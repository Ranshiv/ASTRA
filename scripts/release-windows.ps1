<#
.SYNOPSIS
Builds a Windows release with reproducible checksums and a production signing gate.

Development mode creates the same artifacts and checksum manifest but allows an
unsigned local build. Production mode is deliberately fail-closed: a code-
signing certificate, timestamp service, valid sidecar signature, and valid
installer signature are all required.

Set these in CI rather than putting credentials in the repository:
  ASTRA_SIGNING_CERT_THUMBPRINT
  ASTRA_SIGNING_CERT_STORE       (optional; defaults to Cert:\CurrentUser\My)
  ASTRA_TIMESTAMP_URL
#>
[CmdletBinding()]
param(
  [ValidateSet("Development", "Production")]
  [string]$Mode = "Development",
  [string]$CertificateThumbprint = $env:ASTRA_SIGNING_CERT_THUMBPRINT,
  [string]$CertificateStore = $(if ($env:ASTRA_SIGNING_CERT_STORE) {
      $env:ASTRA_SIGNING_CERT_STORE
    } else { "Cert:\CurrentUser\My" }),
  [string]$TimestampUrl = $env:ASTRA_TIMESTAMP_URL,
  [switch]$SkipEngineBuild,
  [switch]$SkipTauriBuild
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$engineExe = Join-Path $root "engine\dist\astra-engine\astra-engine.exe"
$tauriCli = Join-Path $root "node_modules\.bin\tauri.cmd"

function Require-ProductionSigningConfiguration {
  if ($Mode -ne "Production") { return }
  if ([string]::IsNullOrWhiteSpace($CertificateThumbprint)) {
    throw "Production releases require ASTRA_SIGNING_CERT_THUMBPRINT (or -CertificateThumbprint)."
  }
  if ([string]::IsNullOrWhiteSpace($TimestampUrl)) {
    throw "Production releases require ASTRA_TIMESTAMP_URL (or -TimestampUrl)."
  }
}

function Get-SigningCertificate {
  $normalizedThumbprint = ($CertificateThumbprint -replace "\s", "").ToUpperInvariant()
  $certificatePath = Join-Path $CertificateStore $normalizedThumbprint
  $certificate = Get-Item -LiteralPath $certificatePath -ErrorAction Stop
  if (-not $certificate.HasPrivateKey) {
    throw "Signing certificate $normalizedThumbprint has no accessible private key."
  }
  return $certificate
}

function Assert-ValidSignature([string]$Path) {
  $signature = Get-AuthenticodeSignature -FilePath $Path
  if ($signature.Status -ne "Valid") {
    throw "Authenticode verification failed for ${Path}: $($signature.Status) $($signature.StatusMessage)"
  }
}

function Sign-ReleaseFile([string]$Path, $Certificate) {
  $result = Set-AuthenticodeSignature -FilePath $Path -Certificate $Certificate `
    -TimestampServer $TimestampUrl -HashAlgorithm SHA256
  if ($result.Status -ne "Valid") {
    throw "Authenticode signing failed for ${Path}: $($result.Status) $($result.StatusMessage)"
  }
  Assert-ValidSignature $Path
}

function New-ChecksumManifest([string[]]$ArtifactPaths, [string]$OutputDirectory) {
  $artifacts = @($ArtifactPaths | Sort-Object -Unique | ForEach-Object {
    if (-not (Test-Path -LiteralPath $_ -PathType Leaf)) {
      throw "Cannot checksum missing artifact: $_"
    }
    $item = Get-Item -LiteralPath $_
    $hash = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    [PSCustomObject]@{
      path = $_.Substring($root.Length).TrimStart([char[]]@('\', '/'))
      bytes = [int64]$item.Length
      sha256 = $hash
    }
  })
  if ($artifacts.Count -eq 0) { throw "No release artifacts were found to checksum." }

  $manifest = [PSCustomObject]@{
    product = "ASTRA"
    generated_utc = [DateTime]::UtcNow.ToString("o")
    artifacts = $artifacts
  }
  $jsonPath = Join-Path $OutputDirectory "release-checksums.json"
  $shaPath = Join-Path $OutputDirectory "release-checksums.sha256"
  # PowerShell 5 does not recognise the utf8NoBOM enum value introduced by
  # PowerShell 6.  Use .NET explicitly so local Windows builds and CI emit
  # the same BOM-free JSON manifest.
  $manifestJson = $manifest | ConvertTo-Json -Depth 4
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($jsonPath, $manifestJson, $utf8NoBom)
  ($artifacts | ForEach-Object { "$($_.sha256) *$($_.path -replace '\\','/')" }) |
    Set-Content -LiteralPath $shaPath -Encoding ascii

  # Verify the values just written. This catches a malformed manifest or a
  # build step that altered an artifact after its checksum was calculated.
  $written = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
  foreach ($artifact in $written.artifacts) {
    $absolute = Join-Path $root $artifact.path
    $actual = (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $artifact.sha256) {
      throw "Checksum verification failed for $absolute"
    }
  }
  return @($jsonPath, $shaPath)
}

Require-ProductionSigningConfiguration

if (-not $SkipEngineBuild) {
  & (Join-Path $root "scripts\build-engine.ps1")
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if (-not (Test-Path -LiteralPath $engineExe -PathType Leaf)) {
  throw "Packaged sidecar is missing: $engineExe"
}

$certificate = $null
if ($Mode -eq "Production") {
  $certificate = Get-SigningCertificate
  Sign-ReleaseFile $engineExe $certificate
}

if (-not $SkipTauriBuild) {
  if (-not (Test-Path -LiteralPath $tauriCli -PathType Leaf)) {
    throw "Tauri CLI is missing; run npm install first."
  }
  Push-Location $root
  try {
    & $tauriCli build --config "src-tauri/tauri.release.conf.json"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  } finally { Pop-Location }
}

$releaseExe = Join-Path $root "src-tauri\target\release\ASTRA.exe"
$installerDirectory = Join-Path $root "src-tauri\target\release\bundle\nsis"
if (-not (Test-Path -LiteralPath $releaseExe -PathType Leaf)) {
  throw "Tauri application executable is missing: $releaseExe"
}
$installers = @(Get-ChildItem -LiteralPath $installerDirectory -Filter "*.exe" -File -ErrorAction SilentlyContinue)
if ($installers.Count -eq 0) { throw "NSIS installer is missing from $installerDirectory" }

if ($Mode -eq "Production") {
  Sign-ReleaseFile $releaseExe $certificate
  foreach ($installer in $installers) { Sign-ReleaseFile $installer.FullName $certificate }
}

# Vite clears `dist/` on every frontend build.  Keep release attestations out
# of that build directory so a later `npm run build` cannot erase them.
$outputDirectory = Join-Path $root "release"
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$manifestPaths = New-ChecksumManifest `
  -ArtifactPaths (@($engineExe, $releaseExe) + @($installers | ForEach-Object FullName)) `
  -OutputDirectory $outputDirectory

[PSCustomObject]@{
  mode = $Mode
  sidecar = $engineExe
  application = $releaseExe
  installers = @($installers | ForEach-Object FullName)
  checksums = $manifestPaths
} | ConvertTo-Json -Depth 3
