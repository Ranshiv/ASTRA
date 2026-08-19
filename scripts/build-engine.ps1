param([switch]$Clean)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location (Join-Path $root "engine")
try {
  $arguments = @("-m", "PyInstaller", "--noconfirm")
  if ($Clean) { $arguments += "--clean" }
  $arguments += "astra-engine.spec"
  & (Join-Path $root ".venv\Scripts\python.exe") @arguments
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  # Some third-party hooks add optional GPU modules/DLLs as binaries even
  # when the Python modules are excluded.  PyInstaller may place these at the
  # one-folder root or under _internal, so clean both exact generated trees.
  # Dedicated GPU builds can use -Clean and a GPU-specific spec.
  $bundle = Join-Path $root "engine\dist\astra-engine"
  $internal = Join-Path $bundle "_internal"
  $generatedRoots = @($bundle, $internal) | Where-Object { Test-Path -LiteralPath $_ }
  foreach ($name in @("torch", "torchvision", "cupy", "cupyx", "cupy_backends", "dask")) {
    foreach ($generatedRoot in $generatedRoots) {
      $target = Join-Path $generatedRoot $name
      if (Test-Path -LiteralPath $target) {
        $resolvedTarget = (Resolve-Path -LiteralPath $target).Path
        $resolvedRoot = (Resolve-Path -LiteralPath $generatedRoot).Path
        if ($resolvedTarget.StartsWith($resolvedRoot + [IO.Path]::DirectorySeparatorChar)) {
          Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
        } else { throw "Refusing to remove path outside generated sidecar: $resolvedTarget" }
      }
    }
  }

  $exe = Join-Path $bundle "astra-engine.exe"
  if (-not (Test-Path -LiteralPath $exe)) { throw "PyInstaller did not produce $exe" }
  # A CPU release must not expose a partial importable torch tree.  This
  # check catches hook regressions before Tauri bundles a broken sidecar.
  $previousBindingDebug = $env:ASTRA_LIBRARY_BINDING_DEBUG
  $env:ASTRA_LIBRARY_BINDING_DEBUG = "1"
  try { $probe = & $exe --probe }
  finally {
    if ($null -eq $previousBindingDebug) { Remove-Item Env:ASTRA_LIBRARY_BINDING_DEBUG -ErrorAction SilentlyContinue }
    else { $env:ASTRA_LIBRARY_BINDING_DEBUG = $previousBindingDebug }
  }
  if ($LASTEXITCODE -ne 0) { throw "Packaged engine probe failed (exit $LASTEXITCODE)" }
  try {
    $report = ($probe -join "`n") | ConvertFrom-Json
    if (-not $report.version) { throw "probe omitted protocol version" }
    foreach ($library in @("lightkurve", "astroquery")) {
      if ($report.library_cache_binding.$library -like "unbound:*") {
        throw "$library cache binding failed: $($report.library_cache_binding.$library)"
      }
    }
    if ($report.device.torch_available -or
        $report.device.reason -notlike "PyTorch is not installed*") {
      throw "CPU release exposes a partial or unwanted PyTorch runtime: $($report.device.reason)"
    }
  } catch {
    throw "Packaged engine probe did not pass the CPU release gate: $_"
  }
} finally { Pop-Location }
