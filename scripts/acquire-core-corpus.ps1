param(
  [double]$RaDeg = 180.0,
  [double]$DecDeg = 22.0,
  [double]$RadiusArcsec = 120.0,
  [int]$Limit = 200,
  [string]$DatasetId = "",
  [string]$CassetteMode = "off",
  [switch]$RunBenchmark
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# ASTRA_CASSETTE_MODE: "record" captures real HTTP responses into
# research/fixtures/cassettes/ (see engine/astra/research/cassettes.py);
# "replay" reruns entirely offline from a prior recording; "off" (default)
# is today's unmodified live-request behaviour with no cassette involved.
$env:ASTRA_CASSETTE_MODE = $CassetteMode

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
  throw "No .venv found at $venvPython -- create one and 'pip install -e engine' first."
}

$datasetIdArg = if ($DatasetId) { $DatasetId } else { "core-corpus-$(Get-Date -Format yyyyMMdd-HHmmss)" }

$pyArgs = @(
  "-c",
  @"
import sys
sys.path.insert(0, r'$root\engine')
from astra.research.acquire import acquire_core_corpus
from astra.surveys.base import ConeQuery

query = ConeQuery(ra_deg=$RaDeg, dec_deg=$DecDeg, radius_arcsec=$RadiusArcsec)
result = acquire_core_corpus(
    query, dataset_id='$datasetIdArg', limit=$Limit,
    selection_rule='core corpus demonstration cone',
    license='per-survey, see research/datasets/manifests/$datasetIdArg.json',
    citation='per-survey, see research/sources/source_registry.yaml',
)
import json
print(json.dumps(result.to_dict(), indent=2))
"@
)

& $venvPython @pyArgs
if ($LASTEXITCODE -ne 0) { throw "acquire_core_corpus failed (exit $LASTEXITCODE)" }

if ($RunBenchmark) {
  Write-Host "Benchmark run requires a matrix built separately via engine.featurematrix.build" `
    "against the acquired dataset's survey directories, then research.benchmark.run over the RPC" `
    "layer -- see docs/BENCHMARKS.md for the manual sequence until this script grows that step."
}
