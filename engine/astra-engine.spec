# PyInstaller one-folder build used for local smoke testing and release CI.
# The folder is copied as a Tauri resource; the executable stays separately
# discoverable so startup remains fast and diagnostics remain inspectable.
# Keep analysis bounded on Windows: these are the modules imported by the
# JSON-RPC surface; optional torch/CUDA modules are loaded only when requested.
from PyInstaller.utils.hooks import collect_data_files

hiddenimports = [
    "astra.ablation", "astra.acquire", "astra.anomaly", "astra.artifact",
    "astra.cache", "astra.candidates", "astra.config", "astra.crossmatch",
    "astra.catalogs", "astra.credentials",
    "astra.evidence", "astra.experiment", "astra.exports", "astra.featurecache",
    "astra.featurematrix", "astra.features", "astra.fitsio", "astra.hardware",
    "astra.jobs", "astra.logging_config", "astra.manifest", "astra.metadata",
    "astra.image_features", "astra.readiness", "astra.spectral_features",
    "astra.pipeline", "astra.ranker", "astra.review", "astra.rpc", "astra.scoring", "astra.security",
    "astra.store", "astra.tensors", "astra.tess_pixels", "astra.timeframe", "astra.viz",
    "astra.surveys", "astra.surveys.base", "astra.surveys.gaia",
    "astra.surveys.tess", "astra.surveys.ztf", "astra.surveys.sdss",
    "astra.surveys.panstarrs",
]

a = Analysis(["entrypoint.py"], pathex=["."], hiddenimports=hiddenimports,
             # astroquery reads its CITATION file at import time and
             # lightkurve ships runtime metadata.  Explicitly collecting
             # package data keeps the one-folder build self-contained.
             datas=(collect_data_files("astropy") +
                    collect_data_files("astroquery") +
                    collect_data_files("lightkurve") +
                    collect_data_files("pyvo")),
             binaries=[],
             # GPU/deep-learning packages are optional extras and are loaded
             # lazily by the engine; keeping them out makes the default
             # Windows sidecar build practical on CI and CPU-only machines.
             #
             # plotly/polars/duckdb are declared runtime dependencies that no
             # engine module actually imports: polars and duckdb appear only
             # as strings inside rpc._handle_versions (which already reports
             # "not installed" gracefully), and plotly has zero references
             # anywhere — viz.py returns JSON for the frontend to render.
             # Together they are roughly 250 MB of dead weight in the sidecar.
             # scipy and scikit-learn are deliberately NOT excluded:
             # anomaly.detect, ranker.*, candidates.evaluate and ablation.*
             # all import sklearn at call time.
             excludes=["torch", "torchvision", "cupy", "cupy_backends", "dask",
                       "plotly", "polars", "duckdb"],
             # Astropy's configuration discovery uses the live source module
             # on the call stack, and its SAMP constants resolve package data
             # relative to module.__file__.  Keep these packages as external
             # source modules while the rest stays in PYZ; this avoids both
             # call-stack failures and network fallback for bundled data.
             noarchive=False,
             module_collection_mode={"astropy": "py", "lightkurve": "py"})
# PyInstaller's scientific-package hooks can still discover CUDA DLLs through
# optional array-API branches.  In a one-folder app, embedding those binaries
# in EXE would leave an importable namespace after the on-disk cleanup below.
# Filter them before EXE/COLLECT so the CPU sidecar contains neither the files
# nor a partial `torch`/`cupy` namespace package.
_optional_gpu_prefixes = ("torch/", "torchvision/", "cupy/", "cupyx/", "cupy_backends/")
def _is_optional_gpu_entry(entry):
    destination, source, _typecode = entry
    names = f"{destination}|{source}".replace("\\", "/").lower()
    return any(prefix in names for prefix in _optional_gpu_prefixes)

a.binaries[:] = [entry for entry in a.binaries if not _is_optional_gpu_entry(entry)]
a.datas[:] = [entry for entry in a.datas if not _is_optional_gpu_entry(entry)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="astra-engine",
          console=True, debug=False, exclude_binaries=True)
coll = COLLECT(exe, a.binaries, a.datas, name="astra-engine")
