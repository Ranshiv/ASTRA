"""Build and persist feature matrices over the canonical store.

A feature matrix is the small, permanent research asset described in the
storage strategy: 100k objects x 25 float32 features is about 10 MB, against
tens of gigabytes for the light curves it summarises. Matrices are kept;
light curves can always be re-materialised from a manifest.
"""

from __future__ import annotations

import os
import hashlib
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config, features, store
from .features import FEATURE_NAMES, FEATURE_VERSION, schema_hash

IDENTITY_COLUMNS = ("object_id", "survey", "release", "band",
                    "coverage_tier", "path")

# Four, not the core count. Measured on this machine (12 logical cores) with
# identical Lomb-Scargle tasks:
#
#   workers   per-task   throughput   parallel efficiency
#         1     1097 ms    0.77 /s          100%
#         2     1533 ms    1.14 /s           75%
#         4     2743 ms    1.33 /s           44%
#         8     5941 ms    1.27 /s           21%
#
# The work per task never changes, only contention: each period search streams
# a ~274,000-point frequency grid plus FFT workspace, far larger than L3, so
# every worker reads from RAM. Throughput peaks around four workers and then
# DECLINES — the limit is memory bandwidth, not cores. Spawning ten workers is
# measurably worse than four.
DEFAULT_WORKERS = 4
MEASURED_SCALING_LIMIT = 4

# Below this, process startup costs more than the work saved. Measured at
# roughly one second per curve for the period search, so a handful of curves
# is genuinely faster in-process.
PARALLEL_THRESHOLD = 8

# Bounds a wedged pool (crashed worker, IPC desync, etc.) so extraction falls
# back to sequential instead of hanging the caller forever.
POOL_TIMEOUT_S = 1800


@dataclass(frozen=True)
class BatchReport:
    """Durable accounting for a resumable feature extraction run."""

    checkpoint: str
    source_count: int
    completed: int
    failed: int
    resumed: bool
    batches: int

    def to_dict(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "source_count": self.source_count,
            "completed": self.completed,
            "failed": self.failed,
            "resumed": self.resumed,
            "batches": self.batches,
        }


def _extract_from_path(path_str: str, periodic_override: dict | None = None
                       ) -> tuple[str, list[float] | None, dict | None]:
    """Worker entry point: read one curve and extract its features.

    Module-level and taking only a string plus a small picklable dict because
    Windows spawns fresh interpreters for workers, so the callable and its
    arguments must pickle. `periodic_override`, when given, comes from a
    parent-process GPU pre-pass -- see `build`'s `periodogram_backend`
    argument -- so this worker never opens its own CUDA context.
    """
    path = Path(path_str)
    try:
        curve = store.read_curve(path)
    except Exception:  # noqa: BLE001 - a corrupt file must not kill the pool
        return path_str, None, None

    extracted = features.extract(curve, path=path_str,
                                 periodic_override=periodic_override)
    row = [float(extracted.values[name]) for name in FEATURE_NAMES]
    identity = {
        "object_id": extracted.object_id,
        "survey": extracted.survey,
        "release": curve.release,
        "band": extracted.band,
        "coverage_tier": coverage_tier(len(curve.dropna())),
        "path": path_str,
    }
    return path_str, row, identity


@dataclass
class FeatureMatrix:
    """Rows of features plus the identity of the curve each row came from."""

    values: np.ndarray            # (n_rows, n_features) float64
    identities: list[dict]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_version: int = FEATURE_VERSION

    def __len__(self) -> int:
        return len(self.identities)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.feature_names.index(name)]

    def finite_mask(self) -> np.ndarray:
        """Rows usable by a detector: every feature present and finite."""
        return np.all(np.isfinite(self.values), axis=1)

    def subset(self, rows: list[int], feature_names: tuple[str, ...] | None = None
               ) -> "FeatureMatrix":
        names = feature_names or self.feature_names
        columns = [self.feature_names.index(name) for name in names]
        values = self.values[np.ix_(rows, columns)] if rows else \
            np.empty((0, len(columns)))
        return FeatureMatrix(values=values,
                             identities=[self.identities[i] for i in rows],
                             feature_names=names,
                             feature_version=self.feature_version)

    def to_dict(self) -> dict:
        return {
            "rows": len(self),
            "features": len(self.feature_names),
            "feature_version": self.feature_version,
            "feature_schema_hash": schema_hash(),
            "usable_rows": int(np.count_nonzero(self.finite_mask())),
            "feature_names": list(self.feature_names),
        }


def build(survey: str | None = None, limit: int = 10_000,
          root: Path | None = None, workers: int | None = None,
          use_cache: bool = True, cache_root: Path | None = None,
          periodogram_backend: str = "cpu",
          ) -> FeatureMatrix:
    """Extract features for every stored curve.

    Cached rows are reused where the source file and feature version are
    unchanged; the remainder is extracted across a process pool. Curves are
    still read one at a time inside each worker, so peak memory stays flat as
    the store grows.

    Feature extraction is where 98.6% of pipeline time was measured, almost
    all of it in the period search, so these two paths are the whole of the
    Phase 9 speedup for this stage.

    `periodogram_backend="gpu"` computes periods on CUDA instead of astropy's
    approximate fast method -- see `features.periodic_features`. It is opt-in
    and never selected implicitly: the two backends are not required to agree
    bit-for-bit, so cache rows are tagged by backend and a row computed under
    one is never reused for the other (`featurecache.cache_key`). The period
    search itself runs once in this process across the whole pending batch,
    not inside worker processes -- see `_gpu_periodic_prepass`.
    """
    from . import featurecache

    features.backend_token(periodogram_backend)
    if periodogram_backend == "gpu":
        from . import gpu_periodogram
        ok, reason = gpu_periodogram.available()
        if not ok:
            # Checked once here rather than left to each curve's own fallback
            # inside periodic_features: without this, a curve computed via
            # the internal CPU fallback would still be written to the cache
            # under the "gpu" key, and a LATER run with a real GPU present
            # would then treat that CPU-computed row as GPU-exact. Downgrading
            # the whole build up front keeps the cache tag always true.
            import logging
            logging.getLogger(__name__).warning(
                "GPU periodogram requested but unavailable (%s); this build "
                "will use the CPU backend throughout.", reason)
            periodogram_backend = "cpu"
    root = root or config.PATHS.datasets
    search_root = root / survey.upper() if survey else root

    if not search_root.exists():
        return FeatureMatrix(values=np.empty((0, len(FEATURE_NAMES))),
                             identities=[])

    paths = sorted(search_root.rglob("*.parquet"))[:limit]
    if not paths:
        return FeatureMatrix(values=np.empty((0, len(FEATURE_NAMES))),
                             identities=[])

    cache = featurecache.load(cache_root) if use_cache else featurecache.FeatureCache()

    rows: dict[str, np.ndarray] = {}
    identities: dict[str, dict] = {}
    pending: list[Path] = []

    for path in paths:
        cached = cache.get(path, periodogram_backend) if use_cache else None
        if cached is not None:
            rows[str(path)] = cached
            # Identity comes from the cache when it was recorded there; only a
            # pre-existing cache entry costs a read.
            identities[str(path)] = (cache.identity(path, periodogram_backend)
                                     or _identity_from_path(path))
        else:
            pending.append(path)

    if pending:
        overrides = (_gpu_periodic_prepass(pending)
                    if periodogram_backend == "gpu" else None)
        for path_str, row, identity in _extract_many(pending, workers, overrides):
            if row is None or identity is None:
                continue
            values = np.asarray(row, dtype=np.float64)
            rows[path_str] = values
            identities[path_str] = identity
            if use_cache:
                cache.put(Path(path_str), values, identity, periodogram_backend)

        if use_cache:
            featurecache.save(cache, cache_root)

    ordered = [str(p) for p in paths if str(p) in rows]
    stacked = (np.vstack([rows[key] for key in ordered]) if ordered
               else np.empty((0, len(FEATURE_NAMES))))

    return FeatureMatrix(values=stacked,
                         identities=[identities[key] for key in ordered])


def _batch_root(checkpoint: Path | None) -> Path:
    """Resolve a checkpoint path without putting state beside source data."""
    return checkpoint or (config.PATHS.cache / "feature-batches" / "default.json")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.",
                                     suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _batch_fingerprint(paths: list[Path], survey: str | None) -> str:
    digest = hashlib.sha256()
    digest.update(str(survey or "*").encode("utf-8"))
    for path in paths:
        try:
            stat = path.stat()
            digest.update(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
        except OSError:
            digest.update(str(path).encode("utf-8"))
    return digest.hexdigest()[:24]


def _write_batch_part(path: Path, rows: list[list[float]], identities: list[dict]) -> None:
    """Write one immutable Parquet part used by a resumable build."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, pa.Array] = {
        key: pa.array([identity.get(key, "unknown") for identity in identities])
        for key in IDENTITY_COLUMNS
    }
    values = np.asarray(rows, dtype=np.float64)
    for index, name in enumerate(FEATURE_NAMES):
        columns[name] = pa.array(values[:, index] if len(rows) else [],
                                 type=pa.float64())
    table = pa.table(columns, metadata={
        b"feature_version": str(FEATURE_VERSION).encode(),
        b"feature_schema_hash": schema_hash().encode(),
    })
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd", compression_level=6)
    os.replace(temporary, path)


def _read_batch_parts(parts: list[Path]) -> FeatureMatrix:
    matrices: list[FeatureMatrix] = []
    for path in parts:
        try:
            table = pq.read_table(path)
            metadata = table.schema.metadata or {}
            recorded_version = int(metadata.get(b"feature_version", b"0"))
            recorded_hash = metadata.get(b"feature_schema_hash", b"").decode("utf-8")
            if recorded_version != FEATURE_VERSION or recorded_hash != schema_hash():
                # A part from a different feature contract is never safe to
                # combine with the current extraction. Its sources are rebuilt
                # after the checkpoint validation below resets the run.
                continue
            identities = [dict(zip(IDENTITY_COLUMNS, row)) for row in zip(*(
                table.column(name).to_pylist() for name in IDENTITY_COLUMNS))]
            values = np.column_stack([
                table.column(name).to_numpy() for name in FEATURE_NAMES
            ]) if identities else np.empty((0, len(FEATURE_NAMES)))
            matrices.append(FeatureMatrix(values=values, identities=identities))
        except Exception:  # noqa: BLE001 - a partial part is ignored and rebuilt
            continue
    if not matrices:
        return FeatureMatrix(values=np.empty((0, len(FEATURE_NAMES))), identities=[])
    return FeatureMatrix(
        values=np.vstack([item.values for item in matrices]),
        identities=[identity for item in matrices for identity in item.identities],
    )


def build_resumable(
    survey: str | None = None,
    limit: int = 10_000,
    root: Path | None = None,
    workers: int | None = None,
    batch_size: int = 256,
    checkpoint: Path | None = None,
    use_cache: bool = True,
    cache_root: Path | None = None,
    progress: Callable[[dict], None] | None = None,
) -> tuple[FeatureMatrix, BatchReport]:
    """Extract features in durable batches and resume after interruption.

    The checkpoint contains only source identities and immutable Parquet part
    paths; feature rows never travel through JSON.  A completed part is
    published atomically before its source paths are marked complete, so a
    process killed between those writes can safely rebuild the part without
    losing a completed row.  ``progress`` receives a small dictionary after
    every batch and is intentionally optional for CLI/library callers.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = root or config.PATHS.datasets
    search_root = root / survey.upper() if survey else root
    checkpoint_path = _batch_root(checkpoint)
    state_root = checkpoint_path.parent / checkpoint_path.stem
    paths = (sorted(
        path for path in search_root.rglob("*.parquet")
        if state_root not in path.parents
    )[:limit] if search_root.exists() else [])
    fingerprint = _batch_fingerprint(paths, survey)
    state_root.mkdir(parents=True, exist_ok=True)

    resumed = False
    state: dict = {}
    if checkpoint_path.exists():
        try:
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            resumed = (
                state.get("fingerprint") == fingerprint
                and state.get("feature_version") == FEATURE_VERSION
                and state.get("feature_schema_hash") == schema_hash()
            )
        except (OSError, json.JSONDecodeError):
            state = {}
    if not resumed:
        state = {
            "schema": 1,
            "feature_version": FEATURE_VERSION,
            "feature_schema_hash": schema_hash(),
            "fingerprint": fingerprint,
            "survey": survey,
            "paths": [str(path) for path in paths],
            "completed": [],
            "failed": [],
            "parts": [],
        }

    path_strings = [str(path) for path in paths]
    valid_completed = set(state.get("completed", [])) & set(path_strings)
    failed = set(state.get("failed", [])) & set(path_strings)
    # A source marked failed is retried on a fresh invocation; it is retained
    # in the report for auditability but never blocks the rest of the batch.
    pending = [path for path in paths if str(path) not in valid_completed]
    existing_parts = [Path(part) for part in state.get("parts", [])
                      if Path(part).exists()]
    batches = len(existing_parts)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        extracted = _extract_many(batch, workers)
        rows: list[list[float]] = []
        identities: list[dict] = []
        batch_completed: list[str] = []
        for path_str, row, identity in extracted:
            if row is None or identity is None:
                failed.add(path_str)
                continue
            rows.append(row)
            identities.append(identity)
            batch_completed.append(path_str)
        if rows:
            part = state_root / f"part-{batches:06d}.parquet"
            _write_batch_part(part, rows, identities)
            existing_parts.append(part)
            state.setdefault("parts", []).append(str(part))
            batches += 1
        valid_completed.update(batch_completed)
        state["completed"] = sorted(valid_completed)
        state["failed"] = sorted(failed)
        _atomic_json(checkpoint_path, state)
        if progress is not None:
            progress({
                "fraction": len(valid_completed) / max(len(paths), 1),
                "items_done": len(valid_completed),
                "items_total": len(paths),
                "failed": len(failed),
                "phase": "features",
            })

    matrix = _read_batch_parts(existing_parts)
    # Keep the checkpoint after a successful completion: it is useful for an
    # audit and lets a later call with the same source set be a zero-work read.
    state["completed"] = sorted(valid_completed)
    state["failed"] = sorted(failed)
    _atomic_json(checkpoint_path, state)
    report = BatchReport(
        checkpoint=str(checkpoint_path), source_count=len(paths),
        completed=len(valid_completed), failed=len(failed), resumed=resumed,
        batches=batches,
    )
    return matrix, report


# Name used by the UI/plan language; keep the explicit name above for callers
# that want to distinguish this from the in-memory ``build`` fast path.
build_streaming = build_resumable


# Thread-limiting variables read by the numerical libraries at import time.
# Lomb-Scargle's fast path runs an FFT, and NumPy's FFT is already threaded, so
# N worker processes each spawning N threads oversubscribes the machine and the
# workers spend their time contending rather than computing. Pinning one thread
# per worker is what turns process parallelism into an actual speedup.
_THREAD_LIMIT_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS")


def _gpu_periodic_prepass(paths: list[Path]) -> dict[str, dict]:
    """Compute periods for a batch of curves on GPU, in this process only.

    A GPU call inside `periodic_features` would otherwise run once per worker
    process. `DEFAULT_WORKERS` (4) each opening a CUDA context on one 4 GB
    card would serialise on the device and add ~200-300 MB of context
    overhead per worker -- strictly worse than the CPU path it is meant to
    beat. Running the period search here, before dispatch, means workers do
    only the remaining CPU-only statistics.

    A curve that fails to read or is too short for a period search is simply
    absent from the returned dict; the worker then falls back to computing
    its own (CPU) period for that one curve, via `extract`'s normal path.
    """
    overrides: dict[str, dict] = {}
    for path in paths:
        try:
            curve = store.read_curve(path)
        except Exception:  # noqa: BLE001 - a corrupt file is the worker's problem
            continue
        tidy = curve.dropna().sorted_by_time()
        if len(tidy) < features.MIN_POINTS:
            continue
        overrides[str(path)] = features.periodic_features(
            tidy.time, tidy.value, tidy.value_err, backend="gpu")
    return overrides


def _extract_many(paths: list[Path], workers: int | None,
                  overrides: dict[str, dict] | None = None):
    """Run extraction in parallel, falling back to in-process on failure."""
    count = workers if workers is not None else DEFAULT_WORKERS
    count = max(1, min(count, len(paths)))
    overrides = overrides or {}
    override_list = [overrides.get(str(p)) for p in paths]

    if count == 1 or len(paths) < PARALLEL_THRESHOLD:
        return [_extract_from_path(str(p), override)
                for p, override in zip(paths, override_list)]

    # Set before the pool starts: workers are spawned fresh on Windows and read
    # these when they import NumPy, so the parent's own already-imported
    # libraries are unaffected.
    previous = {name: os.environ.get(name) for name in _THREAD_LIMIT_VARS}
    for name in _THREAD_LIMIT_VARS:
        os.environ[name] = "1"

    pool = ProcessPoolExecutor(max_workers=count)
    try:
        return list(pool.map(_extract_from_path,
                             [str(p) for p in paths], override_list,
                             chunksize=4, timeout=POOL_TIMEOUT_S))
    except Exception:  # noqa: BLE001 - a pool that cannot start, or wedges
        # and hits POOL_TIMEOUT_S, must not lose the run; sequential
        # extraction produces identical output.
        return [_extract_from_path(str(p), override)
                for p, override in zip(paths, override_list)]
    finally:
        # wait=False: a pool that timed out may have workers stuck forever
        # (e.g. the Windows frozen-multiprocessing respawn bug); waiting on
        # them here would silently reintroduce the same hang we just bounded.
        pool.shutdown(wait=False, cancel_futures=True)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _identity_from_path(path: Path) -> dict:
    """Recover identity for a cache hit without recomputing features."""
    try:
        curve = store.read_curve(path)
        return {"object_id": curve.source.object_id,
                "survey": curve.source.survey,
                "release": curve.release, "band": curve.band,
                "coverage_tier": coverage_tier(len(curve.dropna())),
                "path": str(path)}
    except Exception:  # noqa: BLE001
        return {"object_id": "unknown", "survey": "unknown",
                "release": "unknown", "band": "unknown",
                "coverage_tier": "C", "path": str(path)}


def coverage_tier(points: int) -> str:
    """A supports periodic features, B non-periodic features, C review only."""
    if points >= features.MIN_POINTS_FOR_PERIOD:
        return "A"
    if points >= features.MIN_POINTS:
        return "B"
    return "C"


def matrix_path(name: str, root: Path | None = None) -> Path:
    root = root or config.PATHS.projects
    return root / "features" / f"{name}_v{FEATURE_VERSION}.parquet"


def save(matrix: FeatureMatrix, name: str, root: Path | None = None) -> Path:
    """Persist as Parquet, with the feature version in the filename."""
    path = matrix_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    defaults = {"release": "unknown", "coverage_tier": "A"}
    columns: dict[str, pa.Array] = {
        key: pa.array([identity.get(key, defaults.get(key, "unknown"))
                       for identity in matrix.identities])
        for key in IDENTITY_COLUMNS
    }
    for index, feature_name in enumerate(matrix.feature_names):
        columns[feature_name] = pa.array(
            matrix.values[:, index] if len(matrix) else [],
            type=pa.float64(),
        )

    table = pa.table(columns, metadata={
        b"feature_version": str(matrix.feature_version).encode(),
        b"feature_schema_hash": schema_hash().encode(),
    })
    pq.write_table(table, path, compression="zstd", compression_level=6)
    return path


def load(name: str, root: Path | None = None) -> FeatureMatrix:
    path = matrix_path(name, root)
    table = pq.read_table(path)

    available = set(table.column_names)
    identity_columns = [key for key in IDENTITY_COLUMNS if key in available]
    identities = [dict(zip(identity_columns, row)) for row in zip(*(
        table.column(key).to_pylist() for key in identity_columns))]
    for identity in identities:
        identity.setdefault("release", "unknown")
        identity.setdefault("coverage_tier", "A")
    values = np.column_stack([
        table.column(name).to_numpy() for name in FEATURE_NAMES
    ]) if len(identities) else np.empty((0, len(FEATURE_NAMES)))

    metadata = table.schema.metadata or {}
    version = int(metadata.get(b"feature_version", str(FEATURE_VERSION).encode()))
    recorded_hash = metadata.get(b"feature_schema_hash")
    if version != FEATURE_VERSION:
        raise ValueError(
            f"feature matrix {name!r} uses feature version {version}; "
            f"this engine requires {FEATURE_VERSION}. Rebuild the matrix."
        )
    # Matrices before explicit schema hashes remain readable for backwards
    # compatibility. Once a hash is present, however, accepting a different
    # contract would make detector scores falsely look comparable.
    if recorded_hash is not None and recorded_hash.decode("utf-8") != schema_hash():
        raise ValueError(
            f"feature matrix {name!r} has a different feature schema; "
            "rebuild it before detection."
        )

    return FeatureMatrix(values=values, identities=identities,
                         feature_version=version)


# Columns appended by join_gaia_columns. "gaia_matched" is 1.0/0.0 rather
# than a boolean so it survives np.isfinite() masking like every other
# feature; a row with no counterpart gets NaN in the rest and 0.0 here, so it
# is visible to inspection but excluded from finite_mask() like any other
# missing measurement.
GAIA_JOIN_COLUMNS = (
    "gaia_parallax", "gaia_parallax_snr", "gaia_pmra", "gaia_pmdec",
    "gaia_phot_g_mean_mag", "gaia_bp_rp", "gaia_distance_pc",
    "gaia_abs_g_mag", "gaia_ra_now_deg", "gaia_dec_now_deg", "gaia_matched",
)


def join_gaia_columns(matrix: FeatureMatrix, radius_arcsec: float | None = None,
                      projects_root: Path | None = None) -> tuple[FeatureMatrix, dict]:
    """Append Gaia catalogue columns to an existing matrix, by position.

    Gaia is a catalogue connector (surveys/gaia.py): its main table has no
    time series, so it can never contribute ROWS to a sequence-based study --
    `fetch_light_curves` returns `[]` by design. What it can contribute is
    astrometric/photometric context for objects another survey already found.

    This is deliberately a column join, not a row union. Stacking Gaia in as
    extra rows would change the object population being scored, and any
    resulting "improvement" would be a population artefact rather than the
    effect of the added information -- the exact trap documented for
    ztf_tess in docs/DEFERRED.txt (404 ZTF against 3 TESS describes two
    different populations, not one compared fairly). Joining by position
    keeps `len(result) == len(matrix)` and the identical set of objects;
    only the feature width changes.

    A row with no Gaia counterpart within `radius_arcsec` gets NaN in every
    appended column rather than an imputed value: match/no-match is itself
    informative (see surveys/gaia.py's masked-column handling) and must not
    be silently hidden by imputation.

    Returns the augmented matrix plus a small diagnostic dict (`matched`,
    `total`, `match_rate`) so a caller can report how much of the join
    actually landed rather than trusting the join happened at all.
    """
    from . import crossmatch, metadata
    from .surveys.base import SourceRef
    from .surveys.gaia import derived_properties

    projects_root = projects_root or config.PATHS.projects
    radius = (radius_arcsec if radius_arcsec is not None
             else crossmatch.DEFAULT_RADIUS_ARCSEC)
    joined_names = matrix.feature_names + GAIA_JOIN_COLUMNS

    if len(matrix) == 0:
        empty = np.empty((0, len(joined_names)))
        return (FeatureMatrix(values=empty, identities=[],
                              feature_names=joined_names,
                              feature_version=matrix.feature_version),
               {"matched": 0, "total": 0, "match_rate": None})

    gaia_refs = [
        SourceRef(survey="Gaia", object_id=row["object_id"],
                 ra_deg=row["ra_deg"], dec_deg=row["dec_deg"],
                 extra=row["extra"])
        for row in metadata.list_sources(projects_root)
        if row["survey"].upper() == "GAIA"
        and row["ra_deg"] is not None and row["dec_deg"] is not None
    ]

    extra = np.full((len(matrix), len(GAIA_JOIN_COLUMNS)), np.nan)
    matched = 0

    if gaia_refs:
        sources: list[SourceRef | None] = []
        for identity in matrix.identities:
            try:
                sources.append(store.read_curve(Path(identity["path"])).source)
            except Exception:  # noqa: BLE001 - a missing curve just stays unmatched
                sources.append(None)

        valid = [(i, s) for i, s in enumerate(sources) if s is not None]
        if valid:
            found = crossmatch.match_catalogs(
                [s for _, s in valid], gaia_refs, radius_arcsec=radius)
            # match_catalogs keeps the exact SourceRef instances passed in
            # (it never copies), so matches can be re-associated by identity
            # rather than by position -- match_catalogs silently omits
            # sources with no counterpart, so the two lists are not aligned.
            by_source_id = {id(m.source): m for m in found}
            for row_index, source in valid:
                # Gaia data exists and this object was checked against it, so
                # "not matched" is itself known rather than merely absent --
                # record it as 0.0 rather than leaving gaia_matched at the
                # NaN it shares with every column when no Gaia data exists at
                # all (see the no-Gaia-data branch below). The other Gaia
                # columns still stay NaN either way: a 0/1 match flag is
                # meaningful without a counterpart, but a distance or
                # parallax is not.
                extra[row_index, -1] = 0.0
                match = by_source_id.get(id(source))
                if match is None:
                    continue
                derived = derived_properties(match.counterpart.extra)
                # Gaia's own ra_deg/dec_deg are fixed at J2016.0 (GAIA_EPOCH).
                # These two columns are the same object propagated to today by
                # its proper motion, so a caller who wants "where is it now"
                # doesn't have to redo that math -- and the survey's own
                # ra_deg/dec_deg on the candidate stays untouched as the
                # detecting observation's immutable ground truth.
                ra_now, dec_now = crossmatch.propagate_position(
                    match.counterpart.ra_deg, match.counterpart.dec_deg,
                    match.counterpart.extra.get("pmra"),
                    match.counterpart.extra.get("pmdec"),
                    crossmatch.GAIA_EPOCH, crossmatch.current_epoch(),
                )
                values = [
                    match.counterpart.extra.get("parallax"),
                    derived.get("parallax_snr"),
                    match.counterpart.extra.get("pmra"),
                    match.counterpart.extra.get("pmdec"),
                    match.counterpart.extra.get("phot_g_mean_mag"),
                    derived.get("bp_rp"),
                    derived.get("distance_pc"),
                    derived.get("abs_g_mag"),
                    ra_now,
                    dec_now,
                    1.0,
                ]
                extra[row_index, :] = [
                    np.nan if v is None else float(v) for v in values
                ]
                matched += 1

    stacked = np.hstack([matrix.values, extra]) if len(matrix) else extra
    diagnostics = {
        "matched": matched, "total": len(matrix),
        "match_rate": round(matched / len(matrix), 4) if len(matrix) else None,
    }
    return (FeatureMatrix(values=stacked, identities=matrix.identities,
                          feature_names=joined_names,
                          feature_version=matrix.feature_version),
           diagnostics)


def list_matrices(root: Path | None = None) -> list[dict]:
    root = root or config.PATHS.projects
    directory = root / "features"
    if not directory.exists():
        return []

    listing = []
    for path in sorted(directory.glob("*.parquet")):
        try:
            metadata = pq.read_metadata(path)
        except Exception:  # noqa: BLE001
            continue
        listing.append({
            "name": path.stem,
            "path": str(path),
            "rows": metadata.num_rows,
            "mb": round(path.stat().st_size / 1024 ** 2, 4),
        })
    return listing
