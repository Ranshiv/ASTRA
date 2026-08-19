"""Content-addressed Parquet store for normalised light curves.

This is the "extract, then discard" half of the storage strategy. A raw TESS
light-curve FITS file is roughly 2 MB, most of it headers and columns the
pipeline never reads; the three columns that matter compress to ~100-150 KB
here. Raw downloads stay in the capped cache and are disposable.

Addressing is by (survey, release, object_id), so experiments that share
objects share one copy on disk rather than one copy each.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .surveys.base import TIME_DTYPE, VALUE_DTYPE, LightCurve, SourceRef

# zstd beats snappy by roughly 2x on smoothly varying photometry and costs
# little on read, which suits a write-once/read-many canonical store.
_COMPRESSION = "zstd"
_COMPRESSION_LEVEL = 6
# Reentrant: write_curve holds this while calling dataset_usage_bytes, which
# takes it again to read the cached running total.
_WRITE_LOCK = threading.RLock()


class DatasetCapacityError(RuntimeError):
    """Raised when a write would exceed the configured canonical-store cap."""

    def __init__(self, required_bytes: int, cap_bytes: int):
        super().__init__(f"dataset cap exceeded: write needs {required_bytes / 1024**3:.3f} GiB "
                         f"but cap is {cap_bytes / 1024**3:.3f} GiB")
        self.required_bytes = required_bytes
        self.cap_bytes = cap_bytes

SCHEMA = pa.schema([
    pa.field("time", pa.float64(), nullable=False),
    pa.field("value", pa.float32(), nullable=False),
    pa.field("value_err", pa.float32(), nullable=False),
])


@dataclass(frozen=True)
class StoredCurve:
    path: Path
    points: int
    bytes_on_disk: int

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "points": self.points,
            "bytes": self.bytes_on_disk,
            "bytes_per_point": round(self.bytes_on_disk / self.points, 1)
            if self.points else 0.0,
        }


def curve_path(curve: LightCurve, root: Path | None = None) -> Path:
    """Shard by the first two hex characters to keep directories small."""
    root = root or config.PATHS.datasets
    key = curve.source.storage_key(curve.release)
    return (
        root / curve.source.survey.upper() / curve.release
        / key[:2] / f"{key}_{curve.band}.parquet"
    )


def write_curve(curve: LightCurve, root: Path | None = None) -> StoredCurve:
    """Persist one light curve, cleaned and time-ordered."""
    tidy = curve.dropna().sorted_by_time()
    path = curve_path(tidy, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_arrays(
        [
            pa.array(tidy.time, type=pa.float64()),
            pa.array(tidy.value, type=pa.float32()),
            pa.array(tidy.value_err, type=pa.float32()),
        ],
        schema=SCHEMA.with_metadata({
            b"survey": tidy.source.survey.encode(),
            b"release": tidy.release.encode(),
            b"object_id": tidy.source.object_id.encode(),
            b"band": tidy.band.encode(),
            b"value_kind": tidy.value_kind.encode(),
            b"time_system": tidy.time_system.encode(),
            b"ra_deg": str(tidy.source.ra_deg).encode(),
            b"dec_deg": str(tidy.source.dec_deg).encode(),
            b"extra": json.dumps(tidy.source.extra).encode(),
        }),
    )

    # Stage beside the final path, measure the complete compressed artifact,
    # then atomically publish it only after the cap check.  Existing files are
    # replaced in place without charging their bytes twice.
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            pq.write_table(table, temporary, compression=_COMPRESSION,
                           compression_level=_COMPRESSION_LEVEL)
            size = temporary.stat().st_size
            cap = int(config.dataset_cap_gb() * 1024 ** 3)
            current = dataset_usage_bytes(root)
            previous = path.stat().st_size if path.exists() else 0
            required = current - previous + size
            if required > cap:
                raise DatasetCapacityError(required, cap)
            os.replace(temporary, path)
            # Publish succeeded, so fold the change into the running total.
            _adjust_usage(root, size - previous)
        finally:
            if temporary.exists():
                temporary.unlink()

    return StoredCurve(path=path, points=len(tidy), bytes_on_disk=size)


# Running total per root, so the cap check is O(1) per write instead of a
# full tree walk. `write_curve` called this on EVERY write, inside the global
# write lock: invisible at 400 curves, but O(n) per write and therefore O(n^2)
# across a campaign, and it serialised every writer. Measured profiling put
# feature extraction at 98.6% of pipeline time; at Stage B scale this was the
# next bottleneck in line.
_usage_cache: dict[str, int] = {}


def _scan_usage_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    # Products are immutable FITS cutouts stored alongside canonical light
    # curves.  Counting only Parquet made the advertised dataset cap a hole:
    # an image-heavy investigation could fill the disk without a refusal.
    for pattern in ("*.parquet", "*.fits", "*.fit", "*.fts", "*.fits.json",
                    "*.fit.json", "*.fts.json"):
        for path in root.rglob(pattern):
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def ensure_product_capacity(size: int, destination: Path,
                            root: Path | None = None) -> None:
    """Refuse a declared product before it is transferred when possible."""
    if size < 0:
        raise ValueError("product size must not be negative")
    root = root or config.PATHS.datasets
    with _WRITE_LOCK:
        current = dataset_usage_bytes(root)
        previous = destination.stat().st_size if destination.exists() else 0
        required = current - previous + size
        cap = int(config.dataset_cap_gb() * 1024 ** 3)
        if required > cap:
            raise DatasetCapacityError(required, cap)


def publish_product(temporary: Path, destination: Path,
                    root: Path | None = None) -> int:
    """Atomically publish a non-Parquet dataset artifact under the shared cap.

    ``temporary`` must be a complete sibling file produced by a downloader.
    The quota is checked while holding the same lock as Parquet writes, so two
    concurrent jobs cannot both observe space that only one of them owns.
    """
    root = root or config.PATHS.datasets
    if not temporary.is_file():
        raise FileNotFoundError(str(temporary))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        size = temporary.stat().st_size
        previous = destination.stat().st_size if destination.exists() else 0
        ensure_product_capacity(size, destination, root)
        os.replace(temporary, destination)
        _adjust_usage(root, size - previous)
    return size


def publish_product_bundle(temporary: Path, destination: Path,
                           sidecar_temporary: Path, sidecar_destination: Path,
                           root: Path | None = None) -> int:
    """Publish a product and its provenance sidecar as one quota transaction.

    Filesystems do not provide a two-file commit primitive, but both files are
    staged first and the quota decision covers their combined footprint.  A
    sidecar can consequently never push the canonical store over its cap
    unnoticed, and a failed sidecar publication is cleaned up before return.
    """
    root = root or config.PATHS.datasets
    if not temporary.is_file() or not sidecar_temporary.is_file():
        raise FileNotFoundError("product bundle staging file is missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar_destination.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        product_size = temporary.stat().st_size
        sidecar_size = sidecar_temporary.stat().st_size
        previous_product = destination.stat().st_size if destination.exists() else 0
        previous_sidecar = (sidecar_destination.stat().st_size
                            if sidecar_destination.exists() else 0)
        current = dataset_usage_bytes(root)
        required = current - previous_product - previous_sidecar + product_size + sidecar_size
        cap = int(config.dataset_cap_gb() * 1024 ** 3)
        if required > cap:
            raise DatasetCapacityError(required, cap)
        product_was_present = destination.exists()
        sidecar_was_present = sidecar_destination.exists()
        try:
            os.replace(temporary, destination)
            os.replace(sidecar_temporary, sidecar_destination)
        except Exception:
            # Do not leave a newly published FITS without its provenance.
            if not product_was_present and destination.exists():
                destination.unlink()
            if not sidecar_was_present and sidecar_destination.exists():
                sidecar_destination.unlink()
            raise
        _adjust_usage(root, product_size + sidecar_size - previous_product - previous_sidecar)
    return product_size


def dataset_usage_bytes(root: Path | None = None,
                        refresh: bool = False) -> int:
    """Bytes held by the canonical store, cached between writes.

    Pass `refresh=True` to re-scan; anything that changes the store outside
    `write_curve` (a manual delete, an eviction) should do so.
    """
    root = root or config.PATHS.datasets
    key = str(root)
    with _WRITE_LOCK:
        if refresh or key not in _usage_cache:
            _usage_cache[key] = _scan_usage_bytes(root)
        return _usage_cache[key]


def _adjust_usage(root: Path, delta: int) -> None:
    """Keep the running total in step with a completed write."""
    key = str(root)
    if key in _usage_cache:
        _usage_cache[key] = max(_usage_cache[key] + delta, 0)


def invalidate_usage_cache(root: Path | None = None) -> None:
    """Force a re-scan next time usage is asked for."""
    if root is None:
        _usage_cache.clear()
    else:
        _usage_cache.pop(str(root), None)


def dataset_status(root: Path | None = None) -> dict:
    used = dataset_usage_bytes(root)
    cap = config.dataset_cap_gb() * 1024 ** 3
    return {"used_gb": round(used / 1024 ** 3, 4),
            "cap_gb": config.dataset_cap_gb(),
            "usage_fraction": round(used / cap, 6) if cap else 1.0,
            "available_gb": round(max(cap - used, 0) / 1024 ** 3, 4)}


def read_curve(path: Path) -> LightCurve:
    """Reconstruct a light curve, including the source it came from."""
    table = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}

    source = SourceRef(
        survey=meta.get("survey", "unknown"),
        object_id=meta.get("object_id", "unknown"),
        ra_deg=float(meta.get("ra_deg", "nan")),
        dec_deg=float(meta.get("dec_deg", "nan")),
        extra=json.loads(meta.get("extra", "{}")),
    )

    return LightCurve(
        source=source,
        release=meta.get("release", "unknown"),
        band=meta.get("band", "unknown"),
        value_kind=meta.get("value_kind", "mag"),  # type: ignore[arg-type]
        time_system=meta.get("time_system", "JD_UTC"),  # type: ignore[arg-type]
        time=table.column("time").to_numpy().astype(TIME_DTYPE),
        value=table.column("value").to_numpy().astype(VALUE_DTYPE),
        value_err=table.column("value_err").to_numpy().astype(VALUE_DTYPE),
    )


def has_curve(curve: LightCurve, root: Path | None = None) -> bool:
    """True when this object/band is already stored, so a re-fetch can be skipped."""
    return curve_path(curve, root).exists()


def survey_usage(root: Path | None = None) -> dict[str, dict]:
    """Per-survey footprint, for the storage panel in the UI."""
    root = root or config.PATHS.datasets
    usage: dict[str, dict] = {}
    if not root.exists():
        return usage

    for survey_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        total = 0
        count = 0
        products = 0
        for pattern in ("*.parquet", "*.fits", "*.fit", "*.fts", "*.fits.json",
                        "*.fit.json", "*.fts.json"):
            for artifact in survey_dir.rglob(pattern):
                try:
                    total += artifact.stat().st_size
                except OSError:
                    continue
                if artifact.suffix.lower() not in {".parquet", ".json"}:
                    products += 1
        count = sum(1 for _ in survey_dir.rglob("*.parquet"))
        usage[survey_dir.name] = {
            "curves": count,
            "products": products,
            "gb": round(total / 1024 ** 3, 4),
        }
    return usage


def verify_precision(curve: LightCurve, root: Path | None = None) -> dict:
    """Confirm a round trip preserves timing resolution.

    float32 time would silently cost ~2 minutes of resolution at BJD 2457000,
    which is larger than a TESS 2-minute cadence. This makes that regression
    detectable rather than invisible.
    """
    stored = read_curve(curve_path(curve, root))
    tidy = curve.dropna().sorted_by_time()
    if len(stored) == 0:
        return {"points": 0, "max_time_error_s": 0.0, "exact": True}

    delta_days = np.abs(stored.time - tidy.time)
    max_error_s = float(np.max(delta_days) * 86400.0)
    return {
        "points": len(stored),
        "max_time_error_s": max_error_s,
        "exact": max_error_s == 0.0,
    }
