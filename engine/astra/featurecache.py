"""On-disk feature cache (plan phase 9).

Profiling put 98.6% of pipeline time in feature extraction, and 98.3% of that
inside the Lomb-Scargle period search — roughly one second per curve. The
period grid cannot be coarsened without changing the answer, so the two
remaining levers are doing the work in parallel and not doing it twice.

This module is the second lever. A cache entry is keyed by the curve file's
path, its modification time and the feature version, so a stored feature row is
reused only when the source data and the extraction code are both unchanged.
Bumping FEATURE_VERSION invalidates the entire cache by construction, which is
the behaviour section 19 needs: a feature version is part of an experiment, and
silently mixing versions would corrupt every comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .features import FEATURE_NAMES, FEATURE_VERSION

CACHE_FILENAME = f"features_v{FEATURE_VERSION}.parquet"

# Stored alongside the feature row so a cache hit needs no file read at all.
# Without these, a hit still had to open the Parquet file to recover which
# object the row belonged to — which meant a "cached" matrix build walked the
# entire store anyway, and the pipeline read every curve twice.
IDENTITY_FIELDS = ("object_id", "survey", "release", "band", "coverage_tier")


def cache_key(path: Path) -> str:
    """Identity of one extraction: which file, which revision of it."""
    try:
        stat = path.stat()
    except OSError:
        return f"{path.as_posix()}::missing"
    return f"{path.as_posix()}::{stat.st_mtime_ns}::{stat.st_size}"


@dataclass
class FeatureCache:
    """Feature rows keyed by source revision, persisted as Parquet."""

    rows: dict[str, np.ndarray] = field(default_factory=dict)
    identities: dict[str, dict] = field(default_factory=dict)
    path: Path | None = None
    hits: int = 0
    misses: int = 0

    @property
    def size(self) -> int:
        return len(self.rows)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def get(self, source: Path) -> np.ndarray | None:
        row = self.rows.get(cache_key(source))
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return row

    def identity(self, source: Path) -> dict | None:
        """Cached identity for a hit, or None when written before this existed.

        Returning None rather than a placeholder matters: the caller falls back
        to reading the file, so an older cache degrades to the previous cost
        instead of producing rows labelled "unknown".
        """
        stored = self.identities.get(cache_key(source))
        if not stored:
            return None
        return {**stored, "path": str(source)}

    def put(self, source: Path, values: np.ndarray,
            identity: dict | None = None) -> None:
        key = cache_key(source)
        self.rows[key] = np.asarray(values, dtype=np.float64)
        if identity:
            self.identities[key] = {field: str(identity.get(field, ""))
                                    for field in IDENTITY_FIELDS}

    def to_dict(self) -> dict:
        return {
            "entries": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "feature_version": FEATURE_VERSION,
            "path": None if self.path is None else str(self.path),
        }


def cache_path(root: Path | None = None) -> Path:
    """Lives under the managed cache directory, so the size cap applies."""
    root = root or config.PATHS.cache
    return root / "features" / CACHE_FILENAME


def load(root: Path | None = None) -> FeatureCache:
    """Read the cache, tolerating absence or corruption."""
    path = cache_path(root)
    cache = FeatureCache(path=path)
    if not path.exists():
        return cache

    try:
        table = pq.read_table(path)
    except Exception:  # noqa: BLE001 - a damaged cache is rebuilt, not fatal
        return cache

    if "cache_key" not in table.column_names:
        return cache

    keys = table.column("cache_key").to_pylist()
    columns = [table.column(name).to_numpy() for name in FEATURE_NAMES
               if name in table.column_names]
    if len(columns) != len(FEATURE_NAMES):
        return cache  # written under a different feature set

    values = np.column_stack(columns) if keys else np.empty((0, len(FEATURE_NAMES)))
    cache.rows = {key: values[i] for i, key in enumerate(keys)}

    # Identity columns are optional: a cache written before they existed still
    # loads, and simply costs the file read it always did.
    if all(field in table.column_names for field in IDENTITY_FIELDS):
        stored = {field: table.column(field).to_pylist() for field in IDENTITY_FIELDS}
        cache.identities = {
            key: {field: stored[field][i] for field in IDENTITY_FIELDS}
            for i, key in enumerate(keys)
            if stored["object_id"][i]
        }
    return cache


def save(cache: FeatureCache, root: Path | None = None) -> Path:
    path = cache.path or cache_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(cache.rows)
    values = (np.vstack([cache.rows[k] for k in keys]) if keys
              else np.empty((0, len(FEATURE_NAMES))))

    columns: dict[str, pa.Array] = {"cache_key": pa.array(keys)}
    for index, name in enumerate(FEATURE_NAMES):
        columns[name] = pa.array(
            values[:, index] if keys else [], type=pa.float64())
    for field_name in IDENTITY_FIELDS:
        columns[field_name] = pa.array(
            [cache.identities.get(key, {}).get(field_name, "") for key in keys],
            type=pa.string())

    pq.write_table(pa.table(columns), path,
                   compression="zstd", compression_level=6)
    return path


def clear(root: Path | None = None) -> bool:
    path = cache_path(root)
    if path.exists():
        path.unlink()
        return True
    return False
