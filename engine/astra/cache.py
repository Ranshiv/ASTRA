"""Size-capped download cache with least-recently-used eviction.

The raw cache is a staging area that data passes through, not a store: files
arrive from an archive, get extracted to Parquet, and become disposable. This
module is what keeps that promise, because neither astroquery nor lightkurve
bounds its own cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config

_BYTES_PER_GB = 1024 ** 3


@dataclass
class CacheReport:
    total_bytes: int
    file_count: int
    cap_bytes: int
    evicted_bytes: int = 0
    evicted_files: int = 0

    @property
    def total_gb(self) -> float:
        return self.total_bytes / _BYTES_PER_GB

    @property
    def cap_gb(self) -> float:
        return self.cap_bytes / _BYTES_PER_GB

    @property
    def usage_fraction(self) -> float:
        return self.total_bytes / self.cap_bytes if self.cap_bytes else 0.0

    def to_dict(self) -> dict:
        return {
            "total_gb": round(self.total_gb, 3),
            "cap_gb": round(self.cap_gb, 3),
            "usage_fraction": round(self.usage_fraction, 4),
            "file_count": self.file_count,
            "evicted_gb": round(self.evicted_bytes / _BYTES_PER_GB, 3),
            "evicted_files": self.evicted_files,
        }


def _walk_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def measure(root: Path | None = None) -> CacheReport:
    """Report current cache usage without modifying anything."""
    root = root or config.PATHS.cache
    files = _walk_files(root)
    total = 0
    for path in files:
        try:
            total += path.stat().st_size
        except OSError:
            continue  # file vanished mid-scan; not an error worth failing on
    return CacheReport(
        total_bytes=total,
        file_count=len(files),
        cap_bytes=int(config.cache_cap_gb() * _BYTES_PER_GB),
    )


def enforce_cap(root: Path | None = None, target_fraction: float = 0.8) -> CacheReport:
    """Evict least-recently-used files until the cache is under the cap.

    Evicts down to `target_fraction` of the cap rather than exactly to it, so
    a cache sitting at the limit does not trigger an eviction on every call.
    """
    root = root or config.PATHS.cache
    report = measure(root)
    if report.total_bytes <= report.cap_bytes:
        return report

    target_bytes = int(report.cap_bytes * target_fraction)
    entries: list[tuple[float, int, Path]] = []
    for path in _walk_files(root):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((stat.st_atime, stat.st_size, path))

    entries.sort(key=lambda item: item[0])  # oldest access first

    remaining = report.total_bytes
    for _atime, size, path in entries:
        if remaining <= target_bytes:
            break
        try:
            path.unlink()
        except OSError:
            continue
        remaining -= size
        report.evicted_bytes += size
        report.evicted_files += 1

    report.total_bytes = remaining
    report.file_count -= report.evicted_files
    return report
