"""LRU eviction keeps the download cache under its configured cap."""

from __future__ import annotations

import os
import time

from astra import cache


def _write(path, size_bytes, atime_offset=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size_bytes)
    if atime_offset:
        when = time.time() - atime_offset
        os.utime(path, (when, when))
    return path


def test_measure_counts_files_recursively(tmp_path):
    _write(tmp_path / "a.fits", 1000)
    _write(tmp_path / "nested" / "b.fits", 2000)

    report = cache.measure(tmp_path)

    assert report.file_count == 2
    assert report.total_bytes == 3000


def test_measure_of_missing_directory_is_empty(tmp_path):
    report = cache.measure(tmp_path / "does-not-exist")
    assert report.file_count == 0
    assert report.total_bytes == 0


def test_no_eviction_when_under_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_CACHE_CAP_GB", "1")
    _write(tmp_path / "small.fits", 5000)

    report = cache.enforce_cap(tmp_path)

    assert report.evicted_files == 0
    assert (tmp_path / "small.fits").exists()


def test_eviction_removes_least_recently_used_first(tmp_path, monkeypatch):
    # 6 KB cap; three 3 KB files must be trimmed to 80% of cap (4.8 KB).
    monkeypatch.setenv("ASTRA_CACHE_CAP_GB", str(6144 / 1024**3))

    _write(tmp_path / "oldest.fits", 3072, atime_offset=9000)
    _write(tmp_path / "middle.fits", 3072, atime_offset=3000)
    _write(tmp_path / "newest.fits", 3072, atime_offset=10)

    report = cache.enforce_cap(tmp_path)

    assert report.evicted_files >= 1
    assert not (tmp_path / "oldest.fits").exists()
    assert (tmp_path / "newest.fits").exists()
    assert report.total_bytes <= report.cap_bytes


def test_report_serialises_for_the_ui(tmp_path):
    _write(tmp_path / "a.fits", 1024)
    payload = cache.measure(tmp_path).to_dict()

    assert {"total_gb", "cap_gb", "usage_fraction", "file_count"} <= payload.keys()
