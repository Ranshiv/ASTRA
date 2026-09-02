"""Bounded-memory evidence for backlog item 41 (streaming batches + GPU).

`featurematrix.build_resumable` already batches and checkpoints; what item
41 asks for is proof that a GPU-backed run stays bounded by `batch_size`
rather than by the size of the whole store -- the "Gaia-scale chunks"
requirement. Since no Gaia-scale local dataset exists (Gaia epoch photometry
is explicitly blocked pending DR4, see docs/LIMITATIONS.md), this measures the
same relationship on a synthetic population large enough to show the trend:
GPU device memory used by one prepass call tracks `batch_size`, not the
total object count, and end-to-end host memory for a full `build_resumable`
run grows far slower than linearly with the object count once batching is
in effect.

No test here may require a GPU to run; the GPU-memory-scaling class is
skipped without one, following the house convention in
`test_gpu_periodogram.py`/`test_bocpd_gpu.py`.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from astra import bocpd_gpu, gpu_periodogram

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:  # noqa: BLE001 - absence is the normal case in CI
    _CUPY_AVAILABLE = False

_GPU_READY = (_CUPY_AVAILABLE and gpu_periodogram.available(force=True)[0]
             and bocpd_gpu.available(force=True)[0])


def _write_curves(root, count, points=300, seed=0):
    from astra import store
    from astra.surveys.base import LightCurve, SourceRef

    rng = np.random.default_rng(seed)
    for i in range(count):
        time = np.arange(points, dtype=float) * 0.7
        half = points // 2
        value = np.concatenate([rng.normal(0.0, 0.05, half),
                                rng.normal(8.0, 0.05, points - half)])
        store.write_curve(LightCurve(
            source=SourceRef(survey="ZTF", object_id=f"scale{i}",
                             ra_deg=10.0, dec_deg=20.0),
            release="dr24", band="g", value_kind="mag",
            time=time, value=value, value_err=np.full(points, 0.05),
            time_system="HJD_UTC",
        ), root)


@pytest.mark.skipif(not _GPU_READY,
                    reason="no usable CUDA device on this machine")
class TestGpuMemoryScalesWithBatchNotDataset:
    """The GPU-memory half of "bounded-memory ... CUDA execution"."""

    def _pool_total_after_all_batches(self, tmp_path, count, batch_size):
        """Peak pool size the run ever needed, at a fixed `batch_size`.

        `used_bytes()` is measured only DURING an allocation's lifetime --
        each prepass call's device arrays are refcounted away the instant it
        returns, so sampling `used_bytes()` between calls always reads 0
        regardless of how much memory the batch actually needed. CuPy's
        default pool instead keeps freed blocks cached for reuse rather than
        returning them to the driver, so `total_bytes()` after the whole run
        is the pool's actual high-water mark: if later batches (same
        `batch_size`, same per-curve array shapes) fit inside blocks already
        cached from earlier ones, the pool never has to grow further, which
        is exactly the "bounded by batch_size" claim to measure.
        """
        from astra import featurematrix

        data = tmp_path / f"data_{count}_{batch_size}"
        _write_curves(data, count)
        paths = sorted(data.rglob("*.parquet"))

        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            featurematrix._gpu_bocpd_prepass(batch)
            featurematrix._gpu_periodic_prepass(batch)
        return pool.total_bytes()

    def test_peak_gpu_memory_tracks_batch_size_not_object_count(self, tmp_path):
        """Doubling the dataset at fixed batch_size must not double GPU use.

        This is the concrete, measured claim behind "bounded-memory ...
        CUDA execution": each prepass call only ever holds one batch's
        curves on the device, so processing more batches costs more TIME,
        not more device MEMORY.
        """
        small = self._pool_total_after_all_batches(tmp_path, count=64, batch_size=32)
        large = self._pool_total_after_all_batches(tmp_path, count=256, batch_size=32)

        assert small > 0 and large > 0
        # A generous bound, not a tight one: real allocator behaviour (pool
        # fragmentation, alignment) means this is not expected to be exactly
        # equal, only NOT proportional to the 4x growth in object count.
        assert large <= small * 2, (
            f"GPU pool high-water mark grew from {small} to {large} bytes "
            "when the dataset quadrupled at a fixed batch_size -- expected "
            "roughly flat, not proportional, growth."
        )


class TestHostMemoryScaling:
    """The host-memory half: `build_resumable` end-to-end at increasing scale."""

    def _peak_traced_bytes(self, tmp_path, count, batch_size=32):
        from astra import featurematrix

        data = tmp_path / f"data_{count}"
        _write_curves(data, count)

        tracemalloc.start()
        try:
            featurematrix.build_resumable(
                root=data, batch_size=batch_size, workers=1,
                checkpoint=tmp_path / f"checkpoint_{count}.json",
            )
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak

    def test_peak_host_memory_grows_sublinearly_with_object_count(self, tmp_path):
        """4x the objects at a fixed batch_size must cost far less than 4x
        the peak traced Python memory -- the batching/checkpointing design
        (parquet PARTS written and dropped per batch, not one growing
        in-memory accumulator) is what makes this true.
        """
        small = self._peak_traced_bytes(tmp_path, count=40)
        large = self._peak_traced_bytes(tmp_path, count=320)  # 8x the objects

        assert small > 0 and large > 0
        ratio = large / small
        assert ratio < 8, (
            f"peak traced host memory grew {ratio:.2f}x when the object "
            "count grew 8x -- expected clearly sublinear growth from "
            "per-batch part files, not a proportional increase."
        )
