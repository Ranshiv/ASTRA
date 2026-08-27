"""GPU bocpd: parity with the CPU recursion, batching, and graceful absence.

No test here may require a GPU to run -- same house convention as
`test_gpu_periodogram.py`: cases that need a real device are skipped via
`pytest.importorskip("cupy")` plus an `available()` check, and availability
itself is tested by monkeypatching the probe.

Parity is checked on a curve with a genuine, clearly-detectable change point.
`bocpd`'s change-probability values are all extremely close to the hazard
floor (1/200) when nothing changed, so argmax over near-ties is not a
meaningful parity target on flat data -- the same caveat `gpu_periodogram`'s
own docstring states about not requiring bit-for-bit agreement in general;
here it specifically means the parity assertion needs a real signal to be
checking anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import bocpd_gpu

try:
    import cupy  # noqa: F401
    _CUPY_AVAILABLE = True
except Exception:  # noqa: BLE001 - absence is the normal case in CI
    _CUPY_AVAILABLE = False

_GPU_READY = _CUPY_AVAILABLE and bocpd_gpu.available(force=True)[0]


def _stepped_curve(seed: int, n: int = 1200, jump: float = 8.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Two constant-brightness segments with an unambiguous jump between them.

    Deliberately NOT a sinusoid-plus-noise curve on irregular timestamps:
    bocpd's noise estimate assumes roughly piecewise-constant brightness, and
    a smooth trend between irregular points inflates that estimate enough to
    push genuine detection down near the hazard floor (measured directly
    while building this fixture -- a 3-magnitude jump on a sinusoidal curve
    scored `max_probability` at 0.0050, barely above the 0.005 floor a flat
    curve gets, with the reported change index essentially arbitrary among
    near-ties). That is a property of comparing an ambiguous case, not a
    backend disagreement, so it is not useful for a parity test. This
    fixture instead reproduces the shape `test_bocpd_gpu` first validated by
    hand: two constant segments, an evenly-spaced time grid, giving a clean
    `max_probability` of 1.0 and a `change_index` both backends must agree on
    exactly.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(n, dtype=float) * 0.7
    half = n // 2
    value = np.concatenate([rng.normal(0.0, 0.05, half),
                            rng.normal(jump, 0.05, n - half)])
    return time, value


class TestAvailability:
    def test_absent_cupy_is_reported_not_raised(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "cupy":
                raise ImportError("no module named cupy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        bocpd_gpu.available(force=True)
        ok, reason = bocpd_gpu.available(force=True)

        assert ok is False
        assert "not installed" in reason

    def test_probe_runs_once_without_force(self, monkeypatch):
        import cupy

        calls = {"n": 0}
        real_get_count = cupy.cuda.runtime.getDeviceCount

        def counting():
            calls["n"] += 1
            return real_get_count()

        monkeypatch.setattr(cupy.cuda.runtime, "getDeviceCount", counting)
        bocpd_gpu.available(force=True)
        before = calls["n"]
        bocpd_gpu.available()
        bocpd_gpu.available()

        assert calls["n"] == before  # no new device queries once cached


@pytest.mark.skipif(not _GPU_READY,
                    reason="no usable CUDA device on this machine")
class TestParityAgainstCpu:
    """These run only on a machine with a real, working GPU."""

    def test_matches_cpu_on_a_curve_with_a_real_change_point(self):
        from astra import features as features_mod

        time, value = _stepped_curve(seed=3)
        cpu = features_mod.bocpd(time, value)
        gpu = bocpd_gpu.compute_batch([(time, value)])[0]

        # Not a loosened tolerance chosen to pass -- measured at ~1e-14
        # relative on this machine, machine precision for float64, the same
        # standard `test_gpu_periodogram.py` holds its own kernel to.
        assert gpu["change_index"] == cpu["change_index"]
        assert gpu["change_probability"] == pytest.approx(
            cpu["change_probability"], rel=1e-9)
        assert gpu["max_probability"] == pytest.approx(
            cpu["max_probability"], rel=1e-9)
        assert gpu["change_time"] == pytest.approx(cpu["change_time"], rel=1e-9)

    def test_batches_multiple_curves_independently(self):
        from astra import features as features_mod

        curves = [_stepped_curve(seed=s, n=n)
                 for s, n in ((1, 800), (2, 1500), (3, 1200))]
        gpu_results = bocpd_gpu.compute_batch(curves)

        for (time, value), gpu in zip(curves, gpu_results):
            cpu = features_mod.bocpd(time, value)
            assert gpu["change_index"] == cpu["change_index"]
            assert gpu["change_probability"] == pytest.approx(
                cpu["change_probability"], rel=1e-9)

    def test_short_curve_guard_matches_cpu(self):
        from astra import features as features_mod

        time = np.arange(2, dtype=float)
        value = np.zeros(2)
        cpu = features_mod.bocpd(time, value)
        gpu = bocpd_gpu.compute_batch([(time, value)])[0]

        assert np.isnan(cpu["change_probability"])
        assert np.isnan(gpu["change_probability"])

    def test_a_batch_mixing_short_and_usable_curves_keeps_correct_alignment(self):
        from astra import features as features_mod

        short = (np.arange(2, dtype=float), np.zeros(2))
        usable = _stepped_curve(seed=5, n=900)
        results = bocpd_gpu.compute_batch([short, usable, short])

        assert np.isnan(results[0]["change_probability"])
        assert np.isnan(results[2]["change_probability"])
        cpu = features_mod.bocpd(*usable)
        assert results[1]["change_index"] == cpu["change_index"]


class TestFeaturematrixGpuBuild:
    """The batching integration: parent-process GPU pre-pass, tagged cache."""

    def _write_stepped_curves(self, root, count=3, n=900):
        from astra import store
        from astra.surveys.base import LightCurve, SourceRef

        for i in range(count):
            time, value = _stepped_curve(seed=i, n=n)
            store.write_curve(LightCurve(
                source=SourceRef(survey="ZTF", object_id=f"step{i}",
                                 ra_deg=10.0, dec_deg=20.0),
                release="dr24", band="g", value_kind="mag",
                time=time, value=value, value_err=np.full(n, 0.05),
                time_system="HJD_UTC",
            ), root)

    @pytest.mark.skipif(not _GPU_READY,
                        reason="no usable CUDA device on this machine")
    def test_gpu_build_matches_cpu_build_closely(self, tmp_path):
        from astra import featurematrix

        data = tmp_path / "data"
        self._write_stepped_curves(data)

        cpu = featurematrix.build(root=data, use_cache=False, workers=1,
                                  bocpd_backend="cpu")
        gpu = featurematrix.build(root=data, use_cache=False, workers=1,
                                  bocpd_backend="gpu")

        assert len(gpu) == len(cpu)
        index_col = cpu.feature_names.index("bocpd_change_index")
        np.testing.assert_allclose(
            gpu.values[:, index_col], cpu.values[:, index_col])

    @pytest.mark.skipif(not _GPU_READY,
                        reason="no usable CUDA device on this machine")
    def test_gpu_and_cpu_rows_are_cached_separately(self, tmp_path):
        from astra import featurecache, featurematrix

        data = tmp_path / "data"
        self._write_stepped_curves(data, count=4, n=400)

        featurematrix.build(root=data, cache_root=tmp_path, workers=1,
                            bocpd_backend="cpu")
        cache = featurecache.load(tmp_path)
        one_path = next(data.rglob("*.parquet"))

        assert cache.get(one_path, backend="cpu") is not None
        assert cache.get(one_path, backend="periodogram=cpu,bocpd=gpu") is None

    def test_rejects_an_unknown_bocpd_backend_name(self, tmp_path):
        from astra import featurematrix

        with pytest.raises(ValueError, match="unknown bocpd backend"):
            featurematrix.build(root=tmp_path, bocpd_backend="quantum")

    def test_default_backend_combo_keeps_the_plain_cpu_cache_tag(self):
        from astra import featurematrix

        assert featurematrix._combined_backend_tag("cpu", "cpu") == "cpu"
        assert featurematrix._combined_backend_tag("gpu", "cpu") != "cpu"
        assert featurematrix._combined_backend_tag("cpu", "gpu") != "cpu"
        assert (featurematrix._combined_backend_tag("gpu", "cpu")
               != featurematrix._combined_backend_tag("cpu", "gpu"))


class TestFallbackWhenUnavailable:
    """The failure path: a GPU request on a machine without one."""

    def test_build_downgrades_to_cpu_tag_when_gpu_unavailable(self, tmp_path, monkeypatch):
        """A build that silently fell back to CPU must be cached as CPU.

        Tagging it as a GPU combo here would mean a LATER run on a machine
        with a real GPU could treat these CPU-computed rows as GPU-computed
        and reuse them -- exactly the silent mixing backend tagging exists
        to prevent (see `test_gpu_periodogram.py`'s identical case for the
        periodogram backend).
        """
        from tests.test_performance import write_curves
        from astra import bocpd_gpu, featurecache, featurematrix

        monkeypatch.setattr(bocpd_gpu, "available",
                            lambda force=False: (False, "simulated absence"))

        data = tmp_path / "data"
        write_curves(data, count=3, points=80)
        featurematrix.build(root=data, cache_root=tmp_path, workers=1,
                            bocpd_backend="gpu")

        cache = featurecache.load(tmp_path)
        one_path = next(data.rglob("*.parquet"))
        assert cache.get(one_path, backend="cpu") is not None
        assert cache.get(one_path, backend="periodogram=cpu,bocpd=gpu") is None
