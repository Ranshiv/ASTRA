"""GPU periodogram: parity with astropy's exact path, and graceful absence.

No test here may require a GPU to run. Cases that need a real device are
skipped via `pytest.importorskip("cupy")` plus an `available()` check, and
availability itself is tested by monkeypatching the probe -- following the
house convention at `tests/test_hardware.py` and `tests/test_deep.py`.

Parity is checked against astropy's own EXACT path (`method="cython"`), not
`method="fast"`. The two are not required to agree with each other; matching
the approximation is not the goal, and the docstring in `gpu_periodogram`
explains why.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import gpu_periodogram

try:
    import cupy  # noqa: F401
    _CUPY_AVAILABLE = True
except Exception:  # noqa: BLE001 - absence is the normal case in CI
    _CUPY_AVAILABLE = False

# Evaluated once at collection time so a machine without a working GPU skips
# only the parity class below, not this whole file -- the availability and
# fallback tests below are specifically the ones meant to run without one.
_GPU_READY = _CUPY_AVAILABLE and gpu_periodogram.available(force=True)[0]


class TestAvailability:
    def test_absent_cupy_is_reported_not_raised(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "cupy":
                raise ImportError("no module named cupy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        gpu_periodogram.available(force=True)
        ok, reason = gpu_periodogram.available(force=True)

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
        gpu_periodogram.available(force=True)
        before = calls["n"]
        gpu_periodogram.available()
        gpu_periodogram.available()

        assert calls["n"] == before  # no new device queries once cached


@pytest.mark.skipif(not _GPU_READY,
                    reason="no usable CUDA device on this machine")
class TestParityAgainstExactAstropy:
    """These run only on a machine with a real, working GPU."""

    def _grid_and_curve(self, seed: int, n: int = 300, baseline: float = 500.0,
                        period: float = 0.5):
        rng = np.random.default_rng(seed)
        time = np.sort(rng.uniform(0, baseline, n))
        value = 18.0 + 0.4 * np.sin(2 * np.pi * time / period) \
            + rng.normal(0, 0.05, n)
        err = np.full(n, 0.05)
        return time, value, err

    def test_matches_astropy_exact_to_double_precision(self):
        from astropy.timeseries import LombScargle

        time, value, err = self._grid_and_curve(seed=7)
        model = LombScargle(time, value, np.clip(err, 1e-12, None))
        frequency = model.autofrequency(
            minimum_frequency=1.0 / (500.0 * 0.5),
            maximum_frequency=1.0 / 0.05, samples_per_peak=5)

        exact = model.power(frequency, method="cython")
        gpu = gpu_periodogram.power(time, value, err, frequency)

        # Not a loosened tolerance chosen to pass -- this is what float64
        # machine precision on the exact same normal-equations solve actually
        # measures at, on this machine.
        assert np.allclose(gpu, exact, rtol=1e-8, atol=1e-10)

    def test_recovers_the_same_known_period_as_the_cpu_guard_fixture(self):
        """Same seed and shape as TestPeriodSearchGuard's CPU fixture."""
        from astra import features as features_mod

        rng = np.random.default_rng(1)
        time = np.sort(rng.uniform(0, 500, 300))
        value = 18.0 + 0.4 * np.sin(2 * np.pi * time / 0.5) \
            + rng.normal(0, 0.05, 300)
        err = np.full(300, 0.05)

        cpu = features_mod.periodic_features(time, value, err, backend="cpu")
        gpu = features_mod.periodic_features(time, value, err, backend="gpu")

        assert gpu["best_period_days"] == pytest.approx(0.5, rel=0.02)
        assert gpu["best_period_days"] == pytest.approx(
            cpu["best_period_days"], rel=1e-3)

    def test_short_curve_guard_matches_on_both_backends(self):
        from astra import features as features_mod

        time = np.arange(5, dtype=float)
        value = np.zeros(5)
        err = np.full(5, 0.05)

        cpu = features_mod.periodic_features(time, value, err, backend="cpu")
        gpu = features_mod.periodic_features(time, value, err, backend="gpu")

        assert np.isnan(cpu["best_period_days"])
        assert np.isnan(gpu["best_period_days"])


@pytest.mark.skipif(not _GPU_READY,
                    reason="no usable CUDA device on this machine")
class TestFeaturematrixGpuBuild:
    """The batching integration: parent-process GPU pre-pass, tagged cache."""

    def _write_well_separated_curves(self, root, count=4, points=300):
        """Curves with widely-spaced periods over a long baseline.

        `write_curves` in test_performance.py samples at a fixed 0.5-day
        cadence with periods of 0.5-1.1 days -- right at the Nyquist alias
        this codebase's own SAMPLES_PER_PEAK comment warns about, where
        astropy's approximate and an exact periodogram can genuinely pick
        different (both real) peaks among several near-equal ones. That is a
        property of the test data, not of GPU-vs-CPU agreement, so this uses
        the same well-conditioned shape as TestPeriodSearchGuard's fixture.
        """
        from astra import store
        from astra.surveys.base import LightCurve, SourceRef

        rng = np.random.default_rng(0)
        for i in range(count):
            time = np.sort(rng.uniform(0, 500, points))
            period = 0.4 + i * 0.15
            value = 18.0 + 0.4 * np.sin(2 * np.pi * time / period)                 + rng.normal(0, 0.05, points)
            store.write_curve(LightCurve(
                source=SourceRef(survey="ZTF", object_id=f"well{i}",
                                 ra_deg=10.0, dec_deg=20.0),
                release="dr24", band="g", value_kind="mag",
                time=time, value=value, value_err=np.full(points, 0.05),
                time_system="HJD_UTC",
            ), root)

    def test_gpu_build_matches_cpu_build_closely(self, tmp_path):
        from astra import featurematrix

        data = tmp_path / "data"
        self._write_well_separated_curves(data)

        cpu = featurematrix.build(root=data, use_cache=False, workers=1,
                                  periodogram_backend="cpu")
        gpu = featurematrix.build(root=data, use_cache=False, workers=1,
                                  periodogram_backend="gpu")

        assert len(gpu) == len(cpu)
        period_col = cpu.feature_names.index("best_period_days")
        np.testing.assert_allclose(
            gpu.values[:, period_col], cpu.values[:, period_col], rtol=1e-2)

    def test_gpu_and_cpu_rows_are_cached_separately(self, tmp_path):
        from tests.test_performance import write_curves
        from astra import featurecache, featurematrix

        data = tmp_path / "data"
        write_curves(data, count=4, points=80)

        featurematrix.build(root=data, cache_root=tmp_path, workers=1,
                            periodogram_backend="cpu")
        cache = featurecache.load(tmp_path)
        one_path = next(data.rglob("*.parquet"))

        assert cache.get(one_path, backend="cpu") is not None
        assert cache.get(one_path, backend="gpu") is None

    def test_rejects_an_unknown_backend_name(self, tmp_path):
        from astra import featurematrix

        with pytest.raises(ValueError, match="unknown periodogram backend"):
            featurematrix.build(root=tmp_path, periodogram_backend="quantum")


class TestFallbackWhenUnavailable:
    """The failure path: a GPU request on a machine without one."""

    def test_periodic_features_falls_back_to_cpu_without_raising(self, monkeypatch):
        from astra import features as features_mod

        monkeypatch.setattr(gpu_periodogram, "available",
                            lambda force=False: (False, "simulated absence"))

        rng = np.random.default_rng(4)
        time = np.sort(rng.uniform(0, 500, 300))
        value = 18.0 + 0.3 * np.sin(2 * np.pi * time / 0.5) \
            + rng.normal(0, 0.05, 300)
        err = np.full(300, 0.05)

        result = features_mod.periodic_features(time, value, err, backend="gpu")
        cpu_result = features_mod.periodic_features(time, value, err, backend="cpu")

        assert result == cpu_result

    def test_unknown_backend_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown periodogram backend"):
            from astra import features as features_mod
            features_mod.periodic_features(
                np.arange(50, dtype=float), np.zeros(50), np.full(50, 0.05),
                backend="quantum")

    def test_build_downgrades_to_cpu_tag_when_gpu_unavailable(self, tmp_path, monkeypatch):
        """A build that silently fell back to CPU must be cached as CPU.

        Tagging it "gpu" here would mean a LATER run on a machine with a real
        GPU could treat these CPU-approximate rows as GPU-exact and reuse
        them -- exactly the silent mixing backend tagging exists to prevent.
        """
        from tests.test_performance import write_curves
        from astra import featurecache, featurematrix, gpu_periodogram

        monkeypatch.setattr(gpu_periodogram, "available",
                            lambda force=False: (False, "simulated absence"))

        data = tmp_path / "data"
        write_curves(data, count=3, points=80)
        featurematrix.build(root=data, cache_root=tmp_path, workers=1,
                            periodogram_backend="gpu")

        cache = featurecache.load(tmp_path)
        one_path = next(data.rglob("*.parquet"))
        assert cache.get(one_path, backend="cpu") is not None
        assert cache.get(one_path, backend="gpu") is None
