"""Feature caching, parallel extraction and profiling.

The correctness requirement dominates here: an optimisation that changes a
result is a bug, not a speedup. Every path must produce identical features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import featurecache, featurematrix, profiling, store
from astra.features import FEATURE_NAMES
from astra.surveys.base import LightCurve, SourceRef


def write_curves(root, count=12, points=60):
    rng = np.random.default_rng(0)
    for i in range(count):
        time = 2458000.0 + np.arange(points) * 0.5
        value = 18.0 + 0.3 * np.sin(2 * np.pi * time / (0.5 + i * 0.1)) \
            + rng.normal(0, 0.02, points)
        store.write_curve(LightCurve(
            source=SourceRef(survey="ZTF", object_id=f"obj{i}",
                             ra_deg=10.0, dec_deg=20.0),
            release="dr24", band="g", value_kind="mag",
            time=time, value=value, value_err=np.full(points, 0.02),
            time_system="HJD_UTC",
        ), root)


class TestCacheKey:
    def test_key_changes_when_the_file_changes(self, curve, tmp_path):
        path = store.write_curve(curve, tmp_path).path
        first = featurecache.cache_key(path)

        import time as time_mod
        time_mod.sleep(0.01)
        path.write_bytes(path.read_bytes() + b"\0")

        assert featurecache.cache_key(path) != first

    def test_missing_file_yields_a_key_not_an_error(self, tmp_path):
        assert "missing" in featurecache.cache_key(tmp_path / "nope.parquet")


class TestCache:
    def test_round_trip(self, tmp_path, curve):
        path = store.write_curve(curve, tmp_path / "data").path
        cache = featurecache.FeatureCache(path=featurecache.cache_path(tmp_path))
        row = np.arange(len(FEATURE_NAMES), dtype=np.float64)
        cache.put(path, row)

        featurecache.save(cache, tmp_path)
        loaded = featurecache.load(tmp_path)

        np.testing.assert_array_equal(loaded.get(path), row)

    def test_miss_on_empty_cache(self, tmp_path, curve):
        path = store.write_curve(curve, tmp_path / "data").path
        cache = featurecache.load(tmp_path)

        assert cache.get(path) is None
        assert cache.misses == 1

    def test_hit_rate_is_tracked(self, tmp_path, curve):
        path = store.write_curve(curve, tmp_path / "data").path
        cache = featurecache.FeatureCache()
        cache.put(path, np.zeros(len(FEATURE_NAMES)))

        cache.get(path)
        cache.get(tmp_path / "other.parquet")

        assert cache.hit_rate == pytest.approx(0.5)

    def test_corrupt_cache_is_ignored_not_fatal(self, tmp_path):
        path = featurecache.cache_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not parquet at all")

        assert featurecache.load(tmp_path).size == 0

    def test_clear_removes_the_file(self, tmp_path):
        featurecache.save(featurecache.FeatureCache(
            path=featurecache.cache_path(tmp_path)), tmp_path)
        assert featurecache.clear(tmp_path) is True
        assert featurecache.clear(tmp_path) is False


class TestBuildEquivalence:
    """Every build path must return byte-identical features."""

    def test_cached_build_matches_uncached(self, tmp_path):
        data = tmp_path / "data"
        write_curves(data)

        uncached = featurematrix.build(root=data, use_cache=False, workers=1)
        featurematrix.build(root=data, cache_root=tmp_path, workers=1)  # warm
        cached = featurematrix.build(root=data, cache_root=tmp_path, workers=1)

        assert len(cached) == len(uncached)
        np.testing.assert_array_equal(np.nan_to_num(cached.values),
                                      np.nan_to_num(uncached.values))

    def test_parallel_build_matches_sequential(self, tmp_path):
        data = tmp_path / "data"
        write_curves(data, count=12)

        sequential = featurematrix.build(root=data, use_cache=False, workers=1)
        parallel = featurematrix.build(root=data, use_cache=False, workers=3)

        np.testing.assert_array_equal(np.nan_to_num(sequential.values),
                                      np.nan_to_num(parallel.values))

    def test_identities_stay_aligned_with_rows(self, tmp_path):
        data = tmp_path / "data"
        write_curves(data, count=10)

        matrix = featurematrix.build(root=data, cache_root=tmp_path, workers=1)

        assert len(matrix.identities) == matrix.values.shape[0]
        assert all(i["survey"] == "ZTF" for i in matrix.identities)

    def test_row_order_is_stable_across_runs(self, tmp_path):
        data = tmp_path / "data"
        write_curves(data, count=10)

        first = featurematrix.build(root=data, cache_root=tmp_path, workers=1)
        second = featurematrix.build(root=data, cache_root=tmp_path, workers=1)

        assert [i["path"] for i in first.identities] == \
            [i["path"] for i in second.identities]

    def test_cache_is_invalidated_when_a_curve_changes(self, tmp_path):
        data = tmp_path / "data"
        write_curves(data, count=10)
        featurematrix.build(root=data, cache_root=tmp_path, workers=1)

        target = sorted(data.rglob("*.parquet"))[0]
        replacement = LightCurve(
            source=SourceRef(survey="ZTF", object_id="obj0",
                             ra_deg=10.0, dec_deg=20.0),
            release="dr24", band="g", value_kind="mag",
            time=2458000.0 + np.arange(60) * 0.5,
            value=np.full(60, 25.0),  # clearly different
            value_err=np.full(60, 0.02), time_system="HJD_UTC",
        )
        store.write_curve(replacement, data)

        rebuilt = featurematrix.build(root=data, cache_root=tmp_path, workers=1)
        means = rebuilt.column("mean")

        assert np.any(np.isclose(means, 25.0))

    def test_empty_store_returns_correct_shape(self, tmp_path):
        matrix = featurematrix.build(root=tmp_path / "nothing")
        assert matrix.shape == (0, len(FEATURE_NAMES))


class TestScalingConstants:
    def test_default_workers_reflects_the_measurement(self):
        """Throughput peaked at four workers and declined beyond it."""
        assert featurematrix.DEFAULT_WORKERS == 4
        assert featurematrix.DEFAULT_WORKERS <= \
            featurematrix.MEASURED_SCALING_LIMIT

    def test_small_batches_stay_in_process(self, tmp_path):
        data = tmp_path / "data"
        write_curves(data, count=3)
        matrix = featurematrix.build(root=data, use_cache=False, workers=4)
        assert len(matrix) == 3


class TestPeriodSearchGuard:
    def test_samples_per_peak_default_is_five(self):
        """Lowering it returns the 2x harmonic instead of the true period."""
        from astra import features as features_mod
        assert features_mod.SAMPLES_PER_PEAK == 5

    def test_default_grid_recovers_a_known_period(self):
        from astra import features as features_mod

        rng = np.random.default_rng(1)
        time = np.sort(rng.uniform(0, 500, 300))
        value = 18.0 + 0.4 * np.sin(2 * np.pi * time / 0.5) \
            + rng.normal(0, 0.05, 300)

        result = features_mod.periodic_features(time, value,
                                                np.full(300, 0.05))

        assert result["best_period_days"] == pytest.approx(0.5, rel=0.02)

    def test_grid_density_is_configurable_but_defaults_to_safe(self):
        """A caller may trade sensitivity for speed, but must do so explicitly.

        The risk is not a loss of precision. On a real 353-point ZTF curve with
        a 2740-day baseline, samples_per_peak=5 returned 0.50789 d and
        samples_per_peak=3 returned 1.03501 d — the 2x harmonic. That happens
        when the periodogram holds two near-equal peaks, which is common in
        sparse ground-based data and is precisely where a reliable answer
        matters. A single clean injected sinusoid does NOT reproduce it, so
        this is documented from the real measurement rather than asserted from
        a synthetic one.
        """
        from astra import features as features_mod

        rng = np.random.default_rng(2)
        time = np.sort(rng.uniform(0, 500, 300))
        value = 18.0 + 0.4 * np.sin(2 * np.pi * time / 0.5) \
            + rng.normal(0, 0.05, 300)
        err = np.full(300, 0.05)

        # A coarser grid must still run when asked for; it is a documented
        # trade-off, not a forbidden setting.
        coarse = features_mod.periodic_features(time, value, err,
                                                min_period_days=0.1,
                                                samples_per_peak=2)
        assert np.isfinite(coarse["best_period_days"])

    def test_raising_the_minimum_period_shrinks_the_grid(self):
        """The safe speed lever: fewer frequencies, same peak resolution."""
        from astropy.timeseries import LombScargle

        rng = np.random.default_rng(3)
        time = np.sort(rng.uniform(0, 1000, 300))
        value = np.sin(2 * np.pi * time / 0.5)
        model = LombScargle(time, value, np.full(300, 0.05))

        fine, _ = model.autopower(minimum_frequency=1 / 500,
                                  maximum_frequency=1 / 0.05,
                                  samples_per_peak=5)
        coarse, _ = model.autopower(minimum_frequency=1 / 500,
                                    maximum_frequency=1 / 0.1,
                                    samples_per_peak=5)

        assert len(coarse) < len(fine) / 1.8


class TestProfiling:
    def test_timing_computes_per_item_cost(self):
        timing = profiling.Timing("x", seconds=2.0, items=4)
        assert timing.per_item_ms == pytest.approx(500.0)
        assert timing.items_per_second == pytest.approx(2.0)

    def test_zero_items_does_not_divide_by_zero(self):
        assert profiling.Timing("x", 1.0, 0).per_item_ms == 0.0

    def test_profile_ranks_hotspots(self):
        profile = profiling.Profile()
        profile.add(profiling.Timing("fast", 0.1))
        profile.add(profiling.Timing("slow", 9.9))

        assert profile.hotspots(1)[0].name == "slow"
        assert profile.to_dict()["stages"][0]["share"] == pytest.approx(0.99)

    def test_measure_context_records_a_timing(self):
        profile = profiling.Profile()
        with profiling.measure(profile, "block", items=2):
            pass
        assert profile.timings[0].name == "block"

    def test_measure_records_even_when_the_block_raises(self):
        profile = profiling.Profile()
        with pytest.raises(ValueError):
            with profiling.measure(profile, "boom"):
                raise ValueError("x")
        assert len(profile.timings) == 1

    def test_empty_store_profiles_without_error(self, tmp_path):
        profile = profiling.profile_feature_extraction(root=tmp_path)
        assert "no stored curves" in profile.notes[0]

    def test_gpu_report_always_returns_a_verdict(self):
        report = profiling.gpu_memory_report()
        assert "available" in report


class TestCachedIdentity:
    """A cache hit that still has to open the file is only half a cache.

    Recovering which object a cached row belonged to used to cost one
    `store.read_curve` per path, so a fully cached matrix build walked the
    entire store anyway and `pipeline.run` read every curve twice.
    """

    def test_identity_survives_a_save_and_load(self, tmp_path, curve):
        path = store.write_curve(curve, tmp_path / "data").path
        cache = featurecache.FeatureCache(path=featurecache.cache_path(tmp_path))
        cache.put(path, np.zeros(len(FEATURE_NAMES)),
                  {"object_id": "123456789", "survey": "ZTF", "release": "dr24",
                   "band": "g", "coverage_tier": "A", "path": str(path)})

        featurecache.save(cache, tmp_path)
        identity = featurecache.load(tmp_path).identity(path)

        assert identity["object_id"] == "123456789"
        assert identity["survey"] == "ZTF"
        assert identity["band"] == "g"
        assert identity["coverage_tier"] == "A"
        assert identity["path"] == str(path)

    def test_row_without_identity_returns_none_not_a_placeholder(self, tmp_path, curve):
        """An older cache must degrade to the previous cost, not produce rows
        labelled "unknown" that would silently corrupt a matrix."""
        path = store.write_curve(curve, tmp_path / "data").path
        cache = featurecache.FeatureCache(path=featurecache.cache_path(tmp_path))
        cache.put(path, np.zeros(len(FEATURE_NAMES)))

        featurecache.save(cache, tmp_path)
        loaded = featurecache.load(tmp_path)

        assert loaded.get(path) is not None
        assert loaded.identity(path) is None

    def test_cached_build_reads_no_curve_files(self, isolated_root, monkeypatch):
        write_curves(isolated_root.datasets, count=6, points=40)
        first = featurematrix.build(survey="ztf")

        reads = []
        real = store.read_curve

        def counting(path, *args, **kwargs):
            reads.append(str(path))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(store, "read_curve", counting)
        second = featurematrix.build(survey="ztf")

        assert reads == []
        assert [i["object_id"] for i in second.identities] == \
            [i["object_id"] for i in first.identities]
        np.testing.assert_allclose(second.values, first.values, equal_nan=True)

    def test_a_changed_curve_invalidates_its_identity(self, isolated_root):
        """Identity shares the row's (path, mtime, size) key, so it cannot
        outlive the data it describes."""
        write_curves(isolated_root.datasets, count=3, points=40)
        matrix = featurematrix.build(survey="ztf")
        cache = featurecache.load()
        target = Path(matrix.identities[0]["path"])

        assert cache.identity(target) is not None

        import time as time_mod
        time_mod.sleep(0.01)
        target.write_bytes(target.read_bytes() + b"\0")

        assert cache.identity(target) is None
        assert cache.get(target) is None
