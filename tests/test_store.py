"""The canonical Parquet store: precision, addressing and footprint."""

from __future__ import annotations

import numpy as np
import pytest

from astra import store
from astra.surveys.base import LightCurve, SourceRef


def test_round_trip_preserves_values(curve, tmp_path):
    written = store.write_curve(curve, tmp_path)
    restored = store.read_curve(written.path)

    assert len(restored) == len(curve)
    assert restored.band == "g"
    assert restored.value_kind == "mag"
    np.testing.assert_allclose(restored.value, curve.value, rtol=1e-6)


def test_round_trip_preserves_timing_resolution(curve, tmp_path):
    """float32 time would cost minutes of resolution at BJD 2458000."""
    store.write_curve(curve, tmp_path)
    check = store.verify_precision(curve, tmp_path)

    assert check["exact"] is True
    assert check["max_time_error_s"] == 0.0


def test_time_system_survives_the_round_trip(curve, tmp_path):
    written = store.write_curve(curve, tmp_path)
    assert store.read_curve(written.path).time_system == "HJD_UTC"


def test_source_identity_survives_the_round_trip(curve, tmp_path):
    written = store.write_curve(curve, tmp_path)
    restored = store.read_curve(written.path)

    assert restored.source.object_id == curve.source.object_id
    assert restored.source.survey == "ZTF"
    assert restored.source.ra_deg == pytest.approx(180.122)


def test_write_sorts_and_cleans(tmp_path, source):
    messy = LightCurve(source=source, release="dr24", band="g",
                       value_kind="mag",
                       time=[3.0, 1.0, np.nan, 2.0],
                       value=[13.0, 11.0, 99.0, 12.0],
                       value_err=[0.1, 0.1, 0.1, 0.1])

    restored = store.read_curve(store.write_curve(messy, tmp_path).path)

    assert list(restored.time) == [1.0, 2.0, 3.0]


def test_same_object_and_release_share_one_path(curve, tmp_path):
    first = store.write_curve(curve, tmp_path)
    second = store.write_curve(curve, tmp_path)
    assert first.path == second.path


def test_different_release_uses_a_different_path(curve, tmp_path):
    other = LightCurve(source=curve.source, release="dr23", band="g",
                       value_kind="mag", time=curve.time, value=curve.value,
                       value_err=curve.value_err)
    assert store.curve_path(curve, tmp_path) != store.curve_path(other, tmp_path)


def test_has_curve_reports_presence(curve, tmp_path):
    assert store.has_curve(curve, tmp_path) is False
    store.write_curve(curve, tmp_path)
    assert store.has_curve(curve, tmp_path) is True


def test_compression_meets_the_storage_budget(curve, tmp_path):
    """The storage plan assumes ~100-150 KB per light curve, not ~2 MB."""
    written = store.write_curve(curve, tmp_path)
    bytes_per_point = written.bytes_on_disk / written.points

    assert bytes_per_point < 40, f"{bytes_per_point:.1f} B/point is too large"


def test_survey_usage_reports_per_survey_footprint(curve, tmp_path):
    store.write_curve(curve, tmp_path)
    usage = store.survey_usage(tmp_path)

    assert usage["ZTF"]["curves"] == 1
    assert usage["ZTF"]["gb"] >= 0.0


def test_empty_curve_stores_without_error(tmp_path, source):
    empty = LightCurve(source=source, release="dr24", band="g",
                       value_kind="mag", time=[], value=[], value_err=[])
    written = store.write_curve(empty, tmp_path)
    assert written.points == 0


class TestDatasetUsageCache:
    """The cap check used to rescan the whole store on every write.

    That was O(n) per write inside the global write lock, so O(n^2) across a
    campaign, and it serialised every writer. A running total has to stay
    exact, or the cap either leaks or falsely refuses.
    """

    def test_usage_tracks_writes_without_rescanning(self, tmp_path, source):
        from astra import store as store_mod

        store_mod.invalidate_usage_cache()
        assert store_mod.dataset_usage_bytes(tmp_path) == 0

        total = 0
        for index in range(4):
            other = SourceRef(survey="ZTF", object_id=f"o{index}",
                              ra_deg=0.0, dec_deg=0.0)
            curve = LightCurve(source=other, release="dr24", band="g",
                               value_kind="mag",
                               time=2458000.0 + np.arange(80) * 0.1,
                               value=np.full(80, 18.0),
                               value_err=np.full(80, 0.01))
            total += store.write_curve(curve, tmp_path).bytes_on_disk

        assert store_mod.dataset_usage_bytes(tmp_path) == total

    def test_cached_total_matches_a_full_rescan(self, tmp_path, curve):
        from astra import store as store_mod

        store_mod.invalidate_usage_cache()
        store.write_curve(curve, tmp_path)

        cached = store_mod.dataset_usage_bytes(tmp_path)
        rescanned = store_mod.dataset_usage_bytes(tmp_path, refresh=True)

        assert cached == rescanned

    def test_overwriting_a_curve_does_not_double_count(self, tmp_path, curve):
        from astra import store as store_mod

        store_mod.invalidate_usage_cache()
        first = store.write_curve(curve, tmp_path).bytes_on_disk
        store.write_curve(curve, tmp_path)

        assert store_mod.dataset_usage_bytes(tmp_path) == first

    def test_refresh_picks_up_external_deletion(self, tmp_path, curve):
        from astra import store as store_mod

        store_mod.invalidate_usage_cache()
        written = store.write_curve(curve, tmp_path)
        written.path.unlink()

        assert store_mod.dataset_usage_bytes(tmp_path, refresh=True) == 0

    def test_cap_is_still_enforced(self, tmp_path, curve, monkeypatch):
        from astra import store as store_mod

        store_mod.invalidate_usage_cache()
        monkeypatch.setenv("ASTRA_DATASET_CAP_GB", str(1 / 1024 ** 3))

        with pytest.raises(store.DatasetCapacityError):
            store.write_curve(curve, tmp_path)

    def test_a_refused_write_does_not_change_usage(self, tmp_path, curve,
                                                   monkeypatch):
        """A rejected write must not leak bytes into the running total."""
        from astra import store as store_mod

        store_mod.invalidate_usage_cache()
        store_mod.dataset_usage_bytes(tmp_path)  # prime the cache at 0
        monkeypatch.setenv("ASTRA_DATASET_CAP_GB", str(1 / 1024 ** 3))

        with pytest.raises(store.DatasetCapacityError):
            store.write_curve(curve, tmp_path)

        assert store_mod.dataset_usage_bytes(tmp_path) == 0
