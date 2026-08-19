"""Downsampling, folding and binning for the light-curve viewer."""

from __future__ import annotations

import numpy as np
import pytest

from astra import store, viz
from astra.surveys.base import LightCurve, SourceRef


@pytest.fixture
def stored(curve, tmp_path):
    return store.write_curve(curve, tmp_path).path


class TestLTTB:
    def test_returns_input_when_target_exceeds_length(self):
        time = np.arange(10, dtype=np.float64)
        value = np.arange(10, dtype=np.float32)
        out_time, _ = viz.lttb(time, value, 50)
        assert len(out_time) == 10

    def test_hits_the_requested_size(self):
        time = np.arange(10_000, dtype=np.float64)
        value = np.sin(time / 100.0).astype(np.float32)
        out_time, out_value = viz.lttb(time, value, 500)
        assert len(out_time) == 500
        assert len(out_value) == 500

    def test_endpoints_are_always_kept(self):
        """The plotted time span must match the data's time span."""
        time = np.arange(5_000, dtype=np.float64)
        value = np.random.default_rng(0).normal(size=5_000).astype(np.float32)
        out_time, _ = viz.lttb(time, value, 200)
        assert out_time[0] == time[0]
        assert out_time[-1] == time[-1]

    def test_output_stays_time_ordered(self):
        time = np.arange(3_000, dtype=np.float64)
        value = np.random.default_rng(1).normal(size=3_000).astype(np.float32)
        out_time, _ = viz.lttb(time, value, 300)
        assert np.all(np.diff(out_time) > 0)

    def test_a_narrow_spike_survives_downsampling(self):
        """Uniform striding would drop this; that is the whole point of LTTB."""
        time = np.arange(10_000, dtype=np.float64)
        value = np.zeros(10_000, dtype=np.float32)
        value[7_431] = 100.0  # a single-epoch flare

        _, out_value = viz.lttb(time, value, 500)

        assert out_value.max() == pytest.approx(100.0)


class TestCurvePayload:
    def test_payload_carries_metadata_and_series(self, stored):
        payload = viz.curve_payload(stored)
        assert payload["band"] == "g"
        assert payload["value_kind"] == "mag"
        assert payload["time_system"] == "HJD_UTC"
        assert len(payload["time"]) == len(payload["value"])

    def test_small_curve_is_not_downsampled(self, stored):
        payload = viz.curve_payload(stored, max_points=5000)
        assert payload["downsampled"] is False
        assert payload["shown_points"] == payload["points"]

    def test_large_curve_is_downsampled(self, tmp_path, source):
        time = 2458000.0 + np.arange(20_000) * 0.001
        big = LightCurve(source=source, release="spoc", band="TESS",
                         value_kind="flux", time=time,
                         value=np.ones(20_000), value_err=np.full(20_000, 0.01))
        path = store.write_curve(big, tmp_path).path

        payload = viz.curve_payload(path, max_points=1000)

        assert payload["downsampled"] is True
        assert payload["shown_points"] == 1000
        assert payload["points"] == 20_000

    def test_error_bars_match_the_kept_points(self, tmp_path, source):
        time = 2458000.0 + np.arange(5_000) * 0.01
        big = LightCurve(source=source, release="dr24", band="g",
                         value_kind="mag", time=time,
                         value=np.full(5_000, 18.0),
                         value_err=np.full(5_000, 0.02))
        path = store.write_curve(big, tmp_path).path

        payload = viz.curve_payload(path, max_points=500)

        assert len(payload["value_err"]) == len(payload["time"])


class TestFolding:
    def test_folding_maps_into_unit_phase(self, stored):
        result = viz.fold(stored, period_days=1.5)
        assert all(0.0 <= p < 1.0 for p in result["phase"])

    def test_folding_recovers_a_known_period(self, tmp_path, source):
        """A 2.5-day sinusoid folded on its true period must be single-valued."""
        period = 2.5
        time = 2458000.0 + np.arange(2_000) * 0.02
        value = 18.0 + 0.5 * np.sin(2 * np.pi * time / period)
        lc = LightCurve(source=source, release="dr24", band="g",
                        value_kind="mag", time=time, value=value,
                        value_err=np.full(2_000, 0.01))
        path = store.write_curve(lc, tmp_path).path

        folded = viz.fold(path, period_days=period, max_points=2000)
        phase = np.asarray(folded["phase"])
        folded_value = np.asarray(folded["value"])

        # Points at similar phase must have similar values; scatter within a
        # narrow phase window should be far below the full amplitude.
        window = (phase > 0.20) & (phase < 0.25)
        assert folded_value[window].std() < 0.05

    def test_wrong_period_does_not_collapse_the_scatter(self, tmp_path, source):
        period = 2.5
        time = 2458000.0 + np.arange(2_000) * 0.02
        value = 18.0 + 0.5 * np.sin(2 * np.pi * time / period)
        lc = LightCurve(source=source, release="dr24", band="g",
                        value_kind="mag", time=time, value=value,
                        value_err=np.full(2_000, 0.01))
        path = store.write_curve(lc, tmp_path).path

        folded = viz.fold(path, period_days=1.234, max_points=2000)
        phase = np.asarray(folded["phase"])
        folded_value = np.asarray(folded["value"])

        window = (phase > 0.20) & (phase < 0.25)
        assert folded_value[window].std() > 0.10

    def test_non_positive_period_is_rejected(self, stored):
        with pytest.raises(ValueError, match="period_days"):
            viz.fold(stored, period_days=0.0)


class TestBinning:
    def test_binning_reduces_the_point_count(self, stored):
        binned = viz.bin_curve(stored, bin_days=10.0)
        assert 0 < binned["bins"] < 200

    def test_bin_values_average_the_input(self, tmp_path, source):
        time = 2458000.0 + np.arange(100) * 0.01  # all inside one 5-day bin
        lc = LightCurve(source=source, release="dr24", band="g",
                        value_kind="mag", time=time,
                        value=np.linspace(17.0, 19.0, 100),
                        value_err=np.full(100, 0.01))
        path = store.write_curve(lc, tmp_path).path

        binned = viz.bin_curve(path, bin_days=5.0)

        assert binned["bins"] == 1
        assert binned["value"][0] == pytest.approx(18.0, abs=0.02)

    def test_non_positive_bin_is_rejected(self, stored):
        with pytest.raises(ValueError, match="bin_days"):
            viz.bin_curve(stored, bin_days=-1.0)


class TestListing:
    def test_listing_summarises_stored_curves(self, curve, tmp_path):
        store.write_curve(curve, tmp_path)
        listed = viz.list_curves(root=tmp_path)

        assert len(listed) == 1
        assert listed[0]["survey"] == "ZTF"
        assert listed[0]["points"] == 200

    def test_listing_filters_by_survey(self, curve, tmp_path):
        store.write_curve(curve, tmp_path)
        assert len(viz.list_curves(survey="ZTF", root=tmp_path)) == 1
        assert viz.list_curves(survey="TESS", root=tmp_path) == []

    def test_listing_empty_root(self, tmp_path):
        assert viz.list_curves(root=tmp_path / "nothing") == []
