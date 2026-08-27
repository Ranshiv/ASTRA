"""stellar_manifold.py: residuals against the real, embedded ZAMS track."""

from __future__ import annotations

import numpy as np
import pytest

from astra import stellar_manifold


class TestNearestTrackPoint:
    def test_a_point_exactly_on_an_anchor_recovers_zero_residual(self):
        # G5V anchor: bp_rp=0.850, abs_g_mag=4.801, teff=5660 K.
        result = stellar_manifold.nearest_track_point(0.850, 4.801)
        assert result["residual_mag"] == pytest.approx(0.0, abs=1e-9)
        assert result["teff_k"] == pytest.approx(5660.0)
        assert result["out_of_range"] is False

    def test_a_known_perpendicular_offset_recovers_that_offset(self):
        result = stellar_manifold.nearest_track_point(0.850, 4.801 + 2.0)
        assert result["residual_mag"] == pytest.approx(2.0, abs=1e-9)

    def test_underluminous_offset_is_negative(self):
        result = stellar_manifold.nearest_track_point(0.850, 4.801 - 1.5)
        assert result["residual_mag"] == pytest.approx(-1.5, abs=1e-9)

    def test_hotter_anchor_has_lower_teff_value_than_cooler_anchor(self):
        hot = stellar_manifold.nearest_track_point(-0.120, 0.515)   # B9V
        cool = stellar_manifold.nearest_track_point(4.65, 14.72)    # M7V
        assert hot["teff_k"] > cool["teff_k"]

    def test_arc_length_fraction_spans_zero_to_one(self):
        hot = stellar_manifold.nearest_track_point(-0.120, 0.515)
        cool = stellar_manifold.nearest_track_point(4.65, 14.72)
        assert hot["arc_length_fraction"] == pytest.approx(0.0)
        assert cool["arc_length_fraction"] == pytest.approx(1.0)

    def test_out_of_range_colour_is_clamped_not_extrapolated(self):
        beyond_cool_end = stellar_manifold.nearest_track_point(10.0, 20.0)
        beyond_hot_end = stellar_manifold.nearest_track_point(-5.0, -2.0)
        assert beyond_cool_end["out_of_range"] is True
        assert beyond_hot_end["out_of_range"] is True
        assert beyond_cool_end["track_abs_g_mag"] == pytest.approx(14.72)
        assert beyond_hot_end["track_abs_g_mag"] == pytest.approx(0.515)

    def test_non_finite_input_is_rejected(self):
        with pytest.raises(ValueError):
            stellar_manifold.nearest_track_point(float("nan"), 5.0)


class TestIsochroneResidual:
    def test_missing_bp_rp_returns_none(self):
        assert stellar_manifold.isochrone_residual(None, 5.0) is None

    def test_missing_abs_g_returns_none(self):
        assert stellar_manifold.isochrone_residual(0.85, None) is None

    def test_non_finite_input_returns_none(self):
        assert stellar_manifold.isochrone_residual(float("nan"), 5.0) is None

    def test_extinction_correction_shifts_the_residual(self):
        uncorrected = stellar_manifold.isochrone_residual(0.850, 4.801 + 1.0)
        corrected = stellar_manifold.isochrone_residual(0.850, 4.801 + 1.0, a_g=1.0)
        assert uncorrected["residual_mag"] == pytest.approx(1.0, abs=1e-9)
        assert corrected["residual_mag"] == pytest.approx(0.0, abs=1e-9)
        assert corrected["a_g_used"] == pytest.approx(1.0)

    def test_colour_extinction_correction_uses_ebpminrp_directly(self):
        # ebpminrp shifts the colour used to look up the track, not the mag.
        reddened_bp_rp = 0.850 + 0.3
        result = stellar_manifold.isochrone_residual(
            reddened_bp_rp, 4.801, ebpminrp=0.3)
        assert result["residual_mag"] == pytest.approx(0.0, abs=1e-9)
        assert result["ebpminrp_used"] == pytest.approx(0.3)

    def test_no_extinction_supplied_leaves_used_fields_none(self):
        result = stellar_manifold.isochrone_residual(0.850, 4.801)
        assert result["a_g_used"] is None
        assert result["ebpminrp_used"] is None

    def test_non_finite_extinction_is_ignored_not_propagated(self):
        result = stellar_manifold.isochrone_residual(
            0.850, 4.801, a_g=float("nan"), ebpminrp=float("nan"))
        assert result["a_g_used"] is None
        assert result["ebpminrp_used"] is None
        assert result["residual_mag"] == pytest.approx(0.0, abs=1e-9)


class TestCompareToSpectroscopicTeff:
    def test_exact_match_is_zero_discrepancy(self):
        assert stellar_manifold.compare_to_spectroscopic_teff(5000.0, 5000.0) == pytest.approx(0.0)

    def test_hotter_cmd_estimate_is_a_positive_discrepancy(self):
        result = stellar_manifold.compare_to_spectroscopic_teff(6000.0, 5000.0)
        assert result == pytest.approx(0.2)

    def test_missing_cmd_teff_returns_none(self):
        assert stellar_manifold.compare_to_spectroscopic_teff(None, 5000.0) is None

    def test_missing_spectroscopic_teff_returns_none(self):
        assert stellar_manifold.compare_to_spectroscopic_teff(5000.0, None) is None

    def test_non_positive_spectroscopic_teff_returns_none(self):
        assert stellar_manifold.compare_to_spectroscopic_teff(5000.0, 0.0) is None
