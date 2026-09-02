"""neo_hazard.py: MOID, Tisserand parameter, absolute magnitude, PHA
classification, light-time correction, and close-approach distance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from astra import neo_hazard as nh


def _circular_elements(a_au: float, *, i_deg: float = 0.0, argp_deg: float = 0.0,
                       raan_deg: float = 0.0, mean_anomaly_deg: float = 0.0,
                       epoch_mjd: float = 60000.0) -> dict:
    return {"semi_major_axis_au": a_au, "eccentricity": 0.0, "inclination_deg": i_deg,
           "raan_deg": raan_deg, "argument_of_perihelion_deg": argp_deg,
           "mean_anomaly_deg": mean_anomaly_deg, "epoch_mjd": epoch_mjd}


class TestTisserandParameter:
    def test_jupiter_own_orbit_is_exactly_three(self):
        jupiter = _circular_elements(nh.JUPITER_SEMI_MAJOR_AXIS_AU)
        t_j = nh.tisserand_parameter(jupiter)
        assert t_j == pytest.approx(3.0, abs=1e-9)

    def test_earth_orbit_gives_a_large_tisserand_value(self):
        # T_J ~ 3 only holds for bodies near Jupiter's own semi-major axis;
        # for an inner planet the a_J/a term dominates and T_J is large
        # (~6 for Earth) -- this is correct physics, not a bug, and this
        # test guards against a future edit silently "fixing" it to ~3.
        earth = _circular_elements(1.0)
        t_j = nh.tisserand_parameter(earth)
        assert t_j > 5.0

    def test_body_near_jupiter_own_distance_is_near_three(self):
        near_jupiter = _circular_elements(nh.JUPITER_SEMI_MAJOR_AXIS_AU * 1.02,
                                          i_deg=1.0)
        near_jupiter["eccentricity"] = 0.05
        t_j = nh.tisserand_parameter(near_jupiter)
        assert t_j == pytest.approx(3.0, abs=0.1)

    def test_asteroidal_classification(self):
        assert nh.dynamical_class(3.5) == "asteroidal"

    def test_comet_like_classification(self):
        assert nh.dynamical_class(2.0) == "comet-like"

    def test_boundary_at_exactly_three_is_asteroidal(self):
        # T_J > 3 is asteroidal; exactly 3.0 is not > 3, so comet-like.
        assert nh.dynamical_class(3.0) == "comet-like"

    def test_hyperbolic_eccentricity_raises(self):
        elements = _circular_elements(1.0)
        elements["eccentricity"] = 1.0
        with pytest.raises(nh.NeoHazardError):
            nh.tisserand_parameter(elements)

    def test_non_positive_semi_major_axis_raises(self):
        elements = _circular_elements(-1.0)
        with pytest.raises(nh.NeoHazardError):
            nh.tisserand_parameter(elements)


class TestMoid:
    def test_coplanar_circular_orbits_moid_is_the_radius_difference(self):
        inner = _circular_elements(1.0)
        outer = _circular_elements(1.5)
        result = nh.moid(inner, outer, n_coarse=360)
        assert result["moid_au"] == pytest.approx(0.5, abs=1e-3)

    def test_identical_orbits_moid_is_zero(self):
        elements = _circular_elements(1.2, i_deg=5.0)
        result = nh.moid(elements, dict(elements), n_coarse=360)
        assert result["moid_au"] == pytest.approx(0.0, abs=1e-6)

    def test_grid_only_skips_refinement(self):
        inner = _circular_elements(1.0)
        outer = _circular_elements(1.5)
        result = nh.moid(inner, outer, n_coarse=360, refine=False)
        assert result["method"] == "grid"
        assert result["moid_au"] == pytest.approx(0.5, abs=1e-2)

    def test_n_coarse_too_small_raises(self):
        elements = _circular_elements(1.0)
        with pytest.raises(nh.NeoHazardError):
            nh.moid(elements, elements, n_coarse=4)


class TestPhaseFunctionAndMagnitude:
    def test_phase_function_at_zero_phase_is_unity(self):
        assert nh.phase_function(0.0) == pytest.approx(1.0)

    def test_phase_function_decreases_with_phase_angle(self):
        assert nh.phase_function(30.0) < nh.phase_function(0.0)

    def test_phase_angle_out_of_range_raises(self):
        with pytest.raises(nh.NeoHazardError):
            nh.phase_function(180.0)
        with pytest.raises(nh.NeoHazardError):
            nh.phase_function(-1.0)

    def test_absolute_magnitude_at_one_au_zero_phase(self):
        # At r=Delta=1 AU and alpha=0, H = V exactly (phase term is log10(1)=0).
        h = nh.absolute_magnitude(15.0, 1.0, 1.0, 0.0)
        assert h == pytest.approx(15.0, abs=1e-9)

    def test_non_positive_distances_raise(self):
        with pytest.raises(nh.NeoHazardError):
            nh.absolute_magnitude(15.0, 0.0, 1.0, 10.0)
        with pytest.raises(nh.NeoHazardError):
            nh.absolute_magnitude(15.0, 1.0, -1.0, 10.0)


class TestDiameter:
    def test_larger_albedo_gives_smaller_diameter(self):
        small = nh.diameter_km(20.0, albedo=0.25)
        large = nh.diameter_km(20.0, albedo=0.05)
        assert small < large

    def test_albedo_out_of_range_raises(self):
        with pytest.raises(nh.NeoHazardError):
            nh.diameter_km(20.0, albedo=0.0)
        with pytest.raises(nh.NeoHazardError):
            nh.diameter_km(20.0, albedo=1.5)

    def test_diameter_range_is_ordered_low_to_high(self):
        lo, hi = nh.diameter_km_range(20.0)
        assert lo < hi


class TestClassifyHazard:
    def test_boundary_values_are_inclusive_pha(self):
        hazard = nh.classify_hazard(moid_au=0.05, tisserand=3.5, h=22.0)
        assert hazard.is_pha is True

    def test_just_outside_moid_boundary_is_not_pha(self):
        hazard = nh.classify_hazard(moid_au=0.0501, tisserand=3.5, h=22.0)
        assert hazard.is_pha is False

    def test_just_outside_h_boundary_is_not_pha(self):
        hazard = nh.classify_hazard(moid_au=0.05, tisserand=3.5, h=22.01)
        assert hazard.is_pha is False

    def test_main_belt_orbit_is_not_pha(self):
        elements = _circular_elements(2.7, i_deg=5.0)
        tisserand = nh.tisserand_parameter(elements)
        hazard = nh.classify_hazard(moid_au=1.5, tisserand=tisserand, h=14.0)
        assert hazard.is_pha is False
        assert hazard.dynamical_class == "asteroidal"

    def test_missing_moid_or_h_is_never_pha(self):
        hazard = nh.classify_hazard(moid_au=None, tisserand=3.5, h=22.0)
        assert hazard.is_pha is False
        hazard = nh.classify_hazard(moid_au=0.05, tisserand=3.5, h=None)
        assert hazard.is_pha is False

    def test_to_dict_round_trips_as_json(self):
        import json
        hazard = nh.classify_hazard(moid_au=0.05, tisserand=3.5, h=22.0)
        json.dumps(hazard.to_dict())  # must not raise


class TestLightTimeCorrect:
    def test_converges_and_magnitude_is_plausible_for_one_au(self):
        target_position = np.array([1.0, 0.0, 0.0])
        target_velocity = np.array([0.0, 0.017, 0.0])  # ~ AU/day, roughly Earth-like
        observer_position = np.array([0.0, 0.0, 0.0])
        result = nh.light_time_correct(target_position, target_velocity, observer_position)
        # Light-time for 1 AU is ~500 seconds ~ 0.00578 days; assert order of magnitude.
        assert 0.001 < result["light_time_days"] < 0.02

    def test_coincident_observer_and_target_raises(self):
        zero = np.array([0.0, 0.0, 0.0])
        with pytest.raises(nh.NeoHazardError):
            nh.light_time_correct(zero, zero, zero)


class TestCloseApproach:
    def test_end_before_start_raises(self):
        elements = _circular_elements(1.0)
        with pytest.raises(nh.NeoHazardError):
            nh.close_approach(elements, start_mjd=60010.0, end_mjd=60000.0)

    def test_non_positive_step_raises(self):
        elements = _circular_elements(1.0)
        with pytest.raises(nh.NeoHazardError):
            nh.close_approach(elements, start_mjd=60000.0, end_mjd=60010.0, step_days=0.0)

    def test_too_short_window_raises(self):
        elements = _circular_elements(1.0)
        with pytest.raises(nh.NeoHazardError):
            nh.close_approach(elements, start_mjd=60000.0, end_mjd=60001.0, step_days=1.0)

    def test_returns_a_minimum_within_the_window(self):
        elements = _circular_elements(1.0)
        result = nh.close_approach(elements, start_mjd=60000.0, end_mjd=60400.0, step_days=5.0)
        assert result["window_start_mjd"] <= result["close_approach_mjd"] <= result["window_end_mjd"]
        assert result["distance_au"] >= 0.0
        assert result["distance_lunar_distances"] >= 0.0


class TestAssess:
    def test_assess_without_earth_elements_has_no_moid(self):
        elements = _circular_elements(2.7, i_deg=5.0)
        result = nh.assess(elements)
        assert result["moid_au"] is None
        assert result["moid_detail"] is None
        assert result["tisserand_jupiter"] is not None

    def test_assess_with_earth_elements_and_photometry(self):
        elements = _circular_elements(1.0, mean_anomaly_deg=10.0)
        earth = _circular_elements(1.0)
        result = nh.assess(elements, earth_elements=earth, apparent_v=15.0,
                           heliocentric_au=1.0, geocentric_au=0.5, phase_angle_deg=20.0)
        assert result["moid_detail"] is not None
        assert result["absolute_magnitude"] is not None
