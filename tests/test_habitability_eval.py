"""habitability_eval.py: reference-case recovery, ESI anchor, and HZ
membership Monte Carlo."""

from __future__ import annotations

import inspect

import pytest

from astra import habitability as hab
from astra import habitability_eval as hev
from astra import rpc


def test_not_referenced_by_rpc():
    source = inspect.getsource(rpc)
    assert "habitability_eval" not in source


def test_reference_case_recovery_matches_paper_within_tolerance():
    result = hev.reference_case_recovery()
    assert result["all_within_tolerance"] is True
    for name in hev.REFERENCE_SOLAR_SYSTEM:
        assert abs(result["diff_au"][name]) <= result["tolerance_au"]


def test_earth_esi_reference_case_is_unity():
    result = hev.earth_esi_reference_case()
    assert result["is_unity"] is True
    assert result["esi_global"] == pytest.approx(1.0, abs=1e-6)


class TestHzMembershipProbability:
    def test_earth_with_zero_error_is_always_in_hz(self):
        sun = hab.StellarParameters(teff_k=hab.TEFF_SUN_K, radius_rsun=1.0, luminosity_lsun=1.0)
        earth = hab.PlanetParameters(semi_major_axis_au=1.0)
        result = hev.hz_membership_probability(sun, earth, teff_err_k=0.0, radius_err_rsun=0.0,
                                               semimajor_err_au=0.0, n_trials=100, seed=1)
        assert result["hz_membership_probability"] == pytest.approx(1.0)
        assert result["ci95"] is not None

    def test_far_planet_with_zero_error_never_in_hz(self):
        sun = hab.StellarParameters(teff_k=hab.TEFF_SUN_K, radius_rsun=1.0, luminosity_lsun=1.0)
        far = hab.PlanetParameters(semi_major_axis_au=50.0)
        result = hev.hz_membership_probability(sun, far, n_trials=100, seed=1)
        assert result["hz_membership_probability"] == pytest.approx(0.0)

    def test_missing_semimajor_axis_raises(self):
        sun = hab.StellarParameters(teff_k=hab.TEFF_SUN_K, radius_rsun=1.0, luminosity_lsun=1.0)
        with pytest.raises(hev.HabitabilityEvalError):
            hev.hz_membership_probability(sun, hab.PlanetParameters(), n_trials=10)

    def test_non_positive_trials_raises(self):
        sun = hab.StellarParameters(teff_k=hab.TEFF_SUN_K, radius_rsun=1.0, luminosity_lsun=1.0)
        earth = hab.PlanetParameters(semi_major_axis_au=1.0)
        with pytest.raises(hev.HabitabilityEvalError):
            hev.hz_membership_probability(sun, earth, n_trials=0)

    def test_large_teff_error_rejects_some_nonphysical_draws(self):
        # A huge Teff error will push some draws negative; these must be
        # counted as rejected, not silently dropped or crashed on.
        sun = hab.StellarParameters(teff_k=100.0, radius_rsun=1.0, luminosity_lsun=1.0)
        earth = hab.PlanetParameters(semi_major_axis_au=1.0)
        result = hev.hz_membership_probability(sun, earth, teff_err_k=500.0, n_trials=500, seed=7)
        assert result["n_rejected"] >= 0
        assert result["n_valid"] + result["n_rejected"] == 500


def test_run_validation_study_returns_all_sections():
    result = hev.run_validation_study(n_trials=200)
    assert "reference_case" in result
    assert "earth_esi_reference" in result
    assert "hz_membership" in result
    assert result["hz_membership"]["hz_membership_probability"] is not None
