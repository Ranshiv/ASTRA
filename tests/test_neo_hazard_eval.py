"""neo_hazard_eval.py: MOID convergence/accuracy, Tisserand stability,
and close-approach sensitivity."""

from __future__ import annotations

import inspect

import pytest

from astra import neo_hazard_eval as nhe
from astra import rpc


def test_not_referenced_by_rpc():
    source = inspect.getsource(rpc)
    assert "neo_hazard_eval" not in source


class TestMoidConvergence:
    def test_error_shrinks_or_stays_small_with_finer_grid(self):
        result = nhe.moid_convergence(grid_sizes=(90, 360, 1440))
        errors = [row["error_au"] for row in result["grid_results"]]
        # The finest grid should not be worse than the coarsest.
        assert errors[-1] <= errors[0] + 1e-6

    def test_truth_is_exact_radius_difference(self):
        result = nhe.moid_convergence(inner_au=1.0, outer_au=1.5)
        assert result["truth_au"] == pytest.approx(0.5)


def test_moid_reference_cases_are_accurate():
    cases = nhe.moid_reference_cases()
    assert cases["coplanar_circular"]["error_au"] < 0.01
    assert cases["identical_orbits"]["error_au"] < 1e-4


class TestTisserandStability:
    def _elements(self):
        return {"semi_major_axis_au": 2.7, "eccentricity": 0.1, "inclination_deg": 5.0,
               "raan_deg": 0.0, "argument_of_perihelion_deg": 0.0,
               "mean_anomaly_deg": 0.0, "epoch_mjd": 60000.0}

    def test_stable_orbit_has_high_stability_fraction(self):
        # Small perturbations on a main-belt-like orbit should almost never
        # flip the asteroidal/comet-like classification.
        result = nhe.tisserand_classification_stability(self._elements(), n_trials=500, seed=1)
        assert result["classification_stability"] > 0.9
        assert result["ci95"] is not None

    def test_non_positive_trials_raises(self):
        with pytest.raises(nhe.NeoHazardEvalError):
            nhe.tisserand_classification_stability(self._elements(), n_trials=0)


class TestCloseApproachSensitivity:
    def _elements(self):
        return {"semi_major_axis_au": 1.05, "eccentricity": 0.2, "inclination_deg": 8.0,
               "raan_deg": 0.0, "argument_of_perihelion_deg": 0.0,
               "mean_anomaly_deg": 0.0, "epoch_mjd": 60000.0}

    def test_returns_spread_statistics(self):
        result = nhe.close_approach_sensitivity(self._elements(), start_mjd=60000.0,
                                                end_mjd=60200.0, n_trials=30, seed=1)
        assert result["n_valid"] > 0
        assert result["std_distance_au"] is not None
        assert result["min_distance_au"] <= result["mean_distance_au"] <= result["max_distance_au"]

    def test_non_positive_trials_raises(self):
        with pytest.raises(nhe.NeoHazardEvalError):
            nhe.close_approach_sensitivity(self._elements(), start_mjd=60000.0,
                                          end_mjd=60200.0, n_trials=0)


def test_run_validation_study_returns_all_sections():
    result = nhe.run_validation_study(n_trials=50, seed=1)
    assert "moid_convergence" in result
    assert "moid_reference_cases" in result
    assert "tisserand_stability" in result
    assert "close_approach_sensitivity" in result
