"""asteroseismology_eval.py: exact round-trip check and injection-recovery
of `measure`, including the Dnu aliasing-rate measurement."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from astra import asteroseismology_eval as asev
from astra import rpc


def test_not_referenced_by_rpc():
    source = inspect.getsource(rpc)
    assert "asteroseismology_eval" not in source


class TestRoundTripRecovery:
    def test_is_algebraically_exact(self):
        result = asev.round_trip_recovery()
        assert result["algebraically_exact"] is True
        assert result["max_radius_fractional_error"] < 1e-9
        assert result["max_mass_fractional_error"] < 1e-9
        assert result["n_cases"] > 0


class TestMeasurementRecovery:
    def test_recovers_reasonably_at_high_snr(self):
        # Small grid, one numax, high SNR -- should recover with low bias.
        result = asev.measurement_recovery(numax_grid=(1200.0,), amplitude_snr_grid=(2.0,),
                                           n_trials_per_cell=3, seed=1)
        assert result["n_valid"] > 0
        cell = result["cells"][0]
        assert abs(cell["numax_fractional_bias"]) < 0.1

    def test_reports_aliasing_rate_with_ci(self):
        result = asev.measurement_recovery(numax_grid=(500.0, 2800.0), amplitude_snr_grid=(1.0,),
                                           n_trials_per_cell=2, seed=2)
        assert "dnu_aliasing_rate" in result
        assert result["n_trials"] == result["n_valid"] + result["n_rejected"]
        if result["n_valid"] > 0:
            assert result["ci95"] is not None

    def test_non_positive_trials_raises(self):
        with pytest.raises(asev.AsteroseismologyEvalError):
            asev.measurement_recovery(n_trials_per_cell=0)

    def test_accepts_explicit_real_cadence(self):
        cadence = np.arange(15000) * (2.0 / 1440.0)
        result = asev.measurement_recovery(real_cadence_days=cadence, numax_grid=(1200.0,),
                                           amplitude_snr_grid=(2.0,), n_trials_per_cell=1, seed=3)
        assert result["n_trials"] == 1


def test_run_validation_study_returns_both_sections():
    result = asev.run_validation_study(n_trials_per_cell=1, seed=1)
    assert "round_trip" in result
    assert "measurement_recovery" in result
