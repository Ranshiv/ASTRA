"""technosignature_eval.py: false-alarm rate, completeness grid, and
cadence-filter RFI-rejection efficiency."""

from __future__ import annotations

import inspect

import pytest

from astra import rpc
from astra import technosignature_eval as tev


def test_not_referenced_by_rpc():
    source = inspect.getsource(rpc)
    assert "technosignature_eval" not in source


class TestFalseAlarmRate:
    def test_pure_noise_gives_near_zero_false_alarm_rate(self):
        result = tev.false_alarm_rate(n_trials=20, seed=1)
        assert result["any_hit_rate"] < 0.2
        assert result["ci95"] is not None

    def test_non_positive_trials_raises(self):
        with pytest.raises(tev.TechnosignatureEvalError):
            tev.false_alarm_rate(n_trials=0)


def test_false_alarm_rate_is_monotone_non_increasing_with_threshold():
    result = tev.false_alarm_rate_vs_threshold(thresholds=(4.0, 8.0, 15.0), n_trials=20, seed=1)
    rates = [row["any_hit_rate"] for row in result["rows"]]
    assert all(rates[i] >= rates[i + 1] - 1e-9 for i in range(len(rates) - 1))


class TestCompletenessGrid:
    def test_high_snz_zero_drift_is_fully_recovered(self):
        result = tev.completeness_vs_snr_and_drift(snr_grid=(50.0,), drift_grid=(0.0,),
                                                    n_trials_per_cell=5, seed=1)
        assert result["rows"][0]["completeness"] == pytest.approx(1.0)

    def test_non_positive_trials_raises(self):
        with pytest.raises(tev.TechnosignatureEvalError):
            tev.completeness_vs_snr_and_drift(n_trials_per_cell=0)


class TestCadenceRejectionEfficiency:
    def test_rfi_rejected_and_signal_retained_reliably(self):
        result = tev.cadence_rejection_efficiency(n_trials=20, seed=1)
        assert result["rfi_rejection_rate"] == pytest.approx(1.0)
        assert result["signal_retention_rate"] == pytest.approx(1.0)

    def test_non_positive_trials_raises(self):
        with pytest.raises(tev.TechnosignatureEvalError):
            tev.cadence_rejection_efficiency(n_trials=0)


def test_run_validation_study_returns_all_sections():
    result = tev.run_validation_study(n_trials=5, seed=1)
    assert "false_alarm_rate" in result
    assert "false_alarm_rate_vs_threshold" in result
    assert "completeness" in result
    assert "cadence_rejection_efficiency" in result
