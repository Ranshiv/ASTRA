"""biosignature_eval.py: false-positive rate, amplitude recovery, and the
flat-line null case."""

from __future__ import annotations

import inspect

import pytest

from astra import biosignature_eval as bev
from astra import rpc


def test_not_referenced_by_rpc():
    source = inspect.getsource(rpc)
    assert "biosignature_eval" not in source


class TestFalsePositiveRate:
    def test_flat_spectrum_gives_low_false_positive_rate(self):
        result = bev.false_positive_rate(n_trials=40, seed=1)
        assert result["false_positive_rate"] is not None
        assert result["false_positive_rate"] < 0.2
        assert result["ci95"] is not None

    def test_non_positive_trials_raises(self):
        with pytest.raises(bev.BiosignatureEvalError):
            bev.false_positive_rate(n_trials=0)


class TestAmplitudeRecovery:
    def test_strong_signal_has_high_completeness(self):
        # A modest default-SNR injection (log10_amplitude=0.0 at the
        # DEFAULT_ERROR_PPM noise level) is realistically near the
        # detection threshold -- a strong, low-noise injection is needed
        # to check the completeness ceiling behaves as expected.
        result = bev.amplitude_recovery(true_log10_amplitude=1.5, n_trials=20, seed=1,
                                        error_ppm=10.0, cross_sections={"H2O": 5.0})
        assert result["detection_completeness"] is not None
        assert result["detection_completeness"] > 0.5

    def test_weak_signal_at_default_snr_has_low_completeness(self):
        result = bev.amplitude_recovery(true_log10_amplitude=0.0, n_trials=20, seed=1)
        assert result["detection_completeness"] is not None
        assert result["detection_completeness"] < 0.5

    def test_non_positive_trials_raises(self):
        with pytest.raises(bev.BiosignatureEvalError):
            bev.amplitude_recovery(n_trials=0)


def test_flat_line_null_case_detects_nothing():
    result = bev.flat_line_null_case()
    assert result["any_detected"] is False
    assert all(detected is False for detected in result["per_molecule"].values())


def test_run_validation_study_returns_all_sections():
    result = bev.run_validation_study(n_trials=10, seed=1)
    assert "false_positive_rate" in result
    assert "amplitude_recovery" in result
    assert "flat_line_null_case" in result
