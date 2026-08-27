"""Evaluation-study correctness for `quality_drift_eval.py`. No
`research` extra needed (no new optional dependency).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astra import quality_drift_eval as qde
from astra.quality_drift import QualityDriftError


def test_evaluate_calibration_achieves_target_fpr_synthetic():
    result = qde.evaluate_calibration_achieves_target_fpr_synthetic(
        target_fpr=0.1, n_stream=200, n_reference=1000, n_trials=300)
    assert result["empirical_fpr"] < 0.2
    assert result["ci95"] is not None


def test_evaluate_calibration_rejects_too_few_reference_points():
    with pytest.raises(QualityDriftError):
        qde.evaluate_calibration_achieves_target_fpr_synthetic(n_reference=5)


def test_evaluate_detection_delay_and_false_alarm_rate_synthetic():
    result = qde.evaluate_detection_delay_and_false_alarm_rate_synthetic(
        n_points=200, injection_index=100, shift_sigma=4.0, n_trials=50)
    assert result["detections"] > 0
    assert result["mean_detection_delay"] is not None
    assert result["mean_detection_delay"] < 100
    assert result["false_alarm_rate"] < 0.5


def test_evaluate_detection_delay_rejects_bad_injection_index():
    with pytest.raises(QualityDriftError):
        qde.evaluate_detection_delay_and_false_alarm_rate_synthetic(n_points=100, injection_index=0)
    with pytest.raises(QualityDriftError):
        qde.evaluate_detection_delay_and_false_alarm_rate_synthetic(n_points=100, injection_index=100)


def test_quality_drift_eval_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "quality_drift" not in rpc_source
