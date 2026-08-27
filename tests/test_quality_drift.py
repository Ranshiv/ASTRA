"""CUSUM arithmetic, change-point detection, and bootstrap-threshold
calibration correctness for `quality_drift.py`. No `research` extra
needed (no new optional dependency).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import quality_drift as qd


# ---------------------------------------------------------------------------
# standardize_stream / cusum_statistics
# ---------------------------------------------------------------------------

def test_standardize_stream_zero_scores_at_the_reference_mean():
    values = np.array([5.0, 5.0, 5.0])
    z = qd.standardize_stream(values, reference_mean=5.0, reference_std=2.0)
    assert np.allclose(z, 0.0)


def test_standardize_stream_rejects_non_positive_std():
    with pytest.raises(qd.QualityDriftError):
        qd.standardize_stream(np.array([1.0]), reference_mean=0.0, reference_std=0.0)


def test_cusum_statistics_grows_under_a_sustained_positive_shift():
    z = np.full(20, 2.0)
    pos, neg = qd.cusum_statistics(z, drift=0.5)
    assert pos[-1] > pos[0]
    assert np.all(neg == 0.0)


def test_cusum_statistics_rejects_negative_drift():
    with pytest.raises(qd.QualityDriftError):
        qd.cusum_statistics(np.array([0.0]), drift=-1.0)


# ---------------------------------------------------------------------------
# detect_changepoints
# ---------------------------------------------------------------------------

def test_detect_changepoints_finds_an_injected_upward_shift():
    rng = np.random.default_rng(0)
    z = rng.normal(size=100)
    z[50:] += 4.0
    events = qd.detect_changepoints(z, drift=0.5, threshold=5.0)
    assert any(e.direction == "upper" and e.index >= 50 for e in events)


def test_detect_changepoints_stays_silent_on_pure_noise():
    rng = np.random.default_rng(1)
    z = rng.normal(size=100)
    events = qd.detect_changepoints(z, drift=0.5, threshold=8.0)
    assert events == []


def test_detect_changepoints_resets_after_an_alarm():
    z = np.concatenate([np.full(10, 3.0), np.zeros(5), np.full(10, 3.0)])
    events = qd.detect_changepoints(z, drift=0.5, threshold=5.0)
    assert len(events) >= 2


def test_detect_changepoints_rejects_non_positive_threshold():
    with pytest.raises(qd.QualityDriftError):
        qd.detect_changepoints(np.array([0.0]), threshold=0.0)


# ---------------------------------------------------------------------------
# calibrate_threshold
# ---------------------------------------------------------------------------

def test_calibrate_threshold_is_positive_and_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(2)
    nominal = rng.normal(size=200)
    first = qd.calibrate_threshold(nominal, seed=7)
    second = qd.calibrate_threshold(nominal, seed=7)
    assert first == second
    assert first > 0


def test_calibrate_threshold_rejects_too_few_points():
    with pytest.raises(qd.QualityDriftError):
        qd.calibrate_threshold(np.arange(5, dtype=np.float64))


def test_calibrate_threshold_rejects_zero_variance_stream():
    with pytest.raises(qd.QualityDriftError):
        qd.calibrate_threshold(np.full(20, 3.0))


# ---------------------------------------------------------------------------
# monitor_stream / monitor_multiple_channels
# ---------------------------------------------------------------------------

def test_monitor_multiple_channels_runs_each_channel_independently():
    rng = np.random.default_rng(3)
    channels = {"zero_point": rng.normal(size=50), "seeing": rng.normal(loc=10, size=50)}
    references = {"zero_point": (0.0, 1.0), "seeing": (10.0, 1.0)}
    result = qd.monitor_multiple_channels(channels, references, threshold=6.0)
    assert set(result) == {"zero_point", "seeing"}


def test_monitor_multiple_channels_rejects_missing_reference():
    with pytest.raises(qd.QualityDriftError):
        qd.monitor_multiple_channels({"a": np.array([1.0])}, {}, threshold=5.0)


def test_quality_drift_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "quality_drift" not in rpc_source
