"""Calibration-split validation, conformal p-value arithmetic, and
selective-risk-curve monotonicity for `conformal.py`. No `research`
extra needed (no new optional dependency).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import conformal


# ---------------------------------------------------------------------------
# split_calibration_stream
# ---------------------------------------------------------------------------

def test_split_calibration_stream_partitions_without_overlap():
    scores = np.arange(20, dtype=np.float64)
    calibration, remainder = conformal.split_calibration_stream(scores, calibration_fraction=0.4, seed=1)
    assert len(calibration) + len(remainder) == len(scores)
    assert set(calibration.tolist()).isdisjoint(remainder.tolist())


def test_split_calibration_stream_rejects_bad_inputs():
    with pytest.raises(conformal.ConformalError):
        conformal.split_calibration_stream(np.arange(10), calibration_fraction=0.0)
    with pytest.raises(conformal.ConformalError):
        conformal.split_calibration_stream(np.array([1.0]))


# ---------------------------------------------------------------------------
# conformal_p_values
# ---------------------------------------------------------------------------

def test_conformal_p_values_matches_hand_computed_tail_probability():
    calibration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test = np.array([3.0])
    p_values = conformal.conformal_p_values(test, calibration)
    # count(calibration >= 3.0) = 3 (3,4,5); p = (3+1)/(5+1)
    assert p_values[0] == pytest.approx(4.0 / 6.0)


def test_conformal_p_values_are_uniform_for_exchangeable_normal_draws():
    from scipy import stats

    rng = np.random.default_rng(3)
    calibration = rng.normal(size=500)
    test = rng.normal(size=500)
    p_values = conformal.conformal_p_values(test, calibration)
    # Kolmogorov-Smirnov test against Uniform(0,1): should not reject at a
    # strict alpha for exchangeable draws from the identical distribution.
    ks_stat, p = stats.kstest(p_values, "uniform")
    assert p > 0.01, f"conformal p-values look non-uniform (KS p={p})"


def test_conformal_p_values_rejects_empty_calibration():
    with pytest.raises(conformal.ConformalError):
        conformal.conformal_p_values(np.array([1.0]), np.array([]))


def test_conformal_p_values_handles_empty_test():
    assert len(conformal.conformal_p_values(np.array([]), np.array([1.0, 2.0]))) == 0


# ---------------------------------------------------------------------------
# flag_at_alpha
# ---------------------------------------------------------------------------

def test_flag_at_alpha_flags_small_p_values():
    p_values = np.array([0.01, 0.5, 0.99])
    flags = conformal.flag_at_alpha(p_values, 0.05)
    assert flags.tolist() == [True, False, False]


def test_flag_at_alpha_rejects_bad_alpha():
    with pytest.raises(conformal.ConformalError):
        conformal.flag_at_alpha(np.array([0.5]), 1.5)


# ---------------------------------------------------------------------------
# selective_predict / selective_risk_coverage_curve
# ---------------------------------------------------------------------------

def test_selective_predict_accepts_the_requested_fraction():
    p_values = np.array([0.01, 0.02, 0.4, 0.45, 0.5, 0.55, 0.6, 0.95, 0.97, 0.99])
    decisions = conformal.selective_predict(p_values, coverage=0.5)
    assert sum(d.accepted for d in decisions) == 5
    # The accepted set should be the most extreme (highest-confidence) points.
    accepted_p = sorted(d.p_value for d in decisions if d.accepted)
    assert accepted_p[:2] == [0.01, 0.02]


def test_selective_predict_rejects_bad_coverage():
    with pytest.raises(conformal.ConformalError):
        conformal.selective_predict(np.array([0.5]), coverage=0.0)


def test_selective_risk_coverage_curve_risk_is_non_increasing_as_coverage_shrinks():
    # A perfectly separable synthetic case: normal p-values near 1
    # (large), anomaly p-values near 0 (small) -- risk should shrink (or
    # stay flat) as the accepted fraction shrinks to the most confident
    # points, since those are exactly the correctly-labelled extremes.
    rng = np.random.default_rng(9)
    normal_p = rng.uniform(0.5, 1.0, size=100)
    anomaly_p = rng.uniform(0.0, 0.15, size=20)  # a few near the boundary, mostly confident
    p_values = np.concatenate([normal_p, anomaly_p])
    true_labels = np.concatenate([np.zeros(100), np.ones(20)])

    curve = conformal.selective_risk_coverage_curve(
        p_values, true_labels, coverage_levels=(0.2, 0.5, 1.0))
    risk_by_coverage = {row["coverage"]: row["risk"] for row in curve}
    assert risk_by_coverage[0.2] <= risk_by_coverage[0.5] <= risk_by_coverage[1.0]


def test_selective_risk_coverage_curve_rejects_mismatched_lengths():
    with pytest.raises(conformal.ConformalError):
        conformal.selective_risk_coverage_curve(np.array([0.1, 0.2]), np.array([0]))


def test_conformal_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "conformal" not in rpc_source
