"""ICCF lag recovery on a known closed-form shift, FR/RSS uncertainty
coverage, and the synthetic lag-recovery study's bias/precision behaviour."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import reverberation as rev


def _shifted_pair(true_lag, span=200.0, cadence=1.0, noise=0.0, seed=0):
    time = np.arange(0.0, span, cadence)
    driving = np.sin(2 * np.pi * time / 40.0) + 0.3 * np.sin(2 * np.pi * time / 13.0)
    response = np.interp(time - true_lag, time, driving)
    if noise > 0:
        rng = np.random.default_rng(seed)
        driving = driving + rng.normal(0.0, noise, size=time.size)
        response = response + rng.normal(0.0, noise, size=time.size)
    err = np.full_like(time, max(noise, 1e-4))
    return time, driving, err, time.copy(), response, err.copy()


# ---------------------------------------------------------------------------
# ICCF
# ---------------------------------------------------------------------------

def test_iccf_recovers_a_known_closed_form_lag_noiseless():
    t1, v1, e1, t2, v2, e2 = _shifted_pair(true_lag=12.0)
    lag_grid = np.arange(-30, 30, 0.5)
    result = rev.interpolated_cross_correlation(t1, v1, t2, v2, lag_grid)
    assert result.peak_lag == pytest.approx(12.0, abs=0.5)
    assert result.centroid_lag == pytest.approx(12.0, abs=0.5)
    assert result.peak_correlation > 0.99


def test_iccf_recovers_lag_with_moderate_noise():
    t1, v1, e1, t2, v2, e2 = _shifted_pair(true_lag=-8.0, noise=0.05, seed=1)
    lag_grid = np.arange(-30, 30, 0.5)
    result = rev.interpolated_cross_correlation(t1, v1, t2, v2, lag_grid)
    assert result.centroid_lag == pytest.approx(-8.0, abs=1.5)


def test_iccf_zero_correlation_is_not_conflated_with_no_overlap():
    # Two independent white-noise series: correlation should hover near
    # zero at most lags, not be silently treated as "no data."
    rng = np.random.default_rng(3)
    time = np.arange(0.0, 100.0, 1.0)
    v1 = rng.normal(0.0, 1.0, size=time.size)
    v2 = rng.normal(0.0, 1.0, size=time.size)
    result = rev.interpolated_cross_correlation(time, v1, time, v2, np.arange(-10, 10, 1.0))
    assert np.any(np.isfinite(result.correlation))


def test_iccf_rejects_too_few_points():
    with pytest.raises(rev.ReverberationError):
        rev.interpolated_cross_correlation(np.arange(3, dtype=float), np.ones(3),
                                           np.arange(3, dtype=float), np.ones(3), np.array([0.0]))


def test_iccf_rejects_empty_lag_grid():
    t1, v1, e1, t2, v2, e2 = _shifted_pair(true_lag=0.0)
    with pytest.raises(rev.ReverberationError):
        rev.interpolated_cross_correlation(t1, v1, t2, v2, np.array([]))


# ---------------------------------------------------------------------------
# FR/RSS uncertainty
# ---------------------------------------------------------------------------

def test_lag_uncertainty_frss_distribution_covers_the_true_lag():
    t1, v1, e1, t2, v2, e2 = _shifted_pair(true_lag=12.0, noise=0.05, seed=2)
    lag_grid = np.arange(-30, 30, 0.5)
    summary = rev.lag_uncertainty_frss(t1, v1, e1, t2, v2, e2, lag_grid, n_trials=40, seed=5)
    assert summary["ci95"] is not None
    assert summary["ci95"][0] <= 12.0 <= summary["ci95"][1]


def test_lag_uncertainty_frss_rejects_bad_trial_count():
    t1, v1, e1, t2, v2, e2 = _shifted_pair(true_lag=0.0)
    with pytest.raises(rev.ReverberationError):
        rev.lag_uncertainty_frss(t1, v1, e1, t2, v2, e2, np.array([0.0]), n_trials=0)


# ---------------------------------------------------------------------------
# Synthetic lag-recovery study (celerite2-backed driving curve)
# ---------------------------------------------------------------------------

celerite2 = pytest.importorskip("celerite2", reason="celerite2 not installed (opt-in 'research' extra)")


def test_evaluate_lag_recovery_reports_small_bias_for_a_clear_signal():
    result = rev.evaluate_lag_recovery(
        true_lag_days=15.0, transfer_width_days=3.0, span_days=200.0,
        cadence_days=2.0, noise_sigma=0.02, n_trials=5, seed=2)
    assert result.recovered_lag["n_trials_used"] > 0
    assert result.recovered_lag["mean"] == pytest.approx(15.0, abs=3.0)


def test_evaluate_lag_recovery_rejects_bad_transfer_width():
    with pytest.raises(rev.ReverberationError):
        rev.evaluate_lag_recovery(true_lag_days=10.0, transfer_width_days=0.0)


def test_evaluate_lag_recovery_rejects_bad_trial_count():
    with pytest.raises(rev.ReverberationError):
        rev.evaluate_lag_recovery(true_lag_days=10.0, transfer_width_days=2.0, n_trials=0)


def test_reverberation_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "reverberation" not in rpc_source
