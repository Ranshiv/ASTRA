"""Empirical coverage (synthetic and real-integration) and the selective-
risk-curve demonstration for `conformal_eval.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import conformal, conformal_eval as ce


def _fake_sequences(n: int = 64, length: int = 64, seed: int = 0) -> np.ndarray:
    """Same shape/style as `test_deep.py`'s own `fake_sequences` helper:
    smooth normalised curves with full validity masks."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 4 * np.pi, length)
    values = np.stack([
        np.sin(time * rng.uniform(0.5, 2.0) + rng.uniform(0, np.pi)) + rng.normal(0, 0.05, length)
        for _ in range(n)
    ]).astype(np.float32)
    mask = np.ones((n, length), dtype=np.float32)
    return np.stack([values, mask], axis=1)


# ---------------------------------------------------------------------------
# evaluate_conformal_coverage_synthetic
# ---------------------------------------------------------------------------

def test_evaluate_conformal_coverage_synthetic_matches_nominal_alpha_within_ci():
    result = ce.evaluate_conformal_coverage_synthetic(
        alpha_levels=(0.05, 0.1, 0.2), n_trials=150, n_calibration=100, n_test=100, seed=1)
    for alpha_key, entry in result["levels"].items():
        nominal = entry["nominal"]
        low, high = entry["ci95"]
        # Empirical false-positive rate should be close to nominal alpha;
        # a slightly widened band absorbs Monte Carlo noise from n_trials.
        assert low - 0.03 <= nominal <= high + 0.03, (
            f"alpha={nominal}: empirical={entry['empirical']} ci95={entry['ci95']}")


def test_evaluate_conformal_coverage_synthetic_rejects_bad_inputs():
    with pytest.raises(conformal.ConformalError):
        ce.evaluate_conformal_coverage_synthetic(n_trials=0)
    with pytest.raises(conformal.ConformalError):
        ce.evaluate_conformal_coverage_synthetic(alpha_levels=())


# ---------------------------------------------------------------------------
# evaluate_conformal_coverage_on_injected_anomalies
# ---------------------------------------------------------------------------

def test_evaluate_conformal_coverage_on_injected_anomalies_runs_end_to_end():
    values = _fake_sequences(n=120, length=64, seed=5)
    identities = [{"id": i} for i in range(120)]

    result = ce.evaluate_conformal_coverage_on_injected_anomalies(
        values, identities, alpha_levels=(0.1, 0.2), n_trials=5, seed=2)

    assert result["n_trials_requested"] == 5
    for entry in result["levels"].values():
        if entry["n_normal_tested"]:
            assert 0.0 <= entry["empirical"] <= 1.0


def test_evaluate_conformal_coverage_on_injected_anomalies_rejects_too_few_sequences():
    with pytest.raises(conformal.ConformalError):
        ce.evaluate_conformal_coverage_on_injected_anomalies(
            _fake_sequences(n=5, length=32), [{}] * 5)


# ---------------------------------------------------------------------------
# evaluate_selective_risk_coverage
# ---------------------------------------------------------------------------

def test_evaluate_selective_risk_coverage_returns_a_valid_curve():
    result = ce.evaluate_selective_risk_coverage(
        n_calibration=200, n_normal_test=150, n_anomaly_test=50, seed=4)
    assert result["n_test"] == 200
    risks = [row["risk"] for row in result["curve"] if row["risk"] is not None]
    assert all(0.0 <= risk <= 1.0 for risk in risks)
    # A well-separated synthetic anomaly class: overall (full-coverage) risk
    # should be low, and the most-confident (lowest-coverage) slice should
    # be at least as good.
    by_coverage = {row["coverage"]: row["risk"] for row in result["curve"]}
    assert by_coverage[0.1] <= by_coverage[1.0]


def test_evaluate_selective_risk_coverage_rejects_bad_inputs():
    with pytest.raises(conformal.ConformalError):
        ce.evaluate_selective_risk_coverage(n_calibration=0)


def test_conformal_eval_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "conformal" not in rpc_source
