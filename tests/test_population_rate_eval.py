"""Synthetic-population generation, rate-posterior coverage, and the
anchor-survey bias comparison for `population_rate_eval.py`.

Gated on `emcee` (via a module-level flag, not `importorskip`, so the
synthesis/validation tests that don't need a fit still run) the same way
`test_population_rate.py` is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import population_rate_eval as pre

try:
    import emcee  # noqa: F401
    _HAS_EMCEE = True
except ImportError:
    _HAS_EMCEE = False

requires_emcee = pytest.mark.skipif(
    not _HAS_EMCEE, reason="emcee not installed (opt-in 'research' extra)")


# ---------------------------------------------------------------------------
# synthesize_population_strata
# ---------------------------------------------------------------------------

def test_synthesize_population_strata_produces_the_requested_count():
    rng = np.random.default_rng(1)
    strata, truth = pre.synthesize_population_strata(
        rng, n_strata=6, true_mu_log_rate=-3.0, true_sigma_log_rate=0.4)
    assert len(strata) == 6
    assert set(truth["rates"]) == {s.label for s in strata}
    assert truth["mu_log_rate"] == -3.0


def test_synthesize_population_strata_rejects_bad_n_strata():
    rng = np.random.default_rng(1)
    with pytest.raises(pre.PopulationRateError):
        pre.synthesize_population_strata(rng, n_strata=0, true_mu_log_rate=-3.0, true_sigma_log_rate=0.3)


# ---------------------------------------------------------------------------
# evaluate_rate_posterior_coverage
# ---------------------------------------------------------------------------

@requires_emcee
def test_evaluate_rate_posterior_coverage_reports_a_plausible_empirical_rate():
    result = pre.evaluate_rate_posterior_coverage(
        n_trials=15, levels=(0.68, 0.9), seed=5, true_mu_log_rate=-3.0,
        true_sigma_log_rate=0.3, n_strata=6, n_steps=1200, n_walkers=24)
    assert result["n_trials"] == 15
    for level_key in ("0.68", "0.9"):
        entry = result["levels"][level_key]["mu_log_rate"]
        assert entry["trials"] > 0
        assert 0.0 <= entry["empirical"] <= 1.0


def test_evaluate_rate_posterior_coverage_rejects_bad_n_trials():
    with pytest.raises(pre.PopulationRateError):
        pre.evaluate_rate_posterior_coverage(n_trials=0, true_mu_log_rate=-3.0, true_sigma_log_rate=0.3)


# ---------------------------------------------------------------------------
# evaluate_anchor_survey_bias
# ---------------------------------------------------------------------------

def test_evaluate_anchor_survey_bias_shows_the_naive_arm_is_worse_than_the_corrected_arm():
    rng = np.random.default_rng(21)
    result = pre.evaluate_anchor_survey_bias(
        rng, true_rate_deg2_day=0.01,
        surveys=("wide_shallow", "narrow_deep"),
        completeness_by_survey={"wide_shallow": 0.95, "narrow_deep": 0.3},
        footprint_area_deg2=200.0, baseline_days=200.0)
    assert result["n_true_objects"] > 0
    # The corrected arm divides each anchor's own count by its own real
    # completeness and should land close to the true rate for both
    # anchors; the naive arm (one shared completeness) should not.
    assert result["naive_rate_spread"] > result["corrected_rate_spread"]
    for anchor, rate in result["corrected_rate_by_anchor"].items():
        assert rate == pytest.approx(result["true_rate_deg2_day"], rel=0.5)


def test_evaluate_anchor_survey_bias_rejects_mismatched_survey_keys():
    rng = np.random.default_rng(1)
    with pytest.raises(pre.PopulationRateError):
        pre.evaluate_anchor_survey_bias(
            rng, surveys=("a", "b"), completeness_by_survey={"a": 0.5})


def test_population_rate_eval_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "population_rate" not in rpc_source
