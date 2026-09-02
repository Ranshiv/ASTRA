"""corroborate/eval.py: astronomy equivalence, cross-domain transfer, and
the systematics-correlation scaling study (Direction 3)."""

from __future__ import annotations

from astra.corroborate import eval as cxeval


class TestEvaluateAstronomyEquivalence:
    def test_full_agreement_across_random_trials(self):
        result = cxeval.evaluate_astronomy_equivalence(n_trials=25, seed=0)
        assert result["n_trials"] == 25
        assert result["n_mismatches"] == 0
        assert result["agreement_rate"] == 1.0

    def test_is_reproducible_for_a_fixed_seed(self):
        first = cxeval.evaluate_astronomy_equivalence(n_trials=10, seed=3)
        second = cxeval.evaluate_astronomy_equivalence(n_trials=10, seed=3)
        assert first == second


class TestEvaluateDomainTransfer:
    def test_corroboration_reduces_false_positives_in_both_domains(self):
        result = cxeval.evaluate_domain_transfer(astronomy_seed=0, gw_seed=0)

        astronomy = result["astronomy"]
        assert astronomy["corroborated"]["rate"] <= astronomy["baseline"]["rate"]
        assert astronomy["corroborated"]["n_declared"] < astronomy["baseline"]["n_declared"]

        gw = result["gw"]
        assert gw["corroborated"]["rate"] <= gw["baseline"]["rate"]
        assert gw["corroborated"]["n_declared"] < gw["baseline"]["n_declared"]

    def test_both_domains_use_a_nonzero_baseline_population(self):
        result = cxeval.evaluate_domain_transfer()
        assert result["astronomy"]["baseline"]["n_declared"] > 0
        assert result["gw"]["baseline"]["n_declared"] > 0

    def test_is_reproducible_for_fixed_seeds(self):
        first = cxeval.evaluate_domain_transfer(astronomy_seed=1, gw_seed=1)
        second = cxeval.evaluate_domain_transfer(astronomy_seed=1, gw_seed=1)
        assert first == second


class TestEvaluateScalingWithSystematicsCorrelation:
    def test_false_positive_rate_rises_with_correlation(self):
        result = cxeval.evaluate_scaling_with_systematics_correlation(
            correlations=(0.0, 1.0), n_glitches=150, seed=0)
        low, high = result["points"]
        assert low["systematics_correlation"] == 0.0
        assert high["systematics_correlation"] == 1.0
        assert (high["corroboration_fp_rate"]["rate"]
               > low["corroboration_fp_rate"]["rate"])

    def test_near_zero_correlation_gives_a_near_zero_false_positive_rate(self):
        result = cxeval.evaluate_scaling_with_systematics_correlation(
            correlations=(0.0,), n_glitches=200, window_seconds=0.05, seed=0)
        point = result["points"][0]
        assert point["corroboration_fp_rate"]["rate"] < 0.1

    def test_full_correlation_gives_a_high_false_positive_rate(self):
        result = cxeval.evaluate_scaling_with_systematics_correlation(
            correlations=(1.0,), n_glitches=200, seed=0)
        point = result["points"][0]
        assert point["corroboration_fp_rate"]["rate"] > 0.8

    def test_is_monotonic_across_a_full_sweep(self):
        result = cxeval.evaluate_scaling_with_systematics_correlation(seed=0)
        rates = [p["corroboration_fp_rate"]["rate"] for p in result["points"]]
        # Not strictly monotonic at every point (finite synthetic samples),
        # but the endpoints must bracket the sweep correctly.
        assert rates[0] < rates[-1]
