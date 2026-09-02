"""Injection-recovery study for discard_pile.py (Direction 2 evaluation
harness)."""

from __future__ import annotations

from astra.discard_pile_eval import evaluate_coherence_precision, evaluate_injection_recovery


class TestEvaluateInjectionRecovery:
    def test_separates_real_from_artifact_runs_above_chance(self):
        result = evaluate_injection_recovery(n_objects=120, seed=0)
        assert result["auprc"] > 0.7  # chance-level AUPRC at 50/50 balance is ~0.5
        assert result["n_real"] + result["n_artifact"] == result["n_objects"]

    def test_is_reproducible_for_a_fixed_seed(self):
        first = evaluate_injection_recovery(n_objects=60, seed=3)
        second = evaluate_injection_recovery(n_objects=60, seed=3)
        assert first == second

    def test_most_runs_survive_the_min_run_length_bar(self):
        result = evaluate_injection_recovery(n_objects=80, run_length=5,
                                             min_run_length=3, seed=5)
        assert result["runs_surviving_min_run_length"] == result["n_objects"]

    def test_reports_a_bootstrap_confidence_interval(self):
        result = evaluate_injection_recovery(n_objects=100, seed=9)
        low, high = result["auprc_ci"]
        assert low <= result["auprc"] <= high


class TestEvaluateCoherencePrecision:
    def test_the_boolean_verdict_is_high_precision_but_conservative(self):
        # Measured, not assumed: `DiscardRecord.coherent`'s `max_step <
        # offset * 1.5` bar rarely calls a noise-shaped run "real" (few
        # false positives) but also misses many genuine half-sine bumps
        # whose largest single step is comparable to the mean offset (low
        # recall). `evaluate_injection_recovery`'s continuous discriminator
        # ranks the same runs far better (AUPRC > 0.7) -- this test
        # documents that gap rather than assuming the boolean alone is a
        # strong classifier.
        result = evaluate_coherence_precision(n_objects=150, seed=1)
        assert result["precision"] > 0.8
        assert 0.15 < result["recall"] < 0.6

    def test_confusion_counts_sum_to_n_objects(self):
        result = evaluate_coherence_precision(n_objects=90, seed=2)
        total = (result["true_positive"] + result["false_positive"]
                + result["false_negative"] + result["true_negative"])
        assert total == result["n_objects"]

    def test_is_reproducible_for_a_fixed_seed(self):
        first = evaluate_coherence_precision(n_objects=50, seed=4)
        second = evaluate_coherence_precision(n_objects=50, seed=4)
        assert first == second
