"""Artifact likelihood assessment and its synthetic-injection calibration."""

from __future__ import annotations

import numpy as np
import pytest

from astra import artifact


class TestAssessReadsWeights:
    def test_indicator_weight_comes_from_the_weights_dict(self, monkeypatch):
        monkeypatch.setitem(artifact.WEIGHTS, "sampling_period", 0.77)
        result = artifact.assess({"best_period_days": 1.0}, resolved_surveys=2)

        indicator = next(i for i in result.indicators if i.name == "sampling_period")
        assert indicator.weight == pytest.approx(0.77)

    def test_feature_indicator_names_are_a_subset_of_weights(self):
        assert set(artifact.FEATURE_INDICATOR_NAMES) <= set(artifact.WEIGHTS)


class TestSyntheticGenerators:
    """Each generator must trip its OWN indicator via the real
    features.extract()/artifact.assess() code paths, and a clean synthetic
    "real" curve must trip none of them -- these are the ground-truth
    labels calibrate_from_injection() treats as true by construction."""

    @pytest.mark.parametrize("kind", artifact.FEATURE_INDICATOR_NAMES)
    def test_each_kind_fires_its_own_indicator(self, kind):
        rng = np.random.default_rng(1)
        for i in range(3):
            curve = artifact._synthetic_artifact(rng, kind, i)
            fired = artifact._fired_indicators(curve)
            assert kind in fired, f"{kind} did not fire on its own injected defect"

    def test_real_curves_fire_no_feature_indicator(self):
        rng = np.random.default_rng(1)
        for i in range(5):
            curve = artifact._synthetic_real(rng, i)
            assert artifact._fired_indicators(curve) == set()

    def test_unknown_kind_is_rejected(self):
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError, match="unknown artifact kind"):
            artifact._synthetic_artifact(rng, "not_a_real_kind", 0)


class TestLikelihoodFromFired:
    def test_no_indicators_fired_gives_zero_likelihood(self):
        assert artifact._likelihood_from_fired(set(), artifact.WEIGHTS) == 0.0

    def test_matches_the_noisy_or_formula(self):
        weights = {"a": 0.5, "b": 0.5}
        # 1 - (1-0.5)(1-0.5) = 0.75
        assert artifact._likelihood_from_fired({"a", "b"}, weights) == pytest.approx(0.75)


class TestCalibrateFromInjection:
    """Small n_per_class / few seeds here -- this exercises the real
    features.extract() pipeline (Lomb-Scargle included) end to end, so kept
    deliberately small for test runtime rather than statistically precise.
    See `TestHardRealPopulationDifferentiatesIndicators` below for the
    larger, still-bounded run that checks the harder population actually
    does its job (pushes indicator precision below the ceiling)."""

    def test_report_covers_every_feature_indicator(self):
        report = artifact.calibrate_from_injection(n_per_class=12, seeds=(1, 2))
        names = {i.name for i in report.indicators}

        assert names == set(artifact.FEATURE_INDICATOR_NAMES)

    def test_weights_and_auc_are_valid_probabilities(self):
        report = artifact.calibrate_from_injection(n_per_class=12, seeds=(1, 2))

        for indicator in report.indicators:
            assert 0.0 <= indicator.weight <= 1.0
            assert indicator.support >= 0
        assert 0.0 <= report.auc_old_weights <= 1.0
        assert 0.0 <= report.auc_new_weights <= 1.0

    def test_result_is_reproducible_for_the_same_seeds(self):
        first = artifact.calibrate_from_injection(n_per_class=12, seeds=(3, 5))
        second = artifact.calibrate_from_injection(n_per_class=12, seeds=(3, 5))

        assert first.to_dict() == second.to_dict()

    def test_train_test_split_covers_the_full_synthetic_set(self):
        report = artifact.calibrate_from_injection(
            n_per_class=12, test_fraction=0.3, seeds=(1,))
        # 2 classes x n_per_class, split once per seed.
        assert report.n_train + report.n_test == 24

    def test_to_dict_is_json_serialisable(self):
        import json

        report = artifact.calibrate_from_injection(n_per_class=12, seeds=(1,))
        json.dumps(report.to_dict())  # must not raise

    def test_no_weight_can_saturate_the_noisy_or(self):
        """A weight of 1.0 would make one indicator proof on its own.

        Indicators combine by noisy-OR, so at weight 1.0 the product saturates:
        the likelihood pins at 100% the instant that indicator fires and every
        other indicator, clearing evidence included, stops mattering.
        """
        report = artifact.calibrate_from_injection(n_per_class=12, seeds=(1, 2))

        for indicator in report.indicators:
            assert indicator.weight <= artifact.MAX_CALIBRATED_WEIGHT


class TestHardRealPopulationDifferentiatesIndicators:
    """Regression check for the harder `_synthetic_real(hard=True)`
    population: before the `coincidental_period`/`quiescent`/
    `real_outburst` flavours were added, 5 of the 6 calibrated indicators
    pinned at exactly `MAX_CALIBRATED_WEIGHT` (measured: only
    `sparse_sampling` calibrated below it), because the ceiling is a HARD
    CAP on the reported weight -- a 5/6 pin meant those five indicators'
    real synthetic false-positive rate never pushed empirical precision
    below 0.85 in the first place, not that the ceiling was merely hiding
    a lower number. This is a real, larger run (not the n_per_class=12
    smoke tests above) that checks the harder population actually
    generates enough real false positives for MORE than one indicator to
    calibrate below the ceiling -- a specific, checkable claim about
    what changed, not just "the code runs"."""

    def test_more_than_one_indicator_calibrates_below_the_ceiling(self):
        report = artifact.calibrate_from_injection(n_per_class=60, seeds=(1, 2, 3))
        below_ceiling = [i for i in report.indicators
                         if i.weight < artifact.MAX_CALIBRATED_WEIGHT]
        assert len(below_ceiling) >= 2

    def test_sampling_period_and_consistent_with_constant_are_now_reachable(self):
        # These two indicators had NO false-positive-generating hard-real
        # case at all before `coincidental_period`/`quiescent` were added
        # -- confirm the new flavours give them real support (fired on at
        # least one held-out real object) rather than remaining unreachable.
        report = artifact.calibrate_from_injection(n_per_class=60, seeds=(1, 2, 3))
        by_name = {i.name: i for i in report.indicators}
        assert by_name["sampling_period"].support > 0
        assert by_name["consistent_with_constant"].support > 0


class TestSmoothedPrecision:
    def test_small_support_is_pulled_toward_a_half(self):
        """Three firings out of three is not evidence of a perfect indicator."""
        confident = artifact.smoothed_precision(200, 200)
        thin = artifact.smoothed_precision(3, 3)

        assert thin < confident
        assert 0.5 < thin < artifact.MAX_CALIBRATED_WEIGHT

    def test_never_returns_one(self):
        assert artifact.smoothed_precision(10_000, 10_000) <= artifact.MAX_CALIBRATED_WEIGHT

    def test_a_wrong_indicator_scores_low(self):
        assert artifact.smoothed_precision(0, 40) < 0.2

    def test_zero_support_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError):
            artifact.smoothed_precision(0, 0)


class TestHardRealPopulation:
    def test_hard_reals_are_drawn_differently_from_clean_ones(self):
        """The awkward population must actually differ, or it measures nothing."""
        rng = np.random.default_rng(7)
        clean = [len(artifact._synthetic_real(rng, i, hard=False)) for i in range(30)]
        rng = np.random.default_rng(7)
        hard = [len(artifact._synthetic_real(rng, i, hard=True)) for i in range(30)]

        assert clean != hard

    def test_hard_reals_raise_a_measurable_false_positive_rate(self):
        """Some indicator must fire on a genuine object, or nothing is learned.

        Calibrating against clean variables alone is what returned weights of
        0.95-1.0: no real object could ever contradict an indicator.
        """
        rng = np.random.default_rng(11)
        fired = [artifact._fired_indicators(artifact._synthetic_real(rng, i, hard=True))
                 for i in range(40)]

        assert any(names for names in fired)
