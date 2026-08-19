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
    The real calibration numbers this produced are recorded in
    docs/DEFERRED.txt, not asserted here."""

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
