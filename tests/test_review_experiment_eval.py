"""review_experiment_eval.py: anchoring effect size, calibration, and
ensemble accounting for the reviewer human-factors experiment (Direction 6).
Pure functions over synthetic vote rows -- no database needed."""

from __future__ import annotations

from astra import review_experiment_eval as rxe


def _vote(candidate_key, reviewer_id, label, arm, *, confidence=None):
    return {"candidate_key": candidate_key, "reviewer_id": reviewer_id, "label": label,
           "arm": arm, "self_reported_confidence": confidence}


class TestStoppingRuleGate:
    def test_default_threshold_matches_preregistered_plan(self):
        from astra.review_experiment import PREREGISTERED_ANALYSIS_PLAN
        truth = {"c1": True}
        votes = [_vote("c1", "r1", "interesting", "score_shown")]
        result = rxe.anchoring_effect_size(votes, truth)
        assert result["min_votes_per_arm"] == PREREGISTERED_ANALYSIS_PLAN["minimum_votes_per_arm"]

    def test_not_ready_below_threshold_and_verdict_forced_none(self):
        truth = {f"c{i}": True for i in range(5)}
        votes = [_vote(f"c{i}", "r1", "interesting", arm)
                for i in range(5) for arm in rxe.ARMS]
        result = rxe.anchoring_effect_size(votes, truth, min_votes_per_arm=30)
        assert result["ready"] is False
        assert set(result["underpowered_arms"]) == set(rxe.ARMS)
        assert result["anchoring_signature_detected"] is None

    def test_ready_once_every_arm_meets_the_threshold(self):
        truth = {f"c{i}": True for i in range(5)}
        votes = [_vote(f"c{i % 5}", f"r{i}", "interesting", arm)
                for i in range(5) for arm in rxe.ARMS]
        result = rxe.anchoring_effect_size(votes, truth, min_votes_per_arm=5)
        assert result["ready"] is True
        assert result["underpowered_arms"] == []

    def test_partially_underpowered_names_only_the_short_arms(self):
        truth = {"c1": True}
        votes = ([_vote("c1", f"r{i}", "interesting", "score_shown") for i in range(5)]
                + [_vote("c1", "r_x", "interesting", "score_blinded")])
        result = rxe.anchoring_effect_size(votes, truth, min_votes_per_arm=5)
        assert result["ready"] is False
        assert "score_shown" not in result["underpowered_arms"]
        assert "score_blinded" in result["underpowered_arms"]
        assert "score_shuffled" in result["underpowered_arms"]


class TestAnchoringEffectSize:
    def test_per_arm_accuracy_against_truth(self):
        truth = {"c1": True, "c2": False, "c3": True, "c4": False}
        votes = [
            _vote("c1", "r1", "interesting", "score_shown"),   # correct
            _vote("c2", "r1", "interesting", "score_shown"),   # wrong (truth False)
            _vote("c3", "r2", "interesting", "score_blinded"),  # correct
            _vote("c4", "r2", "artifact", "score_blinded"),     # correct
        ]
        result = rxe.anchoring_effect_size(votes, truth)

        assert result["by_arm"]["score_shown"]["n_votes"] == 2
        assert result["by_arm"]["score_shown"]["accuracy"] == 0.5
        assert result["by_arm"]["score_blinded"]["n_votes"] == 2
        assert result["by_arm"]["score_blinded"]["accuracy"] == 1.0

    def test_votes_without_an_arm_are_excluded(self):
        truth = {"c1": True}
        votes = [_vote("c1", "r1", "interesting", None)]
        result = rxe.anchoring_effect_size(votes, truth)
        assert result["by_arm"]["score_shown"]["n_votes"] == 0
        assert result["by_arm"]["score_shown"]["accuracy"] is None

    def test_votes_for_candidates_without_truth_are_excluded(self):
        truth = {"c1": True}
        votes = [_vote("c2", "r1", "interesting", "score_shown")]
        result = rxe.anchoring_effect_size(votes, truth)
        assert result["by_arm"]["score_shown"]["n_votes"] == 0

    def test_cross_reviewer_agreement_within_one_arm(self):
        truth = {"c1": True}
        votes = [
            _vote("c1", "r1", "interesting", "score_shown"),
            _vote("c1", "r2", "interesting", "score_shown"),
            _vote("c1", "r3", "artifact", "score_shown"),
        ]
        result = rxe.anchoring_effect_size(votes, truth)
        # 3 pairs total: (r1,r2) agree, (r1,r3) disagree, (r2,r3) disagree
        # -> agreement 1/3.
        assert result["by_arm"]["score_shown"]["cross_reviewer_agreement"] == round(1 / 3, 4)

    def test_detects_the_anchoring_signature(self):
        truth = {f"c{i}": (i % 2 == 0) for i in range(10)}
        votes = []
        for i in range(10):
            candidate = f"c{i}"
            correct_label = "interesting" if truth[candidate] else "artifact"
            wrong_label = "artifact" if truth[candidate] else "interesting"
            # score_shown: consistently wrong (anchored on a misleading
            # displayed number). score_blinded/score_shuffled: consistently
            # correct and identical to each other.
            votes.append(_vote(candidate, "r1", wrong_label, "score_shown"))
            votes.append(_vote(candidate, "r2", correct_label, "score_blinded"))
            votes.append(_vote(candidate, "r3", correct_label, "score_shuffled"))

        result = rxe.anchoring_effect_size(votes, truth, min_votes_per_arm=1)
        assert result["ready"] is True
        assert result["anchoring_signature_detected"] is True

    def test_none_when_an_arm_has_no_usable_votes(self):
        truth = {"c1": True}
        votes = [_vote("c1", "r1", "interesting", "score_shown")]
        result = rxe.anchoring_effect_size(votes, truth)
        assert result["anchoring_signature_detected"] is None


class TestCalibrationCurve:
    def test_bins_confidence_and_computes_accuracy_per_bin(self):
        truth = {"c1": True, "c2": True, "c3": False}
        votes = [
            _vote("c1", "r1", "interesting", "score_shown", confidence=0.95),
            _vote("c2", "r1", "interesting", "score_shown", confidence=0.9),
            _vote("c3", "r1", "interesting", "score_shown", confidence=0.1),  # wrong, low conf
        ]
        result = rxe.calibration_curve(votes, truth, n_bins=5)

        high_bin = result["by_arm"]["score_shown"][-1]
        assert high_bin["n"] == 2
        assert high_bin["accuracy"] == 1.0
        low_bin = result["by_arm"]["score_shown"][0]
        assert low_bin["n"] == 1
        assert low_bin["accuracy"] == 0.0

    def test_votes_without_confidence_are_excluded(self):
        truth = {"c1": True}
        votes = [_vote("c1", "r1", "interesting", "score_shown", confidence=None)]
        result = rxe.calibration_curve(votes, truth, n_bins=4)
        assert all(b["n"] == 0 for b in result["by_arm"]["score_shown"])

    def test_empty_bins_report_none_not_zero(self):
        result = rxe.calibration_curve([], {}, n_bins=4)
        for arm_bins in result["by_arm"].values():
            for bucket in arm_bins:
                assert bucket["n"] == 0
                assert bucket["accuracy"] is None


class TestEnsembleAccounting:
    def test_combined_can_correct_a_model_error_the_human_catches(self):
        truth = {"c1": True, "c2": False, "c3": True, "c4": False}
        model_scores = {"c1": 0.9, "c2": 0.9, "c3": 0.9, "c4": 0.1}  # c2 wrong, confidently
        votes = [
            _vote("c1", "r1", "interesting", "score_blinded"),
            _vote("c2", "r1", "artifact", "score_blinded"),  # human correctly flags c2
            _vote("c3", "r1", "interesting", "score_blinded"),
            _vote("c4", "r1", "artifact", "score_blinded"),
        ]
        result = rxe.ensemble_accounting(votes, truth, model_scores)

        assert result["n_candidates"] == 4
        assert result["model_accuracy"] == 0.75  # wrong on c2
        assert result["human_accuracy"] == 1.0
        assert result["combined_accuracy"] >= result["model_accuracy"]

    def test_only_shown_and_shuffled_arm_votes_are_ignored_for_the_human_signal(self):
        truth = {"c1": True}
        model_scores = {"c1": 0.9}
        votes = [_vote("c1", "r1", "artifact", "score_shown")]  # not blinded -> excluded
        result = rxe.ensemble_accounting(votes, truth, model_scores)
        assert result["n_candidates"] == 0
        assert result["human_accuracy"] is None

    def test_no_overlap_between_truth_model_and_human_yields_no_candidates(self):
        result = rxe.ensemble_accounting([], {}, {})
        assert result["n_candidates"] == 0
        assert result["model_accuracy"] is None
