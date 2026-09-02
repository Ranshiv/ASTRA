"""review_experiment.py: deterministic arm assignment, decoy scoring, and
experimental vote casting (Direction 6, "the review UI as a controlled
experiment")."""

from __future__ import annotations

import pytest

from astra import candidates, review_experiment as rx


@pytest.fixture
def research_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_RESEARCH_ROOT", str(tmp_path))
    return tmp_path


class TestAssignArm:
    def test_is_deterministic_for_the_same_inputs(self):
        first = rx.assign_arm("alice", "ASTRA-000001", seed=0)
        second = rx.assign_arm("alice", "ASTRA-000001", seed=0)
        assert first == second
        assert first in rx.ARMS

    def test_different_seeds_can_change_the_arm(self):
        # Not guaranteed for every pair, but across many pairs at least one
        # must differ, or the seed would not be doing anything.
        pairs = [("alice", f"ASTRA-{i:06d}") for i in range(50)]
        seed0 = [rx.assign_arm(r, c, seed=0) for r, c in pairs]
        seed1 = [rx.assign_arm(r, c, seed=1) for r, c in pairs]
        assert seed0 != seed1

    def test_distributes_reasonably_across_arms(self):
        pairs = [("alice", f"ASTRA-{i:06d}") for i in range(300)]
        arms = [rx.assign_arm(r, c) for r, c in pairs]
        counts = {arm: arms.count(arm) for arm in rx.ARMS}
        for arm in rx.ARMS:
            assert counts[arm] > 50  # roughly a third of 300, not a fluke skew

    def test_different_reviewers_can_get_different_arms_for_the_same_candidate(self):
        reviewers = [f"reviewer{i}" for i in range(30)]
        arms = {rx.assign_arm(reviewer, "ASTRA-000001") for reviewer in reviewers}
        assert len(arms) > 1


class TestPickDecoyCandidateId:
    def test_never_returns_the_candidate_itself(self):
        pool = [f"ASTRA-{i:06d}" for i in range(10)]
        for candidate_id in pool:
            decoy = rx.pick_decoy_candidate_id("alice", candidate_id, pool)
            assert decoy != candidate_id

    def test_is_deterministic(self):
        pool = [f"ASTRA-{i:06d}" for i in range(10)]
        first = rx.pick_decoy_candidate_id("alice", pool[0], pool, seed=0)
        second = rx.pick_decoy_candidate_id("alice", pool[0], pool, seed=0)
        assert first == second

    def test_no_other_candidates_returns_none(self):
        assert rx.pick_decoy_candidate_id("alice", "ASTRA-000001", ["ASTRA-000001"]) is None

    def test_empty_pool_returns_none(self):
        assert rx.pick_decoy_candidate_id("alice", "ASTRA-000001", []) is None


class TestDisplayedScoreFor:
    def test_score_shown_displays_the_real_score(self):
        assert rx.displayed_score_for("score_shown", 0.8, 0.2) == 0.8

    def test_score_blinded_displays_nothing(self):
        assert rx.displayed_score_for("score_blinded", 0.8, 0.2) is None

    def test_score_shuffled_displays_the_decoy(self):
        assert rx.displayed_score_for("score_shuffled", 0.8, 0.2) == 0.2


class TestCastExperimentalVote:
    def test_records_the_arm_and_displayed_score(self, tmp_path):
        score_lookup = {"ASTRA-000001": 0.9, "ASTRA-000002": 0.1}
        vote = rx.cast_experimental_vote(
            "ASTRA-000001", "alice", "interesting", score_lookup=score_lookup,
            root=tmp_path)

        stored = candidates.label_votes("ASTRA-000001", tmp_path)
        assert len(stored) == 1
        assert stored[0]["arm"] == vote["arm"]
        assert stored[0]["arm"] in rx.ARMS

    def test_score_shuffled_arm_shows_a_different_candidates_score(self, tmp_path):
        # Force the shuffled arm by trying reviewer IDs until one lands
        # there -- assignment is deterministic, not seedable per-call here.
        score_lookup = {"ASTRA-000001": 0.9, "ASTRA-000002": 0.1, "ASTRA-000003": 0.5}
        shuffled_vote = None
        for reviewer_index in range(50):
            reviewer_id = f"reviewer{reviewer_index}"
            arm = rx.assign_arm(reviewer_id, "ASTRA-000001")
            if arm == "score_shuffled":
                shuffled_vote = rx.cast_experimental_vote(
                    "ASTRA-000001", reviewer_id, "interesting",
                    score_lookup=score_lookup, root=tmp_path)
                break

        assert shuffled_vote is not None
        assert shuffled_vote["decoy_candidate_id"] in ("ASTRA-000002", "ASTRA-000003")
        assert shuffled_vote["displayed_score"] == score_lookup[shuffled_vote["decoy_candidate_id"]]

    def test_score_blinded_arm_stores_no_displayed_score(self, tmp_path):
        score_lookup = {"ASTRA-000001": 0.9, "ASTRA-000002": 0.1}
        blinded_vote = None
        for reviewer_index in range(50):
            reviewer_id = f"reviewer{reviewer_index}"
            if rx.assign_arm(reviewer_id, "ASTRA-000001") == "score_blinded":
                blinded_vote = rx.cast_experimental_vote(
                    "ASTRA-000001", reviewer_id, "interesting",
                    score_lookup=score_lookup, root=tmp_path)
                break

        assert blinded_vote is not None
        assert blinded_vote["displayed_score"] is None

    def test_experiment_fields_pass_through_to_storage(self, tmp_path):
        vote = rx.cast_experimental_vote(
            "ASTRA-000001", "alice", "interesting",
            score_lookup={"ASTRA-000001": 0.5}, decision_latency_ms=4200,
            self_reported_confidence=0.75, presentation_index=3, root=tmp_path)

        stored = candidates.label_votes("ASTRA-000001", tmp_path)[0]
        assert stored["decision_latency_ms"] == 4200
        assert stored["self_reported_confidence"] == 0.75
        assert stored["presentation_index"] == 3
        assert vote["decision_latency_ms"] == 4200

    def test_a_vote_missing_from_score_lookup_is_treated_like_blinded(self, tmp_path):
        vote = rx.cast_experimental_vote(
            "ASTRA-000001", "alice", "interesting", score_lookup={}, root=tmp_path)
        # Whatever arm this pair lands in, there is no real score to show,
        # so the only possible displayed values are None (blinded/shown-
        # with-nothing-to-show) or a decoy drawn from an empty pool (also
        # None).
        assert vote["displayed_score"] is None

    def test_ordinary_cast_label_vote_leaves_experiment_fields_null(self, tmp_path):
        candidates.cast_label_vote("ASTRA-000001", "alice", "interesting", root=tmp_path)
        stored = candidates.label_votes("ASTRA-000001", tmp_path)[0]
        assert stored["arm"] is None
        assert stored["displayed_score"] is None
        assert stored["decision_latency_ms"] is None


class TestAllLabelVotes:
    def test_spans_multiple_candidates(self, tmp_path):
        candidates.cast_label_vote("ASTRA-000001", "alice", "interesting", root=tmp_path)
        candidates.cast_label_vote("ASTRA-000002", "bob", "artifact", root=tmp_path)

        all_votes = candidates.all_label_votes(tmp_path)
        assert {v["candidate_key"] for v in all_votes} == {"ASTRA-000001", "ASTRA-000002"}


class TestPreregistration:
    def test_saves_the_plan_with_a_content_hash(self, research_root):
        record = rx.save_preregistration()
        assert record["plan"] == rx.PREREGISTERED_ANALYSIS_PLAN
        assert record["content_hash"]

    def test_is_idempotent_for_the_same_plan(self, research_root):
        first = rx.save_preregistration()
        second = rx.save_preregistration()
        assert first == second

    def test_load_returns_none_before_any_save(self, research_root):
        assert rx.load_preregistration() is None

    def test_load_returns_the_saved_record(self, research_root):
        saved = rx.save_preregistration()
        assert rx.load_preregistration() == saved

    def test_refuses_to_overwrite_a_different_registered_plan(self, research_root):
        rx.save_preregistration()
        path = research_root / "experiments" / "preregistrations" / "review_experiment.json"
        import json
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["content_hash"] = "not-the-real-hash"
        path.write_text(json.dumps(tampered), encoding="utf-8")

        with pytest.raises(ValueError, match="already registered"):
            rx.save_preregistration()
