"""Candidate assembly, explanation, ranking and human labelling."""

from __future__ import annotations

import numpy as np
import pytest

from astra import candidates
from astra.surveys.base import LightCurve, SourceRef


def features(**overrides) -> dict[str, float]:
    base = {
        "n_points": 500.0, "robust_amplitude": 0.5, "median_err": 0.02,
        "reduced_chi2": 150.0, "period_snr": 30.0, "eta": 0.5,
        "change_point_score": 4.0, "best_period_days": 0.5668,
        "kurtosis": 0.1, "time_span_days": 1500.0,
    }
    base.update(overrides)
    return base


IDENTITY = {"object_id": "728116300014796", "survey": "ZTF", "band": "g",
            "ra_deg": 291.3663, "dec_deg": 42.7844}


class TestExplanation:
    def test_candidate_answers_every_section_17_question(self):
        candidate = candidates.build_candidate(1, IDENTITY, features(),
                                               anomaly_score=0.9)
        explanation = candidate.explanation

        assert explanation["what_happened"]
        assert explanation["why_flagged"]
        assert explanation["supporting_observations"]["epochs"] == 500
        assert explanation["could_be_artifact"]["verdict"]
        assert isinstance(explanation["resembles"], list)
        assert explanation["recommended_actions"]

    def test_behaviour_description_mentions_the_period(self):
        text = candidates.describe_behaviour(features())
        assert "0.5668" in text
        assert "periodicity" in text

    def test_behaviour_description_notes_a_step_change(self):
        text = candidates.describe_behaviour(features(change_point_score=80.0))
        assert "abrupt level change" in text

    def test_empty_features_do_not_crash_the_description(self):
        assert candidates.describe_behaviour({}) == \
            "Insufficient data to describe."

    def test_why_flagged_names_the_actual_drivers(self):
        candidate = candidates.build_candidate(1, IDENTITY, features(),
                                               anomaly_score=1.0)
        assert any("statistical rarity" in reason
                   for reason in candidate.explanation["why_flagged"])


class TestResembles:
    def test_rr_lyrae_period_suggests_rr_lyrae(self):
        assert "RR Lyrae pulsator" in candidates.resembles(0.5668)

    def test_long_period_suggests_a_cepheid(self):
        assert "Classical Cepheid" in candidates.resembles(10.0)

    def test_missing_period_suggests_nothing(self):
        assert candidates.resembles(None) == []
        assert candidates.resembles(float("nan")) == []


class TestRecommendations:
    def test_single_survey_prompts_seeking_corroboration(self):
        candidate = candidates.build_candidate(1, IDENTITY, features(),
                                               resolved_surveys=1)
        actions = candidate.explanation["recommended_actions"]
        assert any("independent detection" in a for a in actions)

    def test_sparse_curve_prompts_more_epochs(self):
        candidate = candidates.build_candidate(1, IDENTITY,
                                               features(n_points=40.0))
        assert any("more epochs" in a
                   for a in candidate.explanation["recommended_actions"])

    def test_likely_artifact_prompts_inspection_first(self):
        candidate = candidates.build_candidate(
            1, IDENTITY,
            features(robust_amplitude=0.005, median_err=0.05,
                     reduced_chi2=1.0, n_points=15.0, kurtosis=150.0),
            resolved_surveys=1)
        assert any("instrumental explanation" in a
                   for a in candidate.explanation["recommended_actions"])

    def test_missing_parallax_prompts_astrometry(self):
        candidate = candidates.build_candidate(1, IDENTITY, features())
        assert any("parallax" in a
                   for a in candidate.explanation["recommended_actions"])

    def test_there_is_always_at_least_one_action(self):
        candidate = candidates.build_candidate(
            1, IDENTITY, features(), resolved_surveys=3,
            gaia_properties={"abs_g_mag": 0.6, "parallax_snr": 30.0,
                             "phot_variable_flag": "VARIABLE"})
        assert len(candidate.explanation["recommended_actions"]) >= 1


class TestRanking:
    def _candidate(self, index, score, artifact_likelihood):
        candidate = candidates.Candidate(
            candidate_id=candidates.make_candidate_id(index),
            object_id=f"o{index}", survey="ZTF", band="g",
            ra_deg=0.0, dec_deg=0.0,
            score={"total": score},
            artifact={"likelihood": artifact_likelihood},
        )
        return candidate

    def test_higher_score_ranks_first(self):
        ordered = candidates.rank([self._candidate(1, 0.4, 0.0),
                                   self._candidate(2, 0.9, 0.0)])
        assert ordered[0].object_id == "o2"
        assert ordered[0].rank == 1

    def test_likely_artifacts_are_demoted(self):
        ordered = candidates.rank([self._candidate(1, 0.95, 0.9),
                                   self._candidate(2, 0.6, 0.0)])
        assert ordered[0].object_id == "o2"

    def test_artifacts_are_demoted_not_removed(self):
        """Plan section 4: an artifact conclusion is a real result."""
        ordered = candidates.rank([self._candidate(1, 0.95, 0.99),
                                   self._candidate(2, 0.6, 0.0)])
        assert len(ordered) == 2
        assert ordered[-1].object_id == "o1"

    def test_demotion_can_be_disabled(self):
        ordered = candidates.rank([self._candidate(1, 0.95, 0.9),
                                   self._candidate(2, 0.6, 0.0)],
                                  demote_artifacts=False)
        assert ordered[0].object_id == "o1"

    def test_ranks_are_sequential(self):
        ordered = candidates.rank([self._candidate(i, i / 10, 0.0)
                                   for i in range(1, 6)])
        assert [c.rank for c in ordered] == [1, 2, 3, 4, 5]


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        built = [candidates.build_candidate(1, IDENTITY, features(),
                                            anomaly_score=0.8)]
        candidates.save(built, "run1", tmp_path)
        loaded = candidates.load("run1", tmp_path)

        assert len(loaded) == 1
        assert loaded[0].candidate_id == "ASTRA-000001"
        assert loaded[0].explanation["what_happened"]

    def test_candidate_id_format(self):
        assert candidates.make_candidate_id(4921) == "ASTRA-004921"

    def test_saved_candidates_are_json_safe(self, tmp_path):
        import json

        built = [candidates.build_candidate(
            1, IDENTITY, features(reduced_chi2=float("nan")))]
        path = candidates.save(built, "run2", tmp_path)
        json.loads(path.read_text())  # must not raise


class TestLabelling:
    def test_recording_and_reading_a_label(self, tmp_path):
        candidates.record_label("ASTRA-000001", "interesting", "clear RRab",
                                root=tmp_path)
        labels = candidates.load_labels(tmp_path)

        assert labels["ASTRA-000001"]["label"] == "interesting"
        assert labels["ASTRA-000001"]["note"] == "clear RRab"

    def test_unknown_label_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown label"):
            candidates.record_label("ASTRA-000001", "beautiful", root=tmp_path)

    def test_every_plan_section_22_label_is_accepted(self, tmp_path):
        for label in ("interesting", "artifact", "known_object",
                      "uncertain", "needs_follow_up"):
            candidates.record_label(f"ASTRA-{label}", label, root=tmp_path)
        assert candidates.label_summary(tmp_path)["total"] == 5

    def test_relabelling_overwrites(self, tmp_path):
        candidates.record_label("ASTRA-000001", "interesting", root=tmp_path)
        candidates.record_label("ASTRA-000001", "artifact", root=tmp_path)

        summary = candidates.label_summary(tmp_path)

        assert summary["total"] == 1
        assert summary["by_label"]["artifact"] == 1
        assert summary["by_label"]["interesting"] == 0

    def test_labels_survive_regenerating_candidates(self, tmp_path):
        """Human review is the expensive input; a re-run must not destroy it."""
        candidates.record_label("ASTRA-000001", "interesting", root=tmp_path)
        candidates.save([candidates.build_candidate(1, IDENTITY, features())],
                        "rerun", tmp_path)

        assert candidates.load_labels(tmp_path)["ASTRA-000001"]["label"] == \
            "interesting"

    def test_missing_label_file_reads_as_empty(self, tmp_path):
        assert candidates.load_labels(tmp_path) == {}
        assert candidates.label_summary(tmp_path)["total"] == 0


class TestFollowupTracking:
    def test_requesting_followup_creates_a_history_entry(self, tmp_path):
        entry = candidates.request_followup("ASTRA-000001", "Palomar 200-inch",
                                            "worth a spectrum", root=tmp_path)

        assert entry["status"] == "requested"
        history = candidates.followup_history("ASTRA-000001", tmp_path)
        assert len(history) == 1
        assert history[0]["request_id"] == entry["request_id"]
        assert history[0]["facility_name"] == "Palomar 200-inch"

    def test_a_candidate_can_accumulate_multiple_requests(self, tmp_path):
        """Unlike a label (one row per candidate), follow-up requests are
        append-only -- the same candidate can be requested more than once
        over its lifetime."""
        candidates.request_followup("ASTRA-000001", "Facility A", root=tmp_path)
        candidates.request_followup("ASTRA-000001", "Facility B", root=tmp_path)

        history = candidates.followup_history("ASTRA-000001", tmp_path)
        assert len(history) == 2
        assert {row["facility_name"] for row in history} == {"Facility A", "Facility B"}

    def test_recording_a_result_updates_the_matching_request(self, tmp_path):
        entry = candidates.request_followup("ASTRA-000001", root=tmp_path)
        candidates.record_followup_result(entry["request_id"], "observed",
                                          "clean detection", root=tmp_path)

        history = candidates.followup_history("ASTRA-000001", tmp_path)
        assert history[0]["status"] == "observed"
        assert history[0]["result_note"] == "clean detection"
        assert history[0]["result_utc"] is not None

    def test_unknown_result_status_is_rejected(self, tmp_path):
        entry = candidates.request_followup("ASTRA-000001", root=tmp_path)
        with pytest.raises(ValueError, match="unknown follow-up result"):
            candidates.record_followup_result(entry["request_id"], "beautiful",
                                               root=tmp_path)

    def test_result_for_unknown_request_raises(self, tmp_path):
        with pytest.raises(KeyError):
            candidates.record_followup_result("followup_nope", "observed", root=tmp_path)

    def test_history_for_unrequested_candidate_is_empty(self, tmp_path):
        assert candidates.followup_history("ASTRA-999999", tmp_path) == []


class TestLabelVoting:
    def test_casting_and_reading_a_vote(self, tmp_path):
        entry = candidates.cast_label_vote("ASTRA-000001", "alice", "interesting",
                                           "looks real", root=tmp_path)

        assert entry["reviewer_id"] == "alice"
        votes = candidates.label_votes("ASTRA-000001", tmp_path)
        assert len(votes) == 1
        assert votes[0]["vote_id"] == entry["vote_id"]

    def test_multiple_reviewers_accumulate_rather_than_overwrite(self, tmp_path):
        """Unlike a label (one row per candidate, last write wins), votes
        are append-only -- each reviewer's vote survives another's."""
        candidates.cast_label_vote("ASTRA-000001", "alice", "interesting", root=tmp_path)
        candidates.cast_label_vote("ASTRA-000001", "bob", "artifact", root=tmp_path)

        votes = candidates.label_votes("ASTRA-000001", tmp_path)
        assert len(votes) == 2
        assert {v["reviewer_id"] for v in votes} == {"alice", "bob"}

    def test_unknown_label_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown label"):
            candidates.cast_label_vote("ASTRA-000001", "alice", "beautiful", root=tmp_path)

    def test_empty_reviewer_id_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="reviewer_id"):
            candidates.cast_label_vote("ASTRA-000001", "  ", "interesting", root=tmp_path)

    def test_tally_reports_majority_and_agreement(self, tmp_path):
        candidates.cast_label_vote("ASTRA-000001", "alice", "interesting", root=tmp_path)
        candidates.cast_label_vote("ASTRA-000001", "bob", "interesting", root=tmp_path)
        candidates.cast_label_vote("ASTRA-000001", "carol", "artifact", root=tmp_path)

        tally = candidates.label_vote_tally("ASTRA-000001", tmp_path)

        assert tally["total"] == 3
        assert tally["by_label"]["interesting"] == 2
        assert tally["majority_label"] == "interesting"
        assert tally["agreement_fraction"] == pytest.approx(2 / 3)

    def test_tally_with_no_votes_reports_none_not_zero(self, tmp_path):
        tally = candidates.label_vote_tally("ASTRA-999999", tmp_path)

        assert tally["total"] == 0
        assert tally["majority_label"] is None
        assert tally["agreement_fraction"] is None

    def test_promotion_below_vote_count_reports_gate_reason(self, tmp_path):
        candidates.cast_label_vote("ASTRA-000001", "alice", "interesting", root=tmp_path)

        result = candidates.promote_vote_consensus("ASTRA-000001", root=tmp_path)

        assert result["promoted"] is False
        assert result["reason"] == "insufficient votes or agreement"
        assert result["votes"] == 1
        assert candidates.load_labels(tmp_path) == {}

    def test_promotion_below_agreement_reports_gate_reason(self, tmp_path):
        for reviewer, label in [("a", "interesting"), ("b", "artifact"), ("c", "uncertain")]:
            candidates.cast_label_vote("ASTRA-000001", reviewer, label, root=tmp_path)

        result = candidates.promote_vote_consensus("ASTRA-000001", root=tmp_path)

        assert result["promoted"] is False
        assert result["agreement_fraction"] == pytest.approx(1 / 3)

    def test_promotion_at_threshold_updates_the_authoritative_label(self, tmp_path):
        for reviewer in ("a", "b", "c"):
            candidates.cast_label_vote("ASTRA-000001", reviewer, "interesting", root=tmp_path)

        result = candidates.promote_vote_consensus("ASTRA-000001", root=tmp_path)

        assert result["promoted"] is True
        assert result["label"] == "interesting"
        assert candidates.load_labels(tmp_path)["ASTRA-000001"]["label"] == "interesting"

    def test_a_second_promotion_call_is_safe(self, tmp_path):
        for reviewer in ("a", "b", "c"):
            candidates.cast_label_vote("ASTRA-000001", reviewer, "interesting", root=tmp_path)
        candidates.promote_vote_consensus("ASTRA-000001", root=tmp_path)

        result = candidates.promote_vote_consensus("ASTRA-000001", root=tmp_path)

        assert result["promoted"] is True
        assert candidates.load_labels(tmp_path)["ASTRA-000001"]["label"] == "interesting"


class TestTimeline:
    def test_timeline_is_bounded_and_marks_blended_surveys(self, tmp_path, monkeypatch):
        candidate = candidates.build_candidate(
            1,
            {"survey": "ztf", "release": "dr", "object_id": "Z1", "band": "g",
             "ra_deg": 10.0, "dec_deg": 20.0},
            {"n_points": 20},
            blended=["TESS"],
        )
        candidates.save([candidate], "timeline", tmp_path)
        ztf = LightCurve(
            SourceRef("ztf", "Z1", 10.0, 20.0), "dr", "g", "mag",
            np.arange(20, dtype=float), np.arange(20, dtype=float), np.ones(20),
            "BJD_TDB",
        )
        tess = LightCurve(
            SourceRef("TESS", "T1", 10.0005, 20.0), "qlp", "TESS", "flux",
            np.arange(30, dtype=float), np.arange(30, dtype=float), np.ones(30),
            "BJD_TDB",
        )
        monkeypatch.setattr("astra.candidates.evidence.load_curves_by_key",
                            lambda: {("ztf", "Z1"): [ztf], ("TESS", "T1"): [tess]})

        result = candidates.timeline(candidate.candidate_id, "timeline", tmp_path,
                                     radius_arcsec=30, max_curves=2, max_points=5)

        assert len(result["curves"]) == 2
        assert all(len(item["times"]) <= 5 for item in result["curves"])
        assert next(item for item in result["curves"] if item["survey"] == "TESS")["resolved"] is False
        assert result["warning"]
