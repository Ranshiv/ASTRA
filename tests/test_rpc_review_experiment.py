"""review.experiment.vote / review.experiment.preregister RPC handlers
(Direction 6, "the review UI as a controlled experiment")."""

from __future__ import annotations

import pytest

from astra import rpc


@pytest.fixture
def research_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_RESEARCH_ROOT", str(tmp_path))
    return tmp_path


def test_experiment_vote_records_an_arm(isolated_root):
    response = rpc.dispatch({"id": 1, "method": "review.experiment.vote", "params": {
        "candidate_id": "ASTRA-000001", "reviewer_id": "alice", "label": "interesting",
        "score_lookup": {"ASTRA-000001": 0.8, "ASTRA-000002": 0.2},
    }})
    assert response["ok"] is True
    assert response["result"]["arm"] in ("score_shown", "score_blinded", "score_shuffled")


def test_experiment_vote_passes_through_optional_fields(isolated_root):
    response = rpc.dispatch({"id": 2, "method": "review.experiment.vote", "params": {
        "candidate_id": "ASTRA-000001", "reviewer_id": "bob", "label": "artifact",
        "score_lookup": {"ASTRA-000001": 0.5},
        "decision_latency_ms": 3000, "self_reported_confidence": 0.6,
        "presentation_index": 2,
    }})
    assert response["ok"] is True
    assert response["result"]["decision_latency_ms"] == 3000


def test_experiment_vote_rejects_an_unknown_label(isolated_root):
    response = rpc.dispatch({"id": 3, "method": "review.experiment.vote", "params": {
        "candidate_id": "ASTRA-000001", "reviewer_id": "alice", "label": "not_a_real_label",
        "score_lookup": {},
    }})
    assert response["ok"] is False


def test_experiment_preregister_returns_a_content_hash(research_root):
    response = rpc.dispatch({"id": 4, "method": "review.experiment.preregister", "params": {}})
    assert response["ok"] is True
    assert response["result"]["content_hash"]


def test_experiment_preregister_is_idempotent(research_root):
    first = rpc.dispatch({"id": 5, "method": "review.experiment.preregister", "params": {}})
    second = rpc.dispatch({"id": 6, "method": "review.experiment.preregister", "params": {}})
    assert first["result"] == second["result"]
