"""Label-history, reason-yield, reweighting, and reviewer-agreement-proxy
correctness for `active_review.py`. No `research` extra needed (no new
optional dependency; `cohen_kappa_score` comes from the already-required
`scikit-learn`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astra import active_review as ar
from astra import candidates, review


def _candidate(candidate_id, agreement=1.0, likelihood=0.05, tail=0.95):
    return {"candidate_id": candidate_id, "score": {"model_agreement": agreement},
            "artifact": {"likelihood": likelihood},
            "significance": {"tail_probability": tail}, "features": {}}


# ---------------------------------------------------------------------------
# label_history / labels_per_recovery
# ---------------------------------------------------------------------------

def test_label_history_is_chronologically_ordered(tmp_path):
    candidates.record_label("a", "artifact", root=tmp_path)
    candidates.record_label("b", "interesting", root=tmp_path)
    history = ar.label_history(tmp_path)
    assert [row["candidate_id"] for row in history] == ["a", "b"]
    assert history[0]["is_positive"] is False
    assert history[1]["is_positive"] is True


def test_labels_per_recovery_counts_gaps_between_positive_labels(tmp_path):
    for candidate_id in ("a", "b", "c"):
        candidates.record_label(candidate_id, "artifact", root=tmp_path)
    candidates.record_label("d", "interesting", root=tmp_path)
    candidates.record_label("e", "known_object", root=tmp_path)
    candidates.record_label("f", "needs_follow_up", root=tmp_path)

    result = ar.labels_per_recovery(tmp_path)
    assert result["recoveries"] == 2
    assert result["labels_per_recovery"] == [4, 2]
    assert result["mean_labels_per_recovery"] == pytest.approx(3.0)


def test_labels_per_recovery_is_empty_with_no_labels(tmp_path):
    result = ar.labels_per_recovery(tmp_path)
    assert result["recoveries"] == 0
    assert result["mean_labels_per_recovery"] is None


# ---------------------------------------------------------------------------
# reason_yield / learn_reason_weights
# ---------------------------------------------------------------------------

def test_reason_yield_matches_labels_to_select_next_reasons(tmp_path):
    items = [
        _candidate("disagree", agreement=0.3),      # "detectors disagree"
        _candidate("clean", agreement=1.0),         # "diverse candidate"
    ]
    candidates.record_label("disagree", "interesting", root=tmp_path)
    candidates.record_label("clean", "artifact", root=tmp_path)

    yields = ar.reason_yield(items, tmp_path)
    assert yields["by_reason"]["detectors disagree"]["positive"] == 1
    assert yields["by_reason"]["diverse candidate"]["negative"] == 1
    assert yields["by_reason"]["detectors disagree"]["rate"] == pytest.approx(1.0)


def test_reason_yield_rejects_empty_items(tmp_path):
    with pytest.raises(ar.ActiveReviewError):
        ar.reason_yield([], tmp_path)


def test_learn_reason_weights_defaults_to_uniform_with_no_labels(tmp_path):
    items = [_candidate("a", agreement=0.3), _candidate("b", agreement=1.0)]
    weights = ar.learn_reason_weights(items, tmp_path)
    assert all(value == pytest.approx(1.0) for value in weights.weights.values())


def test_learn_reason_weights_favours_the_higher_yield_reason(tmp_path):
    for i in range(10):
        candidates.record_label(f"d{i}", "interesting", root=tmp_path)
        candidates.record_label(f"c{i}", "artifact", root=tmp_path)
    items = ([_candidate(f"d{i}", agreement=0.3) for i in range(10)]
            + [_candidate(f"c{i}", agreement=1.0) for i in range(10)])
    weights = ar.learn_reason_weights(items, tmp_path)
    assert weights.weights["detectors disagree"] > weights.weights["diverse candidate"]


def test_learn_reason_weights_rejects_bad_smoothing(tmp_path):
    with pytest.raises(ar.ActiveReviewError):
        ar.learn_reason_weights([_candidate("a")], tmp_path, smoothing=0.0)


# ---------------------------------------------------------------------------
# reweighted_select_next
# ---------------------------------------------------------------------------

def test_reweighted_select_next_excludes_already_labelled_candidates(tmp_path):
    candidates.record_label("a", "interesting", root=tmp_path)
    items = [_candidate("a"), _candidate("b")]
    weights = ar.ReasonWeights(weights={tag: 1.0 for tag in ar.REASON_TAGS})
    selection = ar.reweighted_select_next(items, weights, tmp_path, limit=10)
    assert {row["candidate_id"] for row in selection} == {"b"}


def test_reweighted_select_next_biases_toward_the_upweighted_reason(tmp_path):
    items = [_candidate("disagree", agreement=0.3), _candidate("clean", agreement=1.0)]
    weights = ar.ReasonWeights(weights={"detectors disagree": 10.0, "diverse candidate": 0.01,
                                        "artifact assessment is uncertain": 0.01,
                                        "significance is near the review boundary": 0.01})
    selection = ar.reweighted_select_next(items, weights, tmp_path, limit=1)
    assert selection[0]["candidate_id"] == "disagree"


def test_reweighted_select_next_respects_limit_and_empty_items(tmp_path):
    weights = ar.ReasonWeights(weights={})
    assert ar.reweighted_select_next([], weights, tmp_path) == []
    assert ar.reweighted_select_next([_candidate("a")], weights, tmp_path, limit=0) == []


# ---------------------------------------------------------------------------
# reviewer_agreement_with_priority
# ---------------------------------------------------------------------------

def test_reviewer_agreement_with_priority_reports_not_ready_below_gate(tmp_path):
    result = ar.reviewer_agreement_with_priority(root=tmp_path)
    assert result["ready"] is False


def test_active_review_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "active_review" not in rpc_source
