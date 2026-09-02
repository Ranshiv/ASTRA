"""Evaluation-study correctness for `active_review_eval.py`. No `research`
extra needed (no new optional dependency).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astra import active_review_eval as are
from astra.active_review import ActiveReviewError


def test_evaluate_reason_informativeness_rejects_empty_items(tmp_path):
    result = are.evaluate_reason_informativeness([], tmp_path)
    assert result["ready"] is False


def test_evaluate_labels_per_recovery_runs_on_an_empty_project(tmp_path):
    result = are.evaluate_labels_per_recovery(tmp_path)
    assert result["recoveries"] == 0


def test_evaluate_reweighted_selection_recovers_high_yield_reason_synthetic(tmp_path):
    result = are.evaluate_reweighted_selection_recovers_high_yield_reason_synthetic(
        tmp_path, n_per_reason=20, limit=10)
    assert result["enriched"] is True
    assert result["selected_high_yield_reason_fraction"] > result["base_rate"]


def test_evaluate_reweighted_selection_rejects_bad_n_per_reason(tmp_path):
    with pytest.raises(ActiveReviewError):
        are.evaluate_reweighted_selection_recovers_high_yield_reason_synthetic(tmp_path, n_per_reason=0)


def test_evaluate_synthetic_dual_reviewer_kappa_is_monotonic_in_agreement():
    result = are.evaluate_synthetic_dual_reviewer_kappa()
    assert result["monotonic_non_decreasing"] is True
    # Perfect injected agreement must yield kappa == 1.0 exactly.
    assert result["levels"][-1]["injected_agreement"] == 1.0
    assert result["levels"][-1]["cohen_kappa"] == pytest.approx(1.0)


def test_evaluate_synthetic_dual_reviewer_kappa_rejects_bad_inputs():
    with pytest.raises(ActiveReviewError):
        are.evaluate_synthetic_dual_reviewer_kappa(agreement_levels=())
    with pytest.raises(ActiveReviewError):
        are.evaluate_synthetic_dual_reviewer_kappa(n_labels=0)


def test_active_review_eval_is_not_wired_into_rpc():
    """active_review.py (the reweighting logic itself) was deliberately
    promoted into rpc.py's `review.next` (docs/DEFERRED.txt, roadmap item 36)
    -- see test_active_review.py's wired-in test. This evaluation-study
    module (`active_review_eval.py`, the synthetic integration/kappa-
    mechanism checks) was not, and stays research-only."""
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "active_review_eval" not in rpc_source
