"""Evaluation-study correctness for `information_gain_eval.py`. No
`research` extra needed (no new optional dependency).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astra import information_gain_eval as ige
from astra.information_gain import InformationGainError


def test_evaluate_entropy_is_maximized_at_p_half():
    result = ige.evaluate_entropy_is_maximized_at_p_half()
    assert result["is_near_one_half"] is True
    assert result["max_entropy_bits"] == pytest.approx(1.0)


def test_evaluate_entropy_is_maximized_at_p_half_rejects_tiny_grid():
    with pytest.raises(InformationGainError):
        ige.evaluate_entropy_is_maximized_at_p_half(grid_size=2)


def test_evaluate_ranking_prioritizes_uncertain_visible_targets_synthetic():
    result = ige.evaluate_ranking_prioritizes_uncertain_visible_targets_synthetic()
    assert result["no_infeasible_ranked_before_feasible"] is True
    assert result["feasible_entropies_sorted_descending"] is True
    assert result["n_observable"] == result["n_candidates"] // 2


def test_evaluate_real_followup_infeasibility_integration():
    result = ige.evaluate_real_followup_infeasibility_integration()
    assert result["direct_followup_plan_visible"] is False
    assert result["consistent"] is True


def test_information_gain_eval_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "information_gain" not in rpc_source
