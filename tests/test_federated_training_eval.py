"""Evaluation-study correctness for `federated_training_eval.py`. No
`research` extra needed (no new optional dependency).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astra import federated_training_eval as fte
from astra.federated_training import FederatedTrainingError


def test_evaluate_secure_aggregation_recovers_exact_sum():
    result = fte.evaluate_secure_aggregation_recovers_exact_sum()
    assert result["exact_sum_recovered"] is True
    assert result["any_masked_vector_reveals_raw_vector"] is False
    assert result["mask_sum_is_zero"] is True


def test_evaluate_secure_aggregation_rejects_too_few_clients():
    with pytest.raises(FederatedTrainingError):
        fte.evaluate_secure_aggregation_recovers_exact_sum(n_clients=1)


def test_evaluate_federated_vs_centralized_accuracy_gap_synthetic():
    result = fte.evaluate_federated_vs_centralized_accuracy_gap_synthetic()
    assert 0.0 <= result["central_accuracy"] <= 1.0
    assert 0.0 <= result["federated_aligned_accuracy"] <= 1.0
    assert result["alignment_narrows_the_gap"] is True


def test_evaluate_federated_vs_centralized_rejects_too_few_institutions():
    with pytest.raises(FederatedTrainingError):
        fte.evaluate_federated_vs_centralized_accuracy_gap_synthetic(n_institutions=1)


def test_federated_training_eval_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "federated_training" not in rpc_source
