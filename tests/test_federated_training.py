"""Partitioning, secure-aggregation, and FedAvg correctness for
`federated_training.py`. No `research` extra needed (no new optional
dependency; `LogisticRegression` comes from the already-required
`scikit-learn`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import federated_training as ft


def _linearly_separable(n=100, n_features=4, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    features = rng.normal(size=(n, n_features)) + labels[:, None] * 2.0
    return features, labels


# ---------------------------------------------------------------------------
# partition_by_institution
# ---------------------------------------------------------------------------

def test_partition_by_institution_covers_every_row_exactly_once():
    features, labels = _linearly_separable(n=50)
    shards = ft.partition_by_institution(features, labels, 4, seed=1)
    assert len(shards) == 4
    total = sum(len(f) for f, _ in shards)
    assert total == 50


def test_partition_by_institution_rejects_more_institutions_than_rows():
    with pytest.raises(ft.FederatedTrainingError):
        ft.partition_by_institution(np.zeros((2, 3)), np.array([0, 1]), 5)


# ---------------------------------------------------------------------------
# secure_aggregate / generate_pairwise_masks
# ---------------------------------------------------------------------------

def test_generate_pairwise_masks_sum_to_zero():
    masks = ft.generate_pairwise_masks(5, 6, seed=3)
    assert np.allclose(sum(masks), 0.0, atol=1e-9)


def test_secure_aggregate_recovers_the_true_sum():
    vectors = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    aggregate = ft.secure_aggregate(vectors, seed=9)
    assert np.allclose(aggregate, np.array([9.0, 12.0]))


def test_secure_aggregate_rejects_mismatched_lengths():
    with pytest.raises(ft.FederatedTrainingError):
        ft.secure_aggregate([np.array([1.0]), np.array([1.0, 2.0])])


def test_secure_aggregate_rejects_too_few_vectors():
    with pytest.raises(ft.FederatedTrainingError):
        ft.secure_aggregate([np.array([1.0])])


# ---------------------------------------------------------------------------
# federated_average / federated_round
# ---------------------------------------------------------------------------

def test_federated_average_matches_plain_mean_of_identical_models():
    features, labels = _linearly_separable()
    from astra.artifact_bank import train_hard_negative_classifier
    model = train_hard_negative_classifier(features, labels, seed=1)
    averaged = ft.federated_average([model, model], use_secure_aggregation=True)
    assert np.allclose(averaged.coef_, model.coef_)
    assert np.allclose(averaged.intercept_, model.intercept_)


def test_federated_round_produces_a_usable_global_model():
    features, labels = _linearly_separable(n=200)
    shards = ft.partition_by_institution(features, labels, 4, seed=2)
    result = ft.federated_round(shards, domain_align=True, seed=5)
    assert result.n_institutions == 4
    predictions = result.global_model.predict(features)
    accuracy = float(np.mean(predictions == labels))
    assert accuracy > 0.7


def test_federated_round_rejects_empty_shards():
    with pytest.raises(ft.FederatedTrainingError):
        ft.federated_round([])


def test_federated_training_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "federated_training" not in rpc_source
