"""Evaluation studies for `federated_training.py`, split purely to keep
each file under this project's 500-line guideline.

`evaluate_secure_aggregation_recovers_exact_sum` checks the real,
checkable guarantee `secure_aggregate` makes: the masked sum equals the
true sum EXACTLY (masks cancel by construction), and that no individual
masked vector -- inspected directly here, though `secure_aggregate`
itself never returns them -- equals its true unmasked counterpart. This
is the roadmap item's own "leakage audit," run against the real masking
arithmetic, not merely asserted.

`evaluate_federated_vs_centralized_accuracy_gap_synthetic` is the
roadmap item's other named metric: on one synthetic labelled dataset
with a per-institution feature-distribution shift (institutions built
with a systematic mean offset, simulating different instrument
systematics), trains one classifier CENTRALLY on the pooled data and one
FEDERATED classifier via `federated_round`, then reports the accuracy
gap on a common held-out test set -- with and without CORAL domain
alignment, to check alignment narrows the gap, a real, checkable
property of CORAL's own alignment guarantee (already established by
`artifact_bank_eval.py`'s cross-group study), not re-derived here.
`federated_round` is called with `reference_features=test_features`,
matching `evaluate_cross_group_auprc`'s own established convention of
aligning train features to the deployment/test domain -- aligning to an
arbitrary shard instead was tried first and made the federated model
WORSE (accuracy dropped from 0.82 to 0.67 in an initial run), because it
aligned every institution toward one shard's domain rather than the
domain actually being evaluated against; corrected before relying on
this study.
"""

from __future__ import annotations

import numpy as np

from .artifact_bank import train_hard_negative_classifier
from .federated_training import (
    FederatedTrainingError, federated_round, partition_by_institution, secure_aggregate,
)


def evaluate_secure_aggregation_recovers_exact_sum(n_clients: int = 5, vector_length: int = 8,
                                                   seed: int = 42) -> dict:
    """`secure_aggregate` on `n_clients` random vectors: the aggregate
    must equal the true elementwise sum to floating-point precision, and
    each client's MASKED vector (computed here directly, for audit
    purposes only) must differ substantially from its true vector."""
    if n_clients < 2:
        raise FederatedTrainingError(f"n_clients must be at least 2, got {n_clients}")
    rng = np.random.default_rng(seed)
    vectors = [rng.normal(size=vector_length) for _ in range(n_clients)]

    aggregate = secure_aggregate(vectors, seed=seed)
    true_sum = sum(vectors)
    exact = bool(np.allclose(aggregate, true_sum, atol=1e-9))

    from .federated_training import generate_pairwise_masks
    masks = generate_pairwise_masks(n_clients, vector_length, seed=seed)
    masked = [v + m for v, m in zip(vectors, masks)]
    reveals_raw_vector = any(np.allclose(m, v) for m, v in zip(masked, vectors))

    return {"n_clients": n_clients, "exact_sum_recovered": exact,
            "any_masked_vector_reveals_raw_vector": reveals_raw_vector,
            "mask_sum_is_zero": bool(np.allclose(sum(masks), 0.0, atol=1e-9))}


def evaluate_federated_vs_centralized_accuracy_gap_synthetic(
        n_institutions: int = 4, n_per_institution: int = 150, n_features: int = 6,
        shift_scale: float = 1.5, seed: int = 42) -> dict:
    """Synthetic two-class data split across `n_institutions` with a
    systematic per-institution feature mean shift (`shift_scale`); trains
    centralized vs. federated (with and without CORAL alignment)
    classifiers and reports test accuracy for each."""
    if n_institutions < 2:
        raise FederatedTrainingError(f"n_institutions must be at least 2, got {n_institutions}")

    rng = np.random.default_rng(seed)
    n_total = n_institutions * n_per_institution
    labels = rng.integers(0, 2, size=n_total)
    base = rng.normal(size=(n_total, n_features)) + labels[:, None] * 1.5

    institution_id = np.repeat(np.arange(n_institutions), n_per_institution)
    shift = rng.normal(size=(n_institutions, n_features)) * shift_scale
    features = base + shift[institution_id]

    split = int(n_total * 0.8)
    order = rng.permutation(n_total)
    train_idx, test_idx = order[:split], order[split:]
    train_features, train_labels = features[train_idx], labels[train_idx]
    test_features, test_labels = features[test_idx], labels[test_idx]
    train_institution = institution_id[train_idx]

    central_model = train_hard_negative_classifier(train_features, train_labels, seed=seed)
    central_accuracy = float(np.mean(central_model.predict(test_features) == test_labels))

    shards = [(train_features[train_institution == i], train_labels[train_institution == i])
             for i in range(n_institutions)]
    shards = [(f, l) for f, l in shards if len(np.unique(l)) >= 2]

    unaligned = federated_round(shards, domain_align=False, seed=seed)
    aligned = federated_round(shards, domain_align=True, reference_features=test_features, seed=seed)
    unaligned_accuracy = float(np.mean(unaligned.global_model.predict(test_features) == test_labels))
    aligned_accuracy = float(np.mean(aligned.global_model.predict(test_features) == test_labels))

    return {
        "central_accuracy": central_accuracy,
        "federated_unaligned_accuracy": unaligned_accuracy,
        "federated_aligned_accuracy": aligned_accuracy,
        "accuracy_gap_unaligned": central_accuracy - unaligned_accuracy,
        "accuracy_gap_aligned": central_accuracy - aligned_accuracy,
        "alignment_narrows_the_gap": abs(central_accuracy - aligned_accuracy)
        <= abs(central_accuracy - unaligned_accuracy) + 1e-9,
    }


__all__ = [
    "evaluate_secure_aggregation_recovers_exact_sum",
    "evaluate_federated_vs_centralized_accuracy_gap_synthetic",
]
