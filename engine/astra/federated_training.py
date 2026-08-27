"""Federated multi-observatory training (roadmap item 39, P2).

Confirmed genuinely missing by grep for "distributed"/"multi.node"/
"federat"/"torch.distributed"/"DataParallel"/"horovod" across
`tensors.py`/`train.py`/`models.py` before this session: zero hits.
This codebase has no distributed/federated scaffolding of any kind to
build on or duplicate.

Scoped realistically to what is checkable OFFLINE, with no real
multi-institution network or auth layer to test against (confirmed
absent, and explicitly out of scope -- see below): data PARTITIONED
across synthetic "institutions" from one pooled dataset, a classifier
per partition, and a real (simplified) secure-aggregation-style average
of the resulting model parameters.

**Classifier**: `sklearn.linear_model.LogisticRegression`, matching this
codebase's established "simple features into LogisticRegression" house
style (`artifact_bank.train_hard_negative_classifier`, reused UNCHANGED
per institution here, not reimplemented) -- not a new torch training
loop, since the roadmap item's own success metric (accuracy gap vs.
centralized, plus a leakage audit) needs a model simple enough that
"average the parameters" is a well-defined, exact operation, which is
NOT true in general for a deep net's non-convex loss landscape. This is
a real, stated scope choice, not an oversight.

**Federated averaging** follows FedAvg (McMahan, Moore, Ramage, Hampson
& y Arcas 2017, "Communication-Efficient Learning of Deep Networks from
Decentralized Data," AISTATS): each institution fits its own local
model, and the server averages the fitted parameter vectors.
`_set_linear_params`/`_get_linear_params` read/write `coef_`/
`intercept_`/`classes_` directly -- `sklearn.linear_model.
LogisticRegression.predict` only reads those three fitted attributes
(confirmed by direct API inspection this session), so constructing an
unfitted estimator and assigning them is a valid, minimal way to hand
back an "averaged model" without inventing a bespoke model class.

**Secure aggregation**: `generate_pairwise_masks`/`secure_aggregate`
implement the CORE additive-masking idea of Bonawitz et al. (2017,
"Practical Secure Aggregation for Federated Learning," CCS) -- for every
client pair `(i, j)`, a shared random mask vector is added to client
`i`'s update and subtracted from client `j`'s, so the masks cancel
EXACTLY once every client's masked update is summed, and the sum can be
computed without any single client's true update being exposed in the
clear. This module does NOT implement Bonawitz et al.'s cryptographic
key-agreement or dropout-recovery layers (Shamir secret sharing,
authenticated key exchange) -- there is no real network transport
between institutions in this codebase to protect, so that machinery
would have nothing real to secure. Stated as a real scope limit, not
worked around.

**Domain adaptation** reuses `artifact_bank.coral_align` UNCHANGED
(CORAL; Sun & Saenko 2016) to align each institution's local features to
a reference domain before local training -- the same justification
`artifact_bank.py` already gives for CORAL, applied here to cross-
institution feature-distribution shift instead of cross-camera shift.
`artifact_bank_eval.evaluate_cross_group_auprc` establishes this
codebase's own CORAL convention: align TRAIN features to the intended
TEST/deployment domain, not an arbitrary other group, "so the SAME
classifier consumes already-aligned features, matching CORAL's intended
use." `federated_round`'s `reference_features` follows that same
convention.

Confirmed UNREACHABLE, stated up front: real cross-institution
deployment concerns -- authentication, network partitioning, differential-
privacy budgeting, and dropout/failure recovery during a real multi-
party protocol. No survey connector or RPC layer in this codebase
performs multi-institution communication of any kind; there is nothing
real for this module to integrate with on that side, so it is entirely
a single-process simulation of the AGGREGATION MATHEMATICS, not a
network-security or deployment demonstration.

Explicitly NOT done: does not modify `artifact_bank.py` -- `coral_align`
and `train_hard_negative_classifier` are called unchanged. Like every
other opt-in research module in this codebase, NOT wired into `rpc.py`,
`scoring.WEIGHTS`, or `evidence.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import artifact_bank


class FederatedTrainingError(ValueError):
    """A partition, aggregation, or federated-round input was invalid."""


def partition_by_institution(features: np.ndarray, labels: np.ndarray, n_institutions: int, *,
                             seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """A random, roughly-equal partition of pooled `(features, labels)`
    into `n_institutions` disjoint shards -- simulates data PARTITIONED
    across institutions, none of which sees another's rows."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if len(features) != len(labels):
        raise FederatedTrainingError("features and labels must have the same length")
    if n_institutions < 1:
        raise FederatedTrainingError(f"n_institutions must be at least 1, got {n_institutions}")
    if len(features) < n_institutions:
        raise FederatedTrainingError("fewer rows than institutions requested")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    shard_indices = np.array_split(order, n_institutions)
    return [(features[idx], labels[idx]) for idx in shard_indices]


def _get_linear_params(model) -> dict:
    return {"coef_": np.array(model.coef_), "intercept_": np.array(model.intercept_),
            "classes_": np.array(model.classes_)}


def _set_linear_params(params: dict):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.coef_ = np.array(params["coef_"])
    model.intercept_ = np.array(params["intercept_"])
    model.classes_ = np.array(params["classes_"])
    model.n_features_in_ = model.coef_.shape[1]
    return model


def generate_pairwise_masks(n_clients: int, vector_length: int, *, seed: int = 42) -> list[np.ndarray]:
    """One mask vector per client: for every pair `(i, j)` with `i < j`,
    a shared random vector is added to client `i`'s mask and subtracted
    from client `j`'s, so `sum(masks) == 0` EXACTLY (Bonawitz et al.
    2017's additive-masking core idea)."""
    if n_clients < 2:
        raise FederatedTrainingError(f"n_clients must be at least 2, got {n_clients}")
    if vector_length < 1:
        raise FederatedTrainingError(f"vector_length must be at least 1, got {vector_length}")
    rng = np.random.default_rng(seed)
    masks = [np.zeros(vector_length, dtype=np.float64) for _ in range(n_clients)]
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            pairwise = rng.normal(size=vector_length)
            masks[i] += pairwise
            masks[j] -= pairwise
    return masks


def secure_aggregate(vectors: list[np.ndarray], *, seed: int = 42) -> np.ndarray:
    """Sums `vectors` via additive masking: each vector is masked before
    "transmission" and the masks cancel exactly on summation, so the
    exact sum is recovered without any single unmasked vector being
    exposed to an aggregator. Returns ONLY the aggregate -- the masked
    per-client vectors are not returned, matching what a real aggregator
    would see."""
    if len(vectors) < 2:
        raise FederatedTrainingError("secure_aggregate needs at least 2 vectors")
    vectors = [np.asarray(v, dtype=np.float64) for v in vectors]
    length = len(vectors[0])
    if any(len(v) != length for v in vectors):
        raise FederatedTrainingError("all vectors must have the same length")

    masks = generate_pairwise_masks(len(vectors), length, seed=seed)
    masked = [v + m for v, m in zip(vectors, masks)]
    return sum(masked)


def federated_average(models: list, *, use_secure_aggregation: bool = True, seed: int = 42):
    """FedAvg (McMahan et al. 2017): averages fitted `coef_`/`intercept_`
    across `models`, returning one new estimator with the averaged
    parameters. Sums via `secure_aggregate` by default, matching real
    federated deployments where the server never sees an individual
    client's raw update."""
    if len(models) < 1:
        raise FederatedTrainingError("federated_average needs at least 1 model")
    params = [_get_linear_params(m) for m in models]
    classes = params[0]["classes_"]
    if any(not np.array_equal(p["classes_"], classes) for p in params):
        raise FederatedTrainingError("all models must share the same classes_")

    coef_shape = params[0]["coef_"].shape
    flat_coefs = [p["coef_"].ravel() for p in params]
    flat_intercepts = [p["intercept_"].ravel() for p in params]

    if use_secure_aggregation and len(models) >= 2:
        coef_sum = secure_aggregate(flat_coefs, seed=seed)
        intercept_sum = secure_aggregate(flat_intercepts, seed=seed + 1)
    else:
        coef_sum = sum(flat_coefs)
        intercept_sum = sum(flat_intercepts)

    n = len(models)
    averaged = {"coef_": (coef_sum / n).reshape(coef_shape),
               "intercept_": intercept_sum / n, "classes_": classes}
    return _set_linear_params(averaged)


@dataclass(frozen=True)
class FederatedRound:
    local_models: list
    global_model: object
    n_institutions: int
    domain_aligned: bool


def federated_round(shards: list[tuple[np.ndarray, np.ndarray]], *,
                    domain_align: bool = True, reference_features: np.ndarray | None = None,
                    seed: int = 42) -> FederatedRound:
    """One federated round: optionally CORAL-align every shard's features
    to `reference_features` (`artifact_bank.coral_align`, UNCHANGED)
    before fitting a local classifier per shard (`artifact_bank.
    train_hard_negative_classifier`, UNCHANGED), then `federated_
    average` the local models. `reference_features` defaults to the
    first shard's own features when not supplied. `artifact_bank_eval.
    evaluate_cross_group_auprc` establishes this codebase's own CORAL
    convention -- align TRAIN features to the DEPLOYMENT/test domain,
    not an arbitrary shard -- so a caller evaluating against a known
    test set should pass its features here, matching that convention;
    a real federated deployment would not know the test domain in
    advance, the same real limitation `evaluate_cross_group_auprc`
    already accepts."""
    if not shards:
        raise FederatedTrainingError("shards must be non-empty")

    target_features = reference_features if reference_features is not None else shards[0][0]
    local_models = []
    for features, labels in shards:
        if domain_align:
            features = artifact_bank.coral_align(features, target_features)
        local_models.append(artifact_bank.train_hard_negative_classifier(features, labels, seed=seed))

    global_model = federated_average(local_models, seed=seed)
    return FederatedRound(local_models=local_models, global_model=global_model,
                          n_institutions=len(shards), domain_aligned=domain_align)


__all__ = [
    "FederatedTrainingError", "partition_by_institution", "generate_pairwise_masks",
    "secure_aggregate", "federated_average", "FederatedRound", "federated_round",
]
