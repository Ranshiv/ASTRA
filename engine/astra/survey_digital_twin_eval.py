"""Digital-twin success criteria (backlog item 42): distance and transfer.

Two questions, both reusing EXISTING, unmodified machinery rather than
inventing parallel copies:

1. How far is the simulated population from the real one, in summary
   statistics? `evaluate.sequence_summary` (the same compact per-sequence
   statistics `evaluate.compare_on_sequences` already feeds to the baseline
   detectors) is computed for both populations, and a per-feature
   Kolmogorov-Smirnov distance reports how separable the two distributions
   are, feature by feature.

2. Does training on the digital twin transfer to real data? The same
   closed-world/open-world comparison shape `open_world_eval.
   evaluate_open_world_generalization` already established: the identical
   autoencoder architecture (via the unmodified `train.train()`) is trained
   once on real data and once on synthetic data, and BOTH are scored on the
   same held-out real injected set. Multi-seed mean/CI95, and this never
   asserts which arm wins -- the same restraint every other comparison in
   this codebase already applies.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEEDS: tuple[int, ...] = (17, 29, 43)

# Column order matches `evaluate.sequence_summary`'s own column construction
# exactly -- named here only for a readable report, not re-derived.
_SUMMARY_FEATURE_NAMES: tuple[str, ...] = (
    "std", "amplitude", "robust_amplitude", "mean_abs_diff", "max_abs_diff",
    "skew_proxy", "kurtosis_proxy", "min", "max", "coverage",
)


def summary_statistic_distance(real_values: np.ndarray,
                               synthetic_values: np.ndarray) -> dict:
    """Per-feature KS distance between real and synthetic sequence summaries.

    A distance of 0 means the two distributions are indistinguishable on
    that statistic; 1 means they share no support at all. This is reported
    per-feature AND as a mean, because a digital twin can match a survey's
    noise floor while still missing its coverage pattern (or vice versa),
    and collapsing that to one number would hide exactly the diagnostic this
    function exists to provide.
    """
    from scipy.stats import ks_2samp

    from . import evaluate

    if len(real_values) < 2 or len(synthetic_values) < 2:
        return {
            "per_feature": {}, "mean_ks_statistic": float("nan"),
            "real_rows": len(real_values), "synthetic_rows": len(synthetic_values),
            "note": "fewer than 2 rows in one population; no distance computed",
        }

    real_summary = evaluate.sequence_summary(real_values)
    synthetic_summary = evaluate.sequence_summary(synthetic_values)

    per_feature: dict[str, float | None] = {}
    for index, name in enumerate(_SUMMARY_FEATURE_NAMES):
        real_column = real_summary[:, index]
        synthetic_column = synthetic_summary[:, index]
        real_column = real_column[np.isfinite(real_column)]
        synthetic_column = synthetic_column[np.isfinite(synthetic_column)]
        if len(real_column) < 2 or len(synthetic_column) < 2:
            per_feature[name] = None
            continue
        statistic, _p_value = ks_2samp(real_column, synthetic_column)
        per_feature[name] = round(float(statistic), 4)

    finite = [value for value in per_feature.values() if value is not None]
    return {
        "per_feature": per_feature,
        "mean_ks_statistic": round(float(np.mean(finite)), 4) if finite else float("nan"),
        "real_rows": len(real_values), "synthetic_rows": len(synthetic_values),
    }


def _ci95_summary(values: list[float]) -> dict | None:
    """Same mean/std/ci95/n shape `open_world_eval._summary` already uses,
    duplicated rather than imported: a small, self-contained helper each
    module's own comparison owns, matching that module's own stated reason
    for keeping its copy separate from `sweep.py`'s."""
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(finite):
        return None
    return {
        "mean": round(float(np.mean(finite)), 4),
        "std": round(float(np.std(finite, ddof=1)), 4) if len(finite) > 1 else 0.0,
        "ci95": [round(float(np.quantile(finite, 0.025)), 4),
                round(float(np.quantile(finite, 0.975)), 4)],
        "n": len(finite),
    }


def evaluate_transfer_performance(real_batch, synthetic_batch, *,
                                  fraction: float = 0.1,
                                  seeds: tuple[int, ...] = DEFAULT_SEEDS,
                                  epochs: int = 15, model_config=None) -> dict:
    """Train on real vs. train on synthetic; evaluate both on the same real
    held-out injected set.

    `real_batch`/`synthetic_batch` are `tensors.SequenceBatch`, e.g. from
    `tensors.build(survey=...)` and `survey_digital_twin.sample_synthetic_batch`
    respectively -- same (n, 2, length) shape, so both feed the identical
    downstream path unchanged.
    """
    from . import evaluate, tensors
    from . import train as train_mod

    if len(seeds) < 2:
        raise ValueError("evaluate_transfer_performance needs at least two seeds")
    if len(real_batch) < 10:
        raise ValueError("real_batch needs at least 10 rows to hold out a test split")
    if len(synthetic_batch) < 10:
        raise ValueError("synthetic_batch needs at least 10 rows to train on")

    length = real_batch.shape[-1]
    # One fixed real train/test split, reused across every seed: the held-
    # out test set must be the same set of real rows for every arm and every
    # seed, or a "transfer" difference could just be an easier test split.
    # `train_test_split` returns raw arrays plus indices (not SequenceBatch
    # objects), so identities are recovered by indexing the same way.
    real_train_values, real_test_values, train_idx, test_idx = tensors.train_test_split(
        real_batch, test_fraction=0.2, seed=seeds[0])
    real_train_identities = [real_batch.identities[i] for i in train_idx]
    real_test_identities = [real_batch.identities[i] for i in test_idx]

    trained_on_real: list[float] = []
    trained_on_synthetic: list[float] = []

    for seed in seeds:
        test_injection = evaluate.build_injected(
            real_test_values, real_test_identities, fraction=fraction, seed=seed)
        real_injection = evaluate.build_injected(
            real_train_values, real_train_identities, fraction=fraction, seed=seed)
        synthetic_injection = evaluate.build_injected(
            synthetic_batch.values, synthetic_batch.identities,
            fraction=fraction, seed=seed)

        for injection, bucket in ((real_injection, trained_on_real),
                                  (synthetic_injection, trained_on_synthetic)):
            train_values, val_values, _, _ = tensors.train_test_split(
                tensors.SequenceBatch(values=injection.values,
                                      identities=injection.identities, length=length),
                test_fraction=0.2, seed=seed,
            )
            cfg = train_mod.TrainConfig(
                kind="autoencoder", epochs=epochs, seed=seed,
                model=model_config or train_mod.ModelConfig(length=length))
            try:
                report = train_mod.train(train_values, val_values, cfg,
                                         name="digital_twin_transfer")
                model, _ = train_mod.load_model(report.checkpoint)
                scores = train_mod.reconstruction_scores(model, test_injection.values)
                result = evaluate.score_method(
                    "digital_twin_transfer", scores, test_injection.labels)
                bucket.append(result.roc_auc)
            except Exception:  # noqa: BLE001 - one bad seed must not abort the study
                bucket.append(float("nan"))

    return {
        "trained_on_real": _ci95_summary(trained_on_real),
        "trained_on_synthetic": _ci95_summary(trained_on_synthetic),
        "held_out_test_injection": test_injection.to_dict(),
    }
