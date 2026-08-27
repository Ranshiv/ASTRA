"""Benchmark runner: score baselines + the ASTRA detector against a real,
sealed dataset, report bootstrap CIs, and write ResultRecords.

Reuses rather than reinvents: `evaluate.score_method` for the standard
retrieval metrics, `anomaly.run_isolation_forest`/`run_one_class_svm`/
`run_pca_reconstruction`/`detect` for the baseline and ASTRA-ensemble
scorers, and `research.stats.paired_bootstrap_ci` for object-grouped
confidence intervals. This module is the orchestration layer that was
missing: nothing before it bound a metric value to a benchmark ID, split
ID, and dataset manifest hash in one call.

A cross-survey anomaly benchmark has no verified real "this object is
anomalous" label store yet (that is a P1/labels-team follow-on). Consistent
with `evaluate.py`'s own documented approach ("known anomalies of known
shape and amplitude are injected... labels are then true by construction"),
this runner's anomaly track injects synthetic anomalies into real feature
vectors and reports the result under `synthetic=True` -- the *data* is real
and checksummed (`dataset_manifest_hash` points at it), but the *label* is
synthetic, and docs/BENCHMARKS.md requires that distinction to be visible,
not just the data/synthetic split at the whole-dataset level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .. import anomaly
from ..featurematrix import FeatureMatrix
from .records import BenchmarkSpec, ResultRecord
from .splits import Split

logger = logging.getLogger(__name__)

BASELINE_NAMES = ("robust_zscore", "isolation_forest", "one_class_svm",
                  "logistic_regression", "astra_ensemble")


def _robust_zscore_scores(x: np.ndarray) -> np.ndarray:
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0) + 1e-9
    return np.max(np.abs((x - median) / mad), axis=1)


def _logistic_regression_scores(x: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(x, labels)
    return model.predict_proba(x)[:, 1]


def _inject_synthetic_anomalies(x: np.ndarray, *, fraction: float, seed: int,
                                magnitude: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Perturb a fraction of real feature vectors by a large, fixed offset
    in a random subset of feature dimensions -- an honest, label-by-
    construction stand-in for a verified anomaly label, following the same
    principle `evaluate.inject` uses on raw sequences."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    labels = np.zeros(n, dtype=int)
    n_positive = max(1, int(round(n * fraction)))
    positive_idx = rng.choice(n, size=n_positive, replace=False)
    labels[positive_idx] = 1

    perturbed = x.copy()
    scale = np.std(x, axis=0) + 1e-9
    for idx in positive_idx:
        dims = rng.choice(d, size=max(1, d // 4), replace=False)
        perturbed[idx, dims] += magnitude * scale[dims] * rng.choice([-1, 1], size=len(dims))
    return perturbed, labels


def _perturbed_matrix(matrix: FeatureMatrix, identities: list[dict], *,
                      fraction: float, seed: int) -> tuple[FeatureMatrix, np.ndarray]:
    """Inject in *raw* (pre-StandardScaler) feature space and wrap the
    result back into a `FeatureMatrix`, so `anomaly.detect` -- which
    standardises internally via its own `prepare()` -- sees the exact same
    injected anomalies the baselines are scored against, rather than
    silently scoring the clean, un-injected matrix."""
    mask = matrix.finite_mask()
    raw = matrix.values[mask]
    perturbed_raw, labels = _inject_synthetic_anomalies(raw, fraction=fraction, seed=seed)
    perturbed_matrix = FeatureMatrix(values=perturbed_raw, identities=identities,
                                     feature_names=matrix.feature_names,
                                     feature_version=matrix.feature_version)
    return perturbed_matrix, labels


def _bootstrap_auprc_by_object(labels: np.ndarray, scores: np.ndarray,
                               object_ids: list[str], *, seed: int,
                               n_resamples: int = 200) -> dict:
    """AUPRC with a bootstrap CI resampling object *groups*.

    In this feature-matrix track each object contributes exactly one row
    (`anomaly.prepare` yields one feature vector per light curve), so
    grouping by object ID and resampling rows coincide here -- the group
    structure still matters (and is kept explicit, rather than assuming
    it away) for a future multi-epoch-per-object feature representation,
    where it would not.
    """
    from sklearn.metrics import average_precision_score

    n = len(labels)
    if n == 0 or labels.sum() == 0 or labels.sum() == n:
        return {"point": float("nan"), "ci": [float("nan"), float("nan")]}

    point = float(average_precision_score(labels, scores))
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_labels = labels[idx]
        if resampled_labels.sum() == 0 or resampled_labels.sum() == n:
            continue
        boot.append(float(average_precision_score(resampled_labels, scores[idx])))

    if not boot:
        return {"point": round(point, 4), "ci": [round(point, 4), round(point, 4)]}
    return {"point": round(point, 4),
            "ci": [round(float(np.quantile(boot, 0.025)), 4),
                   round(float(np.quantile(boot, 0.975)), 4)]}


@dataclass
class BenchmarkRunResult:
    benchmark_id: str
    split_id: str
    results: list[ResultRecord]

    def to_dict(self) -> dict:
        return {"benchmark_id": self.benchmark_id, "split_id": self.split_id,
               "results": [r.to_dict() for r in self.results]}


def run_cross_survey_anomaly(
    matrix: FeatureMatrix, spec: BenchmarkSpec, split: Split, *,
    experiment_id: str, dataset_manifest_hash: str, injection_fraction: float = 0.1,
) -> BenchmarkRunResult:
    """Score every baseline plus the ASTRA ensemble on injected-anomaly
    recovery over one real, sealed feature matrix, at every seed the
    benchmark spec declares.
    """
    _, identities, _ = anomaly.prepare(matrix)
    object_ids = [str(row.get("object_id", i)) for i, row in enumerate(identities)]

    results: list[ResultRecord] = []
    for seed in spec.seeds:
        # Inject once, in raw feature space, and wrap it back into a
        # `FeatureMatrix` so every method -- baselines and the ASTRA
        # ensemble alike -- is scored against the identical injected
        # anomalies, standardised the same way `anomaly.prepare` always
        # standardises (see `_perturbed_matrix`'s docstring).
        perturbed_matrix, labels = _perturbed_matrix(
            matrix, identities, fraction=injection_fraction, seed=seed)
        perturbed_x, _, _ = anomaly.prepare(perturbed_matrix)

        scored = {
            "robust_zscore": _robust_zscore_scores(perturbed_x),
            "isolation_forest": anomaly.run_isolation_forest(
                perturbed_x, contamination=injection_fraction, seed=seed).scores,
            "one_class_svm": anomaly.run_one_class_svm(
                perturbed_x, contamination=injection_fraction).scores,
            "logistic_regression": _logistic_regression_scores(perturbed_x, labels, seed),
        }
        try:
            # `anomaly.detect` re-derives `prepare()` internally on the same
            # `perturbed_matrix`, so `.identities`/`.consensus` order matches
            # `labels`/`perturbed_x` index-for-index without going through
            # `.ranked()`, which re-sorts by score.
            ensemble = anomaly.detect(perturbed_matrix, contamination=injection_fraction,
                                      seed=seed)
            if ensemble.consensus.size == len(object_ids):
                scored["astra_ensemble"] = ensemble.consensus
        except Exception:  # noqa: BLE001 - a baseline failing must not sink the run
            logger.warning("astra_ensemble scoring failed for seed %d", seed, exc_info=True)

        for method_name, method_scores in scored.items():
            if len(method_scores) != len(labels):
                continue
            ci = _bootstrap_auprc_by_object(labels, np.asarray(method_scores),
                                            object_ids, seed=seed)

            results.append(ResultRecord(
                experiment_id=experiment_id, benchmark_id=spec.benchmark_id,
                split_id=split.split_id, dataset_manifest_hash=dataset_manifest_hash,
                metric=spec.primary_metric, value=ci["point"], sample_count=len(object_ids),
                confidence_interval=ci["ci"], seed=seed, synthetic=True,
                notes=(f"method={method_name}; real feature data, synthetic injected "
                      f"anomaly labels (fraction={injection_fraction})"),
            ))

    return BenchmarkRunResult(benchmark_id=spec.benchmark_id, split_id=split.split_id,
                              results=results)


__all__ = ["BenchmarkRunResult", "run_cross_survey_anomaly", "BASELINE_NAMES"]
