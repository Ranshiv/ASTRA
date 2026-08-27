"""Time-to-classification study: macro-F1 vs. days-since-first-detection.

Split from `sn_classification.py` purely to keep each file under this
project's 500-line guideline (same `stellar_manifold.py`/
`stellar_manifold_eval.py` split rationale, not an independent module).

`evaluate_time_to_classification` takes already-fetched labelled light
curves as plain arrays -- it does no network acquisition itself. A caller
typically builds `labeled_curves` from `surveys/alerce.py`'s
`query_classified_objects()` + `fetch_light_curves()` (real ALeRCE broker
classifications taken as ground truth, the same precedent
`open_world_injection.py` already set) or from synthetic
`sn_classification.bazin_model` injections for mechanism validation.

The classifier itself follows `multimodal_eval.linear_probe_macro_f1`'s
exact pattern (`LogisticRegression` -> `f1_score(average="macro")`) --
that function is coupled to this codebase's torch multimodal embeddings
and not directly importable, so this module reimplements the same shape
over plain feature arrays instead.

"Time-to-classification" is DEFINED explicitly here (no prior art in this
codebase or a single universally agreed literature definition): the first
cutoff day at which mean macro-F1 across seeds reaches at least
`asymptotic_fraction` (default 80%) of the asymptotic (full-light-curve)
macro-F1 AND does not drop below that threshold at any later grid point --
`None` when the threshold is never reached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sn_classification import (
    FEATURE_NAMES, SNClassificationError, bazin_features, features_to_vector, truncate_light_curve,
)


@dataclass(frozen=True)
class LabeledCurve:
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    label: str


def _summary(values: list[float]) -> dict | None:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not len(finite):
        return None
    return {
        "mean": round(float(np.mean(finite)), 4),
        "std": round(float(np.std(finite, ddof=1)), 4) if len(finite) > 1 else 0.0,
        "ci95": [round(float(np.quantile(finite, 0.025)), 4),
                round(float(np.quantile(finite, 0.975)), 4)],
    }


def _macro_f1(features_train, labels_train, features_test, labels_test) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    clf = LogisticRegression(max_iter=1000)
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    return float(f1_score(labels_test, predictions, average="macro"))


@dataclass(frozen=True)
class TimeToClassificationResult:
    cutoff_days: list[float]
    macro_f1_by_cutoff: list[dict | None]
    asymptotic_macro_f1: float
    time_to_classification_days: float | None
    n_objects: int
    n_classes: int

    def to_dict(self) -> dict:
        return {
            "cutoff_days": self.cutoff_days, "macro_f1_by_cutoff": self.macro_f1_by_cutoff,
            "asymptotic_macro_f1": round(self.asymptotic_macro_f1, 4),
            "time_to_classification_days": self.time_to_classification_days,
            "n_objects": self.n_objects, "n_classes": self.n_classes,
        }


def evaluate_time_to_classification(labeled_curves: list[LabeledCurve], cutoff_grid_days: list[float], *,
                                    test_fraction: float = 0.3, n_seeds: int = 5, seed: int = 42,
                                    asymptotic_fraction: float = 0.8) -> TimeToClassificationResult:
    if not labeled_curves:
        raise SNClassificationError("labeled_curves must be non-empty")
    if not cutoff_grid_days:
        raise SNClassificationError("cutoff_grid_days must be non-empty")
    labels_all = np.array([c.label for c in labeled_curves])
    n_classes = len(np.unique(labels_all))
    if n_classes < 2:
        raise SNClassificationError("need at least two distinct classes to compute macro-F1")
    cutoff_grid_days = sorted(cutoff_grid_days)

    n = len(labeled_curves)
    # Truncation/feature extraction depends only on (object, cutoff), never
    # on the train/test seed -- computed once per cutoff and reused across
    # every seed trial below, rather than recomputed n_seeds times.
    features_by_cutoff: list[np.ndarray] = []
    for cutoff in cutoff_grid_days:
        rows = []
        for curve in labeled_curves:
            t, f, e = truncate_light_curve(curve.time, curve.flux, curve.flux_err, cutoff)
            if len(t) == 0:
                rows.append(np.zeros(len(FEATURE_NAMES)))
                continue
            rows.append(features_to_vector(bazin_features(t, f, e)))
        features_by_cutoff.append(np.vstack(rows))

    per_cutoff_scores: list[list[float]] = [[] for _ in cutoff_grid_days]
    for trial in range(n_seeds):
        rng = np.random.default_rng(seed + trial)
        order = rng.permutation(n)
        cut = max(1, int(round(n * (1.0 - test_fraction))))
        train_idx, test_idx = order[:cut], order[cut:]
        if len(np.unique(labels_all[train_idx])) < 2 or len(test_idx) == 0:
            continue

        for cutoff_pos, features in enumerate(features_by_cutoff):
            score = _macro_f1(features[train_idx], labels_all[train_idx],
                              features[test_idx], labels_all[test_idx])
            per_cutoff_scores[cutoff_pos].append(score)

    summaries = [_summary(scores) for scores in per_cutoff_scores]
    valid_means = [s["mean"] for s in summaries if s is not None]
    if not valid_means:
        raise SNClassificationError("no cutoff produced a usable train/test split across any seed")
    asymptotic = summaries[-1]["mean"] if summaries[-1] is not None else valid_means[-1]

    threshold = asymptotic_fraction * asymptotic
    time_to_classification = None
    for i, summary in enumerate(summaries):
        if summary is None or summary["mean"] < threshold:
            continue
        if all((s is not None and s["mean"] >= threshold) for s in summaries[i:]):
            time_to_classification = cutoff_grid_days[i]
            break

    return TimeToClassificationResult(
        cutoff_days=cutoff_grid_days, macro_f1_by_cutoff=summaries,
        asymptotic_macro_f1=asymptotic, time_to_classification_days=time_to_classification,
        n_objects=n, n_classes=n_classes,
    )


__all__ = ["LabeledCurve", "TimeToClassificationResult", "evaluate_time_to_classification"]
