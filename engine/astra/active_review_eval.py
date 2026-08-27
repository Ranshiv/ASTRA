"""Evaluation studies for `active_review.py`, split purely to keep each
file under this project's 500-line guideline (same `conformal.py`/
`conformal_eval.py` split rationale).

`evaluate_reason_informativeness`/`evaluate_labels_per_recovery` run the
real, reachable metrics directly against whatever labels exist under
`root` -- both degrade to `None`/zero-count fields rather than raising
when too few labels exist, the same convention `review.evaluate` uses.

`evaluate_reweighted_selection_recovers_high_yield_reason_synthetic` is
the one integration-level check: it builds a labelled SYNTHETIC training
pool via `candidates.record_label` (real storage, real
`review.select_next` reasons -- only the candidate feature values are
synthetic) where one reason tag has a deliberately higher positive
yield, learns weights from it, then applies `active_review.
reweighted_select_next` to a FRESH unlabelled synthetic pool from the
same distribution and checks the top selections are dominated by the
high-yield reason -- a real, checkable property of the reweighting
mechanism, not merely "weights get produced."

`evaluate_synthetic_dual_reviewer_kappa` exercises real Cohen's-kappa
arithmetic on synthetic two-reviewer labels at controlled true-agreement
levels, since real dual-reviewer data does not exist in this codebase
(see `active_review.py`'s module docstring) to check the mechanism
against directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import candidates
from .active_review import (
    ActiveReviewError, learn_reason_weights, labels_per_recovery,
    reason_yield, reweighted_select_next,
)


def evaluate_reason_informativeness(items: list[object], root: Path | None = None) -> dict:
    """Real `reason_yield`/`learn_reason_weights` against whatever labels
    exist under `root`; returns `None` weights only when `items` is
    empty (a genuine usage error, not a data-sparsity case)."""
    if not items:
        return {"ready": False, "reason": "no candidates supplied"}
    yields = reason_yield(items, root)
    weights = learn_reason_weights(items, root)
    return {"ready": True, "yields": yields, "weights": weights.to_dict()}


def evaluate_labels_per_recovery(root: Path | None = None) -> dict:
    """Real `labels_per_recovery` against whatever labels exist under
    `root` -- already degrades to empty/`None` fields with zero labels."""
    return labels_per_recovery(root)


def _reason_candidate(candidate_id: str, reason: str) -> dict:
    """A synthetic candidate whose `review.select_next` reasons are
    EXACTLY `{reason}`, by direct construction from the documented
    thresholds in `review.select_next` (`review.py` lines ~53-71)."""
    presets = {
        "detectors disagree": {"agreement": 0.3, "likelihood": 0.05, "tail": 0.95},
        "artifact assessment is uncertain": {"agreement": 1.0, "likelihood": 0.5, "tail": 0.95},
        "significance is near the review boundary": {"agreement": 1.0, "likelihood": 0.05, "tail": 0.5},
        "diverse candidate": {"agreement": 1.0, "likelihood": 0.05, "tail": 0.95},
    }
    if reason not in presets:
        raise ActiveReviewError(f"unknown reason preset: {reason!r}")
    preset = presets[reason]
    return {
        "candidate_id": candidate_id,
        "score": {"model_agreement": preset["agreement"]},
        "artifact": {"likelihood": preset["likelihood"]},
        "significance": {"tail_probability": preset["tail"]},
        "features": {"x": hash(candidate_id) % 1000 / 1000.0},
    }


def evaluate_reweighted_selection_recovers_high_yield_reason_synthetic(
        root: Path, *, n_per_reason: int = 25, limit: int = 20,
        high_yield_reason: str = "detectors disagree",
        high_yield_rate: float = 0.8, low_yield_rate: float = 0.1,
        seed: int = 42) -> dict:
    """Builds a labelled training pool where `high_yield_reason` has a
    deliberately higher POSITIVE rate than the other three reasons,
    learns weights from it, then checks a FRESH unlabelled pool's
    `reweighted_select_next` top `limit` selections are enriched for
    `high_yield_reason` relative to its base rate in the pool."""
    if n_per_reason < 1:
        raise ActiveReviewError(f"n_per_reason must be at least 1, got {n_per_reason}")
    from .active_review import REASON_TAGS

    rng = np.random.default_rng(seed)
    training_items = []
    for reason in REASON_TAGS:
        rate = high_yield_rate if reason == high_yield_reason else low_yield_rate
        for i in range(n_per_reason):
            candidate_id = f"train_{reason.replace(' ', '_')}_{i}"
            training_items.append(_reason_candidate(candidate_id, reason))
            label = "interesting" if rng.random() < rate else "artifact"
            candidates.record_label(candidate_id, label, root=root)

    weights = learn_reason_weights(training_items, root)

    fresh_items = []
    for reason in REASON_TAGS:
        for i in range(n_per_reason):
            fresh_items.append(_reason_candidate(f"fresh_{reason.replace(' ', '_')}_{i}", reason))

    selection = reweighted_select_next(fresh_items, weights, root, limit=limit)
    selected_high_yield = sum(1 for row in selection if high_yield_reason in row["reasons"])
    base_rate = 1.0 / len(REASON_TAGS)

    return {
        "weights": weights.to_dict(), "limit": limit,
        "selected_high_yield_reason_fraction": selected_high_yield / len(selection) if selection else None,
        "base_rate": base_rate,
        "enriched": (selected_high_yield / len(selection)) > base_rate if selection else False,
    }


def evaluate_synthetic_dual_reviewer_kappa(
        agreement_levels: tuple[float, ...] = (0.5, 0.7, 0.9, 1.0),
        n_labels: int = 200, seed: int = 42) -> dict:
    """Real Cohen's-kappa arithmetic on SYNTHETIC two-reviewer binary
    labels constructed at a controlled true-agreement probability --
    checks kappa is non-decreasing in the injected agreement level, a
    real, checkable property of the metric mechanism itself."""
    if not agreement_levels:
        raise ActiveReviewError("agreement_levels must be non-empty")
    if n_labels < 1:
        raise ActiveReviewError(f"n_labels must be at least 1, got {n_labels}")

    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(seed)
    results = []
    for agreement in agreement_levels:
        reviewer_a = rng.integers(0, 2, size=n_labels)
        flips = rng.random(n_labels) >= agreement
        reviewer_b = np.where(flips, 1 - reviewer_a, reviewer_a)
        kappa = float(cohen_kappa_score(reviewer_a, reviewer_b))
        results.append({"injected_agreement": agreement,
                        "observed_agreement": float(np.mean(reviewer_a == reviewer_b)),
                        "cohen_kappa": kappa})

    kappas = [row["cohen_kappa"] for row in results]
    return {"n_labels": n_labels, "levels": results,
            "monotonic_non_decreasing": all(kappas[i] <= kappas[i + 1] + 1e-9
                                            for i in range(len(kappas) - 1))}


__all__ = [
    "evaluate_reason_informativeness", "evaluate_labels_per_recovery",
    "evaluate_reweighted_selection_recovers_high_yield_reason_synthetic",
    "evaluate_synthetic_dual_reviewer_kappa",
]
