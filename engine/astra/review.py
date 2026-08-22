"""Evaluation against researcher labels, gated against misleading tiny samples."""
from __future__ import annotations

import numpy as np

from . import candidates

POSITIVE = {"interesting", "needs_follow_up"}
NEGATIVE = {"artifact", "known_object"}
# The old metric-only report was useful for smoke tests but too easy to
# over-interpret.  Match the supervised ranker's release gate so a green
# review report always represents a meaningful human-labelled sample.
MIN_LABELS = 50
MIN_PER_CLASS = 10


def _candidate_value(candidate: object, name: str, default: float = 0.0) -> float:
    if isinstance(candidate, dict):
        value = candidate.get(name)
    else:
        value = getattr(candidate, name, None)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _mapping(candidate: object, name: str) -> dict:
    value = candidate.get(name, {}) if isinstance(candidate, dict) else getattr(candidate, name, {})
    return value if isinstance(value, dict) else {}


def select_next(items: list[object], limit: int = 20) -> list[dict]:
    """Select candidates that maximize expected information from review.

    This is deliberately a transparent heuristic, not a hidden model:
    uncertainty (artifact likelihood near 0.5, weak detector agreement, or a
    calibrated tail probability near 0.5) is combined with greedy feature
    diversity.  The returned reasons make every selection auditable.
    """
    if limit < 1:
        return []
    pool: list[dict] = []
    for item in items:
        candidate_id = (item.get("candidate_id") if isinstance(item, dict)
                        else getattr(item, "candidate_id", None))
        if not candidate_id:
            continue
        score = _mapping(item, "score")
        artifact = _mapping(item, "artifact")
        significance = _mapping(item, "significance")
        agreement = _candidate_value(score, "model_agreement", 0.5)
        # Agreement is usually an integer detector count; normalize either
        # count or an already-normalized value to a disagreement score.
        disagreement = 1.0 - (agreement / 4.0 if agreement > 1.0 else agreement)
        artifact_uncertainty = 1.0 - abs(_candidate_value(artifact, "likelihood", 0.5) - 0.5) * 2.0
        tail = significance.get("tail_probability")
        try:
            tail_uncertainty = 1.0 - abs(float(tail) - 0.5) * 2.0
        except (TypeError, ValueError):
            tail_uncertainty = 0.5
        priority = float(np.clip(0.45 * disagreement + 0.35 * artifact_uncertainty
                                 + 0.20 * tail_uncertainty, 0.0, 1.0))
        reasons = []
        if disagreement >= 0.5:
            reasons.append("detectors disagree")
        if artifact_uncertainty >= 0.7:
            reasons.append("artifact assessment is uncertain")
        if 0.25 <= (float(tail) if isinstance(tail, (int, float)) else 0.5) <= 0.75:
            reasons.append("significance is near the review boundary")
        pool.append({"candidate_id": str(candidate_id), "priority": round(priority, 6),
                     "reasons": reasons or ["diverse candidate"],
                     "_features": _mapping(item, "features")})

    selected: list[dict] = []
    while pool and len(selected) < limit:
        if not selected:
            chosen = max(pool, key=lambda row: (row["priority"], row["candidate_id"]))
        else:
            def marginal(row: dict) -> tuple[float, str]:
                vector = row["_features"]
                distances = []
                for prior in selected:
                    prior_vector = prior.get("_features", {})
                    keys = set(vector) & set(prior_vector)
                    values = []
                    for key in keys:
                        try:
                            left, right = float(vector[key]), float(prior_vector[key])
                            if np.isfinite(left) and np.isfinite(right):
                                values.append((left - right) ** 2)
                        except (TypeError, ValueError):
                            continue
                    distances.append(float(np.sqrt(np.mean(values))) if values else 0.0)
                diversity = min(distances) if distances else 0.0
                return (0.8 * row["priority"] + 0.2 * min(1.0, diversity), row["candidate_id"])
            chosen = max(pool, key=marginal)
        pool.remove(chosen)
        result = {key: value for key, value in chosen.items() if not key.startswith("_")}
        selected.append({**result, "_features": chosen.get("_features", {})})

    return [{key: value for key, value in row.items() if not key.startswith("_")}
            for row in selected]


def evaluate(name: str = "default", root=None) -> dict:
    built = candidates.load(name, root)
    labels = candidates.load_labels(root)
    y_true, y_score = [], []
    for item in built:
        label = labels.get(item.candidate_id, {}).get("label")
        if label in POSITIVE | NEGATIVE:
            y_true.append(1 if label in POSITIVE else 0)
            y_score.append(float(item.score.get("total", 0.0)))
    positives, negatives = sum(y_true), len(y_true) - sum(y_true)
    gate = {"minimum_labels": MIN_LABELS, "minimum_per_class": MIN_PER_CLASS,
            "labels": len(y_true), "positives": positives, "negatives": negatives}
    if len(y_true) < MIN_LABELS or min(positives, negatives) < MIN_PER_CLASS:
        return {"ready": False, "reason": "insufficient independent human labels", **gate}
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    y = np.asarray(y_true); scores = np.asarray(y_score)
    # Threshold is deliberately fixed rather than selected on this same set.
    prediction = scores >= 0.5
    return {"ready": True, **gate,
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "threshold": 0.5}
