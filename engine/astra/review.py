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
