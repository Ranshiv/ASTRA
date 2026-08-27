"""Active-learning researcher review (roadmap item 36, P1).

`review.select_next` already does uncertainty sampling (disagreement,
artifact uncertainty, tail uncertainty) plus greedy feature diversity --
a transparent, FIXED-weight heuristic (0.45/0.35/0.20, `review.py` line
~63). What it does NOT do, confirmed by reading it in full: it never
updates that weighting, or which of its own uncertainty signals is most
worth trusting, from accumulated human labels. This module adds exactly
that closed loop, reusing `review.select_next` and `candidates.
load_labels`/`label_summary` UNCHANGED rather than re-deriving
uncertainty or diversity scoring.

The reweighting idea -- learn, from observed label outcomes, which
acquisition signal has been most informative, and bias future batches
toward it -- is a real, named active-learning strategy: online/bandit
selection AMONG acquisition functions (Baram, El-Yaniv & Luz 2004,
"Online Choice of Active Learning Algorithms," JMLR 5). This module
narrows that to `select_next`'s own four fixed, human-readable `reasons`
tags ("detectors disagree", "artifact assessment is uncertain",
"significance is near the review boundary", "diverse candidate") --
each candidate `select_next` already returns is reused, unmodified, as
its own arm identity; `reason_yield`/`learn_reason_weights` never touch
`select_next`'s internal 0.45/0.35/0.20 formula, only its public output.

Confirmed reachable for "labels required per recovered rare-object":
`metadata.labels` (`metadata.py` line ~40) stores a real `recorded_utc`
per label, so a real chronological label history exists in this
codebase -- `labels_per_recovery` uses it directly, not a synthetic
proxy.

Confirmed UNREACHABLE for "reviewer agreement": `metadata.labels`'
schema keys the `labels` table on `candidate_key` ALONE (`metadata.py`
line ~40-43, `PRIMARY KEY`) -- one label per candidate, overwritten on
relabel. True inter-rater agreement (two independent reviewers labelling
the SAME candidate, compared) cannot be reconstructed from this storage
layer; there is nowhere a second, independent label could even be
written. `reviewer_agreement_with_priority` below is therefore a stated
PROXY -- agreement between the human's final label and what `review.
select_next`'s own priority score would have predicted -- not true
inter-rater agreement. `active_review_eval.py`'s synthetic dual-reviewer
study exercises real inter-rater-kappa arithmetic on synthetic labels to
show the metric mechanism is correct, since real dual-reviewer data does
not exist to check it against.

Explicitly NOT done: does not modify `review.py` or `candidates.py` in
any way -- `select_next`'s formula, `record_label`, and `load_labels`
are called, never forked. Does not add a field to `candidates.
Candidate`. Like every other opt-in research module in this codebase,
NOT wired into `rpc.py`, `scoring.WEIGHTS`, or `evidence.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import candidates, review

REASON_TAGS = ("detectors disagree", "artifact assessment is uncertain",
               "significance is near the review boundary", "diverse candidate")


class ActiveReviewError(ValueError):
    """A label history, reason weight, or reweighting input was invalid."""


def label_history(root: Path | None = None) -> list[dict]:
    """Real labels in `metadata.labels`, chronologically ordered by their
    real `recorded_utc` timestamp -- reused unchanged from `candidates.
    load_labels`, just sorted and flagged POSITIVE/negative via `review.
    POSITIVE`/`review.NEGATIVE`."""
    labels = candidates.load_labels(root)
    rows = [
        {"candidate_id": candidate_id, "label": entry.get("label"),
         "recorded_utc": entry.get("recorded_utc", ""),
         "is_positive": entry.get("label") in review.POSITIVE}
        for candidate_id, entry in labels.items()
        if entry.get("label") in review.POSITIVE | review.NEGATIVE
    ]
    rows.sort(key=lambda row: row["recorded_utc"])
    return rows


def labels_per_recovery(root: Path | None = None) -> dict:
    """Real gaps, in number of labels, between successive POSITIVE
    ("interesting"/"needs_follow_up") labels -- the roadmap item's own
    "labels required per recovered rare-object" metric, computed on real
    label timestamps, not simulated."""
    history = label_history(root)
    gaps: list[int] = []
    since_last = 0
    for row in history:
        since_last += 1
        if row["is_positive"]:
            gaps.append(since_last)
            since_last = 0
    return {
        "total_labels": len(history),
        "recoveries": len(gaps),
        "labels_per_recovery": gaps,
        "mean_labels_per_recovery": (sum(gaps) / len(gaps)) if gaps else None,
        "median_labels_per_recovery": (
            sorted(gaps)[len(gaps) // 2] if gaps and len(gaps) % 2 == 1
            else (sorted(gaps)[len(gaps) // 2 - 1] + sorted(gaps)[len(gaps) // 2]) / 2.0
            if gaps else None
        ),
    }


def reason_yield(items: list[object], root: Path | None = None) -> dict:
    """POSITIVE-label rate per `select_next` reason tag, over the FULL
    candidate pool (`limit=len(items)`, so every item gets a reason)
    matched against real labels. Calls `review.select_next` for its
    ranking and reasons UNCHANGED -- does not recompute uncertainty or
    diversity itself."""
    if not items:
        raise ActiveReviewError("items must be non-empty")
    ranked = review.select_next(items, limit=len(items))
    labels = candidates.load_labels(root)

    counts = {tag: {"positive": 0, "negative": 0} for tag in REASON_TAGS}
    for row in ranked:
        entry = labels.get(row["candidate_id"])
        if entry is None or entry.get("label") not in review.POSITIVE | review.NEGATIVE:
            continue
        bucket = "positive" if entry["label"] in review.POSITIVE else "negative"
        for reason in row["reasons"]:
            if reason in counts:
                counts[reason][bucket] += 1

    from . import significance
    rates = {}
    for tag, bucket in counts.items():
        total = bucket["positive"] + bucket["negative"]
        rates[tag] = {
            "positive": bucket["positive"], "negative": bucket["negative"], "total": total,
            "rate": (bucket["positive"] / total) if total else None,
            "ci95": significance._ci_binomial(bucket["positive"], total),
        }
    return {"n_candidates": len(items), "n_labelled": sum(v["total"] for v in rates.values()),
            "by_reason": rates}


@dataclass(frozen=True)
class ReasonWeights:
    weights: dict

    def to_dict(self) -> dict:
        return {"weights": {k: round(v, 6) for k, v in self.weights.items()}}


def learn_reason_weights(items: list[object], root: Path | None = None, *,
                         smoothing: float = 1.0) -> ReasonWeights:
    """Laplace-smoothed POSITIVE rate per reason tag from `reason_yield`,
    defaulting to uniform (1.0) weights when a tag has no labelled
    evidence yet -- a real, checkable cold-start behaviour."""
    if smoothing <= 0:
        raise ActiveReviewError(f"smoothing must be positive, got {smoothing}")
    yields = reason_yield(items, root)
    weights = {}
    for tag, stat in yields["by_reason"].items():
        if stat["total"] == 0:
            weights[tag] = 1.0
        else:
            weights[tag] = (stat["positive"] + smoothing) / (stat["total"] + 2.0 * smoothing)
    return ReasonWeights(weights=weights)


def reweighted_select_next(items: list[object], weights: ReasonWeights,
                           root: Path | None = None, *, limit: int = 20) -> list[dict]:
    """`review.select_next`'s own full-pool ranking, re-scored by the mean
    learned reason weight of each candidate's reasons, then truncated to
    `limit`. Already-labelled candidates are excluded -- an active-review
    queue should surface unlabelled work, not re-suggest reviewed items.
    Reuses `select_next` for ranking/reasons UNCHANGED; only the
    re-sorting after it is new."""
    if limit < 1:
        return []
    if not items:
        return []
    labels = candidates.load_labels(root)
    ranked = review.select_next(items, limit=len(items))
    unlabelled = [row for row in ranked if row["candidate_id"] not in labels]

    def rescored(row: dict) -> float:
        applicable = [weights.weights.get(reason, 1.0) for reason in row["reasons"]]
        multiplier = sum(applicable) / len(applicable) if applicable else 1.0
        return row["priority"] * multiplier

    unlabelled.sort(key=lambda row: (rescored(row), row["candidate_id"]), reverse=True)
    return unlabelled[:limit]


def reviewer_agreement_with_priority(name: str = "default", root: Path | None = None,
                                     *, threshold: float = 0.5) -> dict:
    """PROXY reviewer-agreement metric: how often a human's final
    POSITIVE/NEGATIVE label agrees with whether `review.select_next`'s
    own priority score (recomputed via `reason_yield`'s full-pool
    ranking) crossed `threshold`. NOT true inter-rater agreement -- see
    the module docstring for why that is unreachable in this codebase.
    Gated the same way `review.evaluate` gates, reusing its thresholds."""
    built = candidates.load(name, root)
    if not built:
        return {"ready": False, "reason": "no candidates found", "labels": 0}
    labels = candidates.load_labels(root)
    ranked = {row["candidate_id"]: row["priority"]
             for row in review.select_next(built, limit=len(built))}

    agreements, positives, negatives = 0, 0, 0
    pairs: list[tuple[int, int]] = []
    for candidate_id, entry in labels.items():
        label = entry.get("label")
        if label not in review.POSITIVE | review.NEGATIVE or candidate_id not in ranked:
            continue
        human_positive = label in review.POSITIVE
        predicted_positive = ranked[candidate_id] >= threshold
        positives += int(human_positive)
        negatives += int(not human_positive)
        agreements += int(human_positive == predicted_positive)
        pairs.append((int(human_positive), int(predicted_positive)))

    gate = {"minimum_labels": review.MIN_LABELS, "minimum_per_class": review.MIN_PER_CLASS,
            "labels": len(pairs), "positives": positives, "negatives": negatives}
    if len(pairs) < review.MIN_LABELS or min(positives, negatives) < review.MIN_PER_CLASS:
        return {"ready": False, "reason": "insufficient independent human labels", **gate}

    from sklearn.metrics import cohen_kappa_score
    human = [pair[0] for pair in pairs]
    predicted = [pair[1] for pair in pairs]
    return {"ready": True, **gate, "threshold": threshold,
            "observed_agreement": agreements / len(pairs),
            "cohen_kappa": float(cohen_kappa_score(human, predicted))}


__all__ = [
    "ActiveReviewError", "REASON_TAGS", "label_history", "labels_per_recovery",
    "reason_yield", "ReasonWeights", "learn_reason_weights", "reweighted_select_next",
    "reviewer_agreement_with_priority",
]
