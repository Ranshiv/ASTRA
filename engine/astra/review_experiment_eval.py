"""Evaluation for the reviewer human-factors experiment (Direction 6:
"the review UI as a controlled experiment").

Every function here is a pure function over vote rows -- the shape
`candidates.all_label_votes`/`metadata.all_label_votes` returns, each with
`arm`, `displayed_score`, `decision_latency_ms`, `self_reported_confidence`
from `review_experiment.py`'s additive columns -- and an externally supplied
`truth` mapping, mirroring `discard_corroboration.py`/`discard_adjudication.
py`'s "the caller resolves ground truth, this module only analyses"
layering. A vote whose `arm` is `None` (cast through the ordinary,
non-experimental `candidates.cast_label_vote`) is excluded from every
analysis here -- see `review_experiment.py`'s own docstring for why that is
correct, not an oversight.

`score_shuffled` is the actual control that makes the anchoring claim
falsifiable: if `score_shuffled` reviewers agree with `score_blinded`
reviewers (both saw a number with no relationship, or no number, to this
candidate) but NOT with `score_shown` reviewers, that is anchoring on
whatever number appears on screen, not information carried by the real
score.
"""

from __future__ import annotations

from typing import Any

from . import review

ARMS = ("score_shown", "score_blinded", "score_shuffled")


def _is_positive(label: str) -> bool | None:
    if label in review.POSITIVE:
        return True
    if label in review.NEGATIVE:
        return False
    return None


def _usable_votes(votes: list[dict], truth: dict[str, bool]) -> list[dict]:
    """Votes that belong to the experiment, carry a scoreable label, and
    have known ground truth -- the intersection every analysis below needs."""
    usable = []
    for vote in votes:
        if vote.get("arm") not in ARMS:
            continue
        positive = _is_positive(vote.get("label", ""))
        if positive is None or vote.get("candidate_key") not in truth:
            continue
        usable.append({**vote, "_positive": positive,
                       "_correct": positive == truth[vote["candidate_key"]]})
    return usable


def anchoring_effect_size(votes: list[dict], truth: dict[str, bool]) -> dict[str, Any]:
    """Per-arm accuracy against truth, plus within-arm cross-reviewer
    agreement -- the two numbers together are what separate anchoring from
    real signal (see this module's docstring).
    """
    usable = _usable_votes(votes, truth)
    per_arm: dict[str, dict[str, Any]] = {}

    for arm in ARMS:
        arm_votes = [vote for vote in usable if vote["arm"] == arm]
        n = len(arm_votes)
        accuracy = (sum(vote["_correct"] for vote in arm_votes) / n) if n else None

        # Cross-reviewer agreement: for every candidate with >=2 votes in
        # THIS arm, the fraction of same-labelled pairs among all pairs --
        # the same pairwise-agreement idea `candidates.label_vote_tally`
        # already uses for its majority fraction, computed per arm here
        # instead of pooling every arm together.
        by_candidate: dict[str, list[bool]] = {}
        for vote in arm_votes:
            by_candidate.setdefault(vote["candidate_key"], []).append(vote["_positive"])
        agreeing_pairs = total_pairs = 0
        for positives in by_candidate.values():
            count = len(positives)
            if count < 2:
                continue
            positive_count = sum(positives)
            negative_count = count - positive_count
            agreeing_pairs += (positive_count * (positive_count - 1)
                              + negative_count * (negative_count - 1))
            total_pairs += count * (count - 1)
        cross_reviewer_agreement = (agreeing_pairs / total_pairs) if total_pairs else None

        per_arm[arm] = {
            "n_votes": n, "n_candidates": len(by_candidate),
            "accuracy": (round(accuracy, 4) if accuracy is not None else None),
            "cross_reviewer_agreement": (round(cross_reviewer_agreement, 4)
                                         if cross_reviewer_agreement is not None else None),
        }

    shown = per_arm["score_shown"]["accuracy"]
    blinded = per_arm["score_blinded"]["accuracy"]
    shuffled = per_arm["score_shuffled"]["accuracy"]
    # A real anchoring signature: shuffled tracks blinded (both saw a number
    # unrelated to ground truth, or no number) while shown diverges from
    # both -- computed only when all three arms have a usable accuracy.
    anchoring_signature_detected = None
    if None not in (shown, blinded, shuffled):
        anchoring_signature_detected = bool(
            abs(shuffled - blinded) < abs(shown - blinded)
            and abs(shuffled - blinded) < abs(shown - shuffled))

    return {"by_arm": per_arm, "anchoring_signature_detected": anchoring_signature_detected}


def calibration_curve(votes: list[dict], truth: dict[str, bool], *,
                      n_bins: int = 5) -> dict[str, Any]:
    """Self-reported confidence vs. realized accuracy, per arm and
    confidence bin. A well-calibrated reviewer's accuracy in the "80-100%
    confident" bin should be near 0.9; a reviewer anchored on a displayed
    score may instead show HIGH confidence with LOW realized accuracy in
    `score_shown` specifically.
    """
    usable = [vote for vote in _usable_votes(votes, truth)
             if isinstance(vote.get("self_reported_confidence"), (int, float))]
    edges = [i / n_bins for i in range(n_bins + 1)]

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        bins = []
        arm_votes = [vote for vote in usable if vote["arm"] == arm]
        for index in range(n_bins):
            low, high = edges[index], edges[index + 1]
            in_bin = [vote for vote in arm_votes
                     if low <= vote["self_reported_confidence"] <= high]
            if not in_bin:
                bins.append({"bin_low": low, "bin_high": high, "n": 0,
                            "mean_confidence": None, "accuracy": None})
                continue
            bins.append({
                "bin_low": low, "bin_high": high, "n": len(in_bin),
                "mean_confidence": round(
                    sum(v["self_reported_confidence"] for v in in_bin) / len(in_bin), 4),
                "accuracy": round(sum(v["_correct"] for v in in_bin) / len(in_bin), 4),
            })
        by_arm[arm] = bins

    return {"n_bins": n_bins, "by_arm": by_arm}


def _blinded_human_positivity(votes: list[dict], truth: dict[str, bool]) -> dict[str, float]:
    """Per-candidate fraction of `score_blinded`-arm votes marking positive
    -- the human signal for the ensemble, deliberately restricted to the
    arm that cannot be contaminated by the displayed score.
    """
    usable = [vote for vote in _usable_votes(votes, truth) if vote["arm"] == "score_blinded"]
    by_candidate: dict[str, list[bool]] = {}
    for vote in usable:
        by_candidate.setdefault(vote["candidate_key"], []).append(vote["_positive"])
    return {candidate_id: sum(positives) / len(positives)
           for candidate_id, positives in by_candidate.items()}


def ensemble_accounting(votes: list[dict], truth: dict[str, bool],
                        model_scores: dict[str, float], *,
                        threshold: float = 0.5) -> dict[str, Any]:
    """Model-alone, human-alone (blinded arm only), and combined accuracy
    against truth -- the "does human+model beat either alone once the
    human's contribution is decorrelated from the model's" question the
    research plan asks.

    The combined score is the simplest defensible ensemble: an unweighted
    average of the model score and the blinded-arm human positivity
    fraction, thresholded at 0.5 -- the same fixed, not-selected-on-this-
    set threshold `review.evaluate`'s own `scores >= 0.5` gate already
    uses, not a value tuned to make this comparison look favourable.
    """
    human = _blinded_human_positivity(votes, truth)
    candidate_ids = sorted(set(truth) & set(model_scores) & set(human))

    def _accuracy(predictions: dict[str, bool]) -> float | None:
        if not candidate_ids:
            return None
        correct = sum(predictions[cid] == truth[cid] for cid in candidate_ids)
        return round(correct / len(candidate_ids), 4)

    model_predictions = {cid: model_scores[cid] >= threshold for cid in candidate_ids}
    human_predictions = {cid: human[cid] >= threshold for cid in candidate_ids}
    combined_predictions = {
        cid: (0.5 * model_scores[cid] + 0.5 * human[cid]) >= threshold
        for cid in candidate_ids
    }

    return {
        "n_candidates": len(candidate_ids), "threshold": threshold,
        "model_accuracy": _accuracy(model_predictions),
        "human_accuracy": _accuracy(human_predictions),
        "combined_accuracy": _accuracy(combined_predictions),
    }
