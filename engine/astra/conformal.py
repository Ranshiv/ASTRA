"""Conformal anomaly uncertainty (roadmap item 34, P1).

The core arithmetic this module needs already exists in this codebase,
just not framed or validated as conformal: `significance.calibrate()`/
`annotate()` compute, for each score, `p = (1 + count(reference >=
score)) / (len(reference) + 1)` -- a plus-one-smoothed empirical tail
probability. This is EXACTLY the standard split-conformal p-value formula
(Vovk, Gammerman & Shafer 2005, *Algorithmic Learning in a Random World*,
Ch. 2). `conformal_p_values` below therefore does not reimplement that
arithmetic -- it calls `significance.calibrate`/`annotate` directly.

What this module adds is the three things that formula alone does not
give you, confirmed genuinely missing by exhaustive grep (zero hits for
"conformal"/"selective_risk"/"risk_coverage"/"abstain" anywhere in
`engine/astra` before this session):

1. An ENFORCED calibration/test split under a stated exchangeability
   assumption. `significance.calibrate` accepts any reference set with no
   split discipline; conformal validity specifically requires the
   calibration set and a test point to be exchangeable draws from the
   same ("normal"/non-anomalous) population. `split_calibration_stream`
   states that assumption explicitly rather than leaving it implicit.
2. Empirical coverage VALIDATION (`conformal_eval.py`) -- proving, not
   asserting, that flagging at a target false-positive rate `alpha`
   actually produces an empirical false-positive rate at or below `alpha`
   on repeated held-out trials. This is the roadmap item's own "empirical
   coverage" metric.
3. A selective-risk/risk-coverage curve (`selective_risk_coverage_curve`;
   El-Yaniv & Wiener 2010, "On the Foundations of Noise-free Selective
   Classification") -- genuinely new. Per-point confidence is `2 * |p -
   0.5|`, the complement of `review.select_next`'s own `tail_uncertainty`
   transform (`1.0 - abs(tail - 0.5) * 2.0`) -- reused for continuity
   with an existing in-house convention, not reinvented. This is the
   roadmap item's second named metric.

Framing: conformal anomaly detection, in the sense of Laxhammar & Falkman
(2010, "Conformal prediction for distribution-independent anomaly
detection"), treats an anomaly SCORE (from any existing detector --
`anomaly.py`'s per-detector scores or ensemble consensus, both reused
unchanged, though this module takes plain `np.ndarray`s and does not
import `anomaly.py` itself) as a NONCONFORMITY measure. A small p-value
means the test point's score is more extreme than almost every
calibration ("normal") score -- strong evidence it does not conform to
the normal population, i.e., is anomalous.

Explicitly NOT done: this does not touch or duplicate `significance.
calibrate`'s existing `rpc.py` exposure (`"significance.calibrate"`) --
it calls the same function, it does not fork it. Like every other opt-in
research module in this codebase, NOT wired into `rpc.py`,
`scoring.WEIGHTS`, or `evidence.py`, and does not add a field to
`candidates.Candidate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import significance


class ConformalError(ValueError):
    """A calibration split, p-value, or selective-prediction input was invalid."""


def split_calibration_stream(scores: np.ndarray, *, calibration_fraction: float = 0.5,
                             seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """A plain random split of a held-out "normal" score stream into a
    calibration set and a remainder -- not a new statistical method, just
    the split discipline conformal prediction requires. Exchangeability
    (calibration and test points are draws from the same distribution
    under the null "this point is normal") is a real assumption this
    function does NOT verify -- it holds by construction when both sides
    of the split come from a genuinely un-flagged population, and is a
    real, unverifiable-in-general assumption for arbitrary deployment
    data. Stated here, not glossed over."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ConformalError(f"scores must be 1-D, got shape {scores.shape}")
    if not 0.0 < calibration_fraction < 1.0:
        raise ConformalError(f"calibration_fraction must be in (0, 1), got {calibration_fraction}")
    if len(scores) < 2:
        raise ConformalError("scores must contain at least 2 values to split")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(scores))
    cut = max(1, int(round(len(scores) * calibration_fraction)))
    cut = min(cut, len(scores) - 1)
    calibration_idx, remainder_idx = order[:cut], order[cut:]
    return scores[calibration_idx], scores[remainder_idx]


def conformal_p_values(test_scores: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    """Split-conformal p-values for `test_scores` against a held-out
    `calibration_scores` population -- delegates to `significance.
    calibrate`/`annotate` for the actual arithmetic (reused unchanged)."""
    test_scores = np.asarray(test_scores, dtype=np.float64)
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    if test_scores.ndim != 1:
        raise ConformalError(f"test_scores must be 1-D, got shape {test_scores.shape}")
    if len(calibration_scores) == 0:
        raise ConformalError("calibration_scores must be non-empty")
    if len(test_scores) == 0:
        return np.empty(0, dtype=np.float64)

    calibration_report = significance.calibrate(test_scores, reference_scores=calibration_scores)
    rows = significance.annotate(test_scores, calibration_report, reference_scores=calibration_scores)
    return np.array([row["tail_probability"] for row in rows], dtype=np.float64)


def flag_at_alpha(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """`True` where a conformal p-value is small enough to flag as
    anomalous at a target false-positive rate `alpha`."""
    if not 0.0 < alpha < 1.0:
        raise ConformalError(f"alpha must be in (0, 1), got {alpha}")
    return np.asarray(p_values, dtype=np.float64) <= alpha


@dataclass(frozen=True)
class ConformalDecision:
    p_value: float
    accepted: bool
    predicted_label: str | None  # "anomaly" | "normal" | None (abstained)
    confidence: float

    def to_dict(self) -> dict:
        return {"p_value": round(self.p_value, 6), "accepted": self.accepted,
                "predicted_label": self.predicted_label, "confidence": round(self.confidence, 6)}


def selective_predict(p_values: np.ndarray, *, coverage: float) -> list[ConformalDecision]:
    """Accepts the `coverage`-fraction of highest-confidence points
    (confidence `= 2 * |p - 0.5|`, the complement of `review.py`'s own
    uncertainty transform), predicting "anomaly" when `p < 0.5` else
    "normal"; abstains (`predicted_label=None`) on the rest."""
    p_values = np.asarray(p_values, dtype=np.float64)
    if not 0.0 < coverage <= 1.0:
        raise ConformalError(f"coverage must be in (0, 1], got {coverage}")
    if p_values.ndim != 1 or len(p_values) == 0:
        raise ConformalError("p_values must be a non-empty 1-D array")

    confidence = 2.0 * np.abs(p_values - 0.5)
    n_accept = max(1, int(round(len(p_values) * coverage)))
    # Ties broken by index for determinism, not by confidence value alone.
    order = np.argsort(-confidence, kind="stable")
    accepted_idx = set(order[:n_accept].tolist())

    decisions = []
    for index, (p_value, conf) in enumerate(zip(p_values, confidence)):
        accepted = index in accepted_idx
        label = ("anomaly" if p_value < 0.5 else "normal") if accepted else None
        decisions.append(ConformalDecision(
            p_value=float(p_value), accepted=accepted, predicted_label=label,
            confidence=float(conf)))
    return decisions


def selective_risk_coverage_curve(p_values: np.ndarray, true_labels: np.ndarray, *,
                                  coverage_levels: tuple[float, ...] =
                                  (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> list[dict]:
    """`[{coverage, risk, n_accepted}]` -- at each coverage level, the
    misclassification rate among accepted (non-abstained) decisions
    against `true_labels` (1 = anomaly, 0 = normal). The roadmap item's
    own "selective-risk curve" metric."""
    p_values = np.asarray(p_values, dtype=np.float64)
    true_labels = np.asarray(true_labels)
    if len(p_values) != len(true_labels):
        raise ConformalError("p_values and true_labels must have the same length")
    if not coverage_levels:
        raise ConformalError("coverage_levels must be non-empty")

    curve = []
    for coverage in coverage_levels:
        decisions = selective_predict(p_values, coverage=coverage)
        predicted = np.array([1 if d.predicted_label == "anomaly" else 0
                              for d in decisions if d.accepted])
        truth = np.array([true_labels[i] for i, d in enumerate(decisions) if d.accepted])
        n_accepted = len(predicted)
        risk = float(np.mean(predicted != truth)) if n_accepted else None
        curve.append({"coverage": round(float(coverage), 4), "risk": risk, "n_accepted": n_accepted})
    return curve


__all__ = [
    "ConformalError", "split_calibration_stream", "conformal_p_values", "flag_at_alpha",
    "ConformalDecision", "selective_predict", "selective_risk_coverage_curve",
]
