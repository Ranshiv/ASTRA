"""Empirical coverage and the selective-risk curve -- the two metrics
roadmap item 34 names -- split from `conformal.py` purely to keep each
file under this project's 500-line guideline (same `stellar_manifold.py`/
`stellar_manifold_eval.py` split rationale, not an independent module).

`evaluate_conformal_coverage_synthetic` checks the exact theoretical
property split-conformal prediction guarantees: for exchangeable draws
from the SAME distribution, a conformal p-value is (super-)uniform on
`[0, 1]`, so flagging at `p <= alpha` produces a false-positive rate at or
below `alpha` in expectation. This is verified directly on synthetic data
constructed to be exactly exchangeable by construction, not merely cited.

`evaluate_conformal_coverage_on_injected_anomalies` is the one real
integration-level check in this pair: it reuses `evaluate.py`'s real
injection-recovery harness (`build_injected`/`sequence_summary`) and
`anomaly.detect()`'s real ensemble consensus score (both reused
unchanged) rather than only the abstract synthetic case, grounding the
coverage guarantee in this codebase's actual anomaly pipeline. Real
deployment data is not, in general, provably exchangeable with a held-out
calibration set -- this study is the closest available real check, not a
substitute for that assumption.

`evaluate_selective_risk_coverage` demonstrates `conformal.
selective_risk_coverage_curve` on a synthetic, well-separated anomaly
class, checking risk decreases as coverage (the accepted fraction)
decreases -- a real, checkable monotonicity property of selective
prediction, not merely "a curve gets produced."

Both studies validated on real (integration) or explicitly synthetic
ground truth respectively -- the same "mechanism validated, not yet run
at real Stage-B scale" caveat every eval module in this family states.
"""

from __future__ import annotations

import numpy as np

from . import significance
from .conformal import (
    ConformalError, conformal_p_values, flag_at_alpha, selective_risk_coverage_curve,
    split_calibration_stream,
)


def evaluate_conformal_coverage_synthetic(alpha_levels: tuple[float, ...] = (0.05, 0.1, 0.2),
                                          n_trials: int = 200, n_calibration: int = 200,
                                          n_test: int = 200, seed: int = 42) -> dict:
    """False-positive rate among exchangeable synthetic "normal" test
    points, across `n_trials` independent calibration/test draws from the
    SAME distribution -- checked against each nominal `alpha` with a
    Wilson 95% CI (`significance._ci_binomial`, reused unchanged)."""
    if n_trials < 1:
        raise ConformalError(f"n_trials must be at least 1, got {n_trials}")
    if n_calibration < 1 or n_test < 1:
        raise ConformalError("n_calibration and n_test must be at least 1")
    if not alpha_levels:
        raise ConformalError("alpha_levels must be non-empty")

    rng = np.random.default_rng(seed)
    flagged_counts = {alpha: 0 for alpha in alpha_levels}
    total_tested = 0
    for _ in range(n_trials):
        # Calibration and test scores are i.i.d. draws from the identical
        # distribution -- exchangeable by construction, the exact regime
        # split-conformal's guarantee assumes.
        calibration_scores = rng.normal(size=n_calibration)
        test_scores = rng.normal(size=n_test)
        p_values = conformal_p_values(test_scores, calibration_scores)
        total_tested += len(p_values)
        for alpha in alpha_levels:
            flagged_counts[alpha] += int(flag_at_alpha(p_values, alpha).sum())

    levels = {
        str(alpha): {
            "nominal": alpha,
            "empirical": (flagged_counts[alpha] / total_tested) if total_tested else None,
            "flagged": flagged_counts[alpha], "n_tested": total_tested,
            "ci95": significance._ci_binomial(flagged_counts[alpha], total_tested),
        }
        for alpha in alpha_levels
    }
    return {"n_trials": n_trials, "n_calibration": n_calibration, "n_test": n_test, "levels": levels}


def evaluate_conformal_coverage_on_injected_anomalies(values: np.ndarray, identities: list[dict], *,
                                                       alpha_levels: tuple[float, ...] = (0.05, 0.1, 0.2),
                                                       n_trials: int = 20, fraction: float = 0.1,
                                                       strength: float = 6.0,
                                                       calibration_fraction: float = 0.5,
                                                       seed: int = 42) -> dict:
    """The same false-positive-rate-vs-alpha check as above, but on real
    `anomaly.detect()` ensemble scores from `evaluate.py`'s real
    injection-recovery harness, not a synthetic score distribution."""
    if len(values) < 20:
        raise ConformalError(f"values must contain at least 20 sequences, got {len(values)}")
    if n_trials < 1:
        raise ConformalError(f"n_trials must be at least 1, got {n_trials}")

    from . import anomaly
    from .evaluate import build_injected, sequence_summary
    from .featurematrix import FeatureMatrix

    flagged_counts = {alpha: 0 for alpha in alpha_levels}
    total_tested = 0
    trials_used = 0
    for trial in range(n_trials):
        injection = build_injected(values, identities, fraction=fraction,
                                   strength=strength, seed=seed + trial)
        summary = sequence_summary(injection.values)
        matrix = FeatureMatrix(values=summary, identities=injection.identities,
                               feature_names=tuple(f"seq_{i}" for i in range(summary.shape[1])))
        ensemble = anomaly.detect(matrix, seed=seed + trial)
        if not ensemble.detectors:
            continue

        usable = matrix.finite_mask()
        scores, labels = ensemble.consensus, injection.labels[usable]
        normal_scores, anomaly_scores = scores[labels == 0], scores[labels == 1]
        if len(normal_scores) < 4:
            continue

        calibration_scores, remaining_normal = split_calibration_stream(
            normal_scores, calibration_fraction=calibration_fraction, seed=seed + trial)
        test_scores = np.concatenate([remaining_normal, anomaly_scores])
        test_is_normal = np.concatenate(
            [np.ones(len(remaining_normal), dtype=bool), np.zeros(len(anomaly_scores), dtype=bool)])
        p_values = conformal_p_values(test_scores, calibration_scores)

        trials_used += 1
        total_tested += int(test_is_normal.sum())
        for alpha in alpha_levels:
            flags = flag_at_alpha(p_values, alpha)
            flagged_counts[alpha] += int(np.sum(flags & test_is_normal))

    levels = {
        str(alpha): {
            "nominal": alpha,
            "empirical": (flagged_counts[alpha] / total_tested) if total_tested else None,
            "false_positives": flagged_counts[alpha], "n_normal_tested": total_tested,
            "ci95": significance._ci_binomial(flagged_counts[alpha], total_tested),
        }
        for alpha in alpha_levels
    }
    return {"n_trials_requested": n_trials, "n_trials_used": trials_used, "levels": levels}


def evaluate_selective_risk_coverage(*, n_calibration: int = 200, n_normal_test: int = 150,
                                     n_anomaly_test: int = 50,
                                     coverage_levels: tuple[float, ...] =
                                     (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                                     anomaly_shift: float = 4.0, seed: int = 42) -> dict:
    """`conformal.selective_risk_coverage_curve` on a synthetic,
    well-separated anomaly class (mean-shifted by `anomaly_shift`
    standard deviations from the calibration population)."""
    if n_calibration < 1 or n_normal_test < 1 or n_anomaly_test < 1:
        raise ConformalError("n_calibration, n_normal_test, and n_anomaly_test must be at least 1")

    rng = np.random.default_rng(seed)
    calibration_scores = rng.normal(size=n_calibration)
    normal_test = rng.normal(size=n_normal_test)
    anomaly_test = rng.normal(loc=anomaly_shift, size=n_anomaly_test)
    test_scores = np.concatenate([normal_test, anomaly_test])
    true_labels = np.concatenate([np.zeros(n_normal_test), np.ones(n_anomaly_test)])

    p_values = conformal_p_values(test_scores, calibration_scores)
    curve = selective_risk_coverage_curve(p_values, true_labels, coverage_levels=coverage_levels)
    return {"n_calibration": n_calibration, "n_test": len(test_scores), "curve": curve}


__all__ = [
    "evaluate_conformal_coverage_synthetic", "evaluate_conformal_coverage_on_injected_anomalies",
    "evaluate_selective_risk_coverage",
]
