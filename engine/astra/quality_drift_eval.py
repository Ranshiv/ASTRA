"""Evaluation studies for `quality_drift.py`, split purely to keep each
file under this project's 500-line guideline.

`evaluate_calibration_achieves_target_fpr_synthetic` verifies
`calibrate_threshold`'s own guarantee: a threshold calibrated on a
nominal (drift-free) stream should produce an empirical false-positive
rate at or near `target_fpr` on FRESH nominal streams from the same
distribution -- checked directly, with a Wilson 95% CI (`significance.
_ci_binomial`, reused unchanged), not merely cited. Uses a long
(`n_reference=1000`) reference by default: `calibrate_threshold`'s own
docstring now documents, from a real check made this session, that a
short reference (200 points) under-estimates `reference_mean`/
`reference_std` enough to overshoot the target FPR (38% observed vs. a
10% target) purely from plug-in estimation noise, not a calibration
bug -- this study's default avoids re-triggering that same finite-
sample effect while still checking the real calibration mechanism.

`evaluate_detection_delay_and_false_alarm_rate_synthetic` is the
roadmap item's own two named metrics, computed directly: for synthetic
streams with a KNOWN injected mean-shift at a known index, the number of
samples between injection and the first alarm (detection delay) across
many trials, and, separately, the false-alarm rate on pure-nominal
streams with no injected shift. Real live telemetry does not exist in
this codebase (see `quality_drift.py`'s module docstring) -- both
studies use synthetic streams, honestly labelled.
"""

from __future__ import annotations

import numpy as np

from . import significance
from .quality_drift import (
    QualityDriftError, calibrate_threshold, detect_changepoints, standardize_stream,
)


def evaluate_calibration_achieves_target_fpr_synthetic(
        target_fpr: float = 0.05, n_stream: int = 300, n_reference: int = 1000,
        n_trials: int = 100, drift: float = 0.5, seed: int = 42) -> dict:
    """Calibrates a threshold from an `n_reference`-point nominal
    reference, then checks the empirical alarm rate across `n_trials`
    FRESH `n_stream`-point nominal streams from the identical
    distribution."""
    if n_reference < 10:
        raise QualityDriftError(f"n_reference must be at least 10, got {n_reference}")
    if n_trials < 1:
        raise QualityDriftError("n_trials must be at least 1")

    rng = np.random.default_rng(seed)
    reference = rng.normal(loc=0.0, scale=1.0, size=n_reference)
    threshold = calibrate_threshold(reference, drift=drift, target_fpr=target_fpr,
                                    stream_length=n_stream, seed=seed)
    reference_mean, reference_std = float(np.mean(reference)), float(np.std(reference))

    alarmed = 0
    for trial in range(n_trials):
        fresh = rng.normal(loc=0.0, scale=1.0, size=n_stream)
        z = standardize_stream(fresh, reference_mean=reference_mean, reference_std=reference_std)
        events = detect_changepoints(z, drift=drift, threshold=threshold)
        alarmed += int(bool(events))

    return {"target_fpr": target_fpr, "threshold": threshold, "n_trials": n_trials,
            "empirical_fpr": alarmed / n_trials, "alarmed_trials": alarmed,
            "ci95": significance._ci_binomial(alarmed, n_trials)}


def evaluate_detection_delay_and_false_alarm_rate_synthetic(
        n_points: int = 200, injection_index: int = 100, shift_sigma: float = 3.0,
        n_trials: int = 100, drift: float = 0.5, target_fpr: float = 0.01,
        seed: int = 42) -> dict:
    """Calibrates a threshold on a purely nominal reference, then over
    `n_trials`: (a) injects a known mean shift of `shift_sigma` standard
    deviations at `injection_index` and records samples-to-first-alarm
    at or after that index (detection delay; `None` on a miss), and (b)
    separately runs a pure-nominal stream and records whether it alarms
    at all (a false alarm)."""
    if injection_index < 1 or injection_index >= n_points:
        raise QualityDriftError("injection_index must be within [1, n_points)")
    if n_trials < 1:
        raise QualityDriftError("n_trials must be at least 1")

    rng = np.random.default_rng(seed)
    # A long reference (1000 points) avoids the finite-sample-noise
    # miscalibration `calibrate_threshold`'s own docstring documents.
    reference = rng.normal(size=1000)
    reference_mean, reference_std = float(np.mean(reference)), float(np.std(reference))
    threshold = calibrate_threshold(reference, drift=drift, target_fpr=target_fpr,
                                    stream_length=n_points, seed=seed)

    delays: list[int] = []
    misses = 0
    false_alarms = 0
    for trial in range(n_trials):
        stream = rng.normal(size=n_points)
        stream[injection_index:] += shift_sigma * reference_std
        z = standardize_stream(stream, reference_mean=reference_mean, reference_std=reference_std)
        events = detect_changepoints(z, drift=drift, threshold=threshold)
        after = [e for e in events if e.index >= injection_index]
        if after:
            delays.append(after[0].index - injection_index)
        else:
            misses += 1

        nominal_stream = rng.normal(size=n_points)
        z_nominal = standardize_stream(nominal_stream, reference_mean=reference_mean,
                                       reference_std=reference_std)
        false_alarms += int(bool(detect_changepoints(z_nominal, drift=drift, threshold=threshold)))

    return {
        "n_trials": n_trials, "injection_index": injection_index, "shift_sigma": shift_sigma,
        "detections": len(delays), "misses": misses,
        "mean_detection_delay": (sum(delays) / len(delays)) if delays else None,
        "median_detection_delay": (sorted(delays)[len(delays) // 2] if delays else None),
        "false_alarm_rate": false_alarms / n_trials,
        "false_alarm_ci95": significance._ci_binomial(false_alarms, n_trials),
    }


__all__ = [
    "evaluate_calibration_achieves_target_fpr_synthetic",
    "evaluate_detection_delay_and_false_alarm_rate_synthetic",
]
