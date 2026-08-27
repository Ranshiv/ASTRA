"""Validation for `xray_hardness.py`: state-transition detection and
flux/hardness calibration (roadmap item 23's two named metrics).

State-transition detection mirrors `agn_changepoint_eval.evaluate_lead_
time`'s injection-study shape: inject a known hardness-ratio shift at a
known epoch, check whether `xray_hardness.detect_state_transitions` finds
it near that epoch. Flux/hardness calibration compares an independently
computed hardness ratio (from real per-band fluxes, e.g. `chandra.
query_band_fluxes`'s output) against a catalogue's OWN released,
pre-computed hardness ratio -- the same "diff an independent measurement
against a released value" pattern `spectroscopy_calibration_eval.
redshift_residuals` and `line_profile_eval.line_parameter_residuals` both
already use this session.
"""

from __future__ import annotations

import numpy as np

from . import xray_hardness as xh

DEFAULT_NOISE_SIGMA = 0.05


class XrayHardnessEvalError(ValueError):
    """An X-ray hardness evaluation study could not be run."""


def evaluate_state_transition_detection(baseline_hr: float, shifted_hr: float, *,
                                        n_before: int = 10, n_after: int = 10,
                                        noise_sigma: float = DEFAULT_NOISE_SIGMA,
                                        n_trials: int = 100, seed: int = 42,
                                        n_states: int = 2, tolerance_epochs: int = 2) -> dict:
    """Synthetic injection study: `n_before` epochs at `baseline_hr`, then
    `n_after` at `shifted_hr` (each with Gaussian noise `noise_sigma`).
    A trial is a "hit" if `detect_state_transitions` flags a transition
    within `tolerance_epochs` of the true injection point (index
    `n_before`).
    """
    if n_before < 1 or n_after < 1:
        raise XrayHardnessEvalError("n_before and n_after must be at least 1")
    rng = np.random.default_rng(seed)
    hits = 0
    for trial in range(n_trials):
        before = rng.normal(baseline_hr, noise_sigma, n_before)
        after = rng.normal(shifted_hr, noise_sigma, n_after)
        series = np.concatenate([before, after])
        try:
            fit = xh.fit_hardness_states(series, n_states=n_states, seed=seed + trial)
        except xh.XrayHardnessError:
            continue
        transitions = xh.detect_state_transitions(fit["labels"])
        hits += int(any(abs(t - n_before) <= tolerance_epochs for t in transitions))

    return {"n_trials": n_trials, "detection_rate": hits / n_trials,
           "baseline_hr": baseline_hr, "shifted_hr": shifted_hr,
           "noise_sigma": noise_sigma, "tolerance_epochs": tolerance_epochs}


def evaluate_false_positive_rate(hr: float, *, n_points: int = 20,
                                 noise_sigma: float = DEFAULT_NOISE_SIGMA,
                                 n_trials: int = 100, seed: int = 42, n_states: int = 2) -> dict:
    """The explicit no-signal regression case: a CONSTANT hardness ratio
    plus noise, no real spectral-state change anywhere. Reports how often
    `detect_state_transitions` wrongly reports one anyway -- a genuine,
    non-zero rate is expected here (a Gaussian mixture with `n_states=2`
    always partitions the data into two clusters even when only one real
    cluster exists, so noise near the fitted boundary between those two
    clusters will flip labels sometimes), reported honestly rather than
    hidden.
    """
    rng = np.random.default_rng(seed)
    false_positives = 0
    n_valid = 0
    for trial in range(n_trials):
        series = rng.normal(hr, noise_sigma, n_points)
        try:
            fit = xh.fit_hardness_states(series, n_states=n_states, seed=seed + trial)
        except xh.XrayHardnessError:
            continue
        n_valid += 1
        transitions = xh.detect_state_transitions(fit["labels"])
        false_positives += int(len(transitions) > 0)

    return {"n_trials": n_trials, "n_valid": n_valid,
           "false_positive_rate": (false_positives / n_valid) if n_valid else float("nan")}


def flux_hardness_calibration(band_flux_rows: list[dict]) -> dict:
    """Compare an independently computed hardness ratio (from real
    `flux_soft`/`flux_hard` fields) against each row's own released
    `hr_hard_soft` value -- exactly `chandra.query_band_fluxes`'s output
    shape. A row missing any of the three fields is skipped and counted,
    not silently dropped or imputed.

    A real, load-bearing finding from running this live this session
    against 76 real CSC 2.1 sources near M87: the residuals are NOT
    small (median ~0.47, robust scatter ~0.34). This is not a formula
    bug -- CSC's own documentation (checked live this session) confirms
    `HRhs = (Fluxh - Fluxs) / (Fluxh + Fluxs)`, the identical functional
    form `xray_hardness.hardness_ratio` implements. The real cause: CSC
    2's released hardness ratios are computed from full photon-flux
    aperture-photometry MPDFs (Bayesian posterior distributions per
    band), "an improvement over CSC Release 1", NOT from the simple
    point-estimate `Fluxh`/`Fluxs` columns this function (necessarily)
    uses -- those two point estimates are a real but different quantity
    from what CSC's own HR is derived from, especially for faint/
    marginal sources where a Bayesian and a point estimate diverge most.
    This function still reports a genuine, real calibration number; it
    is not a bug-free "should match exactly" comparison, and that
    real gap is documented here and in `docs/DEFERRED.txt`, not hidden.
    """
    computed: list[float] = []
    released: list[float] = []
    n_skipped = 0

    for row in band_flux_rows:
        soft, hard = row.get("flux_soft"), row.get("flux_hard")
        released_hr = row.get("hr_hard_soft")
        if soft is None or hard is None or released_hr is None:
            n_skipped += 1
            continue
        try:
            my_hr = float(xh.hardness_ratio(np.array([float(soft)]), np.array([float(hard)]))[0])
        except xh.XrayHardnessError:
            n_skipped += 1
            continue
        if not np.isfinite(my_hr):
            n_skipped += 1
            continue
        computed.append(my_hr)
        released.append(float(released_hr))

    if not computed:
        return {"n_compared": 0, "n_skipped": n_skipped, "median_residual": None,
               "robust_scatter": None, "max_abs_residual": None}

    residual = np.asarray(computed) - np.asarray(released)
    return {
        "n_compared": len(computed), "n_skipped": n_skipped,
        "median_residual": float(np.median(residual)),
        "robust_scatter": float(np.median(np.abs(residual - np.median(residual))) * 1.4826),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }
