"""Per-transit timing (TTV/O-C) and false-positive vetting for `transit_ttv.py`.

Split out purely to keep each file under this project's 500-line guideline,
mirroring the `stellar_manifold.py` / `stellar_manifold_eval.py` split --
this is the second half of the same module family, not an independent
research direction. See `transit_ttv.py`'s module docstring for the scope
notes (circular orbits only, O-C timing analysis rather than full N-body
TTV, no pixel-level centroid vetting) that apply to everything below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .transit_ttv import (
    TransitFit, TransitTTVError, _finite_arrays, estimate_duration_days,
    limb_darkened_transit_model,
)


def _in_transit_mask(time: np.ndarray, fit: TransitFit, duration_days: float) -> np.ndarray:
    first_epoch = math.ceil((time.min() - fit.t0) / fit.period_days)
    last_epoch = math.floor((time.max() - fit.t0) / fit.period_days)
    mask = np.zeros(len(time), dtype=bool)
    for epoch in range(first_epoch, last_epoch + 1):
        predicted = fit.t0 + epoch * fit.period_days
        mask |= np.abs(time - predicted) <= duration_days / 2.0
    return mask


# ---------------------------------------------------------------------------
# Per-transit mid-time fitting and observed-minus-calculated (O-C) TTV.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerTransitFit:
    epoch: int
    midpoint: float
    n_points: int


def per_transit_midpoints(time, value, value_err, fit: TransitFit, *,
                          duration_days: float | None = None, window_factor: float = 1.5,
                          min_points_per_transit: int = 5,
                          ) -> tuple[list[PerTransitFit], list[int]]:
    """Fit each transit's own mid-time (all other parameters held fixed at
    `fit`'s values), searching a window of `window_factor * duration_days`
    on either side of the linear-ephemeris prediction. An epoch without
    enough points in its window is skipped, not degraded with a fabricated
    midpoint -- the same `tensors.resample` MIN_POINTS-skip convention this
    codebase uses elsewhere; skipped epochs are returned separately so a
    caller can see what was excluded and why."""
    time, value, value_err = _finite_arrays(time, value, value_err)
    duration_days = duration_days if duration_days is not None else estimate_duration_days(fit)
    if duration_days <= 0:
        raise TransitTTVError("duration_days must be positive")
    half_window = window_factor * duration_days

    first_epoch = math.ceil((time.min() - fit.t0) / fit.period_days)
    last_epoch = math.floor((time.max() - fit.t0) / fit.period_days)
    fits: list[PerTransitFit] = []
    skipped: list[int] = []
    for epoch in range(first_epoch, last_epoch + 1):
        predicted = fit.t0 + epoch * fit.period_days
        mask = np.abs(time - predicted) <= half_window
        if int(mask.sum()) < min_points_per_transit:
            skipped.append(epoch)
            continue
        local_time, local_value, local_err = time[mask], value[mask], value_err[mask]

        def residuals(params: np.ndarray, lt=local_time, lv=local_value, le=local_err) -> np.ndarray:
            model = limb_darkened_transit_model(
                lt, params[0], fit.period_days, fit.rp_rs, fit.a_rs, fit.inc_deg, fit.u1, fit.u2)
            return (lv - model) / le

        result = least_squares(residuals, x0=[predicted],
                               bounds=([predicted - half_window], [predicted + half_window]))
        if not result.success:
            skipped.append(epoch)
            continue
        fits.append(PerTransitFit(epoch=epoch, midpoint=float(result.x[0]), n_points=int(mask.sum())))
    return fits, skipped


@dataclass(frozen=True)
class TTVResult:
    epochs: list[int]
    residuals_minutes: list[float]
    rms_minutes: float
    amplitude_minutes: float
    n_skipped: int


def ttv_o_minus_c(transit_fits: list[PerTransitFit], *, period_days: float, t0: float,
                  n_skipped: int = 0) -> TTVResult:
    """Observed-minus-calculated residuals of measured mid-times against a
    linear ephemeris (`t0 + epoch * period_days`)."""
    if not transit_fits:
        raise TransitTTVError("no transit midpoints were supplied to compute O-C residuals from")
    epochs = [tf.epoch for tf in transit_fits]
    observed = np.array([tf.midpoint for tf in transit_fits])
    calculated = t0 + np.array(epochs, dtype=np.float64) * period_days
    residual_minutes = (observed - calculated) * 24.0 * 60.0
    rms = float(np.sqrt(np.mean(residual_minutes ** 2)))
    amplitude = float((residual_minutes.max() - residual_minutes.min()) / 2.0)
    return TTVResult(epochs=epochs, residuals_minutes=residual_minutes.tolist(),
                     rms_minutes=rms, amplitude_minutes=amplitude, n_skipped=n_skipped)


# ---------------------------------------------------------------------------
# Bounded false-positive vetting heuristics.
#
# Standard, citable checks used by transiting-planet vetting pipelines
# (e.g. the Kepler Robovetter, Thompson et al. 2018, ApJS 235, 38) --
# NOT pixel-level centroid vetting, which needs source-position-resolved
# photometry (`tess_psf.py`-level work) and is explicitly out of scope here.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VettingResult:
    odd_even_mismatch_sigma: float
    odd_even_flagged: bool
    secondary_eclipse_depth: float
    secondary_eclipse_flagged: bool
    shape_flat_fraction: float
    v_shape_flagged: bool


def _standard_error(values: np.ndarray) -> float:
    if len(values) > 1:
        return float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return float(abs(values[0])) * 0.5 if len(values) else 0.0


def _in_transit_depths(time: np.ndarray, value: np.ndarray, fit: TransitFit,
                       duration_days: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-transit-epoch depth: out-of-transit baseline median minus the
    local median value inside each transit's window."""
    first_epoch = math.ceil((time.min() - fit.t0) / fit.period_days)
    last_epoch = math.floor((time.max() - fit.t0) / fit.period_days)
    baseline_mask = np.ones(len(time), dtype=bool)
    epochs, depths = [], []
    for epoch in range(first_epoch, last_epoch + 1):
        predicted = fit.t0 + epoch * fit.period_days
        in_mask = np.abs(time - predicted) <= duration_days / 2.0
        baseline_mask &= ~in_mask
        if int(in_mask.sum()) < 3:
            continue
        epochs.append(epoch)
        depths.append(float(np.median(value[in_mask])))
    baseline = float(np.median(value[baseline_mask])) if baseline_mask.any() else float(np.median(value))
    return np.array(epochs, dtype=np.int64), baseline - np.array(depths, dtype=np.float64)


def _secondary_eclipse_depth(time: np.ndarray, value: np.ndarray, fit: TransitFit,
                             duration_days: float) -> float:
    phase = (time - fit.t0) % fit.period_days
    distance_from_secondary = np.abs(phase - fit.period_days / 2.0)
    secondary_mask = distance_from_secondary <= duration_days / 2.0
    if int(secondary_mask.sum()) < 3 or int((~secondary_mask).sum()) == 0:
        return 0.0
    baseline = float(np.median(value[~secondary_mask]))
    return baseline - float(np.median(value[secondary_mask]))


def _shape_flat_fraction(time: np.ndarray, value: np.ndarray, fit: TransitFit,
                         duration_days: float) -> float:
    in_transit_mask = _in_transit_mask(time, fit, duration_days)
    if int(in_transit_mask.sum()) < 3 or int((~in_transit_mask).sum()) == 0:
        return 0.0
    baseline = float(np.median(value[~in_transit_mask]))
    in_values = value[in_transit_mask]
    depth = baseline - float(np.min(in_values))
    if depth <= 0:
        return 0.0
    return float(np.mean((baseline - in_values) >= 0.9 * depth))


def vet_candidate(time, value, value_err, fit: TransitFit, *, duration_days: float | None = None,
                  odd_even_sigma_threshold: float = 3.0,
                  secondary_eclipse_fraction_threshold: float = 0.5,
                  v_shape_flat_fraction_threshold: float = 0.2) -> VettingResult:
    """Odd/even depth mismatch, secondary-eclipse search, and transit-shape
    (flat-bottom vs. V-shaped) heuristics -- see the module docstring for
    the literature these follow and what they deliberately do not replace."""
    time, value, _ = _finite_arrays(time, value, value_err)
    duration_days = duration_days if duration_days is not None else estimate_duration_days(fit)
    if duration_days <= 0:
        raise TransitTTVError("duration_days must be positive")

    epochs, depths = _in_transit_depths(time, value, fit, duration_days)
    if len(epochs) < 2:
        raise TransitTTVError("need at least two observed transits to vet a candidate")
    primary_depth = float(np.mean(depths))

    odd_depths = depths[epochs % 2 != 0]
    even_depths = depths[epochs % 2 == 0]
    if len(odd_depths) and len(even_depths):
        combined_se = math.sqrt(_standard_error(odd_depths) ** 2 + _standard_error(even_depths) ** 2)
        odd_even_sigma = abs(float(np.mean(odd_depths)) - float(np.mean(even_depths))) / combined_se \
            if combined_se > 0 else 0.0
    else:
        odd_even_sigma = 0.0

    secondary_depth = _secondary_eclipse_depth(time, value, fit, duration_days)
    secondary_flagged = primary_depth > 0 and secondary_depth >= secondary_eclipse_fraction_threshold * primary_depth
    flat_fraction = _shape_flat_fraction(time, value, fit, duration_days)

    return VettingResult(
        odd_even_mismatch_sigma=odd_even_sigma,
        odd_even_flagged=odd_even_sigma >= odd_even_sigma_threshold,
        secondary_eclipse_depth=secondary_depth,
        secondary_eclipse_flagged=secondary_flagged,
        shape_flat_fraction=flat_fraction,
        v_shape_flagged=flat_fraction < v_shape_flat_fraction_threshold,
    )


__all__ = [
    "PerTransitFit", "per_transit_midpoints", "TTVResult", "ttv_o_minus_c",
    "VettingResult", "vet_candidate",
]
