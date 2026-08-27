"""Eclipsing-binary light-curve geometry and least-squares fitting.

Shaped like `transit_ttv.py`, which this module directly builds on: reuses
`transit_ttv._limb_darkening_rings`/`_disk_overlap_area` (the analytic
concentric-ring, two-circle-lens-area integration validated in that module)
rather than reimplementing eclipse geometry from scratch. Two finite,
mutually eclipsing bodies is a genuine generalisation of `transit_ttv.py`'s
one-finite-body-transits-a-star case, not a copy: both bodies contribute
their own limb-darkened flux, and which body is in front swaps every half
period (primary vs. secondary eclipse), which `transit_ttv.py`'s public
`limb_darkened_transit_model` cannot represent -- its own `rp_rs <= 1` input
validation assumes the occulter is always smaller than the occulted star,
which is not generally true for two comparably-sized stars. This module
therefore calls the private ring/overlap helpers directly instead of the
public wrapper, and implements its own primary/secondary bookkeeping and
per-body luminosity weighting.

Parameterisation follows the standard EB-literature convention: `r1_a` =
R1/a and `r2_a` = R2/a (each body's radius in units of the orbital
semi-major axis), rather than `transit_ttv.py`'s "one body's radius in units
of the OTHER body's radius" convention -- this avoids ever needing an
occulter/occulted radius ratio greater than 1 to be treated as invalid, the
concrete reason the public wrapper could not be reused directly.

Scope, stated explicitly, the same "honest limitation" discipline every
module in this family uses:
- Circular orbits only (e=0), inherited from `transit_ttv.py`.
- No reflection effect (mutual irradiation brightening the near hemispheres)
  and no ellipsoidal (tidal-distortion) variation -- both real light-curve-
  shaping physics for close binaries, deliberately not modelled here.
- The secondary-to-primary temperature ratio (`teff_ratio`) sets each body's
  relative luminosity via Stefan-Boltzmann scaling (L ~ R^2 * Teff^4); no
  absolute temperature is computed in THIS module -- that requires an
  external anchor and lives in `eclipsing_binary_dimensions.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .transit_ttv import DEFAULT_N_RINGS, _disk_overlap_area, _finite_arrays, _limb_darkening_rings

MIN_POINTS = 20


class EclipsingBinaryError(ValueError):
    """An eclipsing-binary model, fit, or depth computation could not be completed."""


def _validate_geometry(r1_a: float, r2_a: float, inc_deg: float, teff_ratio: float) -> None:
    if not 0 < r1_a < 1:
        raise EclipsingBinaryError("r1_a must be in (0, 1)")
    if not 0 < r2_a < 1:
        raise EclipsingBinaryError("r2_a must be in (0, 1)")
    if not 0 <= inc_deg <= 90:
        raise EclipsingBinaryError("inc_deg must be in [0, 90]")
    if not math.isfinite(teff_ratio) or teff_ratio <= 0:
        raise EclipsingBinaryError("teff_ratio must be a positive finite number")


def eclipsing_binary_model(time, t0: float, period_days: float, r1_a: float, r2_a: float,
                           inc_deg: float, u1_1: float, u2_1: float, u1_2: float, u2_2: float,
                           teff_ratio: float, *, n_rings: int = DEFAULT_N_RINGS) -> np.ndarray:
    """Normalised (baseline = 1.0) joint flux of two limb-darkened disks in
    mutual circular-orbit eclipse.

    By convention, `t0` marks the eclipse where body 2 passes in front of
    body 1 (`cos(phase) >= 0`); the eclipse where body 1 passes in front of
    body 2 falls at `t0 + period_days / 2`. A caller fitting real data
    typically seeds `t0` from the deeper of the two observed eclipses, which
    is not necessarily "body 1" in any physical sense -- the labels are
    just this function's own bookkeeping convention.
    """
    time = np.asarray(time, dtype=np.float64)
    if period_days <= 0:
        raise EclipsingBinaryError("period_days must be positive")
    _validate_geometry(r1_a, r2_a, inc_deg, teff_ratio)

    r1_in, r1_out, intensity1, ring_area1 = _limb_darkening_rings(n_rings, u1_1, u2_1)
    r2_in, r2_out, intensity2, ring_area2 = _limb_darkening_rings(n_rings, u1_2, u2_2)
    w1 = float(np.sum(intensity1 * ring_area1))
    w2 = float(np.sum(intensity2 * ring_area2))
    if w1 <= 0 or w2 <= 0:
        raise EclipsingBinaryError("limb-darkening coefficients produce non-positive flux for one star")

    phase = 2.0 * np.pi * (time - t0) / period_days
    inc = math.radians(inc_deg)
    # Sky-projected center separation for a circular orbit, in units of the
    # semi-major axis a (same Winn 2010 formula `transit_ttv.py` uses, here
    # with a_rs = 1 since both radii are already expressed in units of a).
    separation = np.sqrt(np.sin(phase) ** 2 + (math.cos(inc) * np.cos(phase)) ** 2)
    body2_in_front = np.cos(phase) >= 0.0

    flux1 = np.ones_like(time)  # body 1's own light, dimmed when body 2 is in front
    idx1 = np.nonzero(body2_in_front)[0]
    if idx1.size:
        outer = _disk_overlap_area(r1_out[:, None] * r1_a, r2_a, separation[None, idx1])
        inner = _disk_overlap_area(r1_in[:, None] * r1_a, r2_a, separation[None, idx1])
        blocked = np.sum(intensity1[:, None] * (outer - inner), axis=0)
        flux1[idx1] = 1.0 - blocked / (w1 * r1_a ** 2)

    flux2 = np.ones_like(time)  # body 2's own light, dimmed when body 1 is in front
    idx2 = np.nonzero(~body2_in_front)[0]
    if idx2.size:
        outer = _disk_overlap_area(r2_out[:, None] * r2_a, r1_a, separation[None, idx2])
        inner = _disk_overlap_area(r2_in[:, None] * r2_a, r1_a, separation[None, idx2])
        blocked = np.sum(intensity2[:, None] * (outer - inner), axis=0)
        flux2[idx2] = 1.0 - blocked / (w2 * r2_a ** 2)

    luminosity1 = w1 * r1_a ** 2
    luminosity2 = (teff_ratio ** 4) * w2 * r2_a ** 2
    total_luminosity = luminosity1 + luminosity2
    return (luminosity1 * flux1 + luminosity2 * flux2) / total_luminosity


# ---------------------------------------------------------------------------
# Least-squares refinement against real data.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EclipsingBinaryFit:
    t0: float
    period_days: float
    r1_a: float
    r2_a: float
    inc_deg: float
    u1_1: float
    u2_1: float
    u1_2: float
    u2_2: float
    teff_ratio: float
    residual_rms: float
    n_evaluations: int


_FIT_PARAM_ORDER = ("t0", "period_days", "r1_a", "r2_a", "inc_deg",
                    "u1_1", "u2_1", "u1_2", "u2_2", "teff_ratio")
_FIT_BOUNDS = {
    "t0": (-np.inf, np.inf), "period_days": (1e-3, np.inf),
    "r1_a": (1e-4, 0.5 - 1e-6), "r2_a": (1e-4, 0.5 - 1e-6),
    "inc_deg": (0.0, 90.0 + 1e-6), "u1_1": (0.0, 1.0), "u2_1": (-1.0, 1.0),
    "u1_2": (0.0, 1.0), "u2_2": (-1.0, 1.0), "teff_ratio": (1e-3, 1e3),
}


def _validate_limb_darkening_guess(u1: float, u2: float, label: str) -> None:
    if u1 + u2 >= 1.0 or u1 - u2 <= -1.0:
        raise EclipsingBinaryError(
            f"initial_guess {label} fall outside the physically valid quadratic "
            "limb-darkening triangle (u1 + u2 < 1, u1 - u2 > -1)")


def fit_eclipsing_binary(time, value, value_err, initial_guess: dict[str, float], *,
                         n_rings: int = DEFAULT_N_RINGS) -> EclipsingBinaryFit:
    """Least-squares refinement of `eclipsing_binary_model`'s ten parameters.

    Accepts a caller-supplied `initial_guess` for every parameter, the same
    "no built-in initial-parameter estimator" restraint
    `transit_ttv.fit_transit_model` states; a caller typically seeds
    `t0`/`period_days` from a periodogram on the real light curve and
    `r1_a`/`r2_a`/`inc_deg` from the two observed eclipse depths/durations.
    """
    time, value, value_err = _finite_arrays(time, value, value_err)
    if len(time) < MIN_POINTS:
        raise EclipsingBinaryError(f"need at least {MIN_POINTS} finite points to fit an EB model, got {len(time)}")
    missing = [name for name in _FIT_PARAM_ORDER if name not in initial_guess]
    if missing:
        raise EclipsingBinaryError(f"initial_guess is missing required parameters: {missing}")

    x0 = np.array([float(initial_guess[name]) for name in _FIT_PARAM_ORDER])
    lower = np.array([_FIT_BOUNDS[name][0] for name in _FIT_PARAM_ORDER])
    upper = np.array([_FIT_BOUNDS[name][1] for name in _FIT_PARAM_ORDER])
    if np.any(x0 <= lower) or np.any(x0 >= upper):
        raise EclipsingBinaryError("initial_guess falls outside the physically valid parameter bounds")
    if float(initial_guess["r1_a"]) + float(initial_guess["r2_a"]) >= 1.0:
        raise EclipsingBinaryError("initial_guess r1_a + r2_a must be < 1 (a detached, non-overlapping orbit)")
    _validate_limb_darkening_guess(float(initial_guess["u1_1"]), float(initial_guess["u2_1"]), "u1_1/u2_1")
    _validate_limb_darkening_guess(float(initial_guess["u1_2"]), float(initial_guess["u2_2"]), "u1_2/u2_2")

    def residuals(params: np.ndarray) -> np.ndarray:
        kwargs = dict(zip(_FIT_PARAM_ORDER, params))
        try:
            model = eclipsing_binary_model(time, n_rings=n_rings, **kwargs)
        except EclipsingBinaryError:
            # A transient trust-region step outside the physically valid
            # r1_a/r2_a/teff_ratio box (already enforced by `bounds` below,
            # but floating-point edges can still trip `_validate_geometry`)
            # is reported to the optimizer as a very poor fit rather than
            # crashing the whole `least_squares` call.
            return np.full(len(value), 1e6)
        return (value - model) / value_err

    result = least_squares(residuals, x0, bounds=(lower, upper), method="trf")
    if not result.success:
        raise EclipsingBinaryError(f"eclipsing-binary model fit did not converge: {result.message}")

    fitted = dict(zip(_FIT_PARAM_ORDER, (float(x) for x in result.x)))
    rms = float(np.sqrt(np.mean(result.fun ** 2)))
    return EclipsingBinaryFit(**fitted, residual_rms=rms, n_evaluations=int(result.nfev))


def primary_secondary_depths(fit: EclipsingBinaryFit, *, n_rings: int = DEFAULT_N_RINGS) -> tuple[float, float]:
    """Mid-eclipse depths (1 - flux) at the fitted primary (`t0`) and
    secondary (`t0 + period_days / 2`) eclipse epochs."""
    times = np.array([fit.t0, fit.t0 + fit.period_days / 2.0])
    flux = eclipsing_binary_model(times, fit.t0, fit.period_days, fit.r1_a, fit.r2_a, fit.inc_deg,
                                  fit.u1_1, fit.u2_1, fit.u1_2, fit.u2_2, fit.teff_ratio, n_rings=n_rings)
    return float(1.0 - flux[0]), float(1.0 - flux[1])


def temperature_ratio_from_depths(primary_depth: float, secondary_depth: float) -> float:
    """Standard eclipse-depth temperature-ratio approximation
    (Teff_secondary / Teff_primary ~ (secondary_depth / primary_depth)**0.25),
    assuming comparable limb darkening and negligible reflection -- a real
    simplifying approximation, not a full spectral-synthesis solution."""
    if primary_depth <= 0:
        raise EclipsingBinaryError("primary_depth must be positive to derive a temperature ratio")
    ratio = secondary_depth / primary_depth
    if ratio <= 0:
        raise EclipsingBinaryError("secondary_depth must be positive to derive a temperature ratio")
    return float(ratio ** 0.25)


__all__ = [
    "EclipsingBinaryError", "eclipsing_binary_model",
    "EclipsingBinaryFit", "fit_eclipsing_binary",
    "primary_secondary_depths", "temperature_ratio_from_depths",
    "MIN_POINTS",
]
