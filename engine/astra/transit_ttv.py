"""Transit candidate search, limb-darkened transit modelling, and TTV vetting.

Shaped like `moving_objects.py`: a standalone, opt-in research module, not a
`SurveyConnector` -- transit vetting operates on a light curve's real time
series (already fetched via `tess_pixels.py`'s TESS TPF path or a
`lightkurve`-based Kepler/K2 pull), not a cone search. None of this is wired
into `evidence.WEIGHTS`/`scoring.combine()`/`rpc.py` -- it lands as visible,
opt-in candidate-review evidence only, the same restraint every other module
in this file's family already applies.

Per-transit timing (TTV/O-C) and the vetting heuristics
(`per_transit_midpoints`, `ttv_o_minus_c`, `vet_candidate`) live in
`transit_vetting.py`, split out purely to keep each file under this
project's 500-line guideline -- mirroring the `stellar_manifold.py` /
`stellar_manifold_eval.py` split, not an independent module.

Two scope decisions worth stating up front, the same "honest limitation, not
a glossed-over gap" discipline `moving_objects.py`'s Gauss-method docstring
uses:

1. `limb_darkened_transit_model` is a direct numerical integration over a
   fixed stellar-disk grid (a quadratic limb-darkening law integrated point
   by point), not the closed-form elliptic-integral solution of Mandel &
   Agol (2002, ApJ 580, L171). It is exact in the grid-resolution limit and
   avoids adding a compiled transit-modelling dependency (e.g. `batman`)
   purely to compute the same physical quantity this codebase can already
   compute directly with `numpy`/`scipy`, both already core dependencies --
   the same "avoid a new dependency when the real answer is directly
   computable" choice `[KNOWN] Gaps are modelled` made for the neural-ODE
   solver instead of `torchdiffeq`. Orbits are circular (e=0) only; a
   caller needing eccentric-orbit transit timing (which changes duration
   and, for TTV, the O-C shape itself) is out of scope here. Primary
   transits only -- occultation (secondary-eclipse) dimming is not modelled
   by this function; `vet_candidate` searches for a secondary eclipse as a
   false-positive signature instead, which does not require its own flux
   model, only a depth comparison at the predicted secondary phase.
2. `ttv_o_minus_c` is observed-minus-calculated timing-residual analysis
   against a linear ephemeris, not a full N-body dynamical TTV integration
   (e.g. the predicted signal from a specific resonant perturber). That
   remains open -- the same "mechanism built, full multi-body physics
   deferred" scope statement `moving_objects.py`'s two-body-only
   `two_body_propagate` already carries for orbit propagation.

`vet_candidate`'s three checks (odd/even depth mismatch, secondary-eclipse
search, transit-shape flatness) are standard, citable heuristics from the
transiting-planet vetting literature (e.g. the Kepler Robovetter, Thompson
et al. 2018, ApJS 235, 38) -- bounded statistical/geometric checks, NOT
pixel-level centroid vetting, which needs source-position-resolved
photometry (`tess_psf.py`-level PSF fitting) and is explicitly out of scope
here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

MIN_POINTS = 20


class TransitTTVError(ValueError):
    """A transit search, model fit, or TTV/vetting computation could not be completed."""


def _finite_arrays(time, value, value_err) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    value_err = np.asarray(value_err, dtype=np.float64)
    if not (len(time) == len(value) == len(value_err)):
        raise TransitTTVError("time, value, and value_err must be the same length")
    finite = np.isfinite(time) & np.isfinite(value) & np.isfinite(value_err) & (value_err > 0)
    return time[finite], value[finite], value_err[finite]


# ---------------------------------------------------------------------------
# Box Least Squares transit-candidate search (astropy.timeseries, no new dep).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BLSResult:
    period_days: float
    t0: float
    duration_days: float
    depth: float
    depth_err: float
    snr: float


def bls_search(time, value, value_err, *, period_min_days: float = 0.5,
               period_max_days: float = 20.0, duration_grid_days: list[float] | None = None,
               n_periods: int = 5000) -> BLSResult:
    """Search for the strongest periodic box-shaped dip.

    Uses an explicit, deterministic linear period grid rather than
    `BoxLeastSquares.autoperiod`'s frequency-oversampling optimisation -- a
    stated simplification that keeps this function's behaviour easy to
    reason about and test; a caller doing a real large-scale search should
    tune `n_periods`/`duration_grid_days` for their own cadence.
    """
    from astropy.timeseries import BoxLeastSquares

    time, value, value_err = _finite_arrays(time, value, value_err)
    if len(time) < MIN_POINTS:
        raise TransitTTVError(f"need at least {MIN_POINTS} finite points for a BLS search, got {len(time)}")
    if period_min_days <= 0 or period_max_days <= period_min_days:
        raise TransitTTVError("period_min_days must be positive and less than period_max_days")
    span = float(time.max() - time.min())
    if period_max_days > span:
        raise TransitTTVError(
            f"period_max_days ({period_max_days}) exceeds the observed baseline "
            f"({span:.3f} days); cannot search a period longer than the data spans")

    durations = np.asarray(duration_grid_days, dtype=np.float64) if duration_grid_days is not None \
        else np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    if np.any(durations <= 0) or np.any(durations >= period_min_days):
        raise TransitTTVError("every candidate duration must be positive and less than period_min_days")

    periods = np.linspace(period_min_days, period_max_days, int(n_periods))
    model = BoxLeastSquares(time, value, dy=value_err)
    result = model.power(periods, durations, objective="snr")
    idx = int(np.argmax(result.power))
    return BLSResult(
        period_days=float(result.period[idx]), t0=float(result.transit_time[idx]),
        duration_days=float(result.duration[idx]), depth=float(result.depth[idx]),
        depth_err=float(result.depth_err[idx]), snr=float(result.power[idx]),
    )


# ---------------------------------------------------------------------------
# Numerically disk-integrated quadratic limb-darkened transit model.
#
# The stellar disk is decomposed into `n_rings` concentric annuli (fixed
# boundaries, independent of any fit parameter), each assigned the limb
# darkening intensity at its midpoint radius. The area of each annulus
# blocked by the planet at a given separation is the analytic lens-shaped
# circle-circle intersection area (both circles are centered on the star's
# own center: the ring boundary and the planet disk), not a point-count over
# a fixed Cartesian grid -- an earlier version of this module used exactly
# that grid-count approach and it produced a genuinely broken fit: the
# blocked point SET only changes when a continuous geometric parameter
# (t0/period/rp_rs/a_rs/inc_deg) moves far enough to cross a grid cell
# boundary, so `scipy.optimize.least_squares`'s finite-difference gradient
# was measured (this session) to be exactly zero for every one of those five
# parameters -- only u1/u2, which enter through a smooth formula rather than
# set membership, ever moved during a real fit. The ring/analytic-overlap
# formulation below is smooth (piecewise-analytic, continuous first
# derivative away from exact tangency) in every parameter, confirmed this
# session to actually converge t0/period/rp_rs/a_rs/inc_deg to their
# injected values on synthetic data, not just u1/u2.
# ---------------------------------------------------------------------------

DEFAULT_N_RINGS = 200


def quadratic_limb_darkening_intensity(mu: np.ndarray, u1: float, u2: float) -> np.ndarray:
    """I(mu)/I(1) for the standard quadratic law (Kopal 1950), clipped at
    zero: a coefficient pair that is physically implausible right at the
    limb (mu -> 0) must not be allowed to go negative -- intensity is never
    negative -- rather than rejecting the whole evaluation. Full physical
    validity of (u1, u2) (u1 + u2 < 1, u1 > -1, ...) is instead checked once
    on a caller-supplied starting point, in `fit_transit_model`, not on every
    intermediate value an optimizer's trust-region step may transiently
    pass through here."""
    return np.clip(1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2, 0.0, None)


def _limb_darkening_rings(n_rings: int, u1: float, u2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_rings < 4:
        raise TransitTTVError("n_rings must be at least 4")
    edges = np.linspace(0.0, 1.0, n_rings + 1)
    r_in, r_out = edges[:-1], edges[1:]
    mu_mid = np.sqrt(np.clip(1.0 - (0.5 * (r_in + r_out)) ** 2, 0.0, 1.0))
    intensity = quadratic_limb_darkening_intensity(mu_mid, u1, u2)
    ring_area = np.pi * (r_out ** 2 - r_in ** 2)
    return r_in, r_out, intensity, ring_area


def _disk_overlap_area(disk_radius: np.ndarray, planet_radius: float, separation: np.ndarray) -> np.ndarray:
    """Analytic area of intersection between a disk of radius `disk_radius`
    centered at the origin and a disk of radius `planet_radius` centered a
    (broadcastable) `separation` away -- the standard two-circle lens-area
    formula."""
    disk_radius, separation = np.broadcast_arrays(np.asarray(disk_radius, dtype=np.float64),
                                                   np.asarray(separation, dtype=np.float64))
    area = np.zeros_like(disk_radius)
    fully_enclosed = separation <= np.abs(disk_radius - planet_radius)
    area[fully_enclosed] = np.pi * np.minimum(disk_radius, planet_radius)[fully_enclosed] ** 2
    partial = (~fully_enclosed) & (separation < disk_radius + planet_radius) & (separation > 0)
    if np.any(partial):
        R, d = disk_radius[partial], separation[partial]
        d1 = (d ** 2 - planet_radius ** 2 + R ** 2) / (2.0 * d)
        d2 = d - d1
        d1_clamped = np.clip(d1 / R, -1.0, 1.0)
        d2_clamped = np.clip(d2 / planet_radius, -1.0, 1.0)
        term1 = R ** 2 * np.arccos(d1_clamped) - d1 * np.sqrt(np.clip(R ** 2 - d1 ** 2, 0.0, None))
        term2 = planet_radius ** 2 * np.arccos(d2_clamped) - d2 * np.sqrt(np.clip(planet_radius ** 2 - d2 ** 2, 0.0, None))
        area[partial] = term1 + term2
    return area


def limb_darkened_transit_model(time, t0: float, period_days: float, rp_rs: float,
                                a_rs: float, inc_deg: float, u1: float, u2: float, *,
                                n_rings: int = DEFAULT_N_RINGS) -> np.ndarray:
    """Normalised (baseline = 1.0) flux for a circular-orbit, quadratic
    limb-darkened transit. Cost is O(n_times * n_rings), the ring
    decomposition being far cheaper than the earlier per-time-point 2-D
    grid it replaced."""
    time = np.asarray(time, dtype=np.float64)
    if period_days <= 0:
        raise TransitTTVError("period_days must be positive")
    if not 0 < rp_rs <= 1:
        raise TransitTTVError("rp_rs must be in (0, 1]")
    if a_rs <= 1:
        raise TransitTTVError("a_rs must exceed 1 (the planet must orbit outside the stellar surface)")
    if not 0 <= inc_deg <= 90:
        raise TransitTTVError("inc_deg must be in [0, 90]")
    if not math.isfinite(u1) or not math.isfinite(u2):
        raise TransitTTVError("u1/u2 must be finite")

    r_in, r_out, intensity, ring_area = _limb_darkening_rings(n_rings, u1, u2)
    total_flux = float(np.sum(intensity * ring_area))
    if total_flux <= 0:
        raise TransitTTVError("limb-darkening coefficients produce non-positive total stellar flux")

    phase = 2.0 * np.pi * (time - t0) / period_days
    inc = math.radians(inc_deg)
    # Sky-projected planet-center offset for a circular orbit (Winn 2010,
    # "Transits and Occultations", eq. 3-4): |offset| = a_rs * sqrt(sin^2 +
    # cos(i)^2 cos^2), with the transit epoch at phase = 0.
    px = a_rs * np.sin(phase)
    py = -a_rs * np.cos(phase) * math.cos(inc)
    separation = np.sqrt(px ** 2 + py ** 2)
    in_front = np.cos(phase) >= 0.0  # planet between observer and star

    flux = np.ones_like(time)
    idx = np.nonzero(in_front)[0]
    if idx.size:
        # (n_rings, n_selected_times) broadcast of the outer/inner overlap.
        outer = _disk_overlap_area(r_out[:, None], rp_rs, separation[None, idx])
        inner = _disk_overlap_area(r_in[:, None], rp_rs, separation[None, idx])
        blocked_flux = np.sum(intensity[:, None] * (outer - inner), axis=0)
        flux[idx] = 1.0 - blocked_flux / total_flux
    return flux


# ---------------------------------------------------------------------------
# Least-squares refinement of the transit model against real data.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransitFit:
    t0: float
    period_days: float
    rp_rs: float
    a_rs: float
    inc_deg: float
    u1: float
    u2: float
    residual_rms: float
    n_evaluations: int


_FIT_PARAM_ORDER = ("t0", "period_days", "rp_rs", "a_rs", "inc_deg", "u1", "u2")
_FIT_BOUNDS = {
    "t0": (-np.inf, np.inf), "period_days": (1e-3, np.inf), "rp_rs": (1e-4, 1.0),
    "a_rs": (1.0 + 1e-6, np.inf),
    # Edge-on (inc_deg == 90) is a physically real, common case (e.g. many
    # hot Jupiters); scipy's `trf` requires a strictly-interior x0, so the
    # box's own upper edge is nudged past 90 rather than excluding it.
    "inc_deg": (0.0, 90.0 + 1e-6), "u1": (0.0, 1.0), "u2": (-1.0, 1.0),
}


def fit_transit_model(time, value, value_err, initial_guess: dict[str, float], *,
                      n_rings: int = DEFAULT_N_RINGS) -> TransitFit:
    """Least-squares refinement of `limb_darkened_transit_model`'s seven
    parameters. Accepts a caller-supplied `initial_guess` for every
    parameter -- this module does not include a full initial-parameter
    estimator (e.g. deriving a_rs/inc from a BLS duration); a caller
    typically seeds `t0`/`period_days` from `bls_search` and supplies a
    plausible `rp_rs`/`a_rs`/`inc_deg`/`u1`/`u2` from prior knowledge."""
    time, value, value_err = _finite_arrays(time, value, value_err)
    if len(time) < MIN_POINTS:
        raise TransitTTVError(f"need at least {MIN_POINTS} finite points to fit a transit model, got {len(time)}")
    missing = [name for name in _FIT_PARAM_ORDER if name not in initial_guess]
    if missing:
        raise TransitTTVError(f"initial_guess is missing required parameters: {missing}")

    x0 = np.array([float(initial_guess[name]) for name in _FIT_PARAM_ORDER])
    lower = np.array([_FIT_BOUNDS[name][0] for name in _FIT_PARAM_ORDER])
    upper = np.array([_FIT_BOUNDS[name][1] for name in _FIT_PARAM_ORDER])
    if np.any(x0 <= lower) or np.any(x0 >= upper):
        raise TransitTTVError("initial_guess falls outside the physically valid parameter bounds")
    guess_u1, guess_u2 = float(initial_guess["u1"]), float(initial_guess["u2"])
    if guess_u1 + guess_u2 >= 1.0 or guess_u1 - guess_u2 <= -1.0:
        raise TransitTTVError(
            "initial_guess u1/u2 fall outside the physically valid quadratic "
            "limb-darkening triangle (u1 + u2 < 1, u1 - u2 > -1)")

    def residuals(params: np.ndarray) -> np.ndarray:
        kwargs = dict(zip(_FIT_PARAM_ORDER, params))
        model = limb_darkened_transit_model(time, n_rings=n_rings, **kwargs)
        return (value - model) / value_err

    result = least_squares(residuals, x0, bounds=(lower, upper), method="trf")
    if not result.success:
        raise TransitTTVError(f"transit model fit did not converge: {result.message}")

    fitted = dict(zip(_FIT_PARAM_ORDER, (float(x) for x in result.x)))
    rms = float(np.sqrt(np.mean(result.fun ** 2)))
    return TransitFit(**fitted, residual_rms=rms, n_evaluations=int(result.nfev))


def estimate_duration_days(fit: TransitFit) -> float:
    """Analytic total transit duration (T14) from a fitted geometry
    (standard circular-orbit formula, e.g. Winn 2010 eq. 14)."""
    b = fit.a_rs * math.cos(math.radians(fit.inc_deg))
    inside = (1.0 + fit.rp_rs) ** 2 - b ** 2
    if inside <= 0:
        raise TransitTTVError("fitted geometry implies no transit occurs (impact parameter too large)")
    sin_inc = math.sin(math.radians(fit.inc_deg))
    if sin_inc <= 0:
        raise TransitTTVError("inc_deg must be > 0 to compute a transit duration")
    argument = max(-1.0, min(1.0, (math.sqrt(inside) / fit.a_rs) / sin_inc))
    return (fit.period_days / math.pi) * math.asin(argument)


__all__ = [
    "TransitTTVError", "BLSResult", "bls_search",
    "quadratic_limb_darkening_intensity", "limb_darkened_transit_model",
    "TransitFit", "fit_transit_model", "estimate_duration_days",
    "MIN_POINTS", "DEFAULT_N_RINGS",
]
