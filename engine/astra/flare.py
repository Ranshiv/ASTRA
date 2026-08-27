"""Stellar-flare template, detection, and least-squares fitting.

Shaped like `transit_ttv.py`: a standalone, opt-in research module, real
algorithm cited from the literature, validated against synthetic ground
truth, never wired into `evidence.WEIGHTS`/`scoring.combine()`/`rpc.py`.

`evaluate.py`'s synthetic `"flare"` injection kind (one of `ANOMALY_KINDS`)
is a crude instant-rise/pure-exponential-decay shape used only as a
generic anomaly-injection sanity check -- confirmed while planning this
module, not adequate for energy modelling (no polynomial rise phase, no
two-component decay) and NOT reused here. `davenport_flare_template`
instead implements the real, published analytic flare template (Davenport
et al. 2014, ApJ 797, 122): a piecewise 4th-order-polynomial rise plus a
two-exponential decay, the same template widely reproduced in flare-fitting
tools (e.g. the appaloosa/altaipony `aflare` implementation) -- the same
"use the published formula" convention `transit_ttv.py` follows for Winn
(2010) and `moving_objects.py` follows for the Gauss method. The template
has a real, small (~0.8%) discontinuity at t'=0 between its own published
rise and decay pieces (1.0 exactly from the rise side, 0.6890+0.3030=0.992
from the decay side) -- an artefact of the original published fit, not a
bug introduced here.

`equivalent_duration` (ED; Gershberg 1972, standard in Hawley et al.
2014/Davenport 2016) is the time-integral of fractional flux excess -- a
real, exact quantity computable directly from data once a local quiescent
baseline is estimated, needing no assumed physics. Converting ED into a
bolometric energy (which does need a physical assumption) lives in
`flare_energy.py`.

Two more scope notes, the same "honest limitation, not glossed over"
discipline every module in this family uses:
- `detect_flare_candidates` is DETECTION only (a bounded "N consecutive
  points above a sigma threshold" heuristic, e.g. Chang et al. 2015) --
  no false-positive vetting beyond the consecutive-point run-length
  requirement.
- `fit_flare_template` fits a SINGLE-peak flare; complex/multi-peak flare
  morphology is out of scope.

`relative_flux_excess` needs no absolute flux calibration for either
survey: TESS's `pdcsap_flux`/`sap_flux` (electrons/sec, no zero-point) and
ZTF's raw magnitudes both convert to a RELATIVE flux ratio
(`flux / quiescent_flux` or `10**(-0.4*(mag - quiescent_mag))`) without
ever needing an absolute calibration -- confirmed while planning this
module that neither survey connector in this codebase publishes one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .transit_ttv import _finite_arrays

MIN_POINTS = 20
SECONDS_PER_DAY = 86400.0

# Davenport et al. (2014, ApJ 797, 122), Table 1 -- rise (t' in [-1, 0]) and
# decay (t' >= 0) coefficients, where t' = (t - t_peak) / fwhm.
_RISE_COEFFS = (1.0, 1.941, -0.175, -2.246, -1.125)
_DECAY_A1, _DECAY_B1 = 0.6890, -1.600
_DECAY_A2, _DECAY_B2 = 0.3030, -0.2783


class FlareError(ValueError):
    """A flare template, detection, or fit computation could not be completed."""


def davenport_flare_template(t_prime) -> np.ndarray:
    """The published piecewise analytic flare template, in units of peak
    amplitude (baseline = 0)."""
    t_prime = np.asarray(t_prime, dtype=np.float64)
    rising = t_prime < 0.0
    result = np.zeros_like(t_prime)
    tr = t_prime[rising]
    result[rising] = sum(coeff * tr ** power for power, coeff in enumerate(_RISE_COEFFS))
    td = t_prime[~rising]
    result[~rising] = _DECAY_A1 * np.exp(_DECAY_B1 * td) + _DECAY_A2 * np.exp(_DECAY_B2 * td)
    return np.clip(result, 0.0, None)


def flare_model(time, t_peak: float, fwhm: float, amplitude: float) -> np.ndarray:
    """Additive relative-flux-excess model (baseline = 0) for a single
    Davenport-template flare, in the caller's own time units (matching
    `equivalent_duration`'s day convention elsewhere in this module)."""
    time = np.asarray(time, dtype=np.float64)
    if fwhm <= 0:
        raise FlareError("fwhm must be positive")
    if amplitude <= 0:
        raise FlareError("amplitude must be positive")
    return amplitude * davenport_flare_template((time - t_peak) / fwhm)


# ---------------------------------------------------------------------------
# Relative flux excess: the common representation both flux- and mag-kind
# light curves are converted into before detection/fitting/ED integration.
# ---------------------------------------------------------------------------

def relative_flux_excess(time, value, value_err, value_kind: str, *,
                         baseline_window_days: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """`(flux / quiescent_flux) - 1` for `value_kind="flux"`, or the
    equivalent computed from magnitudes for `value_kind="mag"` -- neither
    needs an absolute calibration, per the module docstring.

    The local quiescent baseline is a rolling median over
    `baseline_window_days` (excluding the point itself), a real, standard
    robust-baseline technique: a short flare occupying a small minority of
    points in a window wide enough to span many cadences does not
    meaningfully shift the median. Cost is O(n^2) (an explicit per-point
    window search) -- bounded and fine for a candidate-scale light curve,
    the same "not survey-scale batch" caveat `transit_ttv.py`'s ring
    integration states for itself.
    """
    if value_kind not in ("flux", "mag"):
        raise FlareError(f"value_kind must be 'flux' or 'mag', got {value_kind!r}")
    time, value, value_err = _finite_arrays(time, value, value_err)
    if len(time) < MIN_POINTS:
        raise FlareError(f"need at least {MIN_POINTS} finite points, got {len(time)}")
    if baseline_window_days <= 0:
        raise FlareError("baseline_window_days must be positive")

    baseline = np.empty_like(value)
    for i in range(len(time)):
        window = np.abs(time - time[i]) <= baseline_window_days / 2.0
        window[i] = False
        baseline[i] = np.median(value[window]) if window.any() else float(np.median(value))

    if value_kind == "mag":
        excess = 10.0 ** (-0.4 * (value - baseline)) - 1.0
        # d(excess)/d(mag) = -0.4*ln(10)*(excess + 1); baseline's own
        # uncertainty is assumed negligible next to a single point's error,
        # the same "many-point median dominates a single measurement's
        # error" assumption `transit_ttv.py` implicitly makes for its own
        # per-transit local fits.
        excess_err = 0.4 * math.log(10.0) * np.abs(excess + 1.0) * value_err
    else:
        if np.any(baseline <= 0):
            raise FlareError("non-positive flux baseline; cannot form a flux ratio")
        excess = value / baseline - 1.0
        excess_err = value_err / baseline
    return excess, excess_err


# ---------------------------------------------------------------------------
# Detection: a bounded "N consecutive sigma-threshold points" heuristic.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlareCandidate:
    start_time: float
    end_time: float
    peak_time: float
    peak_excess: float
    n_points: int


def detect_flare_candidates(time, value, value_err, value_kind: str, *,
                            sigma_threshold: float = 3.0, min_consecutive_points: int = 3,
                            baseline_window_days: float = 1.0) -> list[FlareCandidate]:
    """Flags points with a relative excess above `sigma_threshold` times
    the excess series' own MAD-based sigma, then keeps only runs of at
    least `min_consecutive_points` consecutive flagged points (the standard
    convention, e.g. Chang et al. 2015) as candidates. Detection only --
    no false-positive vetting beyond the run-length requirement, per the
    module docstring."""
    if sigma_threshold <= 0:
        raise FlareError("sigma_threshold must be positive")
    if min_consecutive_points < 1:
        raise FlareError("min_consecutive_points must be at least 1")
    time, value, value_err = _finite_arrays(time, value, value_err)
    excess, _ = relative_flux_excess(time, value, value_err, value_kind,
                                     baseline_window_days=baseline_window_days)

    mad = float(np.median(np.abs(excess - np.median(excess))))
    sigma = 1.4826 * mad
    if sigma <= 0:
        return []
    flagged = excess > sigma_threshold * sigma

    candidates: list[FlareCandidate] = []
    run_start = None
    for i, flag in enumerate(np.append(flagged, False)):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            run_end = i - 1
            if run_end - run_start + 1 >= min_consecutive_points:
                run_slice = slice(run_start, run_end + 1)
                peak_offset = int(np.argmax(excess[run_slice]))
                candidates.append(FlareCandidate(
                    start_time=float(time[run_start]), end_time=float(time[run_end]),
                    peak_time=float(time[run_start + peak_offset]),
                    peak_excess=float(excess[run_start + peak_offset]),
                    n_points=run_end - run_start + 1,
                ))
            run_start = None
    return candidates


# ---------------------------------------------------------------------------
# Least-squares refinement of the flare template against real data.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlareFit:
    t_peak: float
    fwhm: float
    amplitude: float
    residual_rms: float
    n_evaluations: int


_FIT_PARAM_ORDER = ("t_peak", "fwhm", "amplitude")
_FIT_BOUNDS = {
    "t_peak": (-np.inf, np.inf), "fwhm": (1e-6, np.inf), "amplitude": (1e-8, np.inf),
}


def fit_flare_template(time, excess, excess_err, initial_guess: dict[str, float]) -> FlareFit:
    """Least-squares refinement of `flare_model`'s three parameters against
    an already-computed relative-excess series (`relative_flux_excess`'s
    output) -- kept decoupled from `value_kind` so this function operates
    on one consistent representation regardless of the source survey.

    A real convergence bug was found and fixed this session, running this
    function against a real Kepler light curve (`surveys/kepler.py`'s own
    validation study) for the first time: Kepler's absolute BJD_TDB time
    values are ~2.455e6 (days), and `scipy.optimize.least_squares`'s
    default `xtol` (relative step-size) termination check compares each
    parameter's step against ITS OWN magnitude -- with `t_peak` sitting at
    ~2.455e6 while `fwhm`/`amplitude` sit at ~1e-2/1e-1, the trust-region
    step size the optimizer considers "converged" for `t_peak` is many
    orders of magnitude larger, in absolute terms, than the fit actually
    needs for the other two parameters, so the whole fit falsely reports
    `xtol` convergence after only 2 function evaluations without ever
    correcting the initial guess (confirmed directly: cost stayed exactly
    606,723 across both evaluations). This was invisible in this
    function's own prior synthetic-data tests because those used small,
    near-zero relative time values where the scale mismatch never
    triggers. The fix is the standard one for this exact failure mode:
    fit in TIME RELATIVE TO THE INITIAL `t_peak` GUESS (so `t_peak`
    starts the optimization at exactly 0, the same order of magnitude as
    `fwhm`) and shift the fitted `t_peak` back to absolute time afterward
    -- confirmed to actually converge (`ftol`, not `xtol`, after 31
    evaluations, not 2) on the same real Kepler data that exposed the bug.
    """
    time = np.asarray(time, dtype=np.float64)
    excess = np.asarray(excess, dtype=np.float64)
    excess_err = np.asarray(excess_err, dtype=np.float64)
    finite = np.isfinite(time) & np.isfinite(excess) & np.isfinite(excess_err) & (excess_err > 0)
    time, excess, excess_err = time[finite], excess[finite], excess_err[finite]
    if len(time) < 5:
        raise FlareError(f"need at least 5 finite points to fit a flare, got {len(time)}")
    missing = [name for name in _FIT_PARAM_ORDER if name not in initial_guess]
    if missing:
        raise FlareError(f"initial_guess is missing required parameters: {missing}")

    time_offset = float(initial_guess["t_peak"])
    time_relative = time - time_offset

    x0 = np.array([float(initial_guess[name]) - time_offset if name == "t_peak"
                   else float(initial_guess[name]) for name in _FIT_PARAM_ORDER])
    lower = np.array([_FIT_BOUNDS[name][0] for name in _FIT_PARAM_ORDER])
    upper = np.array([_FIT_BOUNDS[name][1] for name in _FIT_PARAM_ORDER])
    if np.any(x0 <= lower) or np.any(x0 >= upper):
        raise FlareError("initial_guess falls outside the physically valid parameter bounds")

    def residuals(params: np.ndarray) -> np.ndarray:
        kwargs = dict(zip(_FIT_PARAM_ORDER, params))
        return (excess - flare_model(time_relative, **kwargs)) / excess_err

    result = least_squares(residuals, x0, bounds=(lower, upper), method="trf")
    if not result.success:
        raise FlareError(f"flare template fit did not converge: {result.message}")

    fitted = dict(zip(_FIT_PARAM_ORDER, (float(x) for x in result.x)))
    fitted["t_peak"] += time_offset
    rms = float(np.sqrt(np.mean(result.fun ** 2)))
    return FlareFit(**fitted, residual_rms=rms, n_evaluations=int(result.nfev))


# ---------------------------------------------------------------------------
# Equivalent duration (ED).
# ---------------------------------------------------------------------------

def equivalent_duration(time, excess, *, window: tuple[float, float] | None = None) -> float:
    """Trapezoidal time-integral of relative flux excess, in SECONDS
    (`time` is assumed in days, this codebase's convention elsewhere, e.g.
    `transit_ttv.py`'s `period_days`). Exact given real data; needs no
    model fit."""
    time = np.asarray(time, dtype=np.float64)
    excess = np.asarray(excess, dtype=np.float64)
    if len(time) < 2:
        raise FlareError("need at least two points to integrate an equivalent duration")
    if window is not None:
        mask = (time >= window[0]) & (time <= window[1])
        if int(mask.sum()) < 2:
            raise FlareError("window contains fewer than two points")
        time, excess = time[mask], excess[mask]
    return float(np.trapezoid(excess, time) * SECONDS_PER_DAY)


__all__ = [
    "FlareError", "davenport_flare_template", "flare_model",
    "relative_flux_excess", "FlareCandidate", "detect_flare_candidates",
    "FlareFit", "fit_flare_template", "equivalent_duration",
    "MIN_POINTS", "SECONDS_PER_DAY",
]
