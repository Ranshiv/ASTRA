"""AGN damped-random-walk stochastic model and a TDE change-point statistic.

Shaped like `multiband_hier.py`, which this module reuses the exact
`celerite2` invocation pattern from (`_require_celerite2()` lazy-import
guard, `GaussianProcess(term, mean=...)` -> `gp.compute(time,
diag=error**2 + JITTER_FLOOR)` -> `gp.log_likelihood(value)`), swapping
`terms.RotationTerm` (quasi-periodic) for `terms.RealTerm(a, c)` --
celerite2's real, documented pure-exponential kernel `a * exp(-c*|dt|)`,
exactly the standard AGN damped-random-walk (DRW) covariance (Kelly et al.
2009, ApJ 698, 895; MacLeod et al. 2010, ApJ 721, 1014), with
`sigma**2 = a` and damping timescale `tau = 1/c`. `celerite2` is already a
`research`-extra dependency, used unchanged.

`celerite2.GaussianProcess(kernel, mean=...)` accepts a plain Python
callable for `mean` (confirmed by direct API inspection this session,
`celerite2==0.3.3`: `gp.compute(t, ...)` evaluates `mean(t)` itself) --
this is what makes `changepoint_evidence` possible as ONE consistent GP
log-likelihood comparison (DRW-only mean vs. DRW+flare mean) rather than
needing a separate GP-whitened-residual step.

`tde_flare_model` is a pragmatic, citable empirical parametrization (a
continuous Gaussian rise into a `t^-5/3` power-law decay -- `5/3` is the
standard TDE fallback-accretion-rate power law, Rees 1988, Nature 333,
523), in the same spirit as real-time TDE-alert pipelines' broken
empirical models (e.g. van Velzen et al. 2019, ApJ 872, 198) -- NOT a full
relativistic tidal-disruption simulation, a real, stated scope limit.

`calibrate_changepoint_significance` draws synthetic DRW-only realizations
at the SAME fitted `(sigma, tau)` (no injected flare) via `celerite2`'s
own `gp.sample()`, and reports the delta-BIC value at a target false-
positive rate -- a real null-population calibration, in the same spirit
as `association.calibrate_event_graph`'s scrambled-time-slide null (that
function operates on discrete event lists, not one continuous light
curve, so it is not directly reusable; this is the analogous idea
reimplemented for this shape of data). `celerite2.GaussianProcess.sample()`
has no keyword for an explicit random generator (confirmed by direct API
inspection this session) -- it draws from NumPy's global random state, so
reproducibility here means seeding that global state locally
(`np.random.seed(seed)`), a real, discovered constraint of the library,
not a design choice.

Neither ZTF/Swift/XMM's real connectors nor a NEOWISE connector (which
does not exist in this codebase at all, confirmed while planning this
module) change anything about this module -- it operates on any
`time`/`value`/`value_err` arrays, survey-agnostic by design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

MIN_POINTS = 20
JITTER_FLOOR = 1e-6


class AGNChangepointError(ValueError):
    """A DRW fit, flare model, or change-point computation could not be completed."""


def _require_celerite2():
    try:
        from celerite2 import GaussianProcess, terms
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise AGNChangepointError(
            "celerite2 is not installed; install the 'research' extra "
            "(pip install .[research]) to use agn_changepoint.py"
        ) from exc
    return GaussianProcess, terms


def _finite_arrays(time, value, value_err) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    value_err = np.asarray(value_err, dtype=np.float64)
    if not (len(time) == len(value) == len(value_err)):
        raise AGNChangepointError("time, value, and value_err must be the same length")
    finite = np.isfinite(time) & np.isfinite(value) & np.isfinite(value_err) & (value_err > 0)
    return time[finite], value[finite], value_err[finite]


# ---------------------------------------------------------------------------
# DRW stochastic-process fit.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DRWFit:
    sigma: float
    tau: float
    mean_value: float
    log_likelihood: float
    n_points: int


def _drw_log_likelihood(GaussianProcess, terms, time, value, error, mean_value,
                        sigma: float, tau: float) -> float:
    term = terms.RealTerm(a=sigma ** 2, c=1.0 / tau)
    gp = GaussianProcess(term, mean=float(mean_value))
    try:
        gp.compute(time, diag=error ** 2 + JITTER_FLOOR)
        return float(gp.log_likelihood(value))
    except Exception:  # noqa: BLE001 - a degenerate hyperparameter draw is a bad fit, not a crash
        return float("-inf")


def fit_drw(time, value, value_err, *, sigma_guess: float | None = None,
           tau_guess: float | None = None) -> DRWFit:
    """Maximizes the celerite2 DRW GP log-likelihood over `(sigma, tau)`."""
    GaussianProcess, terms = _require_celerite2()
    time, value, value_err = _finite_arrays(time, value, value_err)
    if len(time) < MIN_POINTS:
        raise AGNChangepointError(f"need at least {MIN_POINTS} finite points, got {len(time)}")
    order = np.argsort(time)
    time, value, value_err = time[order], value[order], value_err[order]
    mean_value = float(np.mean(value))
    span = float(time[-1] - time[0])
    if span <= 0:
        raise AGNChangepointError("time span must be positive")

    sigma0 = sigma_guess if sigma_guess is not None else (float(np.std(value)) or 1.0)
    tau0 = tau_guess if tau_guess is not None else max(span / 5.0, 1e-3)

    def neg_log_likelihood(params: np.ndarray) -> float:
        log_sigma, log_tau = params
        value_ll = _drw_log_likelihood(GaussianProcess, terms, time, value, value_err,
                                       mean_value, math.exp(log_sigma), math.exp(log_tau))
        return -value_ll if np.isfinite(value_ll) else 1e10

    bounds = [(math.log(1e-6), math.log(1e6)), (math.log(1e-3), math.log(max(span * 100.0, 1.0)))]
    x0 = np.array([math.log(max(sigma0, 1e-6)), math.log(max(tau0, 1e-3))])
    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise AGNChangepointError(f"DRW fit did not converge: {result.message}")

    sigma, tau = math.exp(result.x[0]), math.exp(result.x[1])
    return DRWFit(sigma=sigma, tau=tau, mean_value=mean_value,
                 log_likelihood=-float(result.fun), n_points=len(time))


# ---------------------------------------------------------------------------
# TDE flare model: continuous Gaussian rise, t^-5/3 power-law decay.
# ---------------------------------------------------------------------------

def tde_flare_model(time, t0: float, amplitude: float, rise_sigma: float,
                    t_decay_ref: float, decay_index: float = 5.0 / 3.0) -> np.ndarray:
    time = np.asarray(time, dtype=np.float64)
    if amplitude <= 0:
        raise AGNChangepointError("amplitude must be positive")
    if rise_sigma <= 0:
        raise AGNChangepointError("rise_sigma must be positive")
    if t_decay_ref <= 0:
        raise AGNChangepointError("t_decay_ref must be positive")
    dt = time - t0
    result = np.empty_like(time)
    rising = dt < 0
    result[rising] = amplitude * np.exp(-0.5 * (dt[rising] / rise_sigma) ** 2)
    decaying = ~rising
    result[decaying] = amplitude * ((dt[decaying] + t_decay_ref) / t_decay_ref) ** (-decay_index)
    return result


# ---------------------------------------------------------------------------
# Change-point evidence: DRW-only vs. DRW+flare, via one consistent GP.
# ---------------------------------------------------------------------------

_FLARE_PARAM_ORDER = ("t0", "amplitude", "rise_sigma", "t_decay_ref")
N_FLARE_PARAMS = len(_FLARE_PARAM_ORDER)


@dataclass(frozen=True)
class ChangepointEvidence:
    log_likelihood_drw_only: float
    log_likelihood_drw_plus_flare: float
    delta_log_likelihood: float
    delta_bic: float
    flare_params: dict[str, float]
    n_points: int


def default_flare_guess(time: np.ndarray, value: np.ndarray, drw_fit: DRWFit) -> dict[str, float]:
    """A reasonable `flare_guess` for `changepoint_evidence`: `t0` at the
    largest excursion from the DRW mean (the most flare-like point in the
    series), not an arbitrary fixed time. `changepoint_evidence`'s
    optimizer is local (`L-BFGS-B`), so a `t0` guess far from any real
    excursion (e.g. a fixed grid time) can converge to a spurious
    near-zero-amplitude solution rather than finding a real injected
    flare -- confirmed this session by exactly that failure mode with a
    fixed end-of-series guess."""
    time = np.asarray(time, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    peak_idx = int(np.argmax(np.abs(value - drw_fit.mean_value)))
    span = float(time[-1] - time[0]) if len(time) > 1 else 1.0
    amplitude = max(abs(float(value[peak_idx] - drw_fit.mean_value)), drw_fit.sigma, 1e-3)
    return {"t0": float(time[peak_idx]), "amplitude": amplitude,
           "rise_sigma": max(span / 10.0, 1e-3), "t_decay_ref": max(span / 10.0, 1e-3)}


def changepoint_evidence(time, value, value_err, drw_fit: DRWFit,
                         flare_guess: dict[str, float]) -> ChangepointEvidence:
    """Delta-log-likelihood/delta-BIC between the DRW-only fit and the best
    DRW+flare fit, both evaluated through the SAME `celerite2` DRW kernel
    (only the GP's mean function differs) -- one consistent comparison,
    not two separately-normalised fits."""
    GaussianProcess, terms = _require_celerite2()
    time, value, value_err = _finite_arrays(time, value, value_err)
    if len(time) < MIN_POINTS:
        raise AGNChangepointError(f"need at least {MIN_POINTS} finite points, got {len(time)}")
    order = np.argsort(time)
    time, value, value_err = time[order], value[order], value_err[order]
    missing = [name for name in _FLARE_PARAM_ORDER if name not in flare_guess]
    if missing:
        raise AGNChangepointError(f"flare_guess is missing required parameters: {missing}")

    term = terms.RealTerm(a=drw_fit.sigma ** 2, c=1.0 / drw_fit.tau)
    diag = value_err ** 2 + JITTER_FLOOR

    gp0 = GaussianProcess(term, mean=drw_fit.mean_value)
    gp0.compute(time, diag=diag)
    ll_drw_only = float(gp0.log_likelihood(value))

    span = float(time[-1] - time[0])
    lower = [time[0] - span, 1e-6, 1e-3, 1e-3]
    upper = [time[-1] + span, np.inf, span * 10.0, span * 10.0]

    def neg_log_likelihood(params: np.ndarray) -> float:
        t0, amplitude, rise_sigma, t_decay_ref = params

        def mean_fn(t):
            try:
                return drw_fit.mean_value + tde_flare_model(t, t0, amplitude, rise_sigma, t_decay_ref)
            except AGNChangepointError:
                return np.full_like(np.asarray(t, dtype=np.float64), np.nan)

        gp = GaussianProcess(term, mean=mean_fn)
        try:
            gp.compute(time, diag=diag)
            value_ll = float(gp.log_likelihood(value))
            return -value_ll if np.isfinite(value_ll) else 1e10
        except Exception:  # noqa: BLE001 - a degenerate draw is a bad fit, not a crash
            return 1e10

    x0 = np.array([float(flare_guess[name]) for name in _FLARE_PARAM_ORDER])
    x0 = np.clip(x0, lower, upper)
    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=list(zip(lower, upper)))
    ll_drw_plus_flare = -float(result.fun)
    fitted = dict(zip(_FLARE_PARAM_ORDER, (float(x) for x in result.x)))

    delta_ll = ll_drw_plus_flare - ll_drw_only
    delta_bic = -2.0 * delta_ll + N_FLARE_PARAMS * math.log(len(time))
    return ChangepointEvidence(
        log_likelihood_drw_only=ll_drw_only, log_likelihood_drw_plus_flare=ll_drw_plus_flare,
        delta_log_likelihood=delta_ll, delta_bic=delta_bic, flare_params=fitted, n_points=len(time),
    )


def calibrate_changepoint_significance(drw_fit: DRWFit, time_grid, value_err, *,
                                       n_realizations: int = 200, target_fpr: float = 0.01,
                                       seed: int = 42) -> float:
    """Delta-BIC threshold at `target_fpr` under the null (no injected
    flare), from real synthetic DRW realizations at the fitted `(sigma,
    tau)`. A real observed `delta_bic` at or below this threshold is
    significant at `target_fpr`."""
    GaussianProcess, terms = _require_celerite2()
    time_grid = np.asarray(time_grid, dtype=np.float64)
    value_err = np.asarray(value_err, dtype=np.float64)
    if len(time_grid) < MIN_POINTS:
        raise AGNChangepointError(f"need at least {MIN_POINTS} finite points, got {len(time_grid)}")
    if n_realizations < 1:
        raise AGNChangepointError("n_realizations must be at least 1")
    if not 0.0 < target_fpr < 1.0:
        raise AGNChangepointError("target_fpr must be in (0, 1)")

    order = np.argsort(time_grid)
    time_grid, value_err = time_grid[order], value_err[order]
    term = terms.RealTerm(a=drw_fit.sigma ** 2, c=1.0 / drw_fit.tau)
    gp = GaussianProcess(term, mean=drw_fit.mean_value)
    gp.compute(time_grid, diag=value_err ** 2 + JITTER_FLOOR)

    np.random.seed(seed)  # celerite2's gp.sample() has no explicit-generator kwarg (see module docstring)
    delta_bics = []
    for _ in range(n_realizations):
        sample = gp.sample()
        flare_guess = default_flare_guess(time_grid, sample, drw_fit)
        evidence = changepoint_evidence(time_grid, sample, value_err, drw_fit, flare_guess)
        delta_bics.append(evidence.delta_bic)
    return float(np.quantile(delta_bics, target_fpr))


__all__ = [
    "AGNChangepointError", "DRWFit", "fit_drw", "tde_flare_model",
    "default_flare_guess", "ChangepointEvidence", "changepoint_evidence", "calibrate_changepoint_significance",
    "MIN_POINTS", "N_FLARE_PARAMS",
]
