"""Optimisation and posterior sampling for microlensing models (item 15).

This is the first genuine multi-parameter posterior in this codebase.
`multiband_hier.credible_interval` builds a 1-D grid-HPD (and is itself
never coverage-tested anywhere); `sed.py`'s blackbody fit is a 1-D grid
argmin reporting no uncertainty at all. Here the posterior is sampled
properly and then actually validated, by `microlensing_eval.py`.

Two design choices worth stating:

**scipy, not a new dependency.** `scipy` has been a core dependency all
along but only `scipy.stats.rankdata` was ever used (`anomaly.py`), so
`differential_evolution`/`minimize` cost nothing new. Only the SAMPLER
(emcee) is a new, gated dependency -- and point-lens *optimisation* works
without it.

**Classical fitter first, neural proposal later -- a 2026-informed
choice, not a conservative one.** The current state of the art for binary
lenses is amortised neural posterior estimation (CausticFlow, Ren & Zhu,
arXiv:2607.04955, July 2026: a neural-CDE encoder plus a normalising flow,
trained on KMTNet-like simulations). That paper's own numbers are the
reason the optimiser comes first: the flow alone reaches only ~17% on mass
ratio q and ~3% on separation s, improving to <5% and <1% *only when used
as a proposal for downstream local optimisation*. So the local optimiser
is load-bearing either way. Every `fit_*` function here takes an optional
`proposal` callable, which is exactly where such a flow plugs in later
without touching this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .microlensing import (
    MicrolensingError, PointLensParams, magnification, solve_linear_flux,
)

# Physically motivated default search bounds. tE below ~0.5 d is faster
# than any ground-based survey's cadence can constrain; above ~500 d the
# "event" is not separable from ordinary long-term variability. u0 above 3
# gives a magnification under 1.02, indistinguishable from a flat baseline.
DEFAULT_TE_BOUNDS = (0.5, 500.0)
DEFAULT_U0_BOUNDS = (1e-4, 3.0)


@dataclass
class PointLensFit:
    params: PointLensParams
    f_source: float
    f_blend: float
    chi2: float
    reduced_chi2: float
    n_points: int
    converged: bool
    note: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["params"] = self.params.to_dict()
        return payload


@dataclass
class PosteriorResult:
    """Samples plus the intervals `microlensing_eval` will coverage-test."""

    parameter_names: tuple[str, ...]
    samples: np.ndarray                       # (n_samples, n_parameters)
    intervals: dict = field(default_factory=dict)
    medians: dict = field(default_factory=dict)
    autocorrelation_time: dict = field(default_factory=dict)
    converged: bool = False
    n_steps: int = 0
    n_walkers: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "parameter_names": list(self.parameter_names),
            "n_samples": int(len(self.samples)),
            "intervals": self.intervals,
            "medians": self.medians,
            "autocorrelation_time": self.autocorrelation_time,
            "converged": self.converged,
            "n_steps": self.n_steps,
            "n_walkers": self.n_walkers,
            "note": self.note,
        }


def chi_squared(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                params: PointLensParams) -> float:
    """Weighted chi-square with the two linear flux parameters profiled out.

    Because `f_source`/`f_blend` are solved exactly at every call
    (`solve_linear_flux`), this is the profile chi-square over the three
    nonlinear parameters only -- what the optimiser and sampler both see.
    """
    f_source, f_blend = solve_linear_flux(time, flux, flux_err, params)
    model = f_source * magnification(time, params) + f_blend
    residual = (np.asarray(flux, dtype=np.float64) - model) / np.asarray(
        flux_err, dtype=np.float64)
    value = float(np.sum(residual ** 2))
    return value if np.isfinite(value) else float("inf")


def _bounded_chi2(vector, time, flux, flux_err, bounds):
    """chi2 for a raw parameter vector, returning inf outside the bounds so
    the optimiser and the sampler agree on the supported region."""
    t0, tE, u0 = float(vector[0]), float(vector[1]), float(vector[2])
    (t0_lo, t0_hi), (tE_lo, tE_hi), (u0_lo, u0_hi) = bounds
    if not (t0_lo <= t0 <= t0_hi and tE_lo <= tE <= tE_hi and u0_lo <= u0 <= u0_hi):
        return float("inf")
    try:
        return chi_squared(time, flux, flux_err, PointLensParams(t0=t0, tE=tE, u0=u0))
    except MicrolensingError:
        return float("inf")


def default_bounds(time: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Search bounds derived from the data's own baseline.

    `t0` is bounded by the observed window (padded by a little, since a
    peak just outside the window is still constrainable); `tE`/`u0` use the
    physical defaults above.
    """
    time = np.asarray(time, dtype=np.float64)
    start, end = float(np.min(time)), float(np.max(time))
    span = end - start
    pad = 0.1 * span if span > 0 else 1.0
    return ((start - pad, end + pad), DEFAULT_TE_BOUNDS, DEFAULT_U0_BOUNDS)


def fit_point_lens(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                   bounds: tuple[tuple[float, float], ...] | None = None,
                   seed: int = 42, maxiter: int = 200,
                   proposal=None) -> PointLensFit:
    """Global search then local polish for the three nonlinear parameters.

    `differential_evolution` first because a microlensing chi-square
    surface is genuinely multimodal in `t0` (any bump in the baseline is a
    local minimum), so a purely local optimiser started at a guess would
    be at the mercy of that guess. `minimize` afterwards to refine to
    machine precision, since differential evolution converges slowly at
    the very end.

    `proposal`, when given, is called as `proposal(time, flux, flux_err)`
    and must return a starting parameter vector -- the seam a future
    CausticFlow-style neural posterior estimator plugs into (see module
    docstring). When supplied, the global stage is skipped and only the
    local polish runs, which is exactly the configuration that paper
    reports as its best-performing one.
    """
    from scipy.optimize import differential_evolution, minimize

    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)
    if len(time) < 5:
        raise MicrolensingError("need at least five points to fit a point lens")
    if not (len(time) == len(flux) == len(flux_err)):
        raise MicrolensingError("time, flux and flux_err must have equal lengths")
    if np.any(flux_err <= 0):
        raise MicrolensingError("flux_err must be positive")

    bounds = bounds or default_bounds(time)
    objective = lambda v: _bounded_chi2(v, time, flux, flux_err, bounds)  # noqa: E731

    note = ""
    if proposal is not None:
        start = np.asarray(proposal(time, flux, flux_err), dtype=np.float64)
        note = "started from a supplied proposal; global search skipped"
        converged = True
    else:
        globally = differential_evolution(
            objective, bounds=list(bounds), seed=seed, maxiter=maxiter,
            polish=False, tol=1e-8,
        )
        start = np.asarray(globally.x, dtype=np.float64)
        converged = bool(globally.success)

    polished = minimize(objective, start, method="Nelder-Mead",
                        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
    best = polished.x if np.isfinite(polished.fun) else start

    params = PointLensParams.from_array(best)
    f_source, f_blend = solve_linear_flux(time, flux, flux_err, params)
    chi2 = chi_squared(time, flux, flux_err, params)
    # 3 nonlinear + 2 linear parameters are fitted.
    dof = max(1, len(time) - 5)

    return PointLensFit(
        params=params, f_source=f_source, f_blend=f_blend,
        chi2=chi2, reduced_chi2=chi2 / dof, n_points=len(time),
        converged=converged and np.isfinite(chi2), note=note,
    )


def _require_emcee():
    try:
        import emcee
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MicrolensingError(
            "emcee is not installed; install the 'research' extra "
            "(pip install .[research]) to sample a posterior. "
            "fit_point_lens() itself needs no extra dependency."
        ) from exc
    return emcee


def sample_posterior(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                     start: PointLensFit | PointLensParams,
                     bounds: tuple[tuple[float, float], ...] | None = None,
                     n_walkers: int = 32, n_steps: int = 2000,
                     burn_fraction: float = 0.3, seed: int = 42,
                     levels: tuple[float, ...] = (0.68, 0.9)) -> PosteriorResult:
    """Affine-invariant posterior sampling around a fitted solution.

    Uses a flat prior inside `bounds` and a Gaussian likelihood
    (`-chi2/2`), so the reported intervals are genuine credible intervals
    under that prior rather than optimiser scatter.

    Convergence is REPORTED, not assumed: emcee's integrated
    autocorrelation time is estimated and `converged` is set only when the
    chain is at least 50 autocorrelation times long (emcee's own
    documented rule of thumb). A short chain still returns its samples,
    flagged -- silently presenting an unconverged chain as a posterior is
    exactly the failure this codebase's calibration discipline exists to
    prevent.
    """
    emcee = _require_emcee()

    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)
    bounds = bounds or default_bounds(time)
    params = start.params if isinstance(start, PointLensFit) else start
    centre = params.to_array()

    def log_probability(vector):
        value = _bounded_chi2(vector, time, flux, flux_err, bounds)
        if not np.isfinite(value):
            return -np.inf
        return -0.5 * value

    rng = np.random.default_rng(seed)
    # Small multiplicative jitter around the fitted solution, clipped back
    # inside the bounds so no walker starts at -inf probability.
    scatter = np.abs(centre) * 1e-3 + 1e-6
    positions = centre + scatter * rng.normal(size=(n_walkers, len(centre)))
    for index, (low, high) in enumerate(bounds):
        positions[:, index] = np.clip(positions[:, index], low, high)

    sampler = emcee.EnsembleSampler(n_walkers, len(centre), log_probability)
    sampler.run_mcmc(positions, n_steps, progress=False)

    burn = int(n_steps * burn_fraction)
    chain = sampler.get_chain(discard=burn, flat=True)

    names = ("t0", "tE", "u0")
    try:
        # emcee logs a multi-line warning per call when the chain is short.
        # That information is not lost -- it is returned as
        # `autocorrelation_time` and `converged` below -- so the log itself
        # is silenced to keep a many-trial study's output readable.
        import logging

        logger = logging.getLogger("emcee.autocorr")
        previous_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            tau = sampler.get_autocorr_time(quiet=True)
        finally:
            logger.setLevel(previous_level)
        autocorrelation = {name: (None if not np.isfinite(value) else float(value))
                          for name, value in zip(names, tau)}
        finite_tau = [v for v in tau if np.isfinite(v)]
        converged = bool(finite_tau) and n_steps >= 50 * max(finite_tau)
    except Exception:  # noqa: BLE001 - a failed diagnostic must not lose the samples
        autocorrelation = {name: None for name in names}
        converged = False

    intervals: dict = {}
    medians: dict = {}
    for index, name in enumerate(names):
        column = chain[:, index]
        medians[name] = float(np.median(column))
        intervals[name] = {}
        for level in levels:
            tail = (1.0 - level) / 2.0
            low, high = np.quantile(column, [tail, 1.0 - tail])
            intervals[name][str(level)] = [float(low), float(high)]

    return PosteriorResult(
        parameter_names=names, samples=chain, intervals=intervals,
        medians=medians, autocorrelation_time=autocorrelation,
        converged=converged, n_steps=n_steps, n_walkers=n_walkers,
        note=("" if converged else
             "chain is shorter than 50 autocorrelation times; intervals are "
             "reported but not certified converged"),
    )


def fit_binary_lens(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                    bounds: dict | None = None, seed: int = 42,
                    maxiter: int = 60, proposal=None) -> dict:
    """Global search over the seven binary-lens parameters.

    Stated honestly: this is the hard case. The binary chi-square surface
    is pathological -- a multitude of narrow, deep local minima, plus
    well-known discrete degeneracies (close/wide `s -> 1/s`) that no local
    optimiser escapes. That difficulty is precisely why amortised neural
    posterior estimation exists for this problem (CausticFlow, 2026). This
    function therefore reports its own convergence diagnostics and does
    NOT claim to have found a global optimum; `proposal` is the documented
    route to a better starting point.
    """
    from scipy.optimize import differential_evolution

    from .microlensing import BinaryLensParams, binary_magnification

    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)

    span = float(np.max(time) - np.min(time)) or 1.0
    limits = {
        "t0": (float(np.min(time)), float(np.max(time))),
        "tE": DEFAULT_TE_BOUNDS,
        "u0": DEFAULT_U0_BOUNDS,
        "s": (0.3, 3.0),
        "log_q": (-5.0, 0.0),   # searched in log because q spans decades
        "alpha": (0.0, 2.0 * np.pi),
        **(bounds or {}),
    }
    order = ("t0", "tE", "u0", "s", "log_q", "alpha")
    search_bounds = [limits[name] for name in order]

    def objective(vector):
        try:
            params = BinaryLensParams(
                t0=vector[0], tE=vector[1], u0=vector[2], s=vector[3],
                q=10.0 ** vector[4], alpha=vector[5])
            amplification = binary_magnification(time, params)
        except (MicrolensingError, Exception):  # noqa: BLE001 - a bad draw scores inf
            return float("inf")
        if not np.all(np.isfinite(amplification)):
            return float("inf")
        design = np.column_stack([amplification, np.ones_like(amplification)])
        weights = 1.0 / flux_err ** 2
        try:
            solution = np.linalg.solve(design.T @ (design * weights[:, None]),
                                       design.T @ (flux * weights))
        except np.linalg.LinAlgError:
            return float("inf")
        model = design @ solution
        value = float(np.sum(((flux - model) / flux_err) ** 2))
        return value if np.isfinite(value) else float("inf")

    if proposal is not None:
        best_vector = np.asarray(proposal(time, flux, flux_err), dtype=np.float64)
        success, note = True, "started from a supplied proposal"
    else:
        result = differential_evolution(objective, bounds=search_bounds, seed=seed,
                                        maxiter=maxiter, polish=True, tol=1e-6)
        best_vector, success = result.x, bool(result.success)
        note = ""

    chi2 = objective(best_vector)
    dof = max(1, len(time) - 8)
    return {
        "params": {"t0": float(best_vector[0]), "tE": float(best_vector[1]),
                  "u0": float(best_vector[2]), "s": float(best_vector[3]),
                  "q": float(10.0 ** best_vector[4]),
                  "alpha": float(best_vector[5])},
        "chi2": chi2, "reduced_chi2": chi2 / dof, "n_points": int(len(time)),
        "converged": success and np.isfinite(chi2),
        "note": note or ("binary chi-square surfaces are multimodal; this is the "
                        "best solution found, not a certified global optimum"),
    }
