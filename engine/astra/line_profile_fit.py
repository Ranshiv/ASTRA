"""Optimisation and posterior sampling for `line_profile.py` (item 25).

Mirrors `microlensing_fit.py`'s shape closely: `differential_evolution`
(global, because a line-profile chi-square surface can have a secondary
minimum wherever another nearby feature or noise fluctuation sits) then
`minimize` (local polish), then `emcee` posterior sampling with the same
50-autocorrelation-time convergence discipline. `amplitude` enters the model
LINEARLY (`model = continuum + amplitude * voigt_profile(...)`), exactly
like microlensing's `f_source`/`f_blend` -- so it is solved by weighted
linear least squares at every nonlinear step (`solve_linear_amplitude`)
rather than searched over, turning a 4-parameter fit into a 3-parameter one
(`center`, `sigma`, `gamma`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .line_profile import LineProfileError, LineProfileParams, model_flux, voigt_profile


@dataclass
class LineProfileFit:
    params: LineProfileParams
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
class LineProfilePosteriorResult:
    """Samples plus the intervals `line_profile_eval` coverage-tests."""

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


def solve_linear_amplitude(wavelength, flux, error, continuum,
                           center: float, sigma: float, gamma: float) -> float:
    """Exact weighted-least-squares amplitude for a fixed line shape."""
    profile = voigt_profile(np.asarray(wavelength, dtype=np.float64) - center, sigma, gamma)
    residual = np.asarray(flux, dtype=np.float64) - np.asarray(continuum, dtype=np.float64)
    weights = 1.0 / np.asarray(error, dtype=np.float64) ** 2
    denominator = float(np.sum(weights * profile ** 2))
    if denominator <= 0 or not np.isfinite(denominator):
        return 0.0
    return float(np.sum(weights * profile * residual) / denominator)


def chi_squared(wavelength, flux, error, continuum,
                center: float, sigma: float, gamma: float) -> float:
    """Weighted chi-square with the linear amplitude profiled out."""
    amplitude = solve_linear_amplitude(wavelength, flux, error, continuum, center, sigma, gamma)
    params = LineProfileParams(center=center, sigma=sigma, gamma=gamma, amplitude=amplitude)
    model = model_flux(wavelength, continuum, params)
    residual = (np.asarray(flux, dtype=np.float64) - model) / np.asarray(error, dtype=np.float64)
    value = float(np.sum(residual ** 2))
    return value if np.isfinite(value) else float("inf")


def _bounded_chi2(vector, wavelength, flux, error, continuum, bounds):
    center, sigma, gamma = float(vector[0]), float(vector[1]), float(vector[2])
    (c_lo, c_hi), (s_lo, s_hi), (g_lo, g_hi) = bounds
    if not (c_lo <= center <= c_hi and s_lo <= sigma <= s_hi and g_lo <= gamma <= g_hi):
        return float("inf")
    try:
        return chi_squared(wavelength, flux, error, continuum, center, sigma, gamma)
    except LineProfileError:
        return float("inf")


def default_bounds(wavelength, *, center_hint: float | None = None,
                   window_angstrom: float | None = None
                   ) -> tuple[tuple[float, float], ...]:
    """Search bounds derived from the data's own span.

    `center` is bounded to the observed window by default, or to a narrow
    window around `center_hint` (e.g. a rest line's expected position at a
    known redshift -- the identity `spectroscopy_calibration.
    independent_redshift_from_lines` can supply) when both `center_hint`
    and `window_angstrom` are given. `sigma`/`gamma` are bounded by the
    data's own sample spacing (below which a width is unresolved) and a
    quarter of the observed span (above which "line" and "continuum
    trend" are not distinguishable within this window).
    """
    wave = np.asarray(wavelength, dtype=np.float64)
    lo, hi = float(np.min(wave)), float(np.max(wave))
    span = hi - lo
    spacing = span / max(len(wave) - 1, 1)

    if center_hint is not None and window_angstrom is not None:
        center_bounds = (float(center_hint) - float(window_angstrom),
                         float(center_hint) + float(window_angstrom))
    else:
        center_bounds = (lo, hi)
    width_upper = max(span / 4.0, spacing)
    sigma_bounds = (max(spacing / 2.0, 1e-6), width_upper)
    gamma_bounds = (0.0, width_upper)
    return (center_bounds, sigma_bounds, gamma_bounds)


def fit_line_profile(wavelength, flux, error, continuum,
                     bounds: tuple[tuple[float, float], ...] | None = None,
                     seed: int = 42, maxiter: int = 200, proposal=None,
                     center_hint: float | None = None,
                     window_angstrom: float | None = None) -> LineProfileFit:
    """Global search then local polish for `center`/`sigma`/`gamma`.

    `proposal`, when given, is called as
    `proposal(wavelength, flux, error, continuum)` and must return a
    starting `(center, sigma, gamma)` vector -- the same seam
    `microlensing_fit.fit_point_lens` documents for a future amortised
    estimator. When supplied, the global stage is skipped.
    """
    from scipy.optimize import differential_evolution, minimize

    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    if len(wavelength) < 5:
        raise LineProfileError("need at least five points to fit a line profile")
    if not (len(wavelength) == len(flux) == len(error)):
        raise LineProfileError("wavelength, flux and error must have equal lengths")
    if np.any(error <= 0):
        raise LineProfileError("error must be positive")
    # Validated above BEFORE broadcasting continuum: broadcasting a
    # wrong-length continuum array against `wavelength.shape` raises numpy's
    # own opaque ValueError rather than this module's LineProfileError --
    # a real bug, found via a mismatched-length regression test.
    try:
        continuum = np.broadcast_to(np.asarray(continuum, dtype=np.float64), wavelength.shape)
    except ValueError as exc:
        raise LineProfileError("continuum must be a scalar or match wavelength's length") from exc

    bounds = bounds or default_bounds(wavelength, center_hint=center_hint,
                                      window_angstrom=window_angstrom)
    objective = lambda v: _bounded_chi2(  # noqa: E731
        v, wavelength, flux, error, continuum, bounds)

    note = ""
    if proposal is not None:
        start = np.asarray(proposal(wavelength, flux, error, continuum), dtype=np.float64)
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

    center, sigma, gamma = float(best[0]), float(best[1]), float(best[2])
    amplitude = solve_linear_amplitude(wavelength, flux, error, continuum, center, sigma, gamma)
    params = LineProfileParams(center=center, sigma=sigma, gamma=gamma, amplitude=amplitude)
    chi2 = chi_squared(wavelength, flux, error, continuum, center, sigma, gamma)
    # 3 nonlinear + 1 linear parameter are fitted.
    dof = max(1, len(wavelength) - 4)

    return LineProfileFit(
        params=params, chi2=chi2, reduced_chi2=chi2 / dof, n_points=len(wavelength),
        converged=converged and np.isfinite(chi2), note=note,
    )


def _require_emcee():
    try:
        import emcee
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise LineProfileError(
            "emcee is not installed; install the 'research' extra "
            "(pip install .[research]) to sample a posterior. "
            "fit_line_profile() itself needs no extra dependency."
        ) from exc
    return emcee


def sample_posterior(wavelength, flux, error, continuum,
                     start: LineProfileFit | LineProfileParams,
                     bounds: tuple[tuple[float, float], ...] | None = None,
                     n_walkers: int = 32, n_steps: int = 2000,
                     burn_fraction: float = 0.3, seed: int = 42,
                     levels: tuple[float, ...] = (0.68, 0.9)
                     ) -> LineProfilePosteriorResult:
    """Affine-invariant posterior sampling around a fitted line profile.

    Same discipline as `microlensing_fit.sample_posterior`: a flat prior
    inside `bounds`, a Gaussian likelihood (`-chi2/2`), and convergence
    REPORTED (via emcee's integrated autocorrelation time, the 50-tau rule)
    rather than assumed. Only `center`/`sigma`/`gamma` are sampled;
    `amplitude` stays linearly profiled at every step, exactly as
    `f_source`/`f_blend` are never sampled in the microlensing posterior.
    """
    emcee = _require_emcee()

    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    continuum = np.broadcast_to(np.asarray(continuum, dtype=np.float64), wavelength.shape)
    bounds = bounds or default_bounds(wavelength)
    params = start.params if isinstance(start, LineProfileFit) else start
    centre = np.array([params.center, params.sigma, params.gamma], dtype=np.float64)

    def log_probability(vector):
        value = _bounded_chi2(vector, wavelength, flux, error, continuum, bounds)
        if not np.isfinite(value):
            return -np.inf
        return -0.5 * value

    rng = np.random.default_rng(seed)
    scatter = np.abs(centre) * 1e-3 + 1e-6
    positions = centre + scatter * rng.normal(size=(n_walkers, len(centre)))
    for index, (low, high) in enumerate(bounds):
        positions[:, index] = np.clip(positions[:, index], low, high)

    sampler = emcee.EnsembleSampler(n_walkers, len(centre), log_probability)
    sampler.run_mcmc(positions, n_steps, progress=False)

    burn = int(n_steps * burn_fraction)
    chain = sampler.get_chain(discard=burn, flat=True)

    names = ("center", "sigma", "gamma")
    try:
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

    return LineProfilePosteriorResult(
        parameter_names=names, samples=chain, intervals=intervals,
        medians=medians, autocorrelation_time=autocorrelation,
        converged=converged, n_steps=n_steps, n_walkers=n_walkers,
        note=("" if converged else
             "chain is shorter than 50 autocorrelation times; intervals are "
             "reported but not certified converged"),
    )
