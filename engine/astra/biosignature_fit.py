"""Optimisation, posterior sampling, and band-detection significance for
`biosignature.py` (roadmap: astrophysics & extraterrestrial-study
feature pass). Mirrors `line_profile_fit.py`'s shape closely:
`differential_evolution` (global) then `minimize` (local polish), then
`emcee` posterior sampling with the same 50-autocorrelation-time
convergence-REPORTED-not-assumed discipline.

Unlike `line_profile_fit.py`'s single linear amplitude
(`solve_linear_amplitude`), a band amplitude here enters the model
NONLINEARLY through the exponential in `transit_depth` -- so amplitudes
are fit directly as free parameters (`log10_amp_i`), not profiled out
analytically. This is stated explicitly because it is the one place this
module's fitting discipline diverges from `line_profile_fit.py`'s.

`detection_significance` reports DeltaBIC/DeltaChi2 between the full
model and a flat-line (featureless) null for one molecule at a time --
this is a BAND-DETECTION significance, never converted to a claimed
abundance (see `biosignature.py`'s `[GAP]`). `disequilibrium_flag` is a
co-detection SCREENING HEURISTIC (CH4 with O2/O3 above a significance
threshold, the textbook disequilibrium biosignature pair -- e.g.
Lovelock 1965's original argument, and Krissansen-Totton et al. 2018's
modern false-positive framework), returned with explicit caveats and
never folded into `evidence.WEIGHTS`/`scoring.combine()`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from .biosignature import (
    AtmosphereParameters, BiosignatureError, SystemParameters,
    _finite_arrays, default_bounds, transit_depth,
)


@dataclass
class BiosignatureFit:
    params: AtmosphereParameters
    chi2: float
    reduced_chi2: float
    n_points: int
    converged: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["params"] = self.params.to_dict()
        return payload


@dataclass
class BiosignaturePosteriorResult:
    parameter_names: tuple[str, ...]
    samples: np.ndarray
    intervals: dict = field(default_factory=dict)
    medians: dict = field(default_factory=dict)
    autocorrelation_time: dict = field(default_factory=dict)
    converged: bool = False
    n_steps: int = 0
    n_walkers: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_names": list(self.parameter_names), "n_samples": int(len(self.samples)),
            "intervals": self.intervals, "medians": self.medians,
            "autocorrelation_time": self.autocorrelation_time, "converged": self.converged,
            "n_steps": self.n_steps, "n_walkers": self.n_walkers, "note": self.note,
        }


def chi_squared(wavelength_um, depth, error, atmosphere: AtmosphereParameters,
                system: SystemParameters, *, cross_sections: Mapping[str, float]) -> float:
    model = transit_depth(wavelength_um, atmosphere, system, cross_sections=cross_sections)
    residual = (np.asarray(depth, dtype=np.float64) - model) / np.asarray(error, dtype=np.float64)
    value = float(np.sum(residual ** 2))
    return value if np.isfinite(value) else float("inf")


def _bounded_chi2(vector, wavelength_um, depth, error, system, bounds, molecules,
                  mean_molecular_weight, cross_sections):
    for value, (lo, hi) in zip(vector, bounds):
        if not (lo <= value <= hi):
            return float("inf")
    try:
        atmosphere = AtmosphereParameters.from_array(
            vector, molecules=molecules, mean_molecular_weight=mean_molecular_weight)
        return chi_squared(wavelength_um, depth, error, atmosphere, system,
                           cross_sections=cross_sections)
    except BiosignatureError:
        return float("inf")


def fit_transmission_spectrum(wavelength_um, depth, error, system: SystemParameters, *,
                              molecules: tuple[str, ...], cross_sections: Mapping[str, float],
                              mean_molecular_weight: float = 2.3,
                              bounds: tuple[tuple[float, float], ...] | None = None,
                              seed: int = 42, maxiter: int = 200, proposal=None) -> BiosignatureFit:
    """Global search then local polish for
    `(temperature_k, reference_radius_rjup, log10_amp_1, ...)`.

    `proposal`, when given, is called as
    `proposal(wavelength_um, depth, error)` and must return a starting
    parameter vector in the same order -- the same seam
    `line_profile_fit.fit_line_profile` documents. When supplied, the
    global stage is skipped.
    """
    from scipy.optimize import differential_evolution, minimize

    wave, dep, err = _finite_arrays(wavelength_um, depth, error)
    bounds = bounds or default_bounds(dep, system, molecules=molecules)
    objective = lambda v: _bounded_chi2(  # noqa: E731
        v, wave, dep, err, system, bounds, molecules, mean_molecular_weight, cross_sections)

    note = ""
    if proposal is not None:
        start = np.asarray(proposal(wave, dep, err), dtype=np.float64)
        note = "started from a supplied proposal; global search skipped"
        converged = True
    else:
        globally = differential_evolution(objective, bounds=list(bounds), seed=seed,
                                          maxiter=maxiter, polish=False, tol=1e-8)
        start = np.asarray(globally.x, dtype=np.float64)
        converged = bool(globally.success)

    polished = minimize(objective, start, method="Nelder-Mead",
                        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
    best = polished.x if np.isfinite(polished.fun) else start

    atmosphere = AtmosphereParameters.from_array(best, molecules=molecules,
                                                 mean_molecular_weight=mean_molecular_weight)
    chi2 = chi_squared(wave, dep, err, atmosphere, system, cross_sections=cross_sections)
    n_params = 2 + len(molecules)
    dof = max(1, len(wave) - n_params)

    return BiosignatureFit(params=atmosphere, chi2=chi2, reduced_chi2=chi2 / dof,
                           n_points=len(wave), converged=converged and np.isfinite(chi2), note=note)


def _require_emcee():
    try:
        import emcee
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BiosignatureError(
            "emcee is not installed; install the 'research' extra "
            "(pip install .[research]) to sample a posterior. "
            "fit_transmission_spectrum() itself needs no extra dependency."
        ) from exc
    return emcee


def sample_posterior(wavelength_um, depth, error, system: SystemParameters,
                     start: BiosignatureFit | AtmosphereParameters, *,
                     molecules: tuple[str, ...], cross_sections: Mapping[str, float],
                     mean_molecular_weight: float = 2.3,
                     bounds: tuple[tuple[float, float], ...] | None = None,
                     n_walkers: int = 32, n_steps: int = 2000, burn_fraction: float = 0.3,
                     seed: int = 42, levels: tuple[float, ...] = (0.68, 0.9)
                     ) -> BiosignaturePosteriorResult:
    """Affine-invariant posterior sampling, same 50-tau convergence
    discipline as `line_profile_fit.sample_posterior`."""
    emcee = _require_emcee()

    wave, dep, err = _finite_arrays(wavelength_um, depth, error)
    bounds = bounds or default_bounds(dep, system, molecules=molecules)
    atmosphere = start.params if isinstance(start, BiosignatureFit) else start
    centre = atmosphere.to_array(molecules)

    def log_probability(vector):
        value = _bounded_chi2(vector, wave, dep, err, system, bounds, molecules,
                              mean_molecular_weight, cross_sections)
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
    names = ("temperature_k", "reference_radius_rjup") + tuple(f"log10_amp_{m}" for m in molecules)

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
        converged = bool(finite_tau) and bool(n_steps >= 50 * max(finite_tau))
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

    return BiosignaturePosteriorResult(
        parameter_names=names, samples=chain, intervals=intervals, medians=medians,
        autocorrelation_time=autocorrelation, converged=converged, n_steps=n_steps,
        n_walkers=n_walkers,
        note=("" if converged else
             "chain is shorter than 50 autocorrelation times; intervals are "
             "reported but not certified converged"))


def detection_significance(wavelength_um, depth, error, system: SystemParameters,
                           molecule: str, *, cross_sections: Mapping[str, float],
                           mean_molecular_weight: float = 2.3, seed: int = 42) -> dict[str, Any]:
    """DeltaBIC/DeltaChi2 between the full (`molecule` included) model and
    a flat-line null, for ONE molecule -- a band-detection significance,
    never an abundance (see `biosignature.py`'s `[GAP]`)."""
    wave, dep, err = _finite_arrays(wavelength_um, depth, error)

    full_fit = fit_transmission_spectrum(wave, dep, err, system, molecules=(molecule,),
                                         cross_sections=cross_sections,
                                         mean_molecular_weight=mean_molecular_weight, seed=seed)
    null_atmosphere = AtmosphereParameters(temperature_k=full_fit.params.temperature_k,
                                           mean_molecular_weight=mean_molecular_weight,
                                           reference_radius_rjup=full_fit.params.reference_radius_rjup,
                                           abundances=())
    null_chi2 = chi_squared(wave, dep, err, null_atmosphere, system, cross_sections=cross_sections)

    n = len(wave)
    k_full, k_null = 3, 2  # temperature, radius, [+ one amplitude]
    bic_full = full_fit.chi2 + k_full * np.log(n)
    bic_null = null_chi2 + k_null * np.log(n)
    delta_bic = float(bic_null - bic_full)
    delta_chi2 = float(null_chi2 - full_fit.chi2)

    return {"molecule": molecule, "delta_bic": delta_bic, "delta_chi2": delta_chi2,
           "full_chi2": full_fit.chi2, "null_chi2": null_chi2, "n_points": n,
           "log10_amplitude": dict(full_fit.params.abundances).get(molecule),
           "detected": bool(delta_bic > 10.0)}  # Kass & Raftery (1995): DeltaBIC > 10 is "very strong"


def disequilibrium_flag(significances: Mapping[str, dict[str, Any]], *,
                        delta_bic_threshold: float = 10.0) -> dict[str, Any]:
    """CH4 + (O2 or O3) co-detection screening heuristic. See module
    docstring: a heuristic, never a life detection, never scored."""
    ch4 = significances.get("CH4")
    o2 = significances.get("O2")
    o3 = significances.get("O3")
    ch4_detected = bool(ch4 and ch4["delta_bic"] > delta_bic_threshold)
    oxidant_detected = bool((o2 and o2["delta_bic"] > delta_bic_threshold)
                           or (o3 and o3["delta_bic"] > delta_bic_threshold))
    return {"ch4_detected": ch4_detected, "oxidant_detected": oxidant_detected,
           "co_detection_flag": bool(ch4_detected and oxidant_detected),
           "caveat": ("A co-detection screening heuristic (Lovelock 1965's disequilibrium "
                     "argument), not a life detection. Abiotic false-positive pathways "
                     "(e.g. photochemical O2 without a biological source, volcanic/geologic "
                     "CH4) are not evaluated here -- see Krissansen-Totton et al. (2018).")}


__all__ = [
    "BiosignatureFit", "BiosignaturePosteriorResult", "chi_squared",
    "fit_transmission_spectrum", "sample_posterior", "detection_significance",
    "disequilibrium_flag",
]
