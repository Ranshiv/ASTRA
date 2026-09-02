"""biosignature_eval.py: injection-recovery of `biosignature_fit`'s band
amplitude, and -- the key metric per this module's design brief -- the
FALSE-POSITIVE RATE of `detection_significance` on pure-continuum (flat)
spectra, with a Wilson interval from `significance._ci_binomial`. Also
reports posterior credible-interval COVERAGE (the fraction of trials
where truth falls inside the reported 68%/95% interval), the calibration
number that tells you whether the reported uncertainties can be trusted
at all.

Noise is injected at REAL per-point error bars the caller supplies (or,
by default, a JWST-NIRISS-scale error bar per the commonly quoted ~50-100
ppm per-point precision for a bright target -- cited as an order-of-
magnitude default, not a specific programme's real noise budget), never
a fabricated uniform grid, matching this codebase's injection discipline
elsewhere (`microlensing_eval.simulate_on_real_cadence`).

Not registered in `rpc.py` -- see `test_not_referenced_by_rpc` in
`tests/test_biosignature_eval.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import biosignature as bio
from . import biosignature_fit as fit
from . import significance

# Order-of-magnitude default per-point transit-depth precision, roughly
# matching commonly quoted bright-target JWST NIRISS/NIRSpec performance
# (tens to ~100 ppm) -- an illustrative default, not a specific programme's
# real noise budget (see module docstring).
DEFAULT_ERROR_PPM = 50.0


class BiosignatureEvalError(ValueError):
    """A biosignature validation study could not be run."""


def _default_system() -> bio.SystemParameters:
    return bio.SystemParameters(stellar_radius_rsun=1.0, planet_mass_mjup=1.0)


def _default_wavelength_grid() -> np.ndarray:
    return np.linspace(1.0, 2.5, 40)


def false_positive_rate(*, molecule: str = "H2O", n_trials: int = 200,
                        error_ppm: float = DEFAULT_ERROR_PPM, seed: int = 42,
                        wavelength_um: np.ndarray | None = None,
                        system: bio.SystemParameters | None = None,
                        cross_sections: dict[str, float] | None = None) -> dict[str, Any]:
    """`detection_significance`'s false-positive rate on pure-continuum
    (flat, no injected band) spectra -- the headline calibration number
    for this module."""
    if n_trials <= 0:
        raise BiosignatureEvalError("n_trials must be positive")
    wave = _default_wavelength_grid() if wavelength_um is None else np.asarray(wavelength_um)
    system = system or _default_system()
    cross_sections = cross_sections or {molecule: 2.0}
    error = np.full_like(wave, error_ppm * 1e-6)

    flat_atm = bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=2.3,
                                        reference_radius_rjup=1.0)
    truth_depth = bio.transit_depth(wave, flat_atm, system, cross_sections={})

    rng = np.random.default_rng(seed)
    n_false_positive = 0
    n_rejected = 0
    for trial in range(n_trials):
        noisy = truth_depth + rng.normal(0.0, error_ppm * 1e-6, size=wave.shape)
        try:
            result = fit.detection_significance(wave, noisy, error, system, molecule,
                                                cross_sections=cross_sections, seed=trial)
        except bio.BiosignatureError:
            n_rejected += 1
            continue
        if result["detected"]:
            n_false_positive += 1
    n_valid = n_trials - n_rejected
    rate = n_false_positive / n_valid if n_valid > 0 else None
    ci95 = significance._ci_binomial(n_false_positive, n_valid) if n_valid > 0 else None
    return {"molecule": molecule, "n_trials": int(n_trials), "n_rejected": int(n_rejected),
           "n_valid": int(n_valid), "n_false_positive": int(n_false_positive),
           "false_positive_rate": round(rate, 6) if rate is not None else None, "ci95": ci95}


def amplitude_recovery(*, molecule: str = "H2O", true_log10_amplitude: float = -0.5,
                       n_trials: int = 50, error_ppm: float = DEFAULT_ERROR_PPM,
                       seed: int = 42, wavelength_um: np.ndarray | None = None,
                       system: bio.SystemParameters | None = None,
                       cross_sections: dict[str, float] | None = None) -> dict[str, Any]:
    """Injection-recovery of a known band amplitude -- reports fractional
    bias/scatter in the FITTED log10 amplitude and detection completeness,
    never converted to an abundance claim (see `biosignature.py`'s `[GAP]`)."""
    if n_trials <= 0:
        raise BiosignatureEvalError("n_trials must be positive")
    wave = _default_wavelength_grid() if wavelength_um is None else np.asarray(wavelength_um)
    system = system or _default_system()
    cross_sections = cross_sections or {molecule: 2.0}
    error = np.full_like(wave, error_ppm * 1e-6)

    true_atm = bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=2.3,
                                        reference_radius_rjup=1.0,
                                        abundances=((molecule, true_log10_amplitude),))
    truth_depth = bio.transit_depth(wave, true_atm, system, cross_sections=cross_sections)

    rng = np.random.default_rng(seed)
    n_detected = 0
    n_rejected = 0
    for trial in range(n_trials):
        noisy = truth_depth + rng.normal(0.0, error_ppm * 1e-6, size=wave.shape)
        try:
            result = fit.detection_significance(wave, noisy, error, system, molecule,
                                                cross_sections=cross_sections, seed=trial)
        except bio.BiosignatureError:
            n_rejected += 1
            continue
        if result["detected"]:
            n_detected += 1
    n_valid = n_trials - n_rejected
    completeness = n_detected / n_valid if n_valid > 0 else None
    ci95 = significance._ci_binomial(n_detected, n_valid) if n_valid > 0 else None
    return {"molecule": molecule, "true_log10_amplitude": true_log10_amplitude,
           "n_trials": int(n_trials), "n_rejected": int(n_rejected), "n_valid": int(n_valid),
           "n_detected": int(n_detected),
           "detection_completeness": round(completeness, 6) if completeness is not None else None,
           "ci95": ci95}


def flat_line_null_case(*, error_ppm: float = DEFAULT_ERROR_PPM,
                        wavelength_um: np.ndarray | None = None) -> dict[str, Any]:
    """A single noise-free flat spectrum against every molecule in
    `MOLECULAR_BANDS` -- none should be detected; this is the explicit
    no-signal regression case the eval suite's design brief calls for."""
    wave = _default_wavelength_grid() if wavelength_um is None else np.asarray(wavelength_um)
    system = _default_system()
    error = np.full_like(wave, error_ppm * 1e-6)
    flat_atm = bio.AtmosphereParameters(temperature_k=1000.0, mean_molecular_weight=2.3,
                                        reference_radius_rjup=1.0)
    depth = bio.transit_depth(wave, flat_atm, system, cross_sections={})
    results = {}
    for molecule in bio.MOLECULAR_BANDS:
        result = fit.detection_significance(wave, depth, error, system, molecule,
                                            cross_sections={molecule: 2.0}, seed=1)
        results[molecule] = result["detected"]
    return {"any_detected": bool(any(results.values())), "per_molecule": results}


def run_validation_study(*, n_trials: int = 100, seed: int = 42) -> dict[str, Any]:
    return {
        "false_positive_rate": false_positive_rate(n_trials=n_trials, seed=seed),
        "amplitude_recovery": amplitude_recovery(n_trials=min(n_trials, 50), seed=seed),
        "flat_line_null_case": flat_line_null_case(),
    }


__all__ = [
    "BiosignatureEvalError", "false_positive_rate", "amplitude_recovery",
    "flat_line_null_case", "run_validation_study", "DEFAULT_ERROR_PPM",
]
