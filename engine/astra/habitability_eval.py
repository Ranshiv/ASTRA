"""habitability_eval.py: self-consistency checks against the Kopparapu et
al. (2013) erratum's own published reference values, and Monte Carlo
propagation of stellar/planet parameter uncertainty through
`habitability.score` to a habitable-zone MEMBERSHIP PROBABILITY.

There is no noise process in `habitability.py` itself (it is closed-form
arithmetic on a catalog record), so this is not an injection-recovery
study in the sense `microlensing_eval.py` or `radio_variability_eval.py`
run. Instead it does the two things that are actually checkable: (1) does
this module reproduce the boundary distances the source paper itself
quotes for the Sun and for named example systems, within a stated
tolerance; (2) given a planet's catalog parameter ERRORS, what fraction
of plausible re-draws of (Teff, R*, a) still land inside the conservative
HZ -- `hz_membership_probability`, with a Wilson interval from
`significance._ci_binomial`, the same interval estimator
`radio_variability_eval.py` and `microlensing_eval.py` already use.

Not registered in `rpc.py` -- see `test_not_referenced_by_rpc` in
`tests/test_habitability_eval.py`, the same convention every other
`*_eval.py` module in this codebase follows.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import habitability as hab
from . import significance


class HabitabilityEvalError(ValueError):
    """A habitability validation study could not be run."""


# The Sun's own conservative HZ boundaries, as quoted directly in the
# Kopparapu et al. (2013) erratum's Table 1 (fetched and read this
# session): moist greenhouse 0.99 AU, maximum greenhouse 1.67 AU.
REFERENCE_SOLAR_SYSTEM = {
    "moist_greenhouse_au": 0.99,
    "runaway_greenhouse_au": 0.97,
    "recent_venus_au": 0.75,
    "maximum_greenhouse_au": 1.67,
    "early_mars_au": 1.77,
}


def reference_case_recovery(*, tolerance_au: float = 0.02) -> dict[str, Any]:
    """Compare `habitability.habitable_zone` for a solar-Teff/Lsun star
    against the paper's own quoted Table 1 values for the Sun."""
    sun = hab.StellarParameters(teff_k=hab.TEFF_SUN_K, radius_rsun=1.0, luminosity_lsun=1.0)
    zone = hab.habitable_zone(sun)
    diffs: dict[str, float] = {}
    within_tolerance: dict[str, bool] = {}
    for name, published_au in REFERENCE_SOLAR_SYSTEM.items():
        computed_au = zone[name]
        diff = float(computed_au - published_au)
        diffs[name] = round(diff, 6)
        within_tolerance[name] = bool(abs(diff) <= tolerance_au)
    return {
        "schema_version": hab.SCHEMA_VERSION,
        "computed": {k: zone[k] for k in REFERENCE_SOLAR_SYSTEM},
        "published": REFERENCE_SOLAR_SYSTEM,
        "diff_au": diffs,
        "within_tolerance": within_tolerance,
        "all_within_tolerance": bool(all(within_tolerance.values())),
        "tolerance_au": tolerance_au,
    }


def earth_esi_reference_case() -> dict[str, Any]:
    """Definitional anchor: ESI computed from Earth's own TRUE parameters
    (radius, density, escape velocity, true 288 K surface temperature)
    must equal 1.0 -- this bypasses the T_eq substitution entirely to
    check the ESI formula itself, not the substitution's honesty (that is
    `test_habitability.py`'s job)."""
    radius_term = hab._esi_term(hab.ESI_EARTH_RADIUS_REARTH, hab.ESI_EARTH_RADIUS_REARTH,
                                hab.ESI_WEIGHT_RADIUS)
    density_term = hab._esi_term(hab.ESI_EARTH_DENSITY_GCM3, hab.ESI_EARTH_DENSITY_GCM3,
                                 hab.ESI_WEIGHT_DENSITY)
    vesc_term = hab._esi_term(hab.ESI_EARTH_ESCAPE_VELOCITY_KMS, hab.ESI_EARTH_ESCAPE_VELOCITY_KMS,
                              hab.ESI_WEIGHT_ESCAPE_VELOCITY)
    temp_term = hab._esi_term(hab.ESI_EARTH_SURFACE_TEMP_K, hab.ESI_EARTH_SURFACE_TEMP_K,
                              hab.ESI_WEIGHT_SURFACE_TEMP)
    esi_interior = float((radius_term * density_term) ** 0.5)
    esi_surface = float((vesc_term * temp_term) ** 0.5)
    esi_global = float((esi_interior * esi_surface) ** 0.5)
    return {"esi_interior": round(esi_interior, 9), "esi_surface": round(esi_surface, 9),
            "esi_global": round(esi_global, 9), "is_unity": bool(abs(esi_global - 1.0) < 1e-9)}


def hz_membership_probability(star: hab.StellarParameters, planet: hab.PlanetParameters, *,
                              teff_err_k: float = 0.0, radius_err_rsun: float = 0.0,
                              semimajor_err_au: float = 0.0, n_trials: int = 2000,
                              seed: int = 42) -> dict[str, Any]:
    """Monte Carlo propagation of catalog parameter errors through
    `habitability.score`, reporting the fraction of draws landing in the
    conservative HZ with a Wilson binomial confidence interval."""
    if n_trials <= 0:
        raise HabitabilityEvalError("n_trials must be positive")
    if planet.semi_major_axis_au is None:
        raise HabitabilityEvalError("planet.semi_major_axis_au is required")
    rng = np.random.default_rng(seed)
    teff_draws = star.teff_k + rng.normal(0.0, max(teff_err_k, 0.0), size=n_trials)
    radius_draws = star.radius_rsun + rng.normal(0.0, max(radius_err_rsun, 0.0), size=n_trials)
    a_draws = planet.semi_major_axis_au + rng.normal(0.0, max(semimajor_err_au, 0.0), size=n_trials)

    successes = 0
    n_rejected = 0
    for teff, radius, a in zip(teff_draws, radius_draws, a_draws):
        if teff <= 0 or radius <= 0 or a <= 0:
            n_rejected += 1
            continue
        try:
            trial_star = hab.StellarParameters(teff_k=float(teff), radius_rsun=float(radius))
            trial_planet = hab.PlanetParameters(semi_major_axis_au=float(a))
            result = hab.score(trial_star, trial_planet)
        except hab.HabitabilityError:
            n_rejected += 1
            continue
        if result["in_conservative_hz"]:
            successes += 1
    n_valid = n_trials - n_rejected
    probability = successes / n_valid if n_valid > 0 else None
    ci95 = significance._ci_binomial(successes, n_valid) if n_valid > 0 else None
    return {
        "schema_version": hab.SCHEMA_VERSION,
        "n_trials": int(n_trials),
        "n_rejected": int(n_rejected),
        "n_valid": int(n_valid),
        "successes": int(successes),
        "hz_membership_probability": round(probability, 6) if probability is not None else None,
        "ci95": ci95,
        "seed": seed,
    }


def run_validation_study(*, teff_err_k: float = 50.0, radius_err_rsun: float = 0.02,
                         semimajor_err_au: float = 0.01, n_trials: int = 2000,
                         seed: int = 42) -> dict[str, Any]:
    """End-to-end driver: reference-case recovery + Earth ESI anchor +
    HZ-membership Monte Carlo for a Sun/Earth-like system."""
    sun = hab.StellarParameters(teff_k=hab.TEFF_SUN_K, radius_rsun=1.0, luminosity_lsun=1.0)
    earth = hab.PlanetParameters(radius_rearth=1.0, mass_mearth=1.0, semi_major_axis_au=1.0)
    return {
        "schema_version": hab.SCHEMA_VERSION,
        "reference_case": reference_case_recovery(),
        "earth_esi_reference": earth_esi_reference_case(),
        "hz_membership": hz_membership_probability(
            sun, earth, teff_err_k=teff_err_k, radius_err_rsun=radius_err_rsun,
            semimajor_err_au=semimajor_err_au, n_trials=n_trials, seed=seed),
    }


__all__ = [
    "HabitabilityEvalError", "REFERENCE_SOLAR_SYSTEM", "reference_case_recovery",
    "earth_esi_reference_case", "hz_membership_probability", "run_validation_study",
]
