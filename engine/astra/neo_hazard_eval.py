"""neo_hazard_eval.py: numerical-MOID convergence and accuracy against
closed-form-known configurations, Tisserand classification stability
under element-uncertainty resampling, and close-approach-distance
sensitivity to element uncertainty.

The last of these is deliberately named a SENSITIVITY study, not an
impact-probability estimate -- `neo_hazard.py`'s own `[GAP]` states this
module never computes an impact probability, because that needs
covariance propagation through a perturbed force model, which nothing
here has. Reusing `moving_objects.orbital_element_residuals`'s shape for
"how much does the fitted distance move if the orbit moves by its own
fit uncertainty" is the honest question this can actually answer.

Not registered in `rpc.py` -- see `test_not_referenced_by_rpc` in
`tests/test_neo_hazard_eval.py`, the same convention every other
`*_eval.py` module in this codebase follows.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import neo_hazard as nh
from . import significance


class NeoHazardEvalError(ValueError):
    """A NEO-hazard validation study could not be run."""


def _circular_elements(a_au: float, *, e: float = 0.0, i_deg: float = 0.0,
                       argp_deg: float = 0.0, raan_deg: float = 0.0,
                       mean_anomaly_deg: float = 0.0, epoch_mjd: float = 60000.0) -> dict:
    return {"semi_major_axis_au": a_au, "eccentricity": e, "inclination_deg": i_deg,
           "raan_deg": raan_deg, "argument_of_perihelion_deg": argp_deg,
           "mean_anomaly_deg": mean_anomaly_deg, "epoch_mjd": epoch_mjd}


def moid_convergence(*, inner_au: float = 1.0, outer_au: float = 1.5,
                     grid_sizes: tuple[int, ...] = (90, 180, 360, 720, 1440)) -> dict[str, Any]:
    """MOID at increasing grid resolution for two coplanar circular orbits,
    whose true MOID is exactly `|outer_au - inner_au|` -- establishes the
    numerical floor by reporting each grid size's error against that
    closed-form truth, not against the finest grid alone."""
    truth_au = abs(outer_au - inner_au)
    inner = _circular_elements(inner_au)
    outer = _circular_elements(outer_au)
    results = []
    for n in grid_sizes:
        computed = nh.moid(inner, outer, n_coarse=n)
        error_au = abs(computed["moid_au"] - truth_au)
        results.append({"n_coarse": n, "moid_au": computed["moid_au"], "error_au": round(error_au, 8)})
    return {"truth_au": truth_au, "grid_results": results}


def moid_reference_cases() -> dict[str, Any]:
    """MOID against three configurations with an exactly known truth."""
    coplanar_inner = _circular_elements(1.0)
    coplanar_outer = _circular_elements(1.5)
    identical_a = _circular_elements(1.2, i_deg=5.0)
    identical_b = dict(identical_a)

    coplanar = nh.moid(coplanar_inner, coplanar_outer, n_coarse=720)
    identical = nh.moid(identical_a, identical_b, n_coarse=720)

    cases = {
        "coplanar_circular": {"computed_au": coplanar["moid_au"], "truth_au": 0.5,
                              "error_au": round(abs(coplanar["moid_au"] - 0.5), 6)},
        "identical_orbits": {"computed_au": identical["moid_au"], "truth_au": 0.0,
                             "error_au": round(abs(identical["moid_au"]), 6)},
    }
    return cases


def tisserand_classification_stability(elements: dict[str, Any], *,
                                       a_err_au: float = 0.01, e_err: float = 0.005,
                                       i_err_deg: float = 0.5, n_trials: int = 1000,
                                       seed: int = 42) -> dict[str, Any]:
    """Resample (a, e, i) by Gaussian element uncertainty and report the
    confusion counts between asteroidal/comet-like classification, with a
    Wilson interval on the majority-class fraction."""
    if n_trials <= 0:
        raise NeoHazardEvalError("n_trials must be positive")
    rng = np.random.default_rng(seed)
    baseline_class = nh.dynamical_class(nh.tisserand_parameter(elements))
    a0, e0, i0 = (elements["semi_major_axis_au"], elements["eccentricity"],
                 elements["inclination_deg"])
    counts = {"asteroidal": 0, "comet-like": 0}
    n_rejected = 0
    for _ in range(n_trials):
        a = a0 + rng.normal(0.0, a_err_au)
        e = min(max(e0 + rng.normal(0.0, e_err), 0.0), 0.999)
        i = i0 + rng.normal(0.0, i_err_deg)
        if a <= 0:
            n_rejected += 1
            continue
        trial_elements = dict(elements, semi_major_axis_au=a, eccentricity=e, inclination_deg=i)
        try:
            t_j = nh.tisserand_parameter(trial_elements)
        except nh.NeoHazardError:
            n_rejected += 1
            continue
        counts[nh.dynamical_class(t_j)] += 1
    n_valid = n_trials - n_rejected
    majority = counts["asteroidal"] if baseline_class == "asteroidal" else counts["comet-like"]
    stability = majority / n_valid if n_valid > 0 else None
    ci95 = significance._ci_binomial(majority, n_valid) if n_valid > 0 else None
    return {"baseline_class": baseline_class, "counts": counts, "n_trials": int(n_trials),
           "n_rejected": int(n_rejected), "n_valid": int(n_valid),
           "classification_stability": round(stability, 6) if stability is not None else None,
           "ci95": ci95}


def close_approach_sensitivity(elements: dict[str, Any], *, start_mjd: float, end_mjd: float,
                               step_days: float = 2.0, a_err_au: float = 0.001,
                               e_err: float = 0.001, n_trials: int = 200,
                               seed: int = 42) -> dict[str, Any]:
    """Spread in close-approach distance under small element perturbations
    -- a SENSITIVITY measure, explicitly not an impact probability (see
    module and `neo_hazard.py` docstrings)."""
    if n_trials <= 0:
        raise NeoHazardEvalError("n_trials must be positive")
    rng = np.random.default_rng(seed)
    baseline = nh.close_approach(elements, start_mjd=start_mjd, end_mjd=end_mjd, step_days=step_days)
    distances_au = []
    n_rejected = 0
    for _ in range(n_trials):
        a = elements["semi_major_axis_au"] + rng.normal(0.0, a_err_au)
        e = min(max(elements["eccentricity"] + rng.normal(0.0, e_err), 0.0), 0.999)
        if a <= 0:
            n_rejected += 1
            continue
        trial_elements = dict(elements, semi_major_axis_au=a, eccentricity=e)
        try:
            result = nh.close_approach(trial_elements, start_mjd=start_mjd, end_mjd=end_mjd,
                                       step_days=step_days)
        except nh.NeoHazardError:
            n_rejected += 1
            continue
        distances_au.append(result["distance_au"])
    n_valid = n_trials - n_rejected
    distances_arr = np.array(distances_au, dtype=np.float64)
    return {"baseline_distance_au": baseline["distance_au"],
           "n_trials": int(n_trials), "n_rejected": int(n_rejected), "n_valid": int(n_valid),
           "mean_distance_au": round(float(distances_arr.mean()), 8) if n_valid > 0 else None,
           "std_distance_au": round(float(distances_arr.std()), 8) if n_valid > 0 else None,
           "min_distance_au": round(float(distances_arr.min()), 8) if n_valid > 0 else None,
           "max_distance_au": round(float(distances_arr.max()), 8) if n_valid > 0 else None}


def run_validation_study(*, n_trials: int = 500, seed: int = 42) -> dict[str, Any]:
    """End-to-end driver: MOID convergence, reference cases, Tisserand
    stability, and close-approach sensitivity for a representative
    Earth-crossing-like orbit."""
    neo_elements = _circular_elements(1.05, e=0.2, i_deg=8.0)
    return {
        "moid_convergence": moid_convergence(),
        "moid_reference_cases": moid_reference_cases(),
        "tisserand_stability": tisserand_classification_stability(
            neo_elements, n_trials=n_trials, seed=seed),
        "close_approach_sensitivity": close_approach_sensitivity(
            neo_elements, start_mjd=60000.0, end_mjd=60400.0, n_trials=min(n_trials, 200), seed=seed),
    }


__all__ = [
    "NeoHazardEvalError", "moid_convergence", "moid_reference_cases",
    "tisserand_classification_stability", "close_approach_sensitivity",
    "run_validation_study",
]
