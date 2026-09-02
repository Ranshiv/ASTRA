"""asteroseismology_eval.py: exact round-trip verification of the scaling-
relation inversion, and injection-recovery of `measure` on synthetic
oscillation combs built on REAL light-curve cadence -- never a fabricated
uniform grid, matching every other `*_eval.py` module's injection
discipline in this codebase (e.g. `microlensing_eval.py`'s
`simulate_on_real_cadence`).

The headline metric of `measurement_recovery` is `n_aliased`: the count
of trials where `estimate_delta_nu`'s autocorrelation search locked onto
the wrong integer multiple of the true Dnu (the well-known x2/0.5x
aliasing failure mode of Dnu measurement). `asteroseismology.py`'s
`[GAP]` states this failure mode exists and is not corrected for; this
module MEASURES its rate rather than assuming it away.

Not registered in `rpc.py` -- see `test_not_referenced_by_rpc` in
`tests/test_asteroseismology_eval.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import asteroseismology as ast
from . import significance


class AsteroseismologyEvalError(ValueError):
    """An asteroseismology validation study could not be run."""


def round_trip_recovery(*, mass_grid: tuple[float, ...] = (0.8, 1.0, 1.2, 2.0),
                        radius_grid: tuple[float, ...] = (0.9, 1.0, 1.5, 10.0),
                        teff_grid: tuple[float, ...] = (4800.0, 5777.0, 6200.0)
                        ) -> dict[str, Any]:
    """Exact algebraic round-trip: predict (numax, Dnu) from (M, R, Teff),
    invert back, and report the maximum fractional error over the grid --
    should be at the level of floating-point precision; anything larger
    indicates an algebra bug in `solve_scaling_relations`."""
    max_radius_error = 0.0
    max_mass_error = 0.0
    n_cases = 0
    for mass in mass_grid:
        for radius in radius_grid:
            for teff in teff_grid:
                numax = ast.predict_numax(mass, radius, teff)
                dnu = ast.predict_delta_nu(mass, radius)
                seismic = ast.SeismicParameters(numax_uhz=numax, delta_nu_uhz=dnu, teff_k=teff)
                solution = ast.solve_scaling_relations(seismic)
                max_radius_error = max(max_radius_error,
                                       abs(solution.radius_rsun - radius) / radius)
                max_mass_error = max(max_mass_error, abs(solution.mass_msun - mass) / mass)
                n_cases += 1
    return {"n_cases": n_cases, "max_radius_fractional_error": max_radius_error,
           "max_mass_fractional_error": max_mass_error,
           "algebraically_exact": bool(max_radius_error < 1e-9 and max_mass_error < 1e-9)}


def _synthetic_curve(numax_uhz: float, *, time_days: np.ndarray, amplitude_snr: float,
                     noise_sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dnu_true = ast.STELLO_DNU_COEFF * numax_uhz ** ast.STELLO_DNU_EXPONENT
    lo, hi = ast.envelope_window(numax_uhz)
    sigma = (hi - lo) / 2.355
    flux = np.zeros_like(time_days)
    for f in np.arange(lo, hi, dnu_true):
        gauss_amplitude = amplitude_snr * np.exp(-0.5 * ((f - numax_uhz) / sigma) ** 2)
        freq_per_day = f / ast.UHZ_PER_DAY_INVERSE
        phase = rng.uniform(0.0, 2.0 * np.pi)
        flux += gauss_amplitude * np.sin(2.0 * np.pi * freq_per_day * time_days + phase)
    flux += rng.normal(0.0, noise_sigma, time_days.shape[0])
    return flux, dnu_true


def measurement_recovery(*, real_cadence_days: np.ndarray | None = None,
                         numax_grid: tuple[float, ...] = (500.0, 1200.0, 2800.0),
                         amplitude_snr_grid: tuple[float, ...] = (0.5, 1.0, 2.0),
                         noise_sigma: float = 0.05, n_trials_per_cell: int = 5,
                         seed: int = 42) -> dict[str, Any]:
    """Injection-recovery of `measure` on a synthetic oscillation comb
    riding real (or, if none supplied, a dense 2-minute default) cadence,
    reporting per-numax/per-SNR fractional bias/scatter and the `Dnu`
    aliasing rate."""
    if n_trials_per_cell <= 0:
        raise AsteroseismologyEvalError("n_trials_per_cell must be positive")
    if real_cadence_days is None:
        real_cadence_days = np.arange(20000) * (2.0 / 1440.0)
    time_days = np.asarray(real_cadence_days, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    n_aliased = 0
    n_rejected = 0
    n_total = 0
    rng_seed = seed
    for numax_true in numax_grid:
        for amplitude_snr in amplitude_snr_grid:
            numax_errors: list[float] = []
            dnu_errors: list[float] = []
            for trial in range(n_trials_per_cell):
                rng_seed += 1
                n_total += 1
                flux, dnu_true = _synthetic_curve(numax_true, time_days=time_days,
                                                  amplitude_snr=amplitude_snr,
                                                  noise_sigma=noise_sigma, seed=rng_seed)
                result = ast.measure(time_days, flux)
                if result["numax_uhz"] is None or result["delta_nu_uhz"] is None:
                    n_rejected += 1
                    continue
                numax_errors.append((result["numax_uhz"] - numax_true) / numax_true)
                dnu_recovered = result["delta_nu_uhz"]
                dnu_errors.append((dnu_recovered - dnu_true) / dnu_true)
                # An aliased recovery lands near an integer multiple or
                # sub-multiple of the truth rather than near 1.0.
                ratio = dnu_recovered / dnu_true
                if not (0.85 <= ratio <= 1.15) and any(
                        abs(ratio - k) < 0.1 or abs(ratio - 1.0 / k) < 0.1 for k in (2, 3)):
                    n_aliased += 1
            if numax_errors:
                rows.append({
                    "numax_true_uhz": numax_true, "amplitude_snr": amplitude_snr,
                    "n_recovered": len(numax_errors),
                    "numax_fractional_bias": round(float(np.mean(numax_errors)), 6),
                    "numax_fractional_scatter": round(float(np.std(numax_errors)), 6),
                    "dnu_fractional_bias": round(float(np.mean(dnu_errors)), 6),
                    "dnu_fractional_scatter": round(float(np.std(dnu_errors)), 6),
                })
    n_valid = n_total - n_rejected
    aliasing_rate = n_aliased / n_valid if n_valid > 0 else None
    ci95 = significance._ci_binomial(n_aliased, n_valid) if n_valid > 0 else None
    return {"cells": rows, "n_trials": n_total, "n_rejected": n_rejected, "n_valid": n_valid,
           "n_aliased": n_aliased, "dnu_aliasing_rate": round(aliasing_rate, 6) if aliasing_rate is not None else None,
           "ci95": ci95}


def run_validation_study(*, n_trials_per_cell: int = 3, seed: int = 42) -> dict[str, Any]:
    return {"round_trip": round_trip_recovery(),
           "measurement_recovery": measurement_recovery(n_trials_per_cell=n_trials_per_cell, seed=seed)}


__all__ = [
    "AsteroseismologyEvalError", "round_trip_recovery", "measurement_recovery",
    "run_validation_study",
]
