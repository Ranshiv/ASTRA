"""Coverage of simulated rate posteriors, and bias under anchor-survey
changes -- the two metrics roadmap item 32 names -- split from
`population_rate.py` purely to keep each file under this project's
500-line guideline (same `stellar_manifold.py`/`stellar_manifold_eval.py`
split rationale, not an independent module).

`evaluate_rate_posterior_coverage` reuses `microlensing_eval.CoverageTrial`/
`posterior_coverage` UNCHANGED -- the same "simulate a truth, refit, count
how often the credible region contains it" pattern that module already
established for microlensing parameters, applied here to
`mu_log_rate` instead.

`evaluate_anchor_survey_bias` demonstrates the specific, real failure mode
`crossmatch.grouping_bias_report`'s own docstring already warns about:
switching `anchor_survey` "controls the population denominator; it does
not remove the underlying selection bias." Two arms are compared on the
SAME synthetic multi-survey population: a "corrected" arm
(`population_rate.anchor_survey_rate_sweep`, which divides each anchor's
own raw detected count by ITS OWN real completeness) recovers close to
the true rate regardless of which survey anchors the grouping; a "naive"
arm (dividing every anchor's raw count by one SHARED assumed completeness,
the mistake a careless multi-survey analysis can make) does not -- its
rate estimate swings with the anchor choice. This is the honest content of
"bias under anchor-survey changes": not that `group_sources` itself is
wrong, but that skipping a per-survey selection-function correction after
changing the anchor silently biases the inferred rate.

Both studies validated on SYNTHETIC ground truth only -- the same
"mechanism validated on synthetic data, not yet run at real Stage-B scale"
caveat every eval module in this family states.
"""

from __future__ import annotations

import math

import numpy as np

from .microlensing_eval import CoverageTrial, posterior_coverage
from .population_rate import (
    PopulationRateError, SurveyFootprint, Stratum,
    anchor_survey_rate_sweep, fit_hierarchical_rate, poisson_rate_credible_interval,
)
from .surveys.base import SourceRef


def synthesize_population_strata(rng: np.random.Generator, *, n_strata: int = 5,
                                 true_mu_log_rate: float, true_sigma_log_rate: float,
                                 footprint_area_range: tuple[float, float] = (10.0, 500.0),
                                 baseline_days_range: tuple[float, float] = (30.0, 365.0),
                                 completeness_range: tuple[float, float] = (0.3, 0.9)
                                 ) -> tuple[list[Stratum], dict]:
    """Draws `n_strata` strata from the hierarchical model
    `population_rate.fit_hierarchical_rate` fits, with random footprints
    and completeness values, returning the strata plus the true generating
    parameters as ground truth."""
    if n_strata < 1:
        raise PopulationRateError(f"n_strata must be at least 1, got {n_strata}")

    strata: list[Stratum] = []
    true_rates: dict[str, float] = {}
    z_values: dict[str, float] = {}
    for index in range(n_strata):
        label = f"stratum_{index}"
        z = float(rng.normal())
        rate = math.exp(true_mu_log_rate + true_sigma_log_rate * z)
        footprint = SurveyFootprint(
            survey=label, area_deg2=float(rng.uniform(*footprint_area_range)),
            baseline_days=float(rng.uniform(*baseline_days_range)))
        completeness = float(rng.uniform(*completeness_range))
        expected = rate * footprint.exposure_deg2_days * completeness
        detected = int(rng.poisson(expected))
        strata.append(Stratum(label=label, footprint=footprint,
                              completeness=completeness, detected=detected))
        true_rates[label] = rate
        z_values[label] = z

    truth = {"mu_log_rate": true_mu_log_rate, "sigma_log_rate": true_sigma_log_rate,
             "rates": true_rates, "z": z_values}
    return strata, truth


def evaluate_rate_posterior_coverage(n_trials: int = 200, levels: tuple[float, ...] = (0.68, 0.9),
                                     seed: int = 42, *, true_mu_log_rate: float = -3.0,
                                     true_sigma_log_rate: float = 0.5, n_strata: int = 5,
                                     n_steps: int = 3000, n_walkers: int = 32,
                                     **synth_kwargs) -> dict:
    """Empirical coverage of `fit_hierarchical_rate`'s `mu_log_rate`
    credible interval across `n_trials` independent synthetic populations
    -- this item's own "coverage of simulated rate posteriors" metric."""
    if n_trials < 1:
        raise PopulationRateError(f"n_trials must be at least 1, got {n_trials}")

    rng = np.random.default_rng(seed)
    trials: list[CoverageTrial] = []
    for trial_index in range(n_trials):
        strata, truth = synthesize_population_strata(
            rng, n_strata=n_strata, true_mu_log_rate=true_mu_log_rate,
            true_sigma_log_rate=true_sigma_log_rate, **synth_kwargs)
        fit = fit_hierarchical_rate(
            strata, n_walkers=n_walkers, n_steps=n_steps, levels=levels, seed=seed + trial_index)
        trials.append(CoverageTrial(
            truth={"mu_log_rate": truth["mu_log_rate"]},
            intervals={"mu_log_rate": fit.intervals["mu_log_rate"]},
            samples=None, names=("mu_log_rate",)))

    return posterior_coverage(trials, levels=levels)


def evaluate_anchor_survey_bias(rng: np.random.Generator, *, true_rate_deg2_day: float = 0.002,
                                surveys: tuple[str, ...] = ("wide_shallow", "narrow_deep"),
                                completeness_by_survey: dict[str, float] | None = None,
                                footprint_area_deg2: float = 100.0,
                                baseline_days: float = 180.0,
                                naive_completeness_survey: str | None = None) -> dict:
    """Builds one synthetic true population, observed by several surveys
    with DIFFERENT known completeness values, and compares a "corrected"
    (per-anchor completeness) rate estimate against a "naive" (one shared
    completeness) one across anchor choices -- see module docstring."""
    completeness_by_survey = completeness_by_survey or {"wide_shallow": 0.9, "narrow_deep": 0.4}
    if set(surveys) != set(completeness_by_survey):
        raise PopulationRateError("surveys and completeness_by_survey keys must match")

    exposure = footprint_area_deg2 * baseline_days
    n_true = max(1, int(round(true_rate_deg2_day * exposure)))
    # A fixed (not Poisson-sampled) true population size, deliberately:
    # this metric measures denominator/anchor bias, not sampling noise.
    true_ra = rng.uniform(180.0, 181.0, size=n_true)
    true_dec = rng.uniform(-0.5, 0.5, size=n_true)

    by_survey: dict[str, list[SourceRef]] = {}
    for survey in surveys:
        keep = rng.random(n_true) < completeness_by_survey[survey]
        by_survey[survey] = [
            SourceRef(survey=survey, object_id=f"{survey}_{i}",
                     ra_deg=float(true_ra[i]), dec_deg=float(true_dec[i]))
            for i in range(n_true) if keep[i]
        ]

    footprints = {survey: SurveyFootprint(survey=survey, area_deg2=footprint_area_deg2,
                                          baseline_days=baseline_days) for survey in surveys}
    sweep = anchor_survey_rate_sweep(by_survey, footprints, completeness_by_survey, list(surveys))

    naive_survey = naive_completeness_survey or surveys[0]
    naive_completeness = completeness_by_survey[naive_survey]
    naive_rate_by_anchor: dict[str, float] = {}
    for anchor in surveys:
        detected = sweep["anchors"][anchor]["detected"]
        naive_exposure = footprint_area_deg2 * baseline_days * naive_completeness
        naive_rate_by_anchor[anchor] = poisson_rate_credible_interval(detected, naive_exposure)[0]

    corrected_rate_by_anchor = {anchor: entry["rate_point"] for anchor, entry in sweep["anchors"].items()}

    return {
        "true_rate_deg2_day": true_rate_deg2_day, "n_true_objects": n_true,
        "corrected_rate_by_anchor": corrected_rate_by_anchor,
        "naive_rate_by_anchor": naive_rate_by_anchor,
        "corrected_rate_spread": max(corrected_rate_by_anchor.values()) - min(corrected_rate_by_anchor.values()),
        "naive_rate_spread": max(naive_rate_by_anchor.values()) - min(naive_rate_by_anchor.values()),
    }


__all__ = [
    "synthesize_population_strata", "evaluate_rate_posterior_coverage",
    "evaluate_anchor_survey_bias",
]
