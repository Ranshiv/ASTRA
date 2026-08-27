"""Selection-function and population-rate inference (roadmap item 32, P1).

Infers a population occurrence rate (events per deg^2 per day) from
detected counts, correcting for the two things that turn a raw count into
a biased rate estimate: how much sky and time a survey actually covered
(`SurveyFootprint`), and how completely it recovers a real event there
(reused unchanged from `significance.evaluate_selection`'s Wilson-interval
completeness cells, or `significance.fit_selection_model`'s continuous
recovery-probability surface).

Two estimators are provided, both real and citable, neither invented:

**Closed-form**: `poisson_rate_credible_interval` is the standard Jeffreys-
prior conjugate Gamma posterior for a Poisson rate (`detected ~
Poisson(rate * exposure)`; the Jeffreys prior for a Poisson rate is
`Gamma(1/2, 0)`, giving a `Gamma(detected + 1/2, exposure)` posterior --
see Gelman, Carlin, Stern & Rubin, *Bayesian Data Analysis*, 3rd ed., Ch.
2, for the general conjugate-Gamma-for-Poisson-rate result). Fast, no
sampler needed -- used both as an MCMC-walker starting-point heuristic
below and as the estimator behind `anchor_survey_rate_sweep`, which needs
to be cheap to rerun once per candidate anchor.

**Hierarchical (the roadmap item's own named method)**: `fit_hierarchical_
rate` is a standard partial-pooling Poisson rate model in the non-centered
parametrization (the numerically well-behaved form for MCMC -- Gelman &
Hill 2007, *Data Analysis Using Regression and Multilevel/Hierarchical
Models*, Ch. 12; Betancourt & Girolami 2015 on non-centered
parametrizations). For each stratum `i` (one survey, or one magnitude/
amplitude bin):

    z_i        ~ Normal(0, 1)
    rate_i     = exp(mu_log_rate + sigma_log_rate * z_i)
    expected_i = rate_i * exposure_i * completeness_i
    detected_i ~ Poisson(expected_i)

`expected_i` uses the standard Poisson-thinning identity: a Poisson
process observed with a fixed detection probability is itself Poisson at
the thinned rate, so `completeness_i` multiplies straight into the
expectation rather than needing its own likelihood term. `mu_log_rate`
(population mean) and `sigma_log_rate` (between-stratum spread -- the
actual hierarchical/pooling hyperparameter) are the parameters of
interest; each `z_i` is a per-stratum nuisance parameter, giving
`2 + n_strata` total dimensions. This scales to a handful of strata, not
hundreds -- a real, stated scope limit, not a hidden one.

Follows `microlensing_fit.sample_posterior`'s exact template: `emcee` is
gated behind `_require_emcee()` (already a `research`-extra dependency,
used unchanged, no new dependency added here), flat priors within bounds
on `mu_log_rate`/`log(sigma_log_rate)`, walkers jittered around a starting
guess (here, the closed-form per-stratum point estimate rather than a
least-squares fit), convergence reported (not assumed) via emcee's own
"chain at least 50 autocorrelation times long" rule of thumb, and a
`{name: {level_str: [low, high]}}` interval shape matching `PosteriorResult`.

`anchor_survey_rate_sweep` reuses `crossmatch.group_sources`/
`grouping_bias_report` UNCHANGED: `grouping_bias_report`'s own docstring
already states switching the anchor "controls the population denominator;
it does not remove the underlying selection bias" -- this function makes
that concrete by rerunning the closed-form rate estimator once per
candidate anchor.

No cadence-log data structure is built here: `significance.
evaluate_selection` already accepts `cadence_days` as a stratification
dimension, so cadence's effect on completeness is captured by running
injection-recovery stratified by `cadence_days` upstream of this module,
not by a bespoke cadence log. `SurveyFootprint` is deliberately a single
effective sky area plus time baseline, not a patchy-coverage HEALPix map
-- `healpix_common.py` is where a future full-coverage-map version would
hang; not attempted here.

Like every other opt-in research module in this codebase, NOT wired into
`rpc.py`, `scoring.WEIGHTS`, or `evidence.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import crossmatch
from .surveys.base import SourceRef


class PopulationRateError(ValueError):
    """A footprint, stratum, or rate-inference input/computation was invalid."""


def _require_emcee():
    try:
        import emcee
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise PopulationRateError(
            "emcee is not installed; install the 'research' extra "
            "(pip install .[research]) to fit a hierarchical rate. "
            "poisson_rate_credible_interval() itself needs no extra dependency."
        ) from exc
    return emcee


@dataclass(frozen=True)
class SurveyFootprint:
    """A single effective sky area and time baseline for one survey --
    not a patchy-coverage sky map (see module docstring)."""

    survey: str
    area_deg2: float
    baseline_days: float

    def __post_init__(self) -> None:
        if self.area_deg2 <= 0:
            raise PopulationRateError(f"area_deg2 must be positive, got {self.area_deg2}")
        if self.baseline_days <= 0:
            raise PopulationRateError(f"baseline_days must be positive, got {self.baseline_days}")

    @property
    def exposure_deg2_days(self) -> float:
        return self.area_deg2 * self.baseline_days


@dataclass(frozen=True)
class Stratum:
    """One inference cell: a footprint, a detection completeness (from
    `significance.evaluate_selection`), and the real observed detected
    count."""

    label: str
    footprint: SurveyFootprint
    completeness: float
    detected: int

    def __post_init__(self) -> None:
        if not 0.0 < self.completeness <= 1.0:
            raise PopulationRateError(f"completeness must be in (0, 1], got {self.completeness}")
        if self.detected < 0:
            raise PopulationRateError(f"detected must be non-negative, got {self.detected}")

    @property
    def exposure(self) -> float:
        return self.footprint.exposure_deg2_days * self.completeness


def poisson_rate_credible_interval(detected: int, exposure: float, *,
                                   level: float = 0.9) -> tuple[float, float, float]:
    """`(point, low, high)` -- the Jeffreys-prior conjugate Gamma posterior
    for a Poisson rate, `Gamma(detected + 1/2, scale=1/exposure)`."""
    if detected < 0:
        raise PopulationRateError(f"detected must be non-negative, got {detected}")
    if exposure < 0:
        raise PopulationRateError(f"exposure must be non-negative, got {exposure}")
    if not 0.0 < level < 1.0:
        raise PopulationRateError(f"level must be in (0, 1), got {level}")
    if exposure == 0:
        return (0.0, 0.0, float("inf"))

    from scipy.stats import gamma

    point = detected / exposure
    tail = (1.0 - level) / 2.0
    shape = detected + 0.5
    low = float(gamma.ppf(tail, a=shape, scale=1.0 / exposure))
    high = float(gamma.ppf(1.0 - tail, a=shape, scale=1.0 / exposure))
    return (point, low, high)


@dataclass
class HierarchicalRateFit:
    parameter_names: tuple[str, ...]
    samples: np.ndarray
    intervals: dict = field(default_factory=dict)
    medians: dict = field(default_factory=dict)
    mu_log_rate_median: float = 0.0
    sigma_log_rate_median: float = 0.0
    per_stratum_rate_medians: dict = field(default_factory=dict)
    autocorrelation_time: dict = field(default_factory=dict)
    converged: bool = False
    n_steps: int = 0
    n_walkers: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "parameter_names": list(self.parameter_names),
            "n_samples": int(len(self.samples)),
            "intervals": self.intervals, "medians": self.medians,
            "mu_log_rate_median": self.mu_log_rate_median,
            "sigma_log_rate_median": self.sigma_log_rate_median,
            "per_stratum_rate_medians": self.per_stratum_rate_medians,
            "autocorrelation_time": self.autocorrelation_time,
            "converged": self.converged, "n_steps": self.n_steps,
            "n_walkers": self.n_walkers, "note": self.note,
        }


def fit_hierarchical_rate(strata: list[Stratum], *, n_walkers: int = 32, n_steps: int = 4000,
                          burn_fraction: float = 0.3, seed: int = 42,
                          levels: tuple[float, ...] = (0.68, 0.9),
                          mu_log_rate_bounds: tuple[float, float] = (-25.0, 5.0),
                          sigma_log_rate_bounds: tuple[float, float] = (1e-3, 5.0)) -> HierarchicalRateFit:
    """Non-centered partial-pooling Poisson rate fit -- see module
    docstring for the model. Walkers are seeded from the closed-form
    per-stratum point estimate (`poisson_rate_credible_interval`), the
    same role a point-estimate fit plays in `microlensing_fit.
    sample_posterior`."""
    if not strata:
        raise PopulationRateError("strata must be non-empty")
    if n_steps < 1:
        raise PopulationRateError(f"n_steps must be at least 1, got {n_steps}")

    n = len(strata)
    n_dim = 2 + n
    if n_walkers < 2 * n_dim:
        raise PopulationRateError(
            f"n_walkers must be at least 2 * (2 + n_strata) = {2 * n_dim}, got {n_walkers}")

    emcee = _require_emcee()
    from scipy.stats import poisson as poisson_dist

    exposures = np.array([s.footprint.exposure_deg2_days for s in strata], dtype=np.float64)
    completeness = np.array([s.completeness for s in strata], dtype=np.float64)
    detected = np.array([s.detected for s in strata], dtype=np.int64)

    mu_lo, mu_hi = mu_log_rate_bounds
    sig_lo, sig_hi = sigma_log_rate_bounds
    log_sig_lo, log_sig_hi = math.log(sig_lo), math.log(sig_hi)

    def log_posterior(vector: np.ndarray) -> float:
        mu, log_sigma = float(vector[0]), float(vector[1])
        if not (mu_lo <= mu <= mu_hi and log_sig_lo <= log_sigma <= log_sig_hi):
            return -np.inf
        z = vector[2:]
        sigma = math.exp(log_sigma)
        log_rate = mu + sigma * z
        if np.any(log_rate > 50.0):  # guard exp() overflow before it happens
            return -np.inf
        expected = np.exp(log_rate) * exposures * completeness
        if not np.all(np.isfinite(expected)):
            return -np.inf
        log_like = float(np.sum(poisson_dist.logpmf(detected, expected)))
        if not np.isfinite(log_like):
            return -np.inf
        return log_like - 0.5 * float(np.sum(z ** 2))  # standard-normal prior on each z_i

    point_rates = np.array([
        poisson_rate_credible_interval(int(s.detected), s.exposure)[0] for s in strata])
    log_point_rates = np.log(np.clip(point_rates, 1e-12, None))
    mu_guess = float(np.clip(np.mean(log_point_rates), mu_lo, mu_hi))
    sigma_guess = float(np.clip(np.std(log_point_rates) + 1e-2, sig_lo, sig_hi))
    z_guess = (log_point_rates - mu_guess) / sigma_guess
    centre = np.concatenate(([mu_guess, math.log(sigma_guess)], z_guess))

    rng = np.random.default_rng(seed)
    scatter = np.abs(centre) * 1e-2 + 1e-2
    positions = centre + scatter * rng.normal(size=(n_walkers, n_dim))
    positions[:, 0] = np.clip(positions[:, 0], mu_lo, mu_hi)
    positions[:, 1] = np.clip(positions[:, 1], log_sig_lo, log_sig_hi)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior)
    sampler.run_mcmc(positions, n_steps, progress=False)

    burn = int(n_steps * burn_fraction)
    chain = sampler.get_chain(discard=burn, flat=True)
    parameter_names = ("mu_log_rate", "log_sigma_log_rate") + tuple(f"z_{s.label}" for s in strata)

    import logging

    logger = logging.getLogger("emcee.autocorr")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        autocorrelation = {name: (None if not np.isfinite(v) else float(v))
                          for name, v in zip(parameter_names, tau)}
        finite_tau = [v for v in tau if np.isfinite(v)]
        converged = bool(finite_tau) and n_steps >= 50 * max(finite_tau)
    except Exception:  # noqa: BLE001 - a failed diagnostic must not lose the samples
        autocorrelation = {name: None for name in parameter_names}
        converged = False
    finally:
        logger.setLevel(previous_level)

    intervals: dict = {}
    medians: dict = {}
    for index, name in enumerate(parameter_names):
        column = chain[:, index]
        medians[name] = float(np.median(column))
        intervals[name] = {}
        for level in levels:
            tail = (1.0 - level) / 2.0
            low, high = np.quantile(column, [tail, 1.0 - tail])
            intervals[name][str(level)] = [float(low), float(high)]

    mu_med = medians["mu_log_rate"]
    sigma_med = math.exp(medians["log_sigma_log_rate"])
    per_stratum_rate_medians = {
        stratum.label: float(math.exp(mu_med + sigma_med * medians[f"z_{stratum.label}"]))
        for stratum in strata
    }

    return HierarchicalRateFit(
        parameter_names=parameter_names, samples=chain, intervals=intervals, medians=medians,
        mu_log_rate_median=mu_med, sigma_log_rate_median=sigma_med,
        per_stratum_rate_medians=per_stratum_rate_medians,
        autocorrelation_time=autocorrelation, converged=converged,
        n_steps=n_steps, n_walkers=n_walkers,
        note=("" if converged else
             "chain is shorter than 50 autocorrelation times; intervals are "
             "reported but not certified converged"),
    )


def anchor_survey_rate_sweep(by_survey: dict[str, list[SourceRef]],
                             footprints: dict[str, SurveyFootprint],
                             completeness: dict[str, float],
                             anchor_surveys: list[str], *,
                             radius_arcsec: float = crossmatch.DEFAULT_RADIUS_ARCSEC,
                             level: float = 0.9) -> dict:
    """Reruns the closed-form rate estimator once per candidate anchor
    survey, using `crossmatch.group_sources`/`grouping_bias_report`
    UNCHANGED -- the "bias under anchor-survey changes" metric this
    module's docstring names."""
    if not anchor_surveys:
        raise PopulationRateError("anchor_surveys must be non-empty")

    results: dict = {}
    for anchor in anchor_surveys:
        if anchor not in footprints or anchor not in completeness:
            raise PopulationRateError(
                f"anchor {anchor!r} needs both a footprint and a completeness value")
        groups = crossmatch.group_sources(
            by_survey, radius_arcsec=radius_arcsec, anchor_survey=anchor)
        report = crossmatch.grouping_bias_report(
            by_survey, groups=groups, anchor_survey=anchor)
        detected = int(report["groups"])
        footprint = footprints[anchor]
        stratum_completeness = completeness[anchor]
        exposure = footprint.exposure_deg2_days * stratum_completeness
        point, low, high = poisson_rate_credible_interval(detected, exposure, level=level)
        results[anchor] = {
            "anchor_policy": report["anchor_policy"], "detected": detected,
            "exposure_deg2_days": footprint.exposure_deg2_days,
            "completeness": stratum_completeness,
            "rate_point": point, "rate_interval": [low, high], "level": level,
        }
    return {"anchors": results, "level": level}


__all__ = [
    "PopulationRateError", "SurveyFootprint", "Stratum",
    "poisson_rate_credible_interval", "HierarchicalRateFit", "fit_hierarchical_rate",
    "anchor_survey_rate_sweep",
]
