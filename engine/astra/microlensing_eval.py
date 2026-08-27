"""The three metrics backlog item 15 names: parameter bias, event
efficiency, and posterior coverage.

Two of the three reuse existing machinery rather than reinventing it:

* **Event efficiency** delegates to `significance.evaluate_selection`,
  which already produces stratified completeness cells with Wilson 95%
  intervals, sampling weights and Kish effective sample size. Its
  `dimensions` are plain strings, so microlensing dimensions
  (`tE_days`, `u0`, `baseline_mag`) drop straight in.
* **Posterior coverage** follows the pattern `frb.credible_region_containment`
  established -- simulate a truth, refit, count how often the credible
  region contains it -- generalised here from a HEALPix sky region to
  parametric credible intervals.

**Parameter bias** is measured two ways, and the first is the stronger:
against OGLE's OWN published fit for a real event (`surveys.ogle.
published_parameters`), which is a genuine external baseline this codebase
has never had for any other model; and against injected synthetic truth,
which is self-consistent but only proves the fitter recovers what it was
handed.

Coverage uses the standard 2025-26 toolkit: expected coverage probability
(does the k% interval contain the truth k% of the time?) and
simulation-based calibration rank statistics (Talts et al.) -- whose
histogram must be UNIFORM for a calibrated posterior, sagging in the
middle when the posterior is too wide and peaking at the edges when it is
too narrow. Both are a few dozen lines; neither needs a new dependency.

Note this is the first time any credible interval in this codebase gets
coverage-tested at all -- `multiband_hier.credible_interval` has never
been.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_LEVELS: tuple[float, ...] = (0.5, 0.68, 0.9)
POINT_LENS_NAMES: tuple[str, ...] = ("t0", "tE", "u0")


def parameter_bias(fitted: list[dict], reference: list[dict],
                   names: tuple[str, ...] = POINT_LENS_NAMES) -> dict:
    """Per-parameter bias of fitted values against a reference truth.

    Reports the MEDIAN fractional bias and the robust scatter (MAD-scaled)
    rather than a mean and standard deviation: a single catastrophic fit
    (a degenerate solution, a missed peak) would otherwise dominate both,
    and microlensing fits fail catastrophically rather than gracefully
    when they fail at all.

    `t0` is reported as an ABSOLUTE bias in days, not fractional: it is a
    Julian date, so a fractional bias against ~2.46e6 is meaningless.
    """
    if len(fitted) != len(reference):
        raise ValueError("fitted and reference must have equal lengths")

    result: dict = {"n_events": len(fitted), "parameters": {}}
    for name in names:
        absolute: list[float] = []
        fractional: list[float] = []
        for fit_row, ref_row in zip(fitted, reference):
            fit_value, ref_value = fit_row.get(name), ref_row.get(name)
            if fit_value is None or ref_value is None:
                continue
            fit_value, ref_value = float(fit_value), float(ref_value)
            if not (np.isfinite(fit_value) and np.isfinite(ref_value)):
                continue
            absolute.append(fit_value - ref_value)
            if ref_value != 0:
                fractional.append((fit_value - ref_value) / ref_value)

        entry: dict = {"n_compared": len(absolute)}
        if absolute:
            values = np.asarray(absolute, dtype=float)
            entry["median_absolute_bias"] = round(float(np.median(values)), 6)
            entry["robust_scatter"] = round(
                float(np.median(np.abs(values - np.median(values))) * 1.4826), 6)
        # A Julian date has no meaningful fractional bias.
        if fractional and name != "t0":
            values = np.asarray(fractional, dtype=float)
            entry["median_fractional_bias"] = round(float(np.median(values)), 6)
            entry["fractional_robust_scatter"] = round(
                float(np.median(np.abs(values - np.median(values))) * 1.4826), 6)
        result["parameters"][name] = entry
    return result


def event_efficiency(records: list[dict],
                     dimensions: tuple[str, ...] = ("tE_days", "u0", "baseline_mag"),
                     edges: dict | None = None, seed: int = 42) -> dict:
    """Recovery fraction stratified by real physical event parameters.

    A thin adapter over `significance.evaluate_selection` -- reuse, not
    reimplementation. Each record needs `detected` plus whatever of the
    named dimensions it has; a missing dimension lands in an explicit
    `unknown` bin rather than being dropped, and every cell carries a
    Wilson 95% interval.

    Performance note carried forward from that function: its per-cell
    weight loop rescans all rows, so it is O(n_cells x n_rows). Keep
    injection sets to thousands, not millions, or pre-aggregate.
    """
    from . import significance

    return significance.evaluate_selection(
        records, dimensions=dimensions, edges=edges, seed=seed)


def interval_contains(interval, truth: float) -> bool:
    low, high = float(interval[0]), float(interval[1])
    return bool(low <= float(truth) <= high)


@dataclass
class CoverageTrial:
    """One simulate-refit trial: the truth, and the posterior it produced."""

    truth: dict
    intervals: dict            # {name: {level_str: [low, high]}}
    samples: np.ndarray | None = None
    names: tuple[str, ...] = POINT_LENS_NAMES


def posterior_coverage(trials: list[CoverageTrial],
                       levels: tuple[float, ...] = DEFAULT_LEVELS) -> dict:
    """Expected coverage probability, per parameter and per nominal level.

    For a calibrated posterior the empirical containment fraction equals
    the nominal level: a 90% interval should contain the truth in ~90% of
    trials, not 60% (over-confident) or 99% (over-conservative). Each
    fraction carries a Wilson 95% interval (reusing
    `significance._ci_binomial`) so a reader can tell a real miscalibration
    from small-sample noise -- the same distinction
    `frb.credible_region_containment` draws.
    """
    from .significance import _ci_binomial

    if not trials:
        return {"n_trials": 0, "levels": {}}

    names = trials[0].names
    result: dict = {"n_trials": len(trials), "levels": {}}
    for level in levels:
        key = str(level)
        per_parameter: dict = {}
        for name in names:
            contained = 0
            counted = 0
            for trial in trials:
                interval = trial.intervals.get(name, {}).get(key)
                truth = trial.truth.get(name)
                if interval is None or truth is None:
                    continue
                counted += 1
                contained += int(interval_contains(interval, truth))
            per_parameter[name] = {
                "nominal": level,
                "empirical": (contained / counted) if counted else None,
                "contained": contained,
                "trials": counted,
                "ci95": _ci_binomial(contained, counted) if counted else None,
            }
        result["levels"][key] = per_parameter
    return result


def sbc_ranks(trials: list[CoverageTrial]) -> dict:
    """Simulation-based calibration rank statistics (Talts et al.).

    For each trial and parameter, the rank of the true value among that
    trial's posterior samples. For a correctly calibrated posterior these
    ranks are UNIFORMLY distributed; a sagging middle means the posterior
    is too wide, and peaks at both edges mean it is too narrow. This
    catches shape miscalibration that expected coverage at a few discrete
    levels can miss.

    Requires `trial.samples`; trials without samples are skipped and
    counted rather than silently ignored.
    """
    if not trials:
        return {"n_trials": 0, "parameters": {}}

    names = trials[0].names
    ranks: dict[str, list[int]] = {name: [] for name in names}
    skipped = 0
    n_samples_seen: list[int] = []

    for trial in trials:
        if trial.samples is None or len(trial.samples) == 0:
            skipped += 1
            continue
        n_samples_seen.append(int(len(trial.samples)))
        for index, name in enumerate(names):
            truth = trial.truth.get(name)
            if truth is None:
                continue
            column = np.asarray(trial.samples)[:, index]
            ranks[name].append(int(np.sum(column < float(truth))))

    parameters: dict = {}
    for name, values in ranks.items():
        if not values:
            parameters[name] = {"n": 0}
            continue
        array = np.asarray(values, dtype=float)
        max_rank = max(n_samples_seen) if n_samples_seen else 1
        normalised = array / max(max_rank, 1)
        parameters[name] = {
            "n": len(values),
            "ranks": [int(v) for v in values],
            # A uniform distribution has mean 0.5; a systematic offset is
            # a directional bias in the posterior.
            "normalised_mean": round(float(np.mean(normalised)), 4),
            "uniform_expected_mean": 0.5,
        }
    return {"n_trials": len(trials), "skipped_no_samples": skipped,
            "parameters": parameters}


def simulate_on_real_cadence(time: np.ndarray, flux_err: np.ndarray,
                             truth, rng: np.random.Generator,
                             f_source: float = 5.0, f_blend: float = 1.0) -> np.ndarray:
    """A synthetic event on REAL observation times and REAL error bars.

    Same discipline `open_world_injection.py` established for the transient
    simulator: the model is synthetic (so the truth is known exactly), but
    the sampling, gaps and noise level are real survey data, not an
    invented cadence. A coverage result on a fabricated uniform cadence
    would not describe any real survey.
    """
    from .microlensing import model_flux

    clean = model_flux(time, truth, f_source, f_blend)
    return clean + rng.normal(0.0, flux_err)


def run_validation_study(time: np.ndarray, flux_err: np.ndarray, *,
                         n_trials: int = 40, seed: int = 42,
                         levels: tuple[float, ...] = DEFAULT_LEVELS,
                         n_steps: int = 1500, n_walkers: int = 24,
                         detection_threshold: float = 25.0) -> dict:
    """Simulate, fit, and report all three of item 15's metrics.

    Truths are drawn from broad, physically plausible priors and injected
    onto the supplied REAL cadence/errors. Each trial is fitted and
    sampled exactly as a real event would be, so the reported bias,
    efficiency and coverage all describe the same pipeline a researcher
    would actually run.

    `detection_threshold` is the delta-chi-square between the fitted lens
    model and a flat baseline above which the event counts as "detected"
    for the efficiency metric -- an explicit, reported criterion rather
    than an implicit one.
    """
    from .microlensing import PointLensParams, flux_to_mag
    from . import microlensing_fit as fitting

    time = np.asarray(time, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)
    rng = np.random.default_rng(seed)
    start, end = float(np.min(time)), float(np.max(time))

    fitted_rows: list[dict] = []
    truth_rows: list[dict] = []
    efficiency_rows: list[dict] = []
    trials: list[CoverageTrial] = []

    for index in range(n_trials):
        truth = PointLensParams(
            t0=rng.uniform(start + 0.2 * (end - start), start + 0.8 * (end - start)),
            tE=float(10.0 ** rng.uniform(np.log10(5.0), np.log10(100.0))),
            u0=float(rng.uniform(0.02, 1.0)),
        )
        f_source = float(rng.uniform(2.0, 10.0))
        f_blend = float(rng.uniform(0.0, 3.0))
        observed = simulate_on_real_cadence(time, flux_err, truth, rng,
                                            f_source=f_source, f_blend=f_blend)

        try:
            fit = fitting.fit_point_lens(time, observed, flux_err, seed=seed + index)
        except Exception:  # noqa: BLE001 - one bad trial must not end the study
            continue

        # Detection: how much better is the lens model than a flat line?
        weights = 1.0 / flux_err ** 2
        baseline = float(np.average(observed, weights=weights))
        chi2_flat = float(np.sum(((observed - baseline) / flux_err) ** 2))
        delta_chi2 = chi2_flat - fit.chi2
        detected = bool(delta_chi2 > detection_threshold)

        efficiency_rows.append({
            "detected": detected,
            "tE_days": truth.tE,
            "u0": truth.u0,
            "baseline_mag": float(flux_to_mag(np.array([f_source + f_blend]))[0]),
            "delta_chi2": delta_chi2,
        })

        if not detected:
            continue

        fitted_rows.append(fit.params.to_dict())
        truth_rows.append(truth.to_dict())

        try:
            posterior = fitting.sample_posterior(
                time, observed, flux_err, fit, n_walkers=n_walkers,
                n_steps=n_steps, seed=seed + index, levels=levels)
        except Exception:  # noqa: BLE001 - keep the bias/efficiency rows either way
            continue

        trials.append(CoverageTrial(
            truth=truth.to_dict(), intervals=posterior.intervals,
            samples=posterior.samples))

    return {
        "n_trials_requested": n_trials,
        "n_detected": len(fitted_rows),
        "parameter_bias": parameter_bias(fitted_rows, truth_rows),
        "event_efficiency": event_efficiency(efficiency_rows),
        "posterior_coverage": posterior_coverage(trials, levels=levels),
        "sbc": sbc_ranks(trials),
        "detection_threshold_delta_chi2": detection_threshold,
        "cadence": {"n_points": int(len(time)),
                   "span_days": round(end - start, 3),
                   "source": "real survey cadence and errors supplied by caller"},
    }
