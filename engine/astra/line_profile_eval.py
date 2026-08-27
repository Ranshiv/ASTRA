"""Validation for `line_profile.py`/`line_profile_fit.py`: posterior
coverage (item 25's first named metric) and line-parameter residuals
against real released values (its second) -- roadmap item 25.

Posterior coverage reuses `microlensing_eval.CoverageTrial`,
`posterior_coverage`, and `sbc_ranks` UNCHANGED: their shape (a truth dict,
a fitted posterior's per-level credible intervals, optionally its raw
samples) is generic across any parametric model, not specific to
microlensing. Line-parameter residuals reuse `microlensing_eval.
parameter_bias` UNCHANGED for the identical reason -- the same "reuse
existing eval machinery" discipline `radio_variability_eval`/
`xray_hardness_eval` (later items) are also expected to follow.

Real released values come from `surveys.sdss.extract_sdss_line_measurements`
-- SDSS's own pipeline `SPZLINE` fit, confirmed live this session to be
present in the SAME spec-lite FITS `sdss.fetch_spectrum` already downloads
(see that function's docstring), not a separate value-added-catalog query
this codebase originally expected it might need.
"""

from __future__ import annotations

import numpy as np

from . import line_profile_fit as fitting
from .line_profile import LineProfileError, LineProfileParams, model_flux
from .microlensing_eval import CoverageTrial, parameter_bias, posterior_coverage, sbc_ranks

LINE_PROFILE_NAMES: tuple[str, ...] = ("center", "sigma", "gamma")


def simulate_on_real_baseline(wavelength, error, continuum, truth: LineProfileParams,
                              rng: np.random.Generator) -> np.ndarray:
    """A synthetic line on a REAL wavelength grid, continuum, and error bars.

    Same discipline `microlensing_eval.simulate_on_real_cadence` already
    established: the LINE is synthetic (so the truth is known exactly), but
    the sampling, continuum shape and noise level are real spectroscopic
    data, not an invented spectrum.
    """
    clean = model_flux(wavelength, continuum, truth)
    return clean + rng.normal(0.0, np.asarray(error, dtype=np.float64))


def run_coverage_study(wavelength, error, continuum, *, n_trials: int = 40, seed: int = 42,
                       levels: tuple[float, ...] = (0.5, 0.68, 0.9),
                       n_steps: int = 1500, n_walkers: int = 24,
                       center_range: tuple[float, float] | None = None) -> dict:
    """Simulate, fit, and report posterior coverage/SBC on a real
    wavelength/error/continuum baseline, mirroring
    `microlensing_eval.run_validation_study`'s simulate-fit-sample shape.

    Truths are drawn from broad, physically plausible priors: `center`
    within `center_range` (defaults to the middle 60% of the observed
    window, keeping the line's wings inside the data), `sigma`/`gamma`
    from a few sample-spacings up to a modest fraction of the window, and
    `amplitude` scaled to the supplied noise level so injected lines are
    genuine detections, not buried in noise.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    continuum = np.broadcast_to(np.asarray(continuum, dtype=np.float64), wavelength.shape)
    rng = np.random.default_rng(seed)

    lo, hi = float(np.min(wavelength)), float(np.max(wavelength))
    span = hi - lo
    spacing = span / max(len(wavelength) - 1, 1)
    center_lo, center_hi = center_range or (lo + 0.2 * span, lo + 0.8 * span)
    typical_error = float(np.median(error))

    fitted_rows: list[dict] = []
    truth_rows: list[dict] = []
    trials: list[CoverageTrial] = []

    for index in range(n_trials):
        truth = LineProfileParams(
            center=float(rng.uniform(center_lo, center_hi)),
            sigma=float(rng.uniform(2.0 * spacing, max(0.05 * span, 3.0 * spacing))),
            gamma=float(rng.uniform(0.0, max(0.02 * span, spacing))),
            amplitude=float(rng.choice([-1.0, 1.0])) * float(rng.uniform(10.0, 40.0)) * typical_error,
        )
        observed = simulate_on_real_baseline(wavelength, error, continuum, truth, rng)

        try:
            fit = fitting.fit_line_profile(wavelength, observed, error, continuum,
                                           seed=seed + index)
        except Exception:  # noqa: BLE001 - one bad trial must not end the study
            continue

        fitted_rows.append(fit.params.to_dict())
        truth_rows.append(truth.to_dict())

        try:
            posterior = fitting.sample_posterior(
                wavelength, observed, error, continuum, fit,
                n_walkers=n_walkers, n_steps=n_steps, seed=seed + index, levels=levels)
        except Exception:  # noqa: BLE001 - keep the bias row either way
            continue

        trials.append(CoverageTrial(
            truth=truth.to_dict(), intervals=posterior.intervals,
            samples=posterior.samples, names=LINE_PROFILE_NAMES))

    return {
        "n_trials_requested": n_trials,
        "n_fitted": len(fitted_rows),
        "parameter_bias": parameter_bias(fitted_rows, truth_rows, names=LINE_PROFILE_NAMES),
        "posterior_coverage": posterior_coverage(trials, levels=levels),
        "sbc": sbc_ranks(trials),
        "baseline": {"n_points": int(len(wavelength)), "span_angstrom": round(span, 3),
                    "source": "real spectrum wavelength grid, continuum and error bars"},
    }


def line_parameter_residuals(wavelength, flux, error, continuum,
                             released_lines: list[dict], *,
                             window_sigma_angstrom: float = 15.0) -> dict:
    """Fit each RELEASED (real) line's own window with `line_profile_fit`
    and report residuals against SDSS's own pipeline values --
    `sdss.extract_sdss_line_measurements`'s output is exactly the
    `released_lines` shape expected here.

    A window of +/- `window_sigma_angstrom * max(1, released sigma)`
    around each released line's observed wavelength bounds the fit, so a
    nearby, unrelated line does not get folded into this one. Lines whose
    window would fall outside the spectrum's own wavelength coverage, or
    whose window contains too few points to fit, are skipped and counted
    rather than silently dropped.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    continuum = np.broadcast_to(np.asarray(continuum, dtype=np.float64), wavelength.shape)
    lo, hi = float(np.min(wavelength)), float(np.max(wavelength))

    fitted_rows: list[dict] = []
    reference_rows: list[dict] = []
    n_out_of_range = 0
    n_fit_failed = 0

    for line in released_lines:
        center = float(line["observed_wavelength_angstrom"])
        half_window = window_sigma_angstrom * max(float(line["sigma_angstrom"]), 1.0)
        window_lo, window_hi = center - half_window, center + half_window
        if window_lo < lo or window_hi > hi:
            n_out_of_range += 1
            continue

        mask = (wavelength >= window_lo) & (wavelength <= window_hi)
        if np.count_nonzero(mask) < 5:
            n_out_of_range += 1
            continue

        try:
            fit = fitting.fit_line_profile(
                wavelength[mask], flux[mask], error[mask], continuum[mask],
                center_hint=center, window_angstrom=half_window)
        except LineProfileError:
            n_fit_failed += 1
            continue

        fitted_rows.append({"center": fit.params.center, "sigma": fit.params.sigma})
        reference_rows.append({"center": center, "sigma": float(line["sigma_angstrom"])})

    bias = parameter_bias(fitted_rows, reference_rows, names=("center", "sigma")) if fitted_rows \
        else {"n_events": 0, "parameters": {}}

    return {"n_released_lines": len(released_lines), "n_out_of_range": n_out_of_range,
           "n_fit_failed": n_fit_failed, "n_compared": len(fitted_rows), "bias": bias}
