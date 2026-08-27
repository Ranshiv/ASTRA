"""Hierarchical heteroscedastic multiband period model.

`multiband.py`/`features.multiband_periodic_features` fit a joint period via
astropy's `LombScargleMultiband(method="fast")` -- independent per-band
periodograms combined by weight, deliberately NOT the genuine joint
(`"flexible"`) fit, which that module's own docstring measured at ~39s/object
against ~2.4s for "fast" at real ZTF scale. This module is the different,
opt-in thing `docs/DEFERRED.txt` calls "full hierarchical heteroscedastic
multi-band modelling": one shared latent period across bands, but each
band gets its OWN amplitude and its OWN per-point measurement errors feed
the noise model directly (heteroscedastic), rather than a single SNR
threshold screen.

Modelled with a celerite2 quasi-periodic Gaussian process
(`terms.RotationTerm`) per band, all bands sharing one `period` but each
fitting its own `sigma` (amplitude) and `jitter` (excess per-band noise
beyond the reported measurement error) by maximizing the SUM of the bands'
GP log-likelihoods. celerite2 is a genuinely new dependency (`engine
pyproject.toml`'s `research` extra), gated the same way `torch` already is
in this codebase: lazy-imported inside the functions that need it, excluded
from the packaged build (`scripts/build-engine.ps1`), and never a dependency
of anything on the default candidate-discovery path.

Cost budget, stated explicitly the way `features.py` states its own: a
period grid of size N requires N GP fits per band (each O(n) via celerite2's
banded solver, not the O(n^3) of a naive GP), so this scales as
O(N * n_bands * n_points) -- affordable for a single researcher-run object,
NOT swept into `featurematrix.build()`'s per-object population path the way
single-band Lomb-Scargle is.

Like `significance.py` and every other interpretation layer in this
codebase, this NEVER touches `evidence.WEIGHTS`/`scoring.combine()` --
`analyze_object`'s report is additional, joinable evidence, validated here
against synthetic ground truth, not yet adopted into production scoring.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

# RotationTerm's overtone shape (Q0, dQ, f) is held FIXED and shared across
# bands and across the whole period grid -- only `period`, per-band `sigma`,
# and per-band `jitter` are fitted. Letting Q0/dQ/f float per band would let
# the model absorb amplitude differences into fictitious shape differences
# instead of the band-specific amplitude term this feature is about; a
# future version could fit them, but that is a different, larger model than
# the one validated here.
DEFAULT_Q0 = 1.0
DEFAULT_DQ = 1.0
DEFAULT_F = 0.5
JITTER_FLOOR = 1e-6


class MultibandHierError(ValueError):
    """A hierarchical multiband period fit could not be produced."""


def _require_celerite2():
    try:
        from celerite2 import GaussianProcess, terms
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MultibandHierError(
            "celerite2 is not installed; install the 'research' extra "
            "(pip install .[research]) to use multiband_hier.py"
        ) from exc
    return GaussianProcess, terms


def _band_log_likelihood(GaussianProcess, terms, *, time: np.ndarray, value: np.ndarray,
                         error: np.ndarray, period: float, log_sigma: float,
                         log_jitter: float) -> float:
    term = terms.RotationTerm(sigma=math.exp(log_sigma), period=period,
                              Q0=DEFAULT_Q0, dQ=DEFAULT_DQ, f=DEFAULT_F)
    gp = GaussianProcess(term, mean=0.0)
    diag = error ** 2 + math.exp(log_jitter) ** 2 + JITTER_FLOOR
    try:
        gp.compute(time, diag=diag)
        return float(gp.log_likelihood(value))
    except Exception:  # noqa: BLE001 - a degenerate hyperparameter draw is a bad fit, not a crash
        return float("-inf")


def _fit_band_at_period(GaussianProcess, terms, *, time: np.ndarray, value: np.ndarray,
                        error: np.ndarray, period: float) -> dict[str, Any]:
    """Best (sigma, jitter) for one band at a FIXED period, via a small
    deterministic grid-plus-refine search -- avoids pulling in a general
    nonlinear optimizer dependency for a 2-parameter problem."""
    amplitude_guess = float(np.std(value)) or 1.0
    best = {"log_likelihood": float("-inf"), "sigma": amplitude_guess, "jitter": 0.0}
    for sigma_scale in (0.3, 0.6, 1.0, 1.5, 2.5):
        for jitter_scale in (0.0, 0.5, 1.0, 2.0):
            log_sigma = math.log(max(amplitude_guess * sigma_scale, 1e-6))
            jitter = jitter_scale * float(np.median(error)) if error.size else 0.0
            log_jitter = math.log(max(jitter, 1e-6))
            loglike = _band_log_likelihood(
                GaussianProcess, terms, time=time, value=value, error=error,
                period=period, log_sigma=log_sigma, log_jitter=log_jitter)
            if loglike > best["log_likelihood"]:
                best = {"log_likelihood": loglike, "sigma": math.exp(log_sigma),
                        "jitter": math.exp(log_jitter)}
    return best


def fit_shared_period(bands: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                      period_grid: np.ndarray) -> dict[str, Any]:
    """Fit one shared period across bands, each with its own amplitude/jitter.

    `bands` maps a band label to `(time, value, error)` arrays (already
    baseline-subtracted per band, the same convention
    `features.multiband_periodic_features` uses). Returns the best period
    on `period_grid` by summed log-likelihood across bands, plus the full
    per-period profile (needed for `credible_interval`/`calibrated_fap`
    below) and each band's fitted amplitude/jitter at the best period.
    """
    GaussianProcess, terms = _require_celerite2()
    if len(bands) < 2:
        raise MultibandHierError("at least two bands are required for a joint fit")
    for band, (time, value, error) in bands.items():
        if len({len(time), len(value), len(error)}) != 1:
            raise MultibandHierError(f"band {band!r} has mismatched array lengths")
        if len(time) < 5:
            raise MultibandHierError(f"band {band!r} has too few points to fit")
    period_grid = np.asarray(period_grid, dtype=np.float64)
    if period_grid.size == 0:
        raise MultibandHierError("period_grid must be non-empty")

    profile: list[dict[str, Any]] = []
    for period in period_grid:
        per_band_fit = {
            band: _fit_band_at_period(GaussianProcess, terms, time=time, value=value,
                                      error=error, period=float(period))
            for band, (time, value, error) in bands.items()
        }
        total_log_likelihood = sum(fit["log_likelihood"] for fit in per_band_fit.values())
        profile.append({"period_days": float(period),
                        "log_likelihood": float(total_log_likelihood),
                        "per_band": per_band_fit})

    best = max(profile, key=lambda row: row["log_likelihood"])
    return {
        "best_period_days": best["period_days"],
        "log_likelihood": best["log_likelihood"],
        "per_band": {band: {"sigma": fit["sigma"], "jitter": fit["jitter"]}
                    for band, fit in best["per_band"].items()},
        "profile": profile,
    }


def credible_interval(profile: list[dict[str, Any]], *, level: float = 0.68) -> dict[str, Any]:
    """Highest-posterior-density period interval from a log-likelihood profile.

    Converts the (unnormalized) profile log-likelihoods into relative
    posterior weights (uniform prior over the grid), then accumulates grid
    points in DECREASING weight order until `level` of the total mass is
    covered -- the standard HPD construction, correct for a multimodal or
    asymmetric profile in a way that a symmetric +/- interval around the
    best period would not be.
    """
    if not profile:
        raise MultibandHierError("profile must be non-empty")
    log_likelihoods = np.asarray([row["log_likelihood"] for row in profile])
    periods = np.asarray([row["period_days"] for row in profile])
    weights = np.exp(log_likelihoods - np.max(log_likelihoods))
    weights = weights / np.sum(weights)

    order = np.argsort(-weights)
    cumulative = np.cumsum(weights[order])
    n_included = int(np.searchsorted(cumulative, level) + 1)
    included_periods = periods[order[:n_included]]
    return {
        "level": float(level),
        "lower_days": float(np.min(included_periods)),
        "upper_days": float(np.max(included_periods)),
        "n_grid_points_included": n_included,
        "n_grid_points_total": len(profile),
        "map_period_days": float(periods[order[0]]),
    }


def calibrated_fap(bands: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                   period_grid: np.ndarray, observed_log_likelihood: float, *,
                   n_null: int = 100, seed: int = 42) -> dict[str, Any]:
    """Empirical false-alarm probability via time-scrambled null trials.

    Each null trial keeps every band's observation TIMES fixed (preserving
    the real, possibly irregular, cadence/gap structure) but independently
    permutes each band's VALUES against them, destroying any genuine
    periodicity while preserving each band's own value distribution and
    per-point errors. `fit_shared_period` is rerun (full grid search) on
    each scrambled draw; the FAP is the fraction of null trials whose best
    log-likelihood meets or exceeds the real, observed one -- the same
    "probability an unrelated draw passes by coincidence" shape
    `evidence.period_agreement_fap` already uses for the per-survey period
    check, just estimated by simulation here instead of in closed form.
    """
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_null):
        scrambled = {
            band: (time, rng.permutation(value), error)
            for band, (time, value, error) in bands.items()
        }
        null_fit = fit_shared_period(scrambled, period_grid)
        if null_fit["log_likelihood"] >= observed_log_likelihood:
            exceedances += 1
    return {
        "false_alarm_probability": (exceedances + 1.0) / (n_null + 1.0),
        "n_null": int(n_null), "exceedances": exceedances,
        "seed": int(seed),
    }


def analyze_object(bands: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                   period_grid: np.ndarray, *, credible_level: float = 0.68,
                   n_null: int = 100, seed: int = 42,
                   compute_fap: bool = True) -> dict[str, Any]:
    """Top-level entry point: shared-period fit, HPD interval, and FAP.

    `compute_fap=False` skips the (expensive: `n_null` full grid refits)
    false-alarm simulation -- useful when only the period/amplitude fit is
    wanted, e.g. from a batch job where FAP will be computed separately for
    a smaller, pre-selected set of candidates.
    """
    try:
        fit = fit_shared_period(bands, period_grid)
    except MultibandHierError as exc:
        return {"schema_version": SCHEMA_VERSION, "ready": False, "reason": str(exc)}

    interval = credible_interval(fit["profile"], level=credible_level)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ready": True,
        "best_period_days": fit["best_period_days"],
        "log_likelihood": fit["log_likelihood"],
        "per_band_amplitude": {band: values["sigma"] for band, values in fit["per_band"].items()},
        "per_band_jitter": {band: values["jitter"] for band, values in fit["per_band"].items()},
        "credible_interval": interval,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if compute_fap:
        result["fap"] = calibrated_fap(bands, period_grid, fit["log_likelihood"],
                                       n_null=n_null, seed=seed)
    return result


__all__ = [
    "SCHEMA_VERSION", "MultibandHierError", "fit_shared_period", "credible_interval",
    "calibrated_fap", "analyze_object",
]
