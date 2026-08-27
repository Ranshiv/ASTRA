"""Interpolated cross-correlation function (ICCF) lag recovery.

Shaped like the rest of this module family: opt-in, cites its method,
states its scope limits explicitly, never wired into `evidence.WEIGHTS`/
`scoring.combine()`/`rpc.py`. Survey-agnostic by design (works on any two
`time`/`value`/`value_err` series) -- no NEOWISE connector exists in this
codebase (confirmed while planning roadmap item 20), so this module is
validated on synthetic data plus real ZTF multi-band curves as a
stand-in pair, ready to consume NEOWISE W1/W2 whenever that connector
exists.

`interpolated_cross_correlation` is the real, standard ICCF method
(Gaskell & Peterson 1987, ApJ 314, 738): each series is linearly
interpolated onto the other's (lag-shifted) time grid and the Pearson
correlation is computed at every trial lag; `centroid_lag` is the standard
weighted centroid of lags with correlation at or above `0.8 * peak
correlation` (Peterson et al. 1998, PASP 110, 660's convention), a more
robust point estimate than the bare argmax `peak_lag` alone.

`lag_uncertainty_frss` is the standard Flux Randomization / Random Subset
Selection Monte Carlo (Peterson et al. 1998, 2004, ApJ 613, 682): each
trial independently (a) resamples the point set WITH replacement then
keeps only the unique draws (random subset selection) and (b) perturbs
each remaining point by a Gaussian draw scaled to its own reported error
(flux randomization), recomputing `centroid_lag` -- the resulting
distribution's spread is the standard reported lag uncertainty in real
reverberation-mapping work, not an ad hoc bootstrap invented here.

No new dependency: both techniques are bounded `numpy`/`scipy`
interpolation and correlation arithmetic. `engine/pyproject.toml`'s
`research` extra has nothing reverberation-specific (no `pyccf`/`javelin`).

Explicit scope limit: this is cross-correlation lag recovery between two
ALREADY-EXTRACTED light curves (e.g. a continuum vs. line proxy, or two
WISE bands) -- it does not do spectral decomposition to build a "line"
light curve from a spectrum, and it assumes one dominant, roughly
constant lag (a single top-hat/Gaussian transfer function), not a full
velocity-resolved transfer-function reconstruction (e.g. CREAM/JAVELIN's
regularized inversion) -- a real, stated simplification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_POINTS = 10


class ReverberationError(ValueError):
    """A cross-correlation, uncertainty, or lag-recovery computation could not be completed."""


def _finite_pair(time, value, value_err) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    value_err = np.asarray(value_err, dtype=np.float64)
    if not (len(time) == len(value) == len(value_err)):
        raise ReverberationError("time, value, and value_err must be the same length")
    finite = np.isfinite(time) & np.isfinite(value) & np.isfinite(value_err) & (value_err > 0)
    order = np.argsort(time[finite])
    return time[finite][order], value[finite][order], value_err[finite][order]


# ---------------------------------------------------------------------------
# ICCF.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CCFResult:
    lags: np.ndarray
    correlation: np.ndarray
    peak_lag: float
    peak_correlation: float
    centroid_lag: float


def _pearson_at_lag(time1, value1, time2, value2, lag: float) -> float | None:
    """Interpolates series 2 onto series-1 time points shifted by `lag`
    (the standard ICCF convention: a positive lag means series 2 responds
    AFTER series 1), keeping only the overlapping range, and returns the
    Pearson correlation -- `None` when fewer than 3 points overlap."""
    shifted_time1 = time1 + lag
    within = (shifted_time1 >= time2[0]) & (shifted_time1 <= time2[-1])
    if int(within.sum()) < 3:
        return None
    interpolated = np.interp(shifted_time1[within], time2, value2)
    v1 = value1[within]
    if np.std(v1) == 0 or np.std(interpolated) == 0:
        return None
    return float(np.corrcoef(v1, interpolated)[0, 1])


def interpolated_cross_correlation(time1, value1, time2, value2, lag_grid) -> CCFResult:
    time1 = np.asarray(time1, dtype=np.float64)
    value1 = np.asarray(value1, dtype=np.float64)
    time2 = np.asarray(time2, dtype=np.float64)
    value2 = np.asarray(value2, dtype=np.float64)
    if len(time1) < MIN_POINTS or len(time2) < MIN_POINTS:
        raise ReverberationError(f"need at least {MIN_POINTS} points in each series")
    lag_grid = np.asarray(lag_grid, dtype=np.float64)
    if lag_grid.size == 0:
        raise ReverberationError("lag_grid must be non-empty")

    raw_correlation = [_pearson_at_lag(time1, value1, time2, value2, float(lag)) for lag in lag_grid]
    # A legitimate correlation of exactly 0.0 must not be conflated with
    # "no overlap" (`None`) -- an `or np.nan` shortcut would do exactly that.
    correlation = np.array([c if c is not None else np.nan for c in raw_correlation], dtype=np.float64)
    if not np.any(np.isfinite(correlation)):
        raise ReverberationError("no lag in lag_grid produced an overlapping correlation")

    peak_idx = int(np.nanargmax(correlation))
    peak_lag = float(lag_grid[peak_idx])
    peak_correlation = float(correlation[peak_idx])

    # The centroid is restricted to the CONTIGUOUS run of points around the
    # peak that stays at or above the threshold, not every lag anywhere in
    # the grid that happens to clear it -- for a periodic or quasi-periodic
    # driving signal the CCF has multiple lobes of similar height, and a
    # global mask pulls the centroid toward an unrelated alias lobe rather
    # than the true peak's own neighbourhood. This is the standard ICCF
    # convention (e.g. the PyCCF implementation), not a global threshold.
    threshold = 0.8 * peak_correlation
    left = peak_idx
    while left > 0 and np.isfinite(correlation[left - 1]) and correlation[left - 1] >= threshold:
        left -= 1
    right = peak_idx
    n = len(correlation)
    while right < n - 1 and np.isfinite(correlation[right + 1]) and correlation[right + 1] >= threshold:
        right += 1
    window = slice(left, right + 1)
    weights = np.clip(correlation[window], 0.0, None)
    if np.sum(weights) > 0:
        centroid_lag = float(np.sum(lag_grid[window] * weights) / np.sum(weights))
    else:
        centroid_lag = peak_lag

    return CCFResult(lags=lag_grid, correlation=correlation, peak_lag=peak_lag,
                     peak_correlation=peak_correlation, centroid_lag=centroid_lag)


# ---------------------------------------------------------------------------
# FR/RSS lag uncertainty.
# ---------------------------------------------------------------------------

def lag_uncertainty_frss(time1, value1, value_err1, time2, value2, value_err2, lag_grid, *,
                         n_trials: int = 200, seed: int = 42) -> dict:
    """Flux Randomization / Random Subset Selection Monte Carlo lag
    uncertainty (Peterson et al. 1998, 2004) -- see module docstring."""
    time1, value1, value_err1 = _finite_pair(time1, value1, value_err1)
    time2, value2, value_err2 = _finite_pair(time2, value2, value_err2)
    if n_trials < 1:
        raise ReverberationError("n_trials must be at least 1")

    rng = np.random.default_rng(seed)
    centroid_lags: list[float] = []
    for _ in range(n_trials):
        idx1 = np.unique(rng.integers(0, len(time1), size=len(time1)))
        idx2 = np.unique(rng.integers(0, len(time2), size=len(time2)))
        if len(idx1) < MIN_POINTS or len(idx2) < MIN_POINTS:
            continue
        t1, v1 = time1[idx1], value1[idx1] + rng.normal(0.0, value_err1[idx1])
        t2, v2 = time2[idx2], value2[idx2] + rng.normal(0.0, value_err2[idx2])
        try:
            result = interpolated_cross_correlation(t1, v1, t2, v2, lag_grid)
        except ReverberationError:
            continue
        centroid_lags.append(result.centroid_lag)

    finite = np.asarray(centroid_lags, dtype=np.float64)
    if not len(finite):
        return {"mean": None, "std": None, "ci95": None, "n_trials_used": 0}
    return {
        "mean": round(float(np.mean(finite)), 4),
        "std": round(float(np.std(finite, ddof=1)), 4) if len(finite) > 1 else 0.0,
        "ci95": [round(float(np.quantile(finite, 0.025)), 4),
                round(float(np.quantile(finite, 0.975)), 4)],
        "n_trials_used": len(finite),
    }


# ---------------------------------------------------------------------------
# Synthetic lag-recovery validation.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LagRecoveryResult:
    true_lag_days: float
    n_trials: int
    recovered_lag: dict


def _driving_light_curve(time: np.ndarray, seed: int) -> np.ndarray:
    """A real DRW-shaped synthetic driving curve, via `celerite2` (already
    a `research`-extra dependency, used unchanged) -- reverberation mapping
    is normally applied to genuinely stochastic AGN continuum variability,
    not a periodic or white-noise signal."""
    try:
        import celerite2
        from celerite2 import terms
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ReverberationError(
            "celerite2 is not installed; install the 'research' extra "
            "(pip install .[research]) to use evaluate_lag_recovery()"
        ) from exc
    term = terms.RealTerm(a=1.0, c=1.0 / (0.1 * (time[-1] - time[0])))
    gp = celerite2.GaussianProcess(term, mean=0.0)
    gp.compute(time, diag=1e-6 * np.ones_like(time))
    np.random.seed(seed)  # celerite2's gp.sample() has no explicit-generator kwarg
    return gp.sample()


def evaluate_lag_recovery(*, true_lag_days: float, transfer_width_days: float,
                          span_days: float = 300.0, cadence_days: float = 2.0,
                          noise_sigma: float = 0.02, n_trials: int = 10,
                          lag_grid: np.ndarray | None = None, seed: int = 42) -> LagRecoveryResult:
    """Convolves a synthetic DRW driving curve with a known Gaussian
    transfer function centred at `true_lag_days`, subsamples both onto a
    real-cadence-like grid with per-point noise, and reports the recovered
    lag's bias/precision via `interpolated_cross_correlation` across
    `n_trials` independent realizations."""
    if transfer_width_days <= 0:
        raise ReverberationError("transfer_width_days must be positive")
    if n_trials < 1:
        raise ReverberationError("n_trials must be at least 1")

    fine_dt = min(cadence_days, transfer_width_days) / 5.0
    fine_time = np.arange(-5.0 * transfer_width_days, span_days + 5.0 * transfer_width_days, fine_dt)
    kernel_offsets = np.arange(-4.0 * transfer_width_days, 4.0 * transfer_width_days, fine_dt)
    kernel = np.exp(-0.5 * (kernel_offsets / transfer_width_days) ** 2)
    kernel = kernel / np.sum(kernel)

    observed_time = np.arange(0.0, span_days, cadence_days)
    if lag_grid is None:
        lag_grid = np.arange(true_lag_days - 5.0 * transfer_width_days,
                             true_lag_days + 5.0 * transfer_width_days, fine_dt * 2.0)

    recovered_lags: list[float] = []
    for trial in range(n_trials):
        driving_fine = _driving_light_curve(fine_time, seed=seed + trial)
        response_fine = np.convolve(driving_fine, kernel, mode="same")
        # Shift the response by the true lag: series 2 (response) at time t
        # equals the convolved signal at (t - true_lag_days).
        driving_obs = np.interp(observed_time, fine_time, driving_fine)
        response_obs = np.interp(observed_time - true_lag_days, fine_time, response_fine)

        rng = np.random.default_rng(seed + trial)
        driving_noisy = driving_obs + rng.normal(0.0, noise_sigma, size=observed_time.size)
        response_noisy = response_obs + rng.normal(0.0, noise_sigma, size=observed_time.size)

        try:
            result = interpolated_cross_correlation(
                observed_time, driving_noisy, observed_time, response_noisy, lag_grid)
        except ReverberationError:
            continue
        recovered_lags.append(result.centroid_lag)

    finite = np.asarray(recovered_lags, dtype=np.float64)
    if not len(finite):
        summary = {"mean": None, "std": None, "ci95": None, "n_trials_used": 0}
    else:
        summary = {
            "mean": round(float(np.mean(finite)), 4),
            "std": round(float(np.std(finite, ddof=1)), 4) if len(finite) > 1 else 0.0,
            "ci95": [round(float(np.quantile(finite, 0.025)), 4),
                    round(float(np.quantile(finite, 0.975)), 4)],
            "n_trials_used": len(finite),
        }
    return LagRecoveryResult(true_lag_days=true_lag_days, n_trials=n_trials, recovered_lag=summary)


__all__ = [
    "ReverberationError", "CCFResult", "interpolated_cross_correlation",
    "lag_uncertainty_frss", "LagRecoveryResult", "evaluate_lag_recovery", "MIN_POINTS",
]
