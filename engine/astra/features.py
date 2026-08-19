"""Statistical feature extraction from light curves (plan section 12, stage 3).

Features are chosen to be robust on real survey data rather than elegant on
clean data. Survey photometry is unevenly sampled, contains outliers, and has
per-point uncertainties that vary by an order of magnitude across a curve, so
the standard deviation alone is a poor variability measure: it cannot tell a
genuinely variable star from a faint one with large error bars.

The variability indices here — reduced chi-square against a constant, Stetson
J and K, and the von Neumann eta ratio — are the established way astronomers
separate real variability from measurement noise, and they are what make the
downstream anomaly detection meaningful.

Bumping FEATURE_VERSION invalidates stored feature matrices by design;
plan section 19 requires the feature version to be part of an experiment.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass

import numpy as np

from .surveys.base import LightCurve

FEATURE_VERSION = 2

# Below this a curve cannot support a meaningful variability statistic, let
# alone a period search; features are emitted as NaN rather than as noise.
MIN_POINTS = 10
MIN_POINTS_FOR_PERIOD = 30

# Frequency grid bounds for the period search, in days.
MIN_PERIOD_DAYS = 0.05
MAX_PERIOD_FRACTION = 0.5  # never search beyond half the observed baseline

# Oversampling of each periodogram peak. DO NOT LOWER THIS FOR SPEED.
# Measured on a real 353-point ZTF curve with a 2740-day baseline: at 5 the
# search returns 0.50789 d; at 3 it returns 1.03501 d, the 2x harmonic. The
# coarser grid steps over the true peak and locks onto the alias, so this is
# not a speed/accuracy trade-off — it changes the answer. Wall-clock cost is
# addressed by parallelism and caching in astra.featurematrix instead.
SAMPLES_PER_PEAK = 5


@dataclass(frozen=True)
class FeatureSet:
    """Named features for one light curve, plus the identity behind them."""

    values: dict[str, float]
    object_id: str
    survey: str
    band: str
    path: str

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "survey": self.survey,
            "band": self.band,
            "path": self.path,
            "feature_version": FEATURE_VERSION,
            **self.values,
        }


def _nan_features() -> dict[str, float]:
    return {name: float("nan") for name in FEATURE_NAMES}


def photometric_features(value: np.ndarray,
                         value_err: np.ndarray) -> dict[str, float]:
    """Distribution shape and amplitude, using robust estimators throughout."""
    n = len(value)
    mean = float(np.mean(value))
    std = float(np.std(value, ddof=1)) if n > 1 else 0.0
    median = float(np.median(value))

    # Median absolute deviation, scaled to be comparable with a Gaussian sigma.
    mad = float(np.median(np.abs(value - median))) * 1.4826

    # A 5th-95th percentile range resists the single bad epoch that would
    # dominate a plain max-minus-min amplitude.
    p5, p95 = np.percentile(value, [5.0, 95.0])
    robust_amplitude = float(p95 - p5)

    # Inverse-variance weighting: bright epochs with small errors should
    # dominate the mean, which is not true of the unweighted average.
    weights = 1.0 / np.clip(value_err, 1e-12, None) ** 2
    weighted_mean = float(np.sum(weights * value) / np.sum(weights))

    beyond_1std = float(np.mean(np.abs(value - mean) > std)) if std > 0 else 0.0

    return {
        "n_points": float(n),
        "mean": mean,
        "weighted_mean": weighted_mean,
        "median": median,
        "std": std,
        "mad": mad,
        "amplitude": float(np.max(value) - np.min(value)),
        "robust_amplitude": robust_amplitude,
        "skew": _skew(value),
        "kurtosis": _kurtosis(value),
        "beyond_1std": beyond_1std,
        "median_err": float(np.median(value_err)),
    }


def variability_indices(value: np.ndarray,
                        value_err: np.ndarray) -> dict[str, float]:
    """Statistics that separate real variability from measurement noise."""
    n = len(value)
    err = np.clip(value_err, 1e-12, None)

    weights = 1.0 / err ** 2
    weighted_mean = float(np.sum(weights * value) / np.sum(weights))

    # Reduced chi-square against a constant brightness. Around 1 means the
    # scatter is fully explained by the error bars; much greater than 1 is
    # the primary evidence that a source is genuinely variable.
    chi2 = float(np.sum(((value - weighted_mean) / err) ** 2))
    reduced_chi2 = chi2 / (n - 1) if n > 1 else float("nan")

    # Stetson J and K use consecutive pairs, so correlated variation counts
    # for more than uncorrelated noise of the same amplitude.
    delta = np.sqrt(n / max(n - 1, 1)) * (value - weighted_mean) / err
    if n >= 2:
        pairs = delta[:-1] * delta[1:]
        stetson_j = float(np.sum(np.sign(pairs) * np.sqrt(np.abs(pairs))) / len(pairs))
    else:
        stetson_j = float("nan")

    abs_delta = np.abs(delta)
    denominator = np.sqrt(float(np.sum(delta ** 2)) / n)
    stetson_k = (float(np.mean(abs_delta)) / denominator) if denominator > 0 \
        else float("nan")

    # Von Neumann eta: the ratio of successive-difference variance to overall
    # variance. Near 2 for white noise, well below 2 for smooth variation.
    variance = float(np.var(value, ddof=1)) if n > 1 else 0.0
    eta = (float(np.mean(np.diff(value) ** 2)) / variance) if variance > 0 \
        else float("nan")

    return {
        "reduced_chi2": reduced_chi2,
        "stetson_j": stetson_j,
        "stetson_k": stetson_k,
        "eta": eta,
    }


def temporal_features(time: np.ndarray, value: np.ndarray) -> dict[str, float]:
    """Sampling, trend and abruptness, independent of any period model."""
    span = float(time[-1] - time[0])
    gaps = np.diff(time)

    # Least-squares slope; a strong secular trend is itself interesting and
    # also contaminates period searches if not noticed.
    slope = float(np.polyfit(time - time[0], value, 1)[0]) if len(time) > 2 \
        else float("nan")

    return {
        "time_span_days": span,
        "cadence_median_days": float(np.median(gaps)) if gaps.size else float("nan"),
        "cadence_max_gap_days": float(np.max(gaps)) if gaps.size else float("nan"),
        "linear_trend_per_day": slope,
        "max_step": float(np.max(np.abs(np.diff(value)))) if len(value) > 1
        else float("nan"),
        "change_point_score": _change_point_score(value),
    }


def periodic_features(time: np.ndarray, value: np.ndarray,
                      value_err: np.ndarray,
                      min_period_days: float = MIN_PERIOD_DAYS,
                      samples_per_peak: int = SAMPLES_PER_PEAK
                      ) -> dict[str, float]:
    """Lomb-Scargle period search, which handles uneven sampling correctly.

    Cost scales as baseline x maximum frequency: a 2740-day ZTF curve searched
    down to 0.05 d needs ~274,000 frequencies and about 1.2 seconds. Raising
    `min_period_days` to 0.1 halves that and, on the curve measured, returns
    the same period — but it would miss short-period Delta Scuti pulsators, so
    the default stays at the more sensitive setting.
    """
    if len(time) < MIN_POINTS_FOR_PERIOD:
        return {"best_period_days": float("nan"), "best_power": float("nan"),
                "period_snr": float("nan")}

    span = float(time[-1] - time[0])
    if span <= 0:
        return {"best_period_days": float("nan"), "best_power": float("nan"),
                "period_snr": float("nan")}

    try:
        from astropy.timeseries import LombScargle

        max_period = span * MAX_PERIOD_FRACTION
        if max_period <= min_period_days:
            raise ValueError("baseline too short for the period grid")

        model = LombScargle(time, value, np.clip(value_err, 1e-12, None))
        frequency, power = model.autopower(
            minimum_frequency=1.0 / max_period,
            maximum_frequency=1.0 / min_period_days,
            samples_per_peak=samples_per_peak,
            method="fast",
        )
        if power.size == 0:
            raise ValueError("empty periodogram")

        best = int(np.argmax(power))
        best_power = float(power[best])
        background = float(np.median(power))
        spread = float(np.std(power))
        return {
            "best_period_days": float(1.0 / frequency[best]),
            "best_power": best_power,
            "period_snr": ((best_power - background) / spread
                            if spread > 0 else float("nan")),
        }
    except Exception:  # noqa: BLE001 - a failed search is missing data, not a crash
        return {"best_period_days": float("nan"), "best_power": float("nan"),
                "period_snr": float("nan")}


def multiband_periodic_features(curves: list[LightCurve],
                                min_period_days: float = MIN_PERIOD_DAYS,
                                samples_per_peak: int = SAMPLES_PER_PEAK) -> dict[str, float]:
    """Fit one shared period while allowing each band its own baseline.

    This is a bounded multi-band approximation to a full joint light-curve
    model.  Every band contributes its finite points after subtracting its
    weighted mean; a shared sinusoid is then evaluated on the union.  The
    returned period is therefore not the average of independent per-band
    periods and cannot be dominated by the band with the most observations.
    ``bands`` reports how many distinct passbands contributed.
    """
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    names: set[str] = set()
    for curve in curves:
        tidy = curve.dropna().sorted_by_time()
        if len(tidy) < MIN_POINTS_FOR_PERIOD:
            continue
        err = np.clip(np.asarray(tidy.value_err, dtype=float), 1e-12, None)
        values = np.asarray(tidy.value, dtype=float)
        weights = 1.0 / err ** 2
        values = values - np.sum(weights * values) / np.sum(weights)
        prepared.append((np.asarray(tidy.time, dtype=float), values, err))
        names.add(str(curve.band))
    if not prepared or len(names) < 2:
        return {"best_period_days": float("nan"), "best_power": float("nan"),
                "period_snr": float("nan"), "bands": float(len(names)),
                "points": float(sum(len(item[0]) for item in prepared))}

    time = np.concatenate([item[0] for item in prepared])
    value = np.concatenate([item[1] for item in prepared])
    err = np.concatenate([item[2] for item in prepared])
    span = float(np.max(time) - np.min(time))
    max_period = span * MAX_PERIOD_FRACTION
    if span <= 0 or max_period <= min_period_days:
        return {"best_period_days": float("nan"), "best_power": float("nan"),
                "period_snr": float("nan"), "bands": float(len(names)),
                "points": float(len(time))}
    try:
        from astropy.timeseries import LombScargle
        frequency, power = LombScargle(time, value, err).autopower(
            minimum_frequency=1.0 / max_period,
            maximum_frequency=1.0 / min_period_days,
            samples_per_peak=samples_per_peak,
            method="fast",
        )
        best = int(np.argmax(power))
        background = float(np.median(power))
        spread = float(np.std(power))
        return {
            "best_period_days": float(1.0 / frequency[best]),
            "best_power": float(power[best]),
            "period_snr": ((float(power[best]) - background) / spread
                            if spread > 0 else float("nan")),
            "bands": float(len(names)), "points": float(len(time)),
        }
    except Exception:  # noqa: BLE001 - missing/invalid data is unavailable
        return {"best_period_days": float("nan"), "best_power": float("nan"),
                "period_snr": float("nan"), "bands": float(len(names)),
                "points": float(len(time))}


def bocpd(time: np.ndarray, value: np.ndarray, hazard: float = 1 / 200.0,
          max_run_length: int = 512) -> dict[str, float]:
    """Bayesian online change-point detection with a Gaussian predictive model.

    The implementation follows the constant-hazard BOCPD recursion, with a
    robust fixed observation variance and Normal-Inverse-Gamma-free predictive
    approximation.  It is deliberately bounded: run-length probabilities are
    truncated to ``max_run_length`` and normalized at every observation.  The
    output is a probability for a change at the final observation, the maximum
    posterior probability observed, and its index/time.
    """
    values = np.asarray(value, dtype=float)
    times = np.asarray(time, dtype=float)
    finite = np.isfinite(values) & np.isfinite(times)
    values, times = values[finite], times[finite]
    if len(values) < 3:
        return {"change_probability": float("nan"), "max_probability": float("nan"),
                "change_index": float("nan"), "change_time": float("nan")}
    differences = np.diff(values)
    sigma = 1.4826 * float(np.median(np.abs(differences))) / math.sqrt(2.0)
    sigma = max(sigma, 1e-6 * max(float(np.median(np.abs(values))), 1.0))
    variance = sigma ** 2
    run = np.array([1.0], dtype=float)
    means = np.array([float(values[0])], dtype=float)
    counts = np.array([1.0], dtype=float)
    change_probabilities: list[float] = []
    for observation in values[1:]:
        max_len = min(len(run) + 1, max_run_length)
        predictive = np.exp(-0.5 * ((observation - means) ** 2) / variance)
        predictive /= math.sqrt(2.0 * math.pi * variance)
        growth = run * predictive * (1.0 - hazard)
        change = float(np.sum(run * predictive * hazard))
        next_run = np.zeros(max_len, dtype=float)
        next_run[0] = change
        next_run[1:min(len(growth) + 1, max_len)] = growth[:max_len - 1]
        normalizer = float(np.sum(next_run))
        if not np.isfinite(normalizer) or normalizer <= 0:
            next_run[:] = 0.0
            next_run[0] = 1.0
        else:
            next_run /= normalizer
        # Posterior sufficient statistics for each continuing run length.
        next_means = np.empty(max_len, dtype=float)
        next_counts = np.empty(max_len, dtype=float)
        next_means[0], next_counts[0] = observation, 1.0
        carry = min(len(means), max_len - 1)
        next_counts[1:carry + 1] = counts[:carry] + 1.0
        next_means[1:carry + 1] = (means[:carry] * counts[:carry] + observation) / next_counts[1:carry + 1]
        run, means, counts = next_run, next_means, next_counts
        change_probabilities.append(float(run[0]))
    probabilities = np.asarray(change_probabilities)
    index = int(np.argmax(probabilities)) + 1
    return {
        "change_probability": float(probabilities[-1]),
        "max_probability": float(probabilities[index - 1]),
        "change_index": float(index),
        "change_time": float(times[index]),
    }

def extract(curve: LightCurve, path: str = "") -> FeatureSet:
    """Full feature vector for one light curve."""
    tidy = curve.dropna().sorted_by_time()

    if len(tidy) < MIN_POINTS:
        values = _nan_features()
        values["n_points"] = float(len(tidy))
    else:
        values = {}
        values.update(photometric_features(tidy.value, tidy.value_err))
        values.update(variability_indices(tidy.value, tidy.value_err))
        values.update(temporal_features(tidy.time, tidy.value))
        values.update(periodic_features(tidy.time, tidy.value, tidy.value_err))
        values.update({f"bocpd_{name}": value for name, value in
                       bocpd(tidy.time, tidy.value).items()})

    return FeatureSet(
        values={name: float(values.get(name, float("nan")))
                for name in FEATURE_NAMES},
        object_id=curve.source.object_id,
        survey=curve.source.survey,
        band=curve.band,
        path=path,
    )


def _skew(x: np.ndarray) -> float:
    n = len(x)
    std = np.std(x)
    if n < 3 or std == 0:
        return float("nan")
    return float(np.mean(((x - np.mean(x)) / std) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis: 0 for a Gaussian."""
    n = len(x)
    std = np.std(x)
    if n < 4 or std == 0:
        return float("nan")
    return float(np.mean(((x - np.mean(x)) / std) ** 4) - 3.0)


def _change_point_score(value: np.ndarray, window: int = 5) -> float:
    """Largest mean shift between adjacent windows, as a z-score.

    Normalising by the *global* spread would be self-defeating: a step change
    inflates the global spread it is measured against, so a clean 1-magnitude
    step and pure noise score almost the same. Scaling instead by the standard
    error of the difference of two window means, using the pooled within-window
    variance, keeps the step in the numerator and the noise in the denominator.

    White noise then sits near 3 — the expected maximum of many normal draws —
    while a real level change is far larger. A deliberately simple sliding
    statistic, not a formal change-point model.
    """
    n = len(value)
    if n < 2 * window:
        return float("nan")

    # Noise is estimated once, globally, from the median absolute successive
    # difference. A per-window variance over only `window` samples is far too
    # unstable: wherever it underestimates the noise the z-score explodes, and
    # taking a maximum over many windows finds exactly those places. The median
    # is also unaffected by the single large difference a step change creates,
    # which a standard deviation would absorb.
    differences = np.abs(np.diff(value))
    sigma = 1.4826 * float(np.median(differences)) / math.sqrt(2.0)

    # Floor, so a perfectly clean step yields a large finite score rather than
    # a division by zero.
    floor = 1e-9 * max(float(np.median(np.abs(value))), 1.0)
    standard_error = max(sigma, floor) * math.sqrt(2.0 / window)

    best = 0.0
    for i in range(window, n - window + 1):
        left = float(np.mean(value[i - window:i]))
        right = float(np.mean(value[i:i + window]))
        best = max(best, abs(right - left) / standard_error)

    # Clipped so a noiseless step cannot produce an unusable magnitude that
    # would dominate every standardised feature downstream.
    return float(min(best, 1e6))


# Fixed order, so a feature matrix always has the same columns in the same
# positions regardless of which features a given curve could produce.
FEATURE_NAMES: tuple[str, ...] = (
    "n_points", "mean", "weighted_mean", "median", "std", "mad",
    "amplitude", "robust_amplitude", "skew", "kurtosis", "beyond_1std",
    "median_err",
    "reduced_chi2", "stetson_j", "stetson_k", "eta",
    "time_span_days", "cadence_median_days", "cadence_max_gap_days",
    "linear_trend_per_day", "max_step", "change_point_score",
    "best_period_days", "best_power", "period_snr",
    "bocpd_change_probability", "bocpd_max_probability",
    "bocpd_change_index", "bocpd_change_time",
)


def schema_hash() -> str:
    """Content hash of the named feature schema, independent of code layout."""
    payload = f"v{FEATURE_VERSION}|" + "|".join(FEATURE_NAMES)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
