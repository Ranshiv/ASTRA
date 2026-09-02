"""Human-readable labels/units for raw feature names (roadmap: explainability).

`attribution.py`'s occlusion output names features by their raw column name
(`"robust_amplitude"`, `"bocpd_change_probability"`, ...), which is exactly
right for a developer reading `features.py` but not for an observer deciding
whether a candidate is worth scarce follow-up telescope time. This module is
a pure lookup: it restates what `features.py` and `featurematrix.py` already
compute and document in their own docstrings -- no new external unit
citations, nothing that could drift out of sync with a literature source.

Light-curve values themselves (`mean`, `median`, `std`, `amplitude`, ...) are
deliberately left unitless here: `surveys/base.py`'s `ValueKind` ("mag" vs
"flux") means the same column means different physical things depending on
which survey produced the row, and asserting a fixed unit would be wrong for
whichever survey doesn't match it.
"""

from __future__ import annotations

from typing import TypedDict


class FeatureInfo(TypedDict):
    label: str
    unit: str | None
    description: str


FEATURE_LABELS: dict[str, FeatureInfo] = {
    "n_points": {
        "label": "Point count", "unit": None,
        "description": "Number of finite observations in the light curve.",
    },
    "mean": {
        "label": "Mean brightness", "unit": None,
        "description": "Unweighted mean of the light curve's native value "
                        "(magnitude or flux, survey-dependent).",
    },
    "weighted_mean": {
        "label": "Weighted mean brightness", "unit": None,
        "description": "Inverse-variance-weighted mean; low-error epochs "
                        "dominate more than in the plain mean.",
    },
    "median": {
        "label": "Median brightness", "unit": None,
        "description": "Median of the light curve's native value.",
    },
    "std": {
        "label": "Scatter (std dev)", "unit": None,
        "description": "Sample standard deviation of the light curve.",
    },
    "mad": {
        "label": "Robust scatter (MAD)", "unit": None,
        "description": "Median absolute deviation, scaled to be comparable "
                        "with a Gaussian sigma; resists single bad epochs.",
    },
    "amplitude": {
        "label": "Amplitude (max-min)", "unit": None,
        "description": "Full range between the brightest and faintest "
                        "observation.",
    },
    "robust_amplitude": {
        "label": "Robust amplitude", "unit": None,
        "description": "5th-to-95th percentile range; resists a single "
                        "outlier epoch dominating the plain amplitude.",
    },
    "skew": {
        "label": "Skewness", "unit": None,
        "description": "Asymmetry of the brightness distribution.",
    },
    "kurtosis": {
        "label": "Kurtosis", "unit": None,
        "description": "Tailedness of the brightness distribution relative "
                        "to a Gaussian.",
    },
    "beyond_1std": {
        "label": "Fraction beyond 1 sigma", "unit": None,
        "description": "Share of observations more than one standard "
                        "deviation from the mean.",
    },
    "median_err": {
        "label": "Median reported error", "unit": None,
        "description": "Median of the per-point measurement uncertainties.",
    },
    "reduced_chi2": {
        "label": "Reduced chi-square", "unit": None,
        "description": "Scatter against a constant brightness, normalized "
                        "by the error bars; near 1 means noise-only, much "
                        "greater than 1 is evidence of real variability.",
    },
    "stetson_j": {
        "label": "Stetson J", "unit": None,
        "description": "Correlated-variability index using consecutive "
                        "pairs; weights correlated change over uncorrelated "
                        "noise of the same amplitude.",
    },
    "stetson_k": {
        "label": "Stetson K", "unit": None,
        "description": "Kurtosis-like index of variability shape; near "
                        "0.798 for Gaussian noise.",
    },
    "eta": {
        "label": "Von Neumann eta", "unit": None,
        "description": "Ratio of successive-difference variance to overall "
                        "variance; near 2 for white noise, well below 2 for "
                        "smooth variation.",
    },
    "time_span_days": {
        "label": "Observation baseline", "unit": "days",
        "description": "Time between the first and last observation.",
    },
    "cadence_median_days": {
        "label": "Median cadence", "unit": "days",
        "description": "Median gap between consecutive observations.",
    },
    "cadence_max_gap_days": {
        "label": "Largest cadence gap", "unit": "days",
        "description": "Longest gap between consecutive observations.",
    },
    "linear_trend_per_day": {
        "label": "Linear trend", "unit": "per day",
        "description": "Least-squares slope of brightness over time.",
    },
    "max_step": {
        "label": "Largest single-step jump", "unit": None,
        "description": "Largest brightness change between two consecutive "
                        "observations.",
    },
    "change_point_score": {
        "label": "Change-point score", "unit": None,
        "description": "Strength of the strongest abrupt shift detected in "
                        "the light curve.",
    },
    "best_period_days": {
        "label": "Best-fit period", "unit": "days",
        "description": "Period of the strongest Lomb-Scargle periodogram "
                        "peak.",
    },
    "best_power": {
        "label": "Periodogram peak power", "unit": None,
        "description": "Lomb-Scargle power at the best-fit period.",
    },
    "period_snr": {
        "label": "Period detection S/N", "unit": None,
        "description": "Periodogram peak power relative to the "
                        "background level and its spread.",
    },
    "bocpd_change_probability": {
        "label": "Change probability (latest)", "unit": None,
        "description": "Bayesian online change-point probability at the "
                        "final observation.",
    },
    "bocpd_max_probability": {
        "label": "Peak change probability", "unit": None,
        "description": "Maximum change-point probability observed across "
                        "the light curve.",
    },
    "bocpd_change_index": {
        "label": "Change-point index", "unit": None,
        "description": "Observation index of the peak change-point "
                        "probability.",
    },
    "bocpd_change_time": {
        "label": "Change-point time", "unit": "days",
        "description": "Time of the peak change-point probability.",
    },
    "gaia_parallax": {
        "label": "Gaia parallax", "unit": "mas",
        "description": "Gaia DR3 parallax of the matched source.",
    },
    "gaia_parallax_snr": {
        "label": "Gaia parallax S/N", "unit": None,
        "description": "Gaia parallax divided by its reported uncertainty.",
    },
    "gaia_pmra": {
        "label": "Gaia proper motion (RA)", "unit": "mas/yr",
        "description": "Gaia DR3 proper motion in right ascension.",
    },
    "gaia_pmdec": {
        "label": "Gaia proper motion (Dec)", "unit": "mas/yr",
        "description": "Gaia DR3 proper motion in declination.",
    },
    "gaia_phot_g_mean_mag": {
        "label": "Gaia G magnitude", "unit": "mag",
        "description": "Gaia DR3 mean G-band apparent magnitude.",
    },
    "gaia_bp_rp": {
        "label": "Gaia BP-RP color", "unit": "mag",
        "description": "Gaia DR3 BP minus RP color index.",
    },
    "gaia_distance_pc": {
        "label": "Gaia distance", "unit": "pc",
        "description": "Distance derived from the Gaia parallax.",
    },
    "gaia_abs_g_mag": {
        "label": "Gaia absolute G magnitude", "unit": "mag",
        "description": "Gaia G-band magnitude corrected to absolute scale "
                        "using the derived distance.",
    },
    "gaia_ra_now_deg": {
        "label": "Current RA (proper-motion propagated)", "unit": "deg",
        "description": "Right ascension propagated to the present epoch "
                        "using the Gaia proper motion.",
    },
    "gaia_dec_now_deg": {
        "label": "Current Dec (proper-motion propagated)", "unit": "deg",
        "description": "Declination propagated to the present epoch using "
                        "the Gaia proper motion.",
    },
    "gaia_matched": {
        "label": "Gaia match flag", "unit": None,
        "description": "1.0 if a Gaia counterpart was found within the "
                        "match radius, 0.0 if checked but not found.",
    },
    "manifold_residual_mag": {
        "label": "Isochrone residual", "unit": "mag",
        "description": "Offset of the source from the expected "
                        "color-magnitude isochrone.",
    },
    "manifold_arc_length": {
        "label": "Isochrone arc length", "unit": None,
        "description": "Position of the source's nearest isochrone point, "
                        "measured as arc length along the model track.",
    },
    "manifold_teff_k": {
        "label": "Isochrone effective temperature", "unit": "K",
        "description": "Effective temperature implied by the nearest "
                        "isochrone point.",
    },
    "manifold_matched": {
        "label": "Isochrone match flag", "unit": None,
        "description": "1.0 if an isochrone match was computed, 0.0 "
                        "otherwise.",
    },
}


def describe(name: str) -> FeatureInfo:
    """Look up a feature's label/unit/description, with a graceful fallback.

    An unmapped name (a future feature column this glossary hasn't been
    updated for) never breaks attribution -- it gets a title-cased rendering
    of its raw name instead of raising or silently returning nothing.
    """
    info = FEATURE_LABELS.get(name)
    if info is not None:
        return info
    return {
        "label": name.replace("_", " ").title(),
        "unit": None,
        "description": "Engineered feature; see features.py.",
    }


def format_value(name: str, value: float) -> str:
    """Unit-aware string formatting for one feature's value."""
    if not (value == value):  # NaN check without importing math/numpy here
        return "n/a"
    unit = describe(name)["unit"]
    text = f"{value:.3g}"
    return f"{text} {unit}" if unit else text


__all__ = ["FeatureInfo", "FEATURE_LABELS", "describe", "format_value"]
