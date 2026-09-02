"""Cross-survey zero-point and color-term calibration diagnostic.

Every survey's photometry carries its own instrumental zero-point and
passband, so two surveys observing the same star rarely agree exactly even
after crossmatching -- some of that gap is a real color term (redder stars
look systematically brighter/fainter through a different filter), and some
is a residual, uncorrected instrumental offset. This module estimates both,
per survey and per stratum (e.g. night/camera when the caller supplies one),
against matched Gaia/Pan-STARRS/SDSS pairs assembled by `crossmatch.py`.

Like `significance.py` (its direct template) and every other interpretation
layer in this codebase (`gw.py`, `frb.py`, the artifact-weight injection
study), this is a diagnostic: it never mutates the matched photometry, never
touches `evidence.WEIGHTS`/`scoring.combine()`, and is versioned
(`SCHEMA_VERSION`) so a future adoption into scoring would have to bump
`evidence.WEIGHT_VERSION` explicitly rather than silently changing what an
existing candidate's score means.

Real, explicit limitation: `crossmatch.group_sources` matches on position
only. Turning a `MatchGroup` into a *photometric* pair here additionally
requires each member survey to expose comparable magnitude/error fields --
today that means Gaia (`phot_g_mean_mag`/`phot_bp_mean_mag`/
`phot_rp_mean_mag` plus the flux-derived errors from
`surveys.gaia.photometric_errors`), Pan-STARRS (`g_mean`..`y_mean` plus
`*_mean_error`), and SDSS (`mag_u`..`mag_z` plus `mag_*_error`, sourced from
`PhotoObj` via `surveys/sdss.py::cone_search`'s `bestObjID` join -- only
covers spectroscopic objects with a matched photometric counterpart, since
this connector's `cone_search` still queries `SpecObjAll` first; `coverage`
in the returned report makes a two- vs. three-survey result visible either
way rather than presenting fewer surveys as if all three were included).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import crossmatch
from .surveys import gaia as gaia_survey
from .surveys.base import SourceRef

SCHEMA_VERSION = 1

# Per-survey magnitude/error field names, keyed by a band label shared across
# surveys only loosely (Gaia's G/BP/RP are not Pan-STARRS' g/r/i/z/y -- the
# color term absorbs exactly that passband difference, so no attempt is made
# to pretend they are the same filter).
_GAIA_BANDS = {
    "G": ("phot_g_mean_mag", "phot_g_mean_mag_error"),
    "BP": ("phot_bp_mean_mag", "phot_bp_mean_mag_error"),
    "RP": ("phot_rp_mean_mag", "phot_rp_mean_mag_error"),
}
_PANSTARRS_BANDS = {
    "g": ("g_mean", "g_mean_error"), "r": ("r_mean", "r_mean_error"),
    "i": ("i_mean", "i_mean_error"), "z": ("z_mean", "z_mean_error"),
    "y": ("y_mean", "y_mean_error"),
}
# SDSS `PhotoObj` ugriz model magnitudes, joined onto `SpecObjAll` rows in
# `surveys/sdss.py::cone_search` -- see that module's docstring for the
# real gap this closes (SDSS pairs were previously unavailable here at all).
_SDSS_BANDS = {
    "u": ("mag_u", "mag_u_error"), "g": ("mag_g", "mag_g_error"),
    "r": ("mag_r", "mag_r_error"), "i": ("mag_i", "mag_i_error"),
    "z": ("mag_z", "mag_z_error"),
}
SURVEY_BANDS: dict[str, dict[str, tuple[str, str]]] = {
    "GAIA": _GAIA_BANDS, "PAN-STARRS": _PANSTARRS_BANDS, "SDSS": _SDSS_BANDS,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def source_magnitude(source: SourceRef, band: str) -> tuple[float | None, float | None]:
    """(magnitude, magnitude_error) for one source in one of its own bands.

    Gaia's per-band error is not stored in `extra` directly -- it is derived
    from flux/flux_error on demand via `gaia.photometric_errors`, since the
    archive itself has no `phot_*_mag_error` column.
    """
    bands = SURVEY_BANDS.get(source.survey.upper())
    if bands is None or band not in bands:
        return None, None
    mag_field, error_field = bands[band]
    magnitude = _number(source.extra.get(mag_field))
    if source.survey.upper() == "GAIA":
        error = gaia_survey.photometric_errors(source.extra).get(error_field)
    else:
        error = _number(source.extra.get(error_field))
    return magnitude, error


def derive_stratum_by_object_id(sources: list[SourceRef],
                                field: str) -> dict[str, Any]:
    """Build a `stratum_by_object_id` map from one metadata field already
    present on a survey's own `SourceRef.extra` (e.g. SDSS's `run2d`
    reduction-run label), instead of requiring the caller to hand-assemble
    the dict. This does not invent a night/camera label where a survey
    carries none -- when `field` is absent on a source, that source's id
    maps to `None`, which `fit_zero_point` treats exactly like an
    unassigned stratum (falls into `"default"`), never a fabricated group.
    """
    stratum: dict[str, Any] = {}
    for source in sources:
        stratum[source.object_id] = source.extra.get(field)
    return stratum


def build_matched_pairs(anchor_survey: str, comparison_survey: str,
                        anchor_band: str, comparison_band: str, *,
                        by_survey: dict[str, list[SourceRef]],
                        radius_arcsec: float = crossmatch.DEFAULT_RADIUS_ARCSEC,
                        color_survey: str | None = None,
                        color_bands: tuple[str, str] | None = None,
                        stratum: Any = None) -> list[dict[str, Any]]:
    """Assemble photometric pairs from `crossmatch.group_sources`.

    Each returned row carries the anchor magnitude, the comparison-survey
    magnitude to be calibrated against it, both uncertainties (when
    available), an optional color index (for the color-term fit) from a
    third survey/band pair, and the caller-supplied `stratum` label (e.g. a
    night or camera identifier) passed through unchanged -- this module
    never invents strata, it only groups by whatever the caller provides.
    """
    groups = crossmatch.group_sources(by_survey, radius_arcsec=radius_arcsec,
                                      anchor_survey=anchor_survey)
    rows: list[dict[str, Any]] = []
    for group in groups:
        anchor_source = group.members.get(anchor_survey)
        comparison_source = group.members.get(comparison_survey)
        if anchor_source is None or comparison_source is None:
            continue
        if comparison_survey in group.blended:
            continue
        anchor_mag, anchor_err = source_magnitude(anchor_source, anchor_band)
        comparison_mag, comparison_err = source_magnitude(comparison_source, comparison_band)
        if anchor_mag is None or comparison_mag is None:
            continue
        color = None
        if color_survey is not None and color_bands is not None:
            color_source = group.members.get(color_survey)
            if color_source is not None:
                blue_mag, _ = source_magnitude(color_source, color_bands[0])
                red_mag, _ = source_magnitude(color_source, color_bands[1])
                if blue_mag is not None and red_mag is not None:
                    color = blue_mag - red_mag
        rows.append({
            "anchor_object_id": anchor_source.object_id,
            "comparison_object_id": comparison_source.object_id,
            "anchor_mag": anchor_mag, "anchor_mag_error": anchor_err,
            "comparison_mag": comparison_mag, "comparison_mag_error": comparison_err,
            "color": color, "stratum": stratum,
            "separation_arcsec": group.separations.get(comparison_survey),
        })
    return rows


def _weighted_ridge_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                        regularization: float) -> np.ndarray:
    """Deterministic weighted ridge least squares -- same solver shape as
    `significance.fit_selection_model`'s ridge-logistic fit, minus the
    sigmoid link (this is a linear, not a classification, model)."""
    design = np.column_stack([np.ones(len(x)), x])
    weight_matrix = np.diag(weights)
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    lhs = design.T @ weight_matrix @ design + float(regularization) * penalty
    rhs = design.T @ weight_matrix @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def fit_zero_point(pairs: Iterable[dict[str, Any]], *,
                   regularization: float = 1.0,
                   min_pairs: int = 10) -> dict[str, Any]:
    """Fit `comparison_mag - anchor_mag = zero_point + color_term * color`
    per stratum, weighted by combined measurement uncertainty.

    A row missing a color value is still usable for the zero-point-only
    special case (color_term forced to 0 by having no color column at all),
    but once *any* row in a stratum carries a color, every row in that
    stratum without one is dropped -- fitting a color term against a subset
    that silently mixes "no color available" with "zero color" would bias
    the coefficient. Pairs with fewer than `min_pairs` in a stratum are
    reported as not-ready rather than fit on too little data to be
    meaningful.
    """
    rows = [row for row in pairs if isinstance(row, dict)]
    by_stratum: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_stratum.setdefault(row.get("stratum"), []).append(row)

    results: dict[str, Any] = {}
    for stratum, stratum_rows in by_stratum.items():
        has_color = any(row.get("color") is not None for row in stratum_rows)
        usable = [row for row in stratum_rows
                 if not has_color or row.get("color") is not None]
        key = "default" if stratum is None else str(stratum)
        if len(usable) < min_pairs:
            results[key] = {"ready": False, "reason": "insufficient matched pairs",
                            "n_pairs": len(usable), "n_available": len(stratum_rows)}
            continue
        deltas = np.asarray([row["comparison_mag"] - row["anchor_mag"] for row in usable])
        colors = np.asarray([row.get("color") or 0.0 for row in usable])
        errors = np.asarray([
            math.sqrt((row.get("anchor_mag_error") or 0.1) ** 2
                     + (row.get("comparison_mag_error") or 0.1) ** 2)
            for row in usable
        ])
        weights = 1.0 / np.clip(errors, 1e-3, None) ** 2
        beta = _weighted_ridge_fit(colors, deltas, weights, regularization)
        predicted = beta[0] + beta[1] * colors
        residuals = deltas - predicted
        n_with_errors = int(sum(1 for row in usable
                                if row.get("anchor_mag_error") is not None
                                and row.get("comparison_mag_error") is not None))
        results[key] = {
            "ready": True,
            "n_pairs": len(usable),
            "n_available": len(stratum_rows),
            "has_color_term": bool(has_color),
            "zero_point": float(beta[0]),
            "color_term": float(beta[1]) if has_color else None,
            "residual_rms": float(np.sqrt(np.mean(residuals ** 2))),
            "uncertainty_coverage": n_with_errors / len(usable) if usable else 0.0,
        }
    return results


def calibrate(anchor_survey: str, comparison_survey: str,
             anchor_band: str, comparison_band: str, *,
             by_survey: dict[str, list[SourceRef]],
             radius_arcsec: float = crossmatch.DEFAULT_RADIUS_ARCSEC,
             color_survey: str | None = None,
             color_bands: tuple[str, str] | None = None,
             stratum_by_object_id: dict[str, Any] | None = None,
             stratum_field: str | None = None,
             regularization: float = 1.0, min_pairs: int = 10) -> dict[str, Any]:
    """Top-level entry point: assemble pairs, fit per stratum, and report.

    `stratum_by_object_id` lets a caller assign each comparison-survey
    object to a night/camera label (e.g. from that survey's own metadata);
    when omitted, every pair falls into one `"default"` stratum and this
    degrades gracefully to a single global zero-point/color-term fit.
    `stratum_field` is a convenience alternative: name one field already
    present on the comparison survey's `SourceRef.extra` (e.g. `"run2d"`
    for SDSS) and `derive_stratum_by_object_id` builds the map
    automatically, rather than requiring the caller to assemble it by
    hand. Passing both raises -- pick one derivation path explicitly.
    """
    if stratum_by_object_id and stratum_field:
        raise ValueError("pass at most one of stratum_by_object_id, stratum_field")
    raw_pairs = build_matched_pairs(
        anchor_survey, comparison_survey, anchor_band, comparison_band,
        by_survey=by_survey, radius_arcsec=radius_arcsec,
        color_survey=color_survey, color_bands=color_bands)
    if stratum_field:
        stratum_by_object_id = derive_stratum_by_object_id(
            by_survey.get(comparison_survey, []), stratum_field)
    if stratum_by_object_id:
        for row in raw_pairs:
            row["stratum"] = stratum_by_object_id.get(row["comparison_object_id"])

    if not raw_pairs:
        return {
            "schema_version": SCHEMA_VERSION, "ready": False,
            "reason": "no matched photometric pairs",
            "anchor_survey": anchor_survey, "comparison_survey": comparison_survey,
            "generated_utc": _now(),
        }

    strata_result = fit_zero_point(raw_pairs, regularization=regularization,
                                   min_pairs=min_pairs)
    n_with_uncertainty = sum(1 for row in raw_pairs
                             if row.get("anchor_mag_error") is not None
                             and row.get("comparison_mag_error") is not None)
    return {
        "schema_version": SCHEMA_VERSION,
        "ready": any(entry.get("ready") for entry in strata_result.values()),
        "anchor_survey": anchor_survey, "comparison_survey": comparison_survey,
        "anchor_band": anchor_band, "comparison_band": comparison_band,
        "n_pairs": len(raw_pairs),
        "uncertainty_coverage": n_with_uncertainty / len(raw_pairs),
        "strata": strata_result,
        "generated_utc": _now(),
    }


def save(payload: dict[str, Any], *, root: Path | None = None,
        name: str = "default") -> Path:
    base = (root or Path.cwd()).resolve() / "results" / "photometric_calibration"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "SCHEMA_VERSION", "SURVEY_BANDS", "source_magnitude", "build_matched_pairs",
    "derive_stratum_by_object_id", "fit_zero_point", "calibrate", "save",
]
