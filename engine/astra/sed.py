"""Bounded broadband SED diagnostics.

This is a physical-context layer for candidate review, not a stellar
classifier.  It combines available catalogue magnitudes into a rough color
temperature and a grid-fit blackbody residual.  Extinction can be supplied by
the caller; ASTRA does not invent a reddening correction when one is absent.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import candidates, config

SCHEMA_VERSION = 1
_C2_NM_K = 1.438776877e7
BANDS_NM = {
    "gaia_bp": 532.0, "g": 477.0, "ztf_g": 477.0,
    "gaia_g": 673.0, "gaia_rp": 797.0, "r": 623.0, "ztf_r": 623.0,
    "i": 763.0, "ztf_i": 763.0, "z": 905.0, "y": 971.0, "tess": 786.0,
}
ALIASES = {
    "bp": "gaia_bp", "bp_mag": "gaia_bp", "gaia_bp_mag": "gaia_bp",
    "g_mag": "g", "r_mag": "r", "i_mag": "i", "t_mag": "tess",
    "gaia_g_mag": "gaia_g", "rp": "gaia_rp", "rp_mag": "gaia_rp",
    "gaia_rp_mag": "gaia_rp",
}


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical_photometry(photometry: Mapping[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_name, raw_value in photometry.items():
        name = str(raw_name).strip().lower()
        name = ALIASES.get(name, name)
        if name not in BANDS_NM:
            continue
        value = _finite(raw_value)
        if value is not None and -10 < value < 50:
            result[name] = value
    return result


def _color_temperature(color: float) -> float | None:
    # Ballesteros' two-color approximation; clamp to the range where the
    # approximation remains useful rather than extrapolating absurd values.
    if not math.isfinite(color) or not -0.4 <= color <= 3.0:
        return None
    return 4600.0 * (1.0 / (0.92 * color + 1.7) + 1.0 / (0.92 * color + 0.62))


def _blackbody_temperature(wavelength_nm: np.ndarray, magnitudes: np.ndarray) -> tuple[float | None, float | None]:
    if len(wavelength_nm) < 3:
        return None, None
    observed = 10.0 ** (-0.4 * (magnitudes - np.nanmedian(magnitudes)))
    grid = np.geomspace(2500.0, 50000.0, 192)
    best_temperature = None
    best_residual = None
    for temperature in grid:
        exponent = np.clip(_C2_NM_K / (wavelength_nm * temperature), 1e-6, 700)
        model = wavelength_nm ** -5 / np.expm1(exponent)
        model = model / np.nanmedian(model)
        residual = float(np.sqrt(np.nanmean((np.log(np.clip(observed, 1e-12, None))
                                             - np.log(np.clip(model, 1e-12, None))) ** 2)))
        if best_residual is None or residual < best_residual:
            best_temperature, best_residual = temperature, residual
    return best_temperature, best_residual


def characterize(photometry: Mapping[str, object], *, extinction: Mapping[str, object] | None = None,
                 source: str = "caller") -> dict[str, Any]:
    values = _canonical_photometry(photometry)
    extinction = extinction or {}
    corrected = dict(values)
    extinction_used = {}
    for band, value in values.items():
        correction = _finite(extinction.get(band))
        if correction is not None:
            corrected[band] = value - correction
            extinction_used[band] = correction
    colors = {}
    temperatures = []
    for left, right, label in (
        ("gaia_bp", "gaia_rp", "gaia_bp_rp"),
        ("g", "r", "g_r"), ("ztf_g", "ztf_r", "ztf_g_r"),
        ("r", "i", "r_i"),
    ):
        if left in corrected and right in corrected:
            color = corrected[left] - corrected[right]
            colors[label] = round(float(color), 5)
            temperature = _color_temperature(color)
            if temperature is not None:
                temperatures.append(temperature)
    bands = sorted((band for band in corrected if band in BANDS_NM), key=lambda item: BANDS_NM[item])
    wavelengths = np.asarray([BANDS_NM[band] for band in bands], dtype=np.float64)
    magnitudes = np.asarray([corrected[band] for band in bands], dtype=np.float64)
    blackbody_temperature, residual = _blackbody_temperature(wavelengths, magnitudes)
    all_temperatures = temperatures + ([blackbody_temperature] if blackbody_temperature else [])
    temperature = float(np.median(all_temperatures)) if all_temperatures else None
    uncertainty = float(np.percentile(all_temperatures, 75) - np.percentile(all_temperatures, 25)) \
        if len(all_temperatures) >= 2 else None
    warnings = []
    if len(bands) < 3:
        warnings.append("fewer than three bands: blackbody residual is unavailable or weak")
    if not extinction_used:
        warnings.append("no extinction correction supplied")
    if residual is not None and residual > 0.35:
        warnings.append("broadband colors are inconsistent with a single-temperature blackbody")
    return {
        "schema_version": SCHEMA_VERSION, "source": source,
        "bands_used": bands, "photometry": corrected, "colors": colors,
        "temperature_k": round(temperature, 2) if temperature is not None else None,
        "temperature_spread_k": round(uncertainty, 2) if uncertainty is not None else None,
        "blackbody_temperature_k": round(blackbody_temperature, 2) if blackbody_temperature else None,
        "sed_residual_rms": round(float(residual), 6) if residual is not None else None,
        "extinction_applied": extinction_used, "quality": "usable" if temperature else "insufficient",
        "warnings": warnings,
    }


def characterize_candidate(name: str = "default", *, root: Path | None = None,
                           extinction: Mapping[str, object] | None = None) -> dict[str, Any]:
    """Attach SED context to candidates from already stored feature/catalogue values."""
    root = root or config.PATHS.projects
    built = candidates.load(name, root)
    count = 0
    for candidate in built:
        photometry: dict[str, object] = {}
        for key, value in candidate.features.items():
            normalized = str(key).lower()
            if normalized in BANDS_NM or normalized in ALIASES:
                photometry[normalized] = value
        catalog = candidate.catalog if isinstance(candidate.catalog, dict) else {}
        for provider in ("simbad", "vsx", "tns"):
            for match in (catalog.get("providers", {}).get(provider, {}).get("matches", [])
                          if isinstance(catalog.get("providers", {}).get(provider), dict) else []):
                if isinstance(match, dict):
                    for key, value in match.items():
                        if str(key).lower() in BANDS_NM or str(key).lower() in ALIASES:
                            photometry[str(key).lower()] = value
        result = characterize(photometry, extinction=extinction, source="candidate_features")
        candidate.physical_characterization = result
        candidate.explanation["physical_characterization"] = {
            "quality": result["quality"], "bands_used": result["bands_used"],
            "temperature_k": result["temperature_k"],
        }
        candidate.provenance_refs.append({"kind": "physical_characterization",
                                          "method": "broadband_sed_blackbody_grid",
                                          "schema_version": SCHEMA_VERSION})
        count += result["quality"] == "usable"
    path = candidates.save(built, name, root)
    return {"name": name, "candidates": len(built), "usable": count,
            "output_path": str(path)}
