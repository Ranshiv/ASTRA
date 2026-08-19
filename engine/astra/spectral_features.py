"""Bounded, explainable spectral feature extraction.

The module accepts validated arrays or a simple FITS binary table.  It does
not assign an astrophysical class; it exposes continuum, signal quality, and
line/residual statistics for later evidence models.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

FEATURE_SCHEMA_VERSION = 1
MAX_POINTS = 2_000_000


def _finite_arrays(wavelength, flux, error):
    wave = np.asarray(wavelength, dtype=np.float64)
    values = np.asarray(flux, dtype=np.float64)
    errors = np.asarray(error, dtype=np.float64)
    if not (wave.ndim == values.ndim == errors.ndim == 1):
        raise ValueError("spectral columns must be one-dimensional")
    if not (len(wave) == len(values) == len(errors)):
        raise ValueError("spectral columns must have equal lengths")
    if len(wave) == 0 or len(wave) > MAX_POINTS:
        raise ValueError(f"spectrum must contain 1–{MAX_POINTS} points")
    mask = np.isfinite(wave) & np.isfinite(values) & np.isfinite(errors)
    mask &= errors >= 0
    wave, values, errors = wave[mask], values[mask], errors[mask]
    order = np.argsort(wave, kind="stable")
    wave, values, errors = wave[order], values[order], errors[order]
    if len(wave) < 5 or np.any(np.diff(wave) <= 0):
        raise ValueError("spectrum needs at least five strictly increasing finite wavelengths")
    return wave, values, errors


def extract(wavelength, flux, error, *, frame: str = "unknown",
            units: str = "unknown", source: dict | None = None) -> dict:
    wave, values, errors = _finite_arrays(wavelength, flux, error)
    window = max(5, min(101, (len(values) // 20) * 2 + 1))
    kernel = np.ones(window, dtype=np.float64) / window
    continuum = np.convolve(values, kernel, mode="same")
    residual = values - continuum
    fallback_sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    fallback_sigma = float(fallback_sigma) if np.isfinite(fallback_sigma) and fallback_sigma > 0 else 1.0
    effective_error = np.where(errors > 0, errors, fallback_sigma)
    snr = residual / effective_error
    positive = snr >= 5.0
    negative = snr <= -5.0
    spacing = float(np.median(np.diff(wave)))
    continuum_scale = np.maximum(np.abs(continuum), np.finfo(float).eps)

    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "source": source or {},
        "frame": frame,
        "units": units,
        "features": {
            "points": int(len(wave)),
            "wavelength_start": float(wave[0]),
            "wavelength_end": float(wave[-1]),
            "coverage": float(wave[-1] - wave[0]),
            "median_continuum": float(np.median(continuum)),
            "median_snr": float(np.median(values / effective_error)),
            "residual_robust_sigma": float(fallback_sigma),
            "max_positive_line_snr": float(np.max(snr)),
            "max_negative_line_snr": float(np.min(snr)),
            "emission_peak_count": int(np.count_nonzero(positive)),
            "absorption_peak_count": int(np.count_nonzero(negative)),
            "positive_equivalent_width_proxy": float(np.sum(np.clip(residual, 0, None) / continuum_scale) * spacing),
            "negative_equivalent_width_proxy": float(np.sum(np.clip(-residual, 0, None) / continuum_scale) * spacing),
            "invalid_error_fraction": float(np.mean(errors <= 0)),
        },
        "quality": {
            "method": "median-continuum-residual-thresholds",
            "line_threshold_sigma": 5.0,
            "error_fallback": bool(np.any(errors <= 0)),
        },
    }
    return payload


def from_fits(path: str | Path, hdu: int | None = None) -> dict:
    from astropy.io import fits

    source_path = Path(path).resolve()
    with fits.open(source_path, memmap=True) as hdul:
        table_index = hdu
        if table_index is None:
            table_index = next((i for i, item in enumerate(hdul)
                                if getattr(item, "data", None) is not None
                                and getattr(item, "columns", None) is not None), None)
        if table_index is None:
            raise ValueError("no FITS binary table found")
        data = hdul[table_index].data
        names = {str(name).lower(): str(name) for name in data.names or []}
        def column(*choices):
            for choice in choices:
                if choice in names:
                    return data[names[choice]]
            raise ValueError(f"FITS spectrum is missing one of: {', '.join(choices)}")
        wave = column("wavelength", "wave", "lambda")
        flux = column("flux", "flambda", "spec")
        error = column("error", "flux_error", "sigma", "ivar")
        if "ivar" in names and "error" not in names and "flux_error" not in names and "sigma" not in names:
            inverse = np.asarray(error, dtype=np.float64)
            error = np.where(inverse > 0, 1.0 / np.sqrt(inverse), 0.0)
        header = hdul[table_index].header
    payload = extract(wave, flux, error, frame=str(header.get("FRAME", "unknown")),
                      units=str(header.get("BUNIT", "unknown")),
                      source={"path": str(source_path), "hdu": table_index,
                              "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()})
    return payload


def save(payload: dict, root: str | Path) -> Path:
    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    digest = str(payload.get("source", {}).get("sha256", "unknown"))[:32]
    target = destination / f"spectral_features_{digest}.json"
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=destination)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
