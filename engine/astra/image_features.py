"""Bounded image-derived features for validated FITS products.

This is deliberately a transparent morphology/environment baseline, not a
claim of PSF photometry.  It consumes one image HDU, records the product hash
and extraction contract, and returns nullable measurements when a frame cannot
support a feature.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

FEATURE_SCHEMA_VERSION = 1
MAX_PIXELS = 16_000_000


def _first_image(data: object) -> np.ndarray:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"image data must be 2-D, got shape {array.shape}")
    if array.size > MAX_PIXELS:
        raise ValueError(f"image has {array.size} pixels; limit is {MAX_PIXELS}")
    return array


def _robust_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


def _round(value: float | None, digits: int = 8) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def extract(path: str | Path, hdu: int | None = None,
            target_xy: tuple[float, float] | None = None) -> dict:
    """Extract morphology, local-environment and residual statistics."""
    from astropy.io import fits

    source_path = Path(path).resolve()
    with fits.open(source_path, memmap=True) as hdul:
        index = hdu if hdu is not None else next(
            (i for i, item in enumerate(hdul)
             if getattr(item, "shape", None) and len(item.shape) == 2 and item.data is not None),
            None,
        )
        if index is None:
            raise ValueError(f"no 2-D image HDU found in {source_path}")
        data = _first_image(hdul[index].data)
        header = hdul[index].header

    finite_mask = np.isfinite(data)
    finite = data[finite_mask]
    if finite.size == 0:
        raise ValueError("image contains no finite pixels")

    background = float(np.median(finite))
    background_sigma = _robust_scale(finite)
    if not np.isfinite(background_sigma) or background_sigma <= 0:
        background_sigma = float(np.std(finite)) or 1.0
    residual = data - background
    threshold = background + 5.0 * background_sigma
    detections = finite_mask & (data >= threshold)
    ys, xs = np.nonzero(detections)
    weights = np.clip(residual[detections], 0.0, None)
    total_weight = float(weights.sum())
    if total_weight > 0 and len(xs):
        centroid_x = float(np.average(xs, weights=weights))
        centroid_y = float(np.average(ys, weights=weights))
        dx = xs - centroid_x
        dy = ys - centroid_y
        xx = float(np.average(dx * dx, weights=weights))
        yy = float(np.average(dy * dy, weights=weights))
        xy = float(np.average(dx * dy, weights=weights))
        trace = xx + yy
        ellipticity = (float(np.sqrt((xx - yy) ** 2 + 4 * xy ** 2) / trace)
                       if trace > 0 else float("nan"))
    else:
        centroid_x = centroid_y = ellipticity = float("nan")
        xx = yy = xy = float("nan")

    peak = float(np.nanmax(data))
    local_snr = (peak - background) / background_sigma
    if target_xy is None:
        target_distance = float("nan")
    else:
        target_distance = float(np.hypot(centroid_x - target_xy[0], centroid_y - target_xy[1]))

    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "source": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "hdu": index,
            "shape": list(data.shape),
            "filter": str(header.get("FILTER", "")),
        },
        "features": {
            "background_median": _round(background),
            "background_robust_sigma": _round(background_sigma),
            "finite_fraction": _round(float(finite_mask.mean()), 6),
            "detected_pixel_fraction": _round(float(detections.mean()), 6),
            "detected_pixel_count": int(len(xs)),
            "peak_value": _round(peak),
            "peak_snr": _round(local_snr),
            "positive_residual_flux": _round(total_weight),
            "centroid_x": _round(centroid_x),
            "centroid_y": _round(centroid_y),
            "moment_xx": _round(xx),
            "moment_yy": _round(yy),
            "moment_xy": _round(xy),
            "ellipticity": _round(ellipticity),
            "target_centroid_distance_pixels": _round(target_distance),
            "residual_robust_sigma": _round(_robust_scale(residual[finite_mask])),
        },
        "quality": {
            "finite_pixels": int(finite.size),
            "threshold_sigma": 5.0,
            "method": "background-median-mad-thresholded-moments",
        },
    }
    return payload


def save(payload: dict, root: str | Path) -> Path:
    """Atomically persist one feature payload under a managed results root."""
    destination_root = Path(root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    digest = str(payload.get("source", {}).get("sha256", "unknown"))[:32]
    target = destination_root / f"image_features_{digest}.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=destination_root)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
