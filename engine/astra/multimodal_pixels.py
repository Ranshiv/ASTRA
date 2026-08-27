"""Bounded, fixed-size pixel arrays for the multimodal image branch
(backlog item 11).

`fitsio.raw_pixel_array()` and `tess_pixels.read_tpf_cube()` (added
alongside this module) expose real, unstretched float pixel data for the
first time in this codebase -- neither `image_features.extract()` nor
`fitsio.image_payload()` previously returned an array to a caller, only
scalar summaries or an 8-bit display quantization. This module turns those
real arrays into ONE fixed shape a CNN batch can consume, regardless of
which survey (ZTF cutout or TESS TPF) an object came from.

Deliberately NOT the same stretch `fitsio.py`'s zscale/8-bit pipeline uses:
that stretch is tuned for a human looking at a canvas and is lossy
(quantized to 256 levels). `preprocess_image()` uses an arcsinh stretch
instead -- the standard astronomical convention for compressing dynamic
range without a hard floor clip, appropriate for a gradient-based encoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_INPUT_SIZE = 32


def ztf_cutout_array(path: str | Path, hdu: int | None = None) -> np.ndarray:
    """Real, bounded float pixel array from a stored ZTF cutout FITS."""
    from . import fitsio

    return fitsio.raw_pixel_array(path, hdu=hdu, max_dimension=IMAGE_INPUT_SIZE)


def tess_reference_frame(path: str | Path, quality_mask: int = 0) -> np.ndarray:
    """Median frame across cadences passing the quality mask.

    Same `(quality & quality_mask) == 0` convention
    `tess_pixels.extract_photometry()` already uses, so "good cadence"
    means the same thing in both places. A cadence whose frame is entirely
    non-finite is excluded even if its quality flag passed, since a frame
    of NaNs would otherwise poison the per-pixel median.
    """
    from . import tess_pixels

    data = tess_pixels.read_tpf_cube(path)
    cube = data["flux"]
    quality = np.asarray(data["quality"], dtype=np.uint64)
    good = (quality & np.uint64(quality_mask)) == 0
    good &= np.any(np.isfinite(cube), axis=(1, 2))
    if not np.any(good):
        raise ValueError("no cadence passes the quality mask")
    return np.nanmedian(cube[good], axis=0)


def to_fixed_size(array: np.ndarray, size: int = IMAGE_INPUT_SIZE,
                  fill: float | None = None) -> np.ndarray:
    """Center-crop or background-pad `array` to `(size, size)`.

    ZTF cutouts and TESS TPFs have different native shapes; a CNN batch
    needs one fixed shape. `fill` defaults to the array's own robust
    background estimate (median minus the MAD-based convention
    `image_features.py` already uses for background level), so padding
    reads as "more background," not a fabricated zero that could sit far
    outside the frame's real flux range.
    """
    array = np.asarray(array, dtype=np.float64)
    if fill is None:
        finite = array[np.isfinite(array)]
        fill = float(np.median(finite)) if finite.size else 0.0

    height, width = array.shape
    out = np.full((size, size), fill, dtype=np.float64)

    # Crop first (if larger than target), then paste centered (if smaller).
    crop_h, crop_w = min(height, size), min(width, size)
    src_y0 = max(0, (height - crop_h) // 2)
    src_x0 = max(0, (width - crop_w) // 2)
    dst_y0 = max(0, (size - crop_h) // 2)
    dst_x0 = max(0, (size - crop_w) // 2)
    out[dst_y0:dst_y0 + crop_h, dst_x0:dst_x0 + crop_w] = (
        array[src_y0:src_y0 + crop_h, src_x0:src_x0 + crop_w])
    return out


def preprocess_image(array: np.ndarray) -> np.ndarray:
    """Background-subtract (robust median) then arcsinh-stretch.

    Non-finite input pixels are replaced by the frame's own robust
    background before the stretch, so a single bad pixel cannot propagate
    a NaN through the whole encoder input.
    """
    array = np.asarray(array, dtype=np.float64)
    finite = array[np.isfinite(array)]
    background = float(np.median(finite)) if finite.size else 0.0
    mad = float(np.median(np.abs(finite - background))) * 1.4826 if finite.size else 1.0
    scale = mad if mad > 0 else 1.0

    cleaned = np.where(np.isfinite(array), array, background)
    return np.arcsinh((cleaned - background) / scale).astype(np.float32)
