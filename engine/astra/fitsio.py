"""FITS inspection for the image and FITS viewers (plan section 29, phase 3).

Astronomical images have enormous dynamic range — a saturated star and the
sky background can differ by five orders of magnitude — so a linear map to
256 grey levels shows an almost black frame with a few white dots. ZScale is
the stretch DS9 and every other astronomy viewer defaults to, and it is what
makes the faint structure a researcher needs to judge an artifact visible.

Pixels are returned as plain lists for the interface to render; no image file
is written, so nothing here consumes the storage budget.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Larger frames are decimated before transport: a full ZTF quadrant is
# 3080x3072, which is 9.5 million values and pointless to send to a canvas
# that is a few hundred pixels across.
MAX_DIMENSION = 512

# Header cards worth surfacing; the full header is available separately.
INTERESTING_KEYS = (
    "OBJECT", "TELESCOP", "INSTRUME", "FILTER", "DATE-OBS", "MJD-OBS",
    "EXPTIME", "NAXIS1", "NAXIS2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
    "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
    "BUNIT", "SECTOR",
)


def describe(path: str | Path) -> dict:
    """List the HDUs in a FITS file without loading the pixel data."""
    from astropy.io import fits

    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        hdus = []
        for index, hdu in enumerate(hdul):
            shape = getattr(hdu, "shape", None)
            hdus.append({
                "index": index,
                "name": hdu.name,
                "type": type(hdu).__name__,
                "shape": list(shape) if shape else [],
                "is_image": bool(shape) and len(shape) == 2,
            })
    return {"path": str(path), "hdus": hdus,
            "size_mb": round(path.stat().st_size / 1024 ** 2, 3)}


def read_header(path: str | Path, hdu: int = 0) -> dict:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        header = hdul[hdu].header
        full = {}
        for key in header:
            if not key:
                continue
            try:
                value = header[key]
            except Exception:  # noqa: BLE001 - malformed cards are skipped
                continue
            full[key] = value if isinstance(value, (int, float, bool, str)) \
                else str(value)

    return {
        "hdu": hdu,
        "summary": {k: full[k] for k in INTERESTING_KEYS if k in full},
        "cards": full,
    }


def zscale(data: np.ndarray, contrast: float = 0.25) -> tuple[float, float]:
    """Compute display limits the way astronomical viewers do.

    Falls back to a percentile clip when astropy's implementation cannot
    converge, which happens on nearly uniform frames such as flat fields.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0

    try:
        from astropy.visualization import ZScaleInterval

        low, high = ZScaleInterval(contrast=contrast).get_limits(finite)
        if np.isfinite(low) and np.isfinite(high) and high > low:
            return float(low), float(high)
    except Exception:  # noqa: BLE001 - fall through to the percentile clip
        pass

    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def decimate(data: np.ndarray, max_dimension: int = MAX_DIMENSION) -> np.ndarray:
    """Reduce a large frame by integer striding, preserving orientation."""
    height, width = data.shape
    step = max(1, int(np.ceil(max(height, width) / max_dimension)))
    return data[::step, ::step] if step > 1 else data


def image_payload(path: str | Path, hdu: int | None = None,
                  contrast: float = 0.25,
                  max_dimension: int = MAX_DIMENSION) -> dict:
    """Return a display-ready 8-bit frame plus the statistics behind it."""
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        index = hdu if hdu is not None else _first_image_hdu(hdul)
        if index is None:
            raise ValueError(f"no 2-D image HDU found in {path}")
        data = np.asarray(hdul[index].data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"HDU {index} is not 2-D (shape {data.shape})")

    original_shape = list(data.shape)
    reduced = decimate(data, max_dimension)
    low, high = zscale(reduced, contrast)

    scaled = np.clip((reduced - low) / (high - low), 0.0, 1.0)
    as_bytes = (scaled * 255.0).astype(np.uint8)

    finite = reduced[np.isfinite(reduced)]
    return {
        "hdu": index,
        "shape": list(reduced.shape),
        "original_shape": original_shape,
        "decimated": reduced.shape != tuple(original_shape),
        "vmin": low,
        "vmax": high,
        "stats": {
            "min": float(np.min(finite)) if finite.size else 0.0,
            "max": float(np.max(finite)) if finite.size else 0.0,
            "median": float(np.median(finite)) if finite.size else 0.0,
            "std": float(np.std(finite)) if finite.size else 0.0,
        },
        # Row-major, one byte per pixel; the interface paints this to a canvas.
        "pixels": as_bytes.flatten().tolist(),
    }


def _first_image_hdu(hdul) -> int | None:
    for index, hdu in enumerate(hdul):
        shape = getattr(hdu, "shape", None)
        if shape and len(shape) == 2 and hdu.data is not None:
            return index
    return None
