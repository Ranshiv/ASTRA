"""Bounded TESS target-pixel acquisition and blend-aware photometry.

Bulk TESS light curves remain the cheap survey connector in
``surveys.tess``.  This module is the explicit, candidate-scale path for a
target pixel file (TPF): one coordinate, one sector, one bounded TESScut
request.  Raw FITS files are immutable and accompanied by provenance; the
photometry extractor never presents a TESS aperture as resolved stellar
photometry without reporting its 21-arcsec pixel/blend limitation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import numbers
import re
import stat
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import numpy as np

from . import config, netclient, store
from .surveys.base import LightCurve, SourceRef

TESScUT_URL = "https://mast.stsci.edu/tesscut/api/v0.1/astrocut"
DEFAULT_SIZE_PIXELS = 20
MIN_SIZE_PIXELS = 2
MAX_SIZE_PIXELS = 50
TESS_PIXEL_SCALE_ARCSEC = 21.0
DEFAULT_MAX_BYTES = 128 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
DEFAULT_QUALITY_MASK = 0

_INTEGER_RE = re.compile(r"^[+-]?\d+$")


class TESSProductError(ValueError):
    """A TPF request or file is invalid or cannot be interpreted safely."""


def _finite_float(value: object, name: str) -> float:
    """Convert a scalar to a finite float without leaking TypeError details."""
    if isinstance(value, bool):
        raise TESSProductError(f"{name} must be a number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TESSProductError(f"{name} must be a number") from exc
    if not math.isfinite(converted):
        raise TESSProductError(f"{name} must be finite")
    return converted


def _strict_int(value: object, name: str, *, minimum: int | None = None,
                maximum: int | None = None) -> int:
    """Parse an integer-valued scalar without truncating 1.5 to 1."""
    if isinstance(value, bool):
        raise TESSProductError(f"{name} must be an integer")
    if isinstance(value, str):
        text = value.strip()
        if not _INTEGER_RE.fullmatch(text):
            raise TESSProductError(f"{name} must be an integer")
        converted = int(text)
    elif isinstance(value, numbers.Integral):
        # Preserve large uint64 masks exactly; routing every value through a
        # float would round integers above 2**53.
        converted = int(value)
    elif isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise TESSProductError(f"{name} must be an integer")
        converted = int(numeric)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TESSProductError(f"{name} must be an integer") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise TESSProductError(f"{name} must be an integer")
        converted = int(numeric)
    if minimum is not None and converted < minimum:
        raise TESSProductError(f"{name} must be at least {minimum}")
    if maximum is not None and converted > maximum:
        raise TESSProductError(f"{name} must be at most {maximum}")
    return converted


def _normalise_column_name(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    return str(value).strip().upper()


def _column(data: Any, wanted: str) -> Any:
    """Return a FITS table column with case/whitespace tolerant lookup."""
    names = getattr(data, "names", None) or ()
    expected = _normalise_column_name(wanted)
    for name in names:
        if _normalise_column_name(name) == expected:
            return data[name]
    raise TESSProductError(f"TPF column {wanted} is missing")


def _optional_column(data: Any, wanted: str) -> Any | None:
    try:
        return _column(data, wanted)
    except TESSProductError:
        return None


def _header_sector(header: Any) -> int | None:
    """Read a sector header, treating blank/zero values as unknown."""
    raw = header.get("SECTOR") if header is not None else None
    if raw in (None, ""):
        return None
    try:
        sector = _strict_int(raw, "SECTOR", minimum=1, maximum=999)
    except TESSProductError as exc:
        # A few synthetic/legacy files carry SECTOR=0 or a blank placeholder;
        # preserve the fact that it is unknown rather than publishing a false
        # sector. Any other malformed value is a corrupt product.
        try:
            numeric = float(raw)
        except (TypeError, ValueError, OverflowError):
            raise TESSProductError("TPF SECTOR header is invalid") from exc
        if numeric == 0:
            return None
        raise TESSProductError("TPF SECTOR header is invalid") from exc
    return sector


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TPFRequest:
    ra_deg: float
    dec_deg: float
    sector: int
    size_pixels: int = DEFAULT_SIZE_PIXELS
    target_id: str | None = None
    product: str = "SPOC"

    def __post_init__(self) -> None:
        ra = _finite_float(self.ra_deg, "ra_deg")
        dec = _finite_float(self.dec_deg, "dec_deg")
        if not 0.0 <= ra < 360.0:
            raise TESSProductError("ra_deg must be finite and in [0, 360)")
        if not -90.0 <= dec <= 90.0:
            raise TESSProductError("dec_deg must be finite and in [-90, 90]")
        _strict_int(self.sector, "sector", minimum=1, maximum=999)
        _strict_int(self.size_pixels, "size_pixels",
                    minimum=MIN_SIZE_PIXELS, maximum=MAX_SIZE_PIXELS)
        if str(self.product).upper() != "SPOC":
            raise TESSProductError("only SPOC TESScut products are supported")
        if self.target_id is not None:
            target_id = str(self.target_id)
            if len(target_id) > 128:
                raise TESSProductError("target_id is too long")
            if "\x00" in target_id:
                raise TESSProductError("target_id contains a NUL character")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ra_deg": round(float(self.ra_deg), 8),
            "dec_deg": round(float(self.dec_deg), 8),
            "sector": _strict_int(self.sector, "sector", minimum=1, maximum=999),
            "size_pixels": _strict_int(
                self.size_pixels, "size_pixels",
                minimum=MIN_SIZE_PIXELS, maximum=MAX_SIZE_PIXELS,
            ),
            "target_id": self.target_id,
            "product": str(self.product).upper(),
        }

    def params(self) -> dict[str, str]:
        """Parameters used by the public MAST TESScut service."""
        ra = _finite_float(self.ra_deg, "ra_deg")
        dec = _finite_float(self.dec_deg, "dec_deg")
        return {
            "ra": f"{ra:.8f}",
            "dec": f"{dec:.8f}",
            "x": str(_strict_int(
                self.size_pixels, "size_pixels",
                minimum=MIN_SIZE_PIXELS, maximum=MAX_SIZE_PIXELS,
            )),
            "y": str(_strict_int(
                self.size_pixels, "size_pixels",
                minimum=MIN_SIZE_PIXELS, maximum=MAX_SIZE_PIXELS,
            )),
            "units": "px",
            "sector": str(_strict_int(self.sector, "sector", minimum=1, maximum=999)),
        }

    def url(self) -> str:
        return f"{TESScUT_URL}?{urlencode(self.params())}"

    def product_id(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True,
                               separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_zip_member(name: str) -> bool:
    normal = name.replace("\\", "/")
    # Reject absolute paths, drive-qualified names, traversal, and NULs even
    # though extraction never uses the archive's path. Checking every member
    # (rather than only the selected FITS) prevents a malicious companion entry
    # from becoming dangerous if extraction logic changes later.
    drive, _tail = normal[:2], normal[2:]
    return (
        bool(normal)
        and "\x00" not in normal
        and not normal.startswith("/")
        and not (len(drive) == 2 and drive[1] == ":")
        and ".." not in normal.split("/")
    )


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    """Detect Unix symlink entries (including symlinks named *.fits)."""
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _copy_bounded(source, target, limit: int) -> int:
    written = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        written += len(chunk)
        if written > limit:
            raise TESSProductError("uncompressed TPF exceeds the safety limit")
        target.write(chunk)
    return written


def _extract_fits(archive: Path, destination: Path) -> str:
    """Extract exactly one FITS payload from a TESScut zip, safely."""
    if not zipfile.is_zipfile(archive):
        with archive.open("rb") as source, destination.open("wb") as target:
            _copy_bounded(source, target, MAX_UNCOMPRESSED_BYTES)
        return archive.name

    try:
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            for member in members:
                if not _safe_zip_member(member.filename):
                    raise TESSProductError("TESScut zip contains an unsafe member path")
                if _zip_member_is_symlink(member):
                    raise TESSProductError("TESScut zip contains a symlink member")

            candidates = [info for info in members
                          if not info.is_dir()
                          and info.filename.lower().endswith((".fits", ".fits.gz"))]
            if len(candidates) != 1:
                raise TESSProductError(
                    f"TESScut response contains {len(candidates)} FITS payloads; "
                    "a single-sector request must be unambiguous"
                )
            info = candidates[0]
            if info.file_size > MAX_UNCOMPRESSED_BYTES:
                raise TESSProductError("uncompressed TPF exceeds the safety limit")
            with zipped.open(info, "r") as source, destination.open("wb") as target:
                if info.filename.lower().endswith(".gz"):
                    with gzip.GzipFile(fileobj=source, mode="rb") as decompressed:
                        _copy_bounded(decompressed, target, MAX_UNCOMPRESSED_BYTES)
                else:
                    _copy_bounded(source, target, MAX_UNCOMPRESSED_BYTES)
    except TESSProductError:
        raise
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise TESSProductError("TESScut archive could not be safely extracted") from exc
    if destination.stat().st_size > MAX_UNCOMPRESSED_BYTES:
        raise TESSProductError("uncompressed TPF exceeds the safety limit")
    return info.filename


def _find_table(hdul):
    for index, hdu in enumerate(hdul):
        columns = getattr(hdu, "columns", None)
        names = {
            _normalise_column_name(name)
            for name in (getattr(columns, "names", None) or ())
        }
        # Some lightweight FITS table implementations expose names only on
        # ``data``. Supporting both keeps validation independent of Astropy's
        # particular HDU class and makes fixtures representative of real files.
        names.update(
            _normalise_column_name(name)
            for name in (getattr(getattr(hdu, "data", None), "names", None) or ())
        )
        if {"TIME", "FLUX"} <= names:
            return index, hdu
    raise TESSProductError("FITS file is not a TESS target-pixel file (TIME/FLUX missing)")


def _wcs_from_hdul(hdul):
    from astropy.io import fits
    from astropy.wcs import WCS

    # TESScut products generally carry a conventional WCS in the aperture
    # extension; standard SPOC TPFs encode it with numbered column keywords.
    # Prefer later extensions (where the aperture/table WCS normally lives),
    # but do not assume a fixed HDU count.
    for index in range(len(hdul) - 1, -1, -1):
        header = hdul[index].header
        if not any(key in header for key in ("CTYPE1", "CTYPE2", "WCSAXES")):
            continue
        try:
            candidate = WCS(header)
            if candidate.has_celestial:
                celestial = getattr(candidate, "celestial", candidate)
                return celestial, index
        except Exception:  # noqa: BLE001 - fall through to the numbered form
            pass

    mapping = {}
    for axis in (1, 2):
        for keyword, replacement in (
            ("CTYP", "CTYPE"), ("CRPX", "CRPIX"),
            ("CRVL", "CRVAL"), ("CUNI", "CUNIT"),
            ("CDLT", "CDELT"),
        ):
            mapping[f"{axis}{keyword}5"] = f"{replacement}{axis}"
    for first, second in ((1, 1), (1, 2), (2, 1), (2, 2)):
        mapping[f"{first}{second}PC5"] = f"PC{first}_{second}"
    for index in range(len(hdul) - 1, -1, -1):
        source = hdul[index].header
        if not any(key in source for key in mapping):
            continue
        header = fits.Header()
        for old, new in mapping.items():
            if old in source:
                header[new] = source[old]
        try:
            candidate = WCS(header)
            if candidate.has_celestial:
                return getattr(candidate, "celestial", candidate), index
        except Exception:  # noqa: BLE001
            pass
    return None, None


def _tpf_summary(path: Path, *, expected_sector: int | None = None) -> dict[str, Any]:
    from astropy.io import fits

    # ``memmap=True`` leaves a file mapping alive briefly after the HDU closes
    # on Windows, which prevents the validated staging FITS from being moved
    # atomically into the canonical store. Candidate TPFs are bounded, so the
    # small validation read is worth the deterministic close semantics.
    with fits.open(path, memmap=False, lazy_load_hdus=False) as hdul:
        hdul.verify("exception")
        table_index, table_hdu = _find_table(hdul)
        data = table_hdu.data
        flux = np.asarray(_column(data, "FLUX"))
        if flux.ndim != 3:
            raise TESSProductError(f"TPF FLUX column has unexpected shape {flux.shape}")
        time = np.asarray(_column(data, "TIME"))
        if time.ndim != 1 or len(time) != flux.shape[0]:
            raise TESSProductError("TPF TIME and FLUX cadence dimensions do not match")
        wcs, wcs_hdu = _wcs_from_hdul(hdul)
        header = table_hdu.header
        sector = _header_sector(header)
        if sector is None:
            sector = _header_sector(hdul[0].header)
        if expected_sector is not None:
            expected = _strict_int(expected_sector, "sector", minimum=1, maximum=999)
            if sector is not None and sector != expected:
                raise TESSProductError(
                    f"TPF sector header {sector} does not match requested sector {expected}"
                )
            sector = sector or expected
        return {
            "table_hdu": table_index,
            "wcs_hdu": wcs_hdu,
            "cadences": int(flux.shape[0]),
            "pixel_shape": [int(flux.shape[1]), int(flux.shape[2])],
            "sector": sector,
            "target": hdul[0].header.get("OBJECT", hdul[0].header.get("TICID")),
            "wcs": {
                "has_celestial": bool(wcs is not None),
                "pixel_scale_arcsec": TESS_PIXEL_SCALE_ARCSEC,
            },
        }


def download_tpf(request: TPFRequest, *, root: Path | None = None,
                 project_id: str | None = None,
                 max_bytes: int = DEFAULT_MAX_BYTES,
                 overwrite: bool = False,
                 progress=None) -> dict[str, Any]:
    """Fetch one TESScut TPF, validate it, and publish it atomically."""
    max_bytes = _strict_int(max_bytes, "max_bytes", minimum=1,
                            maximum=DEFAULT_MAX_BYTES)
    if not isinstance(request, TPFRequest):
        raise TESSProductError("request must be a TPFRequest")
    if not isinstance(overwrite, bool):
        raise TESSProductError("overwrite must be a boolean")
    if project_id is not None and "\x00" in str(project_id):
        raise TESSProductError("project_id contains a NUL character")
    if max_bytes <= 0 or max_bytes > DEFAULT_MAX_BYTES:
        raise TESSProductError(f"max_bytes must be between 1 and {DEFAULT_MAX_BYTES}")
    if project_id:
        from . import project

        project.require_active(project_id)

    root = root or config.PATHS.datasets
    request_sector = _strict_int(request.sector, "sector", minimum=1, maximum=999)
    product_id = request.product_id()
    destination = (root / "TESS" / "tpf" / f"sector-{request_sector}"
                   / product_id[:2] / f"{product_id}.fits")
    sidecar = _sidecar(destination)
    if destination.exists() and sidecar.exists() and not overwrite:
        try:
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            if cached.get("product_id") == product_id:
                if _sha256(destination) != cached.get("fits_sha256"):
                    raise TESSProductError("existing TPF failed its provenance checksum")
                return {**cached, "reused": True}
        except json.JSONDecodeError as exc:
            raise TESSProductError("existing TPF provenance is invalid JSON") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.zip")
    fits_stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.fits")
    sidecar_stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.json")
    try:
        archive_transfer = netclient.download(
            request.url(), archive_stage, provider="mast", max_bytes=max_bytes,
            progress=progress,
        )
        member_name = _extract_fits(archive_stage, fits_stage)
        fits_summary = _tpf_summary(fits_stage, expected_sector=request.sector)
        fits_sha = _sha256(fits_stage)
        provenance = {
            "schema_version": 1,
            "product_id": product_id,
            "provider": "MAST/TESScut",
            "product_kind": "target_pixel_file",
            "project_id": project_id,
            "request": request.to_dict(),
            "url": request.url(),
            "params": request.params(),
            "archive_member": member_name,
            "archive_bytes": archive_transfer.bytes_written,
            "archive_sha256": archive_transfer.sha256,
            "fits_bytes": fits_stage.stat().st_size,
            "fits_sha256": fits_sha,
            "downloaded_utc": _now(),
            "path": str(destination),
            "fits": fits_summary,
        }
        _write_json_atomic(sidecar_stage, provenance)
        store.publish_product_bundle(fits_stage, destination,
                                     sidecar_stage, sidecar, root)
        return {**provenance, "reused": False}
    finally:
        for temporary in (archive_stage, fits_stage, sidecar_stage):
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    # Do not obscure the archive/FITS validation failure with
                    # a transient Windows handle race. A later cache cleanup
                    # can remove an orphaned dot-prefixed staging file.
                    pass


def _time_to_bjd(time: np.ndarray, header) -> np.ndarray:
    values = np.asarray(time, dtype=np.float64).copy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    try:
        reference = float(header.get("BJDREFI", 0.0) or 0.0) + float(
            header.get("BJDREFF", 0.0) or 0.0
        )
    except (TypeError, ValueError, OverflowError):
        reference = 0.0
    if not math.isfinite(reference):
        reference = 0.0
    if np.nanmedian(np.abs(finite)) < 1_000_000:
        if reference:
            values += reference
        else:
            values += 2457000.0
    return values


def _read_tpf(path: Path) -> dict[str, Any]:
    from astropy.io import fits

    with fits.open(path, memmap=False, lazy_load_hdus=False) as hdul:
        _summary = _tpf_summary(path)
        _index, table_hdu = _find_table(hdul)
        data = table_hdu.data
        time_header = table_hdu.header.copy()
        # SPOC products commonly place BJDREFI/BJDREFF in the primary header,
        # while TESScut fixtures may put them on the table extension. Accept
        # both layouts without double-applying the reference.
        for key in ("BJDREFI", "BJDREFF", "TIMEUNIT", "TIMESYS"):
            if key not in time_header and key in hdul[0].header:
                time_header[key] = hdul[0].header[key]
        time = _time_to_bjd(
            np.array(_column(data, "TIME"), dtype=np.float64, copy=True),
            time_header,
        )
        flux = np.asarray(_column(data, "FLUX"), dtype=np.float64)
        if flux.ndim != 3 or len(time) != flux.shape[0]:
            raise TESSProductError("TPF TIME and FLUX cadence dimensions do not match")
        raw_errors = _optional_column(data, "FLUX_ERR")
        flux_err = None if raw_errors is None else np.asarray(raw_errors, dtype=np.float64)
        if flux_err is not None and flux_err.shape != flux.shape:
            raise TESSProductError("TPF FLUX_ERR shape does not match FLUX")
        raw_quality = _optional_column(data, "QUALITY")
        if raw_quality is None:
            quality = np.zeros(len(time), dtype=np.uint64)
        else:
            try:
                quality_values = np.asarray(raw_quality)
                if np.issubdtype(quality_values.dtype, np.floating):
                    if np.any(~np.isfinite(quality_values)) or np.any(
                        quality_values != np.floor(quality_values)
                    ):
                        raise ValueError("QUALITY values are not integral")
                if np.any(quality_values < 0):
                    raise ValueError("QUALITY values are negative")
                quality = np.asarray(quality_values, dtype=np.uint64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TESSProductError("TPF QUALITY column is not integral") from exc
            if quality.ndim != 1 or len(quality) != len(time):
                raise TESSProductError("TPF QUALITY length does not match TIME")
        wcs, _wcs_hdu = _wcs_from_hdul(hdul)
        return {
            "time": np.array(time, copy=True),
            "flux": np.array(flux, dtype=np.float64, copy=True),
            "flux_err": None if flux_err is None else np.array(flux_err, copy=True),
            "quality": np.array(quality, copy=True),
            "header": time_header,
            "wcs": wcs,
            "summary": _summary,
        }


def _neighbor_field(neighbor: object, *names: str) -> Any:
    if isinstance(neighbor, dict):
        for name in names:
            if name in neighbor:
                return neighbor[name]
        normalised = {
            re.sub(r"[^a-z0-9]", "", str(key).lower()): value
            for key, value in neighbor.items()
        }
        for name in names:
            value = normalised.get(re.sub(r"[^a-z0-9]", "", name.lower()))
            if value is not None:
                return value
        return None
    for name in names:
        if hasattr(neighbor, name):
            return getattr(neighbor, name)
    return None


def _separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    phi1, phi2 = np.radians(dec1), np.radians(dec2)
    dphi = phi2 - phi1
    dlambda = np.radians(ra2 - ra1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))) * 3600.0)


def _fallback_errors(flux: np.ndarray, aperture_values: np.ndarray,
                     background_pixels: np.ndarray, aperture_count: int,
                     supplied: np.ndarray | None) -> np.ndarray:
    """Build finite per-cadence errors even when a TPF omits FLUX_ERR.

    TESScut files normally carry ``FLUX_ERR`` but alternate/fixture products
    do not always do so.  Persisting NaN errors would make ``LightCurve`` drop
    every point; a zero error is also unsafe for downstream weighted fits. We
    combine the supplied uncertainty, background scatter, and a conservative
    Poisson-like scale, then apply a positive finite floor.
    """
    n_cadences = len(flux)
    if supplied is not None:
        try:
            error = np.sqrt(np.nansum(supplied ** 2, axis=1))
        except (TypeError, ValueError):
            error = np.full(n_cadences, np.nan, dtype=float)
    else:
        error = np.full(n_cadences, np.nan, dtype=float)

    if background_pixels.size:
        background_std = np.nanstd(background_pixels, axis=1)
        background_term = (background_std * aperture_count) / math.sqrt(
            max(int(background_pixels.shape[1]), 1)
        )
        error = np.sqrt(
            np.where(np.isfinite(error), error ** 2, 0.0)
            + np.where(np.isfinite(background_term), background_term ** 2, 0.0)
        )

    # A robust cadence-level scale from the aperture itself is a useful
    # fallback for a flat background and is still finite for zero/negative
    # difference fluxes.
    aperture_std = np.nanstd(aperture_values, axis=1) if aperture_values.size else np.full(n_cadences, np.nan)
    shot_scale = np.sqrt(np.maximum(np.abs(np.nansum(aperture_values, axis=1)), 1.0))
    candidate = np.maximum(
        np.where(np.isfinite(aperture_std), aperture_std, 0.0),
        shot_scale / math.sqrt(max(aperture_count, 1)),
    )
    error = np.where(np.isfinite(error) & (error > 0), error, candidate)
    finite_flux = np.abs(np.asarray(flux, dtype=float))
    floor = np.maximum(1e-12, np.sqrt(np.maximum(finite_flux, 1.0)) * 1e-12)
    error = np.where(np.isfinite(error) & (error > 0), error, floor)
    return np.asarray(error, dtype=np.float64)


def _sector_label(value: object) -> str:
    if value is None or value == "":
        return "unknown"
    try:
        return str(_strict_int(value, "sector", minimum=1, maximum=999))
    except TESSProductError:
        return "unknown"


def _blend_assessment(*, ra_deg: float, dec_deg: float, target_pixel: tuple[float, float],
                      shape: tuple[int, int], aperture: np.ndarray,
                      neighbors: Iterable[object], target_mag: float | None,
                      wcs) -> dict[str, Any]:
    ny, nx = shape
    aperture_radius = max(float(np.sqrt(np.sum(aperture) / np.pi)), 0.5)
    aperture_radius_arcsec = aperture_radius * TESS_PIXEL_SCALE_ARCSEC
    cutout_radius_arcsec = (float(np.hypot(nx, ny)) / 2.0) * TESS_PIXEL_SCALE_ARCSEC
    entries: list[dict[str, Any]] = []
    ratios: list[float] = []
    for neighbor in neighbors:
        nra = _neighbor_field(neighbor, "ra_deg", "ra", "RA")
        ndec = _neighbor_field(neighbor, "dec_deg", "dec", "DEC")
        if nra is None or ndec is None:
            continue
        try:
            nra, ndec = _finite_float(nra, "neighbor.ra_deg"), _finite_float(ndec, "neighbor.dec_deg")
            if not 0.0 <= nra < 360.0 or not -90.0 <= ndec <= 90.0:
                continue
            separation = _separation_arcsec(ra_deg, dec_deg, nra, ndec)
        except TESSProductError:
            continue
        pixel = None
        if wcs is not None:
            try:
                px, py = wcs.world_to_pixel_values(nra, ndec)
                pixel = (float(px), float(py))
            except Exception:  # noqa: BLE001
                pixel = None
        if pixel is not None and all(math.isfinite(value) for value in pixel):
            in_cutout = bool(-0.5 <= pixel[0] < nx - 0.5
                             and -0.5 <= pixel[1] < ny - 0.5)
            pixel_x = int(round(pixel[0]))
            pixel_y = int(round(pixel[1]))
            in_aperture = bool(
                0 <= pixel_y < ny and 0 <= pixel_x < nx
                and aperture[pixel_y, pixel_x]
            )
        else:
            # TESScut is centered on the requested coordinate. Without a
            # usable WCS, angular distance and the documented 21-arcsec pixel
            # scale provide a conservative, explicitly approximate assessment.
            in_aperture = separation <= aperture_radius_arcsec
            in_cutout = separation <= cutout_radius_arcsec
        magnitude = _neighbor_field(neighbor, "phot_g_mean_mag", "g_mag", "mag", "tess_mag")
        ratio = None
        try:
            if target_mag is not None and magnitude is not None:
                target_value = _finite_float(target_mag, "target_mag")
                neighbour_value = _finite_float(magnitude, "neighbor magnitude")
                ratio = float(10 ** (-0.4 * (neighbour_value - target_value)))
                if math.isfinite(ratio):
                    if in_aperture:
                        ratios.append(ratio)
        except (TESSProductError, OverflowError):
            ratio = None
        entries.append({
            "object_id": _neighbor_field(neighbor, "object_id", "id", "source_id"),
            "separation_arcsec": round(separation, 4),
            "pixel": pixel,
            "in_cutout": in_cutout,
            "in_aperture": in_aperture,
            "magnitude": magnitude,
            "relative_flux": ratio,
        })

    in_aperture = [entry for entry in entries if entry["in_aperture"]]
    in_cutout = [entry for entry in entries if entry["in_cutout"]]
    if in_aperture:
        risk = "high"
    elif in_cutout:
        risk = "moderate"
    elif entries:
        risk = "low"
    else:
        risk = "unknown"
    total_ratio = sum(ratios)
    return {
        "resolved": False,
        "risk": risk,
        "pixel_scale_arcsec": TESS_PIXEL_SCALE_ARCSEC,
        "target_pixel": [round(target_pixel[0], 3), round(target_pixel[1], 3)],
        "neighbors_considered": len(entries),
        "neighbors_in_cutout": len(in_cutout),
        "neighbors_in_aperture": len(in_aperture),
        "contamination_fraction": (total_ratio / (1.0 + total_ratio)
                                    if ratios else None),
        "neighbors": entries,
        "warning": "TESS aperture photometry is neighborhood flux; target-specific "
                   "confirmation requires pixel-level deblending/PSF modelling.",
    }


def extract_photometry(path: str | Path, *, ra_deg: float, dec_deg: float,
                       neighbors: Iterable[object] = (),
                       target_mag: float | None = None,
                       aperture_radius_pixels: float = 1.5,
                       quality_mask: int = DEFAULT_QUALITY_MASK) -> dict[str, Any]:
    """Extract background-subtracted aperture flux with an explicit blend report."""
    ra = _finite_float(ra_deg, "ra_deg")
    dec = _finite_float(dec_deg, "dec_deg")
    if not 0.0 <= ra < 360.0:
        raise TESSProductError("ra_deg must be finite and in [0, 360)")
    if not -90.0 <= dec <= 90.0:
        raise TESSProductError("dec_deg must be finite and in [-90, 90]")
    try:
        aperture_radius_pixels = float(aperture_radius_pixels)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TESSProductError("aperture_radius_pixels must be a number") from exc
    if not math.isfinite(aperture_radius_pixels) or not 0.5 <= aperture_radius_pixels <= 5.0:
        raise TESSProductError("aperture_radius_pixels must be between 0.5 and 5")
    quality_mask = _strict_int(quality_mask, "quality_mask", minimum=0,
                               maximum=(1 << 64) - 1)
    data = _read_tpf(Path(path))
    cube = data["flux"]
    n_cadences, ny, nx = cube.shape
    x_target, y_target = (float((nx - 1) / 2), float((ny - 1) / 2))
    position_source = "cutout_center"
    if data["wcs"] is not None:
        try:
            x_value, y_value = data["wcs"].world_to_pixel_values(ra, dec)
            if np.isfinite(x_value) and np.isfinite(y_value):
                x_target, y_target = float(x_value), float(y_value)
                position_source = "wcs"
        except Exception:  # noqa: BLE001 - center fallback is explicit evidence
            pass

    yy, xx = np.indices((ny, nx), dtype=float)
    aperture = (xx - x_target) ** 2 + (yy - y_target) ** 2 <= aperture_radius_pixels ** 2
    if not np.any(aperture):
        aperture[int(np.clip(round(y_target), 0, ny - 1)),
                 int(np.clip(round(x_target), 0, nx - 1))] = True
    background_mask = ~aperture
    if not np.any(background_mask):
        background_mask = np.ones((ny, nx), dtype=bool)
    background_pixels = cube[:, background_mask]
    background = np.nanmedian(background_pixels, axis=1)
    aperture_count = int(np.sum(aperture))
    aperture_values = cube[:, aperture]
    flux = np.nansum(aperture_values, axis=1) - background * aperture_count
    error = _fallback_errors(
        flux, aperture_values, background_pixels, aperture_count,
        None if data["flux_err"] is None else data["flux_err"][:, aperture],
    )

    quality = data["quality"]
    good = (np.asarray(quality, dtype=np.uint64) & np.uint64(quality_mask)) == 0
    good &= np.isfinite(data["time"]) & np.isfinite(flux)
    good &= np.isfinite(error) & (error > 0)
    blend = _blend_assessment(
        ra_deg=ra, dec_deg=dec, target_pixel=(x_target, y_target),
        shape=(ny, nx), aperture=aperture, neighbors=neighbors,
        target_mag=target_mag, wcs=data["wcs"],
    )
    return {
        "path": str(path),
        "time": data["time"][good],
        "flux": flux[good],
        "flux_err": error[good],
        "background": background[good],
        "quality": quality[good],
        "points": int(np.sum(good)),
        "total_cadences": int(n_cadences),
        "aperture_mask": aperture,
        "position_source": position_source,
        "pixel_shape": [int(ny), int(nx)],
        "sector": data["summary"].get("sector"),
        "blend": blend,
    }


def persist_photometry(path: str | Path, *, ra_deg: float, dec_deg: float,
                       target_id: str | None = None,
                       neighbors: Iterable[object] = (),
                       target_mag: float | None = None,
                       aperture_radius_pixels: float = 1.5,
                       quality_mask: int = DEFAULT_QUALITY_MASK,
                       root: Path | None = None) -> dict[str, Any]:
    """Extract and persist the TPF aperture curve in the canonical store."""
    payload = extract_photometry(
        path, ra_deg=ra_deg, dec_deg=dec_deg, neighbors=neighbors,
        target_mag=target_mag, aperture_radius_pixels=aperture_radius_pixels,
        quality_mask=quality_mask,
    )
    if payload["points"] < 1:
        raise TESSProductError("TPF contains no finite cadences after quality filtering")
    source_id = target_id or f"TPF-{Path(path).stem[:16]}"
    if len(str(source_id)) > 128 or "\x00" in str(source_id):
        raise TESSProductError("target_id is invalid")
    sector = _sector_label(payload["sector"])
    source = SourceRef(
        survey="TESS", object_id=source_id, ra_deg=float(ra_deg),
        dec_deg=float(dec_deg), extra={"sector": payload["sector"],
                                       "blend": payload["blend"],
                                       "tpf_path": str(path)},
    )
    curve = LightCurve(
        source=source,
        release=f"tpf-s{sector}",
        band="TESS",
        value_kind="flux",
        time=payload["time"], value=payload["flux"],
        value_err=payload["flux_err"], time_system="BJD_TDB",
    )
    stored = store.write_curve(curve, root=root or config.PATHS.datasets)
    payload["curve_path"] = str(stored.path)
    payload["curve_bytes"] = stored.bytes_on_disk
    payload["source"] = source.object_id
    # Numpy arrays/masks remain useful to Python callers, while the RPC layer
    # turns them into bounded JSON below.
    return payload


def json_payload(payload: dict[str, Any], max_points: int = 5000) -> dict[str, Any]:
    """Convert a photometry result to bounded, JSON-safe transport data."""
    max_points = _strict_int(max_points, "max_points", minimum=1, maximum=50_000)
    if max_points < 1 or max_points > 50_000:
        raise TESSProductError("max_points must be between 1 and 50000")
    total = len(payload["time"])
    indices = np.linspace(0, total - 1, min(total, max_points), dtype=int) if total else np.empty(0, dtype=int)
    result = {key: value for key, value in payload.items()
              if key not in {"time", "flux", "flux_err", "background", "quality", "aperture_mask"}}

    def safe_array(value: Any) -> list[Any]:
        array = np.asarray(value)
        output: list[Any] = []
        for item in array[indices].tolist():
            if isinstance(item, (float, np.floating)):
                output.append(float(item) if math.isfinite(float(item)) else None)
            elif isinstance(item, (int, np.integer)):
                output.append(int(item))
            else:
                output.append(item)
        return output

    def safe_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return safe_value(value.item())
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(key): safe_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    result = safe_value(result)
    result.update({
        "time": safe_array(payload["time"]),
        "flux": safe_array(payload["flux"]),
        "flux_err": safe_array(payload["flux_err"]),
        "background": safe_array(payload["background"]),
        "quality": safe_array(payload["quality"]),
        "shown_points": int(len(indices)),
        "downsampled": bool(len(indices) < total),
        "aperture_mask": np.asarray(payload["aperture_mask"], dtype=bool).tolist(),
    })
    return result


__all__ = [
    "TPFRequest", "TESSProductError", "download_tpf", "extract_photometry",
    "persist_photometry", "json_payload", "TESScUT_URL",
    "TESS_PIXEL_SCALE_ARCSEC", "DEFAULT_SIZE_PIXELS", "DEFAULT_MAX_BYTES",
]
