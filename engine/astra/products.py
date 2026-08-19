"""Immutable image products and ZTF cutouts.

The ZTF IRSA image service exposes a metadata search endpoint and a separate
IBE data endpoint.  Keeping the two steps explicit is important for a
reproducible research application: a cutout is identified by the archive
product *and* the requested sky window, while its sidecar records the exact
query, URLs, archive row, checksum and FITS summary.

This module deliberately does not require a live IRSA service at import time.
Search and download functions accept the shared ``netclient`` and are easy to
exercise with fixtures in tests or an offline installation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from . import config, fitsio, netclient, store

ZTF_METADATA_URL = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
ZTF_DATA_URL = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"
ZTF_RELEASE = "dr"
SEARCH_TIMEOUT_S = 90.0
DOWNLOAD_TIMEOUT_S = 300.0
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
MAX_CUTOUT_ARCSEC = 600.0
MIN_CUTOUT_ARCSEC = 1.0


class ProductError(ValueError):
    """A request or archive metadata row cannot produce a safe product."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ProductError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class CutoutRequest:
    ra_deg: float
    dec_deg: float
    size_arcsec: float = 50.0
    product_kind: str = "science"
    release: str = ZTF_RELEASE

    def __post_init__(self) -> None:
        ra = _finite(float(self.ra_deg), "ra_deg")
        dec = _finite(float(self.dec_deg), "dec_deg")
        size = _finite(float(self.size_arcsec), "size_arcsec")
        if not 0.0 <= ra < 360.0:
            raise ProductError("ra_deg must be in [0, 360)")
        if not -90.0 <= dec <= 90.0:
            raise ProductError("dec_deg must be in [-90, 90]")
        if not MIN_CUTOUT_ARCSEC <= size <= MAX_CUTOUT_ARCSEC:
            raise ProductError(
                f"size_arcsec must be between {MIN_CUTOUT_ARCSEC:g} and "
                f"{MAX_CUTOUT_ARCSEC:g} arcsec"
            )
        # This first product connector deliberately targets the documented
        # ``/ztf/products/sci`` IBE collection.  Reference and difference
        # images use different collection/path contracts and must not be
        # guessed from a science row.
        if self.product_kind != "science":
            raise ProductError("only the ZTF science image product_kind is supported")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", str(self.release)):
            raise ProductError("release contains unsafe characters")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ra_deg": round(float(self.ra_deg), 8),
            "dec_deg": round(float(self.dec_deg), 8),
            "size_arcsec": round(float(self.size_arcsec), 4),
            "product_kind": self.product_kind,
            "release": self.release,
        }


def _normalise_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """Make metadata keys case/underscore insensitive while retaining values."""
    result: dict[str, Any] = {}
    for key, value in row.items():
        result[_normalise_key(key)] = value.strip() if isinstance(value, str) else value
    return result


def parse_metadata_csv(payload: str | bytes) -> list[dict[str, Any]]:
    """Parse IRSA ``CT=csv`` output, including comment/preamble lines."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    lines = [line for line in payload.splitlines()
             if line.strip() and not line.lstrip().startswith(("#", "!"))]
    if not lines:
        return []
    # IPAC table output uses a pipe-delimited header and may include type/null
    # rows.  CT=csv is the normal path, but accepting this format makes saved
    # archive fixtures and older IRSA responses reproducible too.
    pipe_header = next((i for i, line in enumerate(lines)
                        if "|" in line and "filefracday" in line.lower()), None)
    if pipe_header is not None:
        return _parse_ipac_rows(lines, pipe_header)
    # IPAC occasionally emits a status line before the CSV header.  Start at
    # the first line containing the required filefracday column.
    start = next((i for i, line in enumerate(lines)
                  if "filefracday" in line.lower()), 0)
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return [_normalise_row(dict(row)) for row in reader if any(row.values())]


def _parse_ipac_rows(lines: list[str], header_index: int) -> list[dict[str, Any]]:
    header_line = lines[header_index]
    separators = [index for index, char in enumerate(header_line) if char == "|"]
    columns = [part.strip() for part in header_line.strip().strip("|").split("|")]
    if not columns or not separators:
        return []
    result: list[dict[str, Any]] = []
    type_words = {"int", "long", "float", "double", "char", "string",
                  "boolean", "bool", "date", "null"}
    for line in lines[header_index + 1:]:
        if line.lstrip().startswith("\\") or not line.strip():
            continue
        if "|" in line:
            values = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(values) != len(columns):
                continue
            if all(value.lower() in type_words for value in values):
                continue
        else:
            # Fixed-width IPAC rows inherit the header's pipe boundaries.
            if len(separators) < len(columns) + 1:
                continue
            values = [line[separators[index] + 1:separators[index + 1]].strip()
                      for index in range(len(columns))]
        if not any(values):
            continue
        result.append(_normalise_row(dict(zip(columns, values))))
    return result


def _value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        key = _normalise_key(name)
        if key in row and row[key] not in (None, ""):
            return row[key]
    raise ProductError(f"metadata row is missing {names[0]}")


def _integer(row: dict[str, Any], *names: str) -> int:
    raw = _value(row, *names)
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError) as exc:
        raise ProductError(f"metadata field {names[0]} is not an integer") from exc
    if value < 0:
        raise ProductError(f"metadata field {names[0]} must be non-negative")
    return value


def _filefracday(row: dict[str, Any]) -> str:
    raw = str(_value(row, "filefracday")).strip()
    match = re.fullmatch(r"(\d{14})(?:\.0+)?", raw)
    if not match:
        raise ProductError("filefracday must contain 14 digits (YYYYMMDDffffff)")
    return match.group(1)


def product_relative_path(row: dict[str, Any]) -> str:
    """Construct the documented IRSA ZTF science-image path from metadata."""
    stamp = _filefracday(row)
    field = _integer(row, "field")
    ccdid = _integer(row, "ccdid", "ccd_id")
    qid = _integer(row, "qid", "quadrant")
    if field > 999999:
        raise ProductError("field is outside the ZTF field range")
    if not 1 <= ccdid <= 99:
        raise ProductError("ccdid is outside the supported range")
    if not 1 <= qid <= 99:
        raise ProductError("qid is outside the supported range")
    filtercode = str(_value(row, "filtercode", "filter")).strip().lower()
    imgtype = str(_value(row, "imgtypecode", "imgtype", "image_type")).strip().lower()
    if not re.fullmatch(r"[a-z0-9]+", filtercode):
        raise ProductError("filtercode contains unsafe characters")
    if not re.fullmatch(r"[a-z0-9]+", imgtype):
        raise ProductError("imgtypecode contains unsafe characters")
    return (
        f"{stamp[:4]}/{stamp[4:8]}/{stamp[8:]}"
        f"/ztf_{stamp}_{field:06d}_{filtercode}_c{ccdid:02d}_{imgtype}_q{qid}_"
        "sciimg.fits"
    )


def product_url(row: dict[str, Any], base_url: str = ZTF_DATA_URL) -> str:
    relative = product_relative_path(row)
    return base_url.rstrip("/") + "/" + "/".join(quote(part, safe="._-")
                                                   for part in relative.split("/"))


def cutout_url(row: dict[str, Any], request: CutoutRequest,
               base_url: str = ZTF_DATA_URL) -> str:
    """Build an IRSA IBE FITS cutout URL (uncompressed FITS output)."""
    center = f"{request.ra_deg:.8f},{request.dec_deg:.8f}"
    size = f"{request.size_arcsec:g}arcsec"
    return product_url(row, base_url) + f"?center={center}&size={size}&gzip=false"


def _search_params(request: CutoutRequest, limit: int) -> dict[str, str]:
    if limit < 1 or limit > 500:
        raise ProductError("limit must be between 1 and 500")
    return {
        "POS": f"{request.ra_deg:.8f},{request.dec_deg:.8f}",
        "SIZE": f"{request.size_arcsec / 3600.0:.8f}",
        "INTERSECT": "OVERLAPS",
        "COLUMNS": "filefracday,field,filtercode,ccdid,imgtypecode,qid",
        "CT": "csv",
        "ROWS": str(limit),
    }


def search(request: CutoutRequest, *, limit: int = 25,
           metadata_url: str = ZTF_METADATA_URL) -> list[dict[str, Any]]:
    """Search ZTF image metadata near a coordinate."""
    response = netclient.get(metadata_url, _search_params(request, limit),
                             timeout=SEARCH_TIMEOUT_S, provider="irsa")
    text = getattr(response, "text", None)
    if text is None:
        content = getattr(response, "content", b"")
        text = content.decode("utf-8", errors="replace")
    rows = parse_metadata_csv(text)
    valid: list[dict[str, Any]] = []
    for row in rows:
        # Validate before exposing a row to the UI; malformed archive rows are
        # reported as absent rather than becoming unsafe paths.
        product_relative_path(row)
        valid.append(row)
    return valid[:limit]


def _request_key(request: CutoutRequest, row: dict[str, Any]) -> str:
    canonical = json.dumps({"request": request.to_dict(),
                            "product": product_relative_path(row)},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _product_root(request: CutoutRequest, root: Path | None = None) -> Path:
    return (root or config.PATHS.datasets) / "ZTF" / "images" / request.release


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _read_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(_sidecar_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fits_summary(path: Path) -> dict[str, Any]:
    """Validate FITS structure and retain bounded WCS/header evidence."""
    from astropy.io import fits

    wcs_state: dict[str, Any] = {"has_celestial": False}
    with fits.open(path, memmap=True, lazy_load_hdus=False) as hdul:
        # A 200 OK HTML page or a truncated archive response must never be
        # published as scientific data merely because Astropy can infer a
        # partial header from it.
        hdul.verify("exception")
        image_header = next((hdu.header for hdu in hdul
                             if getattr(hdu, "shape", None)
                             and len(hdu.shape) == 2), None)
        if image_header is not None:
            try:
                from astropy.wcs import WCS

                wcs_state["has_celestial"] = bool(WCS(image_header).has_celestial)
            except Exception:  # noqa: BLE001 - absent WCS is evidence, not a failure
                wcs_state["has_celestial"] = False
    description = fitsio.describe(path)
    header = fitsio.read_header(path)
    image_hdus = [hdu for hdu in description["hdus"] if hdu["is_image"]]
    if not image_hdus:
        raise ProductError("downloaded product contains no 2-D image HDU")
    summary = header.get("summary", {})
    wcs = {key: summary[key] for key in
           ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2",
             "CD2_1", "CD2_2", "CTYPE1", "CTYPE2") if key in summary}
    wcs.update(wcs_state)
    return {"description": description, "header_summary": summary, "wcs": wcs}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def download_cutout(request: CutoutRequest, row: dict[str, Any], *,
                    project_id: str | None = None,
                    root: Path | None = None,
                    max_bytes: int = DEFAULT_MAX_BYTES,
                    overwrite: bool = False,
                    data_url: str = ZTF_DATA_URL,
                    progress=None) -> dict[str, Any]:
    """Download, validate, quota-check and publish one immutable cutout."""
    if max_bytes <= 0 or max_bytes > DEFAULT_MAX_BYTES:
        raise ProductError(f"max_bytes must be between 1 and {DEFAULT_MAX_BYTES}")
    if project_id:
        from . import project

        project.require_active(project_id)
    relative = product_relative_path(row)
    key = _request_key(request, row)
    destination = _product_root(request, root) / key[:2] / f"{key}.fits"
    sidecar = _sidecar_path(destination)
    if destination.exists() and sidecar.exists() and not overwrite:
        cached = _read_sidecar(destination)
        if cached and cached.get("product_id") == key and cached.get("sha256"):
            actual = _file_sha256(destination)
            if actual != cached["sha256"]:
                raise ProductError("existing FITS product failed its provenance checksum")
            return {**cached, "reused": True}

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.download")
    try:
        transfer = netclient.download(
            cutout_url(row, request, data_url), staging,
            timeout=DOWNLOAD_TIMEOUT_S, provider="irsa", max_bytes=max_bytes,
            progress=progress,
        )
        actual_size = staging.stat().st_size
        actual_sha256 = _file_sha256(staging)
        if transfer.bytes_written != actual_size or transfer.sha256 != actual_sha256:
            raise ProductError("download integrity result does not match staged FITS bytes")
        summary = _fits_summary(staging)
        provenance = {
            "schema_version": 1,
            "product_id": key,
            "provider": "ZTF/IRSA",
            "product_kind": request.product_kind,
            "project_id": project_id,
            "request": request.to_dict(),
            "metadata": row,
            "relative_product_path": relative,
            "source_url": product_url(row, data_url),
            "cutout_url": cutout_url(row, request, data_url),
            "path": str(destination),
            "bytes": transfer.bytes_written,
            "sha256": transfer.sha256,
            "content_length": transfer.content_length,
            "downloaded_utc": _now(),
            "fits": summary,
        }
        # Build the sidecar in the same staging directory so the FITS and its
        # provenance are charged together and become visible together.
        sidecar_staging = sidecar.with_name(
            f".{sidecar.name}.{uuid.uuid4().hex}.download"
        )
        _write_json_atomic(sidecar_staging, provenance)
        size = store.publish_product_bundle(
            staging, destination, sidecar_staging, sidecar,
            root or config.PATHS.datasets,
        )
        provenance["bytes"] = size
        return {**provenance, "reused": False}
    except Exception:
        if staging.exists():
            staging.unlink()
        if "sidecar_staging" in locals() and sidecar_staging.exists():
            sidecar_staging.unlink()
        raise


def list_products(*, root: Path | None = None, limit: int = 500,
                  project_id: str | None = None,
                  survey: str | None = None) -> list[dict[str, Any]]:
    """List provenance sidecars, newest first, without reading FITS pixels."""
    if limit < 1 or limit > 5000:
        raise ProductError("limit must be between 1 and 5000")
    base = (root or config.PATHS.datasets)
    if survey:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", survey):
            raise ProductError("survey contains unsafe characters")
        base = base / survey.upper()
    result: list[dict[str, Any]] = []
    if not base.exists():
        return result
    sidecars = []
    for path in base.rglob("*.fits.json"):
        try:
            sidecars.append((path.stat().st_mtime, path))
        except OSError:
            continue
    for _mtime, sidecar in sorted(sidecars, reverse=True):
        payload = _read_sidecar(sidecar.with_suffix(""))
        if not payload or (project_id is not None and payload.get("project_id") != project_id):
            continue
        result.append(payload)
        if len(result) >= limit:
            break
    return result


def get_product(product_id: str, *, root: Path | None = None) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", product_id):
        raise ProductError("invalid product_id")
    base = (root or config.PATHS.datasets)
    matches = list(base.rglob(f"{product_id}.fits.json")) if base.exists() else []
    if not matches:
        raise FileNotFoundError(product_id)
    payload = _read_sidecar(matches[0].with_suffix(""))
    if payload is None:
        raise ValueError("product provenance is invalid JSON")
    return payload


def product_path(product_id: str, *, root: Path | None = None) -> Path:
    payload = get_product(product_id, root=root)
    path = Path(str(payload["path"])).resolve()
    managed = (root or config.PATHS.datasets).resolve()
    try:
        path.relative_to(managed)
    except ValueError as exc:
        raise PermissionError("product path escapes managed datasets") from exc
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


# Descriptive aliases keep the public connector discoverable without making
# callers depend on the internal verb chosen for the RPC handler.
ZTFImageRequest = CutoutRequest
parse_metadata = parse_metadata_csv
search_images = search


__all__ = [
    "CutoutRequest", "ZTFImageRequest", "ProductError", "parse_metadata_csv",
    "parse_metadata",
    "product_relative_path", "product_url", "cutout_url", "search",
    "search_images",
    "download_cutout", "list_products", "get_product", "product_path",
    "ZTF_METADATA_URL", "ZTF_DATA_URL", "DEFAULT_MAX_BYTES",
]
