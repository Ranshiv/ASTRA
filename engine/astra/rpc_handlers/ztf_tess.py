"""ZTF cutout search/download and TESS target-pixel-file download/photometry.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

from typing import Any

from .common import Handler

from .. import config, products, security, tess_pixels

def _cutout_request(params: dict[str, Any]) -> products.CutoutRequest:
    return products.CutoutRequest(
        ra_deg=float(params["ra_deg"]),
        dec_deg=float(params["dec_deg"]),
        size_arcsec=float(params.get("size_arcsec", 50.0)),
        product_kind=str(params.get("product_kind", "science")),
        release=str(params.get("release", products.ZTF_RELEASE)),
    )


def _handle_ztf_images_search(params: dict[str, Any]) -> list[dict[str, Any]]:
    request = _cutout_request(params)
    rows = products.search(request, limit=int(params.get("limit", 25)))
    return [{**row, "product_url": products.product_url(row),
             "cutout_url": products.cutout_url(row, request)} for row in rows]


def _handle_ztf_images_download(params: dict[str, Any], progress=None) -> dict[str, Any]:
    request = _cutout_request(params)
    row = params.get("metadata") or params.get("row")
    if not isinstance(row, dict):
        raise ValueError("metadata must be an object returned by ztf.images.search")
    return products.download_cutout(
        request, row,
        project_id=params.get("project_id"),
        max_bytes=int(params.get("max_bytes", products.DEFAULT_MAX_BYTES)),
        overwrite=bool(params.get("overwrite", False)),
        progress=(lambda received, total: progress.update(
            phase="download", message="Downloading ZTF FITS cutout",
            bytes_downloaded=received, bytes_total=total,
        ) if progress is not None else None),
    )


def _tpf_request(params: dict[str, Any]) -> tess_pixels.TPFRequest:
    """Build and validate a candidate-scale TESS TPF request."""
    if params.get("sector") is None:
        raise tess_pixels.TESSProductError("sector is required for a TESS TPF request")
    size_pixels = params.get("size_pixels")
    if size_pixels is None:
        size_pixels = tess_pixels.DEFAULT_SIZE_PIXELS
    return tess_pixels.TPFRequest(
        ra_deg=params.get("ra_deg"),
        dec_deg=params.get("dec_deg"),
        sector=params.get("sector"),
        size_pixels=size_pixels,
        target_id=params.get("target_id"),
        product=params.get("product", "SPOC"),
    )


def _optional_bool(params: dict[str, Any], name: str, default: bool) -> bool:
    value = params.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _handle_tess_tpf_download(params: dict[str, Any], progress=None) -> dict[str, Any]:
    request = _tpf_request(params)

    def report(received: int, total: int | None) -> None:
        if progress is None:
            return
        progress.update(
            phase="download",
            message="Downloading TESS target-pixel file",
            bytes_downloaded=received,
            bytes_total=total,
        )

    max_bytes = params.get("max_bytes")
    if max_bytes is None:
        max_bytes = tess_pixels.DEFAULT_MAX_BYTES
    return tess_pixels.download_tpf(
        request,
        project_id=params.get("project_id"),
        max_bytes=max_bytes,
        overwrite=_optional_bool(params, "overwrite", False),
        progress=report,
    )


def _handle_tess_tpf_photometry(params: dict[str, Any]) -> dict[str, Any]:
    raw_path = params.get("path") or params.get("tpf_path")
    if not raw_path:
        raise ValueError("path is required for TESS TPF photometry")
    path = security.authorized_path(str(raw_path))
    neighbours = params.get("neighbors") or []
    if not isinstance(neighbours, (list, tuple)):
        raise ValueError("neighbors must be an array")
    aperture = params.get("aperture_radius_pixels")
    if aperture is None:
        aperture = 1.5
    quality = params.get("quality_mask")
    if quality is None:
        quality = tess_pixels.DEFAULT_QUALITY_MASK
    common = {
        "ra_deg": params.get("ra_deg"),
        "dec_deg": params.get("dec_deg"),
        "neighbors": neighbours,
        "target_mag": params.get("target_mag"),
        "aperture_radius_pixels": aperture,
        "quality_mask": quality,
    }
    if _optional_bool(params, "persist", True):
        payload = tess_pixels.persist_photometry(
            path,
            target_id=params.get("target_id"),
            root=config.PATHS.datasets,
            **common,
        )
    else:
        payload = tess_pixels.extract_photometry(path, **common)
    max_points = params.get("max_points")
    if max_points is None:
        max_points = 5000
    return tess_pixels.json_payload(payload, max_points=max_points)


HANDLERS: dict[str, Handler] = {
    "ztf.images.search": _handle_ztf_images_search,
    "ztf.images.download": _handle_ztf_images_download,
    "tess.tpf.download": _handle_tess_tpf_download,
    "tess.tpf.photometry": _handle_tess_tpf_photometry,
}
