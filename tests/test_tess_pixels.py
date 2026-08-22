"""Bounded TESScut acquisition and blend-aware TPF photometry."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest

from astra import netclient, rpc, store, tess_pixels


def _tpf_bytes(*, sector: int | None = 7, with_errors: bool = True,
               with_wcs: bool = False, lower_columns: bool = False) -> bytes:
    from astropy.io import fits

    n, ny, nx = 12, 5, 5
    time = 1000.0 + np.arange(n, dtype=np.float64)
    flux = np.full((n, ny, nx), 100.0, dtype=np.float32)
    flux[:, 2, 2] += np.arange(n, dtype=np.float32) * 2.0
    quality = np.zeros(n, dtype=np.uint32)
    quality[3] = 1
    names = {"time": "time" if lower_columns else "TIME",
             "flux": "flux" if lower_columns else "FLUX",
             "quality": "quality" if lower_columns else "QUALITY",
             "error": "flux_err" if lower_columns else "FLUX_ERR"}
    columns = [
        fits.Column(name=names["time"], format="D", array=time),
        fits.Column(name=names["flux"], format=f"{ny * nx}E",
                    dim=f"({nx},{ny})", array=flux),
        fits.Column(name=names["quality"], format="J", array=quality),
    ]
    if with_errors:
        columns.append(fits.Column(name=names["error"], format=f"{ny * nx}E",
                                   dim=f"({nx},{ny})",
                                   array=np.full_like(flux, 1.0)))
    table = fits.BinTableHDU.from_columns(columns)
    table.header["BJDREFI"] = 2457000
    if sector is not None:
        table.header["SECTOR"] = sector
    primary = fits.PrimaryHDU()
    if sector is not None:
        primary.header["SECTOR"] = sector
    if with_wcs:
        header = primary.header
        header["CRPIX1"] = 3.0
        header["CRPIX2"] = 3.0
        header["CRVAL1"] = 100.0
        header["CRVAL2"] = 20.0
        header["CDELT1"] = -21.0 / 3600.0
        header["CDELT2"] = 21.0 / 3600.0
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
    output = io.BytesIO()
    fits.HDUList([primary, table]).writeto(output)
    return output.getvalue()


def _zip_bytes(payload: bytes, name: str = "cutout.fits") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
    return output.getvalue()


def test_tpf_request_rejects_truncating_or_non_numeric_values():
    for sector in (1.5, "1.5", True, None):
        with pytest.raises(tess_pixels.TESSProductError):
            tess_pixels.TPFRequest(10, 20, sector)
    with pytest.raises(tess_pixels.TESSProductError):
        tess_pixels.TPFRequest(10, 20, 1, size_pixels=2.2)
    request = tess_pixels.TPFRequest(10, 20, "7", size_pixels="4")
    assert request.to_dict()["sector"] == 7
    assert "sector=7" in request.url()


def test_zip_extraction_rejects_traversal_and_symlink(tmp_path):
    payload = _tpf_bytes()
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escape.fits", payload)
    with pytest.raises(tess_pixels.TESSProductError):
        tess_pixels._extract_fits(traversal, tmp_path / "out.fits")

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link.fits")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(info, b"target.fits")
    with pytest.raises(tess_pixels.TESSProductError):
        tess_pixels._extract_fits(symlink, tmp_path / "out2.fits")


def test_download_publishes_verified_tpf_and_reuses_it(monkeypatch, tmp_path):
    payload = _tpf_bytes(sector=7)
    archive_payload = _zip_bytes(payload, "nested/sector-7.fits")
    calls: list[str] = []

    def fake_download(url, destination, **kwargs):
        calls.append(url)
        Path(destination).write_bytes(archive_payload)
        return netclient.DownloadResult(
            path=Path(destination), bytes_written=len(archive_payload),
            sha256=hashlib.sha256(archive_payload).hexdigest(),
            content_length=len(archive_payload),
        )

    monkeypatch.setattr(tess_pixels.netclient, "download", fake_download)
    store.invalidate_usage_cache()
    request = tess_pixels.TPFRequest(100, 20, 7, size_pixels=4, target_id="TIC 1")
    first = tess_pixels.download_tpf(request, root=tmp_path)
    assert first["reused"] is False
    path = Path(first["path"])
    sidecar = path.with_suffix(path.suffix + ".json")
    assert path.is_file() and sidecar.is_file()
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    assert saved["fits"]["sector"] == 7
    assert saved["fits_sha256"] == hashlib.sha256(payload).hexdigest()
    second = tess_pixels.download_tpf(request, root=tmp_path)
    assert second["reused"] is True
    assert len(calls) == 1


def test_download_rejects_sector_mismatch_and_cleans_staging(monkeypatch, tmp_path):
    payload = _zip_bytes(_tpf_bytes(sector=8))

    def fake_download(url, destination, **kwargs):
        Path(destination).write_bytes(payload)
        return netclient.DownloadResult(
            path=Path(destination), bytes_written=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(), content_length=len(payload),
        )

    monkeypatch.setattr(tess_pixels.netclient, "download", fake_download)
    with pytest.raises(tess_pixels.TESSProductError, match="does not match"):
        tess_pixels.download_tpf(tess_pixels.TPFRequest(1, 2, 7), root=tmp_path)
    assert not list(tmp_path.rglob("*.fits"))
    assert not list(tmp_path.rglob("*.json"))


def test_photometry_filters_quality_reports_blend_and_uses_finite_fallback(tmp_path):
    path = tmp_path / "no-errors.fits"
    path.write_bytes(_tpf_bytes(sector=None, with_errors=False, lower_columns=True))
    result = tess_pixels.extract_photometry(
        path, ra_deg=100, dec_deg=20, quality_mask=1,
        neighbors=[{"id": "near", "ra": 100, "dec": 20, "mag": 12}],
        target_mag=14,
    )
    assert result["points"] == 11
    assert np.all(np.isfinite(result["flux_err"]))
    assert result["blend"]["risk"] == "high"
    assert result["blend"]["source_attribution"]
    assert result["blend"]["attribution_method"] == "catalog_relative_flux_prior"
    assert result["blend"]["attribution_diagnostics"]["quality"] == "informative"
    assert result["blend"]["attribution_diagnostics"]["target_fraction_sensitivity"] >= 0
    assert result["sector"] is None
    payload = tess_pixels.json_payload(result, max_points=3)
    json.dumps(payload, allow_nan=False)
    assert payload["shown_points"] == 3


def test_persist_photometry_uses_explicit_unknown_sector_label(isolated_root, tmp_path):
    path = tmp_path / "unknown.fits"
    path.write_bytes(_tpf_bytes(sector=None, with_errors=False))
    result = tess_pixels.persist_photometry(
        path, ra_deg=100, dec_deg=20, target_id="candidate-1",
        root=isolated_root.datasets,
    )
    assert "tpf-sunknown" in result["curve_path"]
    assert Path(result["curve_path"]).is_file()


def test_rpc_registers_tpf_handlers(monkeypatch):
    seen = {}

    def fake_download(request, **kwargs):
        seen["request"] = request
        return {"product_id": request.product_id(), "reused": False}

    monkeypatch.setattr(rpc.tess_pixels, "download_tpf", fake_download)
    response = rpc.dispatch({
        "id": 1, "method": "tess.tpf.download",
        "params": {"ra_deg": 10, "dec_deg": 20, "sector": 3, "size_pixels": 6},
    })
    assert response["ok"] is True
    assert seen["request"].sector == 3
    assert "tess.tpf.photometry" in rpc.HANDLERS


def test_rpc_photometry_persists_an_authorized_tpf(isolated_root):
    path = isolated_root.datasets / "TESS" / "input.fits"
    path.parent.mkdir(parents=True)
    path.write_bytes(_tpf_bytes(sector=7, with_errors=False))

    response = rpc.dispatch({
        "id": 2,
        "method": "tess.tpf.photometry",
        "params": {
            "path": str(path), "ra_deg": 100, "dec_deg": 20,
            "target_id": "candidate-2", "max_points": 4,
        },
    })

    assert response["ok"] is True
    result = response["result"]
    assert result["shown_points"] == 4
    assert Path(result["curve_path"]).is_file()
