"""Fixture-backed ZTF image metadata, cutouts, and provenance."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from astra import netclient, products, store


ROW = {
    "filefracday": "20180411467847",
    "field": "535",
    "filtercode": "zr",
    "ccdid": "11",
    "imgtypecode": "o",
    "qid": "3",
}


def _fits_bytes() -> bytes:
    from astropy.io import fits

    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    hdu = fits.PrimaryHDU(data)
    hdu.header["CRVAL1"] = 255.57691
    hdu.header["CRVAL2"] = 12.28378
    hdu.header["CRPIX1"] = 2.0
    hdu.header["CRPIX2"] = 2.0
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    output = io.BytesIO()
    hdu.writeto(output)
    return output.getvalue()


def test_ztf_path_and_cutout_url_match_irsa_contract():
    request = products.CutoutRequest(255.57691, 12.28378, 50)

    relative = products.product_relative_path(ROW)
    assert relative == (
        "2018/0411/467847/"
        "ztf_20180411467847_000535_zr_c11_o_q3_sciimg.fits"
    )
    url = products.cutout_url(ROW, request)
    assert "center=255.57691000,12.28378000" in url
    assert "size=50arcsec" in url
    assert url.endswith("gzip=false")


def test_csv_and_ipac_metadata_are_normalised():
    csv_text = "# status: OK\nfilefracday,field,filtercode,ccdid,imgtypecode,qid\n" \
               "20180411467847,535,zr,11,o,3\n"
    assert products.parse_metadata_csv(csv_text)[0]["field"] == "535"

    ipac_text = """| filefracday | field | filtercode | ccdid | imgtypecode | qid |
| long | int | char | int | char | int |
| null | null | null | null | null | null |
| 20180411467847 | 535 | zr | 11 | o | 3 |
"""
    assert products.parse_metadata_csv(ipac_text)[0]["qid"] == "3"


def test_search_forwards_bounded_query(monkeypatch):
    seen = {}

    class Response:
        text = "filefracday,field,filtercode,ccdid,imgtypecode,qid\n" \
               "20180411467847,535,zr,11,o,3\n"

    def fake_get(url, params, timeout, provider):
        seen.update(url=url, params=params, timeout=timeout, provider=provider)
        return Response()

    monkeypatch.setattr(products.netclient, "get", fake_get)
    request = products.CutoutRequest(255.5, 12.2, 40)
    rows = products.search(request, limit=4)

    assert len(rows) == 1
    assert seen["provider"] == "irsa"
    assert seen["params"]["POS"] == "255.50000000,12.20000000"
    assert seen["params"]["CT"] == "csv"
    assert seen["params"]["ROWS"] == "4"


def test_invalid_metadata_cannot_become_a_path():
    with pytest.raises(products.ProductError):
        products.product_relative_path({**ROW, "filefracday": "../../escape"})
    with pytest.raises(products.ProductError):
        products.product_relative_path({**ROW, "filtercode": "../x"})


def test_download_publishes_fits_and_provenance_once(monkeypatch, tmp_path):
    payload = _fits_bytes()
    calls = []

    def fake_download(url, destination, **kwargs):
        calls.append(url)
        Path(destination).write_bytes(payload)
        return netclient.DownloadResult(
            path=Path(destination), bytes_written=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            content_length=len(payload),
        )

    monkeypatch.setattr(products.netclient, "download", fake_download)
    store.invalidate_usage_cache()
    request = products.CutoutRequest(255.57691, 12.28378, 50)

    first = products.download_cutout(request, ROW, root=tmp_path)
    assert first["reused"] is False
    path = Path(first["path"])
    sidecar = Path(f"{path}.json")
    assert path.is_file() and sidecar.is_file()
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    assert saved["sha256"] == hashlib.sha256(payload).hexdigest()
    assert saved["fits"]["wcs"]["CTYPE1"] == "RA---TAN"
    assert store.dataset_usage_bytes(tmp_path, refresh=True) == path.stat().st_size + sidecar.stat().st_size

    second = products.download_cutout(request, ROW, root=tmp_path)
    assert second["reused"] is True
    assert len(calls) == 1
    assert products.list_products(root=tmp_path)[0]["product_id"] == first["product_id"]
    assert products.get_product(first["product_id"], root=tmp_path)["path"] == first["path"]
    assert products.product_path(first["product_id"], root=tmp_path) == path.resolve()


def test_download_rejects_invalid_fits_and_cleans_staging(monkeypatch, tmp_path):
    def fake_download(url, destination, **kwargs):
        Path(destination).write_bytes(b"not a FITS file")
        return netclient.DownloadResult(
            path=Path(destination), bytes_written=15,
            sha256=hashlib.sha256(b"not a FITS file").hexdigest(),
            content_length=15,
        )

    monkeypatch.setattr(products.netclient, "download", fake_download)
    with pytest.raises(Exception):
        products.download_cutout(products.CutoutRequest(1, 2), ROW, root=tmp_path)
    assert not list(tmp_path.rglob("*.fits"))
    assert not list(tmp_path.rglob("*.download"))


def test_product_quota_refusal_leaves_no_published_artifact(monkeypatch, tmp_path):
    payload = _fits_bytes()

    def fake_download(url, destination, **kwargs):
        Path(destination).write_bytes(payload)
        return netclient.DownloadResult(
            path=Path(destination), bytes_written=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            content_length=len(payload),
        )

    monkeypatch.setattr(products.netclient, "download", fake_download)
    monkeypatch.setenv("ASTRA_DATASET_CAP_GB", str(1 / 1024 ** 3))
    store.invalidate_usage_cache()
    with pytest.raises(store.DatasetCapacityError):
        products.download_cutout(products.CutoutRequest(1, 2), ROW, root=tmp_path)
    assert not list(tmp_path.rglob("*.fits"))
    assert not list(tmp_path.rglob("*.json"))
