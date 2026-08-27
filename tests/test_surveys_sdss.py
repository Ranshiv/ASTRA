"""SDSS connector contract: cone search parsing, capabilities, spectrum
acquisition, and the SDSS-specific spectral feature adapter."""

from __future__ import annotations

import numpy as np
import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.sdss import (
    SDSSConnector, SdssSpectrumError, extract_sdss_line_measurements,
    extract_sdss_spectrum, fetch_spectrum, parse_csv,
    query_spectroscopic_redshifts, spectrum_url,
)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


# SpecObjAll-shaped rows, matching the real live schema verified while
# building this connector (bestObjID, specObjID, plate, mjd, fiberID, run2d,
# class) -- NOT PhotoObj, which does not carry these columns (confirmed
# live: a PhotoObj query for them returns HTTP 500).
VALID_CSV = (
    "bestObjID,specObjID,ra,dec,plate,mjd,fiberID,run2d,class\n"
    "1237648720693379140,299489980023179264,180.122,22.411,751,52251,131,26,STAR\n"
    "0,299489980023179265,180.130,22.420,752,52252,132,26,STAR\n"
)


class TestParseCsv:
    def test_parses_rows_into_dicts(self):
        rows = parse_csv(VALID_CSV)
        assert rows[0]["bestObjID"] == "1237648720693379140"
        assert rows[0]["ra"] == "180.122"

    def test_respects_limit(self):
        assert len(parse_csv(VALID_CSV, limit=1)) == 1

    def test_empty_payload_yields_no_rows(self):
        assert parse_csv("") == []

    def test_strips_a_leading_table1_comment_line(self):
        # A real bug, found and fixed via a live check while building
        # `query_spectroscopic_redshifts`: the live SkyServer endpoint
        # prefixes its CSV with a `#Table1` line, which `csv.DictReader`
        # would otherwise misparse as the header row -- silently dropping
        # every real row (confirmed live, see `parse_csv`'s docstring).
        prefixed = "#Table1\n" + VALID_CSV
        rows = parse_csv(prefixed)
        assert rows[0]["bestObjID"] == "1237648720693379140"
        assert rows[0]["ra"] == "180.122"
        assert len(rows) == 2


class TestSDSSConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = SDSSConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_queries_specobjall_not_photoobj(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["sql"] = params["cmd"]
            return _FakeResponse(VALID_CSV)

        monkeypatch.setattr(netclient, "get", fake_get)
        SDSSConnector().cone_search(cone, limit=10)
        assert "FROM SpecObjAll" in captured["sql"]
        assert "PhotoObj" not in captured["sql"]

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_CSV))
        sources = SDSSConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "SDSS"
        assert sources[0].object_id == "1237648720693379140"
        assert sources[0].ra_deg == pytest.approx(180.122)
        assert sources[0].extra["spectrum_ready"] is True
        assert sources[0].extra["run2d"] == "26"
        assert sources[0].extra["photometric_match"] is True

    def test_zero_bestobjid_falls_back_to_specobjid(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_CSV))
        sources = SDSSConnector().cone_search(cone, limit=10)
        assert sources[1].object_id == "299489980023179265"
        assert sources[1].extra["photometric_match"] is False

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = "bestObjID,specObjID,ra,dec,plate,mjd,fiberID,run2d,class\n1,2,,22.4,,,,,\n"
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert SDSSConnector().cone_search(cone) == []

    def test_cone_search_clamps_limit(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["sql"] = params["cmd"]
            return _FakeResponse(VALID_CSV)

        monkeypatch.setattr(netclient, "get", fake_get)
        SDSSConnector().cone_search(cone, limit=10_000)
        assert "TOP 200" in captured["sql"]

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="SDSS", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert SDSSConnector().fetch_light_curves(source) == []


class TestSpectrumUrl:
    def _source(self, **extra_overrides) -> SourceRef:
        extra = {"plate": 266, "mjd": 51630, "fiber_id": 336, "run2d": "26"}
        extra.update(extra_overrides)
        return SourceRef(survey="SDSS", object_id="1", ra_deg=0.0, dec_deg=0.0, extra=extra)

    def test_builds_the_real_verified_url_pattern(self):
        # Real, live-verified URL for this exact plate/mjd/fiber/run2d
        # combination (see surveys/sdss.py's module docstring).
        url = spectrum_url(self._source())
        assert url == (
            "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/"
            "lite/0266/spec-0266-51630-0336.fits"
        )

    def test_missing_plate_raises_a_clear_error(self):
        source = self._source(plate=None)
        with pytest.raises(SdssSpectrumError):
            spectrum_url(source)

    def test_source_with_no_extra_raises_a_clear_error(self):
        source = SourceRef(survey="SDSS", object_id="1", ra_deg=0.0, dec_deg=0.0)
        with pytest.raises(SdssSpectrumError):
            spectrum_url(source)


class TestFetchSpectrum:
    def test_downloads_to_the_expected_dataset_path(self, tmp_path, monkeypatch):
        source = SourceRef(survey="SDSS", object_id="1", ra_deg=0.0, dec_deg=0.0,
                           extra={"plate": 266, "mjd": 51630, "fiber_id": 336, "run2d": "26"})
        captured: dict = {}
        expected = tmp_path / "SDSS" / "spectra" / "spec-0266-51630-0336.fits"

        class FakeResult:
            path = expected

        def fake_download(url, destination, *, provider, overwrite):
            captured["url"] = url
            captured["destination"] = destination
            captured["provider"] = provider
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"fake fits bytes")
            return FakeResult()

        monkeypatch.setattr(netclient, "download", fake_download)
        result = fetch_spectrum(source, dataset_root=tmp_path)

        assert captured["provider"] == "sdss"
        assert captured["destination"] == expected
        assert result == expected

    def test_reuses_an_existing_file_without_downloading_again(self, tmp_path, monkeypatch):
        source = SourceRef(survey="SDSS", object_id="1", ra_deg=0.0, dec_deg=0.0,
                           extra={"plate": 266, "mjd": 51630, "fiber_id": 336, "run2d": "26"})
        existing = tmp_path / "SDSS" / "spectra" / "spec-0266-51630-0336.fits"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"already here")

        def fail_download(*args, **kwargs):
            raise AssertionError("should not re-download an existing file")

        monkeypatch.setattr(netclient, "download", fail_download)
        result = fetch_spectrum(source, dataset_root=tmp_path)
        assert result == existing


class TestExtractSdssSpectrum:
    def _fixture_path(self, tmp_path, *, and_mask_bad_fraction=0.0,
                      elodie_teff: float | None = 6890.0):
        from astropy.io import fits

        n = 200
        loglam = np.linspace(3.55, 3.75, n)  # ~3548-5623 Angstrom
        flux = np.full(n, 20.0) + np.sin(np.linspace(0, 6, n))
        ivar = np.full(n, 0.1)
        and_mask = np.zeros(n, dtype=np.int32)
        if and_mask_bad_fraction:
            bad = int(n * and_mask_bad_fraction)
            and_mask[:bad] = 1

        coadd = fits.BinTableHDU.from_columns([
            fits.Column(name="flux", format="E", array=flux.astype(np.float32)),
            fits.Column(name="loglam", format="E", array=loglam.astype(np.float32)),
            fits.Column(name="ivar", format="E", array=ivar.astype(np.float32)),
            fits.Column(name="and_mask", format="J", array=and_mask),
        ], name="COADD")

        specobj_columns = [
            fits.Column(name="CLASS", format="6A", array=np.array(["STAR"])),
            fits.Column(name="SUBCLASS", format="4A", array=np.array(["G5"])),
            fits.Column(name="ELODIE_TEFF", format="E",
                       array=np.array([elodie_teff if elodie_teff is not None else 0.0],
                                     dtype=np.float32)),
            fits.Column(name="ELODIE_LOGG", format="E", array=np.array([4.5], dtype=np.float32)),
            fits.Column(name="ELODIE_FEH", format="E", array=np.array([-0.2], dtype=np.float32)),
            fits.Column(name="ELODIE_SPTYPE", format="4A", array=np.array(["G5V"])),
        ]
        specobj = fits.BinTableHDU.from_columns(specobj_columns, name="SPECOBJ")

        hdul = fits.HDUList([fits.PrimaryHDU(), coadd, specobj])
        path = tmp_path / f"spec-test-{and_mask_bad_fraction}-{elodie_teff}.fits"
        hdul.writeto(path, overwrite=True)
        return path

    def test_converts_loglam_to_wavelength(self, tmp_path):
        path = self._fixture_path(tmp_path)
        payload = extract_sdss_spectrum(path)
        # 10**3.55 ~= 3548 Angstrom, 10**3.75 ~= 5623 Angstrom.
        assert 3500 < payload["features"]["wavelength_start"] < 3600
        assert 5500 < payload["features"]["wavelength_end"] < 5700

    def test_and_mask_flagged_points_are_excluded(self, tmp_path):
        clean = extract_sdss_spectrum(self._fixture_path(tmp_path, and_mask_bad_fraction=0.0))
        half_bad = extract_sdss_spectrum(self._fixture_path(tmp_path, and_mask_bad_fraction=0.5))
        assert half_bad["features"]["points"] < clean["features"]["points"]

    def test_real_elodie_teff_is_surfaced(self, tmp_path):
        payload = extract_sdss_spectrum(self._fixture_path(tmp_path, elodie_teff=6890.0))
        assert payload["stellar_parameters"]["teff_k"] == pytest.approx(6890.0)
        assert payload["stellar_parameters"]["source"] == "sdss_elodie_legacy_pipeline"
        assert payload["class"] == "STAR"

    def test_zero_elodie_teff_is_treated_as_unavailable_not_fabricated(self, tmp_path):
        payload = extract_sdss_spectrum(self._fixture_path(tmp_path, elodie_teff=0.0))
        assert payload["stellar_parameters"] is None


class TestExtractSdssLineMeasurements:
    def _fixture_path(self, tmp_path, *, rows: list[dict]):
        from astropy.io import fits

        n = len(rows)
        columns = [
            fits.Column(name="LINENAME", format="13A",
                       array=np.array([r["name"] for r in rows])),
            fits.Column(name="LINEWAVE", format="D",
                       array=np.array([r["rest_wave"] for r in rows], dtype=np.float64)),
            fits.Column(name="LINEZ", format="E",
                       array=np.array([r["z"] for r in rows], dtype=np.float32)),
            fits.Column(name="LINEZ_ERR", format="E", array=np.zeros(n, dtype=np.float32)),
            fits.Column(name="LINESIGMA", format="E",
                       array=np.array([r["sigma_kms"] for r in rows], dtype=np.float32)),
            fits.Column(name="LINESIGMA_ERR", format="E", array=np.zeros(n, dtype=np.float32)),
            fits.Column(name="LINEAREA", format="E",
                       array=np.array([r["area"] for r in rows], dtype=np.float32)),
            fits.Column(name="LINEAREA_ERR", format="E", array=np.ones(n, dtype=np.float32)),
            fits.Column(name="LINEEW", format="E", array=np.ones(n, dtype=np.float32)),
            fits.Column(name="LINEEW_ERR", format="E", array=np.ones(n, dtype=np.float32)),
            fits.Column(name="LINECONTLEVEL", format="E", array=np.full(n, 100.0, dtype=np.float32)),
            fits.Column(name="LINECHI2", format="E", array=np.ones(n, dtype=np.float32)),
        ]
        spzline = fits.BinTableHDU.from_columns(columns, name="SPZLINE")
        hdul = fits.HDUList([fits.PrimaryHDU(), spzline])
        path = tmp_path / "spz-test.fits"
        hdul.writeto(path, overwrite=True)
        return path

    def test_parses_a_real_detected_line(self, tmp_path):
        path = self._fixture_path(tmp_path, rows=[
            {"name": "H-beta", "rest_wave": 4861.35, "z": 0.05, "sigma_kms": 80.0, "area": 500.0},
        ])
        measurements = extract_sdss_line_measurements(path)
        assert len(measurements) == 1
        assert measurements[0]["name"] == "H-beta"
        assert measurements[0]["observed_wavelength_angstrom"] == pytest.approx(
            4861.35 * 1.05)
        assert measurements[0]["sigma_angstrom"] > 0

    def test_undetected_lines_are_excluded_not_fabricated_as_zero(self, tmp_path):
        path = self._fixture_path(tmp_path, rows=[
            {"name": "Ly_alpha", "rest_wave": 1215.67, "z": 0.0, "sigma_kms": 0.0, "area": 0.0},
            {"name": "H-alpha", "rest_wave": 6562.79, "z": 0.05, "sigma_kms": 90.0, "area": 700.0},
        ])
        measurements = extract_sdss_line_measurements(path)
        assert len(measurements) == 1
        assert measurements[0]["name"] == "H-alpha"

    def test_missing_spzline_extension_returns_empty_list(self, tmp_path):
        from astropy.io import fits

        hdul = fits.HDUList([fits.PrimaryHDU()])
        path = tmp_path / "no-spzline.fits"
        hdul.writeto(path, overwrite=True)
        assert extract_sdss_line_measurements(path) == []


REDSHIFT_CSV = (
    "bestObjID,specObjID,ra,dec,z,zErr,class\n"
    "1237648720693379140,299489980023179264,180.122,22.411,0.0992,0.00026,GALAXY\n"
    "0,299489980023179265,180.130,22.420,0.1302,0.0000294,GALAXY\n"
)


class TestQuerySpectroscopicRedshifts:
    def test_queries_specobjall_for_z_and_zerr(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["sql"] = params["cmd"]
            captured["provider"] = provider
            return _FakeResponse(REDSHIFT_CSV)

        monkeypatch.setattr(netclient, "get", fake_get)
        query_spectroscopic_redshifts(180.122, 22.411, 10.0)
        assert "FROM SpecObjAll" in captured["sql"]
        assert " z, zErr" in captured["sql"] or "z, zErr" in captured["sql"]
        assert captured["provider"] == "sdss"

    def test_parses_real_redshift_rows(self, monkeypatch):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(REDSHIFT_CSV))
        sources = query_spectroscopic_redshifts(180.122, 22.411, 10.0)
        assert len(sources) == 2
        assert sources[0].object_id == "1237648720693379140"
        assert sources[0].extra["z"] == pytest.approx(0.0992)
        assert sources[0].extra["z_err"] == pytest.approx(0.00026)
        assert sources[0].extra["class"] == "GALAXY"

    def test_zero_bestobjid_falls_back_to_specobjid(self, monkeypatch):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(REDSHIFT_CSV))
        sources = query_spectroscopic_redshifts(180.122, 22.411, 10.0)
        assert sources[1].object_id == "299489980023179265"

    def test_skips_rows_with_missing_or_non_finite_z(self, monkeypatch):
        bad_csv = (
            "bestObjID,specObjID,ra,dec,z,zErr,class\n"
            "1237648720693379140,299489980023179264,180.122,22.411,,0.00026,GALAXY\n"
        )
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(bad_csv))
        assert query_spectroscopic_redshifts(180.122, 22.411, 10.0) == []


@pytest.mark.live
class TestQuerySpectroscopicRedshiftsLive:
    """Confirmed live this session (2026-08-24): a real cone search around
    RA=185.0, Dec=15.0 with a 3600 arcsec radius returns real SpecObjAll
    rows with populated z/zErr -- and, before `parse_csv`'s `#Table1`
    fix, returned zero rows despite the raw HTTP response containing real
    data, a genuine silent-failure bug this test guards against
    regressing."""

    def test_returns_real_redshifts(self):
        sources = query_spectroscopic_redshifts(185.0, 15.0, 3600.0, limit=5)
        assert len(sources) > 0, (
            "query_spectroscopic_redshifts returned zero rows against the live "
            "SkyServer endpoint for a position known (this session) to have real "
            "SpecObjAll matches -- either the service is down, or parse_csv's "
            "#Table1-stripping has regressed; check both before assuming this "
            "is a flaky network failure")
        assert all(s.extra["z"] > 0 for s in sources)


@pytest.mark.live
class TestSDSSVReleaseLive:
    """Confirmed live this session (2026-08-24), for roadmap item 24: SDSS-V
    (DR19) needs no new connector -- `release="dr19"` works against this
    same `SkyServerWS/SqlSearch` endpoint, same schema, same `#Table1`
    quirk. See surveys/sdss.py's module docstring for the full finding,
    including that DR19's SpecObjAll currently carries only legacy
    (pre-FPS) reductions."""

    def test_dr19_cone_search_returns_real_rows(self, cone: ConeQuery):
        sources = SDSSConnector(release="dr19").cone_search(
            ConeQuery(ra_deg=185.0, dec_deg=15.0, radius_arcsec=3600.0), limit=5)
        assert len(sources) > 0, (
            "DR19 SpecObjAll cone search returned zero rows for a position "
            "known (this session) to have real matches -- either the service "
            "is down or something about DR19's schema/endpoint has changed")

    def test_dr19_spectrum_file_is_reachable_at_the_stable_dr17_path(self):
        # A real DR19-cataloged legacy row (plate=7261, mjd=56603,
        # fiber=458, run2d=v5_13_2), confirmed this session to resolve at
        # the SAME dr17-hosted SAS path fetch_spectrum already uses --
        # SDSS spectrum files are not duplicated per release.
        source = SourceRef(survey="SDSS", object_id="1", ra_deg=0.0, dec_deg=0.0,
                           extra={"plate": 7261, "mjd": 56603, "fiber_id": 458,
                                  "run2d": "v5_13_2"})
        url = spectrum_url(source)
        response = netclient.get(url, {}, timeout=30, provider="sdss")
        assert response.status_code == 200
