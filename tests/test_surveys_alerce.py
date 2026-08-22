"""ALeRCE connector contract: cone search parsing, capabilities, real fetch."""

from __future__ import annotations

import pytest

import numpy as np

from astra import netclient
from astra.surveys.alerce import ALeRCEConnector, parse_rows, photometric_residual
from astra.surveys.base import ConeQuery, LightCurve, SourceRef

VALID_OBJECT_ROWS = [
    {"oid": "ZTF20abcdefg", "meanra": 180.122, "meandec": 22.411,
     "classifier": "lc_classifier", "class_name": "RRL", "probability": 0.91,
     "ndet": 42, "firstmjd": 58500.1, "lastmjd": 59000.3},
    {"oid": "ZTF21hijklmn", "ra": 180.130, "dec": 22.420},
]


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _Broken:
    def json(self):
        raise ValueError("not json")


class TestParseRows:
    def test_accepts_bare_list_payload(self):
        assert parse_rows(VALID_OBJECT_ROWS) == VALID_OBJECT_ROWS

    def test_extracts_items_key_from_dict_payload(self):
        assert parse_rows({"items": VALID_OBJECT_ROWS}) == VALID_OBJECT_ROWS

    def test_extracts_results_key_when_items_absent(self):
        assert parse_rows({"results": VALID_OBJECT_ROWS}) == VALID_OBJECT_ROWS

    def test_non_list_non_dict_payload_yields_no_rows(self):
        assert parse_rows("oops") == []

    def test_dict_payload_missing_both_keys_yields_no_rows(self):
        assert parse_rows({"error": "bad request"}) == []

    def test_respects_limit(self):
        assert len(parse_rows(VALID_OBJECT_ROWS, limit=1)) == 1

    def test_keeps_only_dict_rows(self):
        assert parse_rows(["not-a-dict", {"a": 1}]) == [{"a": 1}]


class TestALeRCEConnectorShape:
    def test_capabilities_declare_light_curve_support(self):
        connector = ALeRCEConnector()
        assert "light_curve" in connector.capabilities
        assert "catalogue" in connector.capabilities

    def test_credential_not_required_and_opt_in(self):
        connector = ALeRCEConnector()
        assert connector.credential_required is False
        assert connector.enabled_by_default is False

    def test_default_release_is_ztf(self):
        assert ALeRCEConnector().release == "ztf"

    def test_invalid_release_raises_value_error(self):
        with pytest.raises(ValueError, match="unsupported ALeRCE survey release"):
            ALeRCEConnector(release="sdss")

    def test_release_is_case_insensitive(self):
        assert ALeRCEConnector(release="LSST").release == "lsst"

    def test_resolution_differs_by_release(self):
        assert ALeRCEConnector("ztf").resolution_arcsec == pytest.approx(1.0)
        assert ALeRCEConnector("lsst").resolution_arcsec == pytest.approx(0.2)


class TestConeSearch:
    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_OBJECT_ROWS))
        sources = ALeRCEConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "ALeRCE"
        assert sources[0].object_id == "ZTF20abcdefg"
        assert sources[0].ra_deg == pytest.approx(180.122)
        assert sources[0].extra["class_name"] == "RRL"
        assert sources[0].extra["probability"] == pytest.approx(0.91)
        # second row falls back to ra/dec when meanra/meandec are absent.
        assert sources[1].object_id == "ZTF21hijklmn"
        assert sources[1].ra_deg == pytest.approx(180.130)

    def test_cone_search_sends_survey_param_for_ztf(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(VALID_OBJECT_ROWS)

        monkeypatch.setattr(netclient, "get", fake_get)
        ALeRCEConnector("ztf").cone_search(cone)
        assert captured["params"]["survey"] == "ztf"
        assert captured["provider"] == "alerce"

    def test_cone_search_sends_survey_param_for_lsst(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            return _FakeResponse(VALID_OBJECT_ROWS)

        monkeypatch.setattr(netclient, "get", fake_get)
        ALeRCEConnector("lsst").cone_search(cone)
        assert captured["params"]["survey"] == "lsst"

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = [{"oid": "x"}]
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert ALeRCEConnector().cone_search(cone) == []

    def test_cone_search_skips_rows_missing_oid(self, monkeypatch, cone: ConeQuery):
        payload = [{"ra": 10.0, "dec": 20.0}]
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert ALeRCEConnector().cone_search(cone) == []

    def test_cone_search_handles_non_json_response(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _Broken())
        assert ALeRCEConnector().cone_search(cone) == []

    def test_cone_search_clamps_limit(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            return _FakeResponse(VALID_OBJECT_ROWS)

        monkeypatch.setattr(netclient, "get", fake_get)
        ALeRCEConnector().cone_search(cone, limit=10_000)
        assert captured["params"]["page_size"] == 200


ZTF_DETECTION_ROWS = [
    {"mjd": 59000.1, "magpsf": 18.1, "sigmapsf": 0.05, "fid": 1},
    {"mjd": 59000.2, "magpsf": 17.9, "sigmapsf": 0.04, "fid": 2},
    {"mjd": 59001.1, "magpsf": 18.0, "sigmapsf": 0.05, "fid": "1"},
]

LSST_DETECTION_ROWS = [
    {"mjd": 61000.1, "magpsf": 21.1, "sigmapsf": 0.03, "band": "g"},
    {"mjd": 61000.2, "magpsf": 20.4, "sigmapsf": 0.02, "band": "r"},
    {"mjd": 61000.3, "magpsf": 22.0, "sigmapsf": 0.05, "band": "Y"},
]


class TestFetchLightCurves:
    @pytest.fixture
    def source(self) -> SourceRef:
        return SourceRef(survey="ALeRCE", object_id="ZTF20abcdefg", ra_deg=180.0, dec_deg=22.0)

    def test_maps_ztf_fid_to_band(self, monkeypatch, source: SourceRef):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(ZTF_DETECTION_ROWS))
        curves = ALeRCEConnector("ztf").fetch_light_curves(source)
        bands = {curve.band for curve in curves}
        assert bands == {"g", "r"}

    def test_maps_lsst_band_letters(self, monkeypatch, source: SourceRef):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(LSST_DETECTION_ROWS))
        curves = ALeRCEConnector("lsst").fetch_light_curves(source)
        bands = {curve.band for curve in curves}
        assert bands == {"g", "r", "y"}

    def test_groups_multiple_bands_into_separate_curves_sorted_by_band(self, monkeypatch, source: SourceRef):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(ZTF_DETECTION_ROWS))
        curves = ALeRCEConnector("ztf").fetch_light_curves(source)
        assert [curve.band for curve in curves] == ["g", "r"]
        g_curve = curves[0]
        assert len(g_curve) == 2

    def test_skips_malformed_detection_rows(self, monkeypatch, source: SourceRef):
        payload = ZTF_DETECTION_ROWS + [{"mjd": 59002.0, "fid": 1}]  # missing magpsf/sigmapsf
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        curves = ALeRCEConnector("ztf").fetch_light_curves(source)
        total_points = sum(len(curve) for curve in curves)
        assert total_points == len(ZTF_DETECTION_ROWS)

    def test_unknown_band_falls_back_to_unknown_label(self, monkeypatch, source: SourceRef):
        payload = [{"mjd": 59000.0, "magpsf": 19.0, "sigmapsf": 0.05, "fid": 9}]
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        curves = ALeRCEConnector("ztf").fetch_light_curves(source)
        assert [curve.band for curve in curves] == ["unknown"]

    def test_uses_mag_value_kind_and_mjd_utc_time_system(self, monkeypatch, source: SourceRef):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(ZTF_DETECTION_ROWS))
        curves = ALeRCEConnector("ztf").fetch_light_curves(source)
        for curve in curves:
            assert curve.value_kind == "mag"
            assert curve.time_system == "MJD_UTC"
            assert curve.release == "ztf"

    def test_empty_response_returns_empty_list(self, monkeypatch, source: SourceRef):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse([]))
        assert ALeRCEConnector().fetch_light_curves(source) == []

    def test_non_json_response_returns_empty_list(self, monkeypatch, source: SourceRef):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _Broken())
        assert ALeRCEConnector().fetch_light_curves(source) == []


def _mag_curve(band: str, values: list[float], time_kind: str = "mag") -> LightCurve:
    source = SourceRef(survey="ALeRCE", object_id="obj1", ra_deg=180.0, dec_deg=10.0)
    n = len(values)
    return LightCurve(
        source=source, release="ztf", band=band, value_kind=time_kind,
        time=np.linspace(59000.0, 59010.0, n), value=np.array(values, dtype=np.float32),
        value_err=np.full(n, 0.02, dtype=np.float32), time_system="MJD_UTC",
    )


class TestPhotometricResidual:
    def test_recovers_a_known_injected_offset(self):
        curve = _mag_curve("g", [18.0, 18.0, 18.0, 18.0])
        result = photometric_residual([curve], {"g": 17.5})
        assert result["g"]["residual_mag"] == pytest.approx(0.5, abs=1e-6)
        assert result["g"]["n_points"] == 4
        assert result["g"]["scatter_mag"] == pytest.approx(0.0, abs=1e-6)

    def test_band_missing_from_reference_is_skipped(self):
        curve = _mag_curve("g", [18.0, 18.1])
        assert photometric_residual([curve], {"r": 17.0}) == {}

    def test_empty_curve_is_skipped(self):
        curve = _mag_curve("g", [])
        assert photometric_residual([curve], {"g": 17.0}) == {}

    def test_non_mag_value_kind_is_skipped(self):
        curve = _mag_curve("g", [18.0, 18.0], time_kind="flux")
        assert photometric_residual([curve], {"g": 17.0}) == {}

    def test_multiple_bands_are_reported_independently(self):
        g_curve = _mag_curve("g", [18.0, 18.0])
        r_curve = _mag_curve("r", [17.0, 17.2])
        result = photometric_residual([g_curve, r_curve], {"g": 17.5, "r": 17.0})
        assert set(result) == {"g", "r"}
        assert result["g"]["residual_mag"] == pytest.approx(0.5, abs=1e-6)


class TestSchemaContract:
    """Encodes ALeRCE's DOCUMENTED response shape, not the current parsing
    implementation -- if a real fetch's field names silently diverge from
    this fixture, updating the fixture to match is the signal that the
    contract changed, rather than a silent pass.
    """

    OBJECTS_CONTRACT = {
        "oid": str, "meanra": float, "meandec": float,
        "classifier": str, "class_name": str, "probability": float,
        "ndet": int, "firstmjd": float, "lastmjd": float,
    }
    DETECTIONS_CONTRACT = {
        "mjd": float, "magpsf": float, "sigmapsf": float, "fid": int,
    }

    def _contract_row(self, contract: dict, overrides: dict) -> dict:
        row = dict(overrides)
        for field, kind in contract.items():
            row.setdefault(field, kind())
        return row

    def test_objects_contract_fields_all_survive_cone_search(self, monkeypatch, cone: ConeQuery):
        row = self._contract_row(self.OBJECTS_CONTRACT, {
            "oid": "ZTF20abcdefg", "meanra": 180.5, "meandec": 22.1,
            "classifier": "lc_classifier", "class_name": "RRL",
            "probability": 0.87, "ndet": 12, "firstmjd": 58000.0, "lastmjd": 59000.0,
        })
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse([row]))

        sources = ALeRCEConnector().cone_search(cone)

        assert len(sources) == 1
        source = sources[0]
        assert source.object_id == row["oid"]
        assert source.ra_deg == pytest.approx(row["meanra"])
        assert source.dec_deg == pytest.approx(row["meandec"])
        assert source.extra["classifier"] == row["classifier"]
        assert source.extra["class_name"] == row["class_name"]
        assert source.extra["probability"] == pytest.approx(row["probability"])
        assert source.extra["ndet"] == row["ndet"]
        assert source.extra["firstmjd"] == pytest.approx(row["firstmjd"])
        assert source.extra["lastmjd"] == pytest.approx(row["lastmjd"])

    def test_detections_contract_fields_all_survive_fetch(self, monkeypatch):
        source = SourceRef(survey="ALeRCE", object_id="ZTF20abcdefg", ra_deg=180.0, dec_deg=22.0)
        row = self._contract_row(self.DETECTIONS_CONTRACT, {
            "mjd": 59000.5, "magpsf": 18.3, "sigmapsf": 0.04, "fid": 2,
        })
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse([row]))

        curves = ALeRCEConnector("ztf").fetch_light_curves(source)

        assert len(curves) == 1
        curve = curves[0]
        assert curve.band == "r"  # fid=2 per the documented ZTF_FID_TO_BAND contract
        assert curve.time[0] == pytest.approx(row["mjd"])
        assert curve.value[0] == pytest.approx(row["magpsf"], abs=1e-3)
        assert curve.value_err[0] == pytest.approx(row["sigmapsf"], abs=1e-4)
