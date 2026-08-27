"""VLASS connector contract: cone search parsing (via tap.parse_votable),
capabilities, no-op fetch, and the NVSS cross-match helper."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.vlass import DEFAULT_CATALOG, VLASSConnector, query_nvss_flux_1_4ghz


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.headers = {"Content-Type": "application/x-votable+xml"}


def _votable(fields: list[str], rows: list[list[str]]) -> str:
    field_xml = "".join(f'<FIELD name="{name}"/>' for name in fields)
    row_xml = "".join(
        "<TR>" + "".join(f"<TD>{value}</TD>" for value in row) + "</TR>" for row in rows)
    return (
        '<?xml version="1.0"?><VOTABLE><RESOURCE><TABLE>'
        f"<DATA><TABLEDATA>{row_xml}</TABLEDATA></DATA>"
        "</TABLE></RESOURCE></VOTABLE>"
    ).replace("<TABLE>", f"<TABLE>{field_xml}")


VLASS_FIELDS = ["CompName", "RAJ2000", "DEJ2000", "Ftot", "e_Ftot", "Fpeak", "e_Fpeak",
               "DupFlag", "QualFlag", "NVSSdist", "FIRSTdist"]
VLASS_ROW = ["VLASS1QLCIR J120000.00+200000.0", "180.000000", "20.000000",
            "8.450", "0.381", "6.765", "0.186", "0", "1", "3.881", "0.341"]


class TestVLASSConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = VLASSConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        payload = _votable(VLASS_FIELDS, [VLASS_ROW])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        sources = VLASSConnector().cone_search(cone, limit=10)
        assert len(sources) == 1
        source = sources[0]
        assert source.survey == "VLASS"
        assert source.object_id == "VLASS1QLCIR J120000.00+200000.0"
        assert source.ra_deg == pytest.approx(180.0)
        assert source.extra["flux_total_mjy"] == 8.45
        assert source.extra["epoch"] == "1"

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = _votable(["CompName", "RAJ2000"], [["X", ""]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert VLASSConnector().cone_search(cone) == []

    def test_cone_search_handles_empty_response(self, monkeypatch, cone: ConeQuery):
        payload = _votable(VLASS_FIELDS, [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert VLASSConnector().cone_search(cone) == []

    def test_cone_search_uses_the_vizier_provider(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(_votable(VLASS_FIELDS, [VLASS_ROW]))

        monkeypatch.setattr(netclient, "get", fake_get)
        VLASSConnector().cone_search(cone, limit=10)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == VLASSConnector().release

    def test_cone_search_clamps_limit(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            return _FakeResponse(_votable(VLASS_FIELDS, [VLASS_ROW]))

        monkeypatch.setattr(netclient, "get", fake_get)
        VLASSConnector().cone_search(cone, limit=10_000)
        assert captured["params"]["-out.max"] == 200

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="VLASS", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert VLASSConnector().fetch_light_curves(source) == []

    def test_describe_does_not_touch_the_network(self):
        # A real bug, found via the full test suite (not this module's own
        # tests, which never called describe()): SurveyConnector.describe()
        # reads self.release unconditionally, and this connector originally
        # exposed only self.catalog, raising AttributeError the moment
        # surveys.describe_all() iterated over the registry.
        description = VLASSConnector().describe()
        assert description["name"] == "VLASS"
        assert description["release"] == DEFAULT_CATALOG


NVSS_FIELDS = ["NVSS", "S1.4", "e_S1.4"]


class TestQueryNvssFlux:
    def test_parses_a_real_matching_row(self, monkeypatch):
        payload = _votable(NVSS_FIELDS, [["120010-000003", "6.6", "0.4"]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        result = query_nvss_flux_1_4ghz(180.0, 20.0)
        assert result["frequency_ghz"] == 1.4
        assert result["flux_mjy"] == pytest.approx(6.6)
        assert result["flux_err_mjy"] == pytest.approx(0.4)
        assert result["nvss_name"] == "120010-000003"

    def test_returns_none_when_no_match(self, monkeypatch):
        payload = _votable(NVSS_FIELDS, [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert query_nvss_flux_1_4ghz(180.0, 20.0) is None

    def test_uses_the_vizier_provider(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["provider"] = provider
            captured["params"] = params
            return _FakeResponse(_votable(NVSS_FIELDS, []))

        monkeypatch.setattr(netclient, "get", fake_get)
        query_nvss_flux_1_4ghz(180.0, 20.0)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == "VIII/65/nvss"


@pytest.mark.live
class TestVLASSLive:
    """Confirmed live this session (2026-08-25): VizieR's Simple Cone
    Search hosts the real "VLASS QL Ep.1 Catalog, CIRADA version"
    (`J/ApJS/255/30/comp`) and NVSS (`VIII/65/nvss`) -- see
    `surveys/vlass.py`'s module docstring for the full finding, including
    why this targets VizieR rather than CADC/CIRADA's own TAP endpoint."""

    def test_cone_search_returns_real_rows(self):
        sources = VLASSConnector().cone_search(
            ConeQuery(ra_deg=180.0, dec_deg=0.0, radius_arcsec=1800.0), limit=5)
        assert len(sources) > 0
        assert all(source.survey == "VLASS" for source in sources)
        assert all(source.extra["flux_total_mjy"] is not None for source in sources)

    def test_nvss_cross_match_returns_a_real_source(self):
        result = query_nvss_flux_1_4ghz(180.044, -0.001, radius_arcsec=15.0)
        assert result is not None
        assert result["flux_mjy"] > 0
