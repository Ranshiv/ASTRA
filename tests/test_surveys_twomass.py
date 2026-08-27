"""2MASS connector contract: cone search parsing (via tap.parse_votable),
capabilities, no-op fetch. Same shape as test_surveys_vlass.py."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.twomass import DEFAULT_CATALOG, TwoMASSConnector


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


TWOMASS_FIELDS = ["2MASS", "RAJ2000", "DEJ2000", "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag"]
TWOMASS_ROW = ["12000000+0000000", "180.000000", "0.000000",
              "12.1", "0.03", "11.5", "0.03", "11.3", "0.03"]


class TestTwoMASSConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = TwoMASSConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        payload = _votable(TWOMASS_FIELDS, [TWOMASS_ROW])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        sources = TwoMASSConnector().cone_search(cone, limit=10)
        assert len(sources) == 1
        source = sources[0]
        assert source.survey == "2MASS"
        assert source.object_id == "12000000+0000000"
        assert source.extra["j_mag"] == pytest.approx(12.1)
        assert source.extra["k_mag"] == pytest.approx(11.3)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = _votable(["2MASS", "RAJ2000"], [["X", ""]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert TwoMASSConnector().cone_search(cone) == []

    def test_cone_search_handles_empty_response(self, monkeypatch, cone: ConeQuery):
        payload = _votable(TWOMASS_FIELDS, [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert TwoMASSConnector().cone_search(cone) == []

    def test_cone_search_uses_the_vizier_provider(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(_votable(TWOMASS_FIELDS, [TWOMASS_ROW]))

        monkeypatch.setattr(netclient, "get", fake_get)
        TwoMASSConnector().cone_search(cone, limit=10)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == TwoMASSConnector().release

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="2MASS", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert TwoMASSConnector().fetch_light_curves(source) == []

    def test_describe_does_not_touch_the_network(self):
        description = TwoMASSConnector().describe()
        assert description["name"] == "2MASS"
        assert description["release"] == DEFAULT_CATALOG


@pytest.mark.live
class TestTwoMASSLive:
    """Confirmed live this session (2026-08-25): VizieR hosts the real
    "2MASS All-Sky Catalog of Point Sources" (`II/246/out`) via the same
    Simple Cone Search endpoint `vlass.py` uses."""

    def test_cone_search_returns_real_rows(self):
        sources = TwoMASSConnector().cone_search(
            ConeQuery(ra_deg=180.0, dec_deg=0.0, radius_arcsec=60.0), limit=5)
        assert len(sources) > 0
        assert all(source.survey == "2MASS" for source in sources)
