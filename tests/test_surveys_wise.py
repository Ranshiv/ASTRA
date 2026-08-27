"""WISE connector contract: cone search parsing (via tap.parse_votable),
capabilities, no-op fetch. Same shape as test_surveys_vlass.py."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.wise import DEFAULT_CATALOG, WISEConnector


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


WISE_FIELDS = ["AllWISE", "RAJ2000", "DEJ2000", "W1mag", "e_W1mag", "W2mag", "e_W2mag",
              "W3mag", "e_W3mag", "W4mag", "e_W4mag"]
WISE_ROW = ["J120000.00+000000.0", "180.000000", "0.000000",
           "10.1", "0.02", "9.9", "0.02", "9.5", "0.05", "9.0", "0.15"]


class TestWISEConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = WISEConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        payload = _votable(WISE_FIELDS, [WISE_ROW])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        sources = WISEConnector().cone_search(cone, limit=10)
        assert len(sources) == 1
        source = sources[0]
        assert source.survey == "WISE"
        assert source.object_id == "J120000.00+000000.0"
        assert source.extra["w1_mag"] == pytest.approx(10.1)
        assert source.extra["w4_mag"] == pytest.approx(9.0)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = _votable(["AllWISE", "RAJ2000"], [["X", ""]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert WISEConnector().cone_search(cone) == []

    def test_cone_search_handles_empty_response(self, monkeypatch, cone: ConeQuery):
        payload = _votable(WISE_FIELDS, [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert WISEConnector().cone_search(cone) == []

    def test_cone_search_uses_the_vizier_provider(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(_votable(WISE_FIELDS, [WISE_ROW]))

        monkeypatch.setattr(netclient, "get", fake_get)
        WISEConnector().cone_search(cone, limit=10)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == WISEConnector().release

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="WISE", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert WISEConnector().fetch_light_curves(source) == []

    def test_describe_does_not_touch_the_network(self):
        description = WISEConnector().describe()
        assert description["name"] == "WISE"
        assert description["release"] == DEFAULT_CATALOG


@pytest.mark.live
class TestWISELive:
    """Confirmed live this session (2026-08-25): VizieR hosts the real
    "AllWISE Data Release" (`II/328/allwise`) via the same Simple Cone
    Search endpoint `vlass.py` uses."""

    def test_cone_search_returns_real_rows(self):
        sources = WISEConnector().cone_search(
            ConeQuery(ra_deg=180.0, dec_deg=0.0, radius_arcsec=60.0), limit=5)
        assert len(sources) > 0
        assert all(source.survey == "WISE" for source in sources)
