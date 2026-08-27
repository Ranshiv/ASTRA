"""GALEX connector contract: cone search parsing (via tap.parse_votable),
capabilities, no-op fetch. Same shape as test_surveys_vlass.py."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.galex import DEFAULT_CATALOG, GALEXConnector


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


GALEX_FIELDS = ["Name", "RAJ2000", "DEJ2000", "FUV", "e_FUV", "NUV", "e_NUV"]
GALEX_ROW = ["J120000.0+000000", "180.000000", "0.000000", "19.5", "0.1", "18.2", "0.05"]


class TestGALEXConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = GALEXConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        payload = _votable(GALEX_FIELDS, [GALEX_ROW])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        sources = GALEXConnector().cone_search(cone, limit=10)
        assert len(sources) == 1
        source = sources[0]
        assert source.survey == "GALEX"
        assert source.object_id == "J120000.0+000000"
        assert source.ra_deg == pytest.approx(180.0)
        assert source.extra["fuv_mag"] == pytest.approx(19.5)
        assert source.extra["nuv_mag"] == pytest.approx(18.2)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = _votable(["Name", "RAJ2000"], [["X", ""]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert GALEXConnector().cone_search(cone) == []

    def test_cone_search_handles_empty_response(self, monkeypatch, cone: ConeQuery):
        payload = _votable(GALEX_FIELDS, [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert GALEXConnector().cone_search(cone) == []

    def test_cone_search_uses_the_vizier_provider(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(_votable(GALEX_FIELDS, [GALEX_ROW]))

        monkeypatch.setattr(netclient, "get", fake_get)
        GALEXConnector().cone_search(cone, limit=10)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == GALEXConnector().release

    def test_cone_search_clamps_limit(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            return _FakeResponse(_votable(GALEX_FIELDS, [GALEX_ROW]))

        monkeypatch.setattr(netclient, "get", fake_get)
        GALEXConnector().cone_search(cone, limit=10_000)
        assert captured["params"]["-out.max"] == 200

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="GALEX", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert GALEXConnector().fetch_light_curves(source) == []

    def test_describe_does_not_touch_the_network(self):
        description = GALEXConnector().describe()
        assert description["name"] == "GALEX"
        assert description["release"] == DEFAULT_CATALOG


@pytest.mark.live
class TestGALEXLive:
    """Confirmed live this session (2026-08-25): VizieR hosts the real
    "Revised catalog of GALEX UV sources (GUVcat_AIS GR6+7)"
    (`II/335/galex_ais`) via the same Simple Cone Search endpoint
    `vlass.py` uses."""

    def test_cone_search_returns_real_rows(self):
        sources = GALEXConnector().cone_search(
            ConeQuery(ra_deg=180.0, dec_deg=0.0, radius_arcsec=180.0), limit=5)
        assert len(sources) > 0
        assert all(source.survey == "GALEX" for source in sources)
