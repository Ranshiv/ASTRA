"""eROSITA connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.erosita import DEFAULT_CATALOG, EROSITAConnector


def _votable(fields: list[str], rows: list[list[str]]) -> str:
    field_xml = "".join(f'<FIELD name="{name}"/>' for name in fields)
    row_xml = "".join(
        "<TR>" + "".join(f"<TD>{value}</TD>" for value in row) + "</TR>" for row in rows)
    return (
        '<?xml version="1.0"?><VOTABLE><RESOURCE><TABLE>'
        f"<DATA><TABLEDATA>{row_xml}</TABLEDATA></DATA>"
        "</TABLE></RESOURCE></VOTABLE>"
    ).replace("<TABLE>", f"<TABLE>{field_xml}")


class _VotableResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.headers = {"Content-Type": "application/x-votable+xml"}


EROSITA_FIELDS = ["IAUName", "RA_ICRS", "DE_ICRS", "MLcts1", "MLFlux1"]
EROSITA_ROW = ["1eRASS J120000.0+200000", "180.0", "20.0", "42.3", "1.1e-13"]


class TestEROSITAConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = EROSITAConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_describe_does_not_touch_the_network(self):
        # Regression case for the real bug found this session in
        # vlass.py: SurveyConnector.describe() reads self.release
        # unconditionally; every new connector needs it.
        description = EROSITAConnector().describe()
        assert description["name"] == "eROSITA"
        assert description["release"] == DEFAULT_CATALOG

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        payload = _votable(EROSITA_FIELDS, [EROSITA_ROW])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _VotableResponse(payload))
        sources = EROSITAConnector().cone_search(cone, limit=10)
        assert len(sources) == 1
        assert sources[0].survey == "eROSITA"
        assert sources[0].object_id == "1eRASS J120000.0+200000"
        assert sources[0].ra_deg == pytest.approx(180.0)
        assert sources[0].extra["flux_band1"] == pytest.approx(1.1e-13)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = _votable(["IAUName", "RA_ICRS"], [["X", ""]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _VotableResponse(payload))
        assert EROSITAConnector().cone_search(cone) == []

    def test_cone_search_handles_empty_response(self, monkeypatch, cone: ConeQuery):
        payload = _votable(EROSITA_FIELDS, [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _VotableResponse(payload))
        assert EROSITAConnector().cone_search(cone) == []

    def test_cone_search_uses_the_vizier_provider(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _VotableResponse(_votable(EROSITA_FIELDS, [EROSITA_ROW]))

        monkeypatch.setattr(netclient, "get", fake_get)
        EROSITAConnector().cone_search(cone, limit=10)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == DEFAULT_CATALOG

    def test_cone_search_clamps_limit(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            return _VotableResponse(_votable(EROSITA_FIELDS, [EROSITA_ROW]))

        monkeypatch.setattr(netclient, "get", fake_get)
        EROSITAConnector().cone_search(cone, limit=10_000)
        assert captured["params"]["-out.max"] == 200

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="eROSITA", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert EROSITAConnector().fetch_light_curves(source) == []


@pytest.mark.live
class TestEROSITALive:
    """Confirmed live this session (2026-08-25): VizieR `J/A+A/682/A34`
    (SRG/eROSITA eRASS1, Merloni et al. 2024) is real and returns real
    rows -- see erosita.py's module docstring for the full finding."""

    def test_cone_search_returns_real_rows(self):
        sources = EROSITAConnector().cone_search(
            ConeQuery(ra_deg=180.0, dec_deg=0.0, radius_arcsec=1080.0), limit=5)
        assert len(sources) > 0
        assert all(source.survey == "eROSITA" for source in sources)
