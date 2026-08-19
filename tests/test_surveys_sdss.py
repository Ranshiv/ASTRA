"""SDSS connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery
from astra.surveys.sdss import SDSSConnector, parse_csv


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


VALID_CSV = (
    "objID,ra,dec,plate,mjd,fiberID\n"
    "1237648720693379140,180.122,22.411,751,52251,131\n"
    "1237648720693379141,180.130,22.420,,,\n"
)


class TestParseCsv:
    def test_parses_rows_into_dicts(self):
        rows = parse_csv(VALID_CSV)
        assert rows[0]["objID"] == "1237648720693379140"
        assert rows[0]["ra"] == "180.122"

    def test_respects_limit(self):
        assert len(parse_csv(VALID_CSV, limit=1)) == 1

    def test_empty_payload_yields_no_rows(self):
        assert parse_csv("") == []


class TestSDSSConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = SDSSConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_CSV))
        sources = SDSSConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "SDSS"
        assert sources[0].object_id == "1237648720693379140"
        assert sources[0].ra_deg == pytest.approx(180.122)
        assert sources[0].extra["spectrum_ready"] is True

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = "objID,ra,dec,plate,mjd,fiberID\n1,,22.4,,,\n"
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
        from astra.surveys.base import SourceRef

        source = SourceRef(survey="SDSS", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert SDSSConnector().fetch_light_curves(source) == []
