"""Pan-STARRS connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.panstarrs import PanSTARRSConnector, parse_rows

VALID_ROWS = [
    {"objID": 190231234567890123, "raMean": 180.122, "decMean": 22.411,
     "gMeanPSFMag": 18.1, "rMeanPSFMag": 17.8},
    {"objid": 190231234567890124, "ra": 180.130, "dec": 22.420},
]


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload


class TestParseRows:
    def test_keeps_only_dict_rows(self):
        assert parse_rows(["not-a-dict", {"a": 1}]) == [{"a": 1}]

    def test_non_list_payload_yields_no_rows(self):
        assert parse_rows({"error": "bad request"}) == []

    def test_respects_limit(self):
        assert len(parse_rows(VALID_ROWS, limit=1)) == 1


class TestPanSTARRSConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = PanSTARRSConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_ROWS))
        sources = PanSTARRSConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "Pan-STARRS"
        assert sources[0].object_id == "190231234567890123"
        assert sources[0].ra_deg == pytest.approx(180.122)
        assert sources[0].extra["g_mean"] == pytest.approx(18.1)
        # second row has no photometry keys at all, so they read as None.
        assert sources[1].extra["g_mean"] is None

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = [{"objID": 1}]
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert PanSTARRSConnector().cone_search(cone) == []

    def test_cone_search_handles_non_json_response(self, monkeypatch, cone: ConeQuery):
        class _Broken:
            def json(self):
                raise ValueError("not json")

        monkeypatch.setattr(netclient, "get", lambda *a, **k: _Broken())
        assert PanSTARRSConnector().cone_search(cone) == []

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="Pan-STARRS", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert PanSTARRSConnector().fetch_light_curves(source) == []
