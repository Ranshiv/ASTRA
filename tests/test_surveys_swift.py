"""Swift connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.swift import SwiftConnector, parse_rows

VALID_ROWS = [
    {"IAUName": "SXPS J120021.5+223321", "RA": 180.122, "Decl": 22.411,
     "ObsID": "00012345001", "NumObs": 3, "Rate": 0.021},
    {"source_id": 987654, "ra_deg": 180.130, "dec_deg": 22.420},
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


class TestSwiftConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = SwiftConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_ROWS))
        sources = SwiftConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "Swift"
        assert sources[0].object_id == "SXPS J120021.5+223321"
        assert sources[0].ra_deg == pytest.approx(180.122)
        assert sources[0].extra["num_obs"] == 3
        # second row falls back to source_id / ra_deg / dec_deg.
        assert sources[1].object_id == "987654"
        assert sources[1].ra_deg == pytest.approx(180.130)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = [{"IAUName": "x"}]
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert SwiftConnector().cone_search(cone) == []

    def test_cone_search_handles_non_json_response(self, monkeypatch, cone: ConeQuery):
        class _Broken:
            def json(self):
                raise ValueError("not json")

        monkeypatch.setattr(netclient, "get", lambda *a, **k: _Broken())
        assert SwiftConnector().cone_search(cone) == []

    def test_cone_search_clamps_limit(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(VALID_ROWS)

        monkeypatch.setattr(netclient, "get", fake_get)
        SwiftConnector().cone_search(cone, limit=10_000)
        assert captured["params"]["limit"] == 200
        assert captured["provider"] == "swift"

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="Swift", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert SwiftConnector().fetch_light_curves(source) == []
