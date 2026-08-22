"""JWST connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import json

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.jwst import JWSTConnector, parse_rows

VALID_PAYLOAD = {
    "status": "COMPLETE",
    "fields": [{"name": "obsid"}, {"name": "s_ra"}, {"name": "s_dec"}],
    "data": [
        {"obsid": "8765432100", "obs_id": "jw02733-o001", "s_ra": 180.122,
         "s_dec": 22.411, "instrument_name": "NIRCam", "filters": "F200W",
         "t_min": 59800.25, "t_exptime": 3600.0, "dataproduct_type": "image"},
        {"obs_id": "jw02733-o002", "s_ra": 180.130, "s_dec": 22.420},
    ],
}


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload


class TestParseRows:
    def test_keeps_only_dict_rows(self):
        assert parse_rows({"data": ["not-a-dict", {"a": 1}]}) == [{"a": 1}]

    def test_non_dict_payload_yields_no_rows(self):
        assert parse_rows(["unexpected"]) == []

    def test_missing_data_key_yields_no_rows(self):
        assert parse_rows({"status": "ERROR", "msg": "bad request"}) == []

    def test_respects_limit(self):
        assert len(parse_rows(VALID_PAYLOAD, limit=1)) == 1


class TestJWSTConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = JWSTConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_PAYLOAD))
        sources = JWSTConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "JWST"
        assert sources[0].object_id == "8765432100"
        assert sources[0].ra_deg == pytest.approx(180.122)
        assert sources[0].extra["instrument"] == "NIRCam"
        # second row falls back to obs_id when obsid is absent.
        assert sources[1].object_id == "jw02733-o002"
        assert sources[1].ra_deg == pytest.approx(180.130)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = {"data": [{"obsid": "1"}]}
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert JWSTConnector().cone_search(cone) == []

    def test_cone_search_handles_non_json_response(self, monkeypatch, cone: ConeQuery):
        class _Broken:
            def json(self):
                raise ValueError("not json")

        monkeypatch.setattr(netclient, "get", lambda *a, **k: _Broken())
        assert JWSTConnector().cone_search(cone) == []

    def test_cone_search_pages_until_a_short_page(self, monkeypatch, cone: ConeQuery):
        def make_payload(start: int, count: int) -> dict:
            return {"data": [
                {"obsid": str(start + i), "s_ra": 180.0 + i * 0.0001, "s_dec": 22.0}
                for i in range(count)
            ]}

        pages = [make_payload(1, 200), make_payload(201, 30)]
        calls: list[dict] = []

        def fake_get(url, params, timeout, provider):
            calls.append(json.loads(params["request"]))
            return _FakeResponse(pages[len(calls) - 1])

        monkeypatch.setattr(netclient, "get", fake_get)
        sources = JWSTConnector().cone_search(cone, limit=10_000)

        assert len(calls) == 2
        assert calls[0]["page"] == 1
        assert calls[1]["page"] == 2
        assert len(sources) == 230

    def test_cone_search_clamps_limit(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(VALID_PAYLOAD)

        monkeypatch.setattr(netclient, "get", fake_get)
        JWSTConnector().cone_search(cone, limit=10_000)
        request = json.loads(captured["params"]["request"])
        assert request["pagesize"] == 200
        assert captured["provider"] == "mast"

    def test_cone_search_filters_to_jwst_collection(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            return _FakeResponse(VALID_PAYLOAD)

        monkeypatch.setattr(netclient, "get", fake_get)
        JWSTConnector().cone_search(cone)
        request = json.loads(captured["params"]["request"])
        assert request["params"]["filters"] == [
            {"paramName": "obs_collection", "values": ["JWST"]},
        ]

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="JWST", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert JWSTConnector().fetch_light_curves(source) == []
