"""XMM-Newton connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.xmm import XMMConnector, parse_rows

VALID_ROWS = [
    {"SRCID": 200123456789, "SC_RA": 180.122, "SC_DEC": 22.411,
     "OBS_ID": "0123456789", "SC_EP_8_FLUX": 3.4e-14, "N_DETECTIONS": 2},
    {"iauname": "4XMM J120021.5+223321", "ra_deg": 180.130, "dec_deg": 22.420},
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


class TestXMMConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = XMMConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_ROWS))
        sources = XMMConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "XMM-Newton"
        assert sources[0].object_id == "200123456789"
        assert sources[0].ra_deg == pytest.approx(180.122)
        assert sources[0].extra["n_detections"] == 2
        # second row falls back to iauname / ra_deg / dec_deg.
        assert sources[1].object_id == "4XMM J120021.5+223321"
        assert sources[1].ra_deg == pytest.approx(180.130)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = [{"SRCID": 1}]
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert XMMConnector().cone_search(cone) == []

    def test_cone_search_handles_non_json_response(self, monkeypatch, cone: ConeQuery):
        class _Broken:
            def json(self):
                raise ValueError("not json")

        monkeypatch.setattr(netclient, "get", lambda *a, **k: _Broken())
        assert XMMConnector().cone_search(cone) == []

    def test_cone_search_clamps_limit(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(VALID_ROWS)

        monkeypatch.setattr(netclient, "get", fake_get)
        XMMConnector().cone_search(cone, limit=10_000)
        assert captured["params"]["limit"] == 200
        assert captured["provider"] == "xmm"

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="XMM-Newton", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert XMMConnector().fetch_light_curves(source) == []
