"""Swift connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.swift import SwiftConnector, parse_rows, query_hardness_ratios

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


SXPS_FIELDS = ["IAUName", "RAJ2000", "DEJ2000", "CR0", "HR1", "HR2"]
SXPS_ROW = ["2SXPS J120000.0+200000", "180.0", "20.0", "0.05", "-0.2", "0.1"]


class TestQueryHardnessRatios:
    def test_parses_a_real_row(self, monkeypatch):
        monkeypatch.setattr(netclient, "get",
                           lambda *a, **k: _VotableResponse(_votable(SXPS_FIELDS, [SXPS_ROW])))
        results = query_hardness_ratios(180.0, 20.0, 60.0)
        assert len(results) == 1
        assert results[0]["object_id"] == "2SXPS J120000.0+200000"
        assert results[0]["hr1"] == pytest.approx(-0.2)
        assert results[0]["hr2"] == pytest.approx(0.1)

    def test_uses_the_vizier_provider(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["provider"] = provider
            captured["params"] = params
            return _VotableResponse(_votable(SXPS_FIELDS, []))

        monkeypatch.setattr(netclient, "get", fake_get)
        query_hardness_ratios(180.0, 20.0, 60.0)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == "IX/58"

    def test_empty_response(self, monkeypatch):
        monkeypatch.setattr(netclient, "get",
                           lambda *a, **k: _VotableResponse(_votable(SXPS_FIELDS, [])))
        assert query_hardness_ratios(180.0, 20.0, 60.0) == []


@pytest.mark.live
class TestQueryHardnessRatiosLive:
    """Confirmed live this session (2026-08-25): VizieR IX/58 (2SXPS,
    Evans et al. 2020) is real and returns real count rates and
    pre-computed hardness ratios -- see swift.py's module docstring for
    the full finding.

    Queries M87 (RA=187.7059, Dec=12.3911), not an arbitrary position:
    Swift, like Chandra/XMM, is a POINTED observatory with patchy sky
    coverage (unlike VLASS/eROSITA) -- confirmed directly this session,
    an arbitrary position (RA=180, Dec=0) returned zero rows for exactly
    this reason, not a connector bug.
    """

    def test_returns_real_rows(self):
        results = query_hardness_ratios(187.7059, 12.3911, 60.0)
        assert len(results) > 0
