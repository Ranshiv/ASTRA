"""Chandra connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.chandra import ChandraConnector, parse_rows, query_band_fluxes

VALID_ROWS = [
    {"name": "2CXO J120021.5+223321", "ra": 180.122, "dec": 22.411,
     "obsid": 12345, "instrument": "ACIS-S", "flux_aper_b": 1.2e-14},
    {"src_id": 987654, "ra_deg": 180.130, "dec_deg": 22.420},
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


class TestChandraConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = ChandraConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_ROWS))
        sources = ChandraConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "Chandra"
        assert sources[0].object_id == "2CXO J120021.5+223321"
        assert sources[0].ra_deg == pytest.approx(180.122)
        assert sources[0].extra["instrument"] == "ACIS-S"
        # second row falls back to src_id / ra_deg / dec_deg.
        assert sources[1].object_id == "987654"
        assert sources[1].ra_deg == pytest.approx(180.130)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = [{"name": "x"}]
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert ChandraConnector().cone_search(cone) == []

    def test_cone_search_handles_non_json_response(self, monkeypatch, cone: ConeQuery):
        class _Broken:
            def json(self):
                raise ValueError("not json")

        monkeypatch.setattr(netclient, "get", lambda *a, **k: _Broken())
        assert ChandraConnector().cone_search(cone) == []

    def test_cone_search_clamps_limit(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(VALID_ROWS)

        monkeypatch.setattr(netclient, "get", fake_get)
        ChandraConnector().cone_search(cone, limit=10_000)
        assert captured["params"]["limit"] == 200
        assert captured["provider"] == "chandra"

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="Chandra", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert ChandraConnector().fetch_light_curves(source) == []


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


CSC_FIELDS = ["2CXO", "RAICRS", "DEICRS", "Fluxb", "Fluxs", "Fluxm", "Fluxh",
             "HRhm", "HRhs", "HRms"]
CSC_ROW = ["2CXO J120000.0+200000", "180.0", "20.0", "1.5e-14", "5.0e-15", "6.0e-15",
          "4.0e-15", "-0.2", "-0.1", "0.05"]


class TestQueryBandFluxes:
    def test_parses_a_real_row(self, monkeypatch):
        monkeypatch.setattr(netclient, "get",
                           lambda *a, **k: _VotableResponse(_votable(CSC_FIELDS, [CSC_ROW])))
        results = query_band_fluxes(180.0, 20.0, 60.0)
        assert len(results) == 1
        assert results[0]["object_id"] == "2CXO J120000.0+200000"
        assert results[0]["flux_hard"] == pytest.approx(4.0e-15)
        assert results[0]["hr_hard_soft"] == pytest.approx(-0.1)

    def test_uses_the_vizier_provider(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["provider"] = provider
            captured["params"] = params
            return _VotableResponse(_votable(CSC_FIELDS, []))

        monkeypatch.setattr(netclient, "get", fake_get)
        query_band_fluxes(180.0, 20.0, 60.0)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == "IX/70"

    def test_empty_response(self, monkeypatch):
        monkeypatch.setattr(netclient, "get",
                           lambda *a, **k: _VotableResponse(_votable(CSC_FIELDS, [])))
        assert query_band_fluxes(180.0, 20.0, 60.0) == []


@pytest.mark.live
class TestQueryBandFluxesLive:
    """Confirmed live this session (2026-08-25): VizieR IX/70 (Chandra
    Source Catalog 2.1, Evans et al. 2024) is real and returns real
    per-band fluxes and hardness ratios -- see chandra.py's module
    docstring for the full finding, including why this targets VizieR
    rather than this connector's own (undocumented-parameter) CONE_URL.

    Queries M87 (RA=187.7059, Dec=12.3911), not an arbitrary position:
    unlike VLASS/eROSITA (genuinely all-sky surveys), Chandra is a
    POINTED observatory whose catalogue only covers its observed fields
    -- confirmed directly this session, a first attempt at RA=180, Dec=0
    (a position with no known Chandra pointing) returned zero rows, which
    is real patchy-coverage behaviour, not a connector bug. M87 is one of
    the most Chandra-observed targets in the sky (99 real rows returned).
    """

    def test_returns_real_rows(self):
        results = query_band_fluxes(187.7059, 12.3911, 60.0)
        assert len(results) > 0
        assert all(r["object_id"].startswith("2CXO") for r in results)
