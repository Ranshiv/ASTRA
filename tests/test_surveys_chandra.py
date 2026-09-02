"""Chandra connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.chandra import ChandraConnector, query_band_fluxes


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


# `cone_search` now reads the same VizieR IX/70 (CSC 2.1) shape as
# `query_band_fluxes` -- see chandra.py's module docstring for why this
# connector moved off its previous (dead-parameter) CONE_URL entirely.
CSC_FIELDS = ["2CXO", "RAICRS", "DEICRS", "Fluxb", "Fluxs", "Fluxm", "Fluxh",
             "HRhm", "HRhs", "HRms"]
CSC_ROW = ["2CXO J120000.0+200000", "180.0", "20.0", "1.5e-14", "5.0e-15", "6.0e-15",
          "4.0e-15", "-0.2", "-0.1", "0.05"]


class TestChandraConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = ChandraConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get",
                           lambda *a, **k: _VotableResponse(_votable(CSC_FIELDS, [CSC_ROW])))
        sources = ChandraConnector().cone_search(cone, limit=10)
        assert len(sources) == 1
        assert sources[0].survey == "Chandra"
        assert sources[0].object_id == "2CXO J120000.0+200000"
        assert sources[0].ra_deg == pytest.approx(180.0)
        assert sources[0].extra["flux_hard"] == pytest.approx(4.0e-15)
        assert sources[0].extra["hr_hard_soft"] == pytest.approx(-0.1)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        fields = ["2CXO", "Fluxb"]
        row = ["2CXO J120000.0+200000", "1.5e-14"]
        monkeypatch.setattr(netclient, "get",
                           lambda *a, **k: _VotableResponse(_votable(fields, [row])))
        assert ChandraConnector().cone_search(cone) == []

    def test_cone_search_uses_the_vizier_provider(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["provider"] = provider
            captured["params"] = params
            return _VotableResponse(_votable(CSC_FIELDS, []))

        monkeypatch.setattr(netclient, "get", fake_get)
        ChandraConnector().cone_search(cone, limit=10_000)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == "IX/70"
        # limit clamps the same way it always did, just as `-out.max` now.
        assert captured["params"]["-out.max"] == 200

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="Chandra", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert ChandraConnector().fetch_light_curves(source) == []


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
    rather than this connector's previous (dead-parameter) CONE_URL.

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


@pytest.mark.live
class TestConeSearchLive:
    """Confirmed live: `cone_search` itself, not just `query_band_fluxes`,
    now returns real CSC 2.1 rows from the same VizieR fetch -- the whole
    point of rerouting it off the dead `CONE_URL` this session."""

    def test_returns_real_sources(self):
        sources = ChandraConnector().cone_search(
            ConeQuery(ra_deg=187.7059, dec_deg=12.3911, radius_arcsec=60.0), limit=20)
        assert len(sources) > 0
        assert all(s.object_id.startswith("2CXO") for s in sources)
