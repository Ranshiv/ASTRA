"""XMM-Newton connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.xmm import XMMConnector, query_hardness_ratios


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


# `cone_search` now reads the same VizieR IX/69 (4XMM-DR13) shape as
# `query_hardness_ratios` -- see xmm.py's module docstring for why this
# connector moved off its previous (unresolvable-host) CONE_URL entirely.
XMM_FIELDS = ["4XMM", "RA_ICRS", "DE_ICRS", "Flux8", "e_Flux8", "HR1", "HR2", "HR3", "HR4"]
XMM_ROW = ["J120000.0+200000", "180.0", "20.0", "2.1e-14", "3.0e-15",
          "-0.3", "-0.1", "0.1", "0.2"]


class TestXMMConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = XMMConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get",
                           lambda *a, **k: _VotableResponse(_votable(XMM_FIELDS, [XMM_ROW])))
        sources = XMMConnector().cone_search(cone, limit=10)
        assert len(sources) == 1
        assert sources[0].survey == "XMM-Newton"
        assert sources[0].object_id == "J120000.0+200000"
        assert sources[0].ra_deg == pytest.approx(180.0)
        assert sources[0].extra["hr4"] == pytest.approx(0.2)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        fields = ["4XMM", "Flux8"]
        row = ["J120000.0+200000", "2.1e-14"]
        monkeypatch.setattr(netclient, "get",
                           lambda *a, **k: _VotableResponse(_votable(fields, [row])))
        assert XMMConnector().cone_search(cone) == []

    def test_cone_search_uses_the_vizier_provider(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["provider"] = provider
            captured["params"] = params
            return _VotableResponse(_votable(XMM_FIELDS, []))

        monkeypatch.setattr(netclient, "get", fake_get)
        XMMConnector().cone_search(cone, limit=10_000)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == "IX/69"
        assert captured["params"]["-out.max"] == 200

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="XMM-Newton", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert XMMConnector().fetch_light_curves(source) == []


class TestQueryHardnessRatios:
    def test_parses_a_real_row(self, monkeypatch):
        monkeypatch.setattr(netclient, "get",
                           lambda *a, **k: _VotableResponse(_votable(XMM_FIELDS, [XMM_ROW])))
        results = query_hardness_ratios(180.0, 20.0, 60.0)
        assert len(results) == 1
        assert results[0]["object_id"] == "J120000.0+200000"
        assert results[0]["hr1"] == pytest.approx(-0.3)
        assert results[0]["hr4"] == pytest.approx(0.2)

    def test_uses_the_vizier_provider(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["provider"] = provider
            captured["params"] = params
            return _VotableResponse(_votable(XMM_FIELDS, []))

        monkeypatch.setattr(netclient, "get", fake_get)
        query_hardness_ratios(180.0, 20.0, 60.0)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == "IX/69"

    def test_empty_response(self, monkeypatch):
        monkeypatch.setattr(netclient, "get",
                           lambda *a, **k: _VotableResponse(_votable(XMM_FIELDS, [])))
        assert query_hardness_ratios(180.0, 20.0, 60.0) == []


@pytest.mark.live
class TestQueryHardnessRatiosLive:
    """Confirmed live this session (2026-08-25): VizieR IX/69 (4XMM-DR13,
    Webb et al. 2023) is real and returns real total flux and pre-computed
    hardness ratios -- see xmm.py's module docstring for the full finding.

    Queries M87 (RA=187.7059, Dec=12.3911), not an arbitrary position:
    XMM, like Chandra, is a POINTED observatory with patchy sky coverage
    (unlike VLASS/eROSITA) -- confirmed directly this session, an
    arbitrary position (RA=180, Dec=0) returned zero rows for exactly
    this reason, not a connector bug.
    """

    def test_returns_real_rows(self):
        # Confirmed live this session: VizieR's `4XMM` column holds only
        # the coordinate-designation suffix (e.g. "J123049.2+122330"), NOT
        # a "4XMM "-prefixed full name -- a real naming assumption this
        # test originally got wrong, corrected here rather than papered
        # over with a looser assertion.
        results = query_hardness_ratios(187.7059, 12.3911, 60.0)
        assert len(results) > 0
        assert all(r["object_id"].startswith("J") for r in results)


@pytest.mark.live
class TestConeSearchLive:
    """Confirmed live: `cone_search` itself, not just `query_hardness_
    ratios`, now returns real 4XMM rows from the same VizieR fetch -- the
    whole point of rerouting it off the dead (unresolvable host) `CONE_URL`
    this session."""

    def test_returns_real_sources(self):
        sources = XMMConnector().cone_search(
            ConeQuery(ra_deg=187.7059, dec_deg=12.3911, radius_arcsec=60.0), limit=20)
        assert len(sources) > 0
        assert all(s.object_id.startswith("J") for s in sources)
