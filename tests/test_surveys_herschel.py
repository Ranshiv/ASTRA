"""Herschel connector contract: cone search parsing (via tap.parse_votable),
flux-error derivation from SNR, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.herschel import DEFAULT_CATALOG, HerschelConnector, _flux_error_mjy


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.headers = {"Content-Type": "application/x-votable+xml"}


def _votable(fields: list[str], rows: list[list[str]]) -> str:
    field_xml = "".join(f'<FIELD name="{name}"/>' for name in fields)
    row_xml = "".join(
        "<TR>" + "".join(f"<TD>{value}</TD>" for value in row) + "</TR>" for row in rows)
    return (
        '<?xml version="1.0"?><VOTABLE><RESOURCE><TABLE>'
        f"<DATA><TABLEDATA>{row_xml}</TABLEDATA></DATA>"
        "</TABLE></RESOURCE></VOTABLE>"
    ).replace("<TABLE>", f"<TABLE>{field_xml}")


HERSCHEL_FIELDS = ["Name", "Band", "RAJ2000", "DEJ2000", "Flux", "snr"]
HERSCHEL_ROW = ["HPPSC070A_J120000.0+000000", "70", "180.000000", "0.000000", "120.5", "8.0"]


class TestFluxErrorMjy:
    def test_derives_error_from_flux_and_snr(self):
        assert _flux_error_mjy(100.0, 5.0) == pytest.approx(20.0)

    def test_returns_none_for_non_positive_snr(self):
        assert _flux_error_mjy(100.0, 0.0) is None
        assert _flux_error_mjy(100.0, -1.0) is None

    def test_returns_none_for_non_numeric_input(self):
        assert _flux_error_mjy(None, 5.0) is None
        assert _flux_error_mjy(100.0, None) is None


class TestHerschelConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = HerschelConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        payload = _votable(HERSCHEL_FIELDS, [HERSCHEL_ROW])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        sources = HerschelConnector().cone_search(cone, limit=10)
        assert len(sources) == 1
        source = sources[0]
        assert source.survey == "Herschel"
        assert source.object_id == "HPPSC070A_J120000.0+000000"
        assert source.extra["band"] == 70
        assert source.extra["flux_mjy"] == pytest.approx(120.5)
        assert source.extra["flux_error_mjy"] == pytest.approx(120.5 / 8.0)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = _votable(["Name", "RAJ2000"], [["X", ""]])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert HerschelConnector().cone_search(cone) == []

    def test_cone_search_handles_empty_response(self, monkeypatch, cone: ConeQuery):
        payload = _votable(HERSCHEL_FIELDS, [])
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert HerschelConnector().cone_search(cone) == []

    def test_cone_search_uses_the_vizier_provider(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(_votable(HERSCHEL_FIELDS, [HERSCHEL_ROW]))

        monkeypatch.setattr(netclient, "get", fake_get)
        HerschelConnector().cone_search(cone, limit=10)
        assert captured["provider"] == "vizier"
        assert captured["params"]["-source"] == HerschelConnector().release

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="Herschel", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert HerschelConnector().fetch_light_curves(source) == []

    def test_describe_does_not_touch_the_network(self):
        description = HerschelConnector().describe()
        assert description["name"] == "Herschel"
        assert description["release"] == DEFAULT_CATALOG


@pytest.mark.live
class TestHerschelLive:
    """Confirmed live this session (2026-08-25): VizieR hosts the real
    "Herschel/PACS Point Source Catalogs" (`VIII/106`) via the same Simple
    Cone Search endpoint `vlass.py` uses. PACS is pointed, not all-sky --
    a cone at RA=180/Dec=0 (the coordinate every other connector's live
    test uses) returned zero rows, confirmed live this session, because no
    PACS pointing covers that field. The ECDFS deep field (RA=53.13,
    Dec=-27.80) does have real PACS coverage, confirmed live this session
    by a direct query returning a real `HPPSC070A_J033236.1-275118` row;
    this test uses that field instead. Still not guaranteed to stay
    non-empty forever if catalogue coverage changes."""

    def test_cone_search_returns_real_rows(self):
        sources = HerschelConnector().cone_search(
            ConeQuery(ra_deg=53.13, dec_deg=-27.80, radius_arcsec=720.0), limit=5)
        assert len(sources) > 0
        assert all(source.survey == "Herschel" for source in sources)
