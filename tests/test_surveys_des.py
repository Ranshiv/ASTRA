"""DES connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.des import DESConnector, parse_csv


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


VALID_CSV = (
    "coadd_object_id,ra,dec,mag_auto_g,mag_auto_r,mag_auto_i,mag_auto_z\n"
    "61234567,180.122,22.411,21.4,20.8,20.5,20.3\n"
    "61234568,180.130,22.420,,,,\n"
)


class TestParseCsv:
    def test_parses_rows_into_dicts(self):
        rows = parse_csv(VALID_CSV)
        assert rows[0]["coadd_object_id"] == "61234567"
        assert rows[0]["ra"] == "180.122"

    def test_respects_limit(self):
        assert len(parse_csv(VALID_CSV, limit=1)) == 1

    def test_empty_payload_yields_no_rows(self):
        assert parse_csv("") == []


class TestDESConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = DESConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_CSV))
        sources = DESConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "DES"
        assert sources[0].object_id == "61234567"
        assert sources[0].ra_deg == pytest.approx(180.122)
        assert sources[0].extra["g_mean"] == "21.4"
        # A row with no photometry is still a usable positional counterpart.
        assert sources[1].object_id == "61234568"
        assert sources[1].ra_deg == pytest.approx(180.130)

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = "coadd_object_id,ra,dec\n61234567,,22.411\n"
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert DESConnector().cone_search(cone) == []

    def test_cone_search_handles_empty_response(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(""))
        assert DESConnector().cone_search(cone) == []

    def test_cone_search_pages_until_a_short_page(self, monkeypatch, cone: ConeQuery):
        """A cone wider than one page must keep requesting keyset pages,
        each continuing from the previous page's last coadd_object_id, until
        a short page signals the cone is exhausted."""

        def make_csv(start_id: int, count: int) -> str:
            header = "coadd_object_id,ra,dec,mag_auto_g,mag_auto_r,mag_auto_i,mag_auto_z\n"
            rows = "".join(
                f"{start_id + i},{180.0 + i * 0.0001},22.0,20.0,20.0,20.0,20.0\n"
                for i in range(count)
            )
            return header + rows

        pages = [make_csv(1, 200), make_csv(201, 50)]
        calls: list[dict] = []

        def fake_get(url, params, timeout, provider):
            calls.append(params)
            return _FakeResponse(pages[len(calls) - 1])

        monkeypatch.setattr(netclient, "get", fake_get)
        sources = DESConnector().cone_search(cone, limit=10_000)

        assert len(calls) == 2
        assert "coadd_object_id > 1" not in calls[0]["QUERY"]
        assert "coadd_object_id > 200" in calls[1]["QUERY"]
        assert len(sources) == 250

    def test_cone_search_clamps_limit(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(VALID_CSV)

        monkeypatch.setattr(netclient, "get", fake_get)
        DESConnector().cone_search(cone, limit=10_000)
        assert captured["params"]["MAXREC"] == 200
        assert "TOP 200" in captured["params"]["QUERY"]
        assert captured["provider"] == "datalab"

    def test_query_targets_release_table_and_cone(self, cone: ConeQuery):
        sql = DESConnector().build_query(cone, 50)
        assert "FROM des_dr2.main" in sql
        assert "CONTAINS(POINT('ICRS', ra, dec)" in sql
        assert f"CIRCLE('ICRS', {cone.ra_deg}, {cone.dec_deg}" in sql

    def test_query_honours_non_default_release(self, cone: ConeQuery):
        assert "FROM des_dr1.main" in DESConnector(release="dr1").build_query(cone, 5)

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="DES", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert DESConnector().fetch_light_curves(source) == []
