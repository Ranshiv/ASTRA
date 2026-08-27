"""DES connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.des import DES_PIXEL_SCALE_ARCSEC, DESConnector, DESQueryError, parse_csv


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


# `cone` (from conftest.py) is centred at ra=180.122/dec=22.411 with a
# 10-arcsec radius: both rows below fall well within that real circular
# radius (the second is offset ~4 arcsec, not the ~27 arcsec a naive
# corner-of-the-bounding-box row would be), so the real post-filter added
# with the bounding-box fix keeps both, same as before that fix.
VALID_CSV = (
    "coadd_object_id,ra,dec,mag_auto_g,mag_auto_r,mag_auto_i,mag_auto_z,flux_radius_r\n"
    "61234567,180.122,22.411,21.4,20.8,20.5,20.3,7.0\n"
    "61234568,180.1225,22.4115,,,,,\n"
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
        assert sources[0].extra["flux_radius_r_arcsec"] == pytest.approx(7.0 * DES_PIXEL_SCALE_ARCSEC)
        # A row with no photometry is still a usable positional counterpart.
        assert sources[1].object_id == "61234568"
        assert sources[1].ra_deg == pytest.approx(180.1225)
        assert sources[1].extra["flux_radius_r_arcsec"] is None  # empty CSV field, not fabricated

    def test_cone_search_filters_a_bounding_box_corner_outside_the_true_radius(
            self, monkeypatch, cone: ConeQuery):
        # The bounding-box ADQL workaround returns a SQUARE region; a row
        # near a corner (within the box, but outside the real circular
        # radius) must be post-filtered out, not returned as a false
        # positive. cone is ra=180.122/dec=22.411, radius=10 arcsec
        # (~0.00278 deg); this row sits ~0.0025 deg off in BOTH ra and
        # dec -- inside the box, ~12.7 arcsec from centre -- outside the
        # circle.
        payload = ("coadd_object_id,ra,dec,mag_auto_g,mag_auto_r,mag_auto_i,mag_auto_z\n"
                  "61234569,180.1245,22.4135,20.0,20.0,20.0,20.0\n")
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert DESConnector().cone_search(cone) == []

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = "coadd_object_id,ra,dec\n61234567,,22.411\n"
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert DESConnector().cone_search(cone) == []

    def test_cone_search_handles_empty_response(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(""))
        assert DESConnector().cone_search(cone) == []

    def test_cone_search_raises_on_a_real_shaped_tap_error_response(self, monkeypatch, cone: ConeQuery):
        # The exact real error body confirmed live this session against
        # the current NOIRLab Data Lab TAP service.
        error_body = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n<VOTABLE '
            'xmlns="http://www.ivoa.net/xml/VOTable/v1.2" version="1.2">\r\n'
            '  <RESOURCE type="results">\r\n'
            '    <INFO name="QUERY_STATUS" value="ERROR">PSQLException: '
            'ERROR: function point(unknown, double precision, double precision) '
            'does not exist</INFO>\r\n  </RESOURCE>\r\n</VOTABLE>\r\n'
        )
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(error_body))
        with pytest.raises(DESQueryError):
            DESConnector().cone_search(cone)

    def test_cone_search_pages_until_a_short_page(self, monkeypatch, cone: ConeQuery):
        """A cone wider than one page must keep requesting keyset pages,
        each continuing from the previous page's last coadd_object_id, until
        a short page signals the cone is exhausted."""

        def make_csv(start_id: int, count: int) -> str:
            header = "coadd_object_id,ra,dec,mag_auto_g,mag_auto_r,mag_auto_i,mag_auto_z\n"
            # Offsets small enough (<= 250 * 1e-6 deg ~ 0.9 arcsec) to stay
            # inside cone's real 10-arcsec radius after the post-filter.
            rows = "".join(
                f"{start_id + i},{180.122 + i * 0.000001},22.411,20.0,20.0,20.0,20.0\n"
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
        # Bounding-box ADQL, not CONTAINS/POINT/CIRCLE -- see this
        # module's own docstring for why the geometry-function form
        # currently fails against the real live service.
        sql = DESConnector().build_query(cone, 50)
        assert "FROM des_dr2.main" in sql
        assert "CONTAINS" not in sql
        assert "POINT(" not in sql
        assert "dec BETWEEN" in sql
        assert "ra BETWEEN" in sql

    def test_query_bounding_box_is_centred_on_the_cone(self, cone: ConeQuery):
        import math

        sql = DESConnector().build_query(cone, 50)
        assert f"{cone.dec_deg - cone.radius_deg:.8f}" in sql
        assert f"{cone.dec_deg + cone.radius_deg:.8f}" in sql
        cos_dec = math.cos(math.radians(cone.dec_deg))
        ra_half_width = cone.radius_deg / cos_dec
        assert f"{cone.ra_deg - ra_half_width:.8f}" in sql

    def test_query_honours_non_default_release(self, cone: ConeQuery):
        assert "FROM des_dr1.main" in DESConnector(release="dr1").build_query(cone, 5)

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="DES", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert DESConnector().fetch_light_curves(source) == []


@pytest.mark.live
class TestDESConnectorLive:
    """Confirmed live this session (2026-08-26): the bounding-box ADQL
    workaround (this module's own docstring explains why the previous
    `CONTAINS(POINT(...), CIRCLE(...))` form fails service-wide) returns
    real DES DR2 rows in ~4 seconds against a real, DES-covered
    coordinate (RA=45.0, Dec=-40.0, near the SPT-DES deep field)."""

    def test_cone_search_returns_real_rows(self):
        from astra.surveys.base import ConeQuery

        sources = DESConnector().cone_search(
            ConeQuery(ra_deg=45.0, dec_deg=-40.0, radius_arcsec=72.0), limit=5)
        assert len(sources) > 0
        assert all(source.survey == "DES" for source in sources)
        assert all(source.ra_deg is not None for source in sources)
