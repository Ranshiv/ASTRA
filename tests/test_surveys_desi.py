"""DESI connector contract: bounding-box query construction, exact
circular-distance filtering, capabilities, no-op spectrum fetch.

FIXED this session (see surveys/desi.py's module docstring for the full
finding): `cone_search` now goes through `tap.async_query` with a
bounding-box `BETWEEN` prefilter (not `CONTAINS`/`CIRCLE`, which fails with
a real, permanent Postgres error on this table regardless of transport),
followed by an exact `crossmatch.angular_separation_arcsec` filter on the
returned rows. A `live`-marked class exercises the real round trip."""

from __future__ import annotations

import pytest

from astra import tap
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.desi import DESIConnector, _bounding_box


def _fake_result(rows: list[dict]) -> dict:
    return {"rows": rows, "state": "match" if rows else "no_match", "error": None}


VALID_ROWS = [
    {"targetid": 39628379988689587, "mean_fiber_ra": 180.122, "mean_fiber_dec": 22.411,
     "healpix": 9239, "survey": "main", "program": "bright", "spectype": "GALAXY",
     "z": 0.2727416, "zerr": 1.567e-05, "zwarn": 0, "deltachi2": 8028.8,
     "desiname": "DESI J251.5358+25.2900"},
    # Deliberately close to VALID_ROWS[0]'s position (within the shared
    # `cone` fixture's 10" radius) -- only its zwarn/z fields are the point
    # of this second row, not a different position.
    {"targetid": 39628369037361415, "mean_fiber_ra": 180.1225, "mean_fiber_dec": 22.4115,
     "healpix": 9237, "survey": "main", "program": "bright", "spectype": "GALAXY",
     "z": None, "zerr": None, "zwarn": -1, "deltachi2": None,
     "desiname": "DESI J252.2906+24.7855"},
]


class TestBoundingBox:
    def test_covers_the_requested_center(self):
        box = _bounding_box(180.0, 20.0, 1.0)
        assert box["dec_lo"] < 20.0 < box["dec_hi"]
        lo, hi = box["ra_ranges"][0]
        assert lo < 180.0 < hi

    def test_widens_with_declination_via_cos_scaling(self):
        equator = _bounding_box(180.0, 0.0, 1.0)
        mid_lat = _bounding_box(180.0, 60.0, 1.0)
        equator_width = equator["ra_ranges"][0][1] - equator["ra_ranges"][0][0]
        mid_lat_width = mid_lat["ra_ranges"][0][1] - mid_lat["ra_ranges"][0][0]
        assert mid_lat_width > equator_width

    def test_splits_into_two_ranges_at_the_ra_wraparound(self):
        box = _bounding_box(0.5, 0.0, 2.0)
        assert len(box["ra_ranges"]) == 2

    def test_falls_back_to_full_ra_range_near_a_pole(self):
        box = _bounding_box(180.0, 89.5, 1.0)
        assert box["ra_ranges"] == [(0.0, 360.0)]

    def test_dec_bounds_are_clamped_to_valid_range(self):
        box = _bounding_box(180.0, 89.9, 5.0)
        assert box["dec_hi"] == 90.0


class TestDESIConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = DESIConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(tap, "async_query", lambda *a, **k: _fake_result(VALID_ROWS))
        sources = DESIConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "DESI"
        assert sources[0].object_id == "39628379988689587"
        assert sources[0].ra_deg == pytest.approx(180.122)
        assert sources[0].extra["z"] == pytest.approx(0.2727416)
        assert sources[0].extra["healpix"] == 9239
        # A row with an unresolved fit (zwarn=-1, no z) is still a usable
        # positional counterpart -- z/z_err fall back to None, not 0.0.
        assert sources[1].extra["z"] is None
        assert sources[1].extra["z_err"] is None

    def test_cone_search_filters_out_rows_outside_the_exact_radius(self, monkeypatch):
        query = ConeQuery(ra_deg=180.0, dec_deg=0.0, radius_arcsec=5.0)
        # This row sits inside the bounding box the query would build, but
        # well outside the exact 5" circle -- must be dropped, not returned.
        far_row = {"targetid": 1, "mean_fiber_ra": 180.01, "mean_fiber_dec": 0.0,
                  "healpix": None, "survey": None, "program": None, "spectype": None,
                  "z": None, "zerr": None, "zwarn": None, "deltachi2": None, "desiname": None}
        monkeypatch.setattr(tap, "async_query", lambda *a, **k: _fake_result([far_row]))
        assert DESIConnector().cone_search(query) == []

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        payload = [{"targetid": 1, "mean_fiber_ra": None, "mean_fiber_dec": 22.411}]
        monkeypatch.setattr(tap, "async_query", lambda *a, **k: _fake_result(payload))
        assert DESIConnector().cone_search(cone) == []

    def test_cone_search_handles_empty_response(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(tap, "async_query", lambda *a, **k: _fake_result([]))
        assert DESIConnector().cone_search(cone) == []

    def test_cone_search_pages_until_a_short_page(self, monkeypatch, cone: ConeQuery):
        def make_rows(start_id: int, count: int) -> list[dict]:
            return [{"targetid": start_id + i, "mean_fiber_ra": cone.ra_deg,
                    "mean_fiber_dec": cone.dec_deg, "z": 0.1, "zerr": 0.001,
                    "healpix": None, "survey": None, "program": None, "spectype": None,
                    "zwarn": None, "deltachi2": None, "desiname": None}
                   for i in range(count)]

        pages = [make_rows(1, 200), make_rows(201, 50)]
        calls: list[str] = []

        def fake_async_query(service, adql, **kwargs):
            calls.append(adql)
            return _fake_result(pages[len(calls) - 1])

        monkeypatch.setattr(tap, "async_query", fake_async_query)
        sources = DESIConnector().cone_search(cone, limit=10_000)

        assert len(calls) == 2
        assert "targetid > 1" not in calls[0]
        assert "targetid > 200" in calls[1]
        assert len(sources) == 250

    def test_cone_search_clamps_limit(self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_async_query(service, adql, **kwargs):
            captured["service"] = service
            captured["adql"] = adql
            captured["kwargs"] = kwargs
            return _fake_result(VALID_ROWS)

        monkeypatch.setattr(tap, "async_query", fake_async_query)
        DESIConnector().cone_search(cone, limit=10_000)
        assert captured["kwargs"]["max_rows"] == 200
        assert "TOP 200" in captured["adql"]
        assert captured["kwargs"]["provider"] == "datalab"

    def test_query_targets_release_table_with_a_bounding_box(self, cone: ConeQuery):
        sql = DESIConnector().build_query(cone, 50)
        assert "FROM desi_dr1.zpix" in sql
        assert "BETWEEN" in sql
        assert "CONTAINS" not in sql
        assert "CIRCLE" not in sql

    def test_query_honours_non_default_release(self, cone: ConeQuery):
        assert "FROM desi_edr.zpix" in DESIConnector(release="edr").build_query(cone, 5)

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="DESI", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert DESIConnector().fetch_light_curves(source) == []


@pytest.mark.live
class TestDESILive:
    """Confirmed live this session (2026-08-25): the rewritten `cone_search`
    (async TAP + bounding-box prefilter + exact circular-distance filter)
    works end-to-end against the real `desi_dr1.zpix` table -- see
    surveys/desi.py's module docstring for the full finding. A real query
    at RA=180, Dec=0 returned real DESI targets, all confirmed within the
    exact requested radius."""

    def test_cone_search_returns_real_rows_within_the_exact_radius(self):
        from astra.crossmatch import angular_separation_arcsec

        query = ConeQuery(ra_deg=180.0, dec_deg=0.0, radius_arcsec=1800.0)
        sources = DESIConnector().cone_search(query, limit=20)
        assert len(sources) > 0
        assert all(source.survey == "DESI" for source in sources)
        assert all(
            angular_separation_arcsec(query.ra_deg, query.dec_deg,
                                      source.ra_deg, source.dec_deg) <= query.radius_arcsec
            for source in sources)
