"""ASAS-SN connector contract: cone search over the real lookup_cone POST
protocol (parsed from Parquet bytes despite the server's "arrow" format
field name), capabilities, no-op fetch."""

from __future__ import annotations

import io
import logging

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from astra import netclient
from astra.surveys.asassn import ASASSNConnector, DEFAULT_CATALOG
from astra.surveys.base import ConeQuery, SourceRef


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content


def _parquet_bytes(columns: dict[str, list]) -> bytes:
    table = pa.table({name: pa.array(values) for name, values in columns.items()})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


ROWS_PAYLOAD = _parquet_bytes({
    "asas_sn_id": [335007755071, 549756950226],
    "ra_deg": [280.00621296, 279.98004362],
    "dec_deg": [-20.02029242, -19.9570338],
    "gaia_mag": [15.093, 12.539],
    "gaia_b_mag": [16.114, 13.513],
    "gaia_r_mag": [14.082, 11.571],
})


class TestASASSNConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = ASASSNConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is False

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "post", lambda *a, **k: _FakeResponse(ROWS_PAYLOAD))
        sources = ASASSNConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "ASAS-SN"
        assert sources[0].object_id == "335007755071"
        assert sources[0].ra_deg == pytest.approx(280.00621296)
        assert sources[0].extra["gaia_mag"] == pytest.approx(15.093)

    def test_cone_search_uses_a_post_with_a_json_body_and_the_asassn_provider(
            self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_post(url, json, timeout, provider, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["provider"] = provider
            return _FakeResponse(ROWS_PAYLOAD)

        monkeypatch.setattr(netclient, "post", fake_post)
        ASASSNConnector().cone_search(cone, limit=10)

        assert captured["provider"] == "asassn"
        assert captured["json"]["catalog"] == DEFAULT_CATALOG
        assert captured["json"]["download"] is False
        assert f"ra{cone.ra_deg}" in captured["url"]
        assert f"dec{cone.dec_deg}" in captured["url"]

    def test_cone_search_handles_empty_response(self, monkeypatch, cone: ConeQuery):
        empty = _parquet_bytes({"asas_sn_id": [], "ra_deg": [], "dec_deg": []})
        monkeypatch.setattr(netclient, "post", lambda *a, **k: _FakeResponse(empty))
        assert ASASSNConnector().cone_search(cone) == []

    def test_all_rows_failing_to_parse_logs_a_schema_drift_warning(
            self, monkeypatch, cone: ConeQuery, caplog):
        drifted = _parquet_bytes({"some_other_id": [1, 2], "x": [1.0, 2.0]})
        monkeypatch.setattr(netclient, "post", lambda *a, **k: _FakeResponse(drifted))

        with caplog.at_level(logging.WARNING):
            sources = ASASSNConnector().cone_search(cone)

        assert sources == []
        assert any("none parsed as a source" in record.message for record in caplog.records)

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="ASAS-SN", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert ASASSNConnector().fetch_light_curves(source) == []

    def test_describe_does_not_touch_the_network(self):
        description = ASASSNConnector().describe()
        assert description["name"] == "ASAS-SN"
        assert description["release"] == DEFAULT_CATALOG


@pytest.mark.live
class TestASASSNLive:
    """Confirmed live this session (2026-09-01): ASAS-SN Sky Patrol's
    lookup_cone server (asassn-lb01.ifa.hawaii.edu:9006) answers a real
    POST for the `stellar_main` catalog with real asas_sn_id/ra_deg/dec_deg/
    gaia_mag/gaia_b_mag/gaia_r_mag rows -- a cone at RA=280.0, Dec=-20.0
    with a 180 arcsec radius returned 10+ real rows repeatedly this
    session. See asassn.py's module docstring for the full finding."""

    def test_cone_search_returns_real_rows(self):
        sources = ASASSNConnector().cone_search(
            ConeQuery(ra_deg=280.0, dec_deg=-20.0, radius_arcsec=180.0), limit=10)
        assert len(sources) > 0
        assert all(source.survey == "ASAS-SN" for source in sources)
        assert all(source.extra["gaia_mag"] is not None for source in sources)
