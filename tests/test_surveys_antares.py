"""ANTARES connector contract: the Elasticsearch cone-search query shape,
JSON:API response parsing, upper-limit exclusion from light curves,
capabilities."""

from __future__ import annotations

import json
import logging

import pytest

from astra import netclient
from astra.surveys.antares import ANTARESConnector
from astra.surveys.base import ConeQuery, SourceRef


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload


LOCI_PAYLOAD = {"data": [
    {"type": "locus_listing", "id": "ANT2021y65ce",
     "attributes": {"ra": 37.284397, "dec": 9.258595,
                    "properties": {"ztf_object_id": "ZTF21abxxjrh", "num_alerts": 14,
                                  "newest_alert_magnitude": 19.45, "anomaly_type": "TBD"}}},
]}

ALERTS_PAYLOAD = {"data": [
    {"id": "ztf_upper_limit:ZTF21abxxjrh-1", "attributes": {
        "mjd": 59435.4, "properties": {"ant_maglim": 20.5}}},
    {"id": "ztf_candidate:1707471650515015019", "attributes": {
        "mjd": 59461.47165510012,
        "properties": {"ztf_magpsf": 19.4539, "ztf_sigmapsf": 0.0766, "ztf_fid": 2}}},
]}


class TestANTARESConnector:
    def test_capabilities_declare_light_curve(self):
        connector = ANTARESConnector()
        assert "light_curve" in connector.capabilities
        assert connector.credential_required is False
        assert connector.enabled_by_default is False

    def test_cone_search_builds_the_real_elasticsearch_query_shape(
            self, monkeypatch, cone: ConeQuery):
        captured: dict = {}

        def fake_get(url, params, timeout, provider):
            captured["url"] = url
            captured["params"] = params
            captured["provider"] = provider
            return _FakeResponse(LOCI_PAYLOAD)

        monkeypatch.setattr(netclient, "get", fake_get)
        ANTARESConnector().cone_search(cone, limit=10)

        assert captured["provider"] == "antares"
        query = json.loads(captured["params"]["elasticsearch_query[locus_listing]"])
        filter_ = query["query"]["bool"]["filter"]["sky_distance"]
        assert filter_["htm16"]["center"] == f"{cone.ra_deg} {cone.dec_deg}"
        assert filter_["distance"] == f"{cone.radius_deg} degree"

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(LOCI_PAYLOAD))
        sources = ANTARESConnector().cone_search(cone, limit=10)
        assert len(sources) == 1
        assert sources[0].survey == "ANTARES"
        assert sources[0].object_id == "ANT2021y65ce"
        assert sources[0].ra_deg == pytest.approx(37.284397)
        assert sources[0].extra["ztf_object_id"] == "ZTF21abxxjrh"

    def test_cone_search_handles_empty_response(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse({"data": []}))
        assert ANTARESConnector().cone_search(cone) == []

    def test_all_rows_failing_to_parse_logs_a_schema_drift_warning(
            self, monkeypatch, cone: ConeQuery, caplog):
        drifted = {"data": [{"id": "x", "attributes": {"latitude": 1.0}}]}
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(drifted))

        with caplog.at_level(logging.WARNING):
            sources = ANTARESConnector().cone_search(cone)

        assert sources == []
        assert any("none parsed as a source" in record.message for record in caplog.records)

    def test_fetch_light_curves_excludes_upper_limits(self, monkeypatch):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(ALERTS_PAYLOAD))
        source = SourceRef(survey="ANTARES", object_id="ANT2021y65ce",
                           ra_deg=37.284397, dec_deg=9.258595)
        curves = ANTARESConnector().fetch_light_curves(source)

        assert len(curves) == 1
        assert curves[0].band == "r"  # fid=2
        assert len(curves[0]) == 1
        assert curves[0].time[0] == pytest.approx(59461.47165510012)
        assert curves[0].value[0] == pytest.approx(19.4539, abs=1e-3)

    def test_fetch_light_curves_handles_no_detections(self, monkeypatch):
        only_limits = {"data": [ALERTS_PAYLOAD["data"][0]]}
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(only_limits))
        source = SourceRef(survey="ANTARES", object_id="x", ra_deg=0.0, dec_deg=0.0)
        assert ANTARESConnector().fetch_light_curves(source) == []

    def test_describe_does_not_touch_the_network(self):
        description = ANTARESConnector().describe()
        assert description["name"] == "ANTARES"
        assert description["release"] == "ztf"


@pytest.mark.live
class TestANTARESLive:
    """Confirmed live this session (2026-09-01): ANTARES's real
    Elasticsearch-filtered cone search (`GET /v1/loci` with an
    `elasticsearch_query[locus_listing]` sky_distance/htm16 filter --  a
    naive flat `?ra=&dec=&radius=` request is silently ignored and returns
    an unrelated default listing with HTTP 200, confirmed live) returns the
    real locus `ANT2021y65ce` (ZTF object `ZTF21abxxjrh`) for a cone at its
    own real position, and `fetch_light_curves` against it returns its one
    real PSF-fit detection (mjd=59461.47165510012, r-band, mag~19.45),
    excluding its 13 real ztf_upper_limit non-detections. See antares.py's
    module docstring for the full finding."""

    def test_cone_search_and_light_curve_return_real_data(self):
        sources = ANTARESConnector().cone_search(
            ConeQuery(ra_deg=37.284397, dec_deg=9.258595, radius_arcsec=60.0), limit=10)
        assert len(sources) > 0
        assert all(source.survey == "ANTARES" for source in sources)

        target = next((s for s in sources if s.object_id == "ANT2021y65ce"), sources[0])
        curves = target.survey and ANTARESConnector().fetch_light_curves(target)
        assert isinstance(curves, list)
