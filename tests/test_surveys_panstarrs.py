"""Pan-STARRS connector contract: cone search parsing, capabilities, no-op fetch."""

from __future__ import annotations

import logging

import pytest

from astra import netclient
from astra.surveys.base import ConeQuery, SourceRef
from astra.surveys.panstarrs import PanSTARRSConnector, parse_rows

# The real `mean` endpoint never returns a bare list of row dicts -- it
# returns {"info": [...column definitions...], "data": [...positional row
# arrays...]}, confirmed live against the actual MAST API (a prior version
# of this test file assumed a bare list of dicts, which the real API never
# sends; `parse_rows` silently discarded every real response as a result).
# `VALID_COLUMNS` mirrors that shape at a reduced (5-column) scale for
# readability; `info[i]["name"]` lines up 1:1 with `data[row][i]`, exactly
# as the real response does with its full ~130 columns.
VALID_COLUMNS = ["objID", "raMean", "decMean", "gMeanPSFMag", "rMeanPSFMag"]
VALID_INFO = [{"name": name} for name in VALID_COLUMNS]
VALID_DATA = [
    [190231234567890123, 180.122, 22.411, 18.1, 17.8],
    [190231234567890124, 180.130, 22.420, None, None],
]
VALID_PAYLOAD = {"info": VALID_INFO, "data": VALID_DATA}


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload


class TestParseRows:
    def test_zips_columns_with_positional_row_values(self):
        assert parse_rows(VALID_PAYLOAD) == [
            {"objID": 190231234567890123, "raMean": 180.122, "decMean": 22.411,
             "gMeanPSFMag": 18.1, "rMeanPSFMag": 17.8},
            {"objID": 190231234567890124, "raMean": 180.130, "decMean": 22.420,
             "gMeanPSFMag": None, "rMeanPSFMag": None},
        ]

    def test_non_dict_payload_yields_no_rows(self):
        assert parse_rows(["not-a-dict-payload"]) == []

    def test_missing_info_or_data_yields_no_rows(self):
        assert parse_rows({"info": VALID_INFO}) == []
        assert parse_rows({"data": VALID_DATA}) == []

    def test_row_length_mismatch_is_skipped(self):
        payload = {"info": VALID_INFO, "data": [[1, 2, 3]]}  # too short
        assert parse_rows(payload) == []

    def test_respects_limit(self):
        assert len(parse_rows(VALID_PAYLOAD, limit=1)) == 1


class TestPanSTARRSConnector:
    def test_capabilities_declare_no_light_curve(self):
        connector = PanSTARRSConnector()
        assert "light_curve" not in connector.capabilities
        assert connector.enabled_by_default is True

    def test_cone_search_parses_valid_rows(self, monkeypatch, cone: ConeQuery):
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(VALID_PAYLOAD))
        sources = PanSTARRSConnector().cone_search(cone, limit=10)
        assert len(sources) == 2
        assert sources[0].survey == "Pan-STARRS"
        assert sources[0].object_id == "190231234567890123"
        assert sources[0].ra_deg == pytest.approx(180.122)
        assert sources[0].extra["g_mean"] == pytest.approx(18.1)
        # second row's photometry columns came back null from the archive.
        assert sources[1].extra["g_mean"] is None

    def test_cone_search_skips_rows_missing_position(self, monkeypatch, cone: ConeQuery):
        info = [{"name": "objID"}, {"name": "raMean"}]
        payload = {"info": info, "data": [[1, None]]}
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))
        assert PanSTARRSConnector().cone_search(cone) == []

    def test_cone_search_handles_non_json_response(self, monkeypatch, cone: ConeQuery):
        class _Broken:
            def json(self):
                raise ValueError("not json")

        monkeypatch.setattr(netclient, "get", lambda *a, **k: _Broken())
        assert PanSTARRSConnector().cone_search(cone) == []

    def test_fetch_light_curves_returns_empty(self):
        source = SourceRef(survey="Pan-STARRS", object_id="1", ra_deg=0.0, dec_deg=0.0)
        assert PanSTARRSConnector().fetch_light_curves(source) == []

    def test_all_rows_failing_to_parse_logs_a_schema_drift_warning(self, monkeypatch, cone, caplog):
        # Every row lacks raMean/decMean, so every row hits cone_search's
        # except clause -- indistinguishable from a genuinely empty cone
        # unless this is logged (see panstarrs.py's cone_search).
        info = [{"name": "objID"}, {"name": "someOtherColumn"}]
        payload = {"info": info, "data": [[1, "x"], [2, "y"]]}
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))

        with caplog.at_level(logging.WARNING):
            sources = PanSTARRSConnector().cone_search(cone)

        assert sources == []
        assert any("none parsed as a source" in record.message for record in caplog.records)

    def test_a_genuinely_empty_cone_logs_nothing(self, monkeypatch, cone, caplog):
        payload = {"info": VALID_INFO, "data": []}
        monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))

        with caplog.at_level(logging.WARNING):
            sources = PanSTARRSConnector().cone_search(cone)

        assert sources == []
        assert not any("none parsed as a source" in record.message for record in caplog.records)
