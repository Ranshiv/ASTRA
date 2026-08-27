"""surveys/ogle.py: real OGLE EWS connector (backlog item 15)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from astra.surveys import ogle
from astra.surveys.base import ConeQuery, SourceRef

# A tiny excerpt shaped like the REAL live season index: unclosed <TD> cells,
# a leading empty cell before the event link, sexagesimal coordinates.
SAMPLE_INDEX_HTML = """
<html><body><table>
<tr><TD><TD> 1<TD><a href="#">2019-BLG-0001</a><TD>BLG500<TD>01  <TD ALIGN="RIGHT"> 17:54:33.42 <TD ALIGN="RIGHT">-29:14:05.9
    <TD ALIGN="RIGHT"> 2458529.58<TD ALIGN="RIGHT">2019-03-19<TD ALIGN="RIGHT"> 54.55
    <TD ALIGN="RIGHT"> 0.176<TD ALIGN="RIGHT"> 5.87<TD ALIGN="RIGHT"> 1.83
    <TD ALIGN="RIGHT"> 0.42<TD ALIGN="RIGHT"> 19.1<TD ALIGN="RIGHT"> 17.9</tr>
<tr><TD><TD> 2<TD><a href="#">2019-BLG-0002</a><TD>BLG501<TD>02  <TD ALIGN="RIGHT"> 18:02:11.10 <TD ALIGN="RIGHT">+00:15:44.0
    <TD ALIGN="RIGHT"> junk<TD ALIGN="RIGHT">2019-03-20<TD ALIGN="RIGHT"> 12.0
    <TD ALIGN="RIGHT"> 0.5<TD ALIGN="RIGHT"> 2.24<TD ALIGN="RIGHT"> 1.10
    <TD ALIGN="RIGHT"> 0.10<TD ALIGN="RIGHT"> 18.5<TD ALIGN="RIGHT"> 18.2</tr>
<tr><td>not an event row at all</td></tr>
</table></body></html>
"""

SAMPLE_PHOT_DAT = """\
2458500.12345 18.502 0.015 2.1 320.5
2458500.67891 18.411 0.012 1.9 310.0
not a data line
2458501.11111 nan 0.020 2.0 300.0
2458501.55555 18.220 0.000 2.2 305.0
2458502.00000 17.980 0.018 2.3 315.0
"""


def connector_sources_from(rows: list[dict]) -> list[SourceRef]:
    return [SourceRef(survey="OGLE", object_id=row["event"], ra_deg=row["ra_deg"],
                      dec_deg=row["dec_deg"], extra=row) for row in rows]


class TestRowCells:
    def test_splits_on_unclosed_opening_tags(self):
        row = "<TD>a  <TD ALIGN=\"RIGHT\"> b<TD>c"
        assert ogle._row_cells(row) == ["a", "b", "c"]


class TestParseEventTable:
    def test_parses_the_real_shaped_rows(self):
        rows = ogle.parse_event_table(SAMPLE_INDEX_HTML)
        events = {row["event"]: row for row in rows}
        assert "2019-BLG-0001" in events

    def test_recovers_published_parameters(self):
        rows = ogle.parse_event_table(SAMPLE_INDEX_HTML)
        row = next(r for r in rows if r["event"] == "2019-BLG-0001")
        assert row["t0_hjd"] == pytest.approx(2458529.58)
        assert row["tE_days"] == pytest.approx(54.55)
        assert row["u0"] == pytest.approx(0.176)
        assert row["year"] == 2019
        assert row["number"] == 1

    def test_converts_sexagesimal_position(self):
        rows = ogle.parse_event_table(SAMPLE_INDEX_HTML)
        row = next(r for r in rows if r["event"] == "2019-BLG-0001")
        assert row["ra_deg"] == pytest.approx(15.0 * (17 + 54 / 60 + 33.42 / 3600), abs=1e-6)
        assert row["dec_deg"] == pytest.approx(-(29 + 14 / 60 + 5.9 / 3600), abs=1e-6)

    def test_handles_malformed_t0_gracefully(self):
        rows = ogle.parse_event_table(SAMPLE_INDEX_HTML)
        row = next(r for r in rows if r["event"] == "2019-BLG-0002")
        assert row["t0_hjd"] is None
        assert row["tE_days"] == pytest.approx(12.0)

    def test_ignores_rows_with_no_event_cell(self):
        rows = ogle.parse_event_table(SAMPLE_INDEX_HTML)
        assert all("event" in row for row in rows)
        assert len(rows) == 2

    def test_respects_limit(self):
        rows = ogle.parse_event_table(SAMPLE_INDEX_HTML, limit=1)
        assert len(rows) == 1


class TestParsePhotometry:
    def test_keeps_only_valid_finite_positive_error_rows(self):
        time, mag, mag_err = ogle.parse_photometry(SAMPLE_PHOT_DAT)
        assert len(time) == 3
        assert time[0] == pytest.approx(2458500.12345)
        assert mag[1] == pytest.approx(18.411)

    def test_drops_nan_and_zero_error_rows(self):
        time, _, _ = ogle.parse_photometry(SAMPLE_PHOT_DAT)
        assert 2458501.11111 not in time
        assert 2458501.55555 not in time

    def test_empty_text_yields_empty_arrays(self):
        time, mag, mag_err = ogle.parse_photometry("")
        assert len(time) == len(mag) == len(mag_err) == 0


class TestPublishedParameters:
    def test_returns_none_when_any_core_parameter_missing(self):
        source = SourceRef(survey="OGLE", object_id="x", ra_deg=0.0, dec_deg=0.0,
                           extra={"t0_hjd": 100.0, "tE_days": None, "u0": 0.1})
        assert ogle.published_parameters(source) is None

    def test_returns_dict_when_core_parameters_present(self):
        source = SourceRef(survey="OGLE", object_id="x", ra_deg=0.0, dec_deg=0.0,
                           extra={"t0_hjd": 100.0, "tE_days": 20.0, "u0": 0.1})
        params = ogle.published_parameters(source)
        assert params == {"t0": 100.0, "tE": 20.0, "u0": 0.1, "f_bl": None, "I_bl": None}

    def test_returns_none_for_missing_extra(self):
        source = SourceRef(survey="OGLE", object_id="x", ra_deg=0.0, dec_deg=0.0)
        assert ogle.published_parameters(source) is None


class TestOGLEConnector:
    def test_capabilities_and_opt_in_flags(self):
        connector = ogle.OGLEConnector()
        assert connector.capabilities == ("catalogue", "light_curve", "published_model")
        assert connector.credential_required is False
        assert connector.enabled_by_default is False

    def test_list_events_rejects_impossible_years(self):
        connector = ogle.OGLEConnector()
        with pytest.raises(ValueError):
            connector.list_events(1900)

    def test_list_events_parses_fixture_response(self, monkeypatch):
        response = Mock(text=SAMPLE_INDEX_HTML)
        mock_get = Mock(return_value=response)
        monkeypatch.setattr(ogle.netclient, "get", mock_get)

        connector = ogle.OGLEConnector()
        sources = connector.list_events(2019)

        assert len(sources) == 2
        assert mock_get.call_args.kwargs.get("provider") == "ogle"
        assert all(isinstance(s, SourceRef) for s in sources)
        assert sources[0].extra["tE_days"] == pytest.approx(54.55)

    def test_fetch_light_curves_parses_fixture_response(self, monkeypatch):
        response = Mock(text=SAMPLE_PHOT_DAT)
        monkeypatch.setattr(ogle.netclient, "get", Mock(return_value=response))

        connector = ogle.OGLEConnector()
        source = SourceRef(survey="OGLE", object_id="2019-BLG-0001", ra_deg=0.0,
                           dec_deg=0.0, extra={"year": 2019, "number": 1})
        curves = connector.fetch_light_curves(source)

        assert len(curves) == 1
        curve = curves[0]
        assert curve.band == "I"
        assert curve.value_kind == "mag"
        assert len(curve.time) == 3

    def test_fetch_light_curves_parses_event_id_when_extra_missing(self, monkeypatch):
        response = Mock(text=SAMPLE_PHOT_DAT)
        monkeypatch.setattr(ogle.netclient, "get", Mock(return_value=response))

        connector = ogle.OGLEConnector()
        source = SourceRef(survey="OGLE", object_id="2019-BLG-0001", ra_deg=0.0, dec_deg=0.0)
        curves = connector.fetch_light_curves(source)
        assert len(curves) == 1

    def test_fetch_light_curves_raises_without_identifiable_event(self):
        connector = ogle.OGLEConnector()
        source = SourceRef(survey="OGLE", object_id="not-an-event", ra_deg=0.0, dec_deg=0.0)
        with pytest.raises(ValueError):
            connector.fetch_light_curves(source)

    def test_fetch_light_curves_returns_empty_list_for_empty_photometry(self, monkeypatch):
        monkeypatch.setattr(ogle.netclient, "get", Mock(return_value=Mock(text="")))
        connector = ogle.OGLEConnector()
        source = SourceRef(survey="OGLE", object_id="2019-BLG-0001", ra_deg=0.0,
                           dec_deg=0.0, extra={"year": 2019, "number": 1})
        assert connector.fetch_light_curves(source) == []

    def test_cone_search_filters_by_position(self, monkeypatch):
        # `ConeQuery` (base.py) has no `year` field, so `cone_search` reads
        # `getattr(query, "year", 0)` -- always 0 here -- and falls back to
        # the current UTC year. list_events() is mocked directly rather than
        # netclient.get so the test does not depend on wall-clock year.
        rows = ogle.parse_event_table(SAMPLE_INDEX_HTML)
        target = next(r for r in rows if r["event"] == "2019-BLG-0001")
        sources = connector_sources_from(rows)
        connector = ogle.OGLEConnector()
        monkeypatch.setattr(connector, "list_events", Mock(return_value=sources))

        query = ConeQuery(ra_deg=target["ra_deg"], dec_deg=target["dec_deg"],
                          radius_arcsec=5.0)
        matches = connector.cone_search(query)

        assert len(matches) == 1
        assert matches[0].object_id == "2019-BLG-0001"

    def test_cone_search_excludes_far_away_events(self, monkeypatch):
        rows = ogle.parse_event_table(SAMPLE_INDEX_HTML)
        sources = connector_sources_from(rows)
        connector = ogle.OGLEConnector()
        monkeypatch.setattr(connector, "list_events", Mock(return_value=sources))

        query = ConeQuery(ra_deg=0.0, dec_deg=0.0, radius_arcsec=1.0)
        assert connector.cone_search(query) == []
