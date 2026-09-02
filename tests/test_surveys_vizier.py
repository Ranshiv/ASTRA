"""Shared VizierConeConnector base: the schema-drift warning is tested once
here rather than duplicated across all eight connectors that use it."""

from __future__ import annotations

import logging

from astra import netclient
from astra.surveys.base import ConeQuery
from astra.surveys._vizier import VizierConeConnector


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


class _FakeConnector(VizierConeConnector):
    name = "FakeCatalog"
    id_column = "ID"

    def __init__(self, release: str = "fake/catalog") -> None:
        super().__init__(release)

    def extra_fields(self, row: dict) -> dict:
        return {}


def cone() -> ConeQuery:
    return ConeQuery(ra_deg=180.0, dec_deg=0.0, radius_arcsec=30.0)


def test_all_rows_failing_to_parse_logs_a_schema_drift_warning(monkeypatch, caplog):
    # Columns don't match FakeConnector's expected ID/RAJ2000/DEJ2000 -- every
    # row hits the per-row except clause, exactly what a silent VizieR column
    # rename would look like.
    payload = _votable(["SomeOtherId", "SomeOtherRA", "SomeOtherDec"],
                       [["1", "180.0", "0.0"], ["2", "180.1", "0.1"]])
    monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))

    with caplog.at_level(logging.WARNING):
        sources = _FakeConnector().cone_search(cone())

    assert sources == []
    assert any("none parsed as a source" in record.message for record in caplog.records)


def test_a_genuinely_empty_cone_logs_nothing(monkeypatch, caplog):
    payload = _votable(["ID", "RAJ2000", "DEJ2000"], [])
    monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))

    with caplog.at_level(logging.WARNING):
        sources = _FakeConnector().cone_search(cone())

    assert sources == []
    assert not any("none parsed as a source" in record.message for record in caplog.records)


def test_a_successful_parse_logs_nothing(monkeypatch, caplog):
    payload = _votable(["ID", "RAJ2000", "DEJ2000"], [["1", "180.0", "0.0"]])
    monkeypatch.setattr(netclient, "get", lambda *a, **k: _FakeResponse(payload))

    with caplog.at_level(logging.WARNING):
        sources = _FakeConnector().cone_search(cone())

    assert len(sources) == 1
    assert not any("none parsed as a source" in record.message for record in caplog.records)
