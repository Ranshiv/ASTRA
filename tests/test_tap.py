"""Bounded TAP query and cache behavior."""

import pytest

from astra import tap


class _Response:
    headers = {"Content-Type": "text/csv"}
    text = "ra,dec,name\n1.5,-2.0,source\n"


def test_bound_adql_injects_top_and_rejects_mutation():
    query, limit = tap.bound_adql("SELECT ra FROM foo", 4)
    assert query.startswith("SELECT TOP 4")
    assert limit == 4
    with pytest.raises(ValueError):
        tap.bound_adql("DROP TABLE foo")
    with pytest.raises(ValueError):
        tap.bound_adql("SELECT ra FROM foo; SELECT dec FROM foo")


def test_tap_query_parses_and_caches_rows(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, params, timeout, provider):
        calls.append((url, params, provider))
        return _Response()

    monkeypatch.setattr(tap.netclient, "get", fake_get)
    first = tap.query("https://example.invalid/tap/sync", "SELECT ra, dec, name FROM foo",
                      release="demo", root=tmp_path, max_rows=5)
    second = tap.query("https://example.invalid/tap/sync", "SELECT ra, dec, name FROM foo",
                       release="demo", root=tmp_path, max_rows=5)
    assert len(calls) == 1
    assert first["rows"][0]["ra"] == 1.5
    assert second["cache"]["state"] == "hit"


def test_tap_offline_miss_is_distinct(tmp_path):
    result = tap.query("https://example.invalid/tap/sync", "SELECT ra FROM foo",
                       root=tmp_path, offline=True)
    assert result["state"] == "offline"
    assert result["rows"] == []
