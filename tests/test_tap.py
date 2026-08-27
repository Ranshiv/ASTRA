"""Bounded TAP query and cache behavior."""

import pytest

from astra import tap


class _Response:
    headers = {"Content-Type": "text/csv"}
    text = "ra,dec,name\n1.5,-2.0,source\n"


def _votable(field_sets: list[list[str]], row_sets: list[list[list[str]]]) -> str:
    """Build a VOTable with one `<TABLE>` per (fields, rows) pair --
    the real multi-table shape that exposed `parse_votable`'s bug."""
    tables_xml = ""
    for fields, rows in zip(field_sets, row_sets):
        field_xml = "".join(f'<FIELD name="{name}"/>' for name in fields)
        row_xml = "".join(
            "<TR>" + "".join(f"<TD>{value}</TD>" for value in row) + "</TR>" for row in rows)
        tables_xml += (f"<TABLE>{field_xml}<DATA><TABLEDATA>{row_xml}"
                       "</TABLEDATA></DATA></TABLE>")
    return f'<?xml version="1.0"?><VOTABLE><RESOURCE>{tables_xml}</RESOURCE></VOTABLE>'


class TestParseVotable:
    def test_single_table_still_works(self):
        payload = _votable([["ra", "dec"]], [[["1.5", "-2.0"]]])
        rows = tap.parse_votable(payload)
        assert rows == [{"ra": 1.5, "dec": -2.0}]

    def test_real_multi_table_document_does_not_desync_fields(self):
        # Real bug found this session: a second table's FIELD set used to
        # get appended to a single global field list, and its TR values
        # zipped against that wrong list -- every field came back None.
        payload = _votable(
            [["KiDSID", "RAJ2000", "DEJ2000"], ["Nx", "Ny"]],
            [[["J1200-0000", "180.0", "0.0"]], [["601", "601"]]],
        )
        rows = tap.parse_votable(payload)
        assert rows[0] == {"KiDSID": "J1200-0000", "RAJ2000": 180.0, "DEJ2000": 0.0}
        assert rows[1] == {"Nx": 601, "Ny": 601}
        assert None not in rows[0].values()
        assert None not in rows[1].values()

    def test_six_tables_matches_the_real_shape_that_exposed_the_bug(self):
        # Mirrors the real live document this session found (6 <TABLE>
        # blocks, 13 total FIELDs, 1 <RESOURCE>) closely enough to be a
        # real regression test, not just a 2-table sanity check.
        field_sets = [["a"], ["b", "c"], ["d"], ["e", "f", "g"], ["h"], ["i", "j"]]
        row_sets = [[["1"]], [["2", "3"]], [["4"]], [["5", "6", "7"]], [["8"]], [["9", "10"]]]
        payload = _votable(field_sets, row_sets)
        rows = tap.parse_votable(payload)
        assert len(rows) == 6
        assert rows[3] == {"e": 5.0, "f": 6.0, "g": 7.0}
        assert all(None not in row.values() for row in rows)

    def test_respects_limit_across_multiple_tables(self):
        payload = _votable([["a"], ["b"], ["c"]], [[["1"]], [["2"]], [["3"]]])
        rows = tap.parse_votable(payload, limit=2)
        assert len(rows) == 2

    def test_raises_on_unparseable_xml(self):
        with pytest.raises(tap.TapError):
            tap.parse_votable("not xml at all <<<")


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

    def fake_get(url, params, timeout, provider, headers=None):
        calls.append((url, params, provider, headers))
        return _Response()

    monkeypatch.setattr(tap.netclient, "get", fake_get)
    first = tap.query("https://example.invalid/tap/sync", "SELECT ra, dec, name FROM foo",
                      release="demo", root=tmp_path, max_rows=5)
    second = tap.query("https://example.invalid/tap/sync", "SELECT ra, dec, name FROM foo",
                       release="demo", root=tmp_path, max_rows=5)
    assert len(calls) == 1
    assert calls[0][2] == "datalab"
    assert first["rows"][0]["ra"] == 1.5
    assert second["cache"]["state"] == "hit"


def test_tap_offline_miss_is_distinct(tmp_path):
    result = tap.query("https://example.invalid/tap/sync", "SELECT ra FROM foo",
                       root=tmp_path, offline=True)
    assert result["state"] == "offline"
    assert result["rows"] == []


def test_tap_query_generalizes_provider_and_auth_header(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, params, timeout, provider, headers=None):
        calls.append((provider, headers))
        return _Response()

    monkeypatch.setattr(tap.netclient, "get", fake_get)
    tap.query("https://data.lsst.cloud/api/tap", "SELECT ra FROM foo", root=tmp_path,
             provider="rubin", auth_header={"Authorization": "Bearer token"})

    assert calls == [("rubin", {"Authorization": "Bearer token"})]


def test_tap_query_different_providers_do_not_share_cache(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, params, timeout, provider, headers=None):
        calls.append(provider)
        return _Response()

    monkeypatch.setattr(tap.netclient, "get", fake_get)
    tap.query("https://example.invalid/tap/sync", "SELECT ra FROM foo",
             root=tmp_path, provider="datalab")
    tap.query("https://example.invalid/tap/sync", "SELECT ra FROM foo",
             root=tmp_path, provider="rubin")

    assert calls == ["datalab", "rubin"]


def test_cell_preserves_large_integer_precision():
    """A real bug, found and fixed via a live check against DESI's
    `targetid` column while building `async_query`: routing every
    numeric-looking cell through `float()` first silently rounds any
    integer beyond float64's 2**53 exact range -- confirmed live, a query
    for a specific real targetid came back with a DIFFERENT number in its
    last two digits purely from this round trip."""
    large_targetid = "39628379988689587"  # beyond 2**53 ~= 9.007e15
    assert tap._cell(large_targetid) == 39628379988689587
    assert tap._cell("42") == 42
    assert tap._cell("3.5") == 3.5
    assert tap._cell("1e5") == 100000  # scientific-notation "integer" path
    assert tap._cell("") is None
    assert tap._cell("null") is None


class _JobResponse:
    """A fake `requests.Response` for one step of a UWS async job flow."""

    def __init__(self, url: str, text: str, headers: dict | None = None) -> None:
        self.url = url
        self.text = text
        self.headers = headers or {"Content-Type": "text/xml"}


JOB_URL = "https://example.invalid/tap/async/job123"


def _job_xml(phase: str, *, result_href: str | None = None,
            error_message: str | None = None) -> str:
    result_xml = (f'<uws:result id="result" xlink:href="{result_href}"/>'
                 if result_href else "")
    error_xml = (
        f'<uws:errorSummary type="fatal"><uws:message>{error_message}</uws:message>'
        "</uws:errorSummary>" if error_message else "")
    return (
        '<?xml version="1.0"?>'
        '<uws:job xmlns:uws="http://www.ivoa.net/xml/UWS/v1.0" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">'
        f"<uws:phase>{phase}</uws:phase>"
        f"<uws:results>{result_xml}</uws:results>{error_xml}"
        "</uws:job>"
    )


def test_async_query_completes_a_real_job_flow(monkeypatch, tmp_path):
    posts: list[tuple[str, dict]] = []
    gets: list[str] = []

    def fake_post(url, data, timeout, provider, headers=None):
        posts.append((url, data))
        if url.endswith("/async"):
            return _JobResponse(JOB_URL, _job_xml("PENDING"))
        if url.endswith("/phase"):
            return _JobResponse(url, "")
        raise AssertionError(f"unexpected POST to {url}")

    def fake_get(url, params, timeout, provider, headers=None):
        gets.append(url)
        if url == f"{JOB_URL}/phase":
            return _JobResponse(url, "COMPLETED")
        if url == JOB_URL:
            return _JobResponse(url, _job_xml(
                "COMPLETED", result_href="https://example.invalid/resultStore/r.csv"))
        if url == "https://example.invalid/resultStore/r.csv":
            return _Response()
        raise AssertionError(f"unexpected GET to {url}")

    monkeypatch.setattr(tap.netclient, "post", fake_post)
    monkeypatch.setattr(tap.netclient, "get", fake_get)

    result = tap.async_query("https://example.invalid/tap/sync", "SELECT ra, dec, name FROM foo",
                             root=tmp_path, poll_interval=0.0)

    assert result["state"] == "match"
    assert result["rows"][0]["ra"] == 1.5
    # A PENDING job must be explicitly started with PHASE=RUN.
    assert (f"{JOB_URL}/phase", {"PHASE": "RUN"}) in posts


def test_async_query_polls_through_executing_before_completed(monkeypatch, tmp_path):
    phases = iter(["EXECUTING", "EXECUTING", "COMPLETED"])

    def fake_post(url, data, timeout, provider, headers=None):
        return _JobResponse(JOB_URL, _job_xml("EXECUTING"))

    def fake_get(url, params, timeout, provider, headers=None):
        if url == f"{JOB_URL}/phase":
            return _JobResponse(url, next(phases))
        if url == "https://example.invalid/resultStore/r.csv":
            return _Response()
        return _JobResponse(url, _job_xml(
            "COMPLETED", result_href="https://example.invalid/resultStore/r.csv"))

    monkeypatch.setattr(tap.netclient, "post", fake_post)
    monkeypatch.setattr(tap.netclient, "get", fake_get)

    result = tap.async_query("https://example.invalid/tap/sync", "SELECT ra FROM foo",
                             root=tmp_path, poll_interval=0.0)
    assert result["state"] == "match"


def test_async_query_surfaces_the_real_error_message(monkeypatch, tmp_path):
    def fake_post(url, data, timeout, provider, headers=None):
        return _JobResponse(JOB_URL, _job_xml("EXECUTING"))

    def fake_get(url, params, timeout, provider, headers=None):
        if url == f"{JOB_URL}/phase":
            return _JobResponse(url, "ERROR")
        return _JobResponse(url, _job_xml(
            "ERROR", error_message="PSQLException: function point(...) does not exist"))

    monkeypatch.setattr(tap.netclient, "post", fake_post)
    monkeypatch.setattr(tap.netclient, "get", fake_get)

    result = tap.async_query("https://example.invalid/tap/sync", "SELECT ra FROM foo",
                             root=tmp_path, poll_interval=0.0)
    assert result["state"] == "unavailable"
    assert "PSQLException" in result["error"]


def test_async_query_raises_if_never_terminal_within_max_wait(monkeypatch, tmp_path):
    def fake_post(url, data, timeout, provider, headers=None):
        return _JobResponse(JOB_URL, _job_xml("EXECUTING"))

    def fake_get(url, params, timeout, provider, headers=None):
        return _JobResponse(url, "EXECUTING")

    monkeypatch.setattr(tap.netclient, "post", fake_post)
    monkeypatch.setattr(tap.netclient, "get", fake_get)

    result = tap.async_query("https://example.invalid/tap/sync", "SELECT ra FROM foo",
                             root=tmp_path, poll_interval=0.0, max_wait_seconds=0.0)
    assert result["state"] == "unavailable"
    assert "did not reach a terminal phase" in result["error"]


def test_async_query_rejects_a_non_sync_service_url(tmp_path):
    result = tap.async_query("https://example.invalid/tap/notsync", "SELECT ra FROM foo",
                             root=tmp_path)
    assert result["state"] == "unavailable"
    assert "expected it to end with" in result["error"]
