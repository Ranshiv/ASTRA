"""Cached catalogue evidence must remain honest when the network is absent."""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone

from astra import catalogs, credentials, metadata, scoring
from astra.config import Paths


UTC = timezone.utc


def query() -> catalogs.CatalogQuery:
    return catalogs.CatalogQuery("ZTF-42", 180.1234567, 22.5, 2.0)


def test_catalog_cache_hit_avoids_a_second_network_call(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogs, "_sleep_for_rate_limit", lambda _provider: None)
    calls: list[str] = []

    def fetch(_query):
        calls.append("fetch")
        return [{"main_id": "RR Lyr", "is_variable": True}]

    now = datetime(2026, 8, 17, tzinfo=UTC)
    first = catalogs.query_provider("simbad", query(), root=tmp_path,
                                    fetcher=fetch, now=now)
    second = catalogs.query_provider("simbad", query(), root=tmp_path,
                                     fetcher=fetch, now=now + timedelta(days=1))

    assert first["state"] == "match"
    assert second["cache"]["state"] == "hit"
    assert calls == ["fetch"]
    assert metadata.catalog_cache_summary(tmp_path)["total"] == 1


def test_expired_cache_is_refreshed(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogs, "_sleep_for_rate_limit", lambda _provider: None)
    calls = 0

    def fetch(_query):
        nonlocal calls
        calls += 1
        return []

    now = datetime(2026, 8, 17, tzinfo=UTC)
    catalogs.query_provider("vsx", query(), root=tmp_path, fetcher=fetch, now=now)
    result = catalogs.query_provider("vsx", query(), root=tmp_path, fetcher=fetch,
                                     now=now + timedelta(days=catalogs.CACHE_TTL_DAYS + 1))

    assert result["state"] == "no_match"
    assert result["cache"]["state"] == "refreshed"
    assert calls == 2


def test_offline_keeps_no_match_distinct_from_a_missing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogs, "_sleep_for_rate_limit", lambda _provider: None)
    now = datetime(2026, 8, 17, tzinfo=UTC)
    catalogs.query_provider("simbad", query(), root=tmp_path, fetcher=lambda _: [], now=now)

    stale = catalogs.query_provider(
        "simbad", query(), root=tmp_path, offline=True,
        now=now + timedelta(days=catalogs.CACHE_TTL_DAYS + 1))
    missing = catalogs.query_provider(
        "vsx", query(), root=tmp_path, offline=True, now=now)

    assert stale["state"] == "no_match"
    assert stale["cache"] == {"state": "stale_offline", "stale": True}
    assert missing["state"] == "offline"


def test_rate_limit_is_recorded_without_becoming_a_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogs, "_sleep_for_rate_limit", lambda _provider: None)

    def limited(_query):
        raise catalogs.CatalogRateLimitError("slow down")

    now = datetime(2026, 8, 17, tzinfo=UTC)
    result = catalogs.query_provider("vsx", query(), root=tmp_path,
                                     fetcher=limited, now=now)

    assert result["state"] == "rate_limited"
    assert result["matches"] == []
    assert "slow down" in result["error"]
    assert scoring.catalog_novelty(catalog_evidence={
        "summary": {"states": {"simbad": "no_match", "vsx": "rate_limited"},
                    "known_variable": False, "known_object": False},
    })[0] is None


def test_tns_without_a_protected_secret_is_not_queried(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogs, "_sleep_for_rate_limit", lambda _provider: None)
    monkeypatch.setattr(credentials, "load_tns_credentials", lambda: None)
    seen: list[str] = []

    def public(_query):
        seen.append("public")
        return []

    evidence = catalogs.enrich_position(
        "ZTF-42", 180.123, 22.5, root=tmp_path,
        fetchers={"simbad": public, "vsx": public,
                  "tns": lambda _: (_ for _ in ()).throw(AssertionError("TNS called"))},
    )

    assert seen == ["public", "public"]
    assert evidence["providers"]["tns"]["state"] == "not_configured"
    assert evidence["summary"]["public_complete"] is True
    assert scoring.catalog_novelty(catalog_evidence=evidence)[0] == 0.9


def test_windows_dpapi_secret_is_not_written_in_plaintext(tmp_path):
    paths = Paths(tmp_path)
    paths.ensure()
    api_key = "do-not-store-this-in-plain-text"

    saved = credentials.save_tns_credentials(api_key, "123", "ASTRA test", paths)
    raw = credentials.credential_path(paths).read_text(encoding="utf-8")
    loaded = credentials.load_tns_credentials(paths)

    assert saved["configured"] is True
    assert api_key not in raw
    assert loaded is not None and loaded["api_key"] == api_key
    status = credentials.tns_credential_status(paths)
    assert status["usable"] is True
    assert api_key not in str(status)
    assert credentials.clear_tns_credentials(paths) is True
    assert credentials.load_tns_credentials(paths) is None


def test_simbad_empty_cone_warning_is_quiet_but_remains_no_match(monkeypatch):
    """SIMBAD's expected empty-result warning must not pollute engine logs."""
    from astroquery.exceptions import NoResultsWarning

    class FakeSimbad:
        ROW_LIMIT = 20

        def add_votable_fields(self, *_fields):
            return None

        def query_region(self, *_args, **_kwargs):
            warnings.warn("empty cone", NoResultsWarning)
            return None

    # `_fetch_simbad` imports Simbad locally, so provide a tiny module shim
    # only for this test and restore it automatically through monkeypatch.
    import sys
    import types

    simbad_module = types.ModuleType("astroquery.simbad")
    simbad_module.Simbad = FakeSimbad
    monkeypatch.setitem(sys.modules, "astroquery.simbad", simbad_module)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = catalogs._fetch_simbad(query())

    assert result == []
    assert not [warning for warning in seen if warning.category is NoResultsWarning]


def _tns_secret() -> dict:
    return {"api_key": "secret", "bot_id": "123", "bot_name": "ASTRA test"}


def test_fetch_tns_uses_the_shared_throttled_session(monkeypatch):
    """`_fetch_tns` must go through `netclient.post` (shared session, retry
    policy, throttle bucket), not a bare `requests.post` -- see netclient.py's
    module docstring for why a raw call silently drops data under load."""
    monkeypatch.setattr(credentials, "load_tns_credentials", lambda: _tns_secret())
    seen: dict = {}

    class FakeResponse:
        def json(self):
            return {"data": {"reply": [
                {"objname": "2026abc", "object_type": "SN Ia", "discoverydate": "2026-08-01"},
            ]}}

    def fake_post(url, data, timeout, provider="irsa", headers=None):
        seen["url"] = url
        seen["provider"] = provider
        seen["data"] = data
        seen["headers"] = headers
        return FakeResponse()

    from astra import netclient
    monkeypatch.setattr(netclient, "post", fake_post)

    result = catalogs._fetch_tns(query())

    assert seen["provider"] == "tns"
    assert seen["url"] == "https://www.wis-tns.org/api/get/search"
    assert seen["data"]["api_key"] == "secret"
    assert "tns_marker" in seen["headers"]["User-Agent"]
    assert result == [{"name": "2026abc", "object_type": "SN Ia", "discovery_date": "2026-08-01"}]


def test_fetch_tns_rate_limit_response_raises_catalog_rate_limit_error(monkeypatch):
    monkeypatch.setattr(credentials, "load_tns_credentials", lambda: _tns_secret())

    import requests

    def fake_post(url, data, timeout, provider="irsa", headers=None):
        response = requests.Response()
        response.status_code = 429
        raise requests.HTTPError("rate limited", response=response)

    from astra import netclient
    monkeypatch.setattr(netclient, "post", fake_post)

    try:
        catalogs._fetch_tns(query())
        assert False, "expected CatalogRateLimitError"
    except catalogs.CatalogRateLimitError:
        pass


def test_fetch_tns_network_failure_raises_catalog_error(monkeypatch):
    monkeypatch.setattr(credentials, "load_tns_credentials", lambda: _tns_secret())

    import requests

    def fake_post(url, data, timeout, provider="irsa", headers=None):
        raise requests.ConnectionError("no route to host")

    from astra import netclient
    monkeypatch.setattr(netclient, "post", fake_post)

    try:
        catalogs._fetch_tns(query())
        assert False, "expected CatalogError"
    except catalogs.CatalogError:
        pass
