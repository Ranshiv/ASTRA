"""Offline-first literature search and cache semantics."""

from datetime import datetime, timezone

from astra import literature


def test_search_uses_injected_fetchers_and_preserves_provider_provenance(tmp_path):
    calls = []

    def ads(query):
        calls.append(("ads", query.canonical()))
        return [{"provider": "ads", "title": "A study", "year": 2024,
                 "bibcode": "2024A&A...1A...1"}]

    first = literature.search(object_id="ZTF123", terms=["RR Lyrae"],
                              providers=["ads"], root=tmp_path,
                              fetchers={"ads": ads})
    second = literature.search(object_id="ZTF123", terms=["RR Lyrae"],
                               providers=["ads"], root=tmp_path,
                               fetchers={"ads": lambda _query: (_ for _ in ()).throw(AssertionError("cache miss"))})
    assert len(calls) == 1
    assert first["records"][0]["title"] == "A study"
    assert second["providers"]["ads"]["cache"]["state"] == "hit"
    assert first["provenance"][0]["kind"] == "literature"


def test_offline_miss_is_not_a_no_match(tmp_path):
    result = literature.search(object_id="unknown", providers=["ads"],
                               root=tmp_path, offline=True)
    assert result["providers"]["ads"]["state"] == "offline"
    assert not result["complete"]
    assert result["records"] == []


def test_query_provider_records_unavailable_provider_without_raising(tmp_path):
    result = literature.query_provider(
        "ads", literature.LiteratureQuery("ZTF123"), root=tmp_path,
        fetcher=lambda _query: (_ for _ in ()).throw(literature.LiteratureError("token missing")),
        now=datetime(2026, 8, 21, tzinfo=timezone.utc),
    )
    assert result["state"] == "unavailable"
    assert "token missing" in result["error"]
