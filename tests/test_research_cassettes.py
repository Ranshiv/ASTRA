"""Cassette identity, save/load round-trip, checksum tamper rejection, and
that `netclient.get` replays a cassette instead of touching the network."""

from __future__ import annotations

import pytest

from astra.research import cassettes


def test_mode_defaults_to_off_without_env(monkeypatch):
    monkeypatch.delenv("ASTRA_CASSETTE_MODE", raising=False)
    assert cassettes.mode() == "off"


def test_mode_reads_env(monkeypatch):
    monkeypatch.setenv("ASTRA_CASSETTE_MODE", "record")
    assert cassettes.mode() == "record"


def test_identity_is_stable_and_order_independent():
    a = cassettes.identity("ztf", "GET", "https://example.com", {"b": 1, "a": 2})
    b = cassettes.identity("ztf", "GET", "https://example.com", {"a": 2, "b": 1})
    assert a == b


def test_identity_changes_with_params():
    a = cassettes.identity("ztf", "GET", "https://example.com", {"a": 1})
    b = cassettes.identity("ztf", "GET", "https://example.com", {"a": 2})
    assert a != b


def test_save_and_load_round_trip(tmp_path):
    key = cassettes.identity("ztf", "GET", "https://example.com", {})
    recorded = cassettes.RecordedResponse(status_code=200, headers={"X": "1"},
                                          content=b'{"ok": true}',
                                          url="https://example.com")
    cassettes.save(key, recorded, root=tmp_path)
    loaded = cassettes.load(key, root=tmp_path)
    assert loaded.content == b'{"ok": true}'
    assert loaded.status_code == 200


def test_load_missing_raises_cassette_miss(tmp_path):
    with pytest.raises(cassettes.CassetteMissError):
        cassettes.load("nonexistent", root=tmp_path)


def test_load_rejects_tampered_checksum(tmp_path):
    import json

    key = cassettes.identity("ztf", "GET", "https://example.com", {})
    recorded = cassettes.RecordedResponse(status_code=200, headers={},
                                          content=b"original",
                                          url="https://example.com")
    path = cassettes.save(key, recorded, root=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["content_hex"] = b"tampered".hex()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(cassettes.CassetteChecksumError):
        cassettes.load(key, root=tmp_path)


def test_save_redacts_credential_like_headers(tmp_path):
    key = cassettes.identity("ztf", "GET", "https://example.com", {})
    recorded = cassettes.RecordedResponse(
        status_code=200, headers={"Authorization": "secret-token", "Content-Type": "json"},
        content=b"x", url="https://example.com")
    cassettes.save(key, recorded, root=tmp_path)
    loaded = cassettes.load(key, root=tmp_path)
    assert loaded.headers["Authorization"] == "<redacted>"
    assert loaded.headers["Content-Type"] == "json"


def test_netclient_get_replays_cassette(tmp_path, monkeypatch):
    from astra import netclient

    monkeypatch.setenv("ASTRA_CASSETTE_MODE", "replay")
    monkeypatch.setattr(cassettes, "_cassette_dir", lambda root=None: tmp_path)

    key = cassettes.identity("irsa", "GET", "https://example.com/query", {"a": "1"})
    cassettes.save(key, cassettes.RecordedResponse(
        status_code=200, headers={}, content=b'{"result": 1}',
        url="https://example.com/query"), root=tmp_path)

    response = netclient.get("https://example.com/query", {"a": "1"}, timeout=5,
                             provider="irsa")
    assert response.json() == {"result": 1}


def test_netclient_get_replay_miss_raises(tmp_path, monkeypatch):
    from astra import netclient

    monkeypatch.setenv("ASTRA_CASSETTE_MODE", "replay")
    monkeypatch.setattr(cassettes, "_cassette_dir", lambda root=None: tmp_path)

    with pytest.raises(cassettes.CassetteMissError):
        netclient.get("https://example.com/nope", {}, timeout=5, provider="irsa")
