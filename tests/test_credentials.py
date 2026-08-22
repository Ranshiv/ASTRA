"""Generic DPAPI credential storage, keyed by provider name.

TNS's own behaviour is covered by test_catalogs.py; these tests prove the
generalisation works for a second provider (e.g. a future Rubin/LSST TAP
connector) without disturbing TNS's exact file name, entropy, or contract.
"""
from __future__ import annotations

from astra import credentials
from astra.config import Paths


def _paths(tmp_path) -> Paths:
    paths = Paths(tmp_path)
    paths.ensure()
    return paths


def test_generic_round_trip_for_a_second_provider(tmp_path):
    paths = _paths(tmp_path)
    payload = {"token": "do-not-store-this-in-plain-text", "scope": "dp0"}

    saved = credentials.save_credentials("rubin", payload, paths=paths)
    raw = (paths.config / "rubin-credentials.dpapi.json").read_text(encoding="utf-8")
    loaded = credentials.load_credentials("rubin", paths=paths)

    assert saved["configured"] is True
    assert payload["token"] not in raw
    assert loaded == payload
    status = credentials.credential_status("rubin", paths=paths)
    assert status == {"configured": True, "usable": True, "backend": "windows_dpapi"}
    assert credentials.clear_credentials("rubin", paths=paths) is True
    assert credentials.load_credentials("rubin", paths=paths) is None


def test_different_provider_names_do_not_collide(tmp_path):
    paths = _paths(tmp_path)
    credentials.save_credentials("rubin", {"token": "rubin-secret"}, paths=paths)
    credentials.save_credentials("other", {"token": "other-secret"}, paths=paths)

    assert credentials.load_credentials("rubin", paths=paths) == {"token": "rubin-secret"}
    assert credentials.load_credentials("other", paths=paths) == {"token": "other-secret"}
    assert credentials.clear_credentials("rubin", paths=paths) is True
    assert credentials.load_credentials("other", paths=paths) == {"token": "other-secret"}


def test_missing_provider_credential_is_none_not_an_error(tmp_path):
    paths = _paths(tmp_path)
    assert credentials.load_credentials("never_saved", paths=paths) is None
    assert credentials.credential_status("never_saved", paths=paths) == {
        "configured": False, "backend": "windows_dpapi"}
    assert credentials.clear_credentials("never_saved", paths=paths) is False


def test_empty_payload_is_rejected(tmp_path):
    paths = _paths(tmp_path)
    try:
        credentials.save_credentials("rubin", {}, paths=paths)
    except credentials.CredentialError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError("expected CredentialError for an empty payload")


def test_invalid_name_is_rejected(tmp_path):
    paths = _paths(tmp_path)
    for bad_name in ("Rubin", "123rubin", "ru bin", "a" * 40):
        try:
            credentials.save_credentials(bad_name, {"token": "x"}, paths=paths)
        except credentials.CredentialError:
            continue
        raise AssertionError(f"expected CredentialError for name {bad_name!r}")


def test_tns_thin_wrappers_still_use_the_original_file_and_entropy(tmp_path):
    """The generalisation must not change TNS's own on-disk contract."""
    paths = _paths(tmp_path)
    credentials.save_tns_credentials("tns-secret-key", "123", "ASTRA test", paths)

    # save_tns_credentials must still write exactly tns-credentials.dpapi.json,
    # not a name derived differently by the generic path.
    assert (paths.config / "tns-credentials.dpapi.json").exists()
    assert credentials.credential_path(paths) == paths.config / "tns-credentials.dpapi.json"

    # And the generic loader, given the original fixed entropy, can still
    # decrypt what the TNS-specific saver wrote -- proving save_tns_credentials
    # is a thin wrapper over the same generic mechanism, not a fork of it.
    raw = credentials.load_credentials("tns", entropy=credentials._ENTROPY, paths=paths)
    assert raw["api_key"] == "tns-secret-key"
