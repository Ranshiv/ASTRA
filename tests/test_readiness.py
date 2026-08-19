from __future__ import annotations

from astra import readiness


def test_readiness_is_offline_and_does_not_expose_secret_values(monkeypatch):
    monkeypatch.setenv("ASTRA_SIGN_CERT", "secret-certificate-path")
    monkeypatch.setenv("ASTRA_TIMESTAMP_URL", "https://timestamp.invalid")
    report = readiness.status()

    assert report["gaia_epoch"]["enabled"] is False
    assert report["gaia_epoch"]["expected_release"] == "2026-12-02"
    assert report["release"]["authenticode_certificate"] is True
    assert "secret-certificate-path" not in str(report)
    assert any(item["name"] == "SDSS" for item in report["connectors"])
