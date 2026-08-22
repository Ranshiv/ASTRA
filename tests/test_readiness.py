from __future__ import annotations

from astra import readiness
from astra.surveys.gaia import DR4_EXPECTED_RELEASE


def test_readiness_is_offline_and_does_not_expose_secret_values(monkeypatch):
    monkeypatch.setenv("ASTRA_SIGN_CERT", "secret-certificate-path")
    monkeypatch.setenv("ASTRA_TIMESTAMP_URL", "https://timestamp.invalid")
    report = readiness.status()

    assert report["gaia_epoch"]["enabled"] is False
    assert report["gaia_epoch"]["expected_release"] == "2026-12-02"
    assert report["release"]["authenticode_certificate"] is True
    assert "secret-certificate-path" not in str(report)
    assert any(item["name"] == "SDSS" for item in report["connectors"])


def test_gaia_epoch_expected_release_cannot_drift_from_the_connector():
    """readiness.py used to hardcode a second copy of this date; now it is
    imported from surveys/gaia.py, so the two can no longer silently disagree."""
    report = readiness.status()
    assert report["gaia_epoch"]["expected_release"] == DR4_EXPECTED_RELEASE


def test_gaia_epoch_code_ready_is_true_while_enabled_stays_false():
    """code_ready reflects that the chunked ingestion pipeline (surveys/
    gaia_epoch.py) exists in code; enabled is the external DR4-access gate
    and must not move just because the pipeline is ready to use it."""
    report = readiness.status()
    assert report["gaia_epoch"]["code_ready"] is True
    assert report["gaia_epoch"]["enabled"] is False
