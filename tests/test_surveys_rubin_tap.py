"""Rubin/LSST direct TAP connector (dormant, credential-gated, mocked only).

No network anywhere in this suite -- there is no real Rubin data-rights
token available to this project (see rubin_tap.py's module docstring). Every
test mocks `netclient.get` the same way the other TAP-backed tests do.
"""

from __future__ import annotations

from astra import credentials, netclient
from astra.surveys import rubin_tap
from astra.surveys.base import ConeQuery


class _FakeResponse:
    headers = {"Content-Type": "text/csv"}

    def __init__(self, csv_text: str) -> None:
        self.text = csv_text


VALID_CSV = (
    "objectId,coord_ra,coord_dec,mag_g,mag_r,mag_i\n"
    "12345,180.122,22.411,21.5,21.1,20.9\n"
)


class TestConnectorShape:
    def test_credential_required_and_opt_in(self):
        connector = rubin_tap.RubinTAPConnector()
        assert connector.credential_required is True
        assert connector.enabled_by_default is False

    def test_registered_under_rubin_tap(self):
        from astra import surveys

        assert "rubin_tap" in surveys.available(include_experimental=True)
        assert "rubin_tap" not in surveys.available(include_experimental=False)


class TestBuildConeAdql:
    def test_query_is_read_only_select_and_bounded(self):
        adql = rubin_tap.build_cone_adql(180.0, 10.0, 5.0, limit=50)
        assert adql.strip().upper().startswith("SELECT")
        assert "TOP 50" in adql
        assert "CIRCLE" in adql and "CONTAINS" in adql


class TestConeSearchWithoutCredentials:
    def test_raises_when_no_rubin_credential_is_stored(self, isolated_root, cone: ConeQuery):
        connector = rubin_tap.RubinTAPConnector()
        try:
            connector.cone_search(cone)
        except rubin_tap.RubinTAPError as exc:
            assert "credential" in str(exc).lower() or "token" in str(exc).lower()
        else:
            raise AssertionError("expected RubinTAPError without a stored credential")


class TestConeSearchWithCredentials:
    def _store_credential(self, isolated_root, monkeypatch):
        # DPAPI round-tripping is exercised by test_credentials.py; here we
        # only need the connector to see a stored credential, so monkeypatch
        # the generic load path directly rather than depend on the real
        # Windows DPAPI backend being available in the test environment.
        monkeypatch.setattr(credentials, "load_credentials",
                            lambda name, **kwargs: {"token": "fake-token"} if name == "rubin" else None)

    def test_cone_search_parses_valid_rows_and_sends_auth_header(
            self, isolated_root, monkeypatch, cone: ConeQuery):
        self._store_credential(isolated_root, monkeypatch)
        calls = []

        def fake_get(url, params, timeout, provider, headers=None):
            calls.append((provider, headers))
            return _FakeResponse(VALID_CSV)

        monkeypatch.setattr(netclient, "get", fake_get)
        connector = rubin_tap.RubinTAPConnector()

        sources = connector.cone_search(cone)

        assert len(sources) == 1
        assert sources[0].object_id == "12345"
        assert sources[0].ra_deg == 180.122
        assert sources[0].extra["mag_g"] == 21.5
        assert calls[0][0] == "rubin"
        assert calls[0][1] == {"Authorization": "Bearer fake-token"}

    def test_cone_search_skips_rows_missing_position(self, isolated_root, monkeypatch, cone: ConeQuery):
        self._store_credential(isolated_root, monkeypatch)
        bad_csv = "objectId,coord_ra,coord_dec\n12345,,22.411\n"
        monkeypatch.setattr(netclient, "get",
                            lambda *a, **k: _FakeResponse(bad_csv))
        connector = rubin_tap.RubinTAPConnector()

        assert connector.cone_search(cone) == []

    def test_fetch_light_curves_returns_empty(self, isolated_root, monkeypatch, source):
        self._store_credential(isolated_root, monkeypatch)
        connector = rubin_tap.RubinTAPConnector()
        assert connector.fetch_light_curves(source) == []
