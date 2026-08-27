"""credentials.rubin.*: RPC surface for the dormant Rubin/LSST TAP connector's
data-rights token (see surveys/rubin_tap.py). Exercises the real Windows
DPAPI round trip via rpc.dispatch, mirroring test_credentials.py's generic
provider coverage.
"""

from __future__ import annotations

from astra import rpc


class TestRubinCredentialsRpc:
    def test_configure_then_status_then_clear(self, isolated_root):
        configure = rpc.dispatch({"id": 1, "method": "credentials.rubin.configure",
                                  "params": {"token": "secret-token"}})
        assert configure["ok"] is True
        assert configure["result"]["configured"] is True
        # The token itself must never come back through the RPC layer.
        assert "secret-token" not in str(configure["result"])

        status = rpc.dispatch({"id": 2, "method": "credentials.rubin.status", "params": {}})
        assert status["ok"] is True
        assert status["result"]["configured"] is True
        assert status["result"]["usable"] is True

        clear = rpc.dispatch({"id": 3, "method": "credentials.rubin.clear", "params": {}})
        assert clear["ok"] is True
        assert clear["result"]["cleared"] is True

        status_after = rpc.dispatch({"id": 4, "method": "credentials.rubin.status", "params": {}})
        assert status_after["result"]["configured"] is False

    def test_status_without_a_stored_credential_is_not_configured(self, isolated_root):
        response = rpc.dispatch({"id": 1, "method": "credentials.rubin.status", "params": {}})
        assert response["ok"] is True
        assert response["result"]["configured"] is False
