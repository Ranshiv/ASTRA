"""corroborate.equivalence / .domain_transfer / .scaling RPC handlers
(Direction 3, "corroboration as a general multi-instrument anomaly
library"). No network -- every study is over synthetic data."""

from __future__ import annotations

from astra import rpc


def test_equivalence_handler_reports_full_agreement():
    response = rpc.dispatch({"id": 1, "method": "corroborate.equivalence", "params": {
        "n_trials": 10, "seed": 0,
    }})
    assert response["ok"] is True
    assert response["result"]["agreement_rate"] == 1.0


def test_domain_transfer_handler_reports_both_domains():
    response = rpc.dispatch({"id": 2, "method": "corroborate.domain_transfer", "params": {}})
    assert response["ok"] is True
    assert "astronomy" in response["result"]
    assert "gw" in response["result"]


def test_scaling_handler_accepts_custom_correlations():
    response = rpc.dispatch({"id": 3, "method": "corroborate.scaling", "params": {
        "correlations": [0.0, 1.0], "n_glitches": 100, "seed": 0,
    }})
    assert response["ok"] is True
    assert len(response["result"]["points"]) == 2


def test_scaling_handler_uses_a_default_sweep_without_params():
    response = rpc.dispatch({"id": 4, "method": "corroborate.scaling", "params": {}})
    assert response["ok"] is True
    assert len(response["result"]["points"]) > 1
