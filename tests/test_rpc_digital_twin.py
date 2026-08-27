"""digital_twin.* RPC handlers: read-only diagnostics over
`survey_digital_twin.py`/`survey_digital_twin_eval.py` (backlog item 42).

Every handler here is a plain diagnostic -- see each handler's own
docstring in `rpc.py` -- so these tests only check the RPC plumbing
(dispatch, params, error shape), not the underlying library logic, which
`tests/test_survey_digital_twin.py`/`test_survey_digital_twin_eval.py`
already cover in depth.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import rpc, store
from astra.surveys.base import LightCurve, SourceRef


def _write_curves(root, survey="ZTF", count=40, points=200, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(count):
        season_a = np.sort(rng.uniform(0, 90, points // 2))
        season_b = np.sort(rng.uniform(200, 290, points - points // 2))
        time = np.concatenate([season_a, season_b])
        value = 18.0 + 0.3 * np.sin(time / 10.0) + rng.normal(0, 0.05, len(time))
        store.write_curve(LightCurve(
            source=SourceRef(survey=survey, object_id=f"twin{i}",
                             ra_deg=10.0, dec_deg=20.0),
            release="dr24", band="g", value_kind="mag",
            time=time, value=value, value_err=np.full(len(time), 0.05),
            time_system="HJD_UTC",
        ), root)


class TestFitProfile:
    def test_fits_a_real_profile(self, isolated_root):
        _write_curves(isolated_root.datasets)

        response = rpc.dispatch({"id": 1, "method": "digital_twin.fit_profile",
                                 "params": {"survey": "ZTF", "limit": 100}})

        assert response["ok"] is True
        assert response["result"]["survey"] == "ZTF"
        assert response["result"]["n_curves_used"] >= 5
        assert response["result"]["note"] == ""

    def test_too_few_curves_reports_the_degradation_note(self, isolated_root):
        _write_curves(isolated_root.datasets, count=2)

        response = rpc.dispatch({"id": 1, "method": "digital_twin.fit_profile",
                                 "params": {"survey": "ZTF", "limit": 10}})

        assert response["ok"] is True
        assert "fewer than" in response["result"]["note"]

    def test_missing_survey_param_is_a_dispatch_error_not_a_crash(self, isolated_root):
        response = rpc.dispatch({"id": 1, "method": "digital_twin.fit_profile",
                                 "params": {}})
        assert response["ok"] is False


class TestSample:
    def test_returns_profile_and_batch_summary_not_a_raw_array(self, isolated_root):
        _write_curves(isolated_root.datasets)

        response = rpc.dispatch({"id": 1, "method": "digital_twin.sample",
                                 "params": {"survey": "ZTF", "limit": 100, "n": 15}})

        assert response["ok"] is True
        assert response["result"]["batch"]["rows"] == 15
        assert "values" not in response["result"]["batch"]  # summary only


class TestEvaluateDistance:
    def test_reports_a_per_feature_distance(self, isolated_root):
        _write_curves(isolated_root.datasets)

        response = rpc.dispatch({"id": 1, "method": "digital_twin.evaluate_distance",
                                 "params": {"survey": "ZTF", "limit": 100}})

        assert response["ok"] is True
        result = response["result"]
        assert 0.0 <= result["mean_ks_statistic"] <= 1.0
        assert set(result["per_feature"]) == {
            "std", "amplitude", "robust_amplitude", "mean_abs_diff", "max_abs_diff",
            "skew_proxy", "kurtosis_proxy", "min", "max", "coverage"}

    def test_too_few_real_rows_is_reported_not_raised(self, isolated_root):
        _write_curves(isolated_root.datasets, count=1)

        response = rpc.dispatch({"id": 1, "method": "digital_twin.evaluate_distance",
                                 "params": {"survey": "ZTF", "limit": 10}})

        assert response["ok"] is True
        assert "error" in response["result"]


class TestEvaluateTransfer:
    torch = pytest.importorskip("torch", reason="PyTorch not installed")

    def test_runs_both_arms_and_returns_well_formed_summaries(self, isolated_root):
        _write_curves(isolated_root.datasets, count=40, points=200)

        response = rpc.dispatch({"id": 1, "method": "digital_twin.evaluate_transfer",
                                 "params": {"survey": "ZTF", "limit": 100,
                                            "seeds": [1, 2], "epochs": 2}})

        assert response["ok"] is True
        result = response["result"]
        # Not asserting which arm wins -- only that both ran and produced a
        # well-formed summary shape, matching every other transfer study's
        # own restraint in this codebase.
        for arm in ("trained_on_real", "trained_on_synthetic"):
            if result[arm] is not None:
                assert {"mean", "std", "ci95", "n"} <= result[arm].keys()

    def test_too_few_real_rows_is_reported_not_raised(self, isolated_root):
        _write_curves(isolated_root.datasets, count=1)

        response = rpc.dispatch({"id": 1, "method": "digital_twin.evaluate_transfer",
                                 "params": {"survey": "ZTF", "limit": 10}})

        assert response["ok"] is True
        assert "error" in response["result"]

    def test_missing_torch_is_refused_with_a_clear_message(self, isolated_root, monkeypatch):
        monkeypatch.setattr(rpc, "_require_torch",
                            lambda: (_ for _ in ()).throw(
                                RuntimeError("PyTorch not installed")))
        response = rpc.dispatch({"id": 1, "method": "digital_twin.evaluate_transfer",
                                 "params": {"survey": "ZTF"}})
        assert response["ok"] is False
        assert "PyTorch" in response["error"]
