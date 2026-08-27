"""Lead-time study metric arithmetic on a small synthetic injection set.

celerite2-gated, same as `test_agn_changepoint.py`/`test_multiband_hier.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

celerite2 = pytest.importorskip("celerite2", reason="celerite2 not installed (opt-in 'research' extra)")

from astra import agn_changepoint_eval as agne  # noqa: E402


def test_evaluate_lead_time_detects_a_clear_injected_flare():
    result = agne.evaluate_lead_time(
        drw_sigma=0.2, drw_tau=25.0, flare_amplitude=1.5, flare_rise_sigma=10.0,
        flare_t_decay_ref=15.0, cutoff_grid_days=[100.0, 130.0, 150.0, 170.0, 200.0],
        n_trials=3, span_days=200.0, cadence_days=1.0, noise_sigma=0.02,
        target_fpr=0.05, n_calibration_realizations=15, seed=3)
    assert result.n_detected == result.n_trials
    assert result.lead_time_days is not None


def test_evaluate_lead_time_reports_zero_detections_for_a_negligible_flare():
    result = agne.evaluate_lead_time(
        drw_sigma=0.5, drw_tau=25.0, flare_amplitude=1e-4, flare_rise_sigma=10.0,
        flare_t_decay_ref=15.0, cutoff_grid_days=[100.0, 200.0],
        n_trials=2, span_days=200.0, cadence_days=1.0, noise_sigma=0.02,
        target_fpr=0.01, n_calibration_realizations=10, seed=9)
    assert result.n_detected == 0
    assert result.lead_time_days is None


def test_evaluate_lead_time_rejects_empty_cutoff_grid():
    with pytest.raises(agne.AGNChangepointError):
        agne.evaluate_lead_time(
            drw_sigma=0.2, drw_tau=25.0, flare_amplitude=1.5, flare_rise_sigma=10.0,
            flare_t_decay_ref=15.0, cutoff_grid_days=[])


def test_evaluate_lead_time_rejects_bad_trial_count():
    with pytest.raises(agne.AGNChangepointError):
        agne.evaluate_lead_time(
            drw_sigma=0.2, drw_tau=25.0, flare_amplitude=1.5, flare_rise_sigma=10.0,
            flare_t_decay_ref=15.0, cutoff_grid_days=[100.0], n_trials=0)


def test_lead_time_result_to_dict_shape():
    result = agne.LeadTimeResult(cutoff_days_since_first=[10.0, 20.0], n_trials=2,
                                 n_detected=1, lead_time_days={"mean": 5.0, "std": 0.0, "ci95": [5.0, 5.0]})
    payload = result.to_dict()
    assert payload["detection_rate"] == 0.5
    assert payload["n_detected"] == 1


def test_agn_changepoint_eval_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "agn_changepoint" not in rpc_source
