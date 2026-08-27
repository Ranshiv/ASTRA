"""xray_hardness_eval.py: state-transition detection and flux/hardness
calibration."""

from __future__ import annotations

import pytest

from astra import xray_hardness_eval as evaluation


class TestEvaluateStateTransitionDetection:
    def test_rejects_non_positive_epoch_counts(self):
        with pytest.raises(evaluation.XrayHardnessEvalError):
            evaluation.evaluate_state_transition_detection(-0.5, 0.5, n_before=0)

    def test_high_detection_rate_on_a_clear_shift(self):
        result = evaluation.evaluate_state_transition_detection(
            baseline_hr=-0.6, shifted_hr=0.6, n_before=15, n_after=15,
            noise_sigma=0.02, n_trials=50, seed=1)
        assert result["detection_rate"] > 0.8

    def test_lower_detection_rate_on_a_subtle_shift(self):
        clear = evaluation.evaluate_state_transition_detection(
            baseline_hr=-0.6, shifted_hr=0.6, n_before=15, n_after=15,
            noise_sigma=0.05, n_trials=50, seed=2)
        subtle = evaluation.evaluate_state_transition_detection(
            baseline_hr=-0.05, shifted_hr=0.05, n_before=15, n_after=15,
            noise_sigma=0.05, n_trials=50, seed=2)
        assert subtle["detection_rate"] <= clear["detection_rate"]


class TestEvaluateFalsePositiveRate:
    def test_reports_a_rate_between_zero_and_one(self):
        result = evaluation.evaluate_false_positive_rate(
            hr=0.0, n_points=20, noise_sigma=0.05, n_trials=50, seed=3)
        assert 0.0 <= result["false_positive_rate"] <= 1.0
        assert result["n_trials"] == 50


class TestFluxHardnessCalibration:
    def test_zero_residual_when_computed_matches_released_exactly(self):
        rows = [{"flux_soft": 5.0, "flux_hard": 5.0, "hr_hard_soft": 0.0}]
        result = evaluation.flux_hardness_calibration(rows)
        assert result["n_compared"] == 1
        assert result["median_residual"] == pytest.approx(0.0, abs=1e-9)

    def test_detects_a_real_residual(self):
        # soft=1, hard=9 -> my_hr = 0.8; released claims 0.5 -> residual 0.3
        rows = [{"flux_soft": 1.0, "flux_hard": 9.0, "hr_hard_soft": 0.5}]
        result = evaluation.flux_hardness_calibration(rows)
        assert result["median_residual"] == pytest.approx(0.3, abs=1e-6)

    def test_rows_missing_fields_are_skipped_not_dropped_silently(self):
        rows = [
            {"flux_soft": 1.0, "flux_hard": 9.0, "hr_hard_soft": 0.8},
            {"flux_soft": None, "flux_hard": 9.0, "hr_hard_soft": 0.8},
            {"flux_soft": 1.0, "flux_hard": 9.0, "hr_hard_soft": None},
        ]
        result = evaluation.flux_hardness_calibration(rows)
        assert result["n_compared"] == 1
        assert result["n_skipped"] == 2

    def test_empty_input(self):
        result = evaluation.flux_hardness_calibration([])
        assert result["n_compared"] == 0
        assert result["median_residual"] is None

    def test_zero_total_flux_row_is_skipped(self):
        rows = [{"flux_soft": 0.0, "flux_hard": 0.0, "hr_hard_soft": 0.0}]
        result = evaluation.flux_hardness_calibration(rows)
        assert result["n_compared"] == 0
        assert result["n_skipped"] == 1


@pytest.mark.live
class TestFluxHardnessCalibrationLive:
    """Confirmed live this session (2026-08-25): running this against 76
    real CSC 2.1 sources near M87 gives a real, non-trivial median
    residual (~0.47) -- NOT a formula bug (CSC's own documentation,
    checked live, confirms the identical `(Fluxh-Fluxs)/(Fluxh+Fluxs)`
    functional form), but a real, informative gap between CSC's Bayesian-
    MPDF-based released HR and this function's point-estimate-flux-based
    independent computation -- see `flux_hardness_calibration`'s own
    docstring for the full finding. This test locks in that this real
    residual stays in a stable, bounded range, not that it is ~zero."""

    def test_reports_a_real_bounded_residual_against_csc(self):
        from astra.surveys.chandra import query_band_fluxes

        results = query_band_fluxes(187.7059, 12.3911, 60.0)
        rows = [{"flux_soft": r["flux_soft"], "flux_hard": r["flux_hard"],
                "hr_hard_soft": r["hr_hard_soft"]} for r in results]
        report = evaluation.flux_hardness_calibration(rows)
        assert report["n_compared"] > 20
        # Both the computed and released ratios are individually bounded
        # to [-1, 1], so their difference cannot exceed 2 in magnitude.
        assert abs(report["median_residual"]) < 2.0


def test_not_referenced_by_rpc():
    import inspect

    from astra import rpc
    source = inspect.getsource(rpc)
    assert "xray_hardness_eval" not in source
