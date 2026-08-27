"""microlensing_eval.py: parameter bias, event efficiency, and posterior
coverage (backlog item 15)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import microlensing_eval as me


class TestParameterBias:
    def test_zero_bias_when_fitted_equals_reference(self):
        fitted = [{"t0": 100.0, "tE": 20.0, "u0": 0.2}]
        reference = [{"t0": 100.0, "tE": 20.0, "u0": 0.2}]
        result = me.parameter_bias(fitted, reference)
        assert result["parameters"]["tE"]["median_absolute_bias"] == 0.0
        assert result["parameters"]["tE"]["median_fractional_bias"] == 0.0

    def test_reports_correct_sign_of_bias(self):
        fitted = [{"tE": 22.0}]
        reference = [{"tE": 20.0}]
        result = me.parameter_bias(fitted, reference, names=("tE",))
        assert result["parameters"]["tE"]["median_absolute_bias"] == pytest.approx(2.0)
        assert result["parameters"]["tE"]["median_fractional_bias"] == pytest.approx(0.1)

    def test_t0_never_reports_a_fractional_bias(self):
        fitted = [{"t0": 2458530.0}]
        reference = [{"t0": 2458529.0}]
        result = me.parameter_bias(fitted, reference, names=("t0",))
        assert "median_fractional_bias" not in result["parameters"]["t0"]
        assert result["parameters"]["t0"]["median_absolute_bias"] == pytest.approx(1.0)

    def test_a_single_catastrophic_fit_does_not_dominate_the_median(self):
        fitted = [{"tE": 20.1}, {"tE": 19.9}, {"tE": 20.0}, {"tE": 500.0}]
        reference = [{"tE": 20.0}] * 4
        result = me.parameter_bias(fitted, reference, names=("tE",))
        assert abs(result["parameters"]["tE"]["median_fractional_bias"]) < 0.02

    def test_missing_values_are_skipped_not_counted_as_zero(self):
        fitted = [{"tE": 20.0}, {"tE": None}]
        reference = [{"tE": 20.0}, {"tE": 15.0}]
        result = me.parameter_bias(fitted, reference, names=("tE",))
        assert result["parameters"]["tE"]["n_compared"] == 1

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            me.parameter_bias([{"tE": 1.0}], [{"tE": 1.0}, {"tE": 2.0}])

    def test_empty_input_reports_zero_compared(self):
        result = me.parameter_bias([], [])
        assert result["n_events"] == 0
        for name in me.POINT_LENS_NAMES:
            assert result["parameters"][name]["n_compared"] == 0


class TestEventEfficiency:
    def test_delegates_to_significance_evaluate_selection(self, monkeypatch):
        captured = {}

        def fake_evaluate_selection(records, dimensions=None, edges=None, seed=42):
            captured["records"] = records
            captured["dimensions"] = dimensions
            captured["seed"] = seed
            return {"cells": []}

        from astra import significance
        monkeypatch.setattr(significance, "evaluate_selection", fake_evaluate_selection)

        records = [{"detected": True, "tE_days": 20.0, "u0": 0.1, "baseline_mag": 18.0}]
        result = me.event_efficiency(records, seed=7)

        assert result == {"cells": []}
        assert captured["records"] is records
        assert captured["dimensions"] == ("tE_days", "u0", "baseline_mag")
        assert captured["seed"] == 7

    def test_uses_supplied_dimensions(self, monkeypatch):
        captured = {}

        def fake_evaluate_selection(records, dimensions=None, edges=None, seed=42):
            captured["dimensions"] = dimensions
            return {}

        from astra import significance
        monkeypatch.setattr(significance, "evaluate_selection", fake_evaluate_selection)

        me.event_efficiency([], dimensions=("tE_days",))
        assert captured["dimensions"] == ("tE_days",)


class TestIntervalContains:
    def test_true_when_truth_inside(self):
        assert me.interval_contains([1.0, 3.0], 2.0)

    def test_false_when_truth_outside(self):
        assert not me.interval_contains([1.0, 3.0], 5.0)

    def test_inclusive_at_boundaries(self):
        assert me.interval_contains([1.0, 3.0], 1.0)
        assert me.interval_contains([1.0, 3.0], 3.0)


def _trial_with_interval(truth_value, low, high, level="0.9", name="tE"):
    return me.CoverageTrial(
        truth={name: truth_value},
        intervals={name: {level: [low, high]}},
        names=(name,),
    )


class TestPosteriorCoverage:
    def test_calibrated_posterior_shows_near_nominal_coverage(self):
        # 90 trials, truth drawn uniformly inside a fixed-width interval so
        # ~90% of draws land inside a interval sized to match -- a
        # deliberately well-calibrated construction, the positive control.
        rng = np.random.default_rng(0)
        trials = []
        for _ in range(200):
            # Interval [0, 1]; truth uniform on [-0.05, 1.05] means the
            # interval contains the truth with probability ~0.9.
            truth = rng.uniform(-0.05, 1.05)
            trials.append(_trial_with_interval(truth, 0.0, 1.0))
        result = me.posterior_coverage(trials, levels=(0.9,))
        empirical = result["levels"]["0.9"]["tE"]["empirical"]
        assert 0.8 < empirical < 1.0

    def test_artificially_narrow_interval_shows_undercoverage(self):
        # Truth simulated from a WIDE spread but the reported interval is
        # narrow: a deliberately mis-calibrated posterior, over-confident.
        rng = np.random.default_rng(1)
        trials = []
        for _ in range(200):
            truth = rng.normal(0.0, 5.0)  # true scatter much wider than [-0.1, 0.1]
            trials.append(_trial_with_interval(truth, -0.1, 0.1))
        result = me.posterior_coverage(trials, levels=(0.9,))
        empirical = result["levels"]["0.9"]["tE"]["empirical"]
        assert empirical < 0.5

    def test_artificially_wide_interval_shows_overcoverage(self):
        # Truth tightly clustered but the reported interval is enormous:
        # deliberately mis-calibrated the other way, over-conservative.
        rng = np.random.default_rng(2)
        trials = []
        for _ in range(200):
            truth = rng.normal(0.0, 0.01)
            trials.append(_trial_with_interval(truth, -1000.0, 1000.0))
        result = me.posterior_coverage(trials, levels=(0.9,))
        empirical = result["levels"]["0.9"]["tE"]["empirical"]
        assert empirical > 0.99

    def test_empty_trials_reports_zero(self):
        result = me.posterior_coverage([])
        assert result == {"n_trials": 0, "levels": {}}

    def test_missing_interval_is_not_counted(self):
        trial = me.CoverageTrial(truth={"tE": 1.0}, intervals={}, names=("tE",))
        result = me.posterior_coverage([trial], levels=(0.9,))
        assert result["levels"]["0.9"]["tE"]["trials"] == 0
        assert result["levels"]["0.9"]["tE"]["empirical"] is None


class TestSbcRanks:
    def test_uniform_ranks_for_a_correct_posterior(self):
        rng = np.random.default_rng(0)
        trials = []
        for _ in range(300):
            samples = rng.normal(0.0, 1.0, size=(500, 1))
            truth = rng.normal(0.0, 1.0)  # drawn from the SAME distribution as samples
            trials.append(me.CoverageTrial(
                truth={"tE": truth}, intervals={}, samples=samples, names=("tE",)))
        result = me.sbc_ranks(trials)
        assert result["parameters"]["tE"]["normalised_mean"] == pytest.approx(0.5, abs=0.05)

    def test_biased_posterior_skews_the_mean_rank(self):
        rng = np.random.default_rng(1)
        trials = []
        for _ in range(200):
            samples = rng.normal(0.0, 1.0, size=(500, 1))
            truth = rng.normal(3.0, 1.0)  # systematically higher than the posterior
            trials.append(me.CoverageTrial(
                truth={"tE": truth}, intervals={}, samples=samples, names=("tE",)))
        result = me.sbc_ranks(trials)
        assert result["parameters"]["tE"]["normalised_mean"] > 0.8

    def test_trials_without_samples_are_skipped_and_counted(self):
        trials = [
            me.CoverageTrial(truth={"tE": 1.0}, intervals={}, samples=None, names=("tE",)),
            me.CoverageTrial(truth={"tE": 1.0}, intervals={},
                             samples=np.zeros((10, 1)), names=("tE",)),
        ]
        result = me.sbc_ranks(trials)
        assert result["skipped_no_samples"] == 1
        assert result["parameters"]["tE"]["n"] == 1

    def test_empty_trials_reports_zero(self):
        result = me.sbc_ranks([])
        assert result == {"n_trials": 0, "parameters": {}}


class TestSimulateOnRealCadence:
    def test_output_shape_matches_input_cadence(self):
        from astra.microlensing import PointLensParams

        time = np.linspace(0, 100, 50)
        flux_err = np.full(50, 0.1)
        truth = PointLensParams(t0=50.0, tE=20.0, u0=0.2)
        rng = np.random.default_rng(0)
        observed = me.simulate_on_real_cadence(time, flux_err, truth, rng)
        assert observed.shape == time.shape

    def test_noise_free_limit_matches_the_model_exactly(self):
        from astra.microlensing import PointLensParams, model_flux

        time = np.linspace(0, 100, 50)
        flux_err = np.zeros(50)
        truth = PointLensParams(t0=50.0, tE=20.0, u0=0.2)
        rng = np.random.default_rng(0)
        observed = me.simulate_on_real_cadence(time, flux_err, truth, rng)
        expected = model_flux(time, truth, 5.0, 1.0)
        assert observed == pytest.approx(expected)


class TestRunValidationStudy:
    def test_returns_all_three_named_metrics_on_a_small_real_like_cadence(self):
        emcee = pytest.importorskip("emcee", reason="emcee not installed (research extra)")
        rng = np.random.default_rng(0)
        time = np.sort(rng.uniform(0, 200, 120))
        flux_err = np.full(120, 0.05)

        result = me.run_validation_study(
            time, flux_err, n_trials=3, seed=1, n_steps=200, n_walkers=12)

        assert "parameter_bias" in result
        assert "event_efficiency" in result
        assert "posterior_coverage" in result
        assert "sbc" in result
        assert result["cadence"]["n_points"] == 120
        assert result["n_trials_requested"] == 3
