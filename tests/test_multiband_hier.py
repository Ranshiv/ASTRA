"""Hierarchical heteroscedastic multiband period model (multiband_hier.py).

celerite2 is an opt-in research dependency (`engine pyproject.toml`'s
`research` extra), gated like torch -- skip this whole file, the same way
`test_deep.py` skips on a missing torch, when it is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

celerite2 = pytest.importorskip("celerite2", reason="celerite2 not installed (opt-in 'research' extra)")

from astra import multiband_hier as mh

TRUE_PERIOD = 3.3
GRID = np.linspace(1.0, 8.0, 15)


def _synthetic_band(n: int, amplitude: float, error: float, seed: int,
                    period: float = TRUE_PERIOD):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0.0, 25.0, n))
    value = amplitude * np.sin(2 * np.pi * time / period) + rng.normal(0.0, error, n)
    return time, value, np.full(n, error)


class TestFitSharedPeriod:
    def test_recovers_known_period_across_bands(self):
        bands = {
            "g": _synthetic_band(40, 3.0, 0.2, seed=1),
            "r": _synthetic_band(40, 1.5, 0.15, seed=2),
        }
        fit = mh.fit_shared_period(bands, GRID)
        assert fit["best_period_days"] == pytest.approx(TRUE_PERIOD, abs=0.5)

    def test_band_specific_amplitude_ordering_is_preserved(self):
        bands = {
            "g": _synthetic_band(40, 3.0, 0.2, seed=1),
            "r": _synthetic_band(40, 1.0, 0.15, seed=2),
        }
        fit = mh.fit_shared_period(bands, GRID)
        assert fit["per_band"]["g"]["sigma"] > fit["per_band"]["r"]["sigma"]

    def test_requires_at_least_two_bands(self):
        bands = {"g": _synthetic_band(40, 3.0, 0.2, seed=1)}
        with pytest.raises(mh.MultibandHierError):
            mh.fit_shared_period(bands, GRID)

    def test_mismatched_array_lengths_raise(self):
        time, value, error = _synthetic_band(40, 3.0, 0.2, seed=1)
        bands = {"g": (time, value[:-1], error), "r": _synthetic_band(40, 1.0, 0.15, seed=2)}
        with pytest.raises(mh.MultibandHierError):
            mh.fit_shared_period(bands, GRID)

    def test_too_few_points_raises(self):
        bands = {"g": _synthetic_band(3, 3.0, 0.2, seed=1),
                 "r": _synthetic_band(40, 1.0, 0.15, seed=2)}
        with pytest.raises(mh.MultibandHierError):
            mh.fit_shared_period(bands, GRID)

    def test_empty_grid_raises(self):
        bands = {"g": _synthetic_band(40, 3.0, 0.2, seed=1),
                 "r": _synthetic_band(40, 1.0, 0.15, seed=2)}
        with pytest.raises(mh.MultibandHierError):
            mh.fit_shared_period(bands, np.array([]))


class TestCredibleInterval:
    def test_map_period_is_within_the_interval(self):
        bands = {"g": _synthetic_band(40, 3.0, 0.2, seed=1),
                 "r": _synthetic_band(40, 1.5, 0.15, seed=2)}
        fit = mh.fit_shared_period(bands, GRID)
        interval = mh.credible_interval(fit["profile"], level=0.68)
        assert interval["lower_days"] <= interval["map_period_days"] <= interval["upper_days"]

    def test_wider_level_never_narrows_the_interval(self):
        bands = {"g": _synthetic_band(30, 1.2, 0.6, seed=3),
                 "r": _synthetic_band(30, 0.8, 0.5, seed=4)}
        fit = mh.fit_shared_period(bands, GRID)
        narrow = mh.credible_interval(fit["profile"], level=0.5)
        wide = mh.credible_interval(fit["profile"], level=0.95)
        assert wide["n_grid_points_included"] >= narrow["n_grid_points_included"]

    def test_empty_profile_raises(self):
        with pytest.raises(mh.MultibandHierError):
            mh.credible_interval([])


class TestCalibratedFap:
    def test_a_real_periodic_signal_gets_a_low_fap(self):
        bands = {"g": _synthetic_band(40, 3.0, 0.2, seed=1),
                 "r": _synthetic_band(40, 1.5, 0.15, seed=2)}
        fit = mh.fit_shared_period(bands, GRID)
        fap = mh.calibrated_fap(bands, GRID, fit["log_likelihood"], n_null=15, seed=1)
        assert fap["false_alarm_probability"] < 0.2

    def test_pure_noise_gets_a_high_fap(self):
        bands = {"g": _synthetic_band(40, 0.0, 0.3, seed=5),
                 "r": _synthetic_band(40, 0.0, 0.25, seed=6)}
        fit = mh.fit_shared_period(bands, GRID)
        fap = mh.calibrated_fap(bands, GRID, fit["log_likelihood"], n_null=15, seed=1)
        assert fap["false_alarm_probability"] > 0.3

    def test_periodic_signal_fap_is_lower_than_noise_fap(self):
        periodic_bands = {"g": _synthetic_band(40, 3.0, 0.2, seed=1),
                          "r": _synthetic_band(40, 1.5, 0.15, seed=2)}
        noise_bands = {"g": _synthetic_band(40, 0.0, 0.3, seed=5),
                       "r": _synthetic_band(40, 0.0, 0.25, seed=6)}
        periodic_fit = mh.fit_shared_period(periodic_bands, GRID)
        noise_fit = mh.fit_shared_period(noise_bands, GRID)
        periodic_fap = mh.calibrated_fap(
            periodic_bands, GRID, periodic_fit["log_likelihood"], n_null=15, seed=7)
        noise_fap = mh.calibrated_fap(
            noise_bands, GRID, noise_fit["log_likelihood"], n_null=15, seed=7)
        assert periodic_fap["false_alarm_probability"] < noise_fap["false_alarm_probability"]


class TestAnalyzeObject:
    def test_full_report_for_a_periodic_object(self):
        bands = {"g": _synthetic_band(40, 3.0, 0.2, seed=1),
                 "r": _synthetic_band(40, 1.5, 0.15, seed=2)}
        report = mh.analyze_object(bands, GRID, n_null=10, seed=1)
        assert report["ready"]
        assert report["schema_version"] == mh.SCHEMA_VERSION
        assert report["best_period_days"] == pytest.approx(TRUE_PERIOD, abs=0.5)
        assert "credible_interval" in report
        assert "fap" in report

    def test_skips_fap_when_not_requested(self):
        bands = {"g": _synthetic_band(40, 3.0, 0.2, seed=1),
                 "r": _synthetic_band(40, 1.5, 0.15, seed=2)}
        report = mh.analyze_object(bands, GRID, compute_fap=False)
        assert report["ready"]
        assert "fap" not in report

    def test_not_ready_reports_reason_not_a_crash(self):
        bands = {"g": _synthetic_band(40, 3.0, 0.2, seed=1)}
        report = mh.analyze_object(bands, GRID)
        assert report["ready"] is False
        assert "reason" in report
