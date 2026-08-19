"""Feature extraction, validated against curves with known behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from astra import features
from astra.features import FEATURE_NAMES
from astra.surveys.base import LightCurve, SourceRef


def make_curve(value, time=None, err=None, source=None) -> LightCurve:
    value = np.asarray(value, dtype=np.float64)
    n = len(value)
    time = 2458000.0 + np.arange(n) * 0.1 if time is None else np.asarray(time)
    err = np.full(n, 0.01) if err is None else np.asarray(err)
    return LightCurve(
        source=source or SourceRef(survey="ZTF", object_id="t1",
                                   ra_deg=0.0, dec_deg=0.0),
        release="dr24", band="g", value_kind="mag",
        time=time, value=value, value_err=err, time_system="HJD_UTC",
    )


@pytest.fixture
def rng():
    return np.random.default_rng(11)


class TestConstantVersusVariable:
    def test_constant_source_has_reduced_chi2_near_one(self, rng):
        """Scatter fully explained by the error bars must give chi2/dof ~ 1."""
        noise = rng.normal(0.0, 0.01, size=400)
        curve = make_curve(18.0 + noise, err=np.full(400, 0.01))

        result = features.extract(curve).values

        assert 0.7 < result["reduced_chi2"] < 1.4

    def test_variable_source_has_large_reduced_chi2(self, rng):
        time = 2458000.0 + np.arange(400) * 0.1
        signal = 18.0 + 0.5 * np.sin(2 * np.pi * time / 2.5)
        curve = make_curve(signal + rng.normal(0, 0.01, 400), time=time,
                           err=np.full(400, 0.01))

        result = features.extract(curve).values

        assert result["reduced_chi2"] > 100

    def test_eta_is_near_two_for_white_noise(self, rng):
        curve = make_curve(18.0 + rng.normal(0.0, 0.05, size=500))
        assert 1.7 < features.extract(curve).values["eta"] < 2.3

    def test_eta_is_small_for_smooth_variation(self):
        """Successive points of a smooth curve differ far less than the spread."""
        time = 2458000.0 + np.arange(500) * 0.01
        curve = make_curve(18.0 + np.sin(2 * np.pi * time / 3.0), time=time)
        assert features.extract(curve).values["eta"] < 0.5


class TestPhotometric:
    def test_amplitude_matches_the_input_range(self):
        curve = make_curve(np.linspace(17.0, 19.0, 100))
        assert features.extract(curve).values["amplitude"] == pytest.approx(2.0)

    def test_robust_amplitude_resists_a_single_bad_epoch(self, rng):
        value = 18.0 + rng.normal(0.0, 0.02, size=300)
        value[150] = 5.0  # one catastrophic outlier

        result = features.extract(make_curve(value)).values

        assert result["amplitude"] > 10.0        # raw range is destroyed
        assert result["robust_amplitude"] < 0.2  # robust one is not

    def test_weighted_mean_favours_precise_points(self):
        value = np.concatenate([np.full(50, 18.0), np.full(50, 20.0)])
        err = np.concatenate([np.full(50, 0.001), np.full(50, 1.0)])

        result = features.extract(make_curve(value, err=err)).values

        assert result["mean"] == pytest.approx(19.0, abs=0.01)
        assert result["weighted_mean"] < 18.1

    def test_kurtosis_of_gaussian_is_near_zero(self, rng):
        curve = make_curve(18.0 + rng.normal(0.0, 0.1, size=5000))
        assert abs(features.extract(curve).values["kurtosis"]) < 0.3

    def test_skew_detects_an_asymmetric_distribution(self, rng):
        """A flaring star brightens sharply and decays: strongly skewed."""
        value = 18.0 - rng.exponential(0.1, size=2000)
        assert features.extract(make_curve(value)).values["skew"] < -1.0


class TestTemporal:
    def test_linear_trend_recovers_a_known_slope(self):
        time = 2458000.0 + np.arange(200) * 1.0
        curve = make_curve(18.0 + 0.01 * np.arange(200), time=time)
        result = features.extract(curve).values
        assert result["linear_trend_per_day"] == pytest.approx(0.01, rel=1e-6)

    def test_time_span_matches_the_baseline(self):
        curve = make_curve(np.full(100, 18.0))
        assert features.extract(curve).values["time_span_days"] == \
            pytest.approx(9.9)

    def test_change_point_detects_a_level_shift(self):
        value = np.concatenate([np.full(100, 18.0), np.full(100, 17.0)])
        assert features.extract(make_curve(value)).values["change_point_score"] > 50

    def test_change_point_is_near_the_noise_floor_for_a_steady_source(self, rng):
        """White noise sits near 3 — the expected max of many normal draws."""
        curve = make_curve(18.0 + rng.normal(0.0, 0.01, size=200))
        assert features.extract(curve).values["change_point_score"] < 6

    def test_change_point_separates_a_real_step_from_noise(self, rng):
        """The statistic must have dynamic range, not just a threshold."""
        noise = rng.normal(0.0, 0.01, size=200)
        steady = make_curve(18.0 + noise)
        stepped = make_curve(18.0 + noise + np.concatenate(
            [np.zeros(100), np.full(100, 0.5)]))

        quiet = features.extract(steady).values["change_point_score"]
        loud = features.extract(stepped).values["change_point_score"]

        assert loud > 10 * quiet

    def test_max_gap_reports_the_seasonal_break(self):
        time = np.concatenate([
            2458000.0 + np.arange(50) * 0.1,
            2458200.0 + np.arange(50) * 0.1,
        ])
        result = features.extract(make_curve(np.full(100, 18.0), time=time)).values
        assert result["cadence_max_gap_days"] > 190


class TestPeriodic:
    def test_lomb_scargle_recovers_a_known_period(self, rng):
        """The core requirement: find the right period in noisy, uneven data."""
        period = 2.5
        time = 2458000.0 + np.sort(rng.uniform(0, 100, size=400))
        value = 18.0 + 0.4 * np.sin(2 * np.pi * time / period)
        curve = make_curve(value + rng.normal(0, 0.02, 400), time=time,
                           err=np.full(400, 0.02))

        result = features.extract(curve).values

        assert result["best_period_days"] == pytest.approx(period, rel=0.02)
        assert result["period_snr"] > 5

    def test_noise_gives_a_weak_period_peak(self, rng):
        time = 2458000.0 + np.sort(rng.uniform(0, 100, size=400))
        curve = make_curve(18.0 + rng.normal(0, 0.05, 400), time=time,
                           err=np.full(400, 0.05))

        noise_snr = features.extract(curve).values["period_snr"]

        assert noise_snr < 12  # far below the injected-signal case

    def test_short_curve_returns_nan_period(self):
        curve = make_curve(np.full(15, 18.0))
        assert np.isnan(features.extract(curve).values["best_period_days"])

    def test_multiband_period_uses_shared_signal(self, rng):
        period = 3.25
        time = 2458000.0 + np.sort(rng.uniform(0, 120, size=160))
        first = make_curve(18.0 + 0.3 * np.sin(2 * np.pi * time / period)
                           + rng.normal(0, 0.02, len(time)), time=time)
        second = make_curve(17.0 + 0.2 * np.sin(2 * np.pi * time / period + 0.2)
                            + rng.normal(0, 0.02, len(time)), time=time)
        second = LightCurve(source=second.source, release=second.release, band="r",
                            value_kind=second.value_kind, time=second.time,
                            value=second.value, value_err=second.value_err,
                            time_system=second.time_system)
        result = features.multiband_periodic_features([first, second])
        assert result["bands"] == 2
        assert result["best_period_days"] == pytest.approx(period, rel=0.03)

    def test_bocpd_finds_a_level_change(self):
        time = np.arange(200, dtype=float)
        value = np.concatenate([np.zeros(100), np.ones(100)])
        result = features.bocpd(time, value)
        assert result["max_probability"] > 0.1
        assert 70 < result["change_index"] < 130


class TestContract:
    def test_every_declared_feature_is_produced(self, rng):
        result = features.extract(make_curve(18.0 + rng.normal(0, 0.05, 200)))
        assert set(result.values) == set(FEATURE_NAMES)

    def test_too_short_a_curve_yields_nan_not_an_error(self):
        result = features.extract(make_curve([18.0, 18.1, 18.2])).values
        assert result["n_points"] == 3.0
        assert np.isnan(result["reduced_chi2"])

    def test_empty_curve_is_handled(self):
        result = features.extract(make_curve([])).values
        assert result["n_points"] == 0.0

    def test_identity_travels_with_the_features(self):
        source = SourceRef(survey="TESS", object_id="TIC 42",
                           ra_deg=1.0, dec_deg=2.0)
        curve = make_curve(np.full(50, 18.0), source=source)

        result = features.extract(curve, path="p.parquet")

        assert result.survey == "TESS"
        assert result.object_id == "TIC 42"
        assert result.to_dict()["feature_version"] == features.FEATURE_VERSION

    def test_zero_variance_curve_does_not_divide_by_zero(self):
        result = features.extract(make_curve(np.full(100, 18.0))).values
        assert result["std"] == 0.0
        assert not np.isinf(result["beyond_1std"])
