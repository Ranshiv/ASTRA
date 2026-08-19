"""Sequence building: normalisation, gap masking and splitting."""

from __future__ import annotations

import numpy as np
import pytest

from astra import store, tensors
from astra.surveys.base import LightCurve, SourceRef


def make_curve(n=200, value=None, time=None, survey="ZTF", object_id="t1"):
    value = np.full(n, 18.0) if value is None else np.asarray(value, dtype=float)
    time = 2458000.0 + np.arange(len(value)) * 0.5 if time is None \
        else np.asarray(time, dtype=float)
    return LightCurve(
        source=SourceRef(survey=survey, object_id=object_id,
                         ra_deg=0.0, dec_deg=0.0),
        release="dr24", band="g", value_kind="mag",
        time=time, value=value, value_err=np.full(len(value), 0.01),
        time_system="HJD_UTC",
    )


class TestNormalise:
    def test_centres_on_the_median(self):
        out = tensors.normalise(np.array([17.0, 18.0, 19.0]))
        assert out[1] == pytest.approx(0.0)

    def test_scale_is_robust_to_one_outlier(self):
        rng = np.random.default_rng(0)
        clean = 18.0 + rng.normal(0, 0.1, 500)
        spiked = clean.copy()
        spiked[250] = 100.0

        clean_scale = float(np.std(tensors.normalise(clean)))
        spiked_scale = float(np.std(tensors.normalise(spiked)[:200]))

        assert spiked_scale == pytest.approx(clean_scale, rel=0.3)

    def test_constant_curve_yields_zeros_not_nan(self):
        out = tensors.normalise(np.full(50, 18.0))
        assert np.all(out == 0.0)

    def test_output_is_float32(self):
        assert tensors.normalise(np.arange(20.0)).dtype == np.float32


class TestResample:
    def test_output_shape_is_two_channels(self):
        out = tensors.resample(make_curve(200), length=128)
        assert out.shape == (2, 128)

    def test_short_curve_is_rejected(self):
        assert tensors.resample(make_curve(5)) is None

    def test_zero_span_is_rejected(self):
        curve = make_curve(value=np.full(50, 18.0), time=np.full(50, 2458000.0))
        assert tensors.resample(curve) is None

    def test_well_sampled_curve_is_almost_fully_valid(self):
        mask = tensors.resample(make_curve(500), length=128)[1]
        assert mask.mean() > 0.95

    def test_seasonal_gap_is_masked_out(self):
        """Interpolating across a 200-day gap invents data; the mask says so."""
        time = np.concatenate([
            2458000.0 + np.arange(100) * 0.5,
            2458300.0 + np.arange(100) * 0.5,
        ])
        curve = make_curve(value=np.full(200, 18.0), time=time)

        mask = tensors.resample(curve, length=256)[1]

        assert mask.mean() < 0.6  # the gap occupies much of the span

    def test_invented_points_are_zeroed(self):
        time = np.concatenate([
            2458000.0 + np.arange(50) * 0.5,
            2458400.0 + np.arange(50) * 0.5,
        ])
        rng = np.random.default_rng(1)
        curve = make_curve(value=18.0 + rng.normal(0, 0.5, 100), time=time)

        sequence = tensors.resample(curve, length=256)

        assert np.all(sequence[0][sequence[1] == 0] == 0.0)

    def test_units_are_removed_so_surveys_are_comparable(self):
        """A magnitude curve and a flux curve of the same shape must match."""
        rng = np.random.default_rng(5)
        shape = np.sin(np.linspace(0, 8 * np.pi, 300))

        mags = tensors.resample(make_curve(value=18.0 + 0.3 * shape))
        flux = tensors.resample(make_curve(value=15000.0 + 250.0 * shape,
                                           survey="TESS"))

        np.testing.assert_allclose(mags[0], flux[0], atol=0.05)


class TestBuild:
    def test_build_over_the_store(self, curve, tmp_path):
        store.write_curve(curve, tmp_path)
        batch = tensors.build(root=tmp_path, length=64)

        assert len(batch) == 1
        assert batch.shape == (1, 2, 64)
        assert batch.identities[0]["survey"] == "ZTF"

    def test_empty_store(self, tmp_path):
        batch = tensors.build(root=tmp_path / "none", length=64)
        assert len(batch) == 0
        assert batch.shape == (0, 2, 64)

    def test_too_short_curves_are_skipped(self, tmp_path):
        store.write_curve(make_curve(8), tmp_path)
        assert len(tensors.build(root=tmp_path)) == 0

    def test_summary_reports_coverage(self, curve, tmp_path):
        store.write_curve(curve, tmp_path)
        summary = tensors.build(root=tmp_path).to_dict()
        assert 0.0 <= summary["mean_coverage"] <= 1.0


class TestSplit:
    def test_split_sizes(self):
        batch = tensors.SequenceBatch(
            values=np.zeros((100, 2, 64), dtype=np.float32),
            identities=[{}] * 100, length=64)

        train, test, train_idx, test_idx = tensors.train_test_split(batch, 0.2)

        assert len(train) == 80
        assert len(test) == 20
        assert len(set(train_idx) & set(test_idx)) == 0

    def test_split_is_deterministic(self):
        batch = tensors.SequenceBatch(
            values=np.zeros((50, 2, 32), dtype=np.float32),
            identities=[{}] * 50, length=32)

        _, _, first, _ = tensors.train_test_split(batch, seed=7)
        _, _, second, _ = tensors.train_test_split(batch, seed=7)

        np.testing.assert_array_equal(first, second)

    def test_empty_batch_splits_without_error(self):
        batch = tensors.SequenceBatch(
            values=np.empty((0, 2, 64), dtype=np.float32),
            identities=[], length=64)
        train, test, _, _ = tensors.train_test_split(batch)
        assert len(train) == 0 and len(test) == 0


def seasonal_curve(seasons=3, per_season=120, gap_days=445.0, period=0.6,
                   seed=0):
    """Ground-based sampling: nightly for a few months, then nothing.

    ZTF fields set for most of the year, so a uniform grid over the full
    baseline spends most of its points inside gaps -- measured mean validity
    on real data is about 0.40.
    """
    rng = np.random.default_rng(seed)
    time = np.concatenate([
        season * (per_season + gap_days) + np.sort(rng.uniform(0, per_season, per_season))
        for season in range(seasons)
    ])
    value = 18.0 + 0.5 * np.sin(2 * np.pi * time / period) \
        + rng.normal(0, 0.02, len(time))
    return make_curve(value=value, time=2458000.0 + time)


class TestSeasonBounds:
    def test_gaps_split_the_record(self):
        curve = seasonal_curve(seasons=3)
        assert len(tensors.season_bounds(curve.time)) == 3

    def test_continuous_sampling_is_one_season(self):
        assert len(tensors.season_bounds(make_curve(n=200).time)) == 1

    def test_degenerate_input_is_a_single_span(self):
        assert tensors.season_bounds(np.array([2458000.0])) == [(0, 1)]
        assert tensors.season_bounds(np.array([])) == [(0, 0)]

    def test_threshold_scales_with_the_curves_own_cadence(self):
        """A two-minute-cadence sector and a nightly field share no absolute
        definition of a gap, only a relative one."""
        fast = make_curve(n=300, time=2458000.0 + np.arange(300) * (2 / 1440))
        assert len(tensors.season_bounds(fast.time)) == 1


class TestSeasonResampling:
    def test_coverage_improves_dramatically_over_a_uniform_grid(self):
        curve = seasonal_curve()
        uniform = tensors.resample(curve)
        seasonal = tensors.resample_by_season(curve)

        assert uniform[1].mean() < 0.45
        assert seasonal[1].mean() > 0.90

    def test_shape_is_unchanged(self):
        seasonal = tensors.resample_by_season(seasonal_curve())
        assert seasonal.shape == (2, tensors.DEFAULT_LENGTH)

    def test_single_season_falls_back_to_the_time_grid(self):
        curve = make_curve(n=200)
        np.testing.assert_array_equal(tensors.resample_by_season(curve),
                                      tensors.resample(curve))

    def test_short_curves_are_refused_not_padded(self):
        assert tensors.resample_by_season(make_curve(n=4)) is None

    def test_seasons_share_one_normalisation(self):
        """Normalising each season separately would erase real season-to-season
        brightness change, which for a long-period variable is the signal."""
        rng = np.random.default_rng(3)
        time = np.concatenate([s * 565 + np.sort(rng.uniform(0, 120, 120))
                               for s in range(2)])
        # Second season is a magnitude fainter than the first.
        value = np.concatenate([np.full(120, 18.0), np.full(120, 19.0)]) \
            + rng.normal(0, 0.01, 240)
        curve = make_curve(value=value, time=2458000.0 + time)

        sequence = tensors.resample_by_season(curve)
        first, second = sequence[0][:120], sequence[0][-120:]
        assert abs(float(first.mean()) - float(second.mean())) > 1.0


class TestPhaseResampling:
    def test_folding_covers_phase_densely(self):
        curve = seasonal_curve()
        folded = tensors.resample_by_phase(curve, 0.6)

        assert folded.shape == (2, tensors.DEFAULT_LENGTH)
        assert folded[1].mean() > 0.85

    def test_a_nonsense_period_is_refused(self):
        curve = seasonal_curve()
        assert tensors.resample_by_phase(curve, 0.0) is None
        assert tensors.resample_by_phase(curve, float("nan")) is None


class TestResampleModes:
    def test_time_mode_is_byte_identical_to_the_original(self):
        """The default must not move: every stored result depends on it."""
        curve = seasonal_curve()
        np.testing.assert_array_equal(
            tensors.resample_curve(curve, mode="time"), tensors.resample(curve))

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown resample mode"):
            tensors.resample_curve(seasonal_curve(), mode="fourier")
        with pytest.raises(ValueError, match="unknown resample mode"):
            tensors.build(mode="fourier")

    def test_phase_without_a_period_falls_back_rather_than_dropping(self):
        """Dropping the curve would silently change the population trained on."""
        curve = seasonal_curve()
        np.testing.assert_array_equal(
            tensors.resample_curve(curve, mode="phase", period_days=None),
            tensors.resample(curve))

    def test_batch_records_its_mode_and_coverage(self, isolated_root):
        store.write_curve(seasonal_curve())

        default = tensors.build(survey="ztf")
        seasonal = tensors.build(survey="ztf", mode="season")

        assert default.to_dict()["mode"] == "time"
        assert seasonal.to_dict()["mode"] == "season"
        assert seasonal.to_dict()["mean_coverage"] > \
            default.to_dict()["mean_coverage"]


class TestPreprocessingContract:
    def test_each_mode_hashes_differently(self):
        """Sequences built under different modes are not comparable, so the
        provenance hash has to say so without anyone remembering to bump it."""
        from astra import experiment

        hashes = {mode: experiment.preprocessing_schema_hash(mode)
                  for mode in tensors.RESAMPLE_MODES}
        assert len(set(hashes.values())) == len(tensors.RESAMPLE_MODES)

    def test_time_mode_hash_is_unchanged(self):
        """Existing records must not suddenly report drift."""
        from astra import experiment

        assert experiment.preprocessing_schema_hash("time") == \
            experiment.preprocessing_schema_hash()

    def test_experiment_records_the_mode_it_used(self, isolated_root):
        from astra import experiment

        record = experiment.run("resample_test", {"resample_mode": "season"},
                                lambda: {"ok": True})
        assert record.provenance.preprocessing_schema_hash == \
            experiment.preprocessing_schema_hash("season")

        verification = experiment.verify(record.provenance.experiment_id)
        assert "preprocessing_schema_hash" not in verification["drift"]
