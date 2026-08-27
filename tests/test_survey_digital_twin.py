"""survey_digital_twin.py: cadence/noise/artifact profile fitting and
synthetic sampling (backlog item 42).

Tests never touch the network or the real data root -- every curve here is
written to a `tmp_path` store, the same convention `conftest.py` states for
this whole suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import survey_digital_twin as sdt
from astra import store, tensors
from astra.surveys.base import LightCurve, SourceRef


def _write_ztf_like_curves(root, count=40, points=200, seed=0):
    """Curves with real seasonal gaps, so a fitted profile has real gap
    runs to sample from -- not a fully-dense synthetic fixture, which would
    make `mean_coverage`/`gap_run_lengths` trivially 1.0/empty."""
    rng = np.random.default_rng(seed)
    for i in range(count):
        season_a = np.sort(rng.uniform(0, 90, points // 2))
        season_b = np.sort(rng.uniform(200, 290, points - points // 2))
        time = np.concatenate([season_a, season_b])
        value = 18.0 + 0.3 * np.sin(time / 10.0) + rng.normal(0, 0.05, len(time))
        store.write_curve(LightCurve(
            source=SourceRef(survey="ZTF", object_id=f"twin{i}",
                             ra_deg=10.0, dec_deg=20.0),
            release="dr24", band="g", value_kind="mag",
            time=time, value=value, value_err=np.full(len(time), 0.05),
            time_system="HJD_UTC",
        ), root)


class TestFitSurveyProfile:
    def test_too_few_curves_degrades_explicitly(self, tmp_path):
        _write_ztf_like_curves(tmp_path, count=2)

        profile = sdt.fit_survey_profile("ZTF", root=tmp_path, limit=10)

        assert profile.n_curves_used < sdt.MIN_CURVES_FOR_PROFILE
        assert np.isnan(profile.mean_coverage)
        assert np.isnan(profile.noise_std)
        assert profile.gap_run_lengths == ()
        assert "fewer than" in profile.note

    def test_fits_a_real_coverage_and_noise_estimate(self, tmp_path):
        _write_ztf_like_curves(tmp_path, count=40)

        profile = sdt.fit_survey_profile("ZTF", root=tmp_path, limit=100)

        assert profile.n_curves_used >= sdt.MIN_CURVES_FOR_PROFILE
        assert profile.note == ""
        assert 0.0 < profile.mean_coverage < 1.0
        assert profile.noise_std > 0.0
        assert len(profile.gap_run_lengths) > 0

    def test_missing_survey_directory_is_a_clean_empty_profile(self, tmp_path):
        profile = sdt.fit_survey_profile("ZTF", root=tmp_path / "nothing_here")

        assert profile.n_curves_used == 0
        assert np.isnan(profile.mean_coverage)


class TestSampleSyntheticCurve:
    def _fitted_profile(self, tmp_path):
        _write_ztf_like_curves(tmp_path, count=40)
        return sdt.fit_survey_profile("ZTF", root=tmp_path, limit=100)

    def test_shape_and_channel_contract_matches_tensors_resample(self, tmp_path):
        profile = self._fitted_profile(tmp_path)
        rng = np.random.default_rng(0)

        curve = sdt.sample_synthetic_curve(profile, rng=rng)

        assert curve.shape == (2, profile.length)
        value, mask = curve[0], curve[1]
        assert set(np.unique(mask)).issubset({0.0, 1.0})
        # Zeroed-out mask positions must carry no signal, the same contract
        # `tensors.resample` enforces for a real interpolated gap.
        assert np.all(value[mask == 0.0] == 0.0)

    def test_coverage_tracks_the_fitted_profile_on_average(self, tmp_path):
        profile = self._fitted_profile(tmp_path)
        rng = np.random.default_rng(1)

        curves = [sdt.sample_synthetic_curve(profile, rng=rng) for _ in range(60)]
        mean_coverage = float(np.mean([c[1].mean() for c in curves]))

        # Not exact -- the gap-run sampler stops once it REACHES the target,
        # so it can overshoot by up to one run's length -- but it must track
        # the fitted value, not some unrelated default.
        assert abs(mean_coverage - profile.mean_coverage) < 0.15

    def test_an_unfitted_profile_falls_back_to_a_fully_valid_grid(self):
        profile = sdt.SurveyProfile(
            survey="ZTF", n_curves_used=1, mean_coverage=float("nan"),
            gap_run_lengths=(), noise_std=float("nan"), length=32,
        )
        curve = sdt.sample_synthetic_curve(profile, rng=np.random.default_rng(0))

        assert np.all(curve[1] == 1.0)

    def test_artifact_patch_is_spliced_in_when_forced(self, monkeypatch, tmp_path):
        profile = self._fitted_profile(tmp_path)
        monkeypatch.setattr(sdt, "ARTIFACT_INJECTION_RATE", 1.0)

        patch_value = np.full(8, 99.0, dtype=np.float32)
        patch_mask = np.ones(8, dtype=np.float32)
        patches = np.stack([np.stack([patch_value, patch_mask])])

        curve = sdt.sample_synthetic_curve(
            profile, rng=np.random.default_rng(2), artifact_patches=patches)

        # The distinctive patch value (99.0, far outside a normalised curve's
        # normal range) must appear somewhere in the spliced result. NOT
        # `pytest.approx` here: comparing a numpy array against a scalar
        # `approx` reduces to one aggregate bool (effectively `.all()`), not
        # an elementwise array -- `np.any(...)` over that collapses to
        # checking a single value, silently testing the wrong thing.
        assert np.any(np.isclose(curve[0], 99.0))


class TestSampleSyntheticBatch:
    def test_produces_a_well_formed_sequence_batch(self, tmp_path):
        _write_ztf_like_curves(tmp_path, count=40)
        profile = sdt.fit_survey_profile("ZTF", root=tmp_path, limit=100)

        batch = sdt.sample_synthetic_batch(profile, n=25, seed=7)

        assert isinstance(batch, tensors.SequenceBatch)
        assert batch.shape == (25, 2, profile.length)
        assert len(batch.identities) == 25
        assert all(identity["synthetic"] == "1" for identity in batch.identities)
        assert all(identity["survey"] == "ZTF" for identity in batch.identities)

    def test_same_seed_is_reproducible(self, tmp_path):
        _write_ztf_like_curves(tmp_path, count=40)
        profile = sdt.fit_survey_profile("ZTF", root=tmp_path, limit=100)

        first = sdt.sample_synthetic_batch(profile, n=10, seed=3)
        second = sdt.sample_synthetic_batch(profile, n=10, seed=3)

        np.testing.assert_array_equal(first.values, second.values)

    def test_zero_rows_is_a_clean_empty_batch(self, tmp_path):
        _write_ztf_like_curves(tmp_path, count=40)
        profile = sdt.fit_survey_profile("ZTF", root=tmp_path, limit=100)

        batch = sdt.sample_synthetic_batch(profile, n=0, seed=1)

        assert len(batch) == 0
        assert batch.shape == (0, 2, profile.length)
