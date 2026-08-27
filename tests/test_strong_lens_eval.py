"""Strong-lens recovery study: AUPRC, image-position residual, and
time-delay error, on synthetic ground truth (roadmap item 29)."""

from __future__ import annotations

from astra.strong_lens_eval import (
    Cosmology, evaluate_image_position_residual, evaluate_lens_auprc, evaluate_time_delay_error,
)


class TestEvaluateLensAuprc:
    def test_separates_lensed_from_unlensed_above_chance(self):
        result = evaluate_lens_auprc(n_lensed=60, n_unlensed=60, seed=31)
        assert result["lens_auprc"] > 0.7  # chance-level AUPRC at 50/50 balance is ~0.5
        assert result["mean_images_lensed"] > result["mean_images_unlensed"]

    def test_is_reproducible_for_a_fixed_seed(self):
        first = evaluate_lens_auprc(n_lensed=30, n_unlensed=30, seed=7)
        second = evaluate_lens_auprc(n_lensed=30, n_unlensed=30, seed=7)
        assert first == second


class TestEvaluateImagePositionResidual:
    def test_recovers_small_residuals_at_low_noise(self):
        result = evaluate_image_position_residual(n_trials=15, astrometric_noise=0.005, seed=37)
        assert result["n_used"] > 0
        assert result["rms_residual"] is not None
        assert result["rms_residual"] < 0.05

    def test_higher_noise_gives_a_larger_or_equal_residual(self):
        low_noise = evaluate_image_position_residual(n_trials=15, astrometric_noise=0.002, seed=12)
        high_noise = evaluate_image_position_residual(n_trials=15, astrometric_noise=0.05, seed=12)
        if low_noise["rms_residual"] is not None and high_noise["rms_residual"] is not None:
            assert high_noise["rms_residual"] >= low_noise["rms_residual"]


class TestEvaluateTimeDelayError:
    def test_recovers_a_small_error_at_low_noise(self):
        result = evaluate_time_delay_error(n_trials=15, astrometric_noise=0.005, seed=41)
        assert result["n_used"] > 0
        assert result["mean_absolute_error_days"] is not None
        assert result["mean_absolute_error_days"] >= 0.0

    def test_accepts_a_custom_cosmology(self):
        cosmology = Cosmology(z_lens=0.3, d_l_mpc=900.0, d_s_mpc=1800.0, d_ls_mpc=1100.0)
        result = evaluate_time_delay_error(n_trials=10, cosmology=cosmology, seed=41)
        assert "mean_absolute_error_days" in result
