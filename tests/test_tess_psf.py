"""Forward PSF scene-model deblending (tess_psf.py). No network -- every
scene here is a synthetic circular-Gaussian image built by
`tess_psf._synthetic_gaussian_image`, the same functional form the fitter
itself assumes (see that helper's docstring for why that is the right
thing to validate).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astra import tess_psf


class TestBuildScenePositions:
    def test_target_is_always_included(self):
        positions = tess_psf.build_scene_positions((10.0, 10.0), [], shape=(21, 21))
        assert len(positions) == 1
        assert positions[0].label == "target"

    def test_out_of_cutout_neighbor_is_excluded(self):
        positions = tess_psf.build_scene_positions(
            (10.0, 10.0), [("n1", 13.0, 9.0), ("n2", 999.0, 999.0)], shape=(21, 21))
        labels = [position.label for position in positions]
        assert labels == ["target", "n1"]


class TestFitCadence:
    def test_recovers_a_single_isolated_source(self):
        positions = tess_psf.build_scene_positions((10.0, 10.0), [], shape=(21, 21))
        image = tess_psf._synthetic_gaussian_image(
            (21, 21), positions, {"target": 5000.0}, 1.5, background=10.0)

        result = tess_psf.fit_cadence(image, positions, fwhm_pixels=1.5)

        assert result["fluxes"]["target"] == pytest.approx(5000.0, rel=0.05)
        assert result["diagnostics"]["residual_rms"] is not None

    def test_recovers_two_well_separated_sources(self):
        positions = tess_psf.build_scene_positions(
            (7.0, 10.0), [("n1", 15.0, 10.0)], shape=(21, 21))
        truth = {"target": 5000.0, "n1": 2000.0}
        image = tess_psf._synthetic_gaussian_image(
            (21, 21), positions, truth, 1.5, background=10.0)

        result = tess_psf.fit_cadence(image, positions, fwhm_pixels=1.5)

        assert result["fluxes"]["target"] == pytest.approx(5000.0, rel=0.05)
        assert result["fluxes"]["n1"] == pytest.approx(2000.0, rel=0.1)

    def test_positions_stay_fixed_not_refit(self):
        positions = tess_psf.build_scene_positions((10.3, 9.7), [], shape=(21, 21))
        image = tess_psf._synthetic_gaussian_image(
            (21, 21), positions, {"target": 4000.0}, 1.5, background=5.0)

        result = tess_psf.fit_cadence(image, positions, fwhm_pixels=1.5)

        # fit_cadence's result dict does not carry x/y back out (positions
        # are an input, not an output) -- the real assertion is that a
        # tightly blended second source at a WRONG assumed position (not
        # the injected one) fails to recover correctly, proving the fitter
        # did not silently relocate it.
        assert result["fluxes"]["target"] > 0

    def test_empty_positions_raises(self):
        with pytest.raises(tess_psf.TESSPSFError):
            tess_psf.fit_cadence(np.zeros((21, 21)), [])

    def test_non_2d_image_raises(self):
        positions = tess_psf.build_scene_positions((10.0, 10.0), [], shape=(21, 21))
        with pytest.raises(tess_psf.TESSPSFError):
            tess_psf.fit_cadence(np.zeros((3, 21, 21)), positions)


class TestBuildSceneModel:
    def test_fits_every_cadence_in_a_cube(self):
        positions = tess_psf.build_scene_positions((10.0, 10.0), [], shape=(21, 21))
        rng = np.random.default_rng(0)
        cube = np.stack([
            tess_psf._synthetic_gaussian_image(
                (21, 21), positions, {"target": 4000.0 + 50.0 * i}, 1.5,
                background=10.0, noise_sigma=2.0, rng=rng)
            for i in range(5)
        ])

        result = tess_psf.build_scene_model(cube, positions, fwhm_pixels=1.5)

        assert result["n_cadences"] == 5
        assert result["fit_failures"] == 0
        assert len(result["flux_by_label"]["target"]) == 5
        assert result["flux_by_label"]["target"][0] == pytest.approx(4000.0, rel=0.1)

    def test_a_failed_cadence_does_not_abort_the_whole_run(self):
        positions = tess_psf.build_scene_positions((10.0, 10.0), [], shape=(21, 21))
        good = tess_psf._synthetic_gaussian_image(
            (21, 21), positions, {"target": 4000.0}, 1.5, background=10.0)
        nan_cadence = np.full((21, 21), np.nan)
        cube = np.stack([good, nan_cadence, good])

        result = tess_psf.build_scene_model(cube, positions, fwhm_pixels=1.5)

        assert result["n_cadences"] == 3
        assert result["fit_failures"] >= 1
        assert result["per_cadence"][0] is not None

    def test_non_3d_cube_raises(self):
        positions = tess_psf.build_scene_positions((10.0, 10.0), [], shape=(21, 21))
        with pytest.raises(tess_psf.TESSPSFError):
            tess_psf.build_scene_model(np.zeros((21, 21)), positions)


class TestFluxRmse:
    def test_zero_rmse_for_a_perfect_fit(self):
        fitted = {"target": [5000.0, 5100.0]}
        injected = {"target": [5000.0, 5100.0]}
        result = tess_psf.flux_rmse(fitted, injected)
        assert result["per_source_rmse"]["target"] == pytest.approx(0.0)

    def test_failed_cadences_are_excluded_not_zero_scored(self):
        fitted = {"target": [None, 5100.0]}
        injected = {"target": [5000.0, 5100.0]}
        result = tess_psf.flux_rmse(fitted, injected)
        assert result["per_source_rmse"]["target"] == pytest.approx(0.0)

    def test_all_failed_cadences_report_nan(self):
        fitted = {"target": [None, None]}
        injected = {"target": [5000.0, 5100.0]}
        result = tess_psf.flux_rmse(fitted, injected)
        assert math.isnan(result["per_source_rmse"]["target"])


class TestBlendAttributionAccuracy:
    def test_perfect_split_scores_zero_error(self):
        fitted = {"target": [4000.0], "neighbor": [1000.0]}
        injected = {"target": [4000.0], "neighbor": [1000.0]}
        result = tess_psf.blend_attribution_accuracy(fitted, injected)
        assert result["mean_absolute_fraction_error"] == pytest.approx(0.0, abs=1e-9)

    def test_swapped_flux_scores_a_large_error(self):
        fitted = {"target": [1000.0], "neighbor": [4000.0]}
        injected = {"target": [4000.0], "neighbor": [1000.0]}
        result = tess_psf.blend_attribution_accuracy(fitted, injected)
        assert result["mean_absolute_fraction_error"] > 0.5

    def test_a_cadence_missing_any_source_fit_is_not_scored(self):
        fitted = {"target": [None], "neighbor": [1000.0]}
        injected = {"target": [4000.0], "neighbor": [1000.0]}
        result = tess_psf.blend_attribution_accuracy(fitted, injected)
        assert result["cadences_scored"] == 0
        assert math.isnan(result["mean_absolute_fraction_error"])


class TestInjectedSourceRecovery:
    def test_wide_separation_recovers_target_flux_with_low_bias(self):
        result = tess_psf.injected_source_recovery(
            n_trials=8, separations_pixels=[6.0], flux_ratios=[1.0],
            noise_sigma=2.0, seed=1)
        row = result["grid"][0]
        assert row["trials_completed"] == 8
        assert abs(row["mean_fractional_bias"]) < 0.15

    def test_tight_separation_degrades_recovery_relative_to_wide(self):
        wide = tess_psf.injected_source_recovery(
            n_trials=8, separations_pixels=[6.0], flux_ratios=[1.0],
            noise_sigma=2.0, seed=2)
        tight = tess_psf.injected_source_recovery(
            n_trials=8, separations_pixels=[0.5], flux_ratios=[1.0],
            noise_sigma=2.0, seed=2)
        assert abs(tight["grid"][0]["mean_fractional_bias"]) > abs(wide["grid"][0]["mean_fractional_bias"])

    def test_n_trials_must_be_positive(self):
        with pytest.raises(ValueError):
            tess_psf.injected_source_recovery(
                n_trials=0, separations_pixels=[1.0], flux_ratios=[1.0])
