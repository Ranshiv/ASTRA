"""multimodal_synthetic.py: synthetic paired multimodal data (backlog item
11) -- validates the data-generation mechanism itself, since no real
multi-modal-complete objects exist yet in this codebase's acquisition
pipeline."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from astra import multimodal_synthetic as syn


class TestBuildSyntheticPairs:
    def test_every_object_gets_a_valid_class_label(self):
        batch = syn.build_synthetic_pairs(n=50, seed=0)
        assert set(np.unique(batch.class_labels)) <= set(range(len(syn.CLASS_KINDS)))
        assert len(batch.class_labels) == 50

    def test_all_four_modalities_have_matching_row_counts(self):
        batch = syn.build_synthetic_pairs(n=30, seed=1)
        assert len(batch.object_ids) == 30
        assert batch.lightcurve_values.shape[0] == 30
        assert batch.image_arrays.shape[0] == 30
        assert batch.spectrum_arrays.shape[0] == 30
        assert batch.catalog_features.shape[0] == 30

    def test_shapes_match_requested_dimensions(self):
        batch = syn.build_synthetic_pairs(
            n=10, seed=2, lc_length=48, image_size=12, spectrum_length=40)
        assert batch.lightcurve_values.shape == (10, 2, 48)
        assert batch.image_arrays.shape == (10, 1, 12, 12)
        assert batch.spectrum_arrays.shape == (10, 3, 40)

    def test_flux_derived_scales_are_monotonically_correlated(self):
        """lightcurve_scale/image_scale/spectrum_scale are all positive
        multiples of the SAME flux value derived from one shared magnitude
        -- they must be perfectly rank-correlated with each other."""
        batch = syn.build_synthetic_pairs(n=100, seed=3)
        rho_lc_image, _ = spearmanr(batch.lightcurve_scale, batch.image_scale)
        rho_lc_spectrum, _ = spearmanr(batch.lightcurve_scale, batch.spectrum_scale)
        assert rho_lc_image > 0.99
        assert rho_lc_spectrum > 0.99

    def test_catalog_scale_is_anti_correlated_with_flux_scale(self):
        """catalog_scale is the raw magnitude; magnitude and flux move in
        OPPOSITE directions for the same object (brighter = lower
        magnitude, higher flux) -- a real, physically meaningful relation,
        not independent noise."""
        batch = syn.build_synthetic_pairs(n=100, seed=4)
        rho, _ = spearmanr(batch.catalog_scale, batch.lightcurve_scale)
        assert rho < -0.99

    def test_lightcurve_injection_matches_the_object_class(self):
        """A flare_star object's light curve should differ from a plain
        smooth sine curve (evaluate.inject was applied); a quiet_dwarf's
        should not have any injected anomaly."""
        from astra import evaluate

        batch = syn.build_synthetic_pairs(n=200, seed=5)
        quiet_index = int(np.where(
            batch.class_labels == syn.CLASS_KINDS.index("quiet_dwarf"))[0][0])
        flare_index = int(np.where(
            batch.class_labels == syn.CLASS_KINDS.index("flare_star"))[0][0])

        # A flare injection adds a sharp positive excursion; its curve's
        # max value should clear a quiet curve's typical max by a wide
        # margin at the same strength (evaluate.inject's own convention).
        quiet_max = batch.lightcurve_values[quiet_index, 0].max()
        flare_max = batch.lightcurve_values[flare_index, 0].max()
        assert flare_max > quiet_max

    def test_reproducible_given_the_same_seed(self):
        first = syn.build_synthetic_pairs(n=20, seed=7)
        second = syn.build_synthetic_pairs(n=20, seed=7)
        np.testing.assert_array_equal(first.lightcurve_values, second.lightcurve_values)
        np.testing.assert_array_equal(first.class_labels, second.class_labels)

    def test_catalog_features_separate_by_class(self):
        """The linear-probe macro-F1 metric needs classes to occupy
        different regions of catalog-feature space at all -- confirm the
        generator actually does that, not just add noise."""
        batch = syn.build_synthetic_pairs(n=200, seed=8)
        centroids = np.array([
            batch.catalog_features[batch.class_labels == label].mean(axis=0)
            for label in range(len(syn.CLASS_KINDS))
        ])
        # Centroids for different classes should not collapse to the same point.
        pairwise_distances = [
            np.linalg.norm(centroids[i] - centroids[j])
            for i in range(len(centroids)) for j in range(i + 1, len(centroids))
        ]
        assert min(pairwise_distances) > 0.5


class TestAssembleRealMultimodalObject:
    """`assemble_real_multimodal_object`: assembly (not acquisition) from
    already-fetched real-shaped pieces -- see module docstring."""

    @staticmethod
    def _curve(n=80, seed=0):
        from astra.surveys.base import LightCurve, SourceRef

        rng = np.random.default_rng(seed)
        time = np.sort(rng.uniform(0, 200, n))
        value = 15.0 + 0.1 * np.sin(time / 10) + rng.normal(0, 0.02, n)
        err = np.full(n, 0.02)
        return LightCurve(
            source=SourceRef(survey="ZTF", object_id="x", ra_deg=10.0, dec_deg=20.0),
            release="dr1", band="g", value_kind="mag",
            time=time, value=value, value_err=err, time_system="JD_UTC")

    @staticmethod
    def _gaia_extra():
        return {"parallax": 5.0, "parallax_error": 0.1, "pmra": 1.0, "pmdec": -1.0,
               "phot_g_mean_mag": 15.5, "phot_bp_mean_mag": 15.8, "phot_rp_mean_mag": 15.1,
               "ra_deg": 10.001, "dec_deg": 20.001}

    def test_catalog_row_has_the_real_40_dim_contract(self):
        result = syn.assemble_real_multimodal_object(self._curve(), self._gaia_extra())
        assert result["catalog_features"].shape == (syn.CATALOG_FEATURE_COUNT,)
        assert np.all(np.isfinite(result["catalog_features"]))

    def test_lightcurve_tensor_matches_tensors_resample_shape(self):
        result = syn.assemble_real_multimodal_object(self._curve(), self._gaia_extra(),
                                                      lc_length=128)
        assert result["lightcurve_values"].shape == (2, 128)

    def test_gaia_derived_quantities_match_derived_properties_directly(self):
        from astra.surveys.gaia import derived_properties

        gaia_extra = self._gaia_extra()
        result = syn.assemble_real_multimodal_object(self._curve(), gaia_extra)
        derived = derived_properties(gaia_extra)

        from astra import featurematrix
        bp_rp_index = len(featurematrix.FEATURE_NAMES) + featurematrix.GAIA_JOIN_COLUMNS.index(
            "gaia_bp_rp")
        assert result["catalog_features"][bp_rp_index] == np.float32(derived["bp_rp"])

    def test_gaia_matched_flag_is_always_one(self):
        from astra import featurematrix

        result = syn.assemble_real_multimodal_object(self._curve(), self._gaia_extra())
        matched_index = len(featurematrix.FEATURE_NAMES) + featurematrix.GAIA_JOIN_COLUMNS.index(
            "gaia_matched")
        assert result["catalog_features"][matched_index] == 1.0

    def test_missing_spectrum_and_image_are_nan_not_fabricated(self):
        result = syn.assemble_real_multimodal_object(self._curve(), self._gaia_extra())
        assert np.all(np.isnan(result["spectrum_array"]))
        assert np.all(np.isnan(result["image_array"]))

    def test_real_spectrum_is_resampled_via_resample_spectrum(self):
        rng = np.random.default_rng(1)
        wavelength = np.linspace(4000.0, 9000.0, 200)
        flux = rng.normal(1.0, 0.1, 200)
        error = np.full(200, 0.05)
        result = syn.assemble_real_multimodal_object(
            self._curve(), self._gaia_extra(),
            spectrum_wavelength=wavelength, spectrum_flux=flux, spectrum_error=error,
            spectrum_length=64)
        assert result["spectrum_array"].shape == (3, 64)
        assert np.all(np.isfinite(result["spectrum_array"]))

    def test_real_image_is_preprocessed_via_preprocess_image(self):
        rng = np.random.default_rng(2)
        image = rng.normal(100.0, 5.0, (20, 20))
        result = syn.assemble_real_multimodal_object(
            self._curve(), self._gaia_extra(), image_array=image, image_size=16)
        assert result["image_array"].shape == (1, 16, 16)
        assert np.all(np.isfinite(result["image_array"]))

    def test_scale_scalars_derive_from_the_real_flux_relation(self):
        gaia_extra = self._gaia_extra()
        result = syn.assemble_real_multimodal_object(self._curve(), gaia_extra)
        expected = 10 ** (-0.4 * (gaia_extra["phot_g_mean_mag"] - syn.ZEROPOINT))
        assert result["lightcurve_scale"] == np.float32(expected)
        assert result["image_scale"] == np.float32(expected)

    def test_too_short_curve_raises(self):
        curve = self._curve(n=1)
        try:
            syn.assemble_real_multimodal_object(curve, self._gaia_extra())
            assert False, "expected ValueError"
        except ValueError:
            pass
