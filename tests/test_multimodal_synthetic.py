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
