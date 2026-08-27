"""stellar_manifold_eval.py: feature-space injection and precision-at-fixed-
recall -- both newly built for this codebase, so their own correctness is
tested directly (not just the higher-level contribution study)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import stellar_manifold_eval as sme
from astra.featurematrix import (
    FeatureMatrix, GAIA_JOIN_COLUMNS, STELLAR_MANIFOLD_COLUMNS,
)
from astra.features import FEATURE_NAMES


def _gaia_manifold_matrix(n=100, seed=0) -> FeatureMatrix:
    """A fully-finite, fully-matched Gaia+manifold-joined matrix, built
    directly (not via the real crossmatch pipeline) so these tests exercise
    inject_cmd_outliers/evaluate_manifold_contribution's own logic, not the
    Gaia join's."""
    rng = np.random.default_rng(seed)
    n_base = len(FEATURE_NAMES)
    base = rng.normal(0.0, 1.0, size=(n, n_base))

    bp_rp = rng.uniform(0.0, 4.0, size=n)
    from astra import stellar_manifold
    abs_g = np.array([stellar_manifold.nearest_track_point(c, 0.0)["track_abs_g_mag"]
                      for c in bp_rp])

    gaia = np.zeros((n, len(GAIA_JOIN_COLUMNS)))
    gaia[:, GAIA_JOIN_COLUMNS.index("gaia_bp_rp")] = bp_rp
    gaia[:, GAIA_JOIN_COLUMNS.index("gaia_abs_g_mag")] = abs_g
    gaia[:, GAIA_JOIN_COLUMNS.index("gaia_parallax")] = 5.0
    gaia[:, GAIA_JOIN_COLUMNS.index("gaia_parallax_snr")] = 20.0
    gaia[:, GAIA_JOIN_COLUMNS.index("gaia_pmra")] = 0.0
    gaia[:, GAIA_JOIN_COLUMNS.index("gaia_pmdec")] = 0.0
    gaia[:, GAIA_JOIN_COLUMNS.index("gaia_phot_g_mean_mag")] = 15.0
    gaia[:, GAIA_JOIN_COLUMNS.index("gaia_distance_pc")] = 200.0
    gaia[:, GAIA_JOIN_COLUMNS.index("gaia_ra_now_deg")] = 180.0
    gaia[:, GAIA_JOIN_COLUMNS.index("gaia_dec_now_deg")] = 20.0
    gaia[:, GAIA_JOIN_COLUMNS.index("gaia_matched")] = 1.0

    manifold = np.zeros((n, len(STELLAR_MANIFOLD_COLUMNS)))
    manifold[:, STELLAR_MANIFOLD_COLUMNS.index("manifold_residual_mag")] = 0.0
    manifold[:, STELLAR_MANIFOLD_COLUMNS.index("manifold_arc_length")] = 0.5
    manifold[:, STELLAR_MANIFOLD_COLUMNS.index("manifold_teff_k")] = 5000.0
    manifold[:, STELLAR_MANIFOLD_COLUMNS.index("manifold_matched")] = 1.0

    values = np.hstack([base, gaia, manifold])
    identities = [{"object_id": f"s{i}", "survey": "TEST", "band": "g", "path": f"s{i}",
                  "gaia_a_g": 0.0, "gaia_ebpminrp": 0.0}
                 for i in range(n)]
    names = FEATURE_NAMES + GAIA_JOIN_COLUMNS + STELLAR_MANIFOLD_COLUMNS
    return FeatureMatrix(values=values, identities=identities, feature_names=names)


class TestPrecisionAtRecall:
    def test_perfect_ranking_is_one_at_any_achievable_recall(self):
        labels = np.array([0] * 90 + [1] * 10)
        scores = labels.astype(float)
        assert sme.precision_at_recall(labels, scores, 0.5) == pytest.approx(1.0)
        assert sme.precision_at_recall(labels, scores, 1.0) == pytest.approx(1.0)

    def test_degenerate_labels_return_none(self):
        assert sme.precision_at_recall(np.zeros(10), np.random.rand(10), 0.5) is None
        assert sme.precision_at_recall(np.ones(10), np.random.rand(10), 0.5) is None

    def test_random_ranking_is_near_base_rate(self):
        rng = np.random.default_rng(1)
        labels = np.array([0] * 900 + [1] * 100)
        result = sme.precision_at_recall(labels, rng.normal(size=1000), 0.9)
        assert 0.05 < result < 0.3


class TestInjectCmdOutliers:
    def test_requires_gaia_and_manifold_columns(self):
        matrix = FeatureMatrix(values=np.empty((0, len(FEATURE_NAMES))), identities=[])
        with pytest.raises(ValueError, match="Gaia- and manifold-joined"):
            sme.inject_cmd_outliers(matrix)

    def test_labels_the_right_number_of_rows(self):
        matrix = _gaia_manifold_matrix(n=100)
        _, labels = sme.inject_cmd_outliers(matrix, fraction=0.1, seed=3)
        assert labels.sum() == 10
        assert len(labels) == 100

    def test_untouched_rows_are_unchanged(self):
        matrix = _gaia_manifold_matrix(n=50)
        injected, labels = sme.inject_cmd_outliers(matrix, fraction=0.1, seed=3)
        for index in np.where(labels == 0)[0]:
            np.testing.assert_array_equal(injected.values[index], matrix.values[index])

    def test_injected_rows_have_recomputed_manifold_residual(self):
        matrix = _gaia_manifold_matrix(n=50)
        injected, labels = sme.inject_cmd_outliers(matrix, fraction=0.2, offset_mag=3.0, seed=5)
        residual_col = injected.feature_names.index("manifold_residual_mag")
        for index in np.where(labels == 1)[0]:
            # The residual must reflect the injected offset, not the
            # pre-injection value of 0.0 every row starts at.
            assert abs(injected.values[index, residual_col]) == pytest.approx(3.0, abs=1e-6)


class TestEvaluateManifoldContribution:
    def test_raises_when_too_few_matched_rows(self):
        matrix = _gaia_manifold_matrix(n=5)
        with pytest.raises(ValueError, match="at least 20"):
            sme.evaluate_manifold_contribution(matrix)

    def test_runs_both_arms_and_returns_well_formed_summaries(self):
        matrix = _gaia_manifold_matrix(n=150, seed=2)
        results = sme.evaluate_manifold_contribution(
            matrix, fractions=(0.1,), target_recalls=(0.5,), seeds=(1, 2, 3))

        assert 0.1 in results
        arms = results[0.1]["0.5"]
        assert set(arms.keys()) == {"with_manifold", "without_manifold"}
        # Not asserting which arm wins -- only that both ran and produced a
        # well-formed summary shape.
        for arm in arms.values():
            if arm is not None:
                assert {"mean", "std", "ci95", "n"} <= arm.keys()

    def test_requires_at_least_two_seeds(self):
        matrix = _gaia_manifold_matrix(n=150)
        with pytest.raises(ValueError, match="at least two seeds"):
            sme.evaluate_manifold_contribution(matrix, seeds=(1,))
