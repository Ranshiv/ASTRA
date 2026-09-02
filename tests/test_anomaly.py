"""Feature matrices and the baseline detector ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from astra import anomaly, featurematrix, store
from astra.featurematrix import FeatureMatrix
from astra.features import FEATURE_NAMES
from astra.surveys.base import LightCurve, SourceRef


def synthetic_matrix(n_normal=200, n_outliers=5, seed=0) -> FeatureMatrix:
    """A correlated population with a few objects lying off its manifold.

    Real feature vectors are highly correlated — amplitude, chi-square and the
    Stetson indices all rise together for a variable star — so an ordinary
    population occupies a low-dimensional surface inside the feature space.
    The outliers here are displaced *off* that surface rather than merely
    shifted along it, which is what makes them anomalous to a reconstruction
    method as well as to a distance-based one. An offset along the principal
    direction would be invisible to PCA by construction.
    """
    rng = np.random.default_rng(seed)
    width = len(FEATURE_NAMES)
    latent_dim = 3

    loading = rng.normal(0.0, 1.0, size=(latent_dim, width))

    normal = (rng.normal(0.0, 1.0, size=(n_normal, latent_dim)) @ loading
              + rng.normal(0.0, 0.05, size=(n_normal, width)))
    outliers = (rng.normal(0.0, 1.0, size=(n_outliers, latent_dim)) @ loading
                + rng.normal(0.0, 6.0, size=(n_outliers, width)))
    values = np.vstack([normal, outliers])

    identities = [
        {"object_id": f"n{i}", "survey": "TEST", "band": "g", "path": f"n{i}"}
        for i in range(n_normal)
    ] + [
        {"object_id": f"OUT{i}", "survey": "TEST", "band": "g", "path": f"o{i}"}
        for i in range(n_outliers)
    ]
    return FeatureMatrix(values=values, identities=identities)


class TestFeatureMatrixBuild:
    def test_build_over_the_store(self, curve, tmp_path):
        store.write_curve(curve, tmp_path)
        matrix = featurematrix.build(root=tmp_path)

        assert len(matrix) == 1
        assert matrix.shape[1] == len(FEATURE_NAMES)
        assert matrix.identities[0]["survey"] == "ZTF"

    def test_build_on_empty_store(self, tmp_path):
        matrix = featurematrix.build(root=tmp_path / "nothing")
        assert len(matrix) == 0
        assert matrix.shape[1] == len(FEATURE_NAMES)

    def test_build_filters_by_survey(self, curve, tmp_path):
        store.write_curve(curve, tmp_path)
        assert len(featurematrix.build(survey="ZTF", root=tmp_path)) == 1
        assert len(featurematrix.build(survey="TESS", root=tmp_path)) == 0

    def test_limit_is_respected(self, tmp_path, source):
        for i in range(5):
            other = SourceRef(survey="ZTF", object_id=f"obj{i}",
                              ra_deg=0.0, dec_deg=0.0)
            lc = LightCurve(source=other, release="dr24", band="g",
                            value_kind="mag",
                            time=2458000.0 + np.arange(50) * 0.1,
                            value=np.full(50, 18.0), value_err=np.full(50, 0.01))
            store.write_curve(lc, tmp_path)

        assert len(featurematrix.build(limit=3, root=tmp_path)) == 3

    def test_column_lookup_by_name(self, curve, tmp_path):
        store.write_curve(curve, tmp_path)
        matrix = featurematrix.build(root=tmp_path)
        assert matrix.column("n_points")[0] == 200.0

    def test_resumable_build_writes_checkpoint_and_reuses_parts(self, tmp_path, curve):
        store.write_curve(curve, tmp_path)
        checkpoint = tmp_path / "batch.json"
        first, report = featurematrix.build_resumable(
            root=tmp_path, batch_size=1, workers=1, checkpoint=checkpoint,
        )
        assert len(first) == 1
        assert report.completed == 1
        assert report.batches == 1
        assert checkpoint.exists()

        second, resumed = featurematrix.build_resumable(
            root=tmp_path, batch_size=1, workers=1, checkpoint=checkpoint,
        )
        assert resumed.resumed is True
        assert resumed.completed == 1
        np.testing.assert_allclose(first.values, second.values, equal_nan=True)


class TestFeatureMatrixPersistence:
    def test_round_trip(self, tmp_path):
        original = synthetic_matrix(n_normal=20, n_outliers=2)
        featurematrix.save(original, "test", tmp_path)
        loaded = featurematrix.load("test", tmp_path)

        assert len(loaded) == len(original)
        np.testing.assert_allclose(loaded.values, original.values)
        assert loaded.identities[0]["object_id"] == original.identities[0]["object_id"]

    def test_feature_version_is_in_the_filename(self, tmp_path):
        path = featurematrix.matrix_path("x", tmp_path)
        assert f"_v{featurematrix.FEATURE_VERSION}" in path.name

    def test_matrix_is_small_relative_to_the_curves(self, tmp_path):
        """The storage argument: a matrix is the cheap, permanent asset."""
        featurematrix.save(synthetic_matrix(n_normal=5000), "big", tmp_path)
        path = featurematrix.matrix_path("big", tmp_path)

        assert path.stat().st_size < 2 * 1024 * 1024  # under 2 MB for 5k rows

    def test_listing_reports_saved_matrices(self, tmp_path):
        featurematrix.save(synthetic_matrix(n_normal=10), "one", tmp_path)
        listing = featurematrix.list_matrices(tmp_path)

        assert len(listing) == 1
        assert listing[0]["rows"] == 15


class TestGaiaJoin:
    """Gaia is a catalogue connector (fetch_light_curves returns []), so its
    only route into a feature matrix is a positional column join onto rows
    another survey already contributed -- never rows of its own. See
    featurematrix.join_gaia_columns and the ztf_gaia entry in
    docs/LIMITATIONS.md for why a row union would be wrong here."""

    def _ztf_curve(self, object_id: str, ra_deg: float, dec_deg: float) -> LightCurve:
        source = SourceRef(survey="ZTF", object_id=object_id,
                           ra_deg=ra_deg, dec_deg=dec_deg)
        return LightCurve(source=source, release="dr24", band="g",
                          value_kind="mag",
                          time=2458000.0 + np.arange(30, dtype=np.float64) * 0.5,
                          value=np.full(30, 18.0), value_err=np.full(30, 0.02))

    def test_matched_row_gets_gaia_columns(self, isolated_root):
        from astra import metadata

        curve = self._ztf_curve("obj1", ra_deg=180.0, dec_deg=22.0)
        store.write_curve(curve)
        metadata.upsert_sources(isolated_root.projects, [{
            "source_key": "Gaia/dr3/1", "survey": "Gaia", "release": "dr3",
            "object_id": "1", "ra_deg": 180.0, "dec_deg": 22.0,
            # 5 mas parallax -> 200 pc by the standard 1000/parallax relation.
            "extra": {"parallax": 5.0, "parallax_error": 0.1,
                     "pmra": 12.0, "pmdec": -3.0,
                     "phot_g_mean_mag": 15.0, "phot_bp_mean_mag": 15.4,
                     "phot_rp_mean_mag": 14.5},
        }])

        matrix = featurematrix.build(survey="ZTF")
        joined, diagnostics = featurematrix.join_gaia_columns(matrix)

        assert len(joined) == len(matrix)  # column join: row count unchanged
        assert joined.feature_names[:len(matrix.feature_names)] == matrix.feature_names
        assert diagnostics == {"matched": 1, "total": 1, "match_rate": 1.0}
        assert joined.column("gaia_matched")[0] == 1.0
        assert joined.column("gaia_parallax")[0] == pytest.approx(5.0)
        assert joined.column("gaia_distance_pc")[0] == pytest.approx(200.0)
        assert joined.column("gaia_bp_rp")[0] == pytest.approx(0.9, abs=1e-6)
        # Propagated to the current epoch from Gaia's fixed J2016.0 reference
        # -- with nonzero proper motion and roughly a decade elapsed, the
        # current-epoch position must differ from the stored J2016.0 one.
        assert np.isfinite(joined.column("gaia_ra_now_deg")[0])
        assert np.isfinite(joined.column("gaia_dec_now_deg")[0])
        # abs tolerance, not the default relative one: at dec=22 degrees, a
        # relative tolerance would swallow a shift of several arcminutes.
        assert joined.column("gaia_dec_now_deg")[0] != pytest.approx(22.0, abs=1e-9)

    def test_unmatched_row_gets_nan_not_an_imputed_value(self, isolated_root):
        from astra import metadata

        curve = self._ztf_curve("obj1", ra_deg=180.0, dec_deg=22.0)
        store.write_curve(curve)
        # 1 degree away: nowhere near the default 2 arcsec match radius.
        metadata.upsert_sources(isolated_root.projects, [{
            "source_key": "Gaia/dr3/far", "survey": "Gaia", "release": "dr3",
            "object_id": "far", "ra_deg": 181.0, "dec_deg": 22.0,
            "extra": {"parallax": 5.0},
        }])

        matrix = featurematrix.build(survey="ZTF")
        joined, diagnostics = featurematrix.join_gaia_columns(matrix)

        assert diagnostics == {"matched": 0, "total": 1, "match_rate": 0.0}
        assert joined.column("gaia_matched")[0] == 0.0
        assert np.isnan(joined.column("gaia_parallax")[0])
        # An unmatched row fails finite_mask like any other incomplete
        # feature row, so detection silently excludes it rather than
        # training on an invented value.
        assert not joined.finite_mask()[0]

    def test_no_gaia_data_leaves_every_row_nan(self, isolated_root):
        curve = self._ztf_curve("obj1", ra_deg=180.0, dec_deg=22.0)
        store.write_curve(curve)

        matrix = featurematrix.build(survey="ZTF")
        joined, diagnostics = featurematrix.join_gaia_columns(matrix)

        assert diagnostics == {"matched": 0, "total": 1, "match_rate": 0.0}
        assert len(joined) == 1
        assert np.all(np.isnan(joined.values[:, len(matrix.feature_names):-1]))

    def test_empty_matrix_still_gets_the_joined_columns(self, isolated_root):
        empty = FeatureMatrix(values=np.empty((0, len(FEATURE_NAMES))),
                              identities=[])
        joined, diagnostics = featurematrix.join_gaia_columns(empty)

        assert len(joined) == 0
        assert joined.feature_names == FEATURE_NAMES + featurematrix.GAIA_JOIN_COLUMNS
        assert diagnostics == {"matched": 0, "total": 0, "match_rate": None}

    def test_listing_of_empty_root(self, tmp_path):
        assert featurematrix.list_matrices(tmp_path) == []


class TestStellarManifoldJoin:
    """stellar_manifold's join is layered on top of join_gaia_columns: it
    only ever derives physics from columns that join already produced, so
    it requires that join to have run first."""

    def _ztf_curve(self, object_id: str, ra_deg: float, dec_deg: float) -> LightCurve:
        source = SourceRef(survey="ZTF", object_id=object_id,
                           ra_deg=ra_deg, dec_deg=dec_deg)
        return LightCurve(source=source, release="dr24", band="g",
                          value_kind="mag",
                          time=2458000.0 + np.arange(30, dtype=np.float64) * 0.5,
                          value=np.full(30, 18.0), value_err=np.full(30, 0.02))

    def test_requires_gaia_join_to_run_first(self):
        matrix = FeatureMatrix(values=np.empty((0, len(FEATURE_NAMES))), identities=[])
        with pytest.raises(ValueError, match="join_gaia_columns"):
            featurematrix.join_stellar_manifold_columns(matrix)

    def test_matched_row_gets_manifold_columns(self, isolated_root):
        from astra import metadata

        curve = self._ztf_curve("obj1", ra_deg=180.0, dec_deg=22.0)
        store.write_curve(curve)
        metadata.upsert_sources(isolated_root.projects, [{
            "source_key": "Gaia/dr3/1", "survey": "Gaia", "release": "dr3",
            "object_id": "1", "ra_deg": 180.0, "dec_deg": 22.0,
            # phot_bp_mean_mag - phot_rp_mean_mag = 0.85, a real G5V anchor.
            "extra": {"parallax": 5.0, "parallax_error": 0.1,
                     "phot_g_mean_mag": 4.801 + 5.0 * (np.log10(200.0) - 1.0),
                     "phot_bp_mean_mag": 15.85, "phot_rp_mean_mag": 15.0},
        }])

        matrix = featurematrix.build(survey="ZTF")
        gaia_joined, _ = featurematrix.join_gaia_columns(matrix)
        joined, diagnostics = featurematrix.join_stellar_manifold_columns(gaia_joined)

        assert len(joined) == len(matrix)  # column join: row count unchanged
        assert diagnostics == {"matched": 1, "total": 1, "match_rate": 1.0}
        assert joined.column("manifold_matched")[0] == 1.0
        assert joined.column("manifold_residual_mag")[0] == pytest.approx(0.0, abs=1e-3)
        assert joined.column("manifold_teff_k")[0] == pytest.approx(5660.0, abs=1.0)

    def test_unmatched_gaia_row_gets_nan_manifold_columns(self, isolated_root):
        curve = self._ztf_curve("obj1", ra_deg=180.0, dec_deg=22.0)
        store.write_curve(curve)

        matrix = featurematrix.build(survey="ZTF")
        gaia_joined, _ = featurematrix.join_gaia_columns(matrix)
        joined, diagnostics = featurematrix.join_stellar_manifold_columns(gaia_joined)

        assert diagnostics == {"matched": 0, "total": 1, "match_rate": 0.0}
        assert joined.column("manifold_matched")[0] == 0.0
        assert np.isnan(joined.column("manifold_residual_mag")[0])


class TestPrepare:
    def test_rows_with_nan_features_are_excluded(self):
        matrix = synthetic_matrix(n_normal=20, n_outliers=0)
        matrix.values[3, 0] = np.nan

        x, identities, skipped = anomaly.prepare(matrix)

        assert skipped == 1
        assert x.shape[0] == 19
        assert len(identities) == 19

    def test_standardisation_removes_scale_differences(self):
        """Without this, the largest-magnitude column dominates every distance."""
        matrix = synthetic_matrix(n_normal=100, n_outliers=0)
        matrix.values[:, 0] *= 10_000.0

        x, _, _ = anomaly.prepare(matrix)

        assert np.allclose(np.mean(x, axis=0), 0.0, atol=1e-6)
        assert np.allclose(np.std(x, axis=0), 1.0, atol=1e-6)

    def test_empirical_calibration_is_monotonic_and_bounded(self):
        values = anomaly.calibrate_scores(np.array([0.0, 1.0, 2.0, 2.0]))
        assert np.all((values >= 0.0) & (values <= 1.0))
        assert np.all(np.diff(values) >= 0)
        assert anomaly.calibration_report(values)["reference_rows"] == 4


class TestDetectors:
    @pytest.mark.parametrize("runner", [
        anomaly.run_isolation_forest,
        anomaly.run_lof,
        anomaly.run_one_class_svm,
        anomaly.run_pca_reconstruction,
    ])
    def test_each_detector_finds_most_planted_outliers(self, runner):
        """Individually, each detector should surface most of them.

        The bar is the top 10 of 205 rather than the top 5, because these
        methods fail differently — One-Class SVM in particular is sensitive to
        its kernel width. Demanding a perfect top 5 from every detector would
        be testing luck. The ensemble is held to the stricter standard below,
        which is the reason for running four detectors instead of one.
        """
        matrix = synthetic_matrix(n_normal=200, n_outliers=5)
        x, _, _ = anomaly.prepare(matrix)

        try:
            scores = runner(x, 0.05).scores
        except TypeError:
            scores = runner(x, 0.05, 42).scores

        # The five planted outliers are the last five rows.
        top_ten = set(np.argsort(-scores)[:10])
        assert len(top_ten & {200, 201, 202, 203, 204}) >= 4

    def test_kernel_width_avoids_saturation(self):
        """gamma='scale' collapses far-outlier scores to a constant.

        Regression test for a real defect: with sklearn's default the five
        planted outliers all scored exactly 0.0 and could not be ranked.
        """
        from sklearn.svm import OneClassSVM

        matrix = synthetic_matrix(n_normal=200, n_outliers=5)
        x, _, _ = anomaly.prepare(matrix)

        default = -OneClassSVM(nu=0.05, gamma="scale").fit(x).decision_function(x)
        tuned = anomaly.run_one_class_svm(x, 0.05).scores
        planted = {200, 201, 202, 203, 204}

        # How much ordering information survives among the outliers, as a
        # fraction of each detector's own score range.
        default_resolution = float(np.ptp(default[200:]) / np.ptp(default))
        tuned_resolution = float(np.ptp(tuned[200:]) / np.ptp(tuned))

        # Measured on this fixture: ~2.8e-3 default against ~1.8e-2 tuned.
        assert tuned_resolution > 3 * default_resolution
        assert min(tuned[200:]) > float(np.percentile(tuned[:200], 99))

        # The consequence that matters for ranking: the default surfaces 2 of
        # the 5 planted outliers in its top 10, the median heuristic finds all.
        default_hits = len(set(np.argsort(-default)[:10]) & planted)
        tuned_hits = len(set(np.argsort(-tuned)[:10]) & planted)
        assert tuned_hits == 5
        assert tuned_hits > default_hits

    def test_scores_are_normalised_to_unit_range(self):
        matrix = synthetic_matrix()
        x, _, _ = anomaly.prepare(matrix)
        scores = anomaly.run_isolation_forest(x, 0.05).scores

        assert scores.min() == pytest.approx(0.0)
        assert scores.max() == pytest.approx(1.0)


class TestRankConsensus:
    """Consensus averages per-detector RANKS, not raw normalised scores --
    see the anomaly.py module docstring for the measured reason (an
    equal-weight mean of scores scored WORSE than Isolation Forest alone on
    301 real ZTF sequences, docs/LIMITATIONS.md Phase 8). These test the
    mechanics of _rank_consensus directly; the actual claim that this beats
    a plain mean on real data is verified separately by re-running
    ablation.detector_ablation against the live store, not by a
    hand-constructed array here."""

    def test_higher_score_gets_higher_consensus(self):
        stacked = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]])
        consensus = anomaly._rank_consensus(stacked)

        assert consensus[2] > consensus[1] > consensus[0]

    def test_output_is_bounded_in_unit_interval(self):
        rng = np.random.default_rng(0)
        stacked = rng.normal(size=(4, 50))
        consensus = anomaly._rank_consensus(stacked)

        assert consensus.min() > 0.0
        assert consensus.max() <= 1.0

    def test_ties_share_the_average_rank(self):
        """LOF and PCA-reconstruction error both tie exactly on small or
        degenerate batches; a plain argsort would give ties arbitrary
        distinct positions instead of sharing one."""
        stacked = np.array([[0.5, 0.5, 0.5, 1.0]])
        consensus = anomaly._rank_consensus(stacked)

        assert consensus[0] == consensus[1] == consensus[2]
        assert consensus[3] > consensus[0]

    def test_agrees_with_a_single_detector_ranked_directly(self):
        """With one detector, consensus must reduce to that detector's own
        rank order -- there is nothing else to average against."""
        rng = np.random.default_rng(1)
        scores = rng.normal(size=30)
        consensus = anomaly._rank_consensus(scores[None, :])

        assert np.argsort(consensus).tolist() == np.argsort(scores).tolist()

    def test_empty_input_returns_empty(self):
        assert anomaly._rank_consensus(np.empty((3, 0))).size == 0


class TestEnsemble:
    def test_all_four_detectors_run(self):
        result = anomaly.detect(synthetic_matrix())
        assert set(result.detectors) == set(anomaly.DETECTOR_NAMES)

    def test_consensus_ranks_the_planted_outliers_first(self):
        result = anomaly.detect(synthetic_matrix(n_normal=200, n_outliers=5))
        top = result.ranked(top=5)

        assert all(entry["object_id"].startswith("OUT") for entry in top)

    def test_model_agreement_is_recorded(self):
        """Plan section 16 gives 10% of the score to detector agreement."""
        result = anomaly.detect(synthetic_matrix(n_normal=200, n_outliers=5))
        top = result.ranked(top=5)

        assert all(entry["model_agreement"] >= 3 for entry in top)
        assert all(0 <= entry["model_agreement"] <= 4 for entry in result.ranked(50))

    def test_per_detector_scores_are_exposed(self):
        """Explainability: the evidence must survive into the ranking."""
        entry = anomaly.detect(synthetic_matrix()).ranked(top=1)[0]
        for name in anomaly.DETECTOR_NAMES:
            assert f"score_{name}" in entry

    def test_too_few_rows_returns_an_empty_result(self):
        result = anomaly.detect(synthetic_matrix(n_normal=5, n_outliers=0))
        assert result.detectors == {}
        assert result.ranked() == []

    def test_results_are_reproducible(self):
        """Plan section 37: the same input and seed must give the same output."""
        first = anomaly.detect(synthetic_matrix(seed=3), seed=7)
        second = anomaly.detect(synthetic_matrix(seed=3), seed=7)

        np.testing.assert_allclose(first.consensus, second.consensus)

    def test_ranking_is_saved_as_json(self, tmp_path):
        import json

        result = anomaly.detect(synthetic_matrix())
        path = anomaly.save_ranking(result, "run1", top=10, root=tmp_path)
        payload = json.loads(path.read_text())

        assert payload["summary"]["rows_scored"] == 205
        assert len(payload["candidates"]) == 10

    def test_summary_counts_skipped_rows(self):
        matrix = synthetic_matrix(n_normal=50, n_outliers=2)
        matrix.values[0, 0] = np.nan

        assert anomaly.detect(matrix).to_dict()["rows_skipped"] == 1


class TestCalibrationReferencePersistence:
    """Cross-run comparability (docs/LIMITATIONS.md) needs a reference that
    survives past a single `detect` call. These test the persistence layer
    directly; TestCrossRunCalibration below tests it end-to-end through
    EnsembleResult."""

    def test_missing_reference_loads_as_empty(self, tmp_path):
        loaded = anomaly.load_calibration_reference("nope", tmp_path)
        assert loaded.size == 0

    def test_round_trip(self, tmp_path):
        anomaly.update_calibration_reference("run1", tmp_path, np.array([0.1, 0.5, 0.9]))
        loaded = anomaly.load_calibration_reference("run1", tmp_path)

        np.testing.assert_allclose(np.sort(loaded), [0.1, 0.5, 0.9])

    def test_successive_updates_accumulate(self, tmp_path):
        anomaly.update_calibration_reference("run1", tmp_path, np.array([0.1, 0.2]))
        anomaly.update_calibration_reference("run1", tmp_path, np.array([0.3, 0.4]))
        loaded = anomaly.load_calibration_reference("run1", tmp_path)

        np.testing.assert_allclose(np.sort(loaded), [0.1, 0.2, 0.3, 0.4])

    def test_cap_evicts_oldest_first(self, tmp_path):
        anomaly.update_calibration_reference("run1", tmp_path, np.array([1.0, 2.0]), cap=3)
        anomaly.update_calibration_reference("run1", tmp_path, np.array([3.0, 4.0]), cap=3)
        loaded = anomaly.load_calibration_reference("run1", tmp_path)

        # Oldest (1.0) evicted first; the three most recent survive, in order.
        np.testing.assert_allclose(loaded, [2.0, 3.0, 4.0])

    def test_different_names_do_not_share_a_reference(self, tmp_path):
        anomaly.update_calibration_reference("a", tmp_path, np.array([1.0]))
        anomaly.update_calibration_reference("b", tmp_path, np.array([2.0]))

        np.testing.assert_allclose(anomaly.load_calibration_reference("a", tmp_path), [1.0])
        np.testing.assert_allclose(anomaly.load_calibration_reference("b", tmp_path), [2.0])


class TestCrossRunCalibration:
    """The actual comparability claim: two 'runs' whose raw consensus scores
    live on different scales should land on comparable calibrated values once
    both are calibrated against the same persisted reference."""

    def test_calibrated_scores_are_comparable_across_runs(self, tmp_path):
        rng = np.random.default_rng(0)
        # Same underlying distribution, but run 2's raw scores are shifted
        # and rescaled -- the kind of run-to-run drift the raw, batch-relative
        # min-max score cannot see past.
        base = rng.uniform(0.0, 1.0, size=500)
        run1 = anomaly.EnsembleResult(identities=[{} for _ in base], consensus=base,
                                      agreement=np.zeros(len(base), dtype=int))
        run2_raw = base * 3.0 + 10.0
        run2 = anomaly.EnsembleResult(identities=[{} for _ in base], consensus=run2_raw,
                                      agreement=np.zeros(len(base), dtype=int))

        name = "cross-run-test"
        ref = anomaly.load_calibration_reference(name, tmp_path)
        ref = ref if ref.size else None
        calibrated1 = run1.ranked(top=len(base), reference=ref)
        anomaly.update_calibration_reference(name, tmp_path, run1.consensus)

        ref = anomaly.load_calibration_reference(name, tmp_path)
        calibrated2 = run2.ranked(top=len(base), reference=ref)

        by_id1 = {r["rank"]: r["consensus_calibrated"] for r in calibrated1}
        by_id2 = {r["rank"]: r["consensus_calibrated"] for r in calibrated2}
        # Same rank position (i.e. same position in the shared underlying
        # distribution) should land on a near-identical calibrated value,
        # even though raw consensus scores differ by an order of magnitude.
        for rank in list(by_id1)[:10]:
            assert by_id1[rank] == pytest.approx(by_id2[rank], abs=0.05)

    def test_first_run_falls_back_to_batch_relative(self, tmp_path):
        result = anomaly.detect(synthetic_matrix())
        reference = anomaly.load_calibration_reference("first-run", tmp_path)
        reference = reference if reference.size else None

        assert reference is None
        report = result.to_dict(reference=reference)["calibration"]
        assert report["reference_external"] is False
