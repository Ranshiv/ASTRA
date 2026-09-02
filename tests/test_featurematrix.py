"""featurematrix.py's own contract: coverage tiers, FeatureMatrix helpers,
save/load round-tripping (including its schema-version/hash guards), and
matrix listing. `build`/`build_resumable` themselves are already exercised
via test_anomaly.py/test_ablation.py/etc.; this covers what those callers
don't: the module's own persistence and bookkeeping surface."""

from __future__ import annotations

import numpy as np
import pytest

from astra import featurematrix, features, store
from astra.featurematrix import FeatureMatrix
from astra.features import FEATURE_NAMES


class TestCoverageTier:
    def test_enough_points_for_a_period_search_is_tier_a(self):
        assert featurematrix.coverage_tier(features.MIN_POINTS_FOR_PERIOD) == "A"

    def test_enough_for_non_periodic_features_only_is_tier_b(self):
        assert featurematrix.coverage_tier(features.MIN_POINTS) == "B"
        assert featurematrix.coverage_tier(features.MIN_POINTS_FOR_PERIOD - 1) == "B"

    def test_too_few_points_is_tier_c(self):
        assert featurematrix.coverage_tier(features.MIN_POINTS - 1) == "C"
        assert featurematrix.coverage_tier(0) == "C"


class TestBatchReport:
    def test_to_dict_reports_every_field(self):
        report = featurematrix.BatchReport(
            checkpoint="ckpt.json", source_count=10, completed=8,
            failed=2, resumed=True, batches=3)
        assert report.to_dict() == {
            "checkpoint": "ckpt.json", "source_count": 10, "completed": 8,
            "failed": 2, "resumed": True, "batches": 3,
        }


def _matrix(n_rows: int = 2) -> FeatureMatrix:
    values = np.zeros((n_rows, len(FEATURE_NAMES)))
    identities = [
        {"object_id": f"obj{i}", "survey": "ZTF", "release": "dr24",
         "band": "g", "coverage_tier": "A", "path": f"/tmp/obj{i}.parquet"}
        for i in range(n_rows)
    ]
    return FeatureMatrix(values=values, identities=identities)


class TestFeatureMatrixHelpers:
    def test_len_matches_identity_count(self):
        assert len(_matrix(3)) == 3

    def test_shape_matches_the_values_array(self):
        matrix = _matrix(2)
        assert matrix.shape == (2, len(FEATURE_NAMES))

    def test_column_looks_up_by_feature_name(self):
        matrix = _matrix(2)
        matrix.values[:, 0] = [1.0, 2.0]
        assert list(matrix.column(FEATURE_NAMES[0])) == [1.0, 2.0]

    def test_finite_mask_excludes_a_row_with_any_nan(self):
        matrix = _matrix(2)
        matrix.values[1, 0] = float("nan")
        mask = matrix.finite_mask()
        assert mask[0] and not mask[1]

    def test_subset_selects_rows_and_preserves_identity(self):
        matrix = _matrix(3)
        matrix.values[:, 0] = [10.0, 20.0, 30.0]
        subset = matrix.subset([0, 2])
        assert len(subset) == 2
        assert list(subset.column(FEATURE_NAMES[0])) == [10.0, 30.0]
        assert subset.identities[1]["object_id"] == "obj2"

    def test_subset_of_empty_rows_yields_a_zero_row_matrix(self):
        matrix = _matrix(2)
        subset = matrix.subset([])
        assert len(subset) == 0
        assert subset.shape == (0, len(FEATURE_NAMES))

    def test_subset_can_restrict_feature_names(self):
        matrix = _matrix(2)
        names = FEATURE_NAMES[:2]
        subset = matrix.subset([0, 1], feature_names=names)
        assert subset.feature_names == names
        assert subset.shape == (2, 2)

    def test_to_dict_reports_shape_and_usable_rows(self):
        matrix = _matrix(2)
        matrix.values[1, 0] = float("nan")
        payload = matrix.to_dict()
        assert payload["rows"] == 2
        assert payload["features"] == len(FEATURE_NAMES)
        assert payload["usable_rows"] == 1
        assert payload["feature_names"] == list(FEATURE_NAMES)


class TestSaveAndLoad:
    def test_round_trips_values_and_identities(self, tmp_path):
        matrix = _matrix(2)
        matrix.values[:, 0] = [1.5, 2.5]
        featurematrix.save(matrix, "myrun", tmp_path)
        loaded = featurematrix.load("myrun", tmp_path)

        assert len(loaded) == 2
        assert loaded.identities[0]["object_id"] == "obj0"
        assert loaded.identities[0]["survey"] == "ZTF"
        assert loaded.column(FEATURE_NAMES[0])[0] == pytest.approx(1.5)

    def test_round_trips_an_empty_matrix(self, tmp_path):
        matrix = _matrix(0)
        featurematrix.save(matrix, "empty", tmp_path)
        loaded = featurematrix.load("empty", tmp_path)
        assert len(loaded) == 0
        assert loaded.shape == (0, len(FEATURE_NAMES))

    def test_matrix_path_embeds_the_feature_version(self, tmp_path):
        path = featurematrix.matrix_path("myrun", tmp_path)
        assert path.name == f"myrun_v{featurematrix.FEATURE_VERSION}.parquet"

    def test_load_rejects_a_stale_feature_version(self, tmp_path):
        # matrix_path() itself embeds FEATURE_VERSION in the filename, so a
        # real version bump changes which file load() even looks for -- the
        # only way a version mismatch reaches load()'s own check is a file
        # at the CURRENT path whose recorded metadata says otherwise, which
        # is what's built here directly rather than through save().
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = featurematrix.matrix_path("old", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {name: pa.array([0.0], type=pa.float64()) for name in FEATURE_NAMES},
            metadata={b"feature_version": str(featurematrix.FEATURE_VERSION + 1).encode()},
        )
        pq.write_table(table, path)

        with pytest.raises(ValueError, match="feature version"):
            featurematrix.load("old", tmp_path)

    def test_load_rejects_a_changed_feature_schema_hash(self, tmp_path, monkeypatch):
        matrix = _matrix(1)
        featurematrix.save(matrix, "old", tmp_path)
        monkeypatch.setattr(featurematrix, "schema_hash", lambda: "a-different-hash")
        with pytest.raises(ValueError, match="different feature schema"):
            featurematrix.load("old", tmp_path)

    def test_load_fills_default_identity_fields_when_absent(self, tmp_path):
        # identities missing "release"/"coverage_tier" (as an older matrix
        # written before those columns existed would be) still load, with
        # save()'s own stated defaults.
        values = np.zeros((1, len(FEATURE_NAMES)))
        matrix = FeatureMatrix(values=values, identities=[
            {"object_id": "obj0", "survey": "ZTF", "band": "g", "path": "/tmp/x"}])
        featurematrix.save(matrix, "sparse", tmp_path)
        loaded = featurematrix.load("sparse", tmp_path)
        assert loaded.identities[0]["release"] == "unknown"
        assert loaded.identities[0]["coverage_tier"] == "A"


class TestIdentityFromPath:
    def test_recovers_identity_from_a_real_stored_curve(self, tmp_path, curve):
        result = store.write_curve(curve, tmp_path)
        identity = featurematrix._identity_from_path(result.path)
        assert identity["object_id"] == curve.source.object_id
        assert identity["survey"] == curve.source.survey
        assert identity["path"] == str(result.path)

    def test_a_missing_file_yields_a_placeholder_identity_not_a_crash(self, tmp_path):
        identity = featurematrix._identity_from_path(tmp_path / "does-not-exist.parquet")
        assert identity["object_id"] == "unknown"
        assert identity["coverage_tier"] == "C"


class TestListMatrices:
    def test_missing_features_directory_yields_an_empty_list(self, tmp_path):
        assert featurematrix.list_matrices(tmp_path / "nothing") == []

    def test_lists_a_saved_matrix_with_its_row_count(self, tmp_path):
        featurematrix.save(_matrix(3), "myrun", tmp_path)
        listing = featurematrix.list_matrices(tmp_path)
        assert len(listing) == 1
        assert listing[0]["name"] == f"myrun_v{featurematrix.FEATURE_VERSION}"
        assert listing[0]["rows"] == 3

    def test_a_corrupt_matrix_file_is_skipped_not_fatal(self, tmp_path):
        featurematrix.save(_matrix(1), "good", tmp_path)
        bad = tmp_path / "features" / "bad.parquet"
        bad.write_bytes(b"not a real parquet file")
        listing = featurematrix.list_matrices(tmp_path)
        assert [entry["name"] for entry in listing] == [f"good_v{featurematrix.FEATURE_VERSION}"]
