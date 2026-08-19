"""Experiment provenance, reproducibility checking and comparison."""

from __future__ import annotations

import json

import pytest

from astra import experiment


class TestCodeVersion:
    def test_is_stable_across_calls(self):
        assert experiment.code_version() == experiment.code_version()

    def test_changes_when_source_changes(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        first = experiment.code_version(tmp_path)

        (tmp_path / "a.py").write_text("x = 2")
        second = experiment.code_version(tmp_path)

        assert first != second

    def test_ignores_pycache(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        before = experiment.code_version(tmp_path)

        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "a.py").write_text("compiled junk")

        assert experiment.code_version(tmp_path) == before

    def test_is_order_independent_of_filesystem(self, tmp_path):
        (tmp_path / "b.py").write_text("b = 1")
        (tmp_path / "a.py").write_text("a = 1")
        first = experiment.code_version(tmp_path)

        (tmp_path / "a.py").write_text("a = 1")  # touch, same content
        assert experiment.code_version(tmp_path) == first

    def test_git_revision_is_optional(self, tmp_path):
        assert experiment.code_revision(tmp_path) is None


class TestModelVersion:
    def test_none_for_missing_checkpoint(self):
        assert experiment.model_version(None) is None
        assert experiment.model_version("does/not/exist.pt") is None

    def test_hashes_checkpoint_content(self, tmp_path):
        path = tmp_path / "m.pt"
        path.write_bytes(b"weights-v1")
        first = experiment.model_version(path)

        path.write_bytes(b"weights-v2")
        assert experiment.model_version(path) != first


class TestRecording:
    def test_created_record_captures_section_19_fields(self, tmp_path):
        record = experiment.create("detection", {"contamination": 0.05},
                                   root=tmp_path)
        provenance = record.provenance

        assert provenance.experiment_id.startswith("EXP-")
        assert provenance.code_version
        assert provenance.feature_version >= 1
        assert provenance.preprocessing_version >= 1
        assert provenance.seed == 42
        assert provenance.hardware["device"] in {"cpu", "cuda"}
        assert "python" in provenance.environment
        assert provenance.preprocessing_schema_hash == experiment.preprocessing_schema_hash()

    def test_preprocessing_contract_is_json_safe_and_hashed(self):
        contract = experiment.preprocessing_contract()
        assert contract["version"] == experiment.PREPROCESSING_VERSION
        assert contract["schema_hash"] == experiment.preprocessing_schema_hash()
        assert contract["contract"]["time"]["frame"] == "BJD_TDB"

    def test_preprocessing_contract_change_changes_hash(self, monkeypatch):
        original = experiment.PREPROCESSING_CONTRACT["time"]["encoding"]
        monkeypatch.setitem(experiment.PREPROCESSING_CONTRACT["time"], "encoding", "absolute_bjd")
        changed = experiment.preprocessing_schema_hash()
        monkeypatch.setitem(experiment.PREPROCESSING_CONTRACT["time"], "encoding", original)
        assert changed != experiment.preprocessing_schema_hash()

    def test_ids_increment(self, tmp_path):
        first = experiment.create("a", {}, root=tmp_path)
        experiment.save(first, tmp_path)
        second = experiment.create("b", {}, root=tmp_path)

        assert second.provenance.experiment_id != first.provenance.experiment_id
        assert second.provenance.experiment_id == "EXP-0002"

    def test_run_records_results_and_runtime(self, tmp_path):
        record = experiment.run("study", {"n": 3},
                                lambda: {"roc_auc": 0.8}, root=tmp_path)

        assert record.results["roc_auc"] == 0.8
        assert record.runtime_seconds >= 0.0

    def test_failed_run_is_still_recorded(self, tmp_path):
        """An experiment that crashed is a result; losing it repeats the error."""
        def boom():
            raise RuntimeError("detector exploded")

        with pytest.raises(RuntimeError):
            experiment.run("study", {}, boom, root=tmp_path)

        listing = experiment.list_experiments(tmp_path)
        assert len(listing) == 1
        assert listing[0]["failed"] is True

    def test_save_and_load_round_trip(self, tmp_path):
        record = experiment.run("study", {"a": 1},
                                lambda: {"metric": 0.5}, root=tmp_path)
        loaded = experiment.load(record.provenance.experiment_id, tmp_path)

        assert loaded.configuration == {"a": 1}
        assert loaded.results == {"metric": 0.5}
        assert loaded.provenance.code_version == record.provenance.code_version

    def test_record_is_json_safe(self, tmp_path):
        record = experiment.run("study", {}, lambda: {"m": 1.0}, root=tmp_path)
        path = experiment.experiment_path(record.provenance.experiment_id,
                                          tmp_path)
        json.loads(path.read_text())

    def test_listing_of_empty_root(self, tmp_path):
        assert experiment.list_experiments(tmp_path) == []


class TestVerify:
    def test_fresh_experiment_is_reproducible(self, tmp_path):
        record = experiment.run("study", {}, lambda: {"m": 1.0}, root=tmp_path)
        report = experiment.verify(record.provenance.experiment_id, tmp_path)

        assert report["reproducible"] is True
        assert report["drift"] == {}
        assert "reproduce" in report["note"]

    def test_feature_version_drift_is_detected(self, tmp_path, monkeypatch):
        record = experiment.run("study", {}, lambda: {"m": 1.0}, root=tmp_path)

        from astra import features as features_mod
        monkeypatch.setattr(features_mod, "FEATURE_VERSION", 99)

        report = experiment.verify(record.provenance.experiment_id, tmp_path)

        assert report["reproducible"] is False
        assert report["drift"]["feature_version"]["current"] == 99

    def test_library_drift_is_detected(self, tmp_path):
        record = experiment.run("study", {}, lambda: {"m": 1.0}, root=tmp_path)
        record.provenance.environment["numpy"] = "0.0.1-fake"
        experiment.save(record, tmp_path)

        report = experiment.verify(record.provenance.experiment_id, tmp_path)

        assert report["reproducible"] is False
        assert "numpy" in report["drift"]["environment"]

    def test_seed_is_reported_for_rerunning(self, tmp_path):
        record = experiment.run("study", {}, lambda: {"m": 1.0},
                                seed=1234, root=tmp_path)
        assert experiment.verify(record.provenance.experiment_id,
                                 tmp_path)["seed"] == 1234


class TestCompare:
    def test_compare_ranks_by_metric(self, tmp_path):
        low = experiment.run("a", {}, lambda: {"roc_auc": 0.6}, root=tmp_path)
        high = experiment.run("b", {}, lambda: {"roc_auc": 0.9}, root=tmp_path)

        result = experiment.compare([low.provenance.experiment_id,
                                     high.provenance.experiment_id],
                                    "roc_auc", tmp_path)

        assert result["best"]["experiment_id"] == high.provenance.experiment_id
        assert result["comparable"] is True

    def test_incomparable_feature_versions_are_flagged(self, tmp_path):
        """A metric from different inputs is not the same metric."""
        first = experiment.run("a", {}, lambda: {"roc_auc": 0.6}, root=tmp_path)
        second = experiment.run("b", {}, lambda: {"roc_auc": 0.9}, root=tmp_path)

        second.provenance.feature_version = 99
        experiment.save(second, tmp_path)

        result = experiment.compare([first.provenance.experiment_id,
                                     second.provenance.experiment_id],
                                    "roc_auc", tmp_path)

        assert result["comparable"] is False
        assert "not directly comparable" in result["warning"]

    def test_missing_metric_is_tolerated(self, tmp_path):
        record = experiment.run("a", {}, lambda: {"other": 1}, root=tmp_path)
        result = experiment.compare([record.provenance.experiment_id],
                                    "roc_auc", tmp_path)

        assert result["best"] is None
        assert result["rows"][0]["value"] is None

    def test_unknown_experiment_is_skipped(self, tmp_path):
        result = experiment.compare(["EXP-9999"], "roc_auc", tmp_path)
        assert result["rows"] == []
