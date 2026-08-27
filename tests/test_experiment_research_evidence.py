"""v2 provenance fields: benchmark/split/manifest binding, `complete()`/
`require_complete()`, record-hash tamper detection, and that a v1 record on
disk still loads (migration by dataclass default, not a rewrite)."""

from __future__ import annotations

import json

import pytest

from astra import experiment


def _complete_record(tmp_path):
    record = experiment.create(
        "benchmark", {}, root=tmp_path,
        benchmark_id="bench-1", split_id="core_object_split",
        manifest_content_hash="deadbeef", label_set_hash="cafef00d",
        checkpoint_path=None, result_artifact_paths=["research/results/metrics.jsonl"],
    )
    # model_version is None without a checkpoint; complete() requires it, so
    # a benchmark experiment records a real (if synthetic-for-the-test) one.
    record.provenance.model_version = "abc123"
    return record


class TestCompleteness:
    def test_exploratory_experiment_is_incomplete(self, tmp_path):
        record = experiment.create("detection", {}, root=tmp_path)
        assert not record.complete()

    def test_fully_bound_experiment_is_complete(self, tmp_path):
        record = _complete_record(tmp_path)
        assert record.complete()

    def test_require_complete_raises_for_incomplete(self, tmp_path):
        record = experiment.create("detection", {}, root=tmp_path)
        with pytest.raises(experiment.IncompleteExperimentError):
            experiment.require_complete(record)

    def test_require_complete_passes_for_complete(self, tmp_path):
        record = _complete_record(tmp_path)
        experiment.require_complete(record)  # must not raise

    def test_missing_split_id_alone_is_incomplete(self, tmp_path):
        record = _complete_record(tmp_path)
        record.provenance.split_id = None
        assert not record.complete()


class TestRecordHash:
    def test_saved_record_verifies_clean(self, tmp_path):
        record = experiment.run("study", {}, lambda: {"m": 1.0}, root=tmp_path)
        report = experiment.verify_record_hash(record.provenance.experiment_id, tmp_path)
        assert report["matches"]

    def test_tampered_record_fails_verification(self, tmp_path):
        record = experiment.run("study", {}, lambda: {"m": 1.0}, root=tmp_path)
        path = experiment.experiment_path(record.provenance.experiment_id, tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["results"]["m"] = 999.0
        path.write_text(json.dumps(payload), encoding="utf-8")

        report = experiment.verify_record_hash(record.provenance.experiment_id, tmp_path)
        assert not report["matches"]


class TestV1Migration:
    def test_v1_record_without_research_fields_still_loads(self, tmp_path):
        record = experiment.create("detection", {}, root=tmp_path)
        payload = record.to_dict()
        payload["schema_version"] = 1
        # Simulate a genuine pre-v2 record: strip the v2 provenance keys
        # entirely, as they would never have been written by v1 code.
        for key in ("benchmark_id", "split_id", "manifest_content_hash",
                   "label_set_hash", "result_artifact_paths"):
            payload["provenance"].pop(key, None)
        path = experiment.experiment_path(record.provenance.experiment_id, tmp_path)
        path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = experiment.load(record.provenance.experiment_id, tmp_path)
        assert loaded.provenance.benchmark_id is None
        assert loaded.provenance.result_artifact_paths == []
        assert not loaded.complete()
