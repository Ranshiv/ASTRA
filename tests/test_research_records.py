"""LabelRecord/BenchmarkSpec/ResultRecord content-hash stability, and that
`DatasetManifest` really is `manifest.Manifest` re-exported unchanged."""

from __future__ import annotations

from astra.manifest import Manifest
from astra.research.records import (
    BenchmarkSpec, DatasetManifest, LabelRecord, ResultRecord,
)


def test_dataset_manifest_is_manifest_reexported():
    assert DatasetManifest is Manifest


def test_label_record_hash_ignores_timestamp():
    a = LabelRecord(object_id="ZTF1", label="SN Ia", label_source="TNS",
                    source_release="2024a", confidence=0.9,
                    created_utc="2024-01-01T00:00:00")
    b = LabelRecord(object_id="ZTF1", label="SN Ia", label_source="TNS",
                    source_release="2024a", confidence=0.9,
                    created_utc="2099-01-01T00:00:00")
    assert a.content_hash() == b.content_hash()


def test_label_record_hash_changes_with_label():
    a = LabelRecord(object_id="ZTF1", label="SN Ia", label_source="TNS",
                    source_release="2024a", confidence=0.9)
    b = LabelRecord(object_id="ZTF1", label="SN II", label_source="TNS",
                    source_release="2024a", confidence=0.9)
    assert a.content_hash() != b.content_hash()


def test_benchmark_spec_hash_stable_and_sensitive():
    spec = BenchmarkSpec(benchmark_id="bench-1", task_family="anomaly",
                         modalities=["ztf", "gaia"], positive_definition="verified unknown",
                         split_ids=["core_object_split"], primary_metric="auprc")
    same = BenchmarkSpec(benchmark_id="bench-1", task_family="anomaly",
                         modalities=["ztf", "gaia"], positive_definition="verified unknown",
                         split_ids=["core_object_split"], primary_metric="auprc")
    assert spec.content_hash() == same.content_hash()

    changed = BenchmarkSpec(benchmark_id="bench-1", task_family="anomaly",
                            modalities=["ztf", "gaia"], positive_definition="verified unknown",
                            split_ids=["core_object_split"], primary_metric="recall_at_k")
    assert spec.content_hash() != changed.content_hash()


def test_result_record_synthetic_flag_is_part_of_identity():
    real = ResultRecord(experiment_id="EXP-0001", benchmark_id="bench-1",
                        split_id="core_object_split", dataset_manifest_hash="abc",
                        metric="auprc", value=0.7, sample_count=100,
                        confidence_interval=[0.6, 0.8], seed=0, synthetic=False)
    synthetic = ResultRecord(**{**real.to_dict(), "synthetic": True})
    assert real.content_hash() != synthetic.content_hash()
