"""Load/save round-trips for the research record store, and that a
tampered dataset manifest is rejected rather than silently loaded."""

from __future__ import annotations

import pytest

from astra.manifest import Manifest, SurveyQuery
from astra.research import store
from astra.research.records import BenchmarkSpec, LabelRecord, ResultRecord


@pytest.fixture
def research_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_RESEARCH_ROOT", str(tmp_path))
    return tmp_path


def test_dataset_manifest_round_trip(research_root):
    manifest = Manifest.create("core-ztf-2024a")
    manifest.add(SurveyQuery(survey="ZTF", release="dr24", ra_deg=180.0, dec_deg=22.0,
                             radius_arcsec=10.0, limit=10, object_ids=["a", "b"]))
    manifest.seal()
    store.save_dataset_manifest(manifest)

    loaded = store.load_dataset_manifest("core-ztf-2024a")
    assert loaded.content_hash == manifest.content_hash
    assert loaded.total_objects() == 2


def test_dataset_manifest_rejects_tampered_content(research_root):
    manifest = Manifest.create("tampered")
    manifest.add(SurveyQuery(survey="ZTF", release="dr24", ra_deg=1.0, dec_deg=1.0,
                             radius_arcsec=5.0, limit=1, object_ids=["a"]))
    manifest.seal()
    path = store.save_dataset_manifest(manifest)

    text = path.read_text(encoding="utf-8")
    tampered = text.replace('"limit": 1', '"limit": 999')
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(store.ResearchStoreError):
        store.load_dataset_manifest("tampered")


def test_label_records_round_trip(research_root):
    records = [
        LabelRecord(object_id="ZTF1", label="SN Ia", label_source="TNS",
                   source_release="2024a", confidence=0.9),
        LabelRecord(object_id="ZTF2", label="RR Lyr", label_source="VSX",
                   source_release="2023b", confidence=0.8),
    ]
    store.save_label_records(records, name="test_labels")
    loaded = store.load_label_records(name="test_labels")
    assert [r.object_id for r in loaded] == ["ZTF1", "ZTF2"]


def test_benchmark_spec_round_trip(research_root):
    spec = BenchmarkSpec(benchmark_id="bench-1", task_family="anomaly",
                         modalities=["ztf"], positive_definition="verified unknown",
                         split_ids=["core_object_split"], primary_metric="auprc")
    store.save_benchmark_spec(spec)
    loaded = store.load_benchmark_spec("bench-1")
    assert loaded.content_hash() == spec.content_hash()
    assert store.list_benchmark_specs() == ["bench-1"]


def test_result_records_real_and_synthetic_kept_separate(research_root):
    real = ResultRecord(experiment_id="EXP-0001", benchmark_id="bench-1",
                        split_id="s1", dataset_manifest_hash="abc", metric="auprc",
                        value=0.7, sample_count=100, confidence_interval=[0.6, 0.8],
                        seed=0, synthetic=False)
    store.save_result_records([real], synthetic=False)
    assert store.load_result_records(synthetic=False)[0].experiment_id == "EXP-0001"
    assert store.load_result_records(synthetic=True) == []


def test_result_records_rejects_mismatched_synthetic_flag(research_root):
    real = ResultRecord(experiment_id="EXP-0001", benchmark_id="bench-1",
                        split_id="s1", dataset_manifest_hash="abc", metric="auprc",
                        value=0.7, sample_count=100, confidence_interval=[0.6, 0.8],
                        seed=0, synthetic=False)
    with pytest.raises(store.ResearchStoreError):
        store.save_result_records([real], synthetic=True)


def test_source_registry_round_trip(research_root):
    store.save_source_registry({"schema_version": 1, "sources": {"a": 1}})
    assert store.load_source_registry() == {"schema_version": 1, "sources": {"a": 1}}
