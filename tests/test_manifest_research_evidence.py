"""Manifest v2 archive-provenance fields (license/citation/calibration/
selection/artifact stats) and that a v1 manifest on disk still loads."""

from __future__ import annotations

import json

from astra.manifest import Manifest, MANIFEST_VERSION


def test_v2_fields_default_empty_and_do_not_affect_content_hash():
    a = Manifest.create("ds1")
    b = Manifest.create("ds1", license="CC-BY-4.0", citation="Author et al. 2024",
                        calibration_version="dr24", selection_rule="cone<=10as")
    assert a.compute_content_hash() == b.compute_content_hash()


def test_record_artifact_sets_stats_without_touching_content_hash():
    manifest = Manifest.create("ds1")
    manifest.seal()
    before = manifest.content_hash
    manifest.record_artifact(row_count=100, byte_count=2048, checksum="abc123")
    assert manifest.row_count == 100
    assert manifest.byte_count == 2048
    assert manifest.checksum == "abc123"
    assert manifest.content_hash == before  # artifact stats are not query identity


def test_v1_manifest_on_disk_still_loads(tmp_path):
    manifest = Manifest.create("legacy")
    manifest.seal()
    payload = manifest.to_dict()
    payload["version"] = 1
    for key in ("license", "citation", "calibration_version",
               "selection_rule", "row_count", "byte_count", "checksum"):
        payload.pop(key, None)

    from astra import manifest as manifest_mod
    path = manifest_mod.manifest_path("legacy", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = manifest_mod.load("legacy", tmp_path)
    assert loaded.license == ""
    assert loaded.checksum is None
    assert loaded.verify()


def test_current_version_is_2():
    assert MANIFEST_VERSION == 2
