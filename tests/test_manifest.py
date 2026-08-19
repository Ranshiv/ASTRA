"""Manifests must reproduce a dataset without storing a copy of it."""

from __future__ import annotations

from astra import manifest as manifest_mod
from astra.manifest import Manifest, SurveyQuery
from astra.surveys.base import SourceRef


def _sources(count: int, survey: str = "ZTF") -> list[SourceRef]:
    return [
        SourceRef(survey=survey, object_id=str(i), ra_deg=1.0, dec_deg=2.0)
        for i in range(count)
    ]


def test_hash_ignores_creation_time(cone):
    """Two manifests describing the same data must hash identically."""
    a = Manifest.create("a").add(
        SurveyQuery.from_cone("ZTF", "dr24", cone, 10, _sources(3))).seal()
    b = Manifest.create("b").add(
        SurveyQuery.from_cone("ZTF", "dr24", cone, 10, _sources(3))).seal()

    assert a.content_hash == b.content_hash


def test_hash_is_order_independent(cone):
    forward = Manifest.create("f")
    forward.add(SurveyQuery.from_cone("ZTF", "dr24", cone, 10, _sources(2)))
    forward.add(SurveyQuery.from_cone("Gaia", "dr3", cone, 10, _sources(2, "Gaia")))

    reverse = Manifest.create("r")
    reverse.add(SurveyQuery.from_cone("Gaia", "dr3", cone, 10, _sources(2, "Gaia")))
    reverse.add(SurveyQuery.from_cone("ZTF", "dr24", cone, 10, _sources(2)))

    assert forward.seal().content_hash == reverse.seal().content_hash


def test_hash_changes_when_results_change(cone):
    a = Manifest.create("a").add(
        SurveyQuery.from_cone("ZTF", "dr24", cone, 10, _sources(3))).seal()
    b = Manifest.create("b").add(
        SurveyQuery.from_cone("ZTF", "dr24", cone, 10, _sources(4))).seal()

    assert a.content_hash != b.content_hash


def test_hash_changes_when_release_changes(cone):
    a = Manifest.create("a").add(
        SurveyQuery.from_cone("ZTF", "dr23", cone, 10, _sources(3))).seal()
    b = Manifest.create("b").add(
        SurveyQuery.from_cone("ZTF", "dr24", cone, 10, _sources(3))).seal()

    assert a.content_hash != b.content_hash


def test_verify_detects_tampering(cone):
    record = Manifest.create("x").add(
        SurveyQuery.from_cone("ZTF", "dr24", cone, 10, _sources(3))).seal()
    assert record.verify() is True

    record.queries[0].object_ids.append("999")
    assert record.verify() is False


def test_manifest_stays_small_for_a_large_query(cone, tmp_path):
    """The point of a manifest is that it does not scale with the data."""
    record = Manifest.create("big").add(
        SurveyQuery.from_cone("ZTF", "dr24", cone, 50_000, _sources(50_000))).seal()
    path = manifest_mod.save(record, tmp_path)

    assert record.total_objects() == 50_000
    assert path.stat().st_size < 2 * 1024 * 1024  # under 2 MB for 50k objects


def test_save_and_load_round_trip(cone, tmp_path):
    original = Manifest.create("rt").add(
        SurveyQuery.from_cone("TESS", "spoc", cone, 5, _sources(2, "TESS"))).seal()
    manifest_mod.save(original, tmp_path)

    loaded = manifest_mod.load("rt", tmp_path)

    assert loaded.content_hash == original.content_hash
    assert loaded.verify() is True
    assert loaded.queries[0].survey == "TESS"


def test_environment_is_captured():
    env = manifest_mod.capture_environment()
    assert "python" in env
    assert "numpy" in env


def test_list_manifests_summarises(cone, tmp_path):
    manifest_mod.save(Manifest.create("one").add(
        SurveyQuery.from_cone("ZTF", "dr24", cone, 5, _sources(2))).seal(), tmp_path)

    listing = manifest_mod.list_manifests(tmp_path)

    assert len(listing) == 1
    assert listing[0]["dataset_id"] == "one"
    assert listing[0]["surveys"] == ["ZTF"]


def test_list_manifests_of_empty_root(tmp_path):
    assert manifest_mod.list_manifests(tmp_path) == []
