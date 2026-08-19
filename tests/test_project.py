"""Project workspace lifecycle and isolation tests."""

from __future__ import annotations

import json

import pytest

from astra import acquire, project, rpc, surveys
from tests.test_acquire import FakeConnector


def test_create_project_makes_the_complete_workspace(isolated_root):
    created = project.create(
        name="RR Lyrae follow-up",
        selected_surveys=["ZTF", "Gaia", "ZTF"],
        query_regions=[{"ra_deg": 291.3663, "dec_deg": 42.7844, "radius_arcsec": 10}],
        tags=["variables", "2026"],
    )

    assert created["project_id"] == "rr-lyrae-follow-up"
    assert created["selected_surveys"] == ["ZTF", "Gaia"]
    for name, path in created["layout"].items():
        assert name in {"metadata", "manifests", "candidates", "experiments", "results", "reports"}
        assert (isolated_root.projects / created["project_id"] / name).is_dir()
        assert path
    assert (isolated_root.projects / created["project_id"] / "metadata" / "audit.jsonl").is_file()


def test_project_update_archive_and_validate(isolated_root):
    created = project.create(name="Project Alpha")
    updated = project.update(created["project_id"], {
        "description": "A reproducible candidate review.",
        "selected_surveys": ["ZTF", "TESS"],
        "tags": ["stage-b"],
    })

    assert updated["description"].startswith("A reproducible")
    assert updated["selected_surveys"] == ["ZTF", "TESS"]
    assert project.validate(created["project_id"])["valid"] is True

    archived = project.archive(created["project_id"])
    assert archived["status"] == "archived"
    with pytest.raises(ValueError, match="read-only"):
        project.update(created["project_id"], {"name": "Changed"})
    assert project.archive(created["project_id"], archived=False)["status"] == "active"


def test_project_rejects_unsafe_or_duplicate_ids(isolated_root):
    project.create(name="Safe", project_id="safe")
    with pytest.raises(FileExistsError):
        project.create(name="Again", project_id="safe")
    with pytest.raises(ValueError):
        project.create(name="Unsafe", project_id="../escape")
    with pytest.raises(ValueError):
        project.create(name="Reserved", project_id="manifests")


def test_project_migrates_legacy_schema_zero(isolated_root):
    directory = isolated_root.projects / "legacy"
    directory.mkdir()
    payload = {
        "project_id": "legacy", "name": "Legacy project",
        "data_root": str(isolated_root.root), "created_utc": "2026-01-01T00:00:00+00:00",
    }
    (directory / "project.json").write_text(json.dumps(payload), encoding="utf-8")

    opened = project.open_project("legacy")
    assert opened["schema_version"] == 1
    assert opened["status"] == "active"


def test_project_scopes_new_acquisition_manifests(isolated_root, cone):
    surveys.register("project-fake", FakeConnector)
    try:
        created = project.create(name="Scoped acquisition")
        result = acquire.acquire(cone, survey_names=["project-fake"], limit=1,
                                 project_id=created["project_id"])
    finally:
        surveys._REGISTRY.pop("project-fake", None)

    assert result.project_id == created["project_id"]
    assert result.manifest_path is not None
    assert str(isolated_root.projects / created["project_id"] / "manifests") in result.manifest_path
    assert project.validate(created["project_id"])["manifest_count"] == 1


def test_project_rpc_lifecycle(isolated_root):
    created = rpc.dispatch({
        "id": 1,
        "method": "project.create",
        "params": {"name": "RPC project", "query_regions": []},
    })
    assert created["ok"] is True
    project_id = created["result"]["project_id"]

    listed = rpc.dispatch({"id": 2, "method": "project.list", "params": {}})
    assert [entry["project_id"] for entry in listed["result"]] == [project_id]

    validated = rpc.dispatch({
        "id": 3, "method": "project.validate", "params": {"project_id": project_id},
    })
    assert validated["result"]["valid"] is True
