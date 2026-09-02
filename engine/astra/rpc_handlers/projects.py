"""Project lifecycle handlers: manifest listing and project create/list/
open/update/archive/validate.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

from typing import Any

from .common import Handler

from .. import manifest as manifest_mod, project as project_mod

def _handle_manifest_list(params: dict[str, Any]) -> list[dict]:
    project_id = params.get("project_id")
    # `list_manifests` appends "manifests" onto `root` itself -- this must
    # be the project directory, matching `_handle_research_bundle_build`
    # below, not `project_mod.manifest_dir()` (already the "manifests"
    # subdirectory, which would double-nest).
    root = project_mod.project_dir(str(project_id)) if project_id else None
    return manifest_mod.list_manifests(root)


def _handle_project_create(params: dict[str, Any]) -> dict[str, Any]:
    return project_mod.create(
        name=str(params["name"]),
        project_id=params.get("project_id"),
        description=str(params.get("description") or ""),
        selected_surveys=params.get("selected_surveys"),
        query_regions=params.get("query_regions"),
        tags=params.get("tags"),
        data_root=params.get("data_root"),
    )


def _handle_project_list(params: dict[str, Any]) -> list[dict]:
    return project_mod.list_projects(include_archived=bool(params.get("include_archived", True)))


def _handle_project_open(params: dict[str, Any]) -> dict[str, Any]:
    return project_mod.open_project(str(params["project_id"]))


def _handle_project_update(params: dict[str, Any]) -> dict[str, Any]:
    return project_mod.update(str(params["project_id"]), params.get("patch") or {})


def _handle_project_archive(params: dict[str, Any]) -> dict[str, Any]:
    return project_mod.archive(str(params["project_id"]), bool(params.get("archived", True)))


def _handle_project_validate(params: dict[str, Any]) -> dict[str, Any]:
    return project_mod.validate(str(params["project_id"]))


HANDLERS: dict[str, Handler] = {
    "manifest.list": _handle_manifest_list,
    "project.create": _handle_project_create,
    "project.list": _handle_project_list,
    "project.open": _handle_project_open,
    "project.update": _handle_project_update,
    "project.archive": _handle_project_archive,
    "project.validate": _handle_project_validate,
}
