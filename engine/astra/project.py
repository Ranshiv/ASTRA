"""Versioned, project-scoped research workspaces.

The shared canonical survey store deliberately lives outside a project so an
object downloaded for one experiment is not copied for every other one.  The
mutable research record, however, belongs to a project: its manifest,
candidate reviews, experiments, results, reports, and audit trail must never
be confused with another investigation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config, manifest as manifest_mod

PROJECT_SCHEMA_VERSION = 1
_PROJECT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_RESERVED_IDS = {
    "manifests", "metadata", "candidates", "experiments", "results", "reports",
}
_WORKSPACE_DIRS = ("metadata", "manifests", "candidates", "experiments", "results", "reports")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Project:
    project_id: str
    name: str
    description: str = ""
    selected_surveys: list[str] = field(default_factory=list)
    query_regions: list[dict[str, float]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    data_root: str = ""
    status: str = "active"
    created_utc: str = field(default_factory=_now)
    updated_utc: str = field(default_factory=_now)
    archived_utc: str | None = None
    schema_version: int = PROJECT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _root(root: Path | None = None) -> Path:
    return (root or config.PATHS.projects).resolve()


def _validate_project_id(project_id: str) -> str:
    value = project_id.strip().lower()
    if value in _RESERVED_IDS or not _PROJECT_ID.fullmatch(value):
        raise ValueError(
            "project_id must contain 1–63 lowercase letters, digits, or hyphens "
            "and cannot use a reserved workspace name"
        )
    return value


def _slug(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return (text or "project")[:63].rstrip("-") or "project"


def _normalise_name(name: object) -> str:
    value = str(name).strip()
    if not 1 <= len(value) <= 120:
        raise ValueError("project name must contain 1–120 characters")
    return value


def _string_list(value: object, field_name: str, maximum: int = 100) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    items = []
    for raw in value:
        item = str(raw).strip()
        if not item:
            continue
        if len(item) > 120:
            raise ValueError(f"{field_name} entries must be at most 120 characters")
        if item not in items:
            items.append(item)
    if len(items) > maximum:
        raise ValueError(f"{field_name} may contain at most {maximum} entries")
    return items


def _regions(value: object) -> list[dict[str, float]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("query_regions must be a list")
    result: list[dict[str, float]] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("each query region must be an object")
        try:
            ra_deg = float(raw["ra_deg"])
            dec_deg = float(raw["dec_deg"])
            radius_arcsec = float(raw["radius_arcsec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("query regions require numeric ra_deg, dec_deg, and radius_arcsec") from exc
        if not 0.0 <= ra_deg < 360.0 or not -90.0 <= dec_deg <= 90.0 or radius_arcsec <= 0:
            raise ValueError("query region coordinates or radius are out of range")
        result.append({
            "ra_deg": round(ra_deg, 8),
            "dec_deg": round(dec_deg, 8),
            "radius_arcsec": round(radius_arcsec, 4),
        })
    if len(result) > 100:
        raise ValueError("query_regions may contain at most 100 entries")
    return result


def project_dir(project_id: str, root: Path | None = None) -> Path:
    base = _root(root)
    candidate = base / _validate_project_id(project_id)
    if candidate.parent != base:
        raise ValueError("project path escapes the managed Projects directory")
    return candidate


def project_path(project_id: str, root: Path | None = None) -> Path:
    return project_dir(project_id, root) / "project.json"


def manifest_dir(project_id: str, root: Path | None = None) -> Path:
    return project_dir(project_id, root) / "manifests"


def layout(project_id: str, root: Path | None = None) -> dict[str, str]:
    directory = project_dir(project_id, root)
    return {name: str(directory / name) for name in _WORKSPACE_DIRS}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_audit(directory: Path, action: str, details: dict[str, Any]) -> None:
    path = directory / "metadata" / "audit.jsonl"
    entry = {"event_utc": _now(), "action": action, "subject": directory.name, "details": details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _migrate(payload: dict[str, Any]) -> dict[str, Any]:
    version = int(payload.get("schema_version", 0))
    if version > PROJECT_SCHEMA_VERSION:
        raise ValueError(f"project schema {version} is newer than this ASTRA build")
    while version < PROJECT_SCHEMA_VERSION:
        if version == 0:
            payload = {
                **payload,
                "schema_version": 1,
                "description": payload.get("description", ""),
                "selected_surveys": payload.get("selected_surveys", []),
                "query_regions": payload.get("query_regions", []),
                "tags": payload.get("tags", []),
                "status": payload.get("status", "active"),
                "archived_utc": payload.get("archived_utc"),
                "updated_utc": payload.get("updated_utc", payload.get("created_utc", _now())),
            }
            version = 1
            continue
        raise ValueError(f"no migration is registered for project schema {version}")
    return payload


def _from_payload(payload: dict[str, Any], expected_id: str) -> Project:
    migrated = _migrate(dict(payload))
    if _validate_project_id(str(migrated.get("project_id", ""))) != expected_id:
        raise ValueError("project manifest identity does not match its directory")
    project = Project(
        project_id=expected_id,
        name=_normalise_name(migrated.get("name", "")),
        description=str(migrated.get("description", "")).strip(),
        selected_surveys=_string_list(migrated.get("selected_surveys"), "selected_surveys"),
        query_regions=_regions(migrated.get("query_regions")),
        tags=_string_list(migrated.get("tags"), "tags"),
        data_root=str(migrated.get("data_root", "")).strip(),
        status=str(migrated.get("status", "active")),
        created_utc=str(migrated.get("created_utc", "")),
        updated_utc=str(migrated.get("updated_utc", "")),
        archived_utc=migrated.get("archived_utc"),
        schema_version=int(migrated.get("schema_version", PROJECT_SCHEMA_VERSION)),
    )
    if project.status not in {"active", "archived"}:
        raise ValueError("project status must be active or archived")
    if not project.created_utc or not project.updated_utc or not project.data_root:
        raise ValueError("project manifest is missing required metadata")
    return project


def _read(project_id: str, root: Path | None = None) -> Project:
    path = project_path(project_id, root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"project '{project_id}' does not exist") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"project '{project_id}' metadata is invalid JSON") from exc
    return _from_payload(payload, _validate_project_id(project_id))


def _unique_project_id(name: str, root: Path) -> str:
    base = _slug(name)
    candidate = base
    suffix = 2
    while project_dir(candidate, root).exists():
        tail = f"-{suffix}"
        candidate = f"{base[:63 - len(tail)].rstrip('-')}{tail}"
        suffix += 1
    return candidate


def create(*, name: str, project_id: str | None = None, description: str = "",
           selected_surveys: list[str] | None = None,
           query_regions: list[dict[str, Any]] | None = None,
           tags: list[str] | None = None, data_root: str | None = None,
           root: Path | None = None) -> dict[str, Any]:
    base = _root(root)
    base.mkdir(parents=True, exist_ok=True)
    title = _normalise_name(name)
    identifier = _validate_project_id(project_id) if project_id else _unique_project_id(title, base)
    target = project_dir(identifier, base)
    if target.exists():
        raise FileExistsError(f"project '{identifier}' already exists")

    now = _now()
    project = Project(
        project_id=identifier,
        name=title,
        description=str(description or "").strip(),
        selected_surveys=_string_list(selected_surveys, "selected_surveys"),
        query_regions=_regions(query_regions),
        tags=_string_list(tags, "tags"),
        data_root=str(Path(data_root).expanduser().resolve() if data_root else config.PATHS.root.resolve()),
        created_utc=now,
        updated_utc=now,
    )

    stage = Path(tempfile.mkdtemp(prefix=f".{identifier}-", dir=base))
    try:
        for name in _WORKSPACE_DIRS:
            (stage / name).mkdir()
        _atomic_write_json(stage / "project.json", project.to_dict())
        _append_audit(stage, "project.create", {"name": project.name})
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return open_project(identifier, root=base)


def open_project(project_id: str, root: Path | None = None) -> dict[str, Any]:
    project = _read(project_id, root)
    result = project.to_dict()
    result["layout"] = layout(project.project_id, root)
    result["manifest_count"] = len(manifest_mod.list_manifests(manifest_dir(project.project_id, root)))
    return result


def list_projects(root: Path | None = None, include_archived: bool = True) -> list[dict[str, Any]]:
    base = _root(root)
    if not base.exists():
        return []
    result = []
    for metadata_path in sorted(base.glob("*/project.json")):
        try:
            item = open_project(metadata_path.parent.name, root=base)
        except (OSError, ValueError):
            continue
        if include_archived or item["status"] == "active":
            result.append(item)
    return result


def update(project_id: str, patch: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("project patch must be an object")
    allowed = {"name", "description", "selected_surveys", "query_regions", "tags", "data_root"}
    unknown = sorted(set(patch) - allowed)
    if unknown:
        raise ValueError(f"project patch contains immutable or unknown fields: {', '.join(unknown)}")

    project = _read(project_id, root)
    if project.status == "archived":
        raise ValueError("archived projects are read-only")
    if "name" in patch:
        project.name = _normalise_name(patch["name"])
    if "description" in patch:
        project.description = str(patch["description"] or "").strip()
    if "selected_surveys" in patch:
        project.selected_surveys = _string_list(patch["selected_surveys"], "selected_surveys")
    if "query_regions" in patch:
        project.query_regions = _regions(patch["query_regions"])
    if "tags" in patch:
        project.tags = _string_list(patch["tags"], "tags")
    if "data_root" in patch:
        value = str(patch["data_root"] or "").strip()
        if not value:
            raise ValueError("data_root must not be empty")
        project.data_root = str(Path(value).expanduser().resolve())
    project.updated_utc = _now()

    directory = project_dir(project.project_id, root)
    _atomic_write_json(directory / "project.json", project.to_dict())
    _append_audit(directory, "project.update", {"fields": sorted(patch)})
    return open_project(project.project_id, root)


def archive(project_id: str, archived: bool = True, root: Path | None = None) -> dict[str, Any]:
    project = _read(project_id, root)
    project.status = "archived" if archived else "active"
    project.archived_utc = _now() if archived else None
    project.updated_utc = _now()
    directory = project_dir(project.project_id, root)
    _atomic_write_json(directory / "project.json", project.to_dict())
    _append_audit(directory, "project.archive" if archived else "project.restore", {})
    return open_project(project.project_id, root)


def require_active(project_id: str, root: Path | None = None) -> Project:
    project = _read(project_id, root)
    if project.status != "active":
        raise ValueError(f"project '{project_id}' is archived and cannot accept new work")
    return project


def validate(project_id: str, root: Path | None = None) -> dict[str, Any]:
    issues: list[str] = []
    try:
        project = _read(project_id, root)
    except (OSError, ValueError) as exc:
        return {"project_id": project_id, "valid": False, "issues": [str(exc)]}

    directory = project_dir(project.project_id, root)
    for name in _WORKSPACE_DIRS:
        if not (directory / name).is_dir():
            issues.append(f"missing workspace directory: {name}")

    manifests = manifest_mod.list_manifests(manifest_dir(project.project_id, root))
    for summary in manifests:
        try:
            record = manifest_mod.load(str(summary["dataset_id"]), manifest_dir(project.project_id, root))
            if not record.verify():
                issues.append(f"manifest content hash mismatch: {record.dataset_id}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"invalid manifest {summary.get('dataset_id', 'unknown')}: {exc}")

    return {
        "project_id": project.project_id,
        "valid": not issues,
        "issues": issues,
        "schema_version": project.schema_version,
        "status": project.status,
        "layout": layout(project.project_id, root),
        "manifest_count": len(manifests),
    }
