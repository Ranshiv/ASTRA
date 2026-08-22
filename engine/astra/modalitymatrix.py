"""Versioned optional image/spectral/multiband feature sidecar tables.

The baseline light-curve matrix intentionally remains stable and dense. Image,
spectral, and multiband-period extraction are all sparse and product- or
group-dependent, so these features are stored as keyed Parquet sidecars and
joined explicitly by research jobs. A multiband sidecar row spans a *group*
of bands for one object rather than one curve, so it is written under the
sentinel band key "__multiband__" (see multiband.py) rather than needing a
change to KEY_COLUMNS, which image/spectral sidecars already in production
share.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config

SIDECAR_SCHEMA_VERSION = 1
KEY_COLUMNS = ("survey", "release", "object_id", "band")


@dataclass(frozen=True)
class SidecarTable:
    kind: str
    path: str
    rows: int
    columns: tuple[str, ...]
    schema_version: int = SIDECAR_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "path": self.path,
            "rows": self.rows,
            "columns": list(self.columns),
            "schema_version": self.schema_version,
        }


def _safe_kind(kind: str) -> str:
    value = str(kind).strip().lower()
    if value not in {"image", "spectral", "multiband"}:
        raise ValueError("sidecar kind must be image, spectral, or multiband")
    return value


def _identity(payload: dict, identity: dict | None = None) -> dict:
    source = payload.get("source") or {}
    supplied = identity or payload.get("identity") or {}
    return {
        "survey": str(supplied.get("survey") or source.get("survey") or "unknown"),
        "release": str(supplied.get("release") or source.get("release") or "unknown"),
        "object_id": str(supplied.get("object_id") or source.get("object_id") or "unknown"),
        "band": str(supplied.get("band") or source.get("band") or "unknown"),
    }


def _flatten(payload: dict, kind: str, identity: dict | None = None) -> dict:
    row = _identity(payload, identity)
    row["sidecar_kind"] = _safe_kind(kind)
    row["sidecar_schema_version"] = int(payload.get("schema_version", 0))
    source = payload.get("source") or {}
    row["source_sha256"] = str(source.get("sha256") or "")
    row["source_path"] = str(source.get("path") or "")
    row["quality_json"] = json.dumps(payload.get("quality") or {}, sort_keys=True)
    for name, value in (payload.get("features") or {}).items():
        key = f"{kind}__{name}"
        if isinstance(value, (bool, int, float)) and not isinstance(value, bool):
            row[key] = float(value) if np.isfinite(float(value)) else None
        else:
            row[key] = None
    return row


def _atomic_write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd", compression_level=6)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_payloads(payloads: Iterable[dict], kind: str, *, name: str = "default",
                  root: Path | None = None,
                  identities: Iterable[dict | None] | None = None) -> SidecarTable:
    """Persist validated extractor payloads as a keyed, nullable sidecar."""
    kind = _safe_kind(kind)
    identity_rows = list(identities) if identities is not None else []
    rows = [_flatten(payload, kind, identity_rows[index] if index < len(identity_rows) else None)
            for index, payload in enumerate(payloads)]
    destination = (root or config.PATHS.projects) / "features" / "sidecars"
    path = destination / f"{name}_{kind}_v{SIDECAR_SCHEMA_VERSION}.parquet"
    if not rows:
        columns = list(KEY_COLUMNS) + ["sidecar_kind", "sidecar_schema_version",
                                       "source_sha256", "source_path", "quality_json"]
        table = pa.table({column: pa.array([], type=pa.string()) for column in columns}, metadata={
            b"sidecar_schema_version": str(SIDECAR_SCHEMA_VERSION).encode(),
            b"sidecar_kind": kind.encode(),
        })
    else:
        names = list(dict.fromkeys(key for row in rows for key in row))
        columns = {name: pa.array([row.get(name) for row in rows]) for name in names}
        table = pa.table(columns, metadata={
            b"sidecar_schema_version": str(SIDECAR_SCHEMA_VERSION).encode(),
            b"sidecar_kind": kind.encode(),
        })
    _atomic_write(table, path)
    return SidecarTable(kind=kind, path=str(path), rows=len(rows),
                        columns=tuple(table.column_names))


def load(path: str | Path) -> pa.Table:
    table = pq.read_table(Path(path))
    missing = [column for column in KEY_COLUMNS if column not in table.column_names]
    if missing:
        raise ValueError(f"sidecar missing key columns: {', '.join(missing)}")
    return table


def join_rows(base_identities: list[dict], sidecar: pa.Table,
              *, kind: str) -> tuple[list[dict], dict]:
    """Left-join a sidecar onto base identities without imputing evidence."""
    kind = _safe_kind(kind)
    records = sidecar.to_pylist()
    by_key = {(row.get("survey"), row.get("release"), row.get("object_id"), row.get("band")): row
              for row in records if row.get("sidecar_kind") in {None, kind}}
    feature_columns = [name for name in sidecar.column_names
                       if name.startswith(f"{kind}__")]
    joined, matched = [], 0
    for identity in base_identities:
        key = tuple(identity.get(name, "unknown") for name in KEY_COLUMNS)
        row = dict(identity)
        side = by_key.get(key)
        if side is None:
            for column in feature_columns:
                row[column] = None
        else:
            matched += 1
            for column in feature_columns:
                row[column] = side.get(column)
        joined.append(row)
    return joined, {
        "kind": kind,
        "base_rows": len(base_identities),
        "sidecar_rows": len(records),
        "matched_rows": matched,
        "missing_rows": len(base_identities) - matched,
        "match_rate": (round(matched / len(base_identities), 4)
                       if base_identities else None),
        "feature_columns": feature_columns,
        "schema_version": int((sidecar.schema.metadata or {}).get(
            b"sidecar_schema_version", str(SIDECAR_SCHEMA_VERSION).encode())),
    }


def list_sidecars(root: Path | None = None) -> list[dict]:
    directory = (root or config.PATHS.projects) / "features" / "sidecars"
    if not directory.exists():
        return []
    output = []
    for path in sorted(directory.glob("*.parquet")):
        try:
            table = load(path)
            metadata = table.schema.metadata or {}
            output.append(SidecarTable(
                kind=str(metadata.get(b"sidecar_kind", b"unknown"), "utf-8"),
                path=str(path), rows=table.num_rows,
                columns=tuple(table.column_names),
                schema_version=int(metadata.get(b"sidecar_schema_version", b"1")),
            ).to_dict())
        except (OSError, ValueError, pa.ArrowException):
            continue
    return output
