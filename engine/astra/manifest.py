"""Query manifests — reproducibility without duplicating the archives.

Plan section 37 requires every experiment to be reproducible, and section 20
specifies seven experiment groups plus ablations. Snapshotting the data for
each would multiply storage by the number of experiments.

Instead a manifest records what was asked for and a hash of what came back.
ZTF, Gaia and TESS are permanent public archives, so the data can be
re-materialised on demand, and the content hash proves the re-fetch matched.
A manifest is a few hundred kilobytes where a snapshot is tens of gigabytes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .surveys.base import ConeQuery, SourceRef

MANIFEST_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SurveyQuery:
    survey: str
    release: str
    ra_deg: float
    dec_deg: float
    radius_arcsec: float
    limit: int
    object_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_cone(cls, survey: str, release: str, query: ConeQuery,
                  limit: int, sources: list[SourceRef]) -> "SurveyQuery":
        return cls(
            survey=survey,
            release=release,
            ra_deg=query.ra_deg,
            dec_deg=query.dec_deg,
            radius_arcsec=query.radius_arcsec,
            limit=limit,
            object_ids=sorted(s.object_id for s in sources),
        )

    def object_id_hash(self) -> str:
        """Hash the id list so a huge query stays a small manifest."""
        joined = "\n".join(self.object_ids).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()


@dataclass
class Manifest:
    """Everything needed to reproduce one dataset, minus the dataset.

    v2 adds the fields a research evidence package needs to trace a result
    back to a licensed, citable release rather than only a reproducible
    query: `license`, `citation`, `calibration_version`, `selection_rule`,
    and the materialized artefact's `row_count`/`byte_count`/`checksum`.
    A v1 record on disk is missing these; `load()` fills them with the
    empty defaults below rather than refusing to load, since old manifests
    remain valid evidence for the query they describe -- they simply
    predate the archive-provenance fields.
    """

    dataset_id: str
    created_utc: str = field(default_factory=_utc_now)
    version: int = MANIFEST_VERSION
    queries: list[SurveyQuery] = field(default_factory=list)
    pipeline_version: str = "0.1.0"
    environment: dict = field(default_factory=dict)
    content_hash: str | None = None
    # v2 archive-provenance fields (source registry: docs/DATA_SOURCES.md).
    license: str = ""
    citation: str = ""
    calibration_version: str = ""
    selection_rule: str = ""
    row_count: int = 0
    byte_count: int = 0
    checksum: str | None = None

    @classmethod
    def create(cls, dataset_id: str, *, license: str = "", citation: str = "",
              calibration_version: str = "", selection_rule: str = "") -> "Manifest":
        return cls(dataset_id=dataset_id, environment=capture_environment(),
                   license=license, citation=citation,
                   calibration_version=calibration_version,
                   selection_rule=selection_rule)

    def add(self, query: SurveyQuery) -> "Manifest":
        self.queries.append(query)
        return self

    def compute_content_hash(self) -> str:
        """Hash the queries and their results, not the timestamp.

        Two manifests describing the same data must hash identically even
        when created weeks apart, or the reproducibility check is worthless.
        """
        payload = [
            {
                "survey": q.survey,
                "release": q.release,
                "cone": f"{q.ra_deg:.6f},{q.dec_deg:.6f},{q.radius_arcsec:.3f}",
                "limit": q.limit,
                "objects": q.object_id_hash(),
            }
            for q in sorted(self.queries, key=lambda q: (q.survey, q.release))
        ]
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def seal(self) -> "Manifest":
        """Freeze the content hash once the queries are complete."""
        self.content_hash = self.compute_content_hash()
        return self

    def record_artifact(self, *, row_count: int, byte_count: int,
                        checksum: str) -> "Manifest":
        """Record the materialized dataset's stats after acquisition.

        `checksum` is the SHA-256 of the artefact bytes -- distinct from
        `content_hash`, which is a hash of the *query* and stays stable
        even if the artefact is later re-materialized to a different file
        layout. Both are needed: `content_hash` proves the query was
        reproduced; `checksum` proves the resulting file is unmodified.
        """
        self.row_count = row_count
        self.byte_count = byte_count
        self.checksum = checksum
        return self

    def verify(self) -> bool:
        """True when the recorded hash still matches the recorded queries."""
        return self.content_hash == self.compute_content_hash()

    def total_objects(self) -> int:
        return sum(len(q.object_ids) for q in self.queries)

    def to_dict(self) -> dict:
        return asdict(self)


def capture_environment() -> dict:
    """Software and hardware context, per plan section 37."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for module in ("numpy", "astropy", "astroquery", "lightkurve", "pyarrow"):
        try:
            env[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - absence is recorded, not fatal
            env[module] = "absent"
    return env


def manifest_path(dataset_id: str, root: Path | None = None) -> Path:
    root = root or config.PATHS.projects
    return root / "manifests" / f"{dataset_id}.json"


def save(manifest: Manifest, root: Path | None = None) -> Path:
    path = manifest_path(manifest.dataset_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def load(dataset_id: str, root: Path | None = None) -> Manifest:
    path = manifest_path(dataset_id, root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = [SurveyQuery(**q) for q in payload.pop("queries", [])]
    return Manifest(queries=queries, **payload)


def list_manifests(root: Path | None = None) -> list[dict]:
    root = root or config.PATHS.projects
    directory = root / "manifests"
    if not directory.exists():
        return []

    summaries = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summaries.append({
            "dataset_id": payload.get("dataset_id"),
            "created_utc": payload.get("created_utc"),
            "surveys": sorted({q["survey"] for q in payload.get("queries", [])}),
            "objects": sum(len(q.get("object_ids", []))
                           for q in payload.get("queries", [])),
            "content_hash": payload.get("content_hash"),
        })
    return summaries
