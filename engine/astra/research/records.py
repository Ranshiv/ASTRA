"""Record schemas that bind a reported result to its evidence.

`DatasetManifest` is not redefined here: `manifest.Manifest` already carries
dataset_id, release queries, license/citation/calibration-version/selection-
rule (v2, see `manifest.py`), and a content hash sealed at acquisition time.
Re-exporting it as `DatasetManifest` keeps one manifest implementation
instead of two competing ones.

`LabelRecord`, `BenchmarkSpec`, and `ResultRecord` are genuinely new: nothing
in the codebase before this module recorded an externally-sourced label with
its conflict state, declared a benchmark's positive definition and metrics
before execution, or bound a metric value to the dataset/split/benchmark
triple that produced it.

Every record type hashes the same way `manifest.Manifest` does: canonical
(sorted-keys, no whitespace) JSON, SHA-256, so two records describing the
same evidence hash identically regardless of when they were written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from ..manifest import Manifest as DatasetManifest  # noqa: F401 - re-export

__all__ = [
    "DatasetManifest", "LabelRecord", "BenchmarkSpec", "ResultRecord",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _content_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class LabelRecord:
    """One object's externally-sourced class label.

    `confidence` is the source's own confidence, not a model score.
    `conflicts` lists other (source, label) pairs seen for the same object
    when sources disagree, so disagreement is visible rather than silently
    resolved by "last write wins". `adjudication_state` is one of
    `"unreviewed"`, `"confirmed"`, `"disputed"`, `"rejected"`.
    """

    object_id: str
    label: str
    label_source: str
    source_release: str
    confidence: float
    created_utc: str = field(default_factory=_utc_now)
    conflicts: list[dict] = field(default_factory=list)
    adjudication_state: str = "unreviewed"
    reviewer: str | None = None

    def content_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "created_utc"}
        return _content_hash(payload)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkSpec:
    """A benchmark's definition, declared before execution.

    Per the experimental standards (docs/BENCHMARKS.md): primary and
    secondary metrics are declared up front, splits are named (not
    re-derived per run), and seeds are fixed so a later rerun cannot quietly
    cherry-pick a favorable seed.
    """

    benchmark_id: str
    task_family: str
    modalities: list[str]
    positive_definition: str
    split_ids: list[str]
    primary_metric: str
    secondary_metrics: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    created_utc: str = field(default_factory=_utc_now)
    notes: str = ""

    def content_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "created_utc"}
        return _content_hash(payload)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResultRecord:
    """One metric value, bound to the evidence that produced it.

    `sample_count` and `confidence_interval` make a bare mean unreportable:
    every value here is expected to carry both. `artifact_refs` points at
    the result files (calibration plot, failure-case index, ...) this value
    was read from, so a report can link back to the underlying artefact
    rather than only quoting a number.
    """

    experiment_id: str
    benchmark_id: str
    split_id: str
    dataset_manifest_hash: str
    metric: str
    value: float
    sample_count: int
    confidence_interval: list[float]
    seed: int
    created_utc: str = field(default_factory=_utc_now)
    synthetic: bool = False
    artifact_refs: list[str] = field(default_factory=list)
    notes: str = ""

    def content_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "created_utc"}
        return _content_hash(payload)

    def to_dict(self) -> dict:
        return asdict(self)
