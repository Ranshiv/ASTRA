"""Load/save/list for the research record types, and the source registry.

Unlike `config.Paths` (which resolves the *runtime* data root, `$ASTRA_ROOT`,
kept outside the repo and outside version control), the research artefact
tree lives *inside* the repo -- `research/` alongside `engine/` and `docs/`
-- because its contents are small, text-diffable evidence records meant to
be reviewed like code (see `research/README.md`). This module is therefore
the one place that resolves *that* root, separately from `config.PATHS`.

`ASTRA_RESEARCH_ROOT` overrides the resolved path, for tests and for the
packaged engine (where the repo layout does not exist alongside the sidecar
executable and callers must point explicitly at a checkout).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from .records import BenchmarkSpec, DatasetManifest, LabelRecord, ResultRecord


class ResearchStoreError(ValueError):
    """A record failed to load, or its checksum did not match its content."""


def research_root() -> Path:
    override = os.environ.get("ASTRA_RESEARCH_ROOT")
    if override:
        return Path(override)
    # engine/astra/research/store.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[3] / "research"


def _dir(*parts: str) -> Path:
    path = research_root().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- DatasetManifest (manifest.Manifest, re-exported) ----------------------

def save_dataset_manifest(manifest: DatasetManifest) -> Path:
    if manifest.content_hash is None:
        raise ResearchStoreError("manifest must be sealed before saving")
    path = _dir("datasets", "manifests") / f"{manifest.dataset_id}.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def load_dataset_manifest(dataset_id: str) -> DatasetManifest:
    from ..manifest import Manifest, SurveyQuery  # local: avoid import cycle
    path = _dir("datasets", "manifests") / f"{dataset_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = [SurveyQuery(**q) for q in payload.pop("queries", [])]
    manifest = Manifest(queries=queries, **payload)
    if not manifest.verify():
        raise ResearchStoreError(
            f"dataset manifest {dataset_id!r} failed content-hash verification")
    return manifest


def list_dataset_manifests() -> list[str]:
    directory = _dir("datasets", "manifests")
    return sorted(p.stem for p in directory.glob("*.json"))


# --- LabelRecord -------------------------------------------------------

def save_label_records(records: list[LabelRecord], name: str = "object_labels") -> Path:
    """Append-safe: writes newline-delimited JSON so a large label set stays
    diffable and streamable rather than one giant JSON array."""
    path = _dir("labels") / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return path


def load_label_records(name: str = "object_labels") -> list[LabelRecord]:
    path = _dir("labels") / f"{name}.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(LabelRecord(**json.loads(line)))
    return records


# --- BenchmarkSpec -------------------------------------------------------

def save_benchmark_spec(spec: BenchmarkSpec) -> Path:
    path = _dir("benchmarks", "benchmark_specs") / f"{spec.benchmark_id}.json"
    path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    return path


def load_benchmark_spec(benchmark_id: str) -> BenchmarkSpec:
    path = _dir("benchmarks", "benchmark_specs") / f"{benchmark_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkSpec(**payload)


def list_benchmark_specs() -> list[str]:
    directory = _dir("benchmarks", "benchmark_specs")
    return sorted(p.stem for p in directory.glob("*.json"))


# --- ResultRecord --------------------------------------------------------

def save_result_records(records: list[ResultRecord], *, synthetic: bool) -> Path:
    """Real and synthetic results are kept in separate files by construction
    -- a caller cannot accidentally merge them by passing the wrong flag,
    because the flag chooses the filename, not just a column.

    Appends to whatever is already on disk rather than replacing it. A real
    bug this session found by hand: the previous version opened the file in
    `"w"` mode, so a second real call to `research.benchmark.run` (each one
    producing a fresh `experiment_id`, never a re-run of an old one) would
    silently erase every prior leaderboard row the first call had written --
    including, in this session's own case, the original `core-demo-2026`
    25-row demonstration leaderboard, recovered afterward from git history
    rather than lost. A benchmark leaderboard is meant to accumulate one
    row set per real run, the same append-only discipline
    docs/LIMITATIONS.md's own journal already follows -- this function now
    matches that instead of contradicting it.
    """
    name = "metrics_synthetic.parquet" if synthetic else "metrics.parquet"
    # Written as JSON-lines with a .parquet-shaped sibling name reserved for
    # the pandas/polars conversion step in benchmark.py; store.py itself only
    # needs a dependency-free, diffable format for small result sets.
    path = _dir("results") / name.replace(".parquet", ".jsonl")
    existing = load_result_records(synthetic=synthetic)
    combined = existing + records
    with path.open("w", encoding="utf-8") as handle:
        for record in combined:
            if record.synthetic != synthetic:
                raise ResearchStoreError(
                    f"result {record.experiment_id!r} synthetic={record.synthetic} "
                    f"does not match requested synthetic={synthetic}; "
                    "real and synthetic results must not be mixed in one file")
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return path


def load_result_records(*, synthetic: bool) -> list[ResultRecord]:
    name = "metrics_synthetic.jsonl" if synthetic else "metrics.jsonl"
    path = _dir("results") / name
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(ResultRecord(**json.loads(line)))
    return records


# --- source registry -------------------------------------------------------

def load_source_registry() -> dict:
    path = _dir("sources") / "source_registry.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_source_registry(registry: dict) -> Path:
    path = _dir("sources") / "source_registry.yaml"
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    return path


__all__ = [
    "ResearchStoreError", "research_root",
    "save_dataset_manifest", "load_dataset_manifest", "list_dataset_manifests",
    "save_label_records", "load_label_records",
    "save_benchmark_spec", "load_benchmark_spec", "list_benchmark_specs",
    "save_result_records", "load_result_records",
    "load_source_registry", "save_source_registry",
]
