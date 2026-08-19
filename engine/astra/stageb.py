"""Resumable Stage-B injection-recovery comparisons.

The initial method comparison was deliberately only a small smoke study.  This
module is the durable path for the thousands-of-curves rerun: it checkpoints
each independent injection seed, records the exact input population and keeps
the CPU baseline usable in released ASTRA builds.  It never starts itself;
callers explicitly submit this long-running research action.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import config, evaluate, experiment, modalitymatrix, tensors

STAGE_B_SCHEMA_VERSION = 1
DEFAULT_SEEDS = (17, 29, 43, 59, 71)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def dataset_fingerprint(identities: list[dict], *, mode: str, length: int) -> str:
    """Hash the ordered input identities, not the injected labels."""
    digest = hashlib.sha256()
    digest.update(f"mode={mode}|length={length}".encode("utf-8"))
    for identity in identities:
        digest.update(json.dumps(
            {key: identity.get(key) for key in
             ("survey", "release", "object_id", "band", "path")},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"))
    return digest.hexdigest()[:24]


def _strata(identities: list[dict]) -> dict[str, int]:
    grouped = Counter(
        f"{row.get('survey', 'unknown')}|{row.get('release', 'unknown')}|{row.get('band', 'unknown')}"
        for row in identities
    )
    return dict(sorted(grouped.items()))


def _aggregate(seed_results: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in seed_results:
        for method in result.get("methods", []):
            grouped[str(method["name"])].append(method)

    rows = []
    for name, methods in sorted(grouped.items()):
        metrics: dict[str, Any] = {"name": name, "runs": len(methods)}
        for metric in ("roc_auc", "average_precision", "precision_at_k", "recall_at_k"):
            values = np.asarray([item.get(metric) for item in methods], dtype=float)
            values = values[np.isfinite(values)]
            metrics[metric] = (None if not len(values) else {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
                "ci95": [round(float(np.quantile(values, 0.025)), 4),
                         round(float(np.quantile(values, 0.975)), 4)],
            })
        rows.append(metrics)
    return rows


def run(*, survey: str | None = None, seeds: tuple[int, ...] = DEFAULT_SEEDS,
        fraction: float = 0.1, strength: float = 6.0, limit: int = 10_000,
        mode: str = "time", include_deep: bool = False, epochs: int = 20,
        root: Path | None = None, checkpoint: Path | None = None) -> dict:
    """Run or resume a Stage-B comparison and record it as an experiment.

    ``include_deep`` is opt-in because production installers intentionally do
    not ship PyTorch. The baseline study still creates a reproducible Stage-B
    record, rather than pretending a CPU-only build is unable to contribute.
    """
    unique_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if len(unique_seeds) < 2:
        raise ValueError("Stage-B comparison needs at least two independent seeds")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")

    batch = tensors.build(survey=survey, limit=limit, mode=mode)
    fingerprint = dataset_fingerprint(batch.identities, mode=mode, length=batch.length)
    workspace = root or config.PATHS.projects
    checkpoint_path = checkpoint or (workspace / "results" / "stage-b" / "comparison.json")
    sidecars = modalitymatrix.list_sidecars(workspace)
    configuration = {
        "stage": "B",
        "survey": survey,
        "seeds": list(unique_seeds),
        "fraction": fraction,
        "strength": strength,
        "limit": limit,
        "resample_mode": mode,
        "include_deep": include_deep,
        "epochs": epochs,
        "dataset_fingerprint": fingerprint,
        "sequence": batch.to_dict(),
        "survey_release_band_strata": _strata(batch.identities),
        "sidecar_schemas": sidecars,
        "sidecar_policy": "recorded for provenance; not joined into sequence models",
    }

    def work() -> dict:
        state: dict[str, Any] = {}
        if checkpoint_path.exists():
            try:
                state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
        compatible = (
            state.get("schema_version") == STAGE_B_SCHEMA_VERSION
            and state.get("dataset_fingerprint") == fingerprint
            and state.get("configuration") == configuration
        )
        completed: dict[str, dict] = state.get("completed", {}) if compatible else {}
        if len(batch) < 20:
            return {
                "ready": False,
                "reason": f"only {len(batch)} usable sequences; need at least 20",
                "dataset_fingerprint": fingerprint,
                "checkpoint": str(checkpoint_path),
            }

        for seed in unique_seeds:
            key = str(seed)
            if key in completed:
                continue
            injection = evaluate.build_injected(
                batch.values, batch.identities, fraction=fraction, strength=strength, seed=seed,
            )
            completed[key] = evaluate.compare_on_sequences(
                injection, include_deep=include_deep, epochs=epochs, seed=seed,
            ).to_dict()
            _atomic_json(checkpoint_path, {
                "schema_version": STAGE_B_SCHEMA_VERSION,
                "dataset_fingerprint": fingerprint,
                "configuration": configuration,
                "completed": completed,
            })

        per_seed = [{"seed": seed, **completed[str(seed)]} for seed in unique_seeds]
        return {
            "ready": True,
            "dataset_fingerprint": fingerprint,
            "checkpoint": str(checkpoint_path),
            "resumed": compatible and bool(completed),
            "seeds": list(unique_seeds),
            "per_seed": per_seed,
            "aggregate": _aggregate(per_seed),
            "confidence_interval": "empirical 2.5th–97.5th percentile across independent injection seeds",
            "caveat": "Seed intervals are not population confidence intervals; injected shapes bound this claim.",
        }

    record = experiment.run(
        "stage_b_injection_recovery", configuration, work, seed=unique_seeds[0],
        dataset_hash=fingerprint, root=workspace,
        notes="Resumable Stage-B injection-recovery comparison; only explicit runs make performance claims.",
    )
    return {"experiment_id": record.provenance.experiment_id, **record.results}
