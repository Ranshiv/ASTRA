"""Experiment records, signed reproducibility bundles, research benchmarks,
ablation studies, artifact-weight calibration, the Stage-B scale comparison,
and the (rarely run) full pipeline handler.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .common import Handler, _workspace_root

from .. import (ablation, artifact, config, experiment, featurematrix,
                pipeline, security, stageb)
from .. import manifest as manifest_mod
from .. import reproducibility_bundle as bundle_mod

def _handle_experiment_list(params: dict[str, Any]) -> list[dict]:
    return experiment.list_experiments(_workspace_root(params.get("project_id")))


def _handle_experiment_get(params: dict[str, Any]) -> dict[str, Any]:
    return experiment.load(params["experiment_id"], _workspace_root(params.get("project_id"))).to_dict()


def _handle_experiment_verify(params: dict[str, Any]) -> dict[str, Any]:
    return experiment.verify(params["experiment_id"], _workspace_root(params.get("project_id")))


def _handle_experiment_compare(params: dict[str, Any]) -> dict[str, Any]:
    return experiment.compare(params["experiment_ids"],
                              params.get("metric", "roc_auc"),
                              _workspace_root(params.get("project_id")))


def _bundle_dir(root: Path | None) -> Path:
    directory = (root or config.PATHS.projects) / "signed_bundles"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _bundle_keypair(root: Path | None) -> "bundle_mod.KeyPair":
    """One persistent Ed25519 keypair per project root, generated on first
    use. This is a lab's own reproducibility-signing key (proves *which
    installation* produced a bundle, so a tampered copy is detectable) --
    not a security credential, so storing the raw private key alongside the
    other project files is an acceptable tradeoff for what this key
    protects against."""
    key_path = _bundle_dir(root) / "signing_key.hex"
    if key_path.exists():
        seed = bytes.fromhex(key_path.read_text(encoding="utf-8").strip())
        return bundle_mod.generate_keypair(seed=seed)
    keypair = bundle_mod.generate_keypair()
    from cryptography.hazmat.primitives import serialization
    raw = keypair.private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption())
    key_path.write_text(raw.hex(), encoding="utf-8")
    return keypair


def _handle_research_bundle_build(params: dict[str, Any]) -> dict[str, Any]:
    root = _workspace_root(params.get("project_id"))
    dataset_id = params["dataset_id"]
    m = manifest_mod.load(dataset_id, root)
    if m.content_hash is None:
        raise ValueError(f"manifest {dataset_id!r} is not sealed; call manifest.seal first")
    bundle = bundle_mod.build_bundle(m, experiment_record_refs=params.get("experiment_ids", []))
    signed = bundle_mod.sign_bundle(bundle, _bundle_keypair(root))
    path = security.scoped_id_path(_bundle_dir(root), dataset_id)
    path.write_text(json.dumps(signed.to_dict(), indent=2), encoding="utf-8")
    return {**signed.to_dict(), "path": str(path)}


def _handle_research_bundle_verify(params: dict[str, Any]) -> dict[str, Any]:
    root = _workspace_root(params.get("project_id"))
    dataset_id = params["dataset_id"]
    path = security.scoped_id_path(_bundle_dir(root), dataset_id)
    if not path.exists():
        return {"dataset_id": dataset_id, "valid": False, "note": "no bundle recorded"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["experiment_record_refs"] = tuple(payload.get("experiment_record_refs", []))
    bundle = bundle_mod.ReproducibilityBundle(**payload)
    return {"dataset_id": dataset_id, "valid": bundle_mod.verify_bundle(bundle)}


def _handle_research_bundle_rerun(params: dict[str, Any]) -> dict[str, Any]:
    """Query-provenance half of rerun verification: a freshly-built manifest
    (same queries, re-run) should re-derive the bundle's recorded content
    hash. The caller supplies the fresh manifest's queries, mirroring
    `manifest.SurveyQuery`; this endpoint does not itself re-query archives.
    """
    root = _workspace_root(params.get("project_id"))
    dataset_id = params["dataset_id"]
    path = security.scoped_id_path(_bundle_dir(root), dataset_id)
    if not path.exists():
        return {"dataset_id": dataset_id, "matches": False, "note": "no bundle recorded"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["experiment_record_refs"] = tuple(payload.get("experiment_record_refs", []))
    bundle = bundle_mod.ReproducibilityBundle(**payload)

    fresh = manifest_mod.Manifest(dataset_id=dataset_id)
    for query in params.get("fresh_queries", []):
        fresh.add(manifest_mod.SurveyQuery(**query))
    return bundle_mod.verify_manifest_rerun(bundle, fresh)


def _handle_research_benchmark_run(params: dict[str, Any]) -> dict[str, Any]:
    """Run the cross-survey anomaly benchmark against a saved feature
    matrix, writing bound ResultRecords and an experiment record. Slow:
    scores five baselines at every declared seed."""
    from ..research import benchmark as benchmark_mod, splits as splits_mod, store as research_store

    matrix = featurematrix.load(params["matrix_name"],
                                _workspace_root(params.get("project_id")))
    spec = research_store.load_benchmark_spec(params["benchmark_id"])
    split = splits_mod.load(params["split_id"], research_store.research_root())
    manifest = research_store.load_dataset_manifest(params["dataset_id"])

    # A saved feature matrix can carry rows from every acquisition ever run
    # for this survey, not only the ones this manifest's cone matched -- so
    # score exactly the manifest's own object IDs, or `dataset_manifest_hash`
    # below certifies the query but not the rows actually scored (see
    # docs/RESULTS.md's "Reading these numbers correctly").
    manifest_object_ids = {obj for q in manifest.queries for obj in q.object_ids}
    matrix, dropped_out_of_manifest = benchmark_mod.scope_to_manifest(
        matrix, manifest_object_ids)

    record = experiment.create(
        "research_benchmark", {"matrix_name": params["matrix_name"],
                               "dropped_out_of_manifest_rows": dropped_out_of_manifest},
        benchmark_id=spec.benchmark_id, split_id=split.split_id,
        manifest_content_hash=manifest.content_hash,
        result_artifact_paths=["research/results/metrics_synthetic.jsonl"],
        root=_workspace_root(params.get("project_id")))
    record.provenance.model_version = "ensemble+baselines"

    started = time.time()
    run_result = benchmark_mod.run_cross_survey_anomaly(
        matrix, spec, split, experiment_id=record.provenance.experiment_id,
        dataset_manifest_hash=manifest.content_hash or "",
        injection_fraction=float(params.get("injection_fraction", 0.1)))
    record.runtime_seconds = time.time() - started
    record.results = {"n_results": len(run_result.results)}
    experiment.save(record, _workspace_root(params.get("project_id")))

    research_store.save_result_records(run_result.results, synthetic=True)
    return {**run_result.to_dict(), "experiment_id": record.provenance.experiment_id,
            "matrix_rows_scored": len(matrix),
            "dropped_out_of_manifest_rows": dropped_out_of_manifest}


def _handle_ablation(params: dict[str, Any]) -> dict[str, Any]:
    """Run the section 20 experiment groups plus the ablations. Slow."""
    kwargs = {"fraction": float(params.get("fraction", 0.1)),
              "seed": int(params.get("seed", 42)),
              "survey": params.get("survey")}
    if params.get("project_id"):
        kwargs["root"] = _workspace_root(params.get("project_id"))
    return ablation.run_all(**kwargs)


def _handle_ablation_repeated(params: dict[str, Any]) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in params.get("seeds", (17, 29, 43, 59, 71)))
    kwargs = {"fraction": float(params.get("fraction", 0.1)),
              "seeds": seeds, "survey": params.get("survey")}
    if params.get("checkpoint"):
        kwargs["checkpoint"] = security.authorized_write_path(params["checkpoint"], config.PATHS.root)
    if params.get("project_id"):
        kwargs["root"] = _workspace_root(params.get("project_id"))
    return ablation.run_repeated(**kwargs)


def _handle_artifact_calibrate(params: dict[str, Any]) -> dict[str, Any]:
    """Propose artifact indicator weights from injection. Never adopts them."""
    seeds = tuple(int(seed) for seed in params.get("seeds", (17, 29, 43, 59, 71)))
    kwargs: dict[str, Any] = {
        "n_per_class": int(params.get("n_per_class", 150)),
        "test_fraction": float(params.get("test_fraction", 0.3)),
        "seeds": seeds,
        "hard_real_fraction": float(
            params.get("hard_real_fraction", artifact.DEFAULT_HARD_REAL_FRACTION)),
    }
    if params.get("project_id"):
        kwargs["root"] = _workspace_root(params.get("project_id"))
    return artifact.calibrate_recorded(**kwargs)


def _handle_stageb_compare(params: dict[str, Any]) -> dict[str, Any]:
    """Explicit, checkpointed Stage-B scale comparison; never auto-started."""
    seeds = tuple(int(seed) for seed in params.get("seeds", stageb.DEFAULT_SEEDS))
    workspace = _workspace_root(params.get("project_id"))
    checkpoint = (security.authorized_write_path(params["checkpoint"], config.PATHS.root)
                  if params.get("checkpoint") else None)
    return stageb.run(
        survey=params.get("survey"), seeds=seeds,
        fraction=float(params.get("fraction", 0.1)),
        strength=float(params.get("strength", 6.0)),
        limit=int(params.get("limit", 10_000)), mode=str(params.get("mode", "time")),
        include_deep=bool(params.get("include_deep", False)),
        epochs=int(params.get("epochs", 20)), root=workspace, checkpoint=checkpoint,
    )


def _handle_pipeline(params: dict[str, Any]) -> dict[str, Any]:
    """Full candidate generation. Minutes on a real dataset."""
    built, report = pipeline.run(
        survey_names=params.get("surveys"),
        radius_arcsec=float(params.get("radius_arcsec", 15.0)),
        contamination=float(params.get("contamination", 0.05)),
        top=int(params.get("top", 200)),
        name=params.get("name", "default"),
        seed=int(params.get("seed", 42)),
        root=_workspace_root(params.get("project_id")),
        anchor_survey=params.get("anchor_survey"),
    )
    preview = int(params.get("preview", 25))
    return {**report.to_dict(),
            "candidates": [c.to_dict() for c in built[:preview]]}


HANDLERS: dict[str, Handler] = {
    "experiment.list": _handle_experiment_list,
    "experiment.get": _handle_experiment_get,
    "experiment.verify": _handle_experiment_verify,
    "experiment.compare": _handle_experiment_compare,
    "research.bundle.build": _handle_research_bundle_build,
    "research.bundle.verify": _handle_research_bundle_verify,
    "research.bundle.rerun": _handle_research_bundle_rerun,
    "research.benchmark.run": _handle_research_benchmark_run,
    "ablation.run": _handle_ablation,
    "ablation.repeated": _handle_ablation_repeated,
    "artifact.calibrate": _handle_artifact_calibrate,
    "stageb.compare": _handle_stageb_compare,
    "pipeline.run": _handle_pipeline,
}
