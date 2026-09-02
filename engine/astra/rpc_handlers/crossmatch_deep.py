"""Cross-survey matching/profiling, the BJD_TDB frame-offset measurement,
deep-model train/compare/sweep, and the background job queue.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import common
from .common import Handler, _workspace_root

from .. import config, crossmatch, evaluate, evidence, jobs, metadata, surveys, tensors, timeframe

def _sources_by_survey(root: Path | None = None) -> dict[str, list]:
    index = evidence.load_curves_by_key(root=config.PATHS.datasets)
    by_survey: dict[str, list] = {}
    for (survey, _oid), curves in index.items():
        by_survey.setdefault(survey, []).append(curves[0].source)
    from ..surveys.base import SourceRef
    for row in metadata.list_sources(root or config.PATHS.projects):
        entries = by_survey.setdefault(row["survey"], [])
        if not any(source.object_id == row["object_id"] for source in entries):
            entries.append(SourceRef(survey=row["survey"], object_id=row["object_id"],
                                     ra_deg=row["ra_deg"], dec_deg=row["dec_deg"],
                                     extra=row["extra"]))
    return by_survey


def _handle_crossmatch(params: dict[str, Any]) -> dict[str, Any]:
    """Group stored sources across surveys and summarise the result."""
    radius = float(params.get("radius_arcsec", crossmatch.DEFAULT_RADIUS_ARCSEC))
    anchor_survey = params.get("anchor_survey")
    groups = crossmatch.group_sources(
        _sources_by_survey(_workspace_root(params.get("project_id"))),
        radius_arcsec=radius, anchor_survey=anchor_survey)

    summary = crossmatch.summarise(groups)
    summary["resolved_multi_survey"] = sum(1 for g in groups
                                           if g.resolved_surveys > 1)
    summary["grouping_bias"] = crossmatch.grouping_bias_report(
        _sources_by_survey(_workspace_root(params.get("project_id"))), groups,
        anchor_survey=anchor_survey)
    top = int(params.get("top", 50))
    return {
        "summary": summary,
        "groups": [g.to_dict() for g in
                   sorted(groups, key=lambda g: -g.resolved_surveys)[:top]],
    }


def _handle_profile(params: dict[str, Any]) -> dict[str, Any]:
    """Full cross-survey evidence profiles, ranked by consistency."""
    radius = float(params.get("radius_arcsec", crossmatch.DEFAULT_RADIUS_ARCSEC))
    index = evidence.load_curves_by_key(root=config.PATHS.datasets)
    by_survey = _sources_by_survey(_workspace_root(params.get("project_id")))
    groups = crossmatch.group_sources(by_survey, radius_arcsec=radius,
                                      anchor_survey=params.get("anchor_survey"))
    if params.get("multi_survey_only", True):
        groups = [g for g in groups if g.independent_surveys > 1]

    profiles = [evidence.profile_group(g, index)
                for g in groups[:int(params.get("limit", 100))]]
    profiles.sort(key=lambda p: -p.consistency)

    top = int(params.get("top", 25))
    return {
        "profiled": len(profiles),
        "profiles": [p.to_dict() for p in profiles[:top]],
    }


def _handle_frame_offset(params: dict[str, Any]) -> dict[str, Any]:
    """Measured size of the time-frame correction at a given position."""
    offset = timeframe.measure_frame_offset(
        params.get("time_system", "HJD_UTC"),
        float(params["ra_deg"]), float(params["dec_deg"]),
        float(params.get("reference_jd", 2458600.5)),
        params.get("survey", "ZTF"),
    )
    return offset.to_dict()


DEEP_UNAVAILABLE = (
    "PyTorch is not available in this build, so deep models cannot run. "
    "Released ASTRA installers ship a CPU-only engine that deliberately "
    "excludes PyTorch and CUDA — they would add roughly 3.5 GB to the "
    "installer for a capability most sessions never use. Everything else "
    "(acquisition, features, baseline anomaly detection, cross-survey "
    "matching, ranking and export) works normally. To train deep models, "
    "run the engine from a development checkout with the 'gpu' extra "
    "installed: uv pip install -e engine[gpu]"
)


def _handle_deep_train(params: dict[str, Any]) -> dict[str, Any]:
    """Train one deep model on stored sequences. Minutes, not seconds."""
    common._require_torch()
    from .. import train as train_mod

    batch = tensors.build(survey=params.get("survey"),
                          limit=int(params.get("limit", 10_000)))
    if len(batch) < 20:
        return {"error": f"only {len(batch)} usable sequences; need at least 20",
                "rows": len(batch)}

    train_values, val_values, _, _ = tensors.train_test_split(batch)
    cfg = train_mod.TrainConfig(
        kind=params.get("kind", "autoencoder"),
        epochs=int(params.get("epochs", 30)),
        seed=int(params.get("seed", 42)),
        model=train_mod.ModelConfig(length=batch.length),
    )
    name = params.get("name", "default")
    report = train_mod.train(train_values, val_values, cfg, name=name)
    train_mod.save_report(report, name)
    return {**report.to_dict(), "sequences": batch.to_dict()}


def _handle_deep_compare(params: dict[str, Any]) -> dict[str, Any]:
    """Injection-recovery study comparing baselines against the deep models.

    The baseline half of the study runs fine without PyTorch, so only the
    deep-model request is refused; `include_deep=False` remains available in
    a packaged build and still produces a usable comparison.
    """
    if bool(params.get("include_deep", True)):
        common._require_torch()

    batch = tensors.build(survey=params.get("survey"),
                          limit=int(params.get("limit", 10_000)))
    if len(batch) < 20:
        return {"error": f"only {len(batch)} usable sequences; need at least 20",
                "rows": len(batch)}

    injection = evaluate.build_injected(
        batch.values, batch.identities,
        fraction=float(params.get("fraction", 0.1)),
        strength=float(params.get("strength", 6.0)),
        seed=int(params.get("seed", 42)),
    )
    comparison = evaluate.compare_on_sequences(
        injection,
        include_deep=bool(params.get("include_deep", True)),
        epochs=int(params.get("epochs", 20)),
        seed=int(params.get("seed", 42)),
    )
    return comparison.to_dict()


def _handle_deep_sweep(params: dict[str, Any]) -> dict[str, Any]:
    """Grid search over deep-model hyperparameters. Long-running by nature."""
    common._require_torch()
    from .. import sweep as sweep_mod

    seeds = tuple(int(seed) for seed in params.get("seeds", sweep_mod.DEFAULT_SEEDS))
    # `overrides` narrows the grid and `limit` caps the population. Without
    # both, the only sweep reachable over RPC is the full default grid on the
    # whole population, which is the one run nobody can afford to start.
    raw_overrides = params.get("overrides")
    overrides = None
    if raw_overrides:
        overrides = {str(key): tuple(value)
                     for key, value in dict(raw_overrides).items()}
    return sweep_mod.run_recorded(
        kind=params.get("kind", "autoencoder"),
        survey=params.get("survey"),
        seeds=seeds,
        mode=params.get("mode", "time"),
        fraction=float(params.get("fraction", 0.1)),
        epochs=int(params.get("epochs", 20)),
        limit=int(params.get("limit", 10_000)),
        strength=float(params.get("strength", 6.0)),
        overrides=overrides,
    )


def _handle_job_submit(params: dict[str, Any]) -> dict[str, Any]:
    method = str(params["method"])
    if method.startswith("job.") or method not in HANDLERS:
        raise ValueError("job method must name a registered science handler")
    request_params = params.get("params") or {}
    return jobs.submit(
        method,
        request_params,
        HANDLERS[method],
        project_id=params.get("project_id") or request_params.get("project_id"),
        idempotency_key=params.get("idempotency_key") or request_params.get("idempotency_key"),
    )


def _handle_job_status(params: dict[str, Any]) -> dict[str, Any]:
    return jobs.status(str(params["job_id"]))


def _handle_job_cancel(params: dict[str, Any]) -> dict[str, Any]:
    return jobs.cancel(str(params["job_id"]))


def _handle_job_retry(params: dict[str, Any]) -> dict[str, Any]:
    return jobs.retry(str(params["job_id"]), HANDLERS)


def _handle_job_list(params: dict[str, Any]) -> list[dict]:
    statuses = params.get("statuses")
    values = tuple(str(value) for value in statuses) if isinstance(statuses, list) else None
    return jobs.list_all(values)


HANDLERS: dict[str, Handler] = {
    "crossmatch.run": _handle_crossmatch,
    "crossmatch.profile": _handle_profile,
    "timeframe.offset": _handle_frame_offset,
    "deep.train": _handle_deep_train,
    "deep.compare": _handle_deep_compare,
    "deep.sweep": _handle_deep_sweep,
    "job.submit": _handle_job_submit,
    "job.status": _handle_job_status,
    "job.cancel": _handle_job_cancel,
    "job.retry": _handle_job_retry,
    "job.list": _handle_job_list,
}
