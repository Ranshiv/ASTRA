"""Core engine handlers: hardware/paths/cache/versions, survey discovery,
acquisition, product listing, and the baseline feature/anomaly pipeline.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

import sys
from typing import Any

from .common import Handler, PROTOCOL_VERSION, _workspace_root

from .. import (acquire, anomaly, cache, config, featurematrix, features,
                hardware, products, readiness, security, store, surveys)
from ..surveys.base import ConeQuery

def _handle_ping(_params: dict[str, Any]) -> dict[str, Any]:
    return {"pong": True, "protocol": PROTOCOL_VERSION}


def _handle_hardware(_params: dict[str, Any]) -> dict[str, Any]:
    return hardware.select_device().to_dict()


def _handle_paths(_params: dict[str, Any]) -> dict[str, Any]:
    paths = config.PATHS
    return {
        "root": str(paths.root),
        "projects": str(paths.projects),
        "datasets": str(paths.datasets),
        "models": str(paths.models),
        "cache": str(paths.cache),
        "logs": str(paths.logs),
        "config": str(paths.config),
    }


def _handle_cache_status(_params: dict[str, Any]) -> dict[str, Any]:
    return cache.measure().to_dict()


def _handle_cache_enforce(_params: dict[str, Any]) -> dict[str, Any]:
    return cache.enforce_cap().to_dict()


def _handle_versions(_params: dict[str, Any]) -> dict[str, Any]:
    """Report the science stack, so the UI can prove the engine is wired up."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for module in ("numpy", "scipy", "astropy", "astroquery", "lightkurve",
                   "polars", "duckdb", "sklearn", "torch"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - absence is information, not failure
            versions[module] = "not installed"
    return versions


def _handle_surveys_list(_params: dict[str, Any]) -> list[dict]:
    return surveys.describe_all()


def _handle_readiness_status(_params: dict[str, Any]) -> dict[str, Any]:
    return readiness.status()


def _handle_acquire(params: dict[str, Any], progress=None) -> dict[str, Any]:
    """Run one cone acquisition. Long-running: the UI shows a spinner."""
    query = ConeQuery(
        ra_deg=float(params["ra_deg"]),
        dec_deg=float(params["dec_deg"]),
        radius_arcsec=float(params.get("radius_arcsec", 10.0)),
    )
    result = acquire.acquire(
        query,
        survey_names=params.get("surveys"),
        limit=int(params.get("limit", 25)),
        dataset_id=params.get("dataset_id"),
        project_id=params.get("project_id"),
        skip_existing=bool(params.get("skip_existing", True)),
        progress=progress,
        # Per-survey connector kwargs, e.g. {"tess": {"author": "QLP"}} to
        # acquire via QLP instead of the default SPOC. Optional; absent, this
        # reproduces the previous call exactly.
        survey_options=params.get("survey_options"),
    )
    return result.to_dict()


def _handle_acquire_project(params: dict[str, Any], progress=None) -> dict[str, Any]:
    """Run one acquisition per region in a project's query_regions. Long-running."""
    result = acquire.acquire_project(
        str(params["project_id"]),
        survey_names=params.get("surveys"),
        limit=int(params.get("limit", 25)),
        skip_existing=bool(params.get("skip_existing", True)),
        progress=progress,
        survey_options=params.get("survey_options"),
    )
    return result.to_dict()


def _handle_store_usage(_params: dict[str, Any]) -> dict[str, Any]:
    return {"surveys": store.survey_usage(), "dataset": store.dataset_status()}


def _handle_feature_cache_clear(_params: dict[str, Any]) -> dict[str, Any]:
    from .. import featurecache

    removed = featurecache.clear()
    return {"cleared": removed}


def _handle_profile_run(params: dict[str, Any]) -> dict[str, Any]:
    """Measure where pipeline time goes. Slow by nature."""
    from .. import profiling

    return profiling.run_all(limit=int(params.get("limit", 100)))


def _handle_products_list(params: dict[str, Any]) -> list[dict[str, Any]]:
    return products.list_products(limit=int(params.get("limit", 500)),
                                  project_id=params.get("project_id"))


def _handle_products_get(params: dict[str, Any]) -> dict[str, Any]:
    return products.get_product(str(params["product_id"]))


def _handle_features_build(params: dict[str, Any]) -> dict[str, Any]:
    """Extract features over the store and persist the matrix."""
    workspace = _workspace_root(params.get("project_id"))
    matrix = featurematrix.build(
        survey=params.get("survey"),
        limit=int(params.get("limit", 10_000)),
    )
    name = params.get("name", "default")
    path = featurematrix.save(matrix, name, workspace)
    return {**matrix.to_dict(), "name": name, "path": str(path)}


def _handle_features_build_resumable(params: dict[str, Any]) -> dict[str, Any]:
    """Run the checkpointed streaming extractor used for Stage C/D batches."""
    workspace = _workspace_root(params.get("project_id"))
    checkpoint = (security.authorized_write_path(params["checkpoint"], config.PATHS.root)
                  if params.get("checkpoint") else None)
    if checkpoint is None and workspace is not None:
        checkpoint = workspace / "features" / "checkpoints" / f"{params.get('name', 'default')}.json"
    matrix, report = featurematrix.build_resumable(
        survey=params.get("survey"),
        limit=int(params.get("limit", 10_000)),
        batch_size=int(params.get("batch_size", 256)),
        checkpoint=checkpoint,
    )
    name = str(params.get("name", "default"))
    path = featurematrix.save(matrix, name, workspace)
    return {**matrix.to_dict(), **report.to_dict(), "name": name, "path": str(path)}


def _handle_features_list(params: dict[str, Any]) -> list[dict]:
    return featurematrix.list_matrices(_workspace_root(params.get("project_id")))


def _handle_feature_names(_params: dict[str, Any]) -> dict[str, Any]:
    return {"names": list(features.FEATURE_NAMES),
            "feature_version": features.FEATURE_VERSION}


def _handle_detect(params: dict[str, Any]) -> dict[str, Any]:
    """Run the baseline ensemble over a saved feature matrix."""
    name = params.get("name", "default")
    workspace = _workspace_root(params.get("project_id"))
    matrix = featurematrix.load(name, workspace)
    result = anomaly.detect(
        matrix,
        contamination=float(params.get("contamination",
                                       anomaly.DEFAULT_CONTAMINATION)),
        seed=int(params.get("seed", 42)),
    )
    top = int(params.get("top", 50))
    path = anomaly.save_ranking(result, name, top=max(top, 200), root=workspace)

    # Cross-run calibration: score this run against the persisted reference
    # from prior runs of this same feature-matrix name (empty/batch-relative
    # on the first-ever run), then fold this run's scores in for the next one.
    reference = anomaly.load_calibration_reference(name, workspace)
    reference = reference if reference.size else None
    anomaly.update_calibration_reference(name, workspace, result.consensus)

    return {**result.to_dict(reference=reference),
            "candidates": result.ranked(top, reference=reference),
            "ranking_path": str(path)}


HANDLERS: dict[str, Handler] = {
    "ping": _handle_ping,
    "hardware": _handle_hardware,
    "paths": _handle_paths,
    "cache.status": _handle_cache_status,
    "cache.enforce": _handle_cache_enforce,
    "versions": _handle_versions,
    "surveys.list": _handle_surveys_list,
    "readiness.status": _handle_readiness_status,
    "acquire.cone": _handle_acquire,
    "acquire.project": _handle_acquire_project,
    "store.usage": _handle_store_usage,
    "products.list": _handle_products_list,
    "products.get": _handle_products_get,
    "profile.run": _handle_profile_run,
    "cache.features.clear": _handle_feature_cache_clear,
    "features.build": _handle_features_build,
    "features.build_resumable": _handle_features_build_resumable,
    "features.list": _handle_features_list,
    "features.names": _handle_feature_names,
    "anomaly.detect": _handle_detect,
}
