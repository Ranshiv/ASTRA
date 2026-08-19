"""JSON-lines RPC bridge between the Rust core and the Python engine.

Rust spawns this process and speaks one JSON object per line over stdin;
the engine answers with one JSON object per line on stdout. Line-delimited
JSON is used rather than a socket so the transport has no port to collide
with and dies automatically with the parent process.

Every response carries the request `id` so the Rust side can correlate
replies, and errors are returned as values rather than raised, so a failing
handler never kills the engine.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from . import (ablation, acquire, anomaly, cache, candidates as candidates_mod,
               catalogs, config, credentials, crossmatch, evaluate, evidence,
               experiment, exports, featurematrix, features, fitsio, hardware,
               image_features, modalitymatrix, readiness, spectral_features,
               jobs, manifest as manifest_mod, metadata, pipeline, products,
               project as project_mod, review, ranker, security, store, surveys,
               stageb, tensors, tess_pixels, timeframe, viz)
from .surveys.base import ConeQuery

Handler = Callable[[dict[str, Any]], Any]

PROTOCOL_VERSION = 1


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
        progress=progress,
        # Per-survey connector kwargs, e.g. {"tess": {"author": "QLP"}} to
        # acquire via QLP instead of the default SPOC. Optional; absent, this
        # reproduces the previous call exactly.
        survey_options=params.get("survey_options"),
    )
    return result.to_dict()


def _handle_store_usage(_params: dict[str, Any]) -> dict[str, Any]:
    return {"surveys": store.survey_usage(), "dataset": store.dataset_status()}


def _handle_manifest_list(params: dict[str, Any]) -> list[dict]:
    project_id = params.get("project_id")
    root = project_mod.manifest_dir(str(project_id)) if project_id else None
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


def _workspace_root(project_id: object) -> Path | None:
    """Resolve an optional project workspace without weakening path checks."""
    if project_id is None or str(project_id).strip() == "":
        return None
    return project_mod.project_dir(str(project_id))


def _handle_curves_list(params: dict[str, Any]) -> list[dict]:
    return viz.list_curves(survey=params.get("survey"),
                           limit=int(params.get("limit", 500)),
                           root=config.PATHS.datasets)


def _handle_curves_get(params: dict[str, Any]) -> dict[str, Any]:
    return viz.curve_payload(
        security.authorized_path(params["path"]),
        max_points=int(params.get("max_points", viz.DEFAULT_MAX_POINTS)),
        frame=params.get("frame"),
    )


def _handle_curves_fold(params: dict[str, Any]) -> dict[str, Any]:
    return viz.fold(
        security.authorized_path(params["path"]),
        period_days=float(params["period_days"]),
        epoch=float(params["epoch"]) if params.get("epoch") is not None else None,
    )


def _handle_curves_bin(params: dict[str, Any]) -> dict[str, Any]:
    return viz.bin_curve(security.authorized_path(params["path"]),
                         bin_days=float(params["bin_days"]))


def _handle_fits_describe(params: dict[str, Any]) -> dict[str, Any]:
    return fitsio.describe(security.authorized_path(params["path"]))


def _handle_fits_header(params: dict[str, Any]) -> dict[str, Any]:
    return fitsio.read_header(security.authorized_path(params["path"]),
                              hdu=int(params.get("hdu", 0)))


def _handle_fits_image(params: dict[str, Any]) -> dict[str, Any]:
    return fitsio.image_payload(
        security.authorized_path(params["path"]),
        hdu=int(params["hdu"]) if params.get("hdu") is not None else None,
        contrast=float(params.get("contrast", 0.25)),
    )


def _cutout_request(params: dict[str, Any]) -> products.CutoutRequest:
    return products.CutoutRequest(
        ra_deg=float(params["ra_deg"]),
        dec_deg=float(params["dec_deg"]),
        size_arcsec=float(params.get("size_arcsec", 50.0)),
        product_kind=str(params.get("product_kind", "science")),
        release=str(params.get("release", products.ZTF_RELEASE)),
    )


def _handle_ztf_images_search(params: dict[str, Any]) -> list[dict[str, Any]]:
    request = _cutout_request(params)
    rows = products.search(request, limit=int(params.get("limit", 25)))
    return [{**row, "product_url": products.product_url(row),
             "cutout_url": products.cutout_url(row, request)} for row in rows]


def _handle_ztf_images_download(params: dict[str, Any], progress=None) -> dict[str, Any]:
    request = _cutout_request(params)
    row = params.get("metadata") or params.get("row")
    if not isinstance(row, dict):
        raise ValueError("metadata must be an object returned by ztf.images.search")
    return products.download_cutout(
        request, row,
        project_id=params.get("project_id"),
        max_bytes=int(params.get("max_bytes", products.DEFAULT_MAX_BYTES)),
        overwrite=bool(params.get("overwrite", False)),
        progress=(lambda received, total: progress.update(
            phase="download", message="Downloading ZTF FITS cutout",
            bytes_downloaded=received, bytes_total=total,
        ) if progress is not None else None),
    )


def _tpf_request(params: dict[str, Any]) -> tess_pixels.TPFRequest:
    """Build and validate a candidate-scale TESS TPF request."""
    if params.get("sector") is None:
        raise tess_pixels.TESSProductError("sector is required for a TESS TPF request")
    size_pixels = params.get("size_pixels")
    if size_pixels is None:
        size_pixels = tess_pixels.DEFAULT_SIZE_PIXELS
    return tess_pixels.TPFRequest(
        ra_deg=params.get("ra_deg"),
        dec_deg=params.get("dec_deg"),
        sector=params.get("sector"),
        size_pixels=size_pixels,
        target_id=params.get("target_id"),
        product=params.get("product", "SPOC"),
    )


def _optional_bool(params: dict[str, Any], name: str, default: bool) -> bool:
    value = params.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _handle_tess_tpf_download(params: dict[str, Any], progress=None) -> dict[str, Any]:
    request = _tpf_request(params)

    def report(received: int, total: int | None) -> None:
        if progress is None:
            return
        progress.update(
            phase="download",
            message="Downloading TESS target-pixel file",
            bytes_downloaded=received,
            bytes_total=total,
        )

    max_bytes = params.get("max_bytes")
    if max_bytes is None:
        max_bytes = tess_pixels.DEFAULT_MAX_BYTES
    return tess_pixels.download_tpf(
        request,
        project_id=params.get("project_id"),
        max_bytes=max_bytes,
        overwrite=_optional_bool(params, "overwrite", False),
        progress=report,
    )


def _handle_tess_tpf_photometry(params: dict[str, Any]) -> dict[str, Any]:
    raw_path = params.get("path") or params.get("tpf_path")
    if not raw_path:
        raise ValueError("path is required for TESS TPF photometry")
    path = security.authorized_path(str(raw_path))
    neighbours = params.get("neighbors") or []
    if not isinstance(neighbours, (list, tuple)):
        raise ValueError("neighbors must be an array")
    aperture = params.get("aperture_radius_pixels")
    if aperture is None:
        aperture = 1.5
    quality = params.get("quality_mask")
    if quality is None:
        quality = tess_pixels.DEFAULT_QUALITY_MASK
    common = {
        "ra_deg": params.get("ra_deg"),
        "dec_deg": params.get("dec_deg"),
        "neighbors": neighbours,
        "target_mag": params.get("target_mag"),
        "aperture_radius_pixels": aperture,
        "quality_mask": quality,
    }
    if _optional_bool(params, "persist", True):
        payload = tess_pixels.persist_photometry(
            path,
            target_id=params.get("target_id"),
            root=config.PATHS.datasets,
            **common,
        )
    else:
        payload = tess_pixels.extract_photometry(path, **common)
    max_points = params.get("max_points")
    if max_points is None:
        max_points = 5000
    return tess_pixels.json_payload(payload, max_points=max_points)


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
    checkpoint = Path(params["checkpoint"]) if params.get("checkpoint") else None
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
    return {**result.to_dict(), "candidates": result.ranked(top),
            "ranking_path": str(path)}


def _handle_profile_run(params: dict[str, Any]) -> dict[str, Any]:
    """Measure where pipeline time goes. Slow by nature."""
    from . import profiling

    return profiling.run_all(limit=int(params.get("limit", 100)))


def _handle_feature_cache_clear(_params: dict[str, Any]) -> dict[str, Any]:
    from . import featurecache

    removed = featurecache.clear()
    return {"cleared": removed}


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
    if params.get("project_id"):
        kwargs["root"] = _workspace_root(params.get("project_id"))
    return ablation.run_repeated(**kwargs)


def _handle_stageb_compare(params: dict[str, Any]) -> dict[str, Any]:
    """Explicit, checkpointed Stage-B scale comparison; never auto-started."""
    seeds = tuple(int(seed) for seed in params.get("seeds", stageb.DEFAULT_SEEDS))
    workspace = _workspace_root(params.get("project_id"))
    checkpoint = Path(params["checkpoint"]) if params.get("checkpoint") else None
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
    )
    preview = int(params.get("preview", 25))
    return {**report.to_dict(),
            "candidates": [c.to_dict() for c in built[:preview]]}


def _handle_image_features(params: dict[str, Any]) -> dict[str, Any]:
    payload = image_features.extract(
        security.authorized_path(params["path"]),
        hdu=int(params["hdu"]) if params.get("hdu") is not None else None,
        target_xy=(float(params["target_x"]), float(params["target_y"]))
        if params.get("target_x") is not None and params.get("target_y") is not None
        else None,
    )
    if any(params.get(key) is not None for key in ("survey", "release", "object_id", "band")):
        payload["identity"] = {key: params.get(key, "unknown")
                                for key in ("survey", "release", "object_id", "band")}
    project_id = params.get("project_id")
    if project_id:
        output = image_features.save(payload,
                                     project_mod.project_dir(str(project_id)) / "results" / "image_features")
        payload["output_path"] = str(output)
    return payload


def _handle_spectral_features(params: dict[str, Any]) -> dict[str, Any]:
    path = security.authorized_path(params["path"])
    payload = spectral_features.from_fits(path)
    if any(params.get(key) is not None for key in ("survey", "release", "object_id", "band")):
        payload["identity"] = {key: params.get(key, "unknown")
                                for key in ("survey", "release", "object_id", "band")}
    project_id = params.get("project_id")
    if project_id:
        output = spectral_features.save(payload,
                                        project_mod.project_dir(str(project_id)) / "results" / "spectral_features")
        payload["output_path"] = str(output)
    return payload


def _handle_sidecars_list(params: dict[str, Any]) -> list[dict]:
    return modalitymatrix.list_sidecars(_workspace_root(params.get("project_id")))


def _handle_sidecar_save(params: dict[str, Any]) -> dict[str, Any]:
    kind = str(params["kind"])
    payloads = params.get("payloads") or []
    if not isinstance(payloads, list):
        raise ValueError("payloads must be a list")
    result = modalitymatrix.save_payloads(
        payloads, kind, name=str(params.get("name", "default")),
        root=_workspace_root(params.get("project_id")),
        identities=params.get("identities"),
    )
    return result.to_dict()


def _handle_sidecar_join(params: dict[str, Any]) -> dict[str, Any]:
    path = security.authorized_path(params["path"])
    identities = params.get("identities") or []
    if not isinstance(identities, list):
        raise ValueError("identities must be a list")
    rows, report = modalitymatrix.join_rows(
        identities, modalitymatrix.load(path), kind=str(params["kind"]),
    )
    return {"rows": rows, "report": report}


def _handle_candidates_load(params: dict[str, Any]) -> dict[str, Any]:
    root = _workspace_root(params.get("project_id"))
    built = candidates_mod.load(params.get("name", "default"), root)
    top = int(params.get("top", 50))
    labels = candidates_mod.load_labels(root)
    return {
        "count": len(built),
        "candidates": [
            {**c.to_dict(), "label": labels.get(c.candidate_id, {}).get("label")}
            for c in built[:top]
        ],
    }


def _handle_candidate_get(params: dict[str, Any]) -> dict[str, Any]:
    root = _workspace_root(params.get("project_id"))
    built = candidates_mod.load(params.get("name", "default"), root)
    candidate_id = params["candidate_id"]
    labels = candidates_mod.load_labels(root)
    for candidate in built:
        if candidate.candidate_id == candidate_id:
            return {**candidate.to_dict(), "review": labels.get(candidate_id)}
    raise KeyError(f"candidate not found: {candidate_id}")


def _handle_candidate_timeline(params: dict[str, Any]) -> dict[str, Any]:
    return candidates_mod.timeline(
        str(params["candidate_id"]),
        params.get("name", "default"),
        _workspace_root(params.get("project_id")),
        radius_arcsec=float(params.get("radius_arcsec", 30.0)),
        max_curves=int(params.get("max_curves", 24)),
        max_points=int(params.get("max_points", 180)),
    )


def _handle_candidates_export(params: dict[str, Any]) -> dict[str, Any]:
    return exports.export_candidates(params.get("name", "default"),
                                     params.get("format", "csv"),
                                     _workspace_root(params.get("project_id")))


def _handle_label(params: dict[str, Any]) -> dict[str, Any]:
    entry = candidates_mod.record_label(
        params["candidate_id"], params["label"], params.get("note", ""),
        _workspace_root(params.get("project_id")))
    return {"candidate_id": params["candidate_id"], **entry}


def _handle_labels(params: dict[str, Any]) -> dict[str, Any]:
    return candidates_mod.label_summary(_workspace_root(params.get("project_id")))


def _handle_review_evaluate(params: dict[str, Any]) -> dict[str, Any]:
    return review.evaluate(params.get("name", "default"),
                           _workspace_root(params.get("project_id")))


def _handle_catalog_status(_params: dict[str, Any]) -> dict[str, Any]:
    return catalogs.status()


def _handle_catalog_enrich(params: dict[str, Any]) -> dict[str, Any]:
    """Optional explicit enrichment; candidate generation is always offline-safe."""
    return catalogs.enrich_candidates(
        name=params.get("name", "default"),
        radius_arcsec=float(params.get("radius_arcsec", 2.0)),
        refresh=bool(params.get("refresh", False)),
        offline=bool(params.get("offline", False)),
        include_tns=bool(params.get("include_tns", True)),
        root=_workspace_root(params.get("project_id")),
    )


def _handle_tns_credentials_configure(params: dict[str, Any]) -> dict[str, Any]:
    """Store the API key with Windows DPAPI; it is never echoed to the UI."""
    return credentials.save_tns_credentials(
        str(params["api_key"]), str(params.get("bot_id", "")),
        str(params.get("bot_name", "ASTRA")),
    )


def _handle_tns_credentials_clear(_params: dict[str, Any]) -> dict[str, Any]:
    return {"cleared": credentials.clear_tns_credentials()}


def _handle_ranker_train(params: dict[str, Any]) -> dict[str, Any]:
    return ranker.train(
        name=params.get("name", "default"),
        model_name=params.get("model_name", "calibrated-logistic"),
        seed=int(params.get("seed", 42)),
        bootstrap_samples=int(params.get("bootstrap_samples", ranker.BOOTSTRAP_SAMPLES)),
    )


def _handle_ranker_apply(params: dict[str, Any]) -> dict[str, Any]:
    return ranker.apply(name=params.get("name", "default"),
                        model_name=params.get("model_name", "calibrated-logistic"))


def _handle_ranker_list(_params: dict[str, Any]) -> list[dict]:
    return ranker.list_models()


def _sources_by_survey(root: Path | None = None) -> dict[str, list]:
    index = evidence.load_curves_by_key(root=config.PATHS.datasets)
    by_survey: dict[str, list] = {}
    for (survey, _oid), curves in index.items():
        by_survey.setdefault(survey, []).append(curves[0].source)
    from .surveys.base import SourceRef
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
    groups = crossmatch.group_sources(_sources_by_survey(_workspace_root(params.get("project_id"))), radius_arcsec=radius)

    summary = crossmatch.summarise(groups)
    summary["resolved_multi_survey"] = sum(1 for g in groups
                                           if g.resolved_surveys > 1)
    summary["grouping_bias"] = crossmatch.grouping_bias_report(
        _sources_by_survey(_workspace_root(params.get("project_id"))), groups)
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
    groups = crossmatch.group_sources(by_survey, radius_arcsec=radius)
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


def _require_torch() -> None:
    """Fail with an explanation rather than a bare ModuleNotFoundError.

    A packaged build genuinely cannot do this, so the message has to say why
    and what to do instead. "No module named 'torch'" reads like a broken
    installation; this is a deliberate build choice.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(DEEP_UNAVAILABLE) from exc


def _handle_deep_train(params: dict[str, Any]) -> dict[str, Any]:
    """Train one deep model on stored sequences. Minutes, not seconds."""
    _require_torch()
    from . import train as train_mod

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
        _require_torch()

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
    _require_torch()
    from . import sweep as sweep_mod

    seeds = tuple(int(seed) for seed in params.get("seeds", sweep_mod.DEFAULT_SEEDS))
    return sweep_mod.run_recorded(
        kind=params.get("kind", "autoencoder"),
        survey=params.get("survey"),
        seeds=seeds,
        mode=params.get("mode", "time"),
        fraction=float(params.get("fraction", 0.1)),
        epochs=int(params.get("epochs", 20)),
    )


def _handle_feature_names(_params: dict[str, Any]) -> dict[str, Any]:
    return {"names": list(features.FEATURE_NAMES),
            "feature_version": features.FEATURE_VERSION}


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
    "ping": _handle_ping,
    "hardware": _handle_hardware,
    "paths": _handle_paths,
    "cache.status": _handle_cache_status,
    "cache.enforce": _handle_cache_enforce,
    "versions": _handle_versions,
    "surveys.list": _handle_surveys_list,
    "readiness.status": _handle_readiness_status,
    "acquire.cone": _handle_acquire,
    "store.usage": _handle_store_usage,
    "manifest.list": _handle_manifest_list,
    "project.create": _handle_project_create,
    "project.list": _handle_project_list,
    "project.open": _handle_project_open,
    "project.update": _handle_project_update,
    "project.archive": _handle_project_archive,
    "project.validate": _handle_project_validate,
    "curves.list": _handle_curves_list,
    "curves.get": _handle_curves_get,
    "curves.fold": _handle_curves_fold,
    "curves.bin": _handle_curves_bin,
    "fits.describe": _handle_fits_describe,
    "fits.header": _handle_fits_header,
    "fits.image": _handle_fits_image,
    "image.features": _handle_image_features,
    "spectrum.features": _handle_spectral_features,
    "sidecars.list": _handle_sidecars_list,
    "sidecars.save": _handle_sidecar_save,
    "sidecars.join": _handle_sidecar_join,
    "ztf.images.search": _handle_ztf_images_search,
    "ztf.images.download": _handle_ztf_images_download,
    "tess.tpf.download": _handle_tess_tpf_download,
    "tess.tpf.photometry": _handle_tess_tpf_photometry,
    "products.list": _handle_products_list,
    "products.get": _handle_products_get,
    "profile.run": _handle_profile_run,
    "cache.features.clear": _handle_feature_cache_clear,
    "experiment.list": _handle_experiment_list,
    "experiment.get": _handle_experiment_get,
    "experiment.verify": _handle_experiment_verify,
    "experiment.compare": _handle_experiment_compare,
    "ablation.run": _handle_ablation,
    "ablation.repeated": _handle_ablation_repeated,
    "stageb.compare": _handle_stageb_compare,
    "pipeline.run": _handle_pipeline,
    "candidates.load": _handle_candidates_load,
    "candidates.get": _handle_candidate_get,
    "candidates.timeline": _handle_candidate_timeline,
    "candidates.export": _handle_candidates_export,
    "candidates.label": _handle_label,
    "candidates.labels": _handle_labels,
    "candidates.evaluate": _handle_review_evaluate,
    "catalog.status": _handle_catalog_status,
    "catalog.enrich": _handle_catalog_enrich,
    "credentials.tns.configure": _handle_tns_credentials_configure,
    "credentials.tns.clear": _handle_tns_credentials_clear,
    "ranker.train": _handle_ranker_train,
    "ranker.apply": _handle_ranker_apply,
    "ranker.list": _handle_ranker_list,
    "crossmatch.run": _handle_crossmatch,
    "crossmatch.profile": _handle_profile,
    "timeframe.offset": _handle_frame_offset,
    "deep.train": _handle_deep_train,
    "deep.compare": _handle_deep_compare,
    "deep.sweep": _handle_deep_sweep,
    "features.build": _handle_features_build,
    "features.build_resumable": _handle_features_build_resumable,
    "features.list": _handle_features_list,
    "features.names": _handle_feature_names,
    "anomaly.detect": _handle_detect,
    "job.submit": _handle_job_submit,
    "job.status": _handle_job_status,
    "job.cancel": _handle_job_cancel,
    "job.retry": _handle_job_retry,
    "job.list": _handle_job_list,
}


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    """Route one request to its handler, converting failures into error replies."""
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    handler = HANDLERS.get(method)
    if handler is None:
        return {"id": request_id, "ok": False,
                "error": f"unknown method: {method!r}"}

    try:
        return {"id": request_id, "ok": True, "result": handler(params)}
    except Exception as exc:  # noqa: BLE001 - a bad request must not kill the engine
        return {"id": request_id, "ok": False, "error": str(exc),
                "traceback": traceback.format_exc()}


def serve(stdin=None, stdout=None) -> None:
    """Read requests until stdin closes, which happens when Rust exits.

    The reader (Rust) can also disappear WITHOUT stdin closing first -- a dev
    hot-reload, a force-quit, or the window being killed while a response is
    in flight all close the pipe from the other end. On Windows, writing to a
    pipe whose reader is already gone raises `OSError: [Errno 22] Invalid
    argument` instead of the `BrokenPipeError` Unix would give (documented
    CPython behaviour on Windows, not specific to this file). That write used
    to be unguarded here, so it took down the whole engine process with an
    unhandled exception every time it happened. There is nothing to retry a
    failed write to a broken pipe against, so the correct response is to log
    it as an expected shutdown and exit, not let it become an unhandled crash.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    logger = logging.getLogger(__name__)

    # A Tauri restart can interrupt a request after discovery but before the
    # final response. Recover it before accepting new UI commands.
    jobs.recover(HANDLERS)

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {"id": None, "ok": False, "error": f"invalid JSON: {exc}"}
        else:
            response = dispatch(request)

        try:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()  # Rust blocks on a line; buffering would deadlock it
        except OSError:
            logger.info("stdout closed while writing a response; the reader "
                       "(Rust) is gone, so this is a normal shutdown.")
            return
