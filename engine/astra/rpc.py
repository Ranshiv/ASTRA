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

from . import (ablation, acquire, alerts, anomaly, artifact, association, cache,
               candidates as candidates_mod,
               catalogs, config, credentials, crossmatch, evaluate, evidence,
               events, frb, gw, literature, multiband, significance, tap,
               experiment, exports, featurematrix, features, fitsio, followup, hardware,
               image_features, modalitymatrix, readiness, spectral_features,
               jobs, manifest as manifest_mod, metadata, pipeline, products,
               project as project_mod, review, ranker, security, sed, store, surveys,
               stageb, tensors, tess_pixels, timeframe, viz)
from .surveys import gaia_epoch
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


def _event_root(params: dict[str, Any]) -> Path | None:
    """Use a project workspace for mutable event indexes when supplied."""
    return _workspace_root(params.get("project_id"))


def _handle_event_providers(_params: dict[str, Any]) -> list[dict]:
    return events.providers()


def _handle_event_ingest(params: dict[str, Any]) -> dict:
    if "payload" not in params:
        raise ValueError("events.ingest requires a payload")
    return events.ingest(
        str(params.get("provider", "generic")), params["payload"],
        root=_event_root(params), release=str(params.get("release", "unknown")),
        packet_id=params.get("packet_id"),
        packet_version=str(params.get("packet_version", "1")),
        received_utc=params.get("received_utc"), project_id=params.get("project_id"),
    )


def _handle_event_list(params: dict[str, Any]) -> list[dict]:
    return events.list_events(
        root=_event_root(params), provider=params.get("provider"),
        event_id=params.get("event_id"), limit=int(params.get("limit", 500)),
        packets=bool(params.get("packets", False)),
    )


def _handle_event_get(params: dict[str, Any]) -> dict:
    return events.get_packet(str(params["packet_key"]), root=_event_root(params),
                             include_raw=bool(params.get("include_raw", False)))


def _handle_event_replay(params: dict[str, Any]) -> list[dict]:
    return events.replay(root=_event_root(params), provider=params.get("provider"),
                         event_id=params.get("event_id"),
                         limit=int(params.get("limit", 100)))


def _handle_significance_calibrate(params: dict[str, Any]) -> dict:
    payload = significance.calibrate(
        params.get("scores", []), reference_scores=params.get("reference_scores"),
        threshold=params.get("threshold"), strata=params.get("strata"),
        method=str(params.get("method", "empirical_tail")),
    )
    if params.get("save", True):
        path = significance.save(payload, root=(_workspace_root(params.get("project_id"))
                                                or config.PATHS.root),
                                 kind="calibration", name=str(params.get("name", "default")))
        payload["path"] = str(path)
    return payload


def _handle_selection_evaluate(params: dict[str, Any]) -> dict:
    dimensions = tuple(str(item) for item in params.get(
        "dimensions", ("amplitude", "duration_days", "magnitude")))
    payload = significance.evaluate_selection(
        params.get("records", []), dimensions=dimensions, edges=params.get("edges"),
        fit_model=bool(params.get("fit_model", False)),
        model_features=tuple(str(item) for item in params.get("model_features", dimensions)),
        bootstrap_samples=int(params.get("bootstrap_samples", 0)),
        seed=int(params.get("seed", 42)),
    )
    if params.get("save", True):
        path = significance.save(payload, root=(_workspace_root(params.get("project_id"))
                                                or config.PATHS.root),
                                 kind="selection", name=str(params.get("name", "default")))
        payload["path"] = str(path)
    return payload


def _handle_review_next(params: dict[str, Any]) -> list[dict]:
    name = str(params.get("name", "default"))
    rows = candidates_mod.load(name, _workspace_root(params.get("project_id")))
    return review.select_next(rows, limit=int(params.get("limit", 20)))


def _handle_followup_plan(params: dict[str, Any]) -> dict:
    return followup.plan(
        ra_deg=float(params["ra_deg"]), dec_deg=float(params["dec_deg"]),
        start_utc=params.get("start_utc"),
        duration_hours=float(params.get("duration_hours", 12.0)),
        latitude_deg=float(params.get("latitude_deg", 43.65)),
        longitude_deg=float(params.get("longitude_deg", -79.38)),
        min_altitude_deg=float(params.get("min_altitude_deg", 30.0)),
        cadence_minutes=int(params.get("cadence_minutes", 10)),
        target_id=params.get("target_id"),
        twilight_sun_altitude_deg=float(params.get("twilight_sun_altitude_deg", -18.0)),
        min_moon_separation_deg=float(params.get("min_moon_separation_deg", 0.0)),
        max_moon_illumination=float(params.get("max_moon_illumination", 1.0)),
        max_airmass=params.get("max_airmass"),
        weather=params.get("weather"),
        facility_name=params.get("facility_name"),
        facility_constraints=params.get("facility_constraints"),
    )


def _handle_event_associate(params: dict[str, Any]) -> dict:
    return association.associate_candidates(
        name=str(params.get("name", "default")),
        root=_workspace_root(params.get("project_id")),
        provider=params.get("provider"), event_id=params.get("event_id"),
        radius_arcsec=float(params.get("radius_arcsec", association.DEFAULT_RADIUS_ARCSEC)),
        window_days=float(params.get("window_days", association.DEFAULT_WINDOW_DAYS)),
        allow_unknown_time=bool(params.get("allow_unknown_time", False)),
    )


def _handle_event_graph_correlate(params: dict[str, Any]) -> dict[str, Any]:
    """Pairwise cross-messenger Bayes-factor statistics for ingested events.

    Exposes `association.event_to_event_correlation` -- distinct from
    `events.associate` above, which links an event to a candidate, not one
    event to another. Callers wanting a specific provider/event subset should
    pre-filter with `events.list`/`events.get`; this endpoint always runs
    over `association.fetch_latest_events()`'s full deduplicated view so a
    revised packet cannot silently produce inconsistent results between the
    two association RPCs, matching `fetch_latest_events`'s own contract.
    """
    events_list = association.fetch_latest_events(
        root=_workspace_root(params.get("project_id")), provider=params.get("provider"),
        event_id=params.get("event_id"))
    return {
        "events_checked": len(events_list),
        "pairs": association.event_to_event_correlation(
            events_list,
            window_days=float(params.get("window_days", association.DEFAULT_WINDOW_DAYS)),
            background_window_days=float(params.get("background_window_days", 365.0)),
        ),
    }


def _handle_event_graph_calibrate(params: dict[str, Any]) -> dict[str, Any]:
    """Scrambled-time-slide null calibration for the event-graph Bayes factor."""
    events_list = association.fetch_latest_events(
        root=_workspace_root(params.get("project_id")), provider=params.get("provider"),
        event_id=params.get("event_id"))
    return association.calibrate_event_graph(
        events_list,
        window_days=float(params.get("window_days", association.DEFAULT_WINDOW_DAYS)),
        background_window_days=float(params.get("background_window_days", 365.0)),
        n_trials=int(params.get("n_trials", 200)),
        seed=int(params.get("seed", 42)),
    )


def _handle_alert_providers(_params: dict[str, Any]) -> list[dict]:
    return alerts.providers()


def _handle_alert_status(params: dict[str, Any]) -> dict:
    return alerts.status(_workspace_root(params.get("project_id")) or config.PATHS.root)


def _handle_alert_poll(params: dict[str, Any]) -> dict:
    return alerts.poll(
        str(params["provider"]), endpoint=params.get("endpoint"),
        root=_workspace_root(params.get("project_id")) or config.PATHS.root,
        project_id=params.get("project_id"), cursor=params.get("cursor"),
        limit=int(params.get("limit", 100)), offline=bool(params.get("offline", False)),
        payload=params.get("payload"), params=params.get("params"),
    )


def _handle_tap_status(params: dict[str, Any]) -> dict:
    return tap.status(_workspace_root(params.get("project_id")) or config.PATHS.projects)


def _handle_tap_query(params: dict[str, Any]) -> dict:
    return tap.query(
        str(params["service"]), str(params["adql"]),
        release=str(params.get("release", "unknown")),
        root=_workspace_root(params.get("project_id")) or config.PATHS.projects,
        max_rows=int(params.get("max_rows", 200)), fmt=str(params.get("format", "csv")),
        refresh=bool(params.get("refresh", False)), offline=bool(params.get("offline", False)),
        timeout=float(params.get("timeout", 60.0)),
    )


def _handle_literature_status(_params: dict[str, Any]) -> dict:
    return literature.status()


def _handle_literature_search(params: dict[str, Any]) -> dict:
    providers = params.get("providers", ("ads", "arxiv"))
    return literature.search(
        object_id=str(params.get("object_id", "")),
        terms=params.get("terms", []), event_ids=params.get("event_ids", []),
        providers=providers, limit=int(params.get("limit", 20)),
        root=_workspace_root(params.get("project_id")),
        refresh=bool(params.get("refresh", False)), offline=bool(params.get("offline", False)),
    )


def _handle_literature_enrich(params: dict[str, Any]) -> dict:
    return literature.enrich_candidates(
        name=str(params.get("name", "default")),
        root=_workspace_root(params.get("project_id")),
        refresh=bool(params.get("refresh", False)), offline=bool(params.get("offline", False)),
        include_arxiv=bool(params.get("include_arxiv", True)),
        limit=int(params.get("limit", 20)),
    )


def _handle_physical_characterize(params: dict[str, Any]) -> dict:
    return sed.characterize(params.get("photometry", {}),
                            extinction=params.get("extinction"),
                            source=str(params.get("source", "caller")))


def _handle_physical_enrich(params: dict[str, Any]) -> dict:
    return sed.characterize_candidate(
        name=str(params.get("name", "default")),
        root=_workspace_root(params.get("project_id")),
        extinction=params.get("extinction"),
    )


def _handle_digital_twin_fit_profile(params: dict[str, Any]) -> dict[str, Any]:
    """Fit a per-survey cadence/noise profile from real stored curves.

    Read-only diagnostic (backlog item 42): never writes into
    `scoring.WEIGHTS`/`evidence.py`, the same convention every other
    interpretation-only method here (`physical.characterize`,
    `significance.calibrate`) already follows.
    """
    from . import survey_digital_twin as sdt

    profile = sdt.fit_survey_profile(
        str(params["survey"]), limit=int(params.get("limit", 500)),
        length=int(params.get("length", sdt.DEFAULT_LENGTH)),
    )
    return profile.to_dict()


def _handle_digital_twin_sample(params: dict[str, Any]) -> dict[str, Any]:
    """Fit a profile, then sample a synthetic batch from it.

    Returns aggregate stats via `SequenceBatch.to_dict()` (rows, length,
    mean coverage), not the raw `(n, 2, length)` array -- unsuitable as a
    JSON payload and unnecessary for a diagnostic summary. Read-only, same
    scoring/evidence non-goal as every handler in this group.
    """
    from . import survey_digital_twin as sdt

    profile = sdt.fit_survey_profile(str(params["survey"]),
                                     limit=int(params.get("limit", 500)))
    batch = sdt.sample_synthetic_batch(
        profile, n=int(params.get("n", 50)), seed=int(params.get("seed", 42)))
    return {"profile": profile.to_dict(), "batch": batch.to_dict()}


def _handle_digital_twin_evaluate_distance(params: dict[str, Any]) -> dict[str, Any]:
    """Success criterion 1 (item 42): distance between simulated and real
    summary statistics. Read-only diagnostic; never touches ranking."""
    from . import survey_digital_twin as sdt
    from . import survey_digital_twin_eval as sdte

    survey = str(params["survey"])
    limit = int(params.get("limit", 500))
    profile = sdt.fit_survey_profile(survey, limit=limit)
    real = tensors.build(survey=survey, limit=limit)
    if len(real) < 2:
        return {"error": f"only {len(real)} usable real sequences; need at least 2",
                "rows": len(real)}
    synthetic = sdt.sample_synthetic_batch(
        profile, n=len(real), seed=int(params.get("seed", 42)))
    distance = sdte.summary_statistic_distance(real.values, synthetic.values)
    return {"profile": profile.to_dict(), **distance}


def _handle_digital_twin_evaluate_transfer(params: dict[str, Any]) -> dict[str, Any]:
    """Success criterion 2 (item 42): transfer performance. Minutes, not
    seconds -- trains one autoencoder per seed per arm, same as
    `deep.compare`. Read-only diagnostic; never touches ranking."""
    _require_torch()
    from . import survey_digital_twin as sdt
    from . import survey_digital_twin_eval as sdte

    survey = str(params["survey"])
    limit = int(params.get("limit", 500))
    profile = sdt.fit_survey_profile(survey, limit=limit)
    real = tensors.build(survey=survey, limit=limit)
    if len(real) < 10:
        return {"error": f"only {len(real)} usable real sequences; need at least 10",
                "rows": len(real)}
    synthetic = sdt.sample_synthetic_batch(
        profile, n=len(real), seed=int(params.get("seed", 42)))
    seeds = tuple(int(seed) for seed in params.get("seeds", (17, 29, 43)))
    result = sdte.evaluate_transfer_performance(
        real, synthetic, fraction=float(params.get("fraction", 0.1)),
        seeds=seeds, epochs=int(params.get("epochs", 15)))
    return {"profile": profile.to_dict(), **result}


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
    if params.get("checkpoint"):
        kwargs["checkpoint"] = Path(params["checkpoint"])
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
        anchor_survey=params.get("anchor_survey"),
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


# scoring.py:178 already refuses a parallax with SNR < 5 ("too noisy to
# use") when screening period-luminosity consistency. The spatial view reuses
# that exact threshold rather than inventing a second one: a distance judged
# unreliable enough to skip a luminosity check is unreliable enough to skip
# plotting in 3D space.
GAIA_PARALLAX_SNR_THRESHOLD = 5.0


def _handle_candidates_spatial(params: dict[str, Any]) -> dict[str, Any]:
    """RA/Dec/Gaia-distance per candidate, for the 3D spatial view.

    The live candidate pipeline never joins Gaia distance columns -- only the
    offline ablation study does, via featurematrix.join_gaia_columns. Rather
    than changing candidate-build provenance (a FEATURE_VERSION-adjacent
    concern this view has no business touching), this reuses that same join
    against a deliberately empty FeatureMatrix built only from the already-
    loaded candidates' identities. Nothing is re-extracted or recomputed.
    """
    import numpy as np

    root = _workspace_root(params.get("project_id"))
    built = candidates_mod.load(params.get("name", "default"), root)
    top = int(params.get("top", 200))
    subset = built[:top]

    identities = [{"path": c.path, "object_id": c.object_id,
                  "survey": c.survey, "ra_deg": c.ra_deg, "dec_deg": c.dec_deg}
                 for c in subset]
    empty = featurematrix.FeatureMatrix(
        values=np.empty((len(subset), 0)), identities=identities,
        feature_names=())
    joined, diagnostics = featurematrix.join_gaia_columns(
        empty, radius_arcsec=crossmatch.DEFAULT_RADIUS_ARCSEC, projects_root=root)

    distance_col = joined.feature_names.index("gaia_distance_pc")
    abs_g_col = joined.feature_names.index("gaia_abs_g_mag")
    snr_col = joined.feature_names.index("gaia_parallax_snr")
    matched_col = joined.feature_names.index("gaia_matched")
    ra_now_col = joined.feature_names.index("gaia_ra_now_deg")
    dec_now_col = joined.feature_names.index("gaia_dec_now_deg")

    points = []
    reliable = 0
    for candidate, row in zip(subset, joined.values):
        snr = row[snr_col]
        distance_reliable = bool(
            row[matched_col] == 1.0 and np.isfinite(row[distance_col])
            and np.isfinite(snr) and snr >= GAIA_PARALLAX_SNR_THRESHOLD)
        if distance_reliable:
            reliable += 1
        points.append({
            "candidate_id": candidate.candidate_id,
            "ra_deg": candidate.ra_deg,
            "dec_deg": candidate.dec_deg,
            "gaia_distance_pc": (float(row[distance_col])
                                 if np.isfinite(row[distance_col]) else None),
            "gaia_abs_g_mag": (float(row[abs_g_col])
                               if np.isfinite(row[abs_g_col]) else None),
            "gaia_parallax_snr": float(snr) if np.isfinite(snr) else None,
            "distance_reliable": distance_reliable,
            "gaia_ra_now_deg": (float(row[ra_now_col])
                                if np.isfinite(row[ra_now_col]) else None),
            "gaia_dec_now_deg": (float(row[dec_now_col])
                                 if np.isfinite(row[dec_now_col]) else None),
            "score_total": candidate.score.get("total"),
        })

    return {
        "points": points,
        "total": len(subset),
        "reliable": reliable,
        "snr_threshold": GAIA_PARALLAX_SNR_THRESHOLD,
        "gaia_matched": diagnostics["matched"],
        "gaia_match_rate": diagnostics["match_rate"],
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


def _handle_gw_events(params: dict[str, Any]) -> dict[str, Any]:
    """List published GW events in one catalog, without touching candidates."""
    events = gw.fetch_event_catalog(
        catalog=params.get("catalog", gw.DEFAULT_CATALOG),
        refresh=bool(params.get("refresh", False)),
        offline=bool(params.get("offline", False)),
    )
    return {"catalog": params.get("catalog", gw.DEFAULT_CATALOG),
           "events": [event.to_dict() for event in events]}


def _handle_gw_enrich(params: dict[str, Any]) -> dict[str, Any]:
    """Optional explicit GW coincidence check; never moves the composite score."""
    return gw.enrich_candidates_gw(
        name=params.get("name", "default"),
        catalog=params.get("catalog", gw.DEFAULT_CATALOG),
        window_days=float(params.get("window_days", gw.DEFAULT_WINDOW_DAYS)),
        refresh=bool(params.get("refresh", False)),
        offline=bool(params.get("offline", False)),
        root=_workspace_root(params.get("project_id")),
    )


def _handle_gaia_epoch_ingest(params: dict[str, Any], progress=None) -> dict[str, Any]:
    """Chunked, resumable Gaia DR4 epoch ingestion from an offline fixture.

    No live DR4 endpoint exists yet (see surveys/gaia_epoch.py's module
    docstring): the only source today is a caller-supplied JSON fixture file
    shaped as `{"chunks": [[{row, ...}, ...], ...]}`, which exercises the
    same chunked/checkpoint/resume path a real delivery mechanism will use
    once DR4's access terms are verified. Long-running for a real fixture,
    so this is meant to be called via job.submit like acquire.cone.
    """
    fixture_path = Path(str(params["fixture_path"]))
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    checkpoint = (Path(str(params["checkpoint"])) if params.get("checkpoint")
                 else config.PATHS.cache / "gaia_epoch" / f"{params.get('name', 'default')}.json")
    batch_size = int(params.get("batch_size", 256))

    def _on_progress(update: dict[str, Any]) -> None:
        if progress is None:
            return
        progress.raise_if_cancelled()
        progress.update(fraction=update.get("fraction"), phase="gaia_epoch_ingest",
                        items_done=update.get("chunks_completed"),
                        items_total=update.get("chunks_total"))

    report = gaia_epoch.ingest_resumable(
        chunks, checkpoint=checkpoint, batch_size=batch_size, progress=_on_progress)
    return report.to_dict()


def _handle_gaia_epoch_status(params: dict[str, Any]) -> dict[str, Any]:
    """Read a checkpoint's current state without ingesting anything."""
    checkpoint = (Path(str(params["checkpoint"])) if params.get("checkpoint")
                 else config.PATHS.cache / "gaia_epoch" / f"{params.get('name', 'default')}.json")
    if not checkpoint.exists():
        return {"exists": False}
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    rows_available = len(gaia_epoch.read_ingested_rows(checkpoint))
    return {
        "exists": True,
        "chunks_completed": len(state.get("completed_chunk_ids", [])),
        "chunks_failed": len(state.get("failed_chunk_ids", [])),
        "rows_accepted": int(state.get("rows_accepted", 0)),
        "rows_rejected": int(state.get("rows_rejected", 0)),
        "rejection_histogram": dict(state.get("rejection_histogram", {})),
        "rows_available": rows_available,
    }


def _handle_multiband_build(params: dict[str, Any]) -> dict[str, Any]:
    """Explicit, opt-in joint-period sidecar build. Never run by the default
    pipeline -- see multiband.py's module docstring for why."""
    kwargs = {"survey": params.get("survey"),
             "limit": int(params.get("limit", 10_000)),
             "name": params.get("name", "default")}
    if params.get("project_id"):
        kwargs["root"] = _workspace_root(params.get("project_id"))
    return multiband.build_multiband_sidecar(**kwargs)


def _handle_frb_events(params: dict[str, Any]) -> dict[str, Any]:
    """List published CHIME/FRB bursts, without touching candidates."""
    bursts = frb.fetch_burst_catalog(
        refresh=bool(params.get("refresh", False)),
        offline=bool(params.get("offline", False)),
    )
    return {"bursts": [burst.to_dict() for burst in bursts]}


def _handle_frb_enrich(params: dict[str, Any]) -> dict[str, Any]:
    """Optional explicit FRB coincidence check; never moves the composite score."""
    return frb.enrich_candidates_frb(
        name=params.get("name", "default"),
        window_days=float(params.get("window_days", frb.DEFAULT_WINDOW_DAYS)),
        sigma_threshold=float(params.get("sigma_threshold", frb.DEFAULT_SIGMA_THRESHOLD)),
        refresh=bool(params.get("refresh", False)),
        offline=bool(params.get("offline", False)),
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


def _handle_rubin_credentials_configure(params: dict[str, Any]) -> dict[str, Any]:
    """Store a Rubin/LSST data-rights token with Windows DPAPI.

    RubinTAPConnector (surveys/rubin_tap.py) is dormant until this is called
    with a real token -- see that module's docstring. Never echoed back.
    """
    return credentials.save_credentials("rubin", {"token": str(params["token"])})


def _handle_rubin_credentials_status(_params: dict[str, Any]) -> dict[str, Any]:
    return credentials.credential_status("rubin")


def _handle_rubin_credentials_clear(_params: dict[str, Any]) -> dict[str, Any]:
    return {"cleared": credentials.clear_credentials("rubin")}


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
    "acquire.project": _handle_acquire_project,
    "store.usage": _handle_store_usage,
    "events.providers": _handle_event_providers,
    "events.ingest": _handle_event_ingest,
    "events.list": _handle_event_list,
    "events.get": _handle_event_get,
    "events.replay": _handle_event_replay,
    "events.associate": _handle_event_associate,
    "events.graph.correlate": _handle_event_graph_correlate,
    "events.graph.calibrate": _handle_event_graph_calibrate,
    "alerts.providers": _handle_alert_providers,
    "alerts.status": _handle_alert_status,
    "alerts.poll": _handle_alert_poll,
    "tap.status": _handle_tap_status,
    "tap.query": _handle_tap_query,
    "significance.calibrate": _handle_significance_calibrate,
    "selection.evaluate": _handle_selection_evaluate,
    "review.next": _handle_review_next,
    "followup.plan": _handle_followup_plan,
    "literature.status": _handle_literature_status,
    "literature.search": _handle_literature_search,
    "literature.enrich": _handle_literature_enrich,
    "physical.characterize": _handle_physical_characterize,
    "physical.enrich": _handle_physical_enrich,
    "digital_twin.fit_profile": _handle_digital_twin_fit_profile,
    "digital_twin.sample": _handle_digital_twin_sample,
    "digital_twin.evaluate_distance": _handle_digital_twin_evaluate_distance,
    "digital_twin.evaluate_transfer": _handle_digital_twin_evaluate_transfer,
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
    "artifact.calibrate": _handle_artifact_calibrate,
    "pipeline.run": _handle_pipeline,
    "candidates.load": _handle_candidates_load,
    "candidates.spatial": _handle_candidates_spatial,
    "candidates.get": _handle_candidate_get,
    "candidates.timeline": _handle_candidate_timeline,
    "candidates.export": _handle_candidates_export,
    "candidates.label": _handle_label,
    "candidates.labels": _handle_labels,
    "candidates.evaluate": _handle_review_evaluate,
    "catalog.status": _handle_catalog_status,
    "catalog.enrich": _handle_catalog_enrich,
    "gw.events": _handle_gw_events,
    "gw.enrich": _handle_gw_enrich,
    "frb.events": _handle_frb_events,
    "frb.enrich": _handle_frb_enrich,
    "gaia.epoch_ingest": _handle_gaia_epoch_ingest,
    "gaia.epoch_status": _handle_gaia_epoch_status,
    "features.multiband_build": _handle_multiband_build,
    "credentials.tns.configure": _handle_tns_credentials_configure,
    "credentials.tns.clear": _handle_tns_credentials_clear,
    "credentials.rubin.configure": _handle_rubin_credentials_configure,
    "credentials.rubin.status": _handle_rubin_credentials_status,
    "credentials.rubin.clear": _handle_rubin_credentials_clear,
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
