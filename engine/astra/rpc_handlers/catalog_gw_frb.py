"""Public catalogue cross-reference (SIMBAD/VSX/TNS), gravitational-wave and
FRB coincidence checks, Gaia DR4 epoch ingestion, and joint-period multiband
sidecar building.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

import json
from typing import Any

from .common import Handler, _workspace_root

from .. import catalogs, config, frb, gw, multiband, security
from ..surveys import gaia_epoch

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
    fixture_path = security.authorized_path(params["fixture_path"])
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    checkpoint = (security.authorized_write_path(params["checkpoint"], config.PATHS.root)
                 if params.get("checkpoint")
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
    checkpoint = (security.authorized_write_path(params["checkpoint"], config.PATHS.root)
                 if params.get("checkpoint")
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


HANDLERS: dict[str, Handler] = {
    "catalog.status": _handle_catalog_status,
    "catalog.enrich": _handle_catalog_enrich,
    "gw.events": _handle_gw_events,
    "gw.enrich": _handle_gw_enrich,
    "gaia.epoch_ingest": _handle_gaia_epoch_ingest,
    "gaia.epoch_status": _handle_gaia_epoch_status,
    "features.multiband_build": _handle_multiband_build,
    "frb.events": _handle_frb_events,
    "frb.enrich": _handle_frb_enrich,
}
