"""Event/alert broker handlers: transient event ingestion, clustering,
cross-survey association, and public alert-stream polling.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import Handler, _workspace_root

from .. import alerts, association, config, events

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


HANDLERS: dict[str, Handler] = {
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
}
