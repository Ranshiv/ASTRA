"""Explicit poll adapters for alert brokers and VOEvent feeds.

ASTRA does not keep a background network thread alive.  A researcher calls
``alerts.poll`` with a provider and optional endpoint; the bounded response is
normalized into the existing immutable event-packet inbox.  Cursors are stored
in SQLite so a repeated poll can resume, while packet hashes and provider
packet IDs make retries idempotent.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import association, config, events, metadata, netclient

SCHEMA_VERSION = 1
MAX_PACKETS = 500
DEFAULT_ENDPOINTS = {
    "gcn": "https://gcn.nasa.gov/alerts",
    "voevent": "https://voevent.ivoa.net/voevent/",
    "alerce": "https://api.alerce.online/ztf/v1/alerts",
    "fink": "https://api.fink-portal.org/api/v1/alerts",
    # IceCube/Fermi/Swift notices are served through the same NASA GCN
    # Classic/New-GCN infrastructure as the generic "gcn" provider above --
    # the transport is identical (this poller, unchanged); these three keys
    # exist so association.event_to_event_correlation can distinguish
    # messengers by `provider` rather than lumping every high-energy notice
    # under one generic label. The real per-instrument notice-type filter
    # (e.g. restricting to IceCube AMON alerts specifically, as opposed to
    # every GCN notice type) has NOT been live-validated -- the same
    # "documented, not yet confirmed against a live fetch" caveat every
    # other connector in this codebase carries. Pass the real filter via
    # `poll(provider, params={...})` once GCN's actual filtering contract
    # is confirmed against the live service.
    "icecube": "https://gcn.nasa.gov/alerts",
    "fermi": "https://gcn.nasa.gov/alerts",
    "swift": "https://gcn.nasa.gov/alerts",
}


class AlertPollError(RuntimeError):
    """An alert feed could not be read or normalized."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def providers() -> list[dict[str, Any]]:
    return [{"name": name, "endpoint": endpoint, "mode": "bounded_poll",
             "requires_endpoint_override": name in {"voevent", "fink"}}
            for name, endpoint in DEFAULT_ENDPOINTS.items()]


def _list_payload(payload: object) -> tuple[list[object], str | None]:
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        cursor = payload.get("next_cursor") or payload.get("nextCursor") or payload.get("cursor")
        for key in ("alerts", "events", "items", "results", "data", "packets"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows, str(cursor) if cursor is not None else None
        # A single normalized alert is a valid bounded response.
        if any(key in payload for key in ("event_id", "packet_id", "id", "ivorn", "alertId")):
            return [payload], str(cursor) if cursor is not None else None
    if isinstance(payload, str) and payload.lstrip().startswith("<"):
        return [payload], None
    return [], str(payload.get("cursor")) if isinstance(payload, dict) and payload.get("cursor") else None


def _stable_packet_id(item: object) -> str:
    """Return a deterministic fallback ID for packets without provider IDs.

    Feed ordering is not stable, so an index-based ID would make the same
    packet look new whenever a broker reorders its response.  Canonicalizing
    the payload before hashing keeps retries and pagination idempotent while
    retaining a compact, human-searchable prefix.
    """
    if isinstance(item, str):
        canonical = item.encode("utf-8")
    else:
        canonical = json.dumps(item, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, default=str).encode("utf-8")
    return f"hash-{hashlib.sha256(canonical).hexdigest()[:32]}"


def _normalize_item(provider: str, item: object, index: int) -> tuple[object, str | None, str]:
    if isinstance(item, str):
        packet_id = _stable_packet_id(item)
        match = re.search(r"(?:ivorn|id)=['\"]([^'\"]+)", item)
        if match:
            packet_id = match.group(1)
        return item, None, packet_id
    if not isinstance(item, dict):
        raise AlertPollError(f"alert item {index} is not an object or XML packet")
    payload = dict(item)
    packet_id = (payload.get("packet_id") or payload.get("packetId") or
                 payload.get("alert_id") or payload.get("alertId") or
                 payload.get("id") or payload.get("ivorn"))
    event_id = payload.get("event_id") or payload.get("eventId") or payload.get("event")
    if event_id and "event_id" not in payload:
        payload["event_id"] = event_id
    if packet_id and "packet_id" not in payload:
        payload["packet_id"] = packet_id
    return payload, None, str(packet_id or _stable_packet_id(payload))


def _ingest_outcome(record: dict[str, Any], existing_keys: set[tuple[str, str, str]]) -> tuple[dict[str, Any], bool]:
    """Return the packet and whether this poll added a previously unseen row."""
    identity = (str(record.get("provider", "")), str(record.get("packet_id", "")),
                str(record.get("packet_version", "1")))
    is_new = all(identity) and identity not in existing_keys
    existing_keys.add(identity)
    return record, is_new


def _fetch(endpoint: str, provider: str, cursor: str | None, limit: int,
           fetcher: Callable[..., Any] | None = None,
           extra_params: dict[str, Any] | None = None) -> tuple[object, str | None]:
    if fetcher is not None:
        result = fetcher(endpoint=endpoint, cursor=cursor, limit=limit)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, None
    params: dict[str, Any] = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    if extra_params:
        params.update(extra_params)
    response = netclient.get(endpoint, params, timeout=60, provider=provider)
    content_type = str((getattr(response, "headers", {}) or {}).get("Content-Type", "")).lower()
    text = getattr(response, "text", "") or ""
    if "xml" in content_type or text.lstrip().startswith("<"):
        return text, None
    try:
        return response.json(), None
    except (ValueError, AttributeError) as exc:
        raise AlertPollError("alert endpoint returned unreadable JSON/XML") from exc


def _parse_received(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _latency_summary(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Alert-emission-to-ingestion latency, in seconds, over packets whose
    ``event_time`` parses.

    Reuses ``association._parse_time`` (already handles ISO strings and
    JD/MJD numerics consistently with the rest of the event-graph code)
    rather than a third copy of the same parsing logic. Packets without a
    usable ``event_time`` are excluded rather than treated as zero latency.
    """
    deltas: list[float] = []
    for record in records:
        event_time = association._parse_time(record.get("event_time"))
        received = _parse_received(record.get("received_utc"))
        if event_time is None or received is None:
            continue
        deltas.append((received - event_time).total_seconds())
    if not deltas:
        return None
    array = np.asarray(deltas, dtype=float)
    return {"median": float(np.median(array)), "p95": float(np.percentile(array, 95)),
            "n": len(deltas)}


def poll(provider: str, *, endpoint: str | None = None, root: Path | None = None,
         project_id: str | None = None, cursor: str | None = None,
         limit: int = 100, offline: bool = False,
         payload: object | None = None, params: dict[str, Any] | None = None,
         fetcher: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Poll at most ``limit`` packets and ingest them into the event inbox.

    ``params`` is an optional query-parameter override merged into the
    real network request -- e.g. ``alerts.poll("alerce", params={"survey":
    "lsst"})`` polls the same credential-free ALeRCE endpoint for LSST-
    labelled alerts instead of ZTF, with no second endpoint entry needed.
    It has no effect when ``payload``/``fetcher`` bypass the network call.
    """
    provider = str(provider).strip().lower()
    if provider not in DEFAULT_ENDPOINTS:
        raise ValueError(f"unknown alert provider {provider!r}")
    limit = max(1, min(int(limit), MAX_PACKETS))
    root = (root or config.PATHS.root).resolve()
    previous = metadata.get_alert_cursor(root, provider)
    selected_cursor = cursor if cursor is not None else (previous or {}).get("cursor")
    endpoint = endpoint or DEFAULT_ENDPOINTS[provider]
    if payload is None:
        if offline:
            return {"schema_version": SCHEMA_VERSION, "provider": provider,
                    "state": "offline", "endpoint": endpoint, "cursor": selected_cursor,
                    "packets": [], "ingested": 0, "new_packets": 0,
                    "duplicate_rate": None, "latency_summary": None,
                    "errors": [], "polled_utc": _now()}
        try:
            raw, returned_cursor = _fetch(endpoint, provider, selected_cursor, limit,
                                          fetcher, params)
        except Exception as exc:  # noqa: BLE001 - poll state is persisted below
            metadata.put_alert_cursor(root, provider, cursor=selected_cursor,
                                      packet_count=int((previous or {}).get("packet_count", 0)),
                                      last_poll_utc=_now(), last_error=str(exc))
            raise
    else:
        raw, returned_cursor = payload, None
    items, envelope_cursor = _list_payload(raw)
    next_cursor = returned_cursor or envelope_cursor or selected_cursor
    ingested: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    existing_keys: set[tuple[str, str, str]] = {
        (str(packet.get("provider", "")), str(packet.get("packet_id", "")),
         str(packet.get("packet_version", "1")))
        for packet in events.list_events(root=root, provider=provider,
                                         limit=2_000, packets=True)
    }
    new_packets = 0
    for index, item in enumerate(items[:limit]):
        try:
            normalized, _, packet_id = _normalize_item(provider, item, index)
            record = events.ingest(provider, normalized, root=root, packet_id=packet_id,
                                   project_id=project_id)
            record, is_new = _ingest_outcome(record, existing_keys)
            ingested.append(record)
            new_packets += int(is_new)
        except Exception as exc:  # one malformed packet must not drop the poll
            errors.append({"index": index, "error": str(exc)})
    total_count = int((previous or {}).get("packet_count", 0)) + new_packets
    metadata.put_alert_cursor(root, provider, cursor=next_cursor,
                              packet_count=total_count, last_poll_utc=_now(),
                              last_error=None if not errors else f"{len(errors)} packet errors")
    duplicate_rate = (1.0 - (new_packets / len(ingested))) if ingested else None
    return {"schema_version": SCHEMA_VERSION, "provider": provider,
            "state": "ok" if not errors else "partial", "endpoint": endpoint,
            "cursor": next_cursor, "packets": ingested, "ingested": len(ingested),
            "new_packets": new_packets, "duplicate_rate": duplicate_rate,
            "latency_summary": _latency_summary(ingested),
            "errors": errors, "polled_utc": _now()}


def status(root: Path | None = None) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "providers": providers(),
            "cursors": metadata.alert_cursor_summary((root or config.PATHS.root).resolve())}
