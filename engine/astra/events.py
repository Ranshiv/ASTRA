"""Event-native alert storage and normalization.

Survey connectors describe point sources.  Alerts, gravitational-wave
notices, and FRB events have a different lifecycle: one event can receive
multiple revised packets and its sky position can be a probability region.
This module keeps that distinction explicit while still producing stable,
JSON-serialisable records for the candidate and evidence layers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config, metadata

EVENT_SCHEMA_VERSION = 1
MAX_RAW_BYTES = 16 * 1024 * 1024
_PROVIDERS = (
    {"name": "generic", "kind": "packet", "online": False},
    {"name": "voevent", "kind": "packet", "online": True},
    {"name": "gcn", "kind": "multimessenger", "online": True},
    {"name": "alerce", "kind": "alert_broker", "online": True},
    {"name": "fink", "kind": "alert_broker", "online": True},
    {"name": "gw", "kind": "gravitational_wave", "online": False},
    {"name": "frb", "kind": "radio_transient", "online": False},
    # IceCube/Fermi/Swift notices are transport-reachable today through the
    # SAME generic gcn/voevent poller (alerts.py) as every other provider
    # above -- these three entries only make the messenger explicit for
    # downstream cross-messenger correlation (association.event_to_event_
    # correlation), which groups pairs by `provider`. Field-name recognition
    # below is generic-normalizer-only, matching this codebase's "start
    # generic, fork a bespoke parser only once a real payload proves the
    # generic shape too lossy" rule: real GCN Classic notices for IceCube
    # (signalness/energy/far) and Fermi/Swift (instrument-specific error-
    # radius units) carry structure the generic normalizer does not
    # preserve. This is a documented, not-yet-live-validated gap -- see
    # docs/LIMITATIONS.md, matching every other unvalidated connector's caveat.
    {"name": "icecube", "kind": "neutrino", "online": True},
    {"name": "fermi", "kind": "gamma_ray", "online": True},
    {"name": "swift", "kind": "gamma_ray", "online": True},
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _event_root(root: Path | None) -> Path:
    # A project root is preferred.  The global root remains useful for a
    # packet inbox before a researcher has created a project.
    return (root or config.PATHS.root).resolve()


def _raw_bytes(payload: object) -> bytes:
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        try:
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("event payload must be JSON-compatible text or bytes") from exc
    if len(raw) > MAX_RAW_BYTES:
        raise ValueError(f"event payload exceeds the {MAX_RAW_BYTES // (1024 * 1024)} MiB limit")
    return raw


def _payload_object(payload: object) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, bytes):
        text = payload.decode("utf-8", errors="strict")
    elif isinstance(payload, str):
        text = payload
    else:
        raise ValueError("event payload must be an object, JSON text, XML text, or bytes")
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("event JSON must contain an object")
        return value
    except json.JSONDecodeError:
        return _voevent_object(text)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _voevent_object(text: str) -> dict[str, Any]:
    """Extract the stable VOEvent fields without requiring a VOEvent package."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError("event payload is neither valid JSON nor VOEvent XML") from exc

    values: dict[str, str] = {}
    for node in root.iter():
        name = _local_name(node.tag)
        value = (node.attrib.get("value") or node.text or "").strip()
        if value and name not in values:
            values[name] = value
    ivorn = root.attrib.get("ivorn") or root.attrib.get("id")
    params = []
    for node in root.iter():
        if _local_name(node.tag) == "param":
            item = {str(k): str(v) for k, v in node.attrib.items()}
            if item:
                params.append(item)
    location: dict[str, Any] = {}
    for key in ("ra", "rightascension", "raj2000"):
        if key in values:
            location["ra_deg"] = _number(values[key])
            break
    for key in ("dec", "declination", "dej2000"):
        if key in values:
            location["dec_deg"] = _number(values[key])
            break
    if "ra_deg" not in location and "c1" in values:
        location["ra_deg"] = _number(values["c1"])
    if "dec_deg" not in location and "c2" in values:
        location["dec_deg"] = _number(values["c2"])
    for key in ("error", "error2", "errormajor"):
        if key in values:
            location["error_radius_arcsec"] = _number(values[key])
            break
    return {
        "event_id": ivorn or values.get("eventid") or values.get("name"),
        "event_time": values.get("date") or values.get("time") or values.get("iso"),
        "localization": location,
        "parameters": params,
        "classification": values.get("classification") or values.get("class"),
    }


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first(obj: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in obj and obj[name] not in (None, ""):
            return obj[name]
    return None


def _localization(obj: dict[str, Any]) -> dict[str, Any]:
    raw = _first(obj, "localization", "sky_localization", "where", "position")
    if not isinstance(raw, dict):
        raw = {}
    result: dict[str, Any] = {}
    for key, aliases in {
        "ra_deg": ("ra_deg", "ra", "right_ascension"),
        "dec_deg": ("dec_deg", "dec", "declination"),
        # "err90"/"error_radius" cover common IceCube (AMON realtime, 90%
        # containment) and Fermi/Swift GCN Classic field names respectively.
        # Neither is unit-normalized here -- some notice types report
        # degrees or arcmin, not arcsec, and the generic normalizer has no
        # per-provider unit table. This is a documented, not-yet-live-
        # validated gap (see events._PROVIDERS's icecube/fermi/swift
        # comment and docs/LIMITATIONS.md): a bespoke per-notice-type parser
        # would be needed to convert correctly, and is deliberately not
        # built until a real payload is examined.
        "error_radius_arcsec": ("error_radius_arcsec", "radius_arcsec", "error",
                                "err90", "error_radius"),
        "credible_level": ("credible_level", "confidence", "probability"),
        "healpix_nside": ("healpix_nside", "nside"),
    }.items():
        value = _first(raw, *aliases)
        if value is not None:
            number = _number(value)
            if number is not None:
                result[key] = number
    pixels = raw.get("pixels") or raw.get("healpix")
    if isinstance(pixels, list):
        clean_pixels = []
        for item in pixels:
            if not isinstance(item, dict) or "probability" not in item:
                continue
            probability = _number(item.get("probability"))
            index = _number(item.get("index", item.get("pixel")))
            if probability is None or not 0.0 <= probability <= 1.0 or index is None:
                continue
            clean_pixels.append({"index": int(index), "probability": probability})
        result["pixels"] = clean_pixels[:100_000]
        result["type"] = "healpix"
    elif "ra_deg" in result and "dec_deg" in result:
        result["type"] = "point"
    else:
        result["type"] = "unknown"
    if result.get("error_radius_arcsec", 0) < 0:
        result.pop("error_radius_arcsec", None)
    if "ra_deg" in result and not 0.0 <= result["ra_deg"] < 360.0:
        result.pop("ra_deg", None)
    if "dec_deg" in result and not -90.0 <= result["dec_deg"] <= 90.0:
        result.pop("dec_deg", None)
    if "credible_level" in result and not 0.0 <= result["credible_level"] <= 1.0:
        result.pop("credible_level", None)
    if "healpix_nside" in result and result["healpix_nside"] < 1:
        result.pop("healpix_nside", None)
    return result


def _classes(obj: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _first(obj, "classifications", "classification", "classes")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"label": raw, "probability": None}]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append({"label": item, "probability": None})
        elif isinstance(item, dict):
            label = _first(item, "label", "class", "name", "type")
            if label is not None:
                probability = _number(_first(item, "probability", "prob", "score"))
                result.append({"label": str(label), "probability": probability})
    return result[:100]


@dataclass(frozen=True)
class EventPacket:
    event_id: str
    packet_id: str
    provider: str
    release: str
    packet_version: str
    event_time: str | None
    received_utc: str
    localization: dict[str, Any] = field(default_factory=dict)
    classifications: list[dict[str, Any]] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    raw_sha256: str = ""
    raw_path: str = ""
    packet_key: str = ""
    status: str = "received"
    project_id: str | None = None
    schema_version: int = EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize(provider: str, payload: object, *, release: str = "unknown",
              packet_id: str | None = None, packet_version: str = "1",
              received_utc: str | None = None, project_id: str | None = None) -> tuple[EventPacket, bytes]:
    """Normalize JSON or VOEvent XML into an immutable packet record."""
    raw = _raw_bytes(payload)
    obj = _payload_object(payload)
    raw_sha = hashlib.sha256(raw).hexdigest()
    event_id = _first(obj, "event_id", "eventId", "id", "ivorn", "name")
    event_id = str(event_id or f"{provider}:{raw_sha[:24]}").strip()
    if not event_id:
        raise ValueError("event_id must not be empty")
    packet_id = str(packet_id or _first(obj, "packet_id", "packetId", "notice_id")
                    or raw_sha[:32])
    event_time = _first(obj, "event_time", "eventTime", "time", "trigger_time", "date")
    if event_time is not None:
        event_time = str(event_time)
    related = _first(obj, "related_ids", "relatedIds", "citations", "references") or []
    if isinstance(related, str):
        related = [related]
    if not isinstance(related, list):
        related = []
    record = EventPacket(
        event_id=event_id,
        packet_id=packet_id,
        provider=str(provider).strip().lower() or "generic",
        release=str(release),
        packet_version=str(packet_version),
        event_time=event_time,
        received_utc=str(received_utc or _now()),
        localization=_localization(obj),
        classifications=_classes(obj),
        related_ids=[str(item) for item in related[:100]],
        raw_sha256=raw_sha,
        project_id=project_id,
    )
    packet_key = hashlib.sha256(
        f"{record.provider}/{record.packet_id}/{record.raw_sha256}".encode("utf-8")
    ).hexdigest()[:40]
    return EventPacket(**{**record.to_dict(), "packet_key": packet_key}), raw


def _write_raw(root: Path, packet: EventPacket, raw: bytes) -> Path:
    directory = _event_root(root) / "events" / "packets"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{packet.raw_sha256}.bin"
    if not path.exists():
        fd, temporary_name = tempfile.mkstemp(prefix=f".{packet.raw_sha256}.", dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
    return path


def ingest(provider: str, payload: object, *, root: Path | None = None,
           release: str = "unknown", packet_id: str | None = None,
           packet_version: str = "1", received_utc: str | None = None,
           project_id: str | None = None) -> dict[str, Any]:
    packet, raw = normalize(provider, payload, release=release, packet_id=packet_id,
                             packet_version=packet_version, received_utc=received_utc,
                             project_id=project_id)
    raw_path = _write_raw(root, packet, raw)
    indexed = EventPacket(**{**packet.to_dict(), "raw_path": str(raw_path)})
    metadata.put_event_packet(_event_root(root), indexed.to_dict())
    return indexed.to_dict()


def providers() -> list[dict[str, Any]]:
    return [dict(item) for item in _PROVIDERS]


def list_events(*, root: Path | None = None, provider: str | None = None,
                event_id: str | None = None, limit: int = 500,
                packets: bool = False) -> list[dict[str, Any]]:
    base = _event_root(root)
    if packets:
        return metadata.list_event_packets(base, provider=provider,
                                           event_id=event_id, limit=limit)
    return metadata.list_event_clusters(base, provider=provider, limit=limit)


def get_packet(packet_key: str, *, root: Path | None = None,
               include_raw: bool = False) -> dict[str, Any]:
    packet = metadata.get_event_packet(_event_root(root), packet_key)
    if packet is None:
        raise KeyError(f"event packet not found: {packet_key}")
    if include_raw:
        path = Path(packet["raw_path"])
        if not path.is_file():
            raise FileNotFoundError(f"raw event packet is missing: {path}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != packet["raw_sha256"]:
            raise ValueError("raw event packet checksum does not match its index")
        packet["raw"] = raw.decode("utf-8", errors="replace")
    return packet


def replay(*, root: Path | None = None, provider: str | None = None,
           event_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Return normalized packets in deterministic received-time order.

    Replay is intentionally read-only.  A caller can feed the returned raw
    payloads into a new project to reproduce an analysis without mutating the
    source archive.
    """
    rows = metadata.list_event_packets(_event_root(root), provider=provider,
                                       event_id=event_id, limit=limit)
    rows.sort(key=lambda row: (row["received_utc"], row["packet_key"]))
    return rows
