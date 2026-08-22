"""Bounded IVOA TAP/ADQL access with explicit cache and provenance.

TAP is useful for archive metadata that does not fit a light-curve connector.
The public API deliberately supports synchronous, read-only queries only. SQL
mutation statements, comments, multi-statements, and unbounded result sets are
rejected before any network request. Callers may provide a fixed ADQL template
and values, but the service response is always capped and cached in ASTRA's
SQLite metadata index.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from . import config, metadata, netclient

SCHEMA_VERSION = 1
CACHE_TTL_DAYS = 7
ERROR_TTL_MINUTES = 5
MAX_ROWS = 5000
MAX_QUERY_BYTES = 64 * 1024
_FORBIDDEN = re.compile(r"\b(?:insert|update|delete|drop|alter|create|truncate|grant|revoke|call|merge)\b", re.I)
_TOP = re.compile(r"^\s*select\s+top\s+(\d+)\b", re.I)


class TapError(RuntimeError):
    """The TAP service or query contract is unusable."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _validate_service(service: str) -> str:
    text = str(service).strip()
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("TAP service must be an absolute http(s) URL")
    return text


def bound_adql(adql: str, max_rows: int = 200) -> tuple[str, int]:
    """Validate read-only ADQL and inject a bounded TOP clause."""
    text = str(adql).strip()
    if not text or len(text.encode("utf-8")) > MAX_QUERY_BYTES:
        raise ValueError("ADQL query is empty or exceeds the size limit")
    if "--" in text or "/*" in text or "*/" in text or ";" in text:
        raise ValueError("ADQL comments and multi-statements are not allowed")
    if _FORBIDDEN.search(text):
        raise ValueError("TAP queries are read-only SELECT statements")
    if not re.match(r"^select\b", text, re.I):
        raise ValueError("ADQL query must begin with SELECT")
    limit = max(1, min(int(max_rows), MAX_ROWS))
    match = _TOP.match(text)
    if match:
        existing = int(match.group(1))
        if existing > limit:
            text = text[:match.start(1)] + str(limit) + text[match.end(1):]
        return text, min(existing, limit)
    return re.sub(r"^\s*select\b", f"SELECT TOP {limit}", text, count=1, flags=re.I), limit


def _cell(value: object) -> object:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "nan"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def parse_csv(payload: str, limit: int = MAX_ROWS) -> list[dict[str, object]]:
    reader = csv.DictReader(io.StringIO(payload))
    if reader.fieldnames is None:
        return []
    names = [str(name).strip() for name in reader.fieldnames]
    return [{name: _cell(row.get(name)) for name in names}
            for row in list(reader)[:max(1, min(int(limit), MAX_ROWS))]]


def parse_votable(payload: str, limit: int = MAX_ROWS) -> list[dict[str, object]]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise TapError("TAP returned unreadable VOTable XML") from exc
    fields = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1].upper() == "FIELD":
            fields.append(str(node.attrib.get("name") or node.attrib.get("ID") or f"column_{len(fields)}"))
    rows = []
    for tr in root.iter():
        if tr.tag.rsplit("}", 1)[-1].upper() != "TR":
            continue
        values = []
        for td in tr:
            if td.tag.rsplit("}", 1)[-1].upper() == "TD":
                values.append(_cell(td.text))
        rows.append({name: values[index] if index < len(values) else None
                     for index, name in enumerate(fields)})
        if len(rows) >= limit:
            break
    return rows


def parse_response(response: Any, limit: int) -> tuple[list[dict[str, object]], str]:
    text = getattr(response, "text", "") or ""
    content_type = str((getattr(response, "headers", {}) or {}).get("Content-Type", "")).lower()
    if "xml" in content_type or text.lstrip().startswith("<?xml") or "<VOTABLE" in text.upper():
        return parse_votable(text, limit), "votable"
    return parse_csv(text, limit), "csv"


def _identity(service: str, release: str, adql: str, fmt: str, limit: int) -> tuple[str, str, dict[str, Any]]:
    query = {"service": service, "release": release, "adql": adql, "format": fmt, "limit": limit}
    query_hash = hashlib.sha256(json.dumps(query, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"tap:{query_hash}", query_hash, query


def query(service: str, adql: str, *, release: str = "unknown", root: Path | None = None,
          max_rows: int = 200, fmt: str = "csv", refresh: bool = False,
          offline: bool = False, timeout: float = 60.0) -> dict[str, Any]:
    service = _validate_service(service)
    bounded, limit = bound_adql(adql, max_rows)
    fmt = str(fmt).lower()
    if fmt not in {"csv", "votable"}:
        raise ValueError("TAP format must be csv or votable")
    root = root or config.PATHS.projects
    cache_key, query_hash, query_json = _identity(service, str(release), bounded, fmt, limit)
    now = _now()
    cached = metadata.get_tap_cache(root, cache_key)
    if cached is not None and not refresh and _parse_utc(cached["expires_utc"]) > now:
        response = cached.get("response") or {}
        return {"schema_version": SCHEMA_VERSION, "service": service,
                "release": release, "state": cached["status"],
                "rows": response.get("rows", []), "format": response.get("format", fmt),
                "query": query_json, "error": cached.get("error"),
                "fetched_utc": cached["fetched_utc"], "expires_utc": cached["expires_utc"],
                "cache": {"state": "hit", "stale": False}}
    if offline:
        if cached is not None:
            response = cached.get("response") or {}
            return {"schema_version": SCHEMA_VERSION, "service": service,
                    "release": release, "state": cached["status"],
                    "rows": response.get("rows", []), "format": response.get("format", fmt),
                    "query": query_json, "error": cached.get("error"),
                    "fetched_utc": cached["fetched_utc"], "expires_utc": cached["expires_utc"],
                    "cache": {"state": "stale_offline", "stale": True}}
        return {"schema_version": SCHEMA_VERSION, "service": service, "release": release,
                "state": "offline", "rows": [], "format": fmt, "query": query_json,
                "error": None, "fetched_utc": None, "expires_utc": None,
                "cache": {"state": "miss", "stale": False}}
    try:
        response = netclient.get(service, {"REQUEST": "doQuery", "LANG": "ADQL",
                                           "FORMAT": fmt, "MAXREC": limit, "QUERY": bounded},
                                 timeout=timeout, provider="datalab")
        rows, actual_format = parse_response(response, limit)
        status, error, ttl = ("match" if rows else "no_match"), None, CACHE_TTL_DAYS
    except Exception as exc:  # noqa: BLE001 - cache the unavailable state
        rows, actual_format, status, error, ttl = [], fmt, "unavailable", str(exc), ERROR_TTL_MINUTES / (24 * 60)
    fetched = _utc(now)
    expires = _utc(now + timedelta(days=ttl))
    metadata.put_tap_cache(root, cache_key=cache_key, service=service, release=str(release),
                           query_hash=query_hash, query=query_json, status=status,
                           response={"rows": rows, "format": actual_format}, error=error,
                           fetched_utc=fetched, expires_utc=expires)
    return {"schema_version": SCHEMA_VERSION, "service": service, "release": release,
            "state": status, "rows": rows, "format": actual_format, "query": query_json,
            "error": error, "fetched_utc": fetched, "expires_utc": expires,
            "cache": {"state": "refreshed" if cached else "miss", "stale": False}}


def status(root: Path | None = None) -> dict[str, Any]:
    return {"ttl_days": CACHE_TTL_DAYS, "cache": metadata.tap_cache_summary(root or config.PATHS.projects),
            "max_rows": MAX_ROWS, "formats": ["csv", "votable"]}
