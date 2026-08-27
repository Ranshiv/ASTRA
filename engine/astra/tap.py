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
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from . import config, metadata, netclient

# IVOA UWS (Universal Worker Service) namespace/attribute names, verified
# live against a real NOIRLab Data Lab async TAP job while building
# `async_query` below.
_UWS_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
ASYNC_POLL_INTERVAL_SECONDS = 2.0
ASYNC_MAX_WAIT_SECONDS = 300.0

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
    # A real bug, found and fixed via a live check against DESI's
    # `targetid` column while building `async_query`: routing every
    # numeric-looking cell through `float()` first silently loses
    # precision for any integer beyond float64's 2**53 exact range (DESI
    # targetids are ~3.96e16, well past it -- a live query for a specific
    # targetid came back with a DIFFERENT number, off in the last two
    # digits, purely from this round trip). Plain integer text is parsed
    # with `int()` directly (Python's ints are exact and arbitrary
    # precision) before `float` is tried at all; only genuinely
    # non-integer-looking text (scientific notation, decimals) still goes
    # through the float path below.
    try:
        return int(text)
    except ValueError:
        pass
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


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].upper()


def parse_votable(payload: str, limit: int = MAX_ROWS) -> list[dict[str, object]]:
    """Parse every `<TABLE>` in the document, each scoped to its OWN
    `<FIELD>` set, concatenating their rows.

    A real bug found and fixed this session, running this function
    against a real multi-`<TABLE>` VizieR response for the first time
    (a plain, non-cone-search catalogue dump -- 6 real `<TABLE>` blocks
    in one document, confirmed live): the original implementation
    collected every `<FIELD>` and every `<TR>` in the WHOLE document into
    two flat, document-wide lists, then zipped each row's cell values
    against that single global field-name list by POSITION. With more
    than one table present, a table's own `<TR>` values (as many cells as
    ITS OWN field count) got zipped against the WRONG, longer/shorter
    global field list, so every returned row's fields came back `None`
    -- a real, silent, 100%-of-rows failure, not a partial one. Scoping
    both `fields` and the row-building loop to each `<TABLE>` in turn
    (still via the same namespace-agnostic `_local_tag` matching) fixes
    this while being a no-op for the single-table case every existing
    caller (`vlass.py`, `strong_lens.py`, `weak_lensing.py`, and this
    file's own TAP query path) already depends on and is tested against.
    """
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise TapError("TAP returned unreadable VOTable XML") from exc
    rows: list[dict[str, object]] = []
    tables = [node for node in root.iter() if _local_tag(node.tag) == "TABLE"]
    # A VOTable with no explicit <TABLE> wrapper (unusual, but not
    # impossible) still has FIELD/TR reachable from the root -- fall back
    # to treating the whole document as one table, the original
    # single-table behaviour.
    for table in tables or [root]:
        fields: list[str] = []
        for node in table.iter():
            if _local_tag(node.tag) == "FIELD":
                fields.append(str(node.attrib.get("name") or node.attrib.get("ID")
                                  or f"column_{len(fields)}"))
        for tr in table.iter():
            if _local_tag(tr.tag) != "TR":
                continue
            values = [_cell(td.text) for td in tr if _local_tag(td.tag) == "TD"]
            rows.append({name: values[index] if index < len(values) else None
                        for index, name in enumerate(fields)})
            if len(rows) >= limit:
                return rows
    return rows


def parse_response(response: Any, limit: int) -> tuple[list[dict[str, object]], str]:
    text = getattr(response, "text", "") or ""
    content_type = str((getattr(response, "headers", {}) or {}).get("Content-Type", "")).lower()
    if "xml" in content_type or text.lstrip().startswith("<?xml") or "<VOTABLE" in text.upper():
        return parse_votable(text, limit), "votable"
    return parse_csv(text, limit), "csv"


def _identity(service: str, release: str, adql: str, fmt: str, limit: int,
             provider: str) -> tuple[str, str, dict[str, Any]]:
    # `provider` is part of the cache identity (not just the netclient
    # throttle bucket): two providers hitting the same service URL with
    # different credentials/rate limits must never share a cached response.
    query = {"service": service, "release": release, "adql": adql, "format": fmt,
            "limit": limit, "provider": provider}
    query_hash = hashlib.sha256(json.dumps(query, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"tap:{query_hash}", query_hash, query


def query(service: str, adql: str, *, release: str = "unknown", root: Path | None = None,
          max_rows: int = 200, fmt: str = "csv", refresh: bool = False,
          offline: bool = False, timeout: float = 60.0, provider: str = "datalab",
          auth_header: dict[str, str] | None = None) -> dict[str, Any]:
    """Bounded, cached, read-only TAP query.

    `provider` selects both the `netclient.throttle` bucket and, via
    `_identity`, the cache partition -- previously hardcoded to `"datalab"`,
    generalized so a second TAP provider (e.g. a credential-gated Rubin/LSST
    connector) never shares a throttle interval or a cached response with an
    unrelated service. `auth_header`, when supplied, is passed straight
    through to `netclient.get` as request headers -- never cached or logged
    (only the resulting `rows`/`status` are persisted via
    `metadata.put_tap_cache`), matching the credential-handling discipline
    `credentials.py`'s DPAPI storage already applies elsewhere.
    """
    service = _validate_service(service)
    bounded, limit = bound_adql(adql, max_rows)
    fmt = str(fmt).lower()
    if fmt not in {"csv", "votable"}:
        raise ValueError("TAP format must be csv or votable")
    provider = str(provider)
    root = root or config.PATHS.projects
    cache_key, query_hash, query_json = _identity(service, str(release), bounded, fmt, limit, provider)
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
                                 timeout=timeout, provider=provider, headers=auth_header)
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


def _async_service_url(service: str) -> str:
    """The async sibling of a `/tap/sync` service URL."""
    if service.endswith("/sync"):
        return service[: -len("/sync")] + "/async"
    raise TapError(f"cannot derive an async TAP endpoint from {service!r} "
                   "(expected it to end with '/sync')")


def _parse_uws_job(status_xml: str) -> dict[str, str | None]:
    """The real job-status fields this module needs: phase, the completed
    result file's URL, and a failed job's own error message -- verified
    live against a real NOIRLab Data Lab async job (submission, a genuine
    completed job, and a genuine `PSQLException` failure) while building
    this function, not assumed from the UWS spec alone.
    """
    try:
        root = ET.fromstring(status_xml)
    except ET.ParseError as exc:
        raise TapError("async TAP job status was not valid XML") from exc
    phase: str | None = None
    result_href: str | None = None
    error_message: str | None = None
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "phase" and phase is None:
            phase = (node.text or "").strip()
        elif tag == "result" and result_href is None:
            result_href = node.attrib.get(_UWS_XLINK_HREF) or node.attrib.get("href")
        elif tag == "message" and error_message is None:
            error_message = (node.text or "").strip()
    return {"phase": phase, "result_href": result_href, "error_message": error_message}


def _run_async_job(async_service: str, adql: str, fmt: str, limit: int, provider: str,
                   timeout: float, auth_header: dict[str, str] | None,
                   poll_interval: float, max_wait_seconds: float
                   ) -> tuple[list[dict[str, object]], str]:
    """Submit, run, poll, and fetch the result of one real UWS async TAP
    job. See `async_query`'s docstring for the live-verified protocol
    shape this follows.
    """
    submitted = netclient.post(
        async_service, {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": fmt, "QUERY": adql},
        timeout=timeout, provider=provider, headers=auth_header)
    # `requests` follows the submission's `303 See Other` to the job's own
    # status URL (a GET) by default, so `submitted.url` IS the real job
    # URL and `submitted.text` IS that job's initial status XML -- no raw
    # `Location` header handling needed.
    job_url = submitted.url
    job_status = _parse_uws_job(submitted.text)
    phase = job_status["phase"] or "PENDING"

    if phase == "PENDING":
        netclient.post(f"{job_url}/phase", {"PHASE": "RUN"}, timeout=timeout,
                       provider=provider, headers=auth_header)

    deadline = time.monotonic() + max_wait_seconds
    while phase not in ("COMPLETED", "ERROR", "ABORTED"):
        if time.monotonic() >= deadline:
            raise TapError(f"async TAP job did not reach a terminal phase within "
                           f"{max_wait_seconds}s (last phase: {phase})")
        time.sleep(poll_interval)
        phase_response = netclient.get(f"{job_url}/phase", {}, timeout=timeout,
                                       provider=provider, headers=auth_header)
        phase = phase_response.text.strip()

    status_response = netclient.get(job_url, {}, timeout=timeout, provider=provider,
                                    headers=auth_header)
    job_status = _parse_uws_job(status_response.text)

    if phase != "COMPLETED":
        raise TapError(f"async TAP job ended in {phase}: "
                       f"{job_status['error_message'] or 'no error message reported'}")
    if not job_status["result_href"]:
        raise TapError("async TAP job completed but reported no result file")

    result_response = netclient.get(job_status["result_href"], {}, timeout=timeout,
                                    provider=provider, headers=auth_header)
    return parse_response(result_response, limit)


def async_query(service: str, adql: str, *, release: str = "unknown", root: Path | None = None,
                max_rows: int = 200, fmt: str = "csv", refresh: bool = False,
                offline: bool = False, timeout: float = 60.0, provider: str = "datalab",
                auth_header: dict[str, str] | None = None,
                poll_interval: float = ASYNC_POLL_INTERVAL_SECONDS,
                max_wait_seconds: float = ASYNC_MAX_WAIT_SECONDS) -> dict[str, Any]:
    """Bounded, cached, read-only ASYNC TAP query -- for a table where
    `query()`'s synchronous endpoint times out even on a simple, selective
    filter. Confirmed live this session: `desi_dr1.zpix` on NOIRLab Data
    Lab times out via `/tap/sync` on `WHERE targetid = <a real value>`
    (an exact-match lookup, not a large scan), but the IDENTICAL query
    completes via `/tap/async` in under a second -- the sync endpoint's
    connection-timeout window, not query execution speed, was the actual
    blocker. Returns the SAME shape `query()` does, so callers do not need
    to know which transport actually served the result.

    The real UWS (IVOA Universal Worker Service) protocol, verified live
    against NOIRLab Data Lab while building this function: `POST
    {service}/async` returns a job at the URL `requests` follows its
    `303 See Other` redirect to; a fresh job's phase is `PENDING`, NOT
    auto-started, so `PHASE=RUN` must be POSTed to `{job_url}/phase`
    explicitly; polling `GET {job_url}/phase` returns a bare phase string;
    a real completed job's status XML carries `<uws:result id="result"
    xlink:href="...">` pointing at a plain result file (confirmed live: a
    real CSV at a `resultStore` URL); a real failed job's
    `<uws:errorSummary>` carries the actual database error message (a
    genuine `PSQLException` was confirmed live for a geometric ADQL query
    unsupported on this particular table), surfaced here rather than a
    generic "job failed". `max_wait_seconds` defaults well under the
    service's own confirmed 3600s server-side `executionDuration`.
    """
    service = _validate_service(service)
    bounded, limit = bound_adql(adql, max_rows)
    fmt = str(fmt).lower()
    if fmt not in {"csv", "votable"}:
        raise ValueError("TAP format must be csv or votable")
    provider = str(provider)
    root = root or config.PATHS.projects
    # `provider` is suffixed so an async result never shares a cache
    # partition with a `query()` (sync) call against the identical ADQL --
    # they are genuinely different transports that may behave differently
    # against the same service.
    cache_key, query_hash, query_json = _identity(
        service, str(release), bounded, fmt, limit, f"{provider}:async")
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
        # A bug, found via a direct test of a malformed service URL: this
        # derivation was originally called BEFORE the try block below, so
        # its TapError (e.g. a service URL not ending in "/sync") escaped
        # uncaught instead of degrading to a normal "unavailable" result
        # the way every other async-job failure mode already does.
        async_service = _async_service_url(service)
        rows, actual_format = _run_async_job(
            async_service, bounded, fmt, limit, provider, timeout, auth_header,
            poll_interval, max_wait_seconds)
        status, error, ttl = ("match" if rows else "no_match"), None, CACHE_TTL_DAYS
    except Exception as exc:  # noqa: BLE001 - cache the unavailable state
        rows, actual_format = [], fmt
        status, error, ttl = "unavailable", str(exc), ERROR_TTL_MINUTES / (24 * 60)

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
