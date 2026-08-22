"""Offline-first literature enrichment for candidate research.

Literature is context, not a ranking signal.  This module keeps provider
responses versioned and cacheable, supports NASA ADS when a user supplies an
API token, and uses the public arXiv feed as a no-credential fallback.  A
missing provider is represented as ``unavailable`` rather than as evidence
that an object has no relevant literature.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import config, metadata, netclient

SCHEMA_VERSION = 1
CACHE_TTL_DAYS = 30
ERROR_TTL_MINUTES = 5
PROVIDER_RELEASES = {"ads": "ADS-current", "arxiv": "arXiv-export-api"}
MAX_RESULTS = 100


class LiteratureError(RuntimeError):
    """A literature provider could not produce a usable response."""


@dataclass(frozen=True)
class LiteratureQuery:
    object_id: str
    terms: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()
    limit: int = 20

    def canonical(self) -> dict[str, Any]:
        terms = tuple(sorted({str(item).strip() for item in self.terms if str(item).strip()}))
        events = tuple(sorted({str(item).strip() for item in self.event_ids if str(item).strip()}))
        return {
            "object_id": str(self.object_id).strip(),
            "terms": list(terms[:20]),
            "event_ids": list(events[:20]),
            "limit": max(1, min(int(self.limit), MAX_RESULTS)),
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _identity(provider: str, query: LiteratureQuery) -> tuple[str, str, str, dict[str, Any]]:
    if provider not in PROVIDER_RELEASES:
        raise ValueError(f"unknown literature provider {provider!r}")
    canonical = query.canonical()
    query_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"literature:{provider}:{PROVIDER_RELEASES[provider]}:{query_hash}", \
        PROVIDER_RELEASES[provider], query_hash, canonical


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _year(value: object) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def _terms(query: LiteratureQuery) -> list[str]:
    canonical = query.canonical()
    terms = [canonical["object_id"], *canonical["event_ids"], *canonical["terms"]]
    return [term for term in terms if term]


def _ads_query(query: LiteratureQuery) -> str:
    terms = _terms(query)
    if not terms:
        raise LiteratureError("literature search requires an object identifier or terms")
    clauses = []
    for term in terms:
        escaped = term.replace('"', "")
        clauses.append(f'"{escaped}"')
    return " OR ".join(clauses)


def _normalize_ads(doc: dict[str, Any]) -> dict[str, Any]:
    bibcode = _text(doc.get("bibcode"))
    title = doc.get("title")
    if isinstance(title, list):
        title = title[0] if title else None
    authors = doc.get("author") or doc.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    raw_doi = doc.get("doi")
    if isinstance(raw_doi, list):
        raw_doi = raw_doi[0] if raw_doi else None
    return {
        "provider": "ads", "bibcode": bibcode,
        "title": _text(title),
        "authors": [str(item) for item in authors[:20]],
        "abstract": _text(doc.get("abstract")),
        "year": _year(doc.get("year") or doc.get("pub")),
        "doi": _text(raw_doi),
        "url": f"https://ui.adsabs.harvard.edu/abs/{bibcode}" if bibcode else None,
        "citation_count": doc.get("citation_count"),
    }


def _fetch_ads(query: LiteratureQuery) -> list[dict[str, Any]]:
    token = os.environ.get("ADS_API_TOKEN", "").strip()
    if not token:
        raise LiteratureError("ADS_API_TOKEN is not configured")
    response = netclient.get(
        "https://api.adsabs.harvard.edu/v1/search/query",
        {"q": _ads_query(query), "fl": "bibcode,title,author,abstract,year,doi,citation_count",
         "rows": query.canonical()["limit"]},
        timeout=30, provider="ads",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        docs = response.json().get("response", {}).get("docs", [])
    except (ValueError, AttributeError) as exc:
        raise LiteratureError("ADS returned unreadable JSON") from exc
    if not isinstance(docs, list):
        raise LiteratureError("ADS returned an unexpected response")
    return [_normalize_ads(row) for row in docs if isinstance(row, dict)][:MAX_RESULTS]


def _atom_text(entry: ET.Element, local: str) -> str | None:
    for node in entry.iter():
        if node.tag.rsplit("}", 1)[-1].lower() == local.lower():
            return _text(node.text)
    return None


def _fetch_arxiv(query: LiteratureQuery) -> list[dict[str, Any]]:
    terms = _terms(query)
    if not terms:
        raise LiteratureError("literature search requires an object identifier or terms")
    response = netclient.get(
        "https://export.arxiv.org/api/query",
        {"search_query": f"all:{terms[0]}",
         "start": 0, "max_results": query.canonical()["limit"],
         "sortBy": "relevance", "sortOrder": "descending"},
        timeout=30, provider="arxiv",
    )
    try:
        root = ET.fromstring(response.text)
    except (ET.ParseError, AttributeError) as exc:
        raise LiteratureError("arXiv returned unreadable Atom XML") from exc
    entries: list[dict[str, Any]] = []
    for entry in root.iter():
        if entry.tag.rsplit("}", 1)[-1].lower() != "entry":
            continue
        identifier = _atom_text(entry, "id")
        title = _atom_text(entry, "title")
        published = _atom_text(entry, "published")
        authors = []
        for child in entry.iter():
            if child.tag.rsplit("}", 1)[-1].lower() == "name" and child.text:
                authors.append(child.text.strip())
        entries.append({
            "provider": "arxiv", "bibcode": None, "arxiv_id": identifier,
            "title": " ".join((title or "").split()) or None,
            "authors": authors[:20], "abstract": " ".join((_atom_text(entry, "summary") or "").split()) or None,
            "year": _year(published), "doi": None, "url": identifier,
            "citation_count": None,
        })
    return entries[:MAX_RESULTS]


FETCHERS: dict[str, Callable[[LiteratureQuery], list[dict[str, Any]]]] = {
    "ads": _fetch_ads,
    "arxiv": _fetch_arxiv,
}


def _cached(entry: dict, *, cache_state: str, stale: bool = False) -> dict[str, Any]:
    response = entry.get("response") or {}
    return {
        "provider": entry["provider"], "release": entry["release"],
        "state": entry["status"], "records": response.get("records", []),
        "error": entry.get("error"), "fetched_utc": entry["fetched_utc"],
        "expires_utc": entry["expires_utc"],
        "cache": {"state": cache_state, "stale": stale},
    }


def query_provider(provider: str, query: LiteratureQuery, *, root: Path | None = None,
                   refresh: bool = False, offline: bool = False,
                   fetcher: Callable[[LiteratureQuery], list[dict[str, Any]]] | None = None,
                   now: datetime | None = None) -> dict[str, Any]:
    root = root or config.PATHS.projects
    now = now or _now()
    cache_key, release, query_hash, canonical = _identity(provider, query)
    cached = metadata.get_literature_cache(root, cache_key)
    if cached is not None and not refresh and _parse_utc(cached["expires_utc"]) > now:
        return _cached(cached, cache_state="hit")
    if offline:
        if cached is not None:
            return _cached(cached, cache_state="stale_offline", stale=True)
        return {"provider": provider, "release": release, "state": "offline", "records": [],
                "error": None, "fetched_utc": None, "expires_utc": None,
                "cache": {"state": "miss", "stale": False}}
    try:
        records = list((fetcher or FETCHERS[provider])(query) or [])
        status, error, ttl = ("match" if records else "no_match"), None, CACHE_TTL_DAYS
    except LiteratureError as exc:
        records, status, error, ttl = [], "unavailable", str(exc), ERROR_TTL_MINUTES / (24 * 60)
    except Exception:  # provider details do not belong in the RPC error surface
        records, status, error, ttl = [], "unavailable", "literature request failed", ERROR_TTL_MINUTES / (24 * 60)
    fetched = _utc(now)
    expires = _utc(now + timedelta(days=ttl))
    metadata.put_literature_cache(
        root, cache_key=cache_key, provider=provider, release=release,
        query_hash=query_hash, query=canonical, status=status,
        response={"records": records}, error=error,
        fetched_utc=fetched, expires_utc=expires,
    )
    return {"provider": provider, "release": release, "state": status,
            "records": records, "error": error, "fetched_utc": fetched,
            "expires_utc": expires,
            "cache": {"state": "refreshed" if cached else "miss", "stale": False}}


def search(*, object_id: str = "", terms: Iterable[str] = (), event_ids: Iterable[str] = (),
           providers: Iterable[str] = ("ads", "arxiv"), limit: int = 20,
           root: Path | None = None, refresh: bool = False, offline: bool = False,
           fetchers: dict[str, Callable[[LiteratureQuery], list[dict[str, Any]]]] | None = None) -> dict[str, Any]:
    query = LiteratureQuery(str(object_id), tuple(terms), tuple(event_ids), int(limit))
    results = {}
    for provider in providers:
        results[str(provider)] = query_provider(
            str(provider), query, root=root, refresh=refresh, offline=offline,
            fetcher=(fetchers or {}).get(str(provider)),
        )
    records = [record for result in results.values() for record in result.get("records", [])]
    records.sort(key=lambda row: (-(int(row.get("year") or 0)), str(row.get("title") or "")))
    return {
        "schema_version": SCHEMA_VERSION, "query": query.canonical(),
        "providers": results, "records": records[:MAX_RESULTS],
        "complete": all(result.get("state") in {"match", "no_match"}
                         for result in results.values()),
        "provenance": [{"kind": "literature", "provider": name,
                        "release": result.get("release"), "state": result.get("state"),
                        "fetched_utc": result.get("fetched_utc"),
                        "cache": result.get("cache")}
                       for name, result in results.items()],
    }


def enrich_candidates(name: str = "default", *, root: Path | None = None,
                      refresh: bool = False, offline: bool = False,
                      include_arxiv: bool = True, limit: int = 20) -> dict[str, Any]:
    """Attach literature context to candidates without changing their scores."""
    from . import candidates

    root = root or config.PATHS.projects
    built = candidates.load(name, root)
    counts: dict[str, int] = {}
    for candidate in built:
        terms = list(candidate.explanation.get("resembles", []))
        result = search(
            object_id=candidate.object_id, terms=terms, event_ids=candidate.event_ids,
            providers=("ads", "arxiv") if include_arxiv else ("ads",),
            limit=limit, root=root, refresh=refresh, offline=offline,
        )
        candidate.literature = result
        candidate.explanation["literature"] = {
            "records": len(result["records"]), "complete": result["complete"],
            "states": {key: value.get("state") for key, value in result["providers"].items()},
        }
        candidate.provenance_refs.extend(result["provenance"])
        for value in result["providers"].values():
            state = value.get("state", "unknown")
            counts[state] = counts.get(state, 0) + 1
    path = candidates.save(built, name, root)
    return {"name": name, "candidates": len(built), "state_counts": counts,
            "records": sum(len(item.literature.get("records", [])) for item in built),
            "output_path": str(path), "offline": offline, "refresh": refresh}


def status(root: Path | None = None) -> dict[str, Any]:
    root = root or config.PATHS.projects
    return {"ttl_days": CACHE_TTL_DAYS, "providers": dict(PROVIDER_RELEASES),
            "cache": metadata.literature_cache_summary(root),
            "ads_token_configured": bool(os.environ.get("ADS_API_TOKEN", "").strip())}
