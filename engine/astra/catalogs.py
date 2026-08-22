"""Optional, cached catalogue cross-reference for candidate review.

Network catalogues are evidence, not a dependency of discovery.  Candidate
generation never calls this module; a researcher explicitly runs enrichment
after candidates exist.  Responses are cached in the project SQLite index by
provider, release, and a canonical cone-query hash so offline review remains
possible and catalogue changes are auditable.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from . import config, credentials, metadata, scoring

CACHE_TTL_DAYS = 30
ERROR_TTL_MINUTES = 5
PROVIDER_RELEASES = {
    "simbad": "SIMBAD-current",
    "vsx": "VSX-B/vsx/vsx",
    "tns": "TNS-API-current",
}
REQUEST_INTERVAL_SECONDS = {"simbad": 0.25, "vsx": 0.5, "tns": 1.0}
_last_request_at: dict[str, float] = {}


class CatalogError(RuntimeError):
    """A public catalogue could not produce a scientifically usable answer."""


class CatalogRateLimitError(CatalogError):
    """The provider asked ASTRA to slow down; cached work remains usable."""


@dataclass(frozen=True)
class CatalogQuery:
    """A canonical cone lookup identity, rounded only for cache stability."""

    object_id: str
    ra_deg: float
    dec_deg: float
    radius_arcsec: float = 2.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.ra_deg) and math.isfinite(self.dec_deg)):
            raise ValueError("catalogue lookup requires finite sky coordinates")
        if not (0.0 <= self.ra_deg < 360.0 and -90.0 <= self.dec_deg <= 90.0):
            raise ValueError("catalogue lookup coordinates are out of range")
        if not math.isfinite(self.radius_arcsec) or self.radius_arcsec <= 0:
            raise ValueError("catalogue lookup radius must be positive")

    def canonical(self) -> dict[str, Any]:
        return {
            "object_id": str(self.object_id),
            "ra_deg": round(float(self.ra_deg), 7),
            "dec_deg": round(float(self.dec_deg), 7),
            "radius_arcsec": round(float(self.radius_arcsec), 4),
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _cache_identity(provider: str, query: CatalogQuery) -> tuple[str, str, str, dict]:
    if provider not in PROVIDER_RELEASES:
        raise ValueError(f"unknown catalogue provider {provider!r}")
    release = PROVIDER_RELEASES[provider]
    canonical = query.canonical()
    query_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_key = f"{provider}:{release}:{query_hash}"
    return cache_key, release, query_hash, canonical


def _sleep_for_rate_limit(provider: str) -> None:
    interval = REQUEST_INTERVAL_SECONDS[provider]
    previous = _last_request_at.get(provider)
    now = time.monotonic()
    if previous is not None:
        remaining = interval - (now - previous)
        if remaining > 0:
            time.sleep(remaining)
    _last_request_at[provider] = time.monotonic()


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    text = str(value).strip()
    return text or None


def _table_value(row: Any, *names: str) -> str | None:
    columns = {str(name).lower(): str(name) for name in row.colnames}
    for name in names:
        column = columns.get(name.lower())
        if column is not None:
            return _text(row[column])
    return None


def _is_simbad_variable(otype: str | None) -> bool:
    if not otype:
        return False
    value = otype.upper()
    # SIMBAD's compact object-type codes include V*, EB*, RV*, and related
    # variable-star categories.  A generic '*' is intentionally not enough.
    return ("V*" in value or "EB*" in value or "RV*" in value or
            "PULS" in value or "VARIABLE" in value)


def _fetch_simbad(query: CatalogQuery) -> list[dict]:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.exceptions import NoResultsWarning
    from astroquery.simbad import Simbad

    client = Simbad()
    client.ROW_LIMIT = 20
    try:
        client.add_votable_fields("otype")
    except Exception:
        # Older/newer astroquery versions can already include this field.
        pass
    # SIMBAD uses a warning for an ordinary empty cone.  That is scientific
    # evidence (`no_match`), not an operational fault; keep the expected empty
    # result quiet while allowing all other warnings and exceptions through.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NoResultsWarning)
        table = client.query_region(
            SkyCoord(query.ra_deg, query.dec_deg, unit="deg", frame="icrs"),
            radius=query.radius_arcsec * u.arcsec,
        )
    if table is None:
        return []
    return [{
        "main_id": _table_value(row, "main_id"),
        "object_type": _table_value(row, "otype", "otype_v"),
        "is_variable": _is_simbad_variable(_table_value(row, "otype", "otype_v")),
    } for row in table]


def _fetch_vsx(query: CatalogQuery) -> list[dict]:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    client = Vizier(columns=["Name", "Type", "Period", "_RAJ2000", "_DEJ2000"],
                    row_limit=20)
    tables = client.query_region(
        SkyCoord(query.ra_deg, query.dec_deg, unit="deg", frame="icrs"),
        radius=query.radius_arcsec * u.arcsec,
        catalog="B/vsx/vsx",
    )
    if not tables:
        return []
    matches: list[dict] = []
    for table in tables:
        for row in table:
            period = _table_value(row, "period")
            try:
                period_days = float(period) if period is not None else None
            except ValueError:
                period_days = None
            matches.append({
                "name": _table_value(row, "name", "oid"),
                "variable_type": _table_value(row, "type"),
                "period_days": period_days,
            })
    return matches


def _tns_error(response: Any) -> CatalogError:
    status = int(getattr(response, "status_code", 0) or 0)
    if status == 429:
        return CatalogRateLimitError("TNS rate limit reached")
    return CatalogError("TNS request failed")


def _fetch_tns(query: CatalogQuery) -> list[dict]:
    secret = credentials.load_tns_credentials()
    if secret is None:
        raise CatalogError("TNS credentials are not configured")
    try:
        import requests
    except ImportError as exc:
        raise CatalogError("TNS client dependency is unavailable") from exc

    data = {
        "ra": f"{query.ra_deg:.7f}", "dec": f"{query.dec_deg:.7f}",
        "radius": f"{query.radius_arcsec:.4f}", "units": "arcsec",
        "objname": "", "internal_name": "", "public_timestamp": "0",
    }
    marker = {"tns_id": secret["bot_id"], "type": "bot",
              "name": secret["bot_name"]}
    try:
        response = requests.post(
            "https://www.wis-tns.org/api/get/search",
            data={"api_key": secret["api_key"], "data": json.dumps(data)},
            headers={"User-Agent": "tns_marker" + json.dumps(marker, separators=(",", ":"))},
            timeout=(5, 30),
        )
    except requests.RequestException as exc:
        raise CatalogError("TNS request failed") from exc
    if not response.ok:
        raise _tns_error(response)
    try:
        reply = response.json().get("data", {}).get("reply", [])
    except (ValueError, AttributeError) as exc:
        raise CatalogError("TNS returned an unreadable response") from exc
    if not isinstance(reply, list):
        raise CatalogError("TNS returned an unexpected response")
    return [{
        "name": _text(entry.get("objname") or entry.get("objid")),
        "object_type": _text(entry.get("object_type") or entry.get("objtype")),
        "discovery_date": _text(entry.get("discoverydate")),
    } for entry in reply if isinstance(entry, dict)]


FETCHERS: dict[str, Callable[[CatalogQuery], list[dict]]] = {
    "simbad": _fetch_simbad,
    "vsx": _fetch_vsx,
    "tns": _fetch_tns,
}


def _cached_result(entry: dict, *, cache_state: str, stale: bool = False) -> dict:
    response = entry.get("response") or {}
    return {
        "provider": entry["provider"], "release": entry["release"],
        "state": entry["status"], "matches": response.get("matches", []),
        "error": entry.get("error"), "fetched_utc": entry["fetched_utc"],
        "expires_utc": entry["expires_utc"],
        "cache": {"state": cache_state, "stale": stale},
    }


def query_provider(provider: str, query: CatalogQuery, *, root: Path | None = None,
                   refresh: bool = False, offline: bool = False,
                   fetcher: Callable[[CatalogQuery], list[dict]] | None = None,
                   now: datetime | None = None) -> dict:
    """Return one provider's result, using a versioned TTL cache when valid."""
    root = root or config.PATHS.projects
    now = now or _now()
    cache_key, release, query_hash, canonical = _cache_identity(provider, query)
    cached = metadata.get_catalog_cache(root, cache_key)
    if cached is not None and not refresh and _parse_utc(cached["expires_utc"]) > now:
        return _cached_result(cached, cache_state="hit")

    if offline:
        if cached is not None:
            return _cached_result(cached, cache_state="stale_offline", stale=True)
        return {
            "provider": provider, "release": release, "state": "offline",
            "matches": [], "error": None, "fetched_utc": None,
            "expires_utc": None, "cache": {"state": "miss", "stale": False},
        }

    try:
        _sleep_for_rate_limit(provider)
        matches = (fetcher or FETCHERS[provider])(query)
        matches = list(matches or [])
        status, error, ttl = ("match" if matches else "no_match"), None, CACHE_TTL_DAYS
    except CatalogRateLimitError as exc:
        matches, status, error, ttl = [], "rate_limited", str(exc), ERROR_TTL_MINUTES / (24 * 60)
    except (CatalogError, credentials.CredentialError) as exc:
        matches, status, error, ttl = [], "unavailable", str(exc), ERROR_TTL_MINUTES / (24 * 60)
    except Exception:  # Provider exceptions can include request internals; do not leak them.
        matches, status, error, ttl = [], "unavailable", "catalogue request failed", ERROR_TTL_MINUTES / (24 * 60)

    fetched = _utc(now)
    expires = _utc(now + timedelta(days=ttl))
    metadata.put_catalog_cache(
        root, cache_key=cache_key, provider=provider, release=release,
        query_hash=query_hash, query=canonical, object_id=query.object_id,
        ra_deg=query.ra_deg, dec_deg=query.dec_deg, radius_arcsec=query.radius_arcsec,
        status=status, response={"matches": matches}, error=error,
        fetched_utc=fetched, expires_utc=expires,
    )
    return {
        "provider": provider, "release": release, "state": status,
        "matches": matches, "error": error, "fetched_utc": fetched,
        "expires_utc": expires, "cache": {"state": "refreshed" if cached else "miss", "stale": False},
    }


def _not_requested(provider: str, state: str = "not_requested", reason: str | None = None) -> dict:
    return {"provider": provider, "release": PROVIDER_RELEASES[provider],
            "state": state, "matches": [], "error": reason,
            "fetched_utc": None, "expires_utc": None,
            "cache": {"state": "none", "stale": False}}


def summarize(results: dict[str, dict]) -> dict:
    """Reduce provider answers to transparent novelty evidence.

    A no-match answer is used only after the corresponding provider actually
    responded.  Offline, rate-limited, and not-configured are deliberately
    distinct so the scorer never treats absence of a lookup as novelty.
    """
    simbad = results.get("simbad", _not_requested("simbad"))
    vsx = results.get("vsx", _not_requested("vsx"))
    tns = results.get("tns", _not_requested("tns"))
    simbad_matches = simbad.get("matches", [])
    known_variable = bool(vsx.get("matches") or tns.get("matches") or any(
        match.get("is_variable") for match in simbad_matches if isinstance(match, dict)))
    known_object = bool(simbad_matches or vsx.get("matches") or tns.get("matches"))
    public_complete = all(results.get(provider, {}).get("state") in {"match", "no_match"}
                          for provider in ("simbad", "vsx"))
    tns_complete = tns.get("state") in {"match", "no_match"}
    return {
        "known_variable": known_variable,
        "known_object": known_object,
        "public_complete": public_complete,
        "tns_complete": tns_complete,
        "states": {provider: result.get("state", "not_requested")
                   for provider, result in results.items()},
    }


def enrich_position(object_id: str, ra_deg: float, dec_deg: float, *,
                    radius_arcsec: float = 2.0, root: Path | None = None,
                    refresh: bool = False, offline: bool = False,
                    include_tns: bool = True,
                    fetchers: dict[str, Callable[[CatalogQuery], list[dict]]] | None = None) -> dict:
    """Cross-reference one object without making candidate generation wait."""
    query = CatalogQuery(str(object_id), float(ra_deg), float(dec_deg), float(radius_arcsec))
    root = root or config.PATHS.projects
    fetchers = fetchers or {}
    results: dict[str, dict] = {}
    for provider in ("simbad", "vsx"):
        results[provider] = query_provider(
            provider, query, root=root, refresh=refresh, offline=offline,
            fetcher=fetchers.get(provider),
        )
    if not include_tns:
        results["tns"] = _not_requested("tns")
    else:
        try:
            configured = credentials.load_tns_credentials() is not None
        except credentials.CredentialError:
            configured = False
        if not configured:
            results["tns"] = _not_requested("tns", "not_configured",
                                             "protected TNS credentials are not configured")
        else:
            results["tns"] = query_provider(
                "tns", query, root=root, refresh=refresh, offline=offline,
                fetcher=fetchers.get("tns"),
            )
    return {"query": query.canonical(), "providers": results,
            "summary": summarize(results)}


def _apply_to_candidate(candidate: Any, evidence: dict) -> None:
    """Update only the catalog component while preserving prior score evidence."""
    previous = candidate.score.get("components", {})
    components = {name: previous.get(name) for name in scoring.WEIGHTS}
    novelty, reason = scoring.catalog_novelty(catalog_evidence=evidence)
    components["catalog_novelty"] = novelty
    prior_reasons = [entry for entry in candidate.score.get("reasons", [])
                     if not str(entry).startswith("novelty:")]
    candidate.score = scoring.combine(components, prior_reasons + [f"novelty: {reason}"]).to_dict()
    candidate.catalog = evidence
    candidate.explanation["catalogue_cross_reference"] = evidence["summary"]
    actions = [action for action in candidate.explanation.get("recommended_actions", [])
               if "Cross-reference against SIMBAD" not in action]
    if novelty is None:
        actions.append("Catalog cross-reference is incomplete; retry when SIMBAD and VSX are available.")
    candidate.explanation["recommended_actions"] = actions


def enrich_candidates(name: str = "default", *, root: Path | None = None,
                      radius_arcsec: float = 2.0, refresh: bool = False,
                      offline: bool = False, include_tns: bool = True) -> dict:
    """Enrich an existing candidate file as an explicit, resumable evidence job."""
    from . import candidates

    root = root or config.PATHS.projects
    built = candidates.load(name, root)
    counts: dict[str, int] = {}
    for candidate in built:
        try:
            evidence = enrich_position(
                candidate.object_id, candidate.ra_deg, candidate.dec_deg,
                radius_arcsec=radius_arcsec, root=root, refresh=refresh,
                offline=offline, include_tns=include_tns,
            )
        except ValueError as exc:
            evidence = {
                "query": {"object_id": candidate.object_id},
                "providers": {provider: _not_requested(provider, "not_queried", str(exc))
                              for provider in PROVIDER_RELEASES},
                "summary": {"known_variable": False, "known_object": False,
                            "public_complete": False, "tns_complete": False,
                            "states": {provider: "not_queried" for provider in PROVIDER_RELEASES}},
            }
        _apply_to_candidate(candidate, evidence)
        for state in evidence["summary"]["states"].values():
            counts[state] = counts.get(state, 0) + 1

    ranked = candidates.rank(built)
    path = candidates.save(ranked, name, root)
    return {"name": name, "candidates": len(ranked), "state_counts": counts,
            "output_path": str(path), "offline": offline, "refresh": refresh}


def status(root: Path | None = None) -> dict:
    root = root or config.PATHS.projects
    return {"ttl_days": CACHE_TTL_DAYS,
            "cache": metadata.catalog_cache_summary(root),
            "tns_credentials": credentials.tns_credential_status()}
