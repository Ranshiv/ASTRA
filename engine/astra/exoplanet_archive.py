"""NASA Exoplanet Archive cross-check for `transit_ttv.py` candidates.

Reuses `tap.py`'s existing bounded, cached, read-only IVOA TAP/ADQL client
rather than writing new HTTP/parsing code -- the Exoplanet Archive publishes
a real, confirmed-live TAP sync endpoint
(`https://exoplanetarchive.ipac.caltech.edu/TAP/sync`, confirmed this session
via a direct query -- the bare `/TAP` path 404s) against its `ps` (Planetary
Systems) table. Only a fixed, ASTRA-parameterised ADQL template is ever
sent; no caller SQL reaches the service, the same "no user SQL" contract
`des.py`'s NOIRLab Astro Data Lab query already documents for `tap.query`.

This is diagnostic cross-check evidence only -- comparing a recovered
period/depth/duration against a published value -- never a correction
applied to a fitted result and never folded into ranking, the same
"catalog_relative, not resolved-confirmation" restraint
`blend.source_attribution` already applies elsewhere in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import tap

# Confirmed live this session (2026-08-24): the bare `/TAP` path 404s; the
# real synchronous-query endpoint is `/TAP/sync`.
EXOPLANET_ARCHIVE_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
_PS_COLUMNS = (
    "pl_name", "hostname", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
    "pl_trandur", "pl_trandep", "pl_rade", "pl_radj", "pl_tranmid",
)


class ExoplanetArchiveError(ValueError):
    """A confirmed-planet query request could not be built or parsed."""


@dataclass(frozen=True)
class PlanetRecord:
    name: str
    host_name: str
    period_days: float | None
    period_err_days: float | None
    duration_hours: float | None
    depth_ppm: float | None
    radius_earth: float | None
    transit_midpoint_bjd: float | None


def _sql_literal(text: str) -> str:
    """Escape a string for embedding in a single-quoted ADQL literal."""
    return str(text).replace("'", "''")


def _row_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_to_record(row: dict[str, Any]) -> PlanetRecord:
    return PlanetRecord(
        name=str(row.get("pl_name") or ""), host_name=str(row.get("hostname") or ""),
        period_days=_row_float(row, "pl_orbper"), period_err_days=_row_float(row, "pl_orbpererr1"),
        duration_hours=_row_float(row, "pl_trandur"), depth_ppm=_row_float(row, "pl_trandep"),
        radius_earth=_row_float(row, "pl_rade"), transit_midpoint_bjd=_row_float(row, "pl_tranmid"),
    )


def query_confirmed_planets(*, host_name: str | None = None, planet_name: str | None = None,
                           max_rows: int = 50, root=None, refresh: bool = False,
                           offline: bool = False) -> list[PlanetRecord]:
    """Confirmed-planet parameters from the Exoplanet Archive's `ps` table.

    Exactly one of `host_name`/`planet_name` must be supplied -- an
    unbounded query against the whole table is never built here.
    """
    if bool(host_name) == bool(planet_name):
        raise ExoplanetArchiveError("supply exactly one of host_name or planet_name")
    column_list = ", ".join(_PS_COLUMNS)
    if host_name:
        clause = f"hostname = '{_sql_literal(host_name)}'"
    else:
        clause = f"pl_name = '{_sql_literal(planet_name)}'"
    adql = f"SELECT TOP {int(max_rows)} {column_list} FROM ps WHERE {clause} AND default_flag = 1"

    result = tap.query(EXOPLANET_ARCHIVE_TAP_URL, adql, release="exoplanetarchive",
                       max_rows=max_rows, root=root, refresh=refresh, offline=offline,
                       provider="exoplanetarchive")
    if result["state"] == "unavailable":
        raise ExoplanetArchiveError(f"Exoplanet Archive TAP request failed: {result['error']}")
    return [_row_to_record(row) for row in result["rows"]]


def compare_to_published(fitted_period_days: float, fitted_depth: float,
                         published: PlanetRecord) -> dict[str, float | None]:
    """Fractional differences between a fitted candidate and a published
    record -- a diagnostic only, per the module docstring."""
    period_diff = None
    if published.period_days:
        period_diff = (fitted_period_days - published.period_days) / published.period_days
    depth_diff = None
    if published.depth_ppm:
        published_depth_fraction = published.depth_ppm * 1e-6
        depth_diff = (fitted_depth - published_depth_fraction) / published_depth_fraction
    return {"period_fractional_diff": period_diff, "depth_fractional_diff": depth_diff}


__all__ = [
    "ExoplanetArchiveError", "PlanetRecord", "query_confirmed_planets",
    "compare_to_published", "EXOPLANET_ARCHIVE_TAP_URL",
]
