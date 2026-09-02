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

`_PS_COLUMNS` also carries the stellar/insolation/mass columns
`habitability.py` (roadmap: astrophysics & extraterrestrial-study feature
pass) needs to score a planet -- `pl_insol`, `pl_eqt`, `pl_orbsmax`,
`pl_orbeccen`, `pl_bmasse`, `st_teff`, `st_rad`, `st_lum`, `st_mass`,
`sy_dist` -- added to the same `ps` table query rather than a second
round-trip, since the Exoplanet Archive already publishes them on that
table (NASA Exoplanet Archive Planetary Systems (PS) table column
definitions, https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html,
consulted this session; not re-derived from memory). `query_planets_bounded`
adds a second, catalog-scale query path alongside the existing
single-host/single-planet `query_confirmed_planets` -- still a fixed,
ASTRA-parameterised ADQL template with only numeric bounds substituted via
`float()`, never caller SQL, and still bounded by `TOP {max_rows}`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from . import tap

# Confirmed live this session (2026-08-24): the bare `/TAP` path 404s; the
# real synchronous-query endpoint is `/TAP/sync`.
EXOPLANET_ARCHIVE_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
_PS_COLUMNS = (
    "pl_name", "hostname", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
    "pl_trandur", "pl_trandep", "pl_rade", "pl_radj", "pl_tranmid",
    # habitability.py inputs (roadmap: astrophysics & extraterrestrial-study
    # feature pass) -- insolation flux, equilibrium temperature, semi-major
    # axis, eccentricity, planet mass, and stellar Teff/radius/luminosity/
    # mass/distance, all published on the same `ps` table.
    "pl_insol", "pl_eqt", "pl_orbsmax", "pl_orbeccen", "pl_bmasse",
    "st_teff", "st_rad", "st_lum", "st_mass", "sy_dist",
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
    # habitability.py inputs -- all optional because not every confirmed
    # planet has every derived quantity published.
    insolation_earth: float | None = None
    eq_temp_k: float | None = None
    semimajor_au: float | None = None
    eccentricity: float | None = None
    mass_earth: float | None = None
    st_teff_k: float | None = None
    st_radius_rsun: float | None = None
    st_luminosity_lsun: float | None = None
    st_mass_msun: float | None = None
    distance_pc: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        insolation_earth=_row_float(row, "pl_insol"), eq_temp_k=_row_float(row, "pl_eqt"),
        semimajor_au=_row_float(row, "pl_orbsmax"), eccentricity=_row_float(row, "pl_orbeccen"),
        mass_earth=_row_float(row, "pl_bmasse"), st_teff_k=_row_float(row, "st_teff"),
        st_radius_rsun=_row_float(row, "st_rad"), st_luminosity_lsun=_row_float(row, "st_lum"),
        st_mass_msun=_row_float(row, "st_mass"), distance_pc=_row_float(row, "sy_dist"),
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


def query_planets_bounded(*, teff_min: float | None = None, teff_max: float | None = None,
                          insolation_min: float | None = None, insolation_max: float | None = None,
                          max_rows: int = 500, root=None, refresh: bool = False,
                          offline: bool = False) -> list[PlanetRecord]:
    """Catalog-scale planet query for `habitability.rank_planets`.

    Unlike `query_confirmed_planets` (exactly one host/planet name), this
    builds a WHERE clause from numeric bounds only -- each bound is coerced
    through `float()` before formatting, so no caller string ever reaches
    the ADQL text. `max_rows` still caps the request via `TOP {max_rows}`,
    the same "no unbounded query" contract the module docstring states.
    """
    column_list = ", ".join(_PS_COLUMNS)
    clauses = ["default_flag = 1"]
    if teff_min is not None:
        clauses.append(f"st_teff >= {float(teff_min)!r}")
    if teff_max is not None:
        clauses.append(f"st_teff <= {float(teff_max)!r}")
    if insolation_min is not None:
        clauses.append(f"pl_insol >= {float(insolation_min)!r}")
    if insolation_max is not None:
        clauses.append(f"pl_insol <= {float(insolation_max)!r}")
    where = " AND ".join(clauses)
    adql = f"SELECT TOP {int(max_rows)} {column_list} FROM ps WHERE {where}"

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
    "query_planets_bounded", "compare_to_published", "EXOPLANET_ARCHIVE_TAP_URL",
]
