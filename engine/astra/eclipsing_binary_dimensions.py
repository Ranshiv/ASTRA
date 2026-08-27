"""Absolute physical dimensions (radii, masses) for a fitted eclipsing binary.

This is the "RV-inspired" half of roadmap item 17, split from
`eclipsing_binary.py` purely to keep each file under this project's 500-line
guideline (same `stellar_manifold.py`/`stellar_manifold_eval.py` split
rationale, not an independent module).

`eclipsing_binary.fit_eclipsing_binary` only ever recovers RATIOS from a
light curve alone: `r1_a`/`r2_a` (each body's radius in units of the
semi-major axis) and `teff_ratio`. Converting those into physical units
(solar radii, solar masses) normally requires real radial-velocity (RV)
spectroscopy: the RV semi-amplitudes give the semi-major axis `a` in
physical units directly, and from there Kepler's third law gives the total
mass. This codebase has no real RV data source (confirmed by grep across
`engine/astra/` while planning this module -- see `docs/DEFERRED.txt`).

What this module does INSTEAD, stated explicitly as a genuine substitute
technique used in the real literature when RV is unavailable, not a
fabricated shortcut: it anchors one body's PHYSICAL radius from Gaia
photometry (either Gaia's own `astrophysical_parameters.radius_gspphot`
estimate, or a fallback via `stellar_manifold.nearest_track_point`'s
CMD-derived effective temperature), propagates that anchor through the
light-curve-fitted radius RATIO to get both bodies' physical radii and the
physical semi-major axis, applies Kepler's third law for the TOTAL system
mass, then splits that total between the two bodies using a real
mass-Teff relation evaluated at each body's own (anchor-derived) effective
temperature -- not by assuming an equal split.

`MASS_RADIUS_ZAMS_TRACK` extends `stellar_manifold.ZAMS_TRACK`'s own real,
live-verified source (Eric Mamajek's maintained dwarf-star table,
`EEM_dwarf_UBVIJHK_colors_Teff.txt`, version 2022.04.16, fetched live this
session, 2026-08-24, from
https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt)
with that same table's own `R_Rsun`/`Msun` columns, for the identical
B9V-M7V spectral-type rows `stellar_manifold.ZAMS_TRACK` already uses. Like
that track, this is ONE static, approximately solar-metallicity
zero-age-main-sequence relation, not a fitted multi-age/multi-metallicity
grid -- an evolved or interacting EB component is expected to show a real
residual against it, not necessarily an error; the same scope limit
`stellar_manifold.py`'s own docstring states for its track.

`query_component_catalog`/`mass_radius_residuals` provide the "Metrics:
Mass/radius residuals against EB catalogs" cross-check this backlog item
names, reusing `catalogs.py`'s `_fetch_vsx`-shaped VizieR query pattern
(`astroquery.vizier.Vizier(columns=...).query_region(...)`) against a real,
live-verified catalog: VizieR `J/ApJ/709/535` (Brown 2010, "Eclipsing
binary stars with accurate mass and radius estimates"), confirmed live this
session by cone-searching the real, published coordinates of eclipsing
binary V760 Sco and getting back its real component B row (Mass=4.609
Msun, Rad=2.642 Rsun, Teff=16300 K) -- not assumed from documentation. Each
row is ONE binary COMPONENT (e.g. `Name="V760 Sco B"`), often sharing the
same catalogued position as its companion component to within the
catalogue's own astrometric precision -- a cone search can return one or
both components, and pairing them into a single system is left to the
caller. This is diagnostic-only, reported residuals never correct a fitted
result, the same `blend.source_attribution`/
`exoplanet_archive.compare_to_published` restraint every prior
cross-check in this codebase applies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .eclipsing_binary import EclipsingBinaryFit
from .stellar_manifold import nearest_track_point

# 1 AU in solar radii (IAU 2015 nominal solar radius, 6.957e8 m; 1 AU =
# 1.495978707e11 m) -- 1.495978707e11 / 6.957e8.
AU_IN_SOLAR_RADII = 215.032

# (spectral_type, teff_kelvin, mass_solar, radius_solar), same B9V-M7V rows
# and ordering (hot to cool) as `stellar_manifold.ZAMS_TRACK`. Real values,
# see module docstring for provenance.
MASS_RADIUS_ZAMS_TRACK: tuple[tuple[str, float, float, float], ...] = (
    ("B9V", 10700.0, 2.75, 2.49),
    ("A1V", 9300.0, 2.05, 2.136),
    ("A3V", 8600.0, 1.86, 1.861),
    ("A5V", 8100.0, 1.88, 1.785),
    ("A7V", 7760.0, 1.77, 1.750),
    ("A9V", 7400.0, 1.75, 1.747),
    ("F1V", 7020.0, 1.50, 1.679),
    ("F3V", 6750.0, 1.44, 1.578),
    ("F5V", 6550.0, 1.33, 1.473),
    ("F7V", 6280.0, 1.21, 1.324),
    ("F9V", 6050.0, 1.13, 1.167),
    ("G1V", 5860.0, 1.03, 1.060),
    ("G3V", 5720.0, 0.99, 1.002),
    ("G5V", 5660.0, 0.98, 0.977),
    ("G7V", 5550.0, 0.95, 0.927),
    ("G9V", 5380.0, 0.90, 0.853),
    ("K1V", 5170.0, 0.86, 0.797),
    ("K3V", 4830.0, 0.78, 0.755),
    ("K5V", 4440.0, 0.70, 0.701),
    ("K7V", 4100.0, 0.64, 0.630),
    ("K9V", 3930.0, 0.59, 0.608),
    ("M1V", 3660.0, 0.50, 0.501),
    ("M3V", 3430.0, 0.37, 0.361),
    ("M5V", 3060.0, 0.162, 0.196),
    ("M7V", 2680.0, 0.090, 0.120),
)

# np.interp needs ascending x; the table above is ordered hot-to-cool
# (descending Teff), so build ascending (cool-to-hot) arrays here.
_TRACK_TEFF_ASC = np.array([row[1] for row in reversed(MASS_RADIUS_ZAMS_TRACK)], dtype=np.float64)
_TRACK_MASS_ASC = np.array([row[2] for row in reversed(MASS_RADIUS_ZAMS_TRACK)], dtype=np.float64)
_TRACK_RADIUS_ASC = np.array([row[3] for row in reversed(MASS_RADIUS_ZAMS_TRACK)], dtype=np.float64)


class EclipsingBinaryDimensionsError(ValueError):
    """An absolute-dimension computation could not be completed."""


def mass_and_radius_at_teff(teff_k: float) -> tuple[float, float]:
    """ZAMS mass/radius (solar units) at a given effective temperature.

    A temperature outside the track's own [2680, 10700] K range is clamped
    to the nearest end rather than extrapolated, the same convention
    `stellar_manifold.nearest_track_point` uses for its own colour range.
    """
    teff_k = float(teff_k)
    if not math.isfinite(teff_k) or teff_k <= 0:
        raise EclipsingBinaryDimensionsError("teff_k must be a positive finite number")
    clamped = float(np.clip(teff_k, _TRACK_TEFF_ASC[0], _TRACK_TEFF_ASC[-1]))
    mass = float(np.interp(clamped, _TRACK_TEFF_ASC, _TRACK_MASS_ASC))
    radius = float(np.interp(clamped, _TRACK_TEFF_ASC, _TRACK_RADIUS_ASC))
    return mass, radius


def anchor_physical_radius(*, radius_gspphot: float | None = None,
                           bp_rp: float | None = None, abs_g_mag: float | None = None) -> float:
    """Body 1's physical radius (solar radii), from whichever real source
    is available.

    Prefers Gaia's own `astrophysical_parameters.radius_gspphot` estimate
    when supplied and physically valid; falls back to the ZAMS radius at
    the CMD-derived Teff (`stellar_manifold.nearest_track_point`) when
    `bp_rp`/`abs_g_mag` are supplied instead. Raises rather than fabricating
    an anchor when neither real source is available.
    """
    if radius_gspphot is not None:
        try:
            radius = float(radius_gspphot)
        except (TypeError, ValueError):
            radius = float("nan")
        if math.isfinite(radius) and radius > 0:
            return radius
    if bp_rp is not None and abs_g_mag is not None:
        track_point = nearest_track_point(bp_rp, abs_g_mag)
        _, radius = mass_and_radius_at_teff(track_point["teff_k"])
        return radius
    raise EclipsingBinaryDimensionsError(
        "no radius anchor available: supply a valid radius_gspphot, or both bp_rp and abs_g_mag")


@dataclass(frozen=True)
class AbsoluteDimensions:
    r1_solar: float
    r2_solar: float
    a_au: float
    total_mass_solar: float
    m1_solar: float
    m2_solar: float
    teff1_k: float
    teff2_k: float


def absolute_dimensions(fit: EclipsingBinaryFit, anchor_r1_solar: float, teff1_k: float) -> AbsoluteDimensions:
    """Physical radii/masses from a light-curve fit plus one external
    physical anchor (`anchor_r1_solar`, from `anchor_physical_radius`) and
    body 1's absolute effective temperature (`teff1_k`, from the same real
    source the radius anchor came from).

    Kepler's third law gives the TOTAL system mass once the physical
    semi-major axis is known; splitting it between the two bodies uses
    `mass_and_radius_at_teff` at each body's own temperature as a real
    empirical mass-Teff relation, not an assumed equal split -- see the
    module docstring for why this substitutes for real RV.
    """
    if anchor_r1_solar <= 0 or not math.isfinite(anchor_r1_solar):
        raise EclipsingBinaryDimensionsError("anchor_r1_solar must be a positive finite number")
    if teff1_k <= 0 or not math.isfinite(teff1_k):
        raise EclipsingBinaryDimensionsError("teff1_k must be a positive finite number")

    r1_solar = float(anchor_r1_solar)
    r2_solar = r1_solar * (fit.r2_a / fit.r1_a)
    a_solar = r1_solar / fit.r1_a
    a_au = a_solar / AU_IN_SOLAR_RADII
    period_years = fit.period_days / 365.25
    # Kepler's third law in solar units: M_total[Msun] = a[AU]^3 / P[yr]^2.
    total_mass_solar = (a_au ** 3) / (period_years ** 2)

    teff2_k = teff1_k * fit.teff_ratio
    mass1_model, _ = mass_and_radius_at_teff(teff1_k)
    mass2_model, _ = mass_and_radius_at_teff(teff2_k)
    if mass1_model <= 0 or mass2_model <= 0:
        raise EclipsingBinaryDimensionsError("mass-Teff relation returned a non-positive mass")
    mass_fraction_1 = mass1_model / (mass1_model + mass2_model)
    m1_solar = total_mass_solar * mass_fraction_1
    m2_solar = total_mass_solar - m1_solar

    return AbsoluteDimensions(r1_solar=r1_solar, r2_solar=r2_solar, a_au=a_au,
                              total_mass_solar=total_mass_solar, m1_solar=m1_solar,
                              m2_solar=m2_solar, teff1_k=teff1_k, teff2_k=teff2_k)


# ---------------------------------------------------------------------------
# EB mass/radius catalog cross-check (VizieR J/ApJ/709/535, Brown 2010).
# ---------------------------------------------------------------------------

EB_MASS_RADIUS_CATALOG = "J/ApJ/709/535"


def query_component_catalog(ra_deg: float, dec_deg: float, radius_arcsec: float = 30.0) -> list[dict]:
    """Real, published component-level mass/radius/Teff rows near a
    position, from VizieR `J/ApJ/709/535` (Brown 2010).

    Each returned dict is ONE eclipsing-binary component (e.g. `name`
    `"V760 Sco B"`); a system with a catalogued mass/radius solution
    typically returns one row per component within a few arcsec of each
    other, but pairing them into a single system is left to the caller.
    Diagnostic catalogue metadata only, per the module docstring.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    if radius_arcsec <= 0:
        raise EclipsingBinaryDimensionsError("radius_arcsec must be positive")
    client = Vizier(columns=["Name", "Mass", "e_Mass", "Rad", "e_Rad", "Teff", "e_Teff"],
                    row_limit=20)
    tables = client.query_region(
        SkyCoord(ra_deg, dec_deg, unit="deg", frame="icrs"),
        radius=radius_arcsec * u.arcsec, catalog=EB_MASS_RADIUS_CATALOG,
    )
    if not tables:
        return []

    def _optional_float(row, column: str) -> float | None:
        value = row[column]
        if value is None or np.ma.is_masked(value):
            return None
        as_float = float(value)
        return as_float if np.isfinite(as_float) else None

    matches: list[dict] = []
    for table in tables:
        for row in table:
            matches.append({
                "name": str(row["Name"]),
                "mass_solar": _optional_float(row, "Mass"),
                "mass_solar_err": _optional_float(row, "e_Mass"),
                "radius_solar": _optional_float(row, "Rad"),
                "radius_solar_err": _optional_float(row, "e_Rad"),
                "teff_k": _optional_float(row, "Teff"),
            })
    return matches


def mass_radius_residuals(dims: AbsoluteDimensions, catalog_component1: dict,
                          catalog_component2: dict) -> dict:
    """Fractional (fitted - catalogued) / catalogued differences for both
    components' mass and radius -- diagnostic only, per the module
    docstring; never a correction applied to `dims`."""

    def _fractional(fitted: float, published: float | None) -> float | None:
        if published is None or published == 0:
            return None
        return (fitted - published) / published

    return {
        "mass1_fractional_diff": _fractional(dims.m1_solar, catalog_component1.get("mass_solar")),
        "radius1_fractional_diff": _fractional(dims.r1_solar, catalog_component1.get("radius_solar")),
        "mass2_fractional_diff": _fractional(dims.m2_solar, catalog_component2.get("mass_solar")),
        "radius2_fractional_diff": _fractional(dims.r2_solar, catalog_component2.get("radius_solar")),
    }


__all__ = [
    "EclipsingBinaryDimensionsError", "MASS_RADIUS_ZAMS_TRACK", "AU_IN_SOLAR_RADII",
    "mass_and_radius_at_teff", "anchor_physical_radius", "AbsoluteDimensions", "absolute_dimensions",
    "EB_MASS_RADIUS_CATALOG", "query_component_catalog", "mass_radius_residuals",
]
