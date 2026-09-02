"""Habitable-zone position and Earth Similarity Index for confirmed planets
(roadmap: astrophysics & extraterrestrial-study feature pass).

Habitable zone: Kopparapu et al. (2013, ApJ 765, 131) with the erratum
(2013, ApJ 770, 82) that corrects Table 3's coefficients by up to ~3% --
this module transcribes the ERRATUM table, fetched and read directly
this session (both PDFs), not the original publication's superseded
values. The effective stellar flux at a habitable-zone boundary is

    Seff = Seff_sun + a*Tstar + b*Tstar^2 + c*Tstar^3 + d*Tstar^4

with `Tstar = Teff - 5780` (K), valid only for `2600 K <= Teff <= 7200 K`
(the erratum's own stated domain). The corresponding orbital distance is
`d_au = sqrt((L/Lsun) / Seff)` (erratum Equation (3)).

Two HZ definitions the paper itself proposes, both computed here:
  - conservative: [moist_greenhouse, maximum_greenhouse] -- the paper's own
    recommendation for RV/Kepler-style surveys "so that eta_earth is not
    overestimated" (Section 6).
  - optimistic: [recent_venus, early_mars] -- empirical limits from Venus's
    and Mars's own geological history.
`runaway_greenhouse` is also exposed as a third, even-more-conservative
inner-edge alternative (the paper notes it is indistinguishable from
moist_greenhouse for Teff <~ 5000 K) but is not used to build either named
HZ above, matching the paper's own usage in Section 6.

Earth Similarity Index: Schulze-Makuch et al. (2011, Astrobiology 11,
1041; doi:10.1089/ast.2010.0592). The primary paper is paywalled and was
not itself fetched this session; the formula and Earth reference values
below were cross-checked against two independent secondary sources
(Wikipedia's Earth Similarity Index article and NASA Astrobiology's
"Quick Facts: Earth Similarity Index" page, both fetched this session)
that agree on the same four weight exponents, which is treated as
sufficient corroboration but is weaker than reading the primary table
directly -- flagged here rather than silently presented as first-hand.
The index for one property is

    ESI_x = (1 - |x - x0| / (x + x0)) ^ (w_x / n)

with Earth reference values x0 and weights w_x:
  interior (n=2): radius x0=1.0 R_earth w=0.57; density x0=5.51 g/cm^3 w=1.07
  surface  (n=2): escape velocity x0=11.19 km/s w=0.70; surf. temp x0=288 K w=5.58
Sub-indices are geometric means of their two terms; the global ESI is the
geometric mean of the two sub-indices, equivalently the 4-term product
with n=4 -- both forms are used below and are algebraically identical.

[GAP] The HZ is a 1D, cloud-free, radiative-convective result for an
Earth-mass planet with an H2O/CO2/N2 atmosphere (Kasting-class model);
it is not a 3D climate result and the paper's own abstract states real
boundaries "may extend further in both directions" once clouds are
included. Eccentricity is reported on `PlanetRecord` but is NOT
integrated into a time-averaged flux here (the paper's Equation (4),
`<Seff'> = Seff / sqrt(1-e^2)`, is a documented, not-yet-implemented
extension -- left out because it changes the returned quantity's meaning
and needs its own test, not because it is hard).

ESI is a geometric similarity metric to Earth, not a probability of
life, and the paper's own reference implementation does not claim
otherwise -- this module never folds either output into
`evidence.WEIGHTS` or `scoring.combine()`, the same "diagnostic
evidence only" restraint `exoplanet_archive.py`'s own docstring already
states for its `compare_to_published` cross-check.

Equilibrium temperature (`equilibrium_temperature`) is the standard
zero-albedo-adjusted blackbody relation, used here ONLY as a stand-in
for Schulze-Makuch's "surface temperature" input, because ASTRA has no
greenhouse model and no way to measure a real surface temperature. This
substitution is real and is surfaced explicitly: the returned field is
named `esi_surface_from_teq`, never `esi_surface`, and a warning is
always attached when it is used. It systematically underestimates ESI
for any planet with a greenhouse atmosphere -- Earth itself, scored via
T_eq rather than its true 288 K surface temperature, does not reach 1.0.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

# Kopparapu et al. (2013, ApJ 765, 131), coefficients from the ERRATUM
# (2013, ApJ 770, 82, Table 3) -- fetched and read directly this session.
# Keys: (Seff_sun, a, b, c, d) in the quartic above.
HZ_COEFFICIENTS: dict[str, tuple[float, float, float, float, float]] = {
    "recent_venus": (1.7763, 1.4335e-4, 3.3954e-9, -7.6364e-12, -1.1950e-15),
    "runaway_greenhouse": (1.0385, 1.2456e-4, 1.4612e-8, -7.6345e-12, -1.7511e-15),
    "moist_greenhouse": (1.0146, 8.1884e-5, 1.9394e-9, -4.3618e-12, -6.8260e-16),
    "maximum_greenhouse": (0.3507, 5.9578e-5, 1.6707e-9, -3.0058e-12, -5.1925e-16),
    "early_mars": (0.3207, 5.4471e-5, 1.5275e-9, -2.1709e-12, -3.8282e-16),
}
CONSERVATIVE_INNER = "moist_greenhouse"
CONSERVATIVE_OUTER = "maximum_greenhouse"
OPTIMISTIC_INNER = "recent_venus"
OPTIMISTIC_OUTER = "early_mars"
TEFF_MIN_K = 2600.0
TEFF_MAX_K = 7200.0
TEFF_SUN_K = 5780.0

# Schulze-Makuch et al. (2011) Earth reference values and weight exponents,
# cross-checked against two independent secondary sources this session
# (see module docstring).
ESI_EARTH_RADIUS_REARTH = 1.0
ESI_EARTH_DENSITY_GCM3 = 5.51
ESI_EARTH_ESCAPE_VELOCITY_KMS = 11.19
ESI_EARTH_SURFACE_TEMP_K = 288.0
ESI_WEIGHT_RADIUS = 0.57
ESI_WEIGHT_DENSITY = 1.07
ESI_WEIGHT_ESCAPE_VELOCITY = 0.70
ESI_WEIGHT_SURFACE_TEMP = 5.58
DEFAULT_BOND_ALBEDO = 0.3
SCHEMA_VERSION = 1
_G_CGS = 6.67430e-8  # CODATA 2018 gravitational constant, cm^3 g^-1 s^-2
_M_EARTH_G = 5.9722e27
_R_EARTH_CM = 6.371e8


class HabitabilityError(ValueError):
    """Raised when habitability inputs are physically inadmissible."""


@dataclass(frozen=True)
class StellarParameters:
    teff_k: float
    radius_rsun: float
    luminosity_lsun: float | None = None

    def __post_init__(self) -> None:
        if self.teff_k <= 0:
            raise HabitabilityError("teff_k must be positive")
        if self.radius_rsun <= 0:
            raise HabitabilityError("radius_rsun must be positive")
        if self.luminosity_lsun is not None and self.luminosity_lsun <= 0:
            raise HabitabilityError("luminosity_lsun must be positive when given")

    def effective_luminosity_lsun(self) -> float:
        """L/Lsun from Teff and R when not supplied directly (L ~ R^2 T^4)."""
        if self.luminosity_lsun is not None:
            return self.luminosity_lsun
        return (self.radius_rsun ** 2) * (self.teff_k / TEFF_SUN_K) ** 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanetParameters:
    radius_rearth: float | None = None
    mass_mearth: float | None = None
    semi_major_axis_au: float | None = None
    eccentricity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _esi_term(value: float, reference: float, weight: float) -> float:
    similarity = 1.0 - abs(value - reference) / (value + reference)
    similarity = max(0.0, similarity)
    return similarity ** weight


def effective_flux(teff_k: float, boundary: str) -> float:
    """Kopparapu et al. (2013) erratum Seff at one named HZ boundary."""
    if boundary not in HZ_COEFFICIENTS:
        raise HabitabilityError(f"unknown HZ boundary: {boundary!r}")
    seff_sun, a, b, c, d = HZ_COEFFICIENTS[boundary]
    t_star = teff_k - TEFF_SUN_K
    return seff_sun + a * t_star + b * t_star ** 2 + c * t_star ** 3 + d * t_star ** 4


def habitable_zone(star: StellarParameters) -> dict[str, Any]:
    """All five named HZ boundary distances (AU) for one star."""
    extrapolated = not (TEFF_MIN_K <= star.teff_k <= TEFF_MAX_K)
    luminosity = star.effective_luminosity_lsun()
    distances: dict[str, float] = {}
    for name in HZ_COEFFICIENTS:
        seff = effective_flux(star.teff_k, name)
        distances[f"{name}_au"] = round(float(math.sqrt(luminosity / seff)), 6)
    return {
        "schema_version": SCHEMA_VERSION,
        **distances,
        "conservative_inner_au": distances[f"{CONSERVATIVE_INNER}_au"],
        "conservative_outer_au": distances[f"{CONSERVATIVE_OUTER}_au"],
        "optimistic_inner_au": distances[f"{OPTIMISTIC_INNER}_au"],
        "optimistic_outer_au": distances[f"{OPTIMISTIC_OUTER}_au"],
        "extrapolated": bool(extrapolated),
    }


def equilibrium_temperature(teff_k: float, stellar_radius_rsun: float,
                            semimajor_au: float, *,
                            bond_albedo: float = DEFAULT_BOND_ALBEDO) -> float:
    """Zero-dimensional equilibrium temperature; see module `[GAP]` re: use
    as an ESI surface-temperature stand-in."""
    if semimajor_au <= 0:
        raise HabitabilityError("semimajor_au must be positive")
    if not (0.0 <= bond_albedo < 1.0):
        raise HabitabilityError("bond_albedo must be in [0, 1)")
    rsun_au = 0.00465047  # 1 solar radius in AU
    r_star_au = stellar_radius_rsun * rsun_au
    return teff_k * math.sqrt(r_star_au / (2.0 * semimajor_au)) * (1.0 - bond_albedo) ** 0.25


def earth_similarity_index(planet: PlanetParameters, star: StellarParameters, *,
                           bond_albedo: float = DEFAULT_BOND_ALBEDO) -> dict[str, Any]:
    """ESI split into interior/surface sub-indices; see module docstring
    for the `esi_surface_from_teq` substitution and its warning."""
    warnings: list[str] = []
    esi_interior = None
    density_gcm3 = None
    escape_velocity_kms = None

    if planet.radius_rearth is not None and planet.radius_rearth <= 0:
        raise HabitabilityError("radius_rearth must be positive when given")
    if planet.mass_mearth is not None and planet.mass_mearth <= 0:
        raise HabitabilityError("mass_mearth must be positive when given")

    if planet.radius_rearth is not None and planet.mass_mearth is not None:
        radius_cm = planet.radius_rearth * _R_EARTH_CM
        mass_g = planet.mass_mearth * _M_EARTH_G
        volume_cm3 = (4.0 / 3.0) * math.pi * radius_cm ** 3
        density_gcm3 = mass_g / volume_cm3
        escape_velocity_kms = math.sqrt(2.0 * _G_CGS * mass_g / radius_cm) / 1.0e5
        radius_term = _esi_term(planet.radius_rearth, ESI_EARTH_RADIUS_REARTH, ESI_WEIGHT_RADIUS)
        density_term = _esi_term(density_gcm3, ESI_EARTH_DENSITY_GCM3, ESI_WEIGHT_DENSITY)
        esi_interior = float((radius_term * density_term) ** (1.0 / 2.0))
    else:
        warnings.append("planet mass and/or radius unavailable; interior ESI not computed")

    esi_surface_from_teq = None
    t_eq_k = None
    if (planet.semi_major_axis_au is not None and planet.semi_major_axis_au > 0
            and escape_velocity_kms is not None):
        t_eq_k = equilibrium_temperature(star.teff_k, star.radius_rsun,
                                         planet.semi_major_axis_au, bond_albedo=bond_albedo)
        vesc_term = _esi_term(escape_velocity_kms, ESI_EARTH_ESCAPE_VELOCITY_KMS,
                              ESI_WEIGHT_ESCAPE_VELOCITY)
        temp_term = _esi_term(t_eq_k, ESI_EARTH_SURFACE_TEMP_K, ESI_WEIGHT_SURFACE_TEMP)
        esi_surface_from_teq = float((vesc_term * temp_term) ** (1.0 / 2.0))
        warnings.append("esi_surface_from_teq uses equilibrium temperature, not a measured "
                        "surface temperature -- no greenhouse model is applied (see [GAP])")
    else:
        warnings.append("semi-major axis and/or escape velocity unavailable; "
                        "esi_surface_from_teq not computed")

    esi_global = None
    if esi_interior is not None and esi_surface_from_teq is not None:
        esi_global = float(math.sqrt(esi_interior * esi_surface_from_teq))

    return {
        "schema_version": SCHEMA_VERSION,
        "esi_interior": round(esi_interior, 6) if esi_interior is not None else None,
        "esi_surface_from_teq": round(esi_surface_from_teq, 6) if esi_surface_from_teq is not None else None,
        "esi_global": round(esi_global, 6) if esi_global is not None else None,
        "density_gcm3": round(density_gcm3, 6) if density_gcm3 is not None else None,
        "escape_velocity_kms": round(escape_velocity_kms, 6) if escape_velocity_kms is not None else None,
        "equilibrium_temp_k": round(t_eq_k, 6) if t_eq_k is not None else None,
        "warnings": warnings,
    }


def score(star: StellarParameters, planet: PlanetParameters, *,
          bond_albedo: float = DEFAULT_BOND_ALBEDO) -> dict[str, Any]:
    """Combined HZ position + ESI report for one star/planet pair."""
    zone = habitable_zone(star)
    esi = earth_similarity_index(planet, star, bond_albedo=bond_albedo)
    warnings = list(esi["warnings"])

    hz_position = None
    in_conservative_hz = None
    in_optimistic_hz = None
    if planet.semi_major_axis_au is not None and planet.semi_major_axis_au > 0:
        a = planet.semi_major_axis_au
        inner, outer = zone["conservative_inner_au"], zone["conservative_outer_au"]
        # 0 = at the inner (hot) edge, 1 = at the outer (cold) edge; values
        # outside [0, 1] are meaningful (too hot / too cold) and returned as-is.
        hz_position = round(float((a - inner) / (outer - inner)), 6) if outer != inner else None
        in_conservative_hz = bool(inner <= a <= outer)
        in_optimistic_hz = bool(zone["optimistic_inner_au"] <= a <= zone["optimistic_outer_au"])
    else:
        warnings.append("semi-major axis unavailable; HZ membership not evaluated")

    if zone["extrapolated"]:
        warnings.append(f"Teff {star.teff_k} K is outside the Kopparapu et al. (2013) "
                        f"validity range [{TEFF_MIN_K}, {TEFF_MAX_K}] K")

    quality = "usable" if (esi["esi_global"] is not None and hz_position is not None
                           and not zone["extrapolated"]) else "insufficient"

    return {
        "schema_version": SCHEMA_VERSION,
        "habitable_zone": zone,
        "hz_position": hz_position,
        "in_conservative_hz": in_conservative_hz,
        "in_optimistic_hz": in_optimistic_hz,
        "esi_interior": esi["esi_interior"],
        "esi_surface_from_teq": esi["esi_surface_from_teq"],
        "esi_global": esi["esi_global"],
        "equilibrium_temp_k": esi["equilibrium_temp_k"],
        "warnings": warnings,
        "quality": quality,
    }


def score_archive_planet(planet_name: str, *, offline: bool = False, root=None) -> dict[str, Any]:
    """Score a named confirmed planet via `exoplanet_archive.query_confirmed_planets`."""
    from . import exoplanet_archive

    records = exoplanet_archive.query_confirmed_planets(planet_name=planet_name, root=root,
                                                         offline=offline)
    if not records:
        raise HabitabilityError(f"no confirmed-planet record found for {planet_name!r}")
    record = records[0]
    if record.st_teff_k is None or record.st_radius_rsun is None:
        raise HabitabilityError(f"{planet_name!r} is missing stellar Teff/radius; cannot score")
    star = StellarParameters(teff_k=record.st_teff_k, radius_rsun=record.st_radius_rsun,
                             luminosity_lsun=record.st_luminosity_lsun)
    planet = PlanetParameters(radius_rearth=record.radius_earth, mass_mearth=record.mass_earth,
                              semi_major_axis_au=record.semimajor_au,
                              eccentricity=record.eccentricity)
    result = score(star, planet)
    result["planet_name"] = record.name
    result["host_name"] = record.host_name
    return result


def rank_planets(records: list, *, limit: int = 50) -> list[dict[str, Any]]:
    """Score a batch of `exoplanet_archive.PlanetRecord`s, ranked by `esi_global`
    descending (planets with no computable ESI sort last)."""
    scored: list[dict[str, Any]] = []
    for record in records:
        if record.st_teff_k is None or record.st_radius_rsun is None:
            continue
        star = StellarParameters(teff_k=record.st_teff_k, radius_rsun=record.st_radius_rsun,
                                 luminosity_lsun=record.st_luminosity_lsun)
        planet = PlanetParameters(radius_rearth=record.radius_earth, mass_mearth=record.mass_earth,
                                  semi_major_axis_au=record.semimajor_au,
                                  eccentricity=record.eccentricity)
        result = score(star, planet)
        result["planet_name"] = record.name
        result["host_name"] = record.host_name
        scored.append(result)
    scored.sort(key=lambda r: (r["esi_global"] is None, -(r["esi_global"] or 0.0)))
    return scored[: int(limit)]


__all__ = [
    "HabitabilityError", "StellarParameters", "PlanetParameters",
    "HZ_COEFFICIENTS", "TEFF_MIN_K", "TEFF_MAX_K",
    "effective_flux", "habitable_zone", "equilibrium_temperature",
    "earth_similarity_index", "score", "score_archive_planet", "rank_planets",
]
