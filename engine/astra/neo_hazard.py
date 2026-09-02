"""Near-Earth-object hazard quantities built on top of `moving_objects.py`'s
existing orbit machinery (roadmap: astrophysics & extraterrestrial-study
feature pass).

`moving_objects.py` already provides Gauss preliminary orbit determination,
classical-element conversion, and two-body Kepler propagation, all
documented there as pure two-body (no planetary perturbations, no
light-time/aberration/parallax correction). This module adds the hazard
quantities the MPC/NASA CNEOS convention derives from an orbit -- MOID,
the Tisserand parameter, absolute magnitude, and close-approach distance
-- WITHOUT re-deriving any orbital mechanics `moving_objects.py` already
has; every function here takes the same `elements` dict shape that
`moving_objects.state_vector_to_elements`/`two_body_propagate` already
produce and consume.

Tisserand parameter w.r.t. Jupiter: the standard three-body dynamical
invariant (e.g. Murray & Dermott, *Solar System Dynamics*, 1999, Sec. 3.5)

    T_J = a_J/a + 2*sqrt((a/a_J)*(1-e^2))*cos(i)

with `a_J = 5.2038 au` (Jupiter's semi-major axis). `T_J < 3` is the
standard Jupiter-family-comet/Damocloid dynamical (not compositional)
classification; `T_J > 3` asteroidal (Kresak 1980 / Levison & Duncan 1994's
adopted convention).

Absolute magnitude: the IAU H-G two-parameter phase-function system
(Bowell et al. 1989, in *Asteroids II*, Univ. Arizona Press). The two
basis functions

    Phi1(alpha) = exp(-3.33 * tan(alpha/2)^0.63)
    Phi2(alpha) = exp(-1.87 * tan(alpha/2)^1.22)

with coefficients A1=3.33, B1=0.63, A2=1.87, B2=1.22 -- confirmed this
session against a secondary source quoting Bowell et al. (1989) directly
(the primary conference-proceedings chapter is not available as a
fetchable URL); H follows from
`H = V - 5*log10(r*Delta) + 2.5*log10((1-G)*Phi1(alpha) + G*Phi2(alpha))`
with the MPC-standard default slope parameter `G = 0.15`.

Potentially Hazardous Asteroid (PHA) classification uses the MPC/IAU
definition: `MOID <= 0.05 au AND H <= 22.0`.

Diameter from absolute magnitude uses the standard photometric relation
`D_km = (1329/sqrt(p_V)) * 10^(-H/5)` (Fowler & Chillemi 1992, IRAS
minor-planet survey; the `1329` constant is derived from the Sun's
apparent magnitude and 1 AU, and is the number quoted throughout the NEO
literature, e.g. Harris & Harris 1997). Since diameter depends on an
ASSUMED geometric albedo (`p_V`), never a measured one here, this module
always returns a range over `p_V in (0.05, 0.25)` alongside a default
point value (`p_V = 0.14`, the commonly cited NEO-population mean
albedo), not a single confident number.

[GAP] MOID here (`moid`) is a NUMERICAL result from sampling both orbits'
osculating ellipses and refining with `scipy.optimize.minimize` -- NOT
Gronchi's (2005) algebraic method, and it uses the OSCULATING elements at
one epoch with no secular precession of node/argument of perihelion, so
it is a snapshot, not a long-term impact-risk MOID. Close approaches
(`close_approach`) are likewise pure two-body heliocentric propagation of
both bodies via `moving_objects.two_body_propagate` and
`moving_objects.earth_heliocentric_position_au` -- no planetary
perturbations, no Yarkovsky effect, no Earth-Moon barycenter split, and
critically NO COVARIANCE PROPAGATION. This module therefore never reports
an impact probability: a close-approach distance without a propagated
uncertainty region is not an impact-hazard assessment (see JPL Sentry for
that). `light_time_correct` applies first-order light-time iteration and
first-order annual aberration only -- no relativistic corrections.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from . import moving_objects as mo

SCHEMA_VERSION = 1
JUPITER_SEMI_MAJOR_AXIS_AU = 5.2038
C_AU_PER_DAY = 173.1446326846693  # IAU 2012 exact defined speed of light, au/day
DEFAULT_SLOPE_G = 0.15
DEFAULT_ALBEDO = 0.14
ALBEDO_RANGE = (0.05, 0.25)
PHA_MOID_THRESHOLD_AU = 0.05
PHA_H_THRESHOLD = 22.0
_LUNAR_DISTANCE_AU = 0.00256955529


class NeoHazardError(ValueError):
    """Raised when orbital elements or hazard inputs are inadmissible."""


@dataclass(frozen=True)
class HazardAssessment:
    moid_au: float | None
    tisserand_jupiter: float | None
    dynamical_class: str
    absolute_magnitude: float | None
    diameter_km: float | None
    diameter_km_range: tuple[float, float] | None
    is_pha: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_bound_elements(elements: dict[str, Any]) -> tuple[float, float, float]:
    a = float(elements["semi_major_axis_au"])
    e = float(elements["eccentricity"])
    i_deg = float(elements["inclination_deg"])
    if not math.isfinite(a) or a <= 0:
        raise NeoHazardError("elements must describe a bound (finite, positive-a) orbit")
    if not (0.0 <= e < 1.0):
        raise NeoHazardError("eccentricity must be in [0, 1) for a bound orbit")
    return a, e, i_deg


def tisserand_parameter(elements: dict[str, Any], *,
                        planet_semi_major_axis_au: float = JUPITER_SEMI_MAJOR_AXIS_AU) -> float:
    """Tisserand parameter with respect to Jupiter (or another perturber)."""
    a, e, i_deg = _require_bound_elements(elements)
    i_rad = math.radians(i_deg)
    a_p = float(planet_semi_major_axis_au)
    t_j = a_p / a + 2.0 * math.sqrt((a / a_p) * (1.0 - e ** 2)) * math.cos(i_rad)
    return float(t_j)


def dynamical_class(tisserand: float) -> str:
    return "asteroidal" if tisserand > 3.0 else "comet-like"


def _orbit_positions(elements: dict[str, Any], true_anomalies_rad: np.ndarray, *,
                     mu: float = mo.MU_SUN_AU3_PER_DAY2) -> np.ndarray:
    """Heliocentric positions (n, 3) at a grid of true anomalies on one orbit."""
    positions = np.empty((true_anomalies_rad.shape[0], 3), dtype=np.float64)
    base = dict(elements)
    for idx, nu_rad in enumerate(true_anomalies_rad):
        # Reuse elements_to_state_vector by expressing the sample point via
        # its own mean anomaly is unnecessary here -- the perifocal-to-ECI
        # rotation only needs true anomaly, so build the position directly
        # with the same convention `moving_objects.elements_to_state_vector`
        # uses, avoiding a duplicate Kepler solve for a fixed nu.
        a, e = base["semi_major_axis_au"], base["eccentricity"]
        p = a * (1.0 - e ** 2)
        r = p / (1.0 + e * math.cos(nu_rad))
        r_perifocal = np.array([r * math.cos(nu_rad), r * math.sin(nu_rad), 0.0])
        i = math.radians(base["inclination_deg"])
        raan = math.radians(base["raan_deg"])
        argp = math.radians(base["argument_of_perihelion_deg"])

        def rotation_z(angle: float) -> np.ndarray:
            c, s = math.cos(angle), math.sin(angle)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        def rotation_x(angle: float) -> np.ndarray:
            c, s = math.cos(angle), math.sin(angle)
            return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

        rotation = rotation_z(raan) @ rotation_x(i) @ rotation_z(argp)
        positions[idx] = rotation @ r_perifocal
    return positions


def moid(elements_a: dict[str, Any], elements_b: dict[str, Any], *,
         n_coarse: int = 720, refine: bool = True) -> dict[str, Any]:
    """Numerical minimum orbit intersection distance between two orbits.

    Coarse grid search over both orbits' true anomalies, refined by
    `scipy.optimize.minimize` (Nelder-Mead) from the coarse minimum and its
    four neighbouring grid cells. See module `[GAP]`: this is a snapshot at
    the given osculating elements, not Gronchi's algebraic method.
    """
    if n_coarse < 8:
        raise NeoHazardError("n_coarse must be at least 8")
    _require_bound_elements(elements_a)
    _require_bound_elements(elements_b)

    grid = np.linspace(0.0, 2.0 * math.pi, int(n_coarse), endpoint=False)
    positions_a = _orbit_positions(elements_a, grid)
    positions_b = _orbit_positions(elements_b, grid)
    # (n_coarse, n_coarse) distance matrix -- trivial for n_coarse ~ 1000.
    diff = positions_a[:, None, :] - positions_b[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    idx_a, idx_b = np.unravel_index(int(np.argmin(distances)), distances.shape)
    coarse_min_au = float(distances[idx_a, idx_b])
    grid_spacing_au = float(2.0 * math.pi / n_coarse)

    if not refine:
        return {"schema_version": SCHEMA_VERSION, "moid_au": round(coarse_min_au, 8),
               "method": "grid", "moid_grid_spacing_rad": grid_spacing_au,
               "n_coarse": int(n_coarse)}

    def distance_at(angles: np.ndarray) -> float:
        pa = _orbit_positions(elements_a, np.array([angles[0]]))[0]
        pb = _orbit_positions(elements_b, np.array([angles[1]]))[0]
        return float(np.linalg.norm(pa - pb))

    best = coarse_min_au
    best_angles = (float(grid[idx_a]), float(grid[idx_b]))
    step = 2.0 * math.pi / n_coarse
    for da in (-step, 0.0, step):
        for db in (-step, 0.0, step):
            seed = np.array([grid[idx_a] + da, grid[idx_b] + db])
            result = minimize(distance_at, seed, method="Nelder-Mead",
                              options={"xatol": 1e-10, "fatol": 1e-12})
            if result.fun < best:
                best = float(result.fun)
                best_angles = (float(result.x[0] % (2 * math.pi)), float(result.x[1] % (2 * math.pi)))

    return {"schema_version": SCHEMA_VERSION, "moid_au": round(best, 8),
           "method": "grid+neldermead", "moid_grid_spacing_rad": grid_spacing_au,
           "n_coarse": int(n_coarse), "true_anomalies_rad": best_angles}


def phase_function(alpha_deg: float, *, slope_g: float = DEFAULT_SLOPE_G) -> float:
    """The `(1-G)*Phi1 + G*Phi2` composite phase function, Bowell et al. (1989)."""
    if not (0.0 <= alpha_deg < 180.0):
        raise NeoHazardError("phase angle must be in [0, 180) degrees")
    half_tan = math.tan(math.radians(alpha_deg) / 2.0)
    phi1 = math.exp(-3.33 * half_tan ** 0.63)
    phi2 = math.exp(-1.87 * half_tan ** 1.22)
    return (1.0 - slope_g) * phi1 + slope_g * phi2


def absolute_magnitude(apparent_v: float, heliocentric_au: float, geocentric_au: float,
                       phase_angle_deg: float, *, slope_g: float = DEFAULT_SLOPE_G) -> float:
    """IAU H-G absolute magnitude from an observed apparent magnitude."""
    if heliocentric_au <= 0 or geocentric_au <= 0:
        raise NeoHazardError("heliocentric_au and geocentric_au must be positive")
    composite = phase_function(phase_angle_deg, slope_g=slope_g)
    if composite <= 0:
        raise NeoHazardError("phase function is non-positive at this phase angle/slope")
    h = (apparent_v - 5.0 * math.log10(heliocentric_au * geocentric_au)
        + 2.5 * math.log10(composite))
    return float(h)


def diameter_km(h: float, *, albedo: float = DEFAULT_ALBEDO) -> float:
    if albedo <= 0 or albedo > 1.0:
        raise NeoHazardError("albedo must be in (0, 1]")
    return float((1329.0 / math.sqrt(albedo)) * 10.0 ** (-h / 5.0))


def diameter_km_range(h: float, *, albedo_range: tuple[float, float] = ALBEDO_RANGE) -> tuple[float, float]:
    lo_albedo, hi_albedo = albedo_range
    # Larger albedo -> smaller diameter for the same H, so the diameter
    # range is (diameter at the HIGH albedo, diameter at the LOW albedo).
    return diameter_km(h, albedo=hi_albedo), diameter_km(h, albedo=lo_albedo)


def light_time_correct(target_position_au: np.ndarray, target_velocity_au_per_day: np.ndarray,
                       observer_position_au: np.ndarray, *, tolerance_days: float = 1e-9,
                       max_iterations: int = 20) -> dict[str, Any]:
    """Iterative light-time correction: converges `t_emit` such that the
    light from the target's position at `t_emit` reaches the observer at
    `t_obs`. First-order (linear in velocity) target motion is assumed over
    the light-time interval -- adequate for the sub-hour light times typical
    of near-Earth distances, not for a science-grade astrometric reduction.
    """
    target_position_au = np.asarray(target_position_au, dtype=np.float64)
    target_velocity_au_per_day = np.asarray(target_velocity_au_per_day, dtype=np.float64)
    observer_position_au = np.asarray(observer_position_au, dtype=np.float64)
    delta_t_days = 0.0
    for _ in range(max_iterations):
        position_at_emission = target_position_au - target_velocity_au_per_day * delta_t_days
        separation_au = float(np.linalg.norm(position_at_emission - observer_position_au))
        new_delta_t = separation_au / C_AU_PER_DAY
        if abs(new_delta_t - delta_t_days) < tolerance_days:
            delta_t_days = new_delta_t
            break
        delta_t_days = new_delta_t
    else:
        raise NeoHazardError("light-time iteration did not converge")

    corrected_position = target_position_au - target_velocity_au_per_day * delta_t_days
    direction = corrected_position - observer_position_au
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise NeoHazardError("observer and target coincide; direction undefined")
    return {"light_time_days": round(delta_t_days, 10),
           "corrected_direction": (direction / norm).tolist(),
           "corrected_position_au": corrected_position.tolist()}


def close_approach(elements: dict[str, Any], *, start_mjd: float, end_mjd: float,
                   step_days: float = 1.0) -> dict[str, Any]:
    """Minimum geocentric distance of a two-body-propagated orbit over a
    window, refined by parabolic interpolation around the coarse minimum.

    See module `[GAP]`: unperturbed two-body only, no impact probability.
    """
    if end_mjd <= start_mjd:
        raise NeoHazardError("end_mjd must be after start_mjd")
    if step_days <= 0:
        raise NeoHazardError("step_days must be positive")
    _require_bound_elements(elements)

    mjds = np.arange(start_mjd, end_mjd + step_days, step_days)
    if mjds.shape[0] < 3:
        raise NeoHazardError("window is too short for at least three propagation steps")
    distances = np.empty(mjds.shape[0], dtype=np.float64)
    for idx, mjd_value in enumerate(mjds):
        propagated = mo.two_body_propagate(elements, float(mjd_value))
        r_vec, _ = mo.elements_to_state_vector(propagated)
        earth_vec = mo.earth_heliocentric_position_au(float(mjd_value))
        distances[idx] = float(np.linalg.norm(r_vec - earth_vec))

    min_idx = int(np.argmin(distances))
    if 0 < min_idx < mjds.shape[0] - 1:
        y0, y1, y2 = distances[min_idx - 1], distances[min_idx], distances[min_idx + 1]
        denom = y0 - 2.0 * y1 + y2
        offset = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-15 else 0.0
        offset = max(-1.0, min(1.0, offset))
        refined_mjd = float(mjds[min_idx] + offset * step_days)
        refined_distance_au = float(y1 - 0.25 * (y0 - y2) * offset)
    else:
        refined_mjd = float(mjds[min_idx])
        refined_distance_au = float(distances[min_idx])

    return {"schema_version": SCHEMA_VERSION,
           "close_approach_mjd": round(refined_mjd, 6),
           "distance_au": round(refined_distance_au, 8),
           "distance_lunar_distances": round(refined_distance_au / _LUNAR_DISTANCE_AU, 3),
           "window_start_mjd": float(start_mjd), "window_end_mjd": float(end_mjd),
           "step_days": float(step_days)}


def classify_hazard(*, moid_au: float | None, tisserand: float | None,
                    h: float | None) -> HazardAssessment:
    """Bundle MOID/Tisserand/H into one `HazardAssessment`, applying the
    MPC/IAU PHA criterion `MOID <= 0.05 au AND H <= 22.0`."""
    diameter = diameter_km(h) if h is not None else None
    d_range = diameter_km_range(h) if h is not None else None
    is_pha = bool(moid_au is not None and h is not None
                 and moid_au <= PHA_MOID_THRESHOLD_AU and h <= PHA_H_THRESHOLD)
    dyn_class = dynamical_class(tisserand) if tisserand is not None else "unknown"
    return HazardAssessment(
        moid_au=round(moid_au, 6) if moid_au is not None else None,
        tisserand_jupiter=round(tisserand, 6) if tisserand is not None else None,
        dynamical_class=dyn_class,
        absolute_magnitude=round(h, 3) if h is not None else None,
        diameter_km=round(diameter, 3) if diameter is not None else None,
        diameter_km_range=(round(d_range[0], 3), round(d_range[1], 3)) if d_range else None,
        is_pha=is_pha,
    )


def assess(elements: dict[str, Any], *, earth_elements: dict[str, Any] | None = None,
          apparent_v: float | None = None, heliocentric_au: float | None = None,
          geocentric_au: float | None = None, phase_angle_deg: float | None = None,
          slope_g: float = DEFAULT_SLOPE_G) -> dict[str, Any]:
    """End-to-end hazard assessment for one object's orbital elements."""
    tisserand = tisserand_parameter(elements)
    moid_result = None
    if earth_elements is not None:
        moid_result = moid(elements, earth_elements)
    h = None
    if None not in (apparent_v, heliocentric_au, geocentric_au, phase_angle_deg):
        h = absolute_magnitude(apparent_v, heliocentric_au, geocentric_au, phase_angle_deg,
                               slope_g=slope_g)
    hazard = classify_hazard(moid_au=moid_result["moid_au"] if moid_result else None,
                             tisserand=tisserand, h=h)
    result = hazard.to_dict()
    result["schema_version"] = SCHEMA_VERSION
    result["moid_detail"] = moid_result
    return result


__all__ = [
    "NeoHazardError", "HazardAssessment", "JUPITER_SEMI_MAJOR_AXIS_AU",
    "PHA_MOID_THRESHOLD_AU", "PHA_H_THRESHOLD",
    "tisserand_parameter", "dynamical_class", "moid", "phase_function",
    "absolute_magnitude", "diameter_km", "diameter_km_range",
    "light_time_correct", "close_approach", "classify_hazard", "assess",
]
