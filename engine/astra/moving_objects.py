"""Moving-object tracklet linking and preliminary orbit determination.

Shaped like `gw.py`/`frb.py`, not a `SurveyConnector`: a moving object is a
time-ordered sequence of positions, not a static point a `cone_search` can
return, so it does not fit that interface any better than a GW skymap or an
FRB burst does. This is optional, explicit candidate-review evidence, never
a dependency of discovery.

Two live-endpoint checks were made before writing this module, the same
discipline `gw.py`/`frb.py` document for GWOSC/GraceDB/CADC:

1. The Minor Planet Center DOES publish a genuine, credential-free (shared,
   publicly documented basic-auth) query API for orbital elements --
   `https://minorplanetcenter.net/web_service/search_orbits` -- returning
   semimajor axis, eccentricity, inclination, node, argument of perihelion
   and related quantities by designation. This is real, queryable data, not
   the one-way ADES/obs80 submission path this codebase's earlier planning
   assumed was MPC's only interface. `mpc_search_orbits` below uses it, for
   comparing a fitted orbit's elements against MPC's published ones
   (orbital-element residuals).
2. MPC's query API does NOT expose raw per-night detections/tracklets --
   only already-linked, already-fitted orbital elements. Real raw tracklet
   detections would have to come from ZTF/ATLAS/Pan-STARRS themselves, and
   none of those is wired up for this today: ATLAS's public forced-
   photometry service (`fallingstar-data.com/forcedphot/`) requires sign-in
   and returns light curves at supplied/ephemeris positions, not raw
   detection streams; ZTF/Pan-STARRS moving-object detections are a
   separate product from the static-source connectors already in
   `surveys/ztf.py`/`surveys/panstarrs.py` and are not ingested anywhere in
   this codebase. `assemble_tracklets`/`orbit_from_tracklet` below therefore
   operate on caller-supplied detection rows (already-known ra/dec/mjd),
   the same "diagnostic operates on data the caller assembled" shape
   `significance.evaluate_selection` already uses -- they are validated
   against synthetic, known-truth orbits here, not yet runnable against a
   live multi-survey detection stream.

The orbit-determination step is the classical (Gauss, 1809) preliminary
method: three angles-only observations plus the observer's heliocentric
position at each time, solved via the standard iteration-free (single-pass,
3rd-order f/g series) formulation -- e.g. Curtis, *Orbital Mechanics for
Engineering Students*, the algorithm this module's tests validate against
synthetic ground truth. This is a preliminary/first-approximation method,
accurate for closely-spaced observations (the tracklet case this module
targets: hours to about a day apart), NOT an iteratively refined orbit;
light-time, aberration and topocentric parallax corrections are also not
applied. `two_body_propagate` is likewise pure two-body Kepler propagation,
with no perturbations (no planetary perturbations, no non-gravitational
forces) -- adequate for a short-arc residual check, not for a long-term
ephemeris. None of this is wired into `evidence.WEIGHTS`/`scoring.combine()`
-- it lands as visible evidence only, matching every other module in this
file's family.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

from . import netclient

MPC_SEARCH_ORBITS_URL = "https://minorplanetcenter.net/web_service/search_orbits"
# Publicly documented shared credentials for the MPC query service (not a
# per-user API key) -- see the module docstring.
MPC_USERNAME = "mpc_ws"
MPC_PASSWORD = "mpc!!ws"

# Gaussian gravitational constant k (AU^1.5 / day, historically defined so
# that G*M_sun = k^2 in AU/day units) -- the standard unit choice for
# solar-system orbit determination, avoiding km/s-scale conditioning issues.
GAUSSIAN_K = 0.01720209895
MU_SUN_AU3_PER_DAY2 = GAUSSIAN_K ** 2


class MovingObjectError(ValueError):
    """A tracklet or orbit-fit request could not be satisfied."""


# ---------------------------------------------------------------------------
# Geometry: RA/Dec <-> unit vectors, and the observer's heliocentric position.
# ---------------------------------------------------------------------------

def radec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    return np.array([math.cos(dec) * math.cos(ra),
                     math.cos(dec) * math.sin(ra),
                     math.sin(dec)], dtype=np.float64)


def unit_vector_to_radec(vector: np.ndarray) -> tuple[float, float]:
    x, y, z = vector
    r = float(np.linalg.norm(vector))
    if r <= 0:
        raise MovingObjectError("cannot derive RA/Dec from a zero vector")
    dec = math.degrees(math.asin(max(-1.0, min(1.0, z / r))))
    ra = math.degrees(math.atan2(y, x)) % 360.0
    return ra, dec


def earth_heliocentric_position_au(mjd: float) -> np.ndarray:
    """Earth's heliocentric position (AU, equatorial J2000-aligned) at one epoch.

    Uses astropy's builtin low-precision ephemeris -- no network fetch, no
    external ephemeris-kernel download -- adequate for the preliminary-orbit
    use case here; a science-grade astrometric reduction would want a JPL
    DE kernel instead.
    """
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
    from astropy.time import Time

    time = Time(mjd, format="mjd", scale="utc")
    with solar_system_ephemeris.set("builtin"):
        earth = get_body_barycentric("earth", time)
        sun = get_body_barycentric("sun", time)
    helio = (earth - sun).xyz.to_value("AU")
    return np.asarray(helio, dtype=np.float64)


@dataclass(frozen=True)
class Observation:
    """One angles-only observation: an object's direction from a known observer."""

    mjd: float
    ra_deg: float
    dec_deg: float
    observer_position_au: np.ndarray  # heliocentric, AU

    @property
    def line_of_sight(self) -> np.ndarray:
        return radec_to_unit_vector(self.ra_deg, self.dec_deg)


def observation_from_detection(detection: dict[str, Any]) -> Observation:
    """Build an `Observation`, computing the observer's heliocentric position
    from `mjd` unless the caller already supplied one (e.g. a cached value,
    or a synthetic test's own ground-truth observer position)."""
    mjd = float(detection["mjd"])
    observer = detection.get("observer_position_au")
    if observer is None:
        observer = earth_heliocentric_position_au(mjd)
    return Observation(mjd=mjd, ra_deg=float(detection["ra_deg"]),
                       dec_deg=float(detection["dec_deg"]),
                       observer_position_au=np.asarray(observer, dtype=np.float64))


# ---------------------------------------------------------------------------
# Gauss's preliminary orbit determination (single-pass, 3rd-order f/g series).
# ---------------------------------------------------------------------------

def gauss_preliminary_orbit(observations: list[Observation], *,
                            mu: float = MU_SUN_AU3_PER_DAY2) -> dict[str, Any]:
    """Fit a heliocentric state vector at the middle epoch from three
    angles-only observations.

    Returns a dict with `r2_au`/`v2_au_per_day` (the fitted state vector) on
    success. Raises `MovingObjectError` when the geometry is degenerate (the
    three lines of sight are ~coplanar with the observer-Sun geometry, so
    `D0` vanishes) or the range polynomial has no usable positive real root
    -- both real failure modes of angles-only IOD from a short, near-linear
    arc, not implementation bugs to paper over.
    """
    if len(observations) != 3:
        raise MovingObjectError("Gauss's method requires exactly three observations")
    obs1, obs2, obs3 = sorted(observations, key=lambda o: o.mjd)
    tau1 = obs1.mjd - obs2.mjd
    tau3 = obs3.mjd - obs2.mjd
    tau = tau3 - tau1
    if tau == 0:
        raise MovingObjectError("observations must have distinct times")

    rho1, rho2, rho3 = obs1.line_of_sight, obs2.line_of_sight, obs3.line_of_sight
    R1, R2, R3 = obs1.observer_position_au, obs2.observer_position_au, obs3.observer_position_au

    p1 = np.cross(rho2, rho3)
    p2 = np.cross(rho1, rho3)
    p3 = np.cross(rho1, rho2)
    D0 = float(np.dot(rho1, p1))
    if abs(D0) < 1e-12:
        raise MovingObjectError("degenerate geometry: observations are coplanar with the observer")

    D = np.array([[np.dot(R1, p1), np.dot(R1, p2), np.dot(R1, p3)],
                  [np.dot(R2, p1), np.dot(R2, p2), np.dot(R2, p3)],
                  [np.dot(R3, p1), np.dot(R3, p2), np.dot(R3, p3)]])

    A = (1.0 / D0) * (-D[0, 1] * (tau3 / tau) + D[1, 1] + D[2, 1] * (tau1 / tau))
    B = (1.0 / (6.0 * D0)) * (D[0, 1] * (tau3 ** 2 - tau ** 2) * (tau3 / tau)
                             + D[2, 1] * (tau ** 2 - tau1 ** 2) * (tau1 / tau))
    E = float(np.dot(R2, rho2))
    R2sq = float(np.dot(R2, R2))

    a_coeff = -(A ** 2 + 2.0 * A * E + R2sq)
    b_coeff = -2.0 * mu * B * (A + E)
    c_coeff = -(mu ** 2) * (B ** 2)
    # x^8 + a*x^6 + 0*x^5 + 0*x^4 + b*x^3 + 0*x^2 + 0*x + c = 0
    coefficients = [1.0, 0.0, a_coeff, 0.0, 0.0, b_coeff, 0.0, 0.0, c_coeff]
    roots = np.roots(coefficients)
    positive_real_roots = sorted(
        float(root.real) for root in roots
        if abs(root.imag) < 1e-6 * max(abs(root.real), 1.0) and root.real > 0
    )
    if not positive_real_roots:
        raise MovingObjectError("range polynomial has no usable positive real root")
    # The largest positive real root is the standard choice: angles-only
    # Gauss IOD is known to admit spurious small-r roots close to the
    # observer, and the physically relevant solar-system solution is the
    # outermost one (Curtis, sec. 5.10).
    r2 = positive_real_roots[-1]

    u = mu / r2 ** 3
    f1 = 1.0 - 0.5 * u * tau1 ** 2
    f3 = 1.0 - 0.5 * u * tau3 ** 2
    g1 = tau1 - (u * tau1 ** 3) / 6.0
    g3 = tau3 - (u * tau3 ** 3) / 6.0
    denominator = f1 * g3 - f3 * g1
    if abs(denominator) < 1e-12:
        raise MovingObjectError("degenerate f/g geometry (observations too close in time)")

    c1 = g3 / denominator
    c3 = -g1 / denominator
    rho1_range = (1.0 / D0) * (-D[0, 0] + D[1, 0] / c1 - D[2, 0] * (c3 / c1))
    rho2_range = A + mu * B / r2 ** 3
    rho3_range = (1.0 / D0) * (-D[0, 2] * (c1 / c3) + D[1, 2] / c3 - D[2, 2])
    if rho2_range <= 0:
        raise MovingObjectError("fitted slant range is non-physical (negative or zero)")

    r1_vec = R1 + rho1_range * rho1
    r2_vec = R2 + rho2_range * rho2
    r3_vec = R3 + rho3_range * rho3
    v2_vec = (-f3 * r1_vec + f1 * r3_vec) / denominator

    return {
        "epoch_mjd": obs2.mjd,
        "r2_au": r2_vec,
        "v2_au_per_day": v2_vec,
        "slant_ranges_au": {"r1": float(rho1_range), "r2": float(rho2_range),
                            "r3": float(rho3_range)},
        "diagnostics": {"D0": D0, "candidate_roots": positive_real_roots},
    }


# ---------------------------------------------------------------------------
# State vector <-> classical orbital elements, and simple two-body propagation.
# ---------------------------------------------------------------------------

def state_vector_to_elements(r_vec: np.ndarray, v_vec: np.ndarray, *,
                             mu: float = MU_SUN_AU3_PER_DAY2,
                             epoch_mjd: float | None = None) -> dict[str, Any]:
    """Classical orbital elements from a heliocentric state vector.

    Standard closed-form conversion (e.g. Curtis Algorithm 4.1 / Vallado);
    returns angles in degrees and the mean anomaly at `epoch_mjd` so the
    result can be propagated forward with `two_body_propagate`.
    """
    r_vec = np.asarray(r_vec, dtype=np.float64)
    v_vec = np.asarray(v_vec, dtype=np.float64)
    r = float(np.linalg.norm(r_vec))
    v = float(np.linalg.norm(v_vec))
    if r <= 0:
        raise MovingObjectError("position vector must be nonzero")

    h_vec = np.cross(r_vec, v_vec)
    h = float(np.linalg.norm(h_vec))
    if h <= 0:
        raise MovingObjectError("degenerate (zero angular momentum) orbit")

    node_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
    node = float(np.linalg.norm(node_vec))

    e_vec = ((v * v - mu / r) * r_vec - float(np.dot(r_vec, v_vec)) * v_vec) / mu
    eccentricity = float(np.linalg.norm(e_vec))

    energy = (v * v) / 2.0 - mu / r
    if abs(eccentricity - 1.0) > 1e-9:
        semi_major_axis = -mu / (2.0 * energy)
    else:
        semi_major_axis = math.inf  # parabolic; not expected from real IOD input

    inclination = math.degrees(math.acos(max(-1.0, min(1.0, h_vec[2] / h))))

    if node > 1e-12:
        raan = math.degrees(math.acos(max(-1.0, min(1.0, node_vec[0] / node))))
        if node_vec[1] < 0:
            raan = 360.0 - raan
    else:
        raan = 0.0  # equatorial orbit: node undefined, convention 0

    if node > 1e-12 and eccentricity > 1e-12:
        cos_argp = float(np.dot(node_vec, e_vec)) / (node * eccentricity)
        argp = math.degrees(math.acos(max(-1.0, min(1.0, cos_argp))))
        if e_vec[2] < 0:
            argp = 360.0 - argp
    else:
        argp = 0.0

    if eccentricity > 1e-12:
        cos_nu = float(np.dot(e_vec, r_vec)) / (eccentricity * r)
        true_anomaly = math.degrees(math.acos(max(-1.0, min(1.0, cos_nu))))
        if float(np.dot(r_vec, v_vec)) < 0:
            true_anomaly = 360.0 - true_anomaly
    else:
        true_anomaly = 0.0

    eccentric_anomaly = 2.0 * math.atan2(
        math.sqrt(max(0.0, 1.0 - eccentricity)) * math.sin(math.radians(true_anomaly) / 2.0),
        math.sqrt(max(0.0, 1.0 + eccentricity)) * math.cos(math.radians(true_anomaly) / 2.0),
    )
    mean_anomaly = math.degrees(eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly)) % 360.0

    return {
        "semi_major_axis_au": float(semi_major_axis),
        "eccentricity": eccentricity,
        "inclination_deg": inclination,
        "raan_deg": raan,
        "argument_of_perihelion_deg": argp,
        "true_anomaly_deg": true_anomaly,
        "mean_anomaly_deg": mean_anomaly,
        "epoch_mjd": epoch_mjd,
    }


def _solve_kepler(mean_anomaly_rad: float, eccentricity: float, *,
                  tolerance: float = 1e-10, max_iter: int = 50) -> float:
    """Eccentric anomaly via Newton-Raphson on Kepler's equation."""
    e_anomaly = mean_anomaly_rad if eccentricity < 0.8 else math.pi
    for _ in range(max_iter):
        delta = (e_anomaly - eccentricity * math.sin(e_anomaly) - mean_anomaly_rad) \
            / (1.0 - eccentricity * math.cos(e_anomaly))
        e_anomaly -= delta
        if abs(delta) < tolerance:
            break
    return e_anomaly


def elements_to_state_vector(elements: dict[str, Any], *,
                             mu: float = MU_SUN_AU3_PER_DAY2) -> tuple[np.ndarray, np.ndarray]:
    """Classical elements -> heliocentric (r, v), the inverse of
    `state_vector_to_elements` (used for propagation and synthetic tests)."""
    a = float(elements["semi_major_axis_au"])
    e = float(elements["eccentricity"])
    i = math.radians(elements["inclination_deg"])
    raan = math.radians(elements["raan_deg"])
    argp = math.radians(elements["argument_of_perihelion_deg"])
    mean_anomaly = math.radians(elements["mean_anomaly_deg"])

    eccentric_anomaly = _solve_kepler(mean_anomaly, e)
    true_anomaly = 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(eccentric_anomaly / 2.0),
        math.sqrt(1.0 - e) * math.cos(eccentric_anomaly / 2.0),
    )
    p = a * (1.0 - e ** 2)
    r = p / (1.0 + e * math.cos(true_anomaly))

    r_perifocal = np.array([r * math.cos(true_anomaly), r * math.sin(true_anomaly), 0.0])
    h = math.sqrt(mu * p)
    v_perifocal = (mu / h) * np.array([-math.sin(true_anomaly), e + math.cos(true_anomaly), 0.0])

    def rotation_z(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def rotation_x(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    rotation = rotation_z(raan) @ rotation_x(i) @ rotation_z(argp)
    return rotation @ r_perifocal, rotation @ v_perifocal


def two_body_propagate(elements: dict[str, Any], target_mjd: float, *,
                       mu: float = MU_SUN_AU3_PER_DAY2) -> dict[str, Any]:
    """Advance a two-body (unperturbed) orbit's mean anomaly to `target_mjd`.

    No planetary perturbations, no non-gravitational forces -- adequate for
    the short-arc residual check this module targets (compare against MPC's
    own perturbed elements over the SAME short window a tracklet spans),
    not for a long-term ephemeris.
    """
    epoch_mjd = elements.get("epoch_mjd")
    if epoch_mjd is None:
        raise MovingObjectError("elements must carry an epoch_mjd to propagate from")
    a = float(elements["semi_major_axis_au"])
    if not math.isfinite(a) or a <= 0:
        raise MovingObjectError("propagation requires a bound (finite, positive-a) orbit")
    mean_motion_deg_per_day = math.degrees(math.sqrt(mu / a ** 3))
    elapsed_days = float(target_mjd) - float(epoch_mjd)
    propagated = dict(elements)
    propagated["mean_anomaly_deg"] = (
        elements["mean_anomaly_deg"] + mean_motion_deg_per_day * elapsed_days
    ) % 360.0
    propagated["epoch_mjd"] = float(target_mjd)
    return propagated


# ---------------------------------------------------------------------------
# Tracklet assembly from caller-supplied detections.
# ---------------------------------------------------------------------------

DEFAULT_MAX_TRACKLET_SPAN_DAYS = 1.0
DEFAULT_RATE_RESIDUAL_ARCSEC = 3.0


def assemble_tracklets(detections: Iterable[dict[str, Any]], *,
                      max_span_days: float = DEFAULT_MAX_TRACKLET_SPAN_DAYS,
                      rate_residual_arcsec: float = DEFAULT_RATE_RESIDUAL_ARCSEC
                      ) -> list[dict[str, Any]]:
    """Group same-night detections into tracklets by rate-of-motion consistency.

    Each input row needs `ra_deg`, `dec_deg`, `mjd`, and a `survey` label.
    Detections are first grouped by observation night (`max_span_days`
    window from the earliest detection in a candidate group), then a
    tracklet is accepted only when a straight-line fit of RA/Dec against
    time has a residual under `rate_residual_arcsec` for every member --
    real asteroid motion over a night is very close to linear on the sky, so
    a large residual means the group is not one physical object crossing
    the field, and it is reported as `rejected`, not silently linked.
    """
    rows = sorted((row for row in detections if isinstance(row, dict)),
                 key=lambda row: float(row["mjd"]))
    if len(rows) < 3:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if current and float(row["mjd"]) - float(current[0]["mjd"]) > max_span_days:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)

    tracklets: list[dict[str, Any]] = []
    for group in groups:
        if len(group) < 3:
            continue
        times = np.array([float(row["mjd"]) for row in group])
        ras = np.array([float(row["ra_deg"]) for row in group])
        decs = np.array([float(row["dec_deg"]) for row in group])
        cos_dec = np.cos(np.radians(np.mean(decs)))
        # Fit in a locally flat tangent-plane approximation (RA scaled by
        # cos(dec)); adequate for one night's worth of motion, far from the
        # celestial poles -- the same convention `catalogs.py`'s
        # within_error_ellipse already documents for CHIME/FRB.
        ra_fit = np.polyfit(times, ras * cos_dec, 1)
        dec_fit = np.polyfit(times, decs, 1)
        ra_residual_arcsec = np.abs(ras * cos_dec - np.polyval(ra_fit, times)) * 3600.0
        dec_residual_arcsec = np.abs(decs - np.polyval(dec_fit, times)) * 3600.0
        max_residual = float(max(ra_residual_arcsec.max(), dec_residual_arcsec.max()))
        accepted = max_residual <= rate_residual_arcsec
        tracklets.append({
            "detections": group,
            "n_detections": len(group),
            "span_days": float(times.max() - times.min()),
            "max_rate_residual_arcsec": max_residual,
            "accepted": accepted,
        })
    return tracklets


def orbit_from_tracklet(tracklet: dict[str, Any], *,
                        mu: float = MU_SUN_AU3_PER_DAY2) -> dict[str, Any]:
    """Fit a preliminary orbit from one accepted tracklet's detections.

    Uses the first, middle, and last detections as the three Gauss-method
    observations -- the widest usable time baseline within the tracklet,
    which conditions the range polynomial better than three closely bunched
    points would.
    """
    detections = tracklet.get("detections", [])
    if len(detections) < 3:
        raise MovingObjectError("a tracklet needs at least three detections to fit an orbit")
    ordered = sorted(detections, key=lambda row: float(row["mjd"]))
    chosen = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    observations = [observation_from_detection(row) for row in chosen]
    fit = gauss_preliminary_orbit(observations, mu=mu)
    elements = state_vector_to_elements(fit["r2_au"], fit["v2_au_per_day"], mu=mu,
                                        epoch_mjd=fit["epoch_mjd"])
    return {"elements": elements, "state_vector": fit, "n_detections_used": 3,
           "n_detections_available": len(detections)}


# ---------------------------------------------------------------------------
# MPC published-orbit lookup, for orbital-element residual comparison.
# ---------------------------------------------------------------------------

def _mpc_auth_header() -> dict[str, str]:
    token = base64.b64encode(f"{MPC_USERNAME}:{MPC_PASSWORD}".encode("ascii")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def mpc_search_orbits(params: dict[str, Any], *, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Query MPC's public orbital-elements web service.

    `params` follows the service's own documented query fields (e.g.
    `{"name": "Ceres"}` or a designation). Returns the parsed JSON rows
    unchanged; a non-2xx response or a malformed body propagates as the
    underlying `requests`/JSON error rather than a silently empty result,
    matching this codebase's "a real failure must not read as a true
    no-match" convention.
    """
    response = netclient.get(MPC_SEARCH_ORBITS_URL, {**params, "json": 1},
                             timeout=timeout, provider="mpc", headers=_mpc_auth_header())
    payload = response.json()
    return payload if isinstance(payload, list) else [payload]


def orbital_element_residuals(fitted: dict[str, Any], published: dict[str, Any]) -> dict[str, Any]:
    """Compare a fitted orbit's elements against MPC's published ones.

    A diagnostic, not a correction: `published` (from `mpc_search_orbits`,
    or any dict using the same field names) is treated as ground truth for
    reporting purposes only.
    """
    fields = ("semi_major_axis_au", "eccentricity", "inclination_deg",
             "raan_deg", "argument_of_perihelion_deg")
    field_map = {
        "semi_major_axis_au": ("a", "semi_major_axis_au"),
        "eccentricity": ("e", "eccentricity"),
        "inclination_deg": ("i", "inclination_deg"),
        "raan_deg": ("node", "raan_deg"),
        "argument_of_perihelion_deg": ("argper", "argument_of_perihelion_deg"),
    }
    residuals: dict[str, float | None] = {}
    for field in fields:
        fitted_value = fitted.get(field)
        published_value = None
        for key in field_map[field]:
            if key in published:
                published_value = published[key]
                break
        if fitted_value is None or published_value is None:
            residuals[field] = None
            continue
        try:
            residuals[field] = float(fitted_value) - float(published_value)
        except (TypeError, ValueError):
            residuals[field] = None
    return {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "residuals": residuals}


__all__ = [
    "MovingObjectError", "Observation", "radec_to_unit_vector", "unit_vector_to_radec",
    "earth_heliocentric_position_au", "observation_from_detection",
    "gauss_preliminary_orbit", "state_vector_to_elements", "elements_to_state_vector",
    "two_body_propagate", "assemble_tracklets", "orbit_from_tracklet",
    "mpc_search_orbits", "orbital_element_residuals",
    "MU_SUN_AU3_PER_DAY2", "MPC_SEARCH_ORBITS_URL",
]
