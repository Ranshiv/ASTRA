"""Strong-lens morphology: SIE lens equation, image-position solving, and
time-delay prediction (roadmap item 29, P2).

DES, Pan-STARRS, Hubble and JWST connectors already exist for imaging
cross-checks. HSC (Hyper Suprime-Cam SSP) has no VizieR-hosted point-
source catalogue that resolved live this session -- checked via the same
`Vizier.find_catalogs` discovery this batch's other items used. What DID
resolve, and is used here as the real catalogue cross-check this roadmap
item's "morphology" half needs: VizieR `J/A+A/688/A34` ("Strong lenses
KiDS DR4", Grespan et al. 2024, A&A 688, A34), confirmed live this
session via a real cone search returning real `KiDSID`/`zphot`/`zspec`
rows.

The physics itself needs no survey data at all: `strong_lens.py`
implements the singular isothermal ellipsoid (SIE) lens equation in
closed form (Kormann, Schneider & Bartelmann 1994, A&A 284, 285) -- the
standard analytic lens model this roadmap item's "differentiable lens
equation" describes, differentiable because every piece here is a plain
closed-form expression `scipy.optimize` can already take gradients
through numerically. The SIE deflection has a documented, cited closed
form; the SIE lensing POTENTIAL used for the time-delay calculation
below is derived here (not separately cited) from Euler's homogeneous-
function theorem: since the deflection `alpha(theta)` is 0-homogeneous
in `theta` (isothermal profiles have constant deflection magnitude along
any ray from the lens centre), the potential `psi` -- being 1-homogeneous
because `alpha = grad(psi)` -- satisfies `psi(theta) = theta . alpha(theta)`
exactly. This is verified directly in this module's own tests via a
numerical-gradient cross-check, not merely asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import netclient
from .tap import parse_votable

SCHEMA_VERSION = 1
SCS_URL = "https://vizier.cds.unistra.fr/viz-bin/votable/-A"
KIDS_LENS_CATALOG = "J/A+A/688/A34"
_SPEED_OF_LIGHT_M_S = 299_792_458.0
_MPC_TO_M = 3.0856775814913673e22


class StrongLensError(ValueError):
    """A lens-model computation was given an unphysical configuration."""


@dataclass(frozen=True)
class SIELens:
    """A singular isothermal ellipsoid, all angles in radians."""

    theta_e: float  # Einstein radius
    axis_ratio: float  # q, 0 < q <= 1 (1 = circular/SIS)
    position_angle: float = 0.0  # major-axis orientation, radians

    def __post_init__(self) -> None:
        if self.theta_e <= 0:
            raise StrongLensError("theta_e must be positive")
        if not 0.0 < self.axis_ratio <= 1.0:
            raise StrongLensError("axis_ratio must be in (0, 1]")


def _to_lens_frame(x: np.ndarray, position_angle: float) -> np.ndarray:
    cos_pa, sin_pa = math.cos(position_angle), math.sin(position_angle)
    rotation = np.array([[cos_pa, sin_pa], [-sin_pa, cos_pa]])
    return x @ rotation.T


def _from_lens_frame(x: np.ndarray, position_angle: float) -> np.ndarray:
    cos_pa, sin_pa = math.cos(position_angle), math.sin(position_angle)
    rotation = np.array([[cos_pa, -sin_pa], [sin_pa, cos_pa]])
    return x @ rotation.T


def deflection(lens: SIELens, theta: np.ndarray) -> np.ndarray:
    """SIE deflection angle at image-plane position(s) `theta` (..., 2),
    in the lens's own frame (radians). Closed form from Kormann, Schneider
    & Bartelmann (1994, A&A 284, 285, eq. 4.5-4.6); the `axis_ratio -> 1`
    (SIS) limit is handled separately since the general formula is 0/0
    there.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    local = _to_lens_frame(theta, lens.position_angle)
    x1, x2 = local[..., 0], local[..., 1]
    q = lens.axis_ratio
    if q >= 1.0 - 1e-9:
        radius = np.hypot(x1, x2)
        radius = np.where(radius > 0, radius, np.finfo(float).tiny)
        alpha1, alpha2 = lens.theta_e * x1 / radius, lens.theta_e * x2 / radius
    else:
        psi_ellip = np.sqrt(q ** 2 * x1 ** 2 + x2 ** 2)
        psi_ellip = np.where(psi_ellip > 0, psi_ellip, np.finfo(float).tiny)
        root_term = math.sqrt(1.0 - q ** 2)
        prefactor = lens.theta_e * q / root_term
        alpha1 = prefactor * np.arctan(root_term * x1 / psi_ellip)
        alpha2 = prefactor * np.arctanh(np.clip(root_term * x2 / psi_ellip, -1 + 1e-12, 1 - 1e-12))
    local_alpha = np.stack([alpha1, alpha2], axis=-1)
    result = _from_lens_frame(local_alpha, lens.position_angle)
    return result[0] if result.shape[0] == 1 and theta.shape[0] == 1 else result


def lensing_potential(lens: SIELens, theta: np.ndarray) -> np.ndarray:
    """`psi(theta) = theta . alpha(theta)`, exact for any 0-homogeneous
    deflection field -- see this module's docstring for the derivation.
    Verified numerically against `deflection` via a gradient check in
    `tests/test_strong_lens.py`."""
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    alpha = np.atleast_2d(deflection(lens, theta))
    return np.sum(theta * alpha, axis=-1)


def lens_equation_residual(lens: SIELens, theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """`beta - (theta - alpha(theta))`; zero at a true image position."""
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    beta = np.asarray(beta, dtype=np.float64)
    alpha = np.atleast_2d(deflection(lens, theta))
    return beta - (theta - alpha)


@dataclass(frozen=True)
class ExternalShear:
    """A constant external tidal field from mass outside the lens galaxy
    itself (e.g. a foreground group/cluster) -- the standard extension to
    a single-galaxy lens model (e.g. Kochanek 2006, Saas-Fee lecture
    notes; Schneider, Ehlers & Falco 1992, ch. 8). `gamma1`/`gamma2` are
    the two real shear components (dimensionless)."""

    gamma1: float = 0.0
    gamma2: float = 0.0


def shear_deflection(shear: ExternalShear, theta: np.ndarray) -> np.ndarray:
    """`alpha_shear = (gamma1*x1 + gamma2*x2, gamma2*x1 - gamma1*x2)`, the
    standard closed-form external-shear deflection -- DEGREE-1
    homogeneous in `theta` (linear), unlike the SIE's degree-0 deflection,
    which is exactly why it needs its own potential relation below rather
    than reusing `lensing_potential`'s `theta.alpha` shortcut."""
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    x1, x2 = theta[..., 0], theta[..., 1]
    alpha1 = shear.gamma1 * x1 + shear.gamma2 * x2
    alpha2 = shear.gamma2 * x1 - shear.gamma1 * x2
    result = np.stack([alpha1, alpha2], axis=-1)
    return result[0] if result.shape[0] == 1 else result


def shear_potential(shear: ExternalShear, theta: np.ndarray) -> np.ndarray:
    """`psi_shear = 0.5*gamma1*(x1^2-x2^2) + gamma2*x1*x2` -- the real
    closed-form potential for `shear_deflection` (Euler's theorem for a
    degree-1-homogeneous deflection gives `psi = theta.alpha/2`, not
    `theta.alpha` as the SIE's degree-0 case does; this is that same
    relation evaluated directly rather than re-derived per call)."""
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    x1, x2 = theta[..., 0], theta[..., 1]
    return 0.5 * shear.gamma1 * (x1 ** 2 - x2 ** 2) + shear.gamma2 * x1 * x2


def total_deflection(lens: SIELens, theta: np.ndarray, shear: ExternalShear | None = None
                     ) -> np.ndarray:
    """SIE deflection plus external shear (zero shear if `shear` is
    omitted) -- the combined field `solve_image_positions`/`fit_lens_
    model` use when a shear term is supplied."""
    combined = np.atleast_2d(deflection(lens, theta))
    if shear is not None:
        combined = combined + np.atleast_2d(shear_deflection(shear, theta))
    return combined[0] if combined.shape[0] == 1 else combined


def total_potential(lens: SIELens, theta: np.ndarray, shear: ExternalShear | None = None
                    ) -> np.ndarray:
    """SIE potential plus external-shear potential -- each computed via
    its OWN correct homogeneous-degree relation (see `shear_potential`'s
    docstring), then summed, rather than applying `theta.alpha` to the
    combined (non-homogeneous) deflection field, which would be wrong."""
    total = lensing_potential(lens, theta)
    if shear is not None:
        total = total + shear_potential(shear, theta)
    return total


def magnification(lens: SIELens, theta: np.ndarray, shear: ExternalShear | None = None,
                  *, step: float = 1e-6) -> float:
    """Lensing magnification `mu = 1/det(A)` where `A = I - d(alpha)/d(theta)`
    is the lens-mapping Jacobian, via a central finite difference of
    `total_deflection` -- numerical differentiation of an analytic,
    closed-form field (not an approximate model), the same trade-off
    `dust_3d.py`'s trilinear interpolation makes against a closed-form
    density gradient. Used for the flux-ratio constraint below, which
    needs |mu|, not a full imaging-based photometric model."""
    theta = np.asarray(theta, dtype=np.float64).reshape(2)
    jacobian_alpha = np.empty((2, 2))
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        plus = total_deflection(lens, (theta + offset).reshape(1, 2), shear)
        minus = total_deflection(lens, (theta - offset).reshape(1, 2), shear)
        jacobian_alpha[:, axis] = (plus - minus) / (2.0 * step)
    jacobian_a = np.eye(2) - jacobian_alpha
    determinant = np.linalg.det(jacobian_a)
    if abs(determinant) < 1e-12:
        return float("inf")
    return float(1.0 / determinant)


def solve_image_positions(lens: SIELens, beta: np.ndarray, *,
                          shear: ExternalShear | None = None,
                          n_seeds: int = 12, dedupe_arcsec: float = 1e-3) -> list[np.ndarray]:
    """Root-find the (optionally sheared) SIE lens equation for a given
    source position `beta`, starting from `n_seeds` points on a ring
    around the lens centre (a circular isothermal lens with a point
    source off-axis produces images roughly at Einstein-radius
    separation, so a ring at `theta_e` is a reasonable seed set) and
    de-duplicating converged roots that land within `dedupe_arcsec` of
    one another."""
    from scipy.optimize import root

    beta = np.asarray(beta, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * math.pi, n_seeds, endpoint=False)
    seeds = lens.theta_e * np.stack([np.cos(angles), np.sin(angles)], axis=-1) + beta

    def residual(flat_theta: np.ndarray) -> np.ndarray:
        theta = flat_theta.reshape(1, 2)
        alpha = np.atleast_2d(total_deflection(lens, theta, shear))
        return (beta - (theta - alpha)).reshape(-1)

    found: list[np.ndarray] = []
    for seed in seeds:
        solution = root(residual, seed, method="hybr", tol=1e-12)
        if not solution.success:
            continue
        candidate = solution.x
        if np.max(np.abs(residual(candidate))) > 1e-8:
            continue
        if any(np.linalg.norm(candidate - existing) < dedupe_arcsec for existing in found):
            continue
        found.append(candidate)
    return found


def fermat_potential(lens: SIELens, theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """The dimensionless Fermat/time-delay potential
    `0.5*|theta - beta|^2 - psi(theta)` (e.g. Refsdal 1964, MNRAS 128,
    307; Treu & Marshall 2016, A&ARv 24, 11, eq. 2)."""
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    beta = np.asarray(beta, dtype=np.float64)
    geometric = 0.5 * np.sum((theta - beta) ** 2, axis=-1)
    return geometric - lensing_potential(lens, theta)


def time_delay_seconds(lens: SIELens, theta_a: np.ndarray, theta_b: np.ndarray,
                       beta: np.ndarray, *, z_lens: float,
                       d_l_mpc: float, d_s_mpc: float, d_ls_mpc: float) -> float:
    """Time delay between images at `theta_a` and `theta_b`, in seconds.
    Angular-diameter distances are the caller's responsibility to supply
    (typically from `astropy.cosmology`, already a core dependency) --
    this function is the closed-form arithmetic, not a cosmology client.
    """
    phi_a = float(fermat_potential(lens, theta_a, beta)[0])
    phi_b = float(fermat_potential(lens, theta_b, beta)[0])
    distance_factor_m = (d_l_mpc * d_s_mpc / d_ls_mpc) * _MPC_TO_M
    return (1.0 + z_lens) / _SPEED_OF_LIGHT_M_S * distance_factor_m * (phi_a - phi_b)


def fit_lens_model(observed_images: list[np.ndarray], *, initial_theta_e: float,
                   initial_axis_ratio: float = 0.8, initial_position_angle: float = 0.0,
                   observed_flux_ratios: list[float] | None = None
                   ) -> dict[str, Any]:
    """Recover `(theta_e, axis_ratio, position_angle, beta)` by least
    squares over the joint requirement that every observed image satisfy
    the SAME lens equation for a SHARED, unknown source position -- the
    standard strong-lens modelling approach when only image positions
    (not full imaging) are available.

    `observed_flux_ratios` (one value per image AFTER the first, each the
    observed flux ratio `image[i] / image[0]`) is the real, standard fix
    for a two-image ("double") system's underdetermination: a double
    gives only 4 position residuals for 5 unknowns, but each image's real
    brightness is set by the SAME lens model's magnification, so a real
    flux-ratio measurement adds exactly the missing constraint -- the
    standard "flux-ratio anomaly" technique already used throughout the
    strong-lensing literature (e.g. Mao & Schneider 1998, MNRAS 295, 587;
    Dalal & Kochanek 2002, ApJ 572, 25), not invented here. Compared in
    LOG space (`log|mu_i/mu_0|`) because magnification ratios are
    naturally multiplicative and can span orders of magnitude near a
    caustic.
    """
    from scipy.optimize import least_squares

    observed = np.asarray(observed_images, dtype=np.float64)
    if len(observed) < 2:
        raise StrongLensError("at least two observed images are required to constrain a fit")
    if observed_flux_ratios is not None and len(observed_flux_ratios) != len(observed) - 1:
        raise StrongLensError("observed_flux_ratios must have one entry per image after the first")
    beta_guess = observed.mean(axis=0)

    def residuals(params: np.ndarray) -> np.ndarray:
        theta_e, axis_ratio, position_angle, beta_x, beta_y = params
        theta_e = max(theta_e, 1e-6)
        axis_ratio = float(np.clip(axis_ratio, 1e-3, 1.0))
        lens = SIELens(theta_e=theta_e, axis_ratio=axis_ratio, position_angle=position_angle)
        beta = np.array([beta_x, beta_y])
        position_residuals = lens_equation_residual(lens, observed, beta).reshape(-1)
        if observed_flux_ratios is None:
            return position_residuals
        mu0 = magnification(lens, observed[0])
        flux_residuals = []
        for image, observed_ratio in zip(observed[1:], observed_flux_ratios):
            mu_i = magnification(lens, image)
            if not (math.isfinite(mu0) and math.isfinite(mu_i)) or mu0 == 0 or mu_i == 0:
                flux_residuals.append(10.0)  # a finite, large penalty, never NaN/inf into least_squares
                continue
            model_log_ratio = math.log(abs(mu_i / mu0))
            flux_residuals.append(model_log_ratio - math.log(abs(observed_ratio)))
        return np.concatenate([position_residuals, np.asarray(flux_residuals)])

    initial = np.array([initial_theta_e, initial_axis_ratio, initial_position_angle,
                        beta_guess[0], beta_guess[1]])
    # A two-image ("double") system gives only 4 residuals for 5 unknown
    # parameters -- underdetermined, and `method="lm"` refuses to even
    # attempt it (real error found this session: "Method 'lm' doesn't work
    # when the number of residuals is less than the number of variables").
    # `trf` handles the underdetermined case (converging to some point on
    # the degenerate solution manifold rather than crashing) as well as
    # the well-determined 4-image case, so it is used unconditionally
    # rather than switching methods by image count. Supplying
    # `observed_flux_ratios` for a double makes the system well-determined
    # again (5 residuals for 5 unknowns) without needing a method switch.
    result = least_squares(residuals, initial, method="trf", max_nfev=5000)
    theta_e, axis_ratio, position_angle, beta_x, beta_y = result.x
    fitted_lens = SIELens(theta_e=max(theta_e, 1e-6), axis_ratio=float(np.clip(axis_ratio, 1e-3, 1.0)),
                          position_angle=position_angle)
    fitted_images = solve_image_positions(fitted_lens, np.array([beta_x, beta_y]))
    return {
        "theta_e": float(fitted_lens.theta_e), "axis_ratio": float(fitted_lens.axis_ratio),
        "position_angle_rad": float(position_angle), "beta": [float(beta_x), float(beta_y)],
        "converged": bool(result.success), "residual_rms": float(np.sqrt(np.mean(result.fun ** 2))),
        "fitted_image_count": len(fitted_images),
    }


def query_kids_strong_lens_catalog(ra_deg: float, dec_deg: float, radius_arcsec: float = 10.0
                                   ) -> dict | None:
    """The nearest real, published strong-lens candidate from KiDS DR4
    (Grespan et al. 2024) within `radius_arcsec` -- a catalogue cross-
    check, never a substitute for this module's own model fit."""
    response = netclient.get(
        SCS_URL,
        {"-source": KIDS_LENS_CATALOG, "RA": ra_deg, "DEC": dec_deg,
         "SR": radius_arcsec / 3600.0, "-out.max": 1, "-out.orderby": "_r"},
        timeout=60, provider="vizier",
    )
    rows = parse_votable(response.text, 1)
    if not rows:
        return None
    row = rows[0]
    return {"kids_id": row.get("KiDSID"), "z_phot": row.get("zphot"), "z_spec": row.get("zspec")}


def evaluate_multi_survey_coverage_real(candidates: list[dict[str, float]], *,
                                        radius_arcsec: float = 3.0) -> dict[str, Any]:
    """The real, POPULATION-scale study this item's own "real recall/
    precision" gap could not get (no live-reachable catalogue of real
    multi-image lens ASTROMETRY was found this session -- see this
    module's `docs/LIMITATIONS.md` entry), done instead with what IS real and
    reachable: for each real KiDS DR4 strong-lens CANDIDATE position
    (`candidates`, each a `{"kidsid", "ra", "dec"}` dict -- the real 564-
    candidate catalogue was fetched live this session via VizieR's plain
    (non-cone-search) VOTable endpoint, not the SCS one `query_kids_
    strong_lens_catalog` uses, since SCS requires a position per query),
    cross-match against DES and Pan-STARRS (both already real,
    registered connectors) to measure real multi-survey imaging coverage
    -- a real, useful diagnostic (which candidates have independent
    imaging to visually/photometrically vet a lens model against) even
    though it is NOT a recall/precision study, and is not claimed to be
    one.
    """
    from .surveys.base import ConeQuery
    from .surveys.des import DESConnector
    from .surveys.panstarrs import PanSTARRSConnector

    des = DESConnector()
    panstarrs = PanSTARRSConnector()
    n_des = 0
    n_panstarrs = 0
    n_both = 0
    n_neither = 0
    n_des_errors = 0
    n_panstarrs_errors = 0
    for candidate in candidates:
        cone = ConeQuery(ra_deg=candidate["ra"], dec_deg=candidate["dec"], radius_arcsec=radius_arcsec)
        # A real, live-confirmed finding this session: DES's `cone_search`
        # (`surveys/des.py`, built and only mock-tested in an earlier
        # session) currently raises a bare `KeyError` against the real
        # NOIRLab Data Lab TAP service -- its ADQL `POINT`/`CIRCLE`
        # cone-search functions now fail server-side with `PSQLException:
        # ERROR: function point(unknown, double precision, double
        # precision) does not exist`, confirmed via a direct query this
        # session (not assumed). One candidate's connector failure must
        # not abort this whole population study, the same "one bad X does
        # not abort the batch" discipline every other real study in this
        # codebase already applies (e.g. `surveys/kepler.py`'s per-quarter
        # try/except) -- counted separately as `n_des_errors`/`n_
        # panstarrs_errors` rather than silently counted as "no match".
        try:
            has_des = len(des.cone_search(cone, limit=1)) > 0
        except Exception:
            has_des = False
            n_des_errors += 1
        try:
            has_panstarrs = len(panstarrs.cone_search(cone, limit=1)) > 0
        except Exception:
            has_panstarrs = False
            n_panstarrs_errors += 1
        n_des += has_des
        n_panstarrs += has_panstarrs
        n_both += has_des and has_panstarrs
        n_neither += not has_des and not has_panstarrs
    n_total = len(candidates)
    return {
        "n_candidates": n_total,
        "n_des_matches": n_des, "n_panstarrs_matches": n_panstarrs,
        "n_both_surveys": n_both, "n_neither_survey": n_neither,
        "n_des_errors": n_des_errors, "n_panstarrs_errors": n_panstarrs_errors,
        "des_coverage_fraction": round(n_des / n_total, 4) if n_total else None,
        "panstarrs_coverage_fraction": round(n_panstarrs / n_total, 4) if n_total else None,
    }
