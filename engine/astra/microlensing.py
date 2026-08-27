"""Physical microlensing forward models (backlog item 15).

This is the first time-domain PHYSICAL forward model in this codebase.
Everything else here fits either a periodogram (`features.py`,
`multiband.py`), a generic ML model (`models.py`), or a one-parameter grid
(`sed.py`'s 192-point blackbody argmin, which reports no uncertainty at
all). A microlensing light curve, by contrast, has an exact closed-form
shape derived from general relativity, so the model can be written down
rather than learned.

Point lens (Paczynski 1986): a single lens crossing the line of sight to a
background source magnifies it by

    A(u) = (u^2 + 2) / (u * sqrt(u^2 + 4)),   u(t) = sqrt(u0^2 + ((t-t0)/tE)^2)

with three nonlinear parameters -- `t0` (time of closest approach), `tE`
(Einstein-radius crossing time, the physically interesting one: it encodes
the lens mass, distance and transverse velocity) and `u0` (impact
parameter in Einstein radii). Pure numpy, no dependencies, safe in the
packaged CPU build.

The observed flux adds two more parameters, `f_source` and `f_blend`
(unresolved light from neighbours inside the seeing disc):

    F(t) = f_source * A(t) + f_blend

Those enter LINEARLY, so they are never searched over -- `solve_linear_flux`
solves them exactly by weighted linear least squares at each nonlinear
step, turning a 5-parameter fit into a 3-parameter one. That is standard
microlensing practice and a large robustness win, since the nonlinear
search then only has to explore a space where every point already has its
best possible flux scaling.

Binary lens adds `s` (separation), `q` (mass ratio), `alpha` (trajectory
angle) and `rho` (source size), and has no closed form -- its magnification
requires contour integration around caustics. `binary_magnification`
delegates to VBMicrolensing rather than reimplementing that: mature codes
spent years hardening those numerics, and a hand-rolled version would be a
worse, unvalidated forward model. Gated in the `research` extra and
lazy-imported, exactly the `multiband_hier._require_celerite2` pattern.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Below this impact parameter the point-lens magnification diverges as 1/u
# and the point-source approximation stops being physical anyway (a real
# source has finite size). Clamping keeps the model finite without
# pretending the point-source formula is valid there.
MIN_IMPACT_PARAMETER = 1e-6


class MicrolensingError(ValueError):
    """A microlensing model or fit could not be evaluated."""


@dataclass(frozen=True)
class PointLensParams:
    """The three nonlinear point-lens parameters.

    `tE` is the physically interesting one: it is the Einstein-radius
    crossing time, degenerate in lens mass / distance / transverse
    velocity, and the quantity every microlensing survey reports.
    """

    t0: float
    tE: float
    u0: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.t0, self.tE, self.u0]).all():
            raise MicrolensingError("point-lens parameters must be finite")
        if self.tE <= 0:
            raise MicrolensingError("tE must be positive")
        if self.u0 < 0:
            raise MicrolensingError("u0 must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        return np.array([self.t0, self.tE, self.u0], dtype=np.float64)

    @classmethod
    def from_array(cls, values) -> "PointLensParams":
        t0, tE, u0 = (float(v) for v in values)
        return cls(t0=t0, tE=tE, u0=u0)


@dataclass(frozen=True)
class BinaryLensParams:
    """Point-lens parameters plus the four binary-lens ones."""

    t0: float
    tE: float
    u0: float
    s: float
    q: float
    alpha: float
    rho: float = 1e-3

    def __post_init__(self) -> None:
        values = [self.t0, self.tE, self.u0, self.s, self.q, self.alpha, self.rho]
        if not np.isfinite(values).all():
            raise MicrolensingError("binary-lens parameters must be finite")
        if self.tE <= 0:
            raise MicrolensingError("tE must be positive")
        if self.s <= 0:
            raise MicrolensingError("s (separation) must be positive")
        if not 0 < self.q <= 1:
            raise MicrolensingError("q (mass ratio) must be in (0, 1]")
        if self.rho <= 0:
            raise MicrolensingError("rho (source radius) must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


def impact_parameter(time: np.ndarray, params: PointLensParams) -> np.ndarray:
    """u(t) = sqrt(u0^2 + ((t - t0)/tE)^2), the lens-source separation in
    Einstein radii."""
    time = np.asarray(time, dtype=np.float64)
    tau = (time - params.t0) / params.tE
    return np.sqrt(params.u0 ** 2 + tau ** 2)


def magnification(time: np.ndarray, params: PointLensParams) -> np.ndarray:
    """Exact Paczynski point-source point-lens magnification.

    A(u) -> 1 as u -> infinity (unlensed baseline) and A(u) ~ 1/u for
    u << 1 (the high-magnification limit); both are checked directly in
    this module's tests rather than assumed.
    """
    u = np.clip(impact_parameter(time, params), MIN_IMPACT_PARAMETER, None)
    return (u ** 2 + 2.0) / (u * np.sqrt(u ** 2 + 4.0))


def model_flux(time: np.ndarray, params: PointLensParams,
               f_source: float, f_blend: float) -> np.ndarray:
    """F(t) = f_source * A(t) + f_blend."""
    return f_source * magnification(time, params) + f_blend


def mag_to_flux(mag: np.ndarray, zeropoint: float = 18.0) -> np.ndarray:
    """Magnitudes to linear flux. OGLE and ZTF both report magnitudes, but
    the microlensing model is linear in FLUX (a blend adds flux, not
    magnitudes), so fitting must happen in flux space."""
    return 10.0 ** (-0.4 * (np.asarray(mag, dtype=np.float64) - zeropoint))


def flux_to_mag(flux: np.ndarray, zeropoint: float = 18.0) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return zeropoint - 2.5 * np.log10(np.where(flux > 0, flux, np.nan))


def mag_err_to_flux_err(mag: np.ndarray, mag_err: np.ndarray,
                        zeropoint: float = 18.0) -> np.ndarray:
    """Standard error propagation: dF/F = 0.4 * ln(10) * dm."""
    flux = mag_to_flux(mag, zeropoint)
    return flux * 0.4 * np.log(10.0) * np.asarray(mag_err, dtype=np.float64)


def solve_linear_flux(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                      params: PointLensParams) -> tuple[float, float]:
    """Exact weighted least-squares solution for (f_source, f_blend).

    Given the nonlinear parameters, the model is linear in these two, so
    they are SOLVED rather than searched -- the standard microlensing
    trick, reducing the nonlinear search from 5 dimensions to 3. Returns
    `(f_source, f_blend)`; a degenerate design matrix (e.g. a curve with no
    magnification variation at all, so `A(t)` is effectively constant)
    falls back to a flat solution rather than raising, since the caller is
    a fitter that must be able to score even a bad parameter draw.
    """
    amplification = magnification(time, params)
    weights = 1.0 / np.asarray(flux_err, dtype=np.float64) ** 2
    design = np.column_stack([amplification, np.ones_like(amplification)])

    normal = design.T @ (design * weights[:, None])
    target = design.T @ (np.asarray(flux, dtype=np.float64) * weights)
    try:
        solution = np.linalg.solve(normal, target)
    except np.linalg.LinAlgError:
        return 0.0, float(np.average(flux, weights=weights))
    if not np.isfinite(solution).all():
        return 0.0, float(np.average(flux, weights=weights))
    return float(solution[0]), float(solution[1])


def _require_vbmicrolensing():
    try:
        import VBMicrolensing
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MicrolensingError(
            "VBMicrolensing is not installed; install the 'research' extra "
            "(pip install .[research]) to fit binary lenses. Point-lens "
            "fitting needs no extra dependency."
        ) from exc
    return VBMicrolensing


def binary_magnification(time: np.ndarray, params: BinaryLensParams) -> np.ndarray:
    """Finite-source binary-lens magnification, via VBMicrolensing.

    Deliberately NOT reimplemented here: binary-lens magnification needs
    contour integration around caustics, whose numerics mature codes spent
    years hardening. A hand-rolled version would be a worse, unvalidated
    forward model, and every downstream metric in `microlensing_eval.py`
    would then be measuring that error rather than the fitter.
    """
    module = _require_vbmicrolensing()
    solver = module.VBMicrolensing()

    time = np.asarray(time, dtype=np.float64)
    tau = (time - params.t0) / params.tE
    cos_alpha, sin_alpha = np.cos(params.alpha), np.sin(params.alpha)
    # Standard trajectory parameterisation: the source moves in a straight
    # line at angle `alpha` to the binary axis, offset by `u0`.
    x = tau * cos_alpha - params.u0 * sin_alpha
    y = tau * sin_alpha + params.u0 * cos_alpha

    result = np.empty_like(time)
    for index in range(len(time)):
        result[index] = solver.BinaryMag2(
            params.s, params.q, float(x[index]), float(y[index]), params.rho)
    return result
