"""Radio variability: interstellar scintillation and synchrotron spectral
models (roadmap item 22).

Scintillation: `pulse_broadening_time_ms` implements the Bhat et al. 2004
(ApJ 605, 759) empirical pulse-broadening relation, verified this session
against the paper's own quoted formula (not reproduced from memory alone):

    log10(tau_d_ms) = -6.46 + 0.154*log10(DM) + 1.07*log10(DM)^2
                      - 3.86*log10(nu_GHz)

with `DM` in pc/cm^3, `nu` in GHz, `tau_d` in ms. `decorrelation_bandwidth_mhz`
follows from the standard Fourier relation between scattering time and
frequency coherence bandwidth, `delta_nu_d = C1 / (2*pi*tau_d)` (Cordes &
Rickett 1998), with `C1=1.16` for a thin, uniform scattering screen -- a
commonly adopted geometric constant, stated as such rather than re-derived.

Deliberately NOT implemented: a diffractive/refractive scintillation
TIMESCALE. That relation needs an effective scattering-screen distance and
transverse velocity, which in turn need a Galactic electron-density/
turbulence model (NE2001 or YMW16) -- no such model exists anywhere in this
codebase (grepped, zero hits), and fabricating a screen-distance constant
to force a timescale number out would produce an unverified, uncited
result. This is a real, stated `[GAP]`, the same "verify live/from a
citable source, or state the gap plainly" discipline `kilonova.py`'s own
Barnes-2016-interpolation-table gap already uses.

Synchrotron: `synchrotron_flux_mjy` is the standard optically-thin power
law with a self-absorption turnover -- textbook synchrotron physics (e.g.
Rybicki & Lightman 1979, ch. 6), not attributed to a single paper:
`S_nu ~ nu^alpha` above a turnover frequency (`alpha` typically -0.5 to
-1.2 for AGN/afterglow synchrotron emission), steepening to the universal
self-absorbed slope `S_nu ~ nu^2.5` below it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Cordes & Rickett 1998's thin-uniform-screen geometric constant relating
# scattering time to decorrelation bandwidth.
SCATTERING_GEOMETRY_C1 = 1.16
SELF_ABSORBED_SLOPE = 2.5


class RadioVariabilityError(ValueError):
    """A radio-variability model could not be evaluated."""


def pulse_broadening_time_ms(dispersion_measure_pc_cm3: float, frequency_ghz) -> np.ndarray:
    """Bhat et al. 2004's empirical scattering-time relation."""
    dm = float(dispersion_measure_pc_cm3)
    if dm <= 0:
        raise RadioVariabilityError("dispersion_measure_pc_cm3 must be positive")
    nu = np.asarray(frequency_ghz, dtype=np.float64)
    if np.any(nu <= 0):
        raise RadioVariabilityError("frequency_ghz must be positive")
    log_dm = np.log10(dm)
    log_tau_ms = -6.46 + 0.154 * log_dm + 1.07 * log_dm ** 2 - 3.86 * np.log10(nu)
    return 10.0 ** log_tau_ms


def decorrelation_bandwidth_mhz(dispersion_measure_pc_cm3: float, frequency_ghz) -> np.ndarray:
    """Scintillation decorrelation bandwidth from the pulse-broadening time
    via the standard scattering-time/bandwidth Fourier relation."""
    tau_d_s = pulse_broadening_time_ms(dispersion_measure_pc_cm3, frequency_ghz) * 1e-3
    delta_nu_hz = SCATTERING_GEOMETRY_C1 / (2.0 * np.pi * tau_d_s)
    return delta_nu_hz / 1e6


@dataclass(frozen=True)
class SynchrotronSpectrumParams:
    """A power-law synchrotron spectrum with a self-absorption turnover.

    `flux_at_turnover_mjy` is `S(nu_turnover_ghz)`; `alpha_thin` is the
    optically-thin spectral index above the turnover (`S_nu ~ nu^alpha`);
    below the turnover the spectrum follows the fixed, universal
    self-absorbed slope `SELF_ABSORBED_SLOPE` -- not a free parameter,
    since that slope is set by synchrotron self-absorption physics, not by
    the source.
    """

    nu_turnover_ghz: float
    flux_at_turnover_mjy: float
    alpha_thin: float

    def __post_init__(self) -> None:
        values = [self.nu_turnover_ghz, self.flux_at_turnover_mjy, self.alpha_thin]
        if not np.isfinite(values).all():
            raise RadioVariabilityError("synchrotron parameters must be finite")
        if self.nu_turnover_ghz <= 0:
            raise RadioVariabilityError("nu_turnover_ghz must be positive")
        if self.flux_at_turnover_mjy <= 0:
            raise RadioVariabilityError("flux_at_turnover_mjy must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


def synchrotron_flux_mjy(frequency_ghz, params: SynchrotronSpectrumParams) -> np.ndarray:
    nu = np.asarray(frequency_ghz, dtype=np.float64)
    if np.any(nu <= 0):
        raise RadioVariabilityError("frequency_ghz must be positive")
    ratio = nu / params.nu_turnover_ghz
    optically_thin = params.flux_at_turnover_mjy * ratio ** params.alpha_thin
    optically_thick = params.flux_at_turnover_mjy * ratio ** SELF_ABSORBED_SLOPE
    return np.where(nu >= params.nu_turnover_ghz, optically_thin, optically_thick)


def fit_spectral_index(frequency_ghz, flux_mjy, flux_err_mjy) -> dict:
    """Weighted least-squares power-law spectral index in log-log space:
    `log(S) = log(S0) + alpha*log(nu)`.

    Needs at least two frequency points (a spectral index is undefined
    from one flux measurement alone). Errors are propagated into log-space
    via the standard small-error approximation `sigma_logS ~= sigma_S/S`.
    Returns the fitted index, its formal standard error from the weighted
    linear-regression covariance, and the fitted normalization -- a point
    estimate; `radio_variability_eval.py` builds a bootstrap/posterior
    uncertainty around this for cases with only 2-3 points, where the
    formal covariance alone is not a reliable uncertainty.
    """
    nu = np.asarray(frequency_ghz, dtype=np.float64)
    flux = np.asarray(flux_mjy, dtype=np.float64)
    flux_err = np.asarray(flux_err_mjy, dtype=np.float64)
    if not (len(nu) == len(flux) == len(flux_err)):
        raise RadioVariabilityError("frequency_ghz, flux_mjy and flux_err_mjy must have equal lengths")
    if len(nu) < 2:
        raise RadioVariabilityError("at least two frequency points are needed to fit a spectral index")
    if np.any(nu <= 0) or np.any(flux <= 0) or np.any(flux_err <= 0):
        raise RadioVariabilityError("frequency, flux and flux_err must all be positive")

    log_nu = np.log(nu)
    log_flux = np.log(flux)
    log_flux_err = flux_err / flux  # small-error approximation

    design = np.column_stack([np.ones_like(log_nu), log_nu])
    weights = 1.0 / log_flux_err ** 2
    weighted_design = design * weights[:, None]
    covariance_matrix = design.T @ weighted_design
    try:
        covariance = np.linalg.inv(covariance_matrix)
    except np.linalg.LinAlgError as exc:
        raise RadioVariabilityError("spectral-index design matrix is singular") from exc
    solution = covariance @ (design.T @ (weights * log_flux))
    log_s0, alpha = float(solution[0]), float(solution[1])

    return {
        "alpha": alpha, "alpha_stderr": float(np.sqrt(covariance[1, 1])),
        "s0_mjy_at_1ghz": float(np.exp(log_s0)), "n_points": int(len(nu)),
    }
