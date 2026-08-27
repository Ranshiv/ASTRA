"""Voigt/Gaussian spectral line-profile forward model (roadmap item 25).

`spectroscopy_calibration.py` (item 24) already detects a candidate line's
approximate WAVELENGTH via continuum-subtracted S/N peak-finding, and can
match it to a rest-frame identity via `independent_redshift_from_lines`, but
deliberately stops there -- its own docstring states that fitting the
line's actual SHAPE is item 25's job, not something its bounded peak-finder
attempts. This module is that fit target: a physical forward model for one
line's profile, decomposed with real priors rather than a generic Gaussian
convenience fit.

A Voigt profile is a Gaussian (width `sigma`, from Doppler/thermal/
instrumental broadening) convolved with a Lorentzian (half-width `gamma`,
from natural/pressure broadening) -- the standard physical line shape.
`gamma=0` recovers a pure Gaussian exactly, so ONE model here covers both
cases roadmap item 25 names ("Voigt/Gaussian") without a separate code path.
The convolution has an exact closed form via the Faddeeva function
(`scipy.special.wofz`, already available -- `scipy` is a core dependency,
no new package), so this is evaluated directly rather than by numerical
convolution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


class LineProfileError(ValueError):
    """A line-profile model could not be evaluated."""


@dataclass(frozen=True)
class LineProfileParams:
    """One spectral line's physical shape parameters.

    `amplitude` carries the line's sign: positive for emission, negative
    for absorption. `center`/`sigma`/`gamma` are in the same wavelength
    units as the spectrum being fitted (Angstrom for every connector in
    this codebase).
    """

    center: float
    sigma: float
    gamma: float
    amplitude: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.center, self.sigma, self.gamma, self.amplitude]).all():
            raise LineProfileError("line-profile parameters must be finite")
        if self.sigma <= 0:
            raise LineProfileError("sigma must be positive")
        if self.gamma < 0:
            raise LineProfileError("gamma must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        return np.array([self.center, self.sigma, self.gamma, self.amplitude],
                        dtype=np.float64)

    @classmethod
    def from_array(cls, values) -> "LineProfileParams":
        center, sigma, gamma, amplitude = (float(v) for v in values)
        return cls(center=center, sigma=sigma, gamma=gamma, amplitude=amplitude)


def voigt_profile(x, sigma: float, gamma: float) -> np.ndarray:
    """Unit-area Voigt profile at offsets `x` from the line center.

    `Re[wofz((x + i*gamma) / (sigma*sqrt(2)))] / (sigma*sqrt(2*pi))` is the
    standard closed form (e.g. Olivero & Longbothum 1977); `gamma=0` reduces
    it exactly to a Gaussian, since `wofz` on the real axis is
    `exp(-x^2)(1 + erf(ix))`'s real part.
    """
    from scipy.special import wofz

    if sigma <= 0:
        raise LineProfileError("sigma must be positive")
    if gamma < 0:
        raise LineProfileError("gamma must be non-negative")
    offsets = np.asarray(x, dtype=np.float64)
    z = (offsets + 1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def model_flux(wavelength, continuum, params: LineProfileParams) -> np.ndarray:
    """Continuum plus one Voigt/Gaussian line, evaluated on `wavelength`.

    `continuum` may be a scalar or an array matching `wavelength` -- this
    function does not re-derive a continuum estimate (see
    `spectroscopy_calibration._continuum`), it only adds a line on top of
    whichever one the caller already established.
    """
    wave = np.asarray(wavelength, dtype=np.float64)
    base = np.broadcast_to(np.asarray(continuum, dtype=np.float64), wave.shape)
    return base + params.amplitude * voigt_profile(
        wave - params.center, params.sigma, params.gamma)
