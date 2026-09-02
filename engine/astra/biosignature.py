"""Transmission-spectrum forward model for molecular BAND-DETECTION
SIGNIFICANCE (roadmap: astrophysics & extraterrestrial-study feature
pass). See `biosignature_fit.py` for the optimiser/sampler.

Isothermal, hydrostatic, constant-gravity transmission spectrum
(Lecavelier des Etangs et al. 2008, A&A 481, L83 -- the standard analytic
baseline for an exponential atmosphere probed by transit spectroscopy):

    H = k_B * T_eq / (mu * m_u * g)                    (scale height, m)
    z(lambda) = H * ln(sigma(lambda) / sigma_ref) + z_ref
    depth(lambda) = ((Rp + z(lambda)) / Rstar)^2

`sigma_ref`/`z_ref` are degenerate with each other and with the reference
pressure level -- the classic normalisation degeneracy of transmission
spectroscopy -- so this module exposes a single fitted nuisance parameter,
`reference_radius_rjup` (the planet's apparent radius at the reference
wavelength), rather than pretending to fit a reference pressure it has no
way to constrain.

Molecular absorption is a GAUSSIAN-BAND TEMPLATE, not line-by-line or
correlated-k opacity: `MOLECULAR_BANDS` carries only band CENTRES and
WIDTHS (standard, widely-cited IR/visible molecular band locations --
H2O 1.4/1.9/2.7 um, CH4 1.66/2.3/3.3 um, CO2 1.6/2.0/2.7/4.3/15 um,
CO 2.3/4.6 um, O2 A-band 0.76 um, O3 Chappuis ~0.6 um and the mid-IR
9.6 um Hartley-adjacent band). These are textbook band locations (e.g.
the same CO2 15 um / 667 cm^-1 band `kopparapu`-family climate models
already cite), not transcribed from a single paper's numeric table, so
they carry a lower verification bar than the physics constants elsewhere
in this codebase -- but they are still approximate band CENTRES, not a
real absorption cross-section. Peak cross sections are a REQUIRED CALLER
INPUT (`cross_sections: Mapping[str, float]`), never a module default,
so no invented absolute-opacity number ever enters a result silently.

[GAP] There are no ExoMol/HITEMP line lists anywhere in this repository
and none are downloaded (petitRADTRANS-class radiative transfer was
explicitly ruled out for this build -- no new heavy dependency). Molecular
absorption here is a relative band-strength template conditioned on
whatever cross-section normalisation the caller supplied -- fitted band
amplitudes are NOT calibrated absolute volume mixing ratios. A statement
of the form "X ppm of water was detected" is not supported by this
module; `biosignature_fit.detection_significance` measures whether a
band is statistically present, never an abundance. Also absent: no
multiple-scattering radiative transfer (isothermal/isobaric single-slab
only), no non-isothermal T-P profile, no limb inhomogeneity, no
refraction, and no stellar-contamination (unocculted spot/facula)
correction -- Rackham et al. (2018)'s spot-crossing systematic, likely
the single largest real-world systematic in transmission spectroscopy,
is simply not modelled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = 1

# Boltzmann constant (J/K) and atomic mass unit (kg) -- CODATA 2018.
K_BOLTZMANN = 1.380649e-23
AMU_KG = 1.66053906660e-27
G_NEWTON = 6.67430e-11
R_JUPITER_M = 7.1492e7
R_SUN_M = 6.957e8
M_JUPITER_KG = 1.89813e27

# Band centre/width, in microns -- see module docstring for provenance.
# Peak cross sections are NOT included here; callers must supply them.
MOLECULAR_BANDS: dict[str, tuple[tuple[float, float], ...]] = {
    "H2O": ((0.95, 0.03), (1.15, 0.04), (1.4, 0.08), (1.9, 0.08), (2.7, 0.15)),
    "CH4": ((1.66, 0.03), (2.3, 0.08), (3.3, 0.1)),
    "CO2": ((1.6, 0.03), (2.0, 0.03), (2.7, 0.06), (4.3, 0.1), (15.0, 1.0)),
    "CO": ((2.3, 0.03), (4.6, 0.08)),
    "O2": ((0.76, 0.005),),
    "O3": ((0.6, 0.05), (9.6, 0.8)),
}


class BiosignatureError(ValueError):
    """Raised when transmission-spectrum inputs are physically inadmissible."""


def _finite_arrays(wavelength_um, depth, error) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same validation discipline as `spectral_features._finite_arrays`,
    reimplemented here for a MICRON wavelength axis (transmission spectra
    are conventionally reported in microns, unlike the Angstrom convention
    `spectral_features.py` uses for stellar/AGN spectra)."""
    wave = np.asarray(wavelength_um, dtype=np.float64)
    values = np.asarray(depth, dtype=np.float64)
    errors = np.asarray(error, dtype=np.float64)
    if not (wave.ndim == values.ndim == errors.ndim == 1):
        raise BiosignatureError("spectral columns must be one-dimensional")
    if not (len(wave) == len(values) == len(errors)):
        raise BiosignatureError("spectral columns must have equal lengths")
    mask = np.isfinite(wave) & np.isfinite(values) & np.isfinite(errors) & (errors >= 0)
    wave, values, errors = wave[mask], values[mask], errors[mask]
    order = np.argsort(wave, kind="stable")
    wave, values, errors = wave[order], values[order], errors[order]
    if len(wave) < 5 or np.any(np.diff(wave) <= 0):
        raise BiosignatureError("spectrum needs at least five strictly increasing finite wavelengths")
    return wave, values, errors


@dataclass(frozen=True)
class SystemParameters:
    stellar_radius_rsun: float
    planet_mass_mjup: float

    def __post_init__(self) -> None:
        if self.stellar_radius_rsun <= 0 or self.planet_mass_mjup <= 0:
            raise BiosignatureError("stellar_radius_rsun and planet_mass_mjup must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtmosphereParameters:
    temperature_k: float
    mean_molecular_weight: float
    reference_radius_rjup: float
    log10_cloud_pressure_bar: float | None = None
    abundances: tuple[tuple[str, float], ...] = ()  # (molecule, log10 relative amplitude)

    def __post_init__(self) -> None:
        if self.temperature_k <= 0:
            raise BiosignatureError("temperature_k must be positive")
        if self.mean_molecular_weight <= 0:
            raise BiosignatureError("mean_molecular_weight must be positive")
        if self.reference_radius_rjup <= 0:
            raise BiosignatureError("reference_radius_rjup must be positive")
        for name, _ in self.abundances:
            if name not in MOLECULAR_BANDS:
                raise BiosignatureError(f"unknown molecule: {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "abundances": list(self.abundances)}

    def to_array(self, molecules: tuple[str, ...]) -> np.ndarray:
        """`[temperature_k, reference_radius_rjup, log10_amp_1, ...]` --
        `mean_molecular_weight` and `log10_cloud_pressure_bar` are held
        fixed by the fitter, not searched over (see `biosignature_fit.py`)."""
        lookup = dict(self.abundances)
        return np.array([self.temperature_k, self.reference_radius_rjup]
                        + [lookup.get(name, -30.0) for name in molecules], dtype=np.float64)

    @classmethod
    def from_array(cls, values, *, molecules: tuple[str, ...],
                   mean_molecular_weight: float,
                   log10_cloud_pressure_bar: float | None = None) -> "AtmosphereParameters":
        values = np.asarray(values, dtype=np.float64)
        temperature_k, reference_radius_rjup = float(values[0]), float(values[1])
        abundances = tuple((name, float(v)) for name, v in zip(molecules, values[2:]))
        return cls(temperature_k=temperature_k, mean_molecular_weight=mean_molecular_weight,
                  reference_radius_rjup=reference_radius_rjup,
                  log10_cloud_pressure_bar=log10_cloud_pressure_bar, abundances=abundances)


def surface_gravity_ms2(planet_mass_mjup: float, reference_radius_rjup: float) -> float:
    if planet_mass_mjup <= 0 or reference_radius_rjup <= 0:
        raise BiosignatureError("planet_mass_mjup and reference_radius_rjup must be positive")
    mass_kg = planet_mass_mjup * M_JUPITER_KG
    radius_m = reference_radius_rjup * R_JUPITER_M
    return float(G_NEWTON * mass_kg / radius_m ** 2)


def scale_height_m(atmosphere: AtmosphereParameters, system: SystemParameters) -> float:
    g = surface_gravity_ms2(system.planet_mass_mjup, atmosphere.reference_radius_rjup)
    mu_kg = atmosphere.mean_molecular_weight * AMU_KG
    return float(K_BOLTZMANN * atmosphere.temperature_k / (mu_kg * g))


# Arbitrary but fixed continuum-opacity floor. `z = H*ln(sigma/sigma_ref)`
# is only well-conditioned when `sigma` never approaches zero (a pure
# molecular-band sum vanishes between bands, which would make the log
# ratio numerically explode from floating-point noise, not real signal) --
# the absolute scale of the floor is unobservable (degenerate with
# `reference_radius_rjup`, per the module docstring's z_ref/sigma_ref
# note), only its role as a nonzero baseline matters.
CONTINUUM_OPACITY_FLOOR = 1.0


def _band_optical_depth(wavelength_um: np.ndarray, *, molecules: tuple[str, ...],
                        log10_amplitudes: np.ndarray,
                        cross_sections: Mapping[str, float]) -> np.ndarray:
    """Continuum floor plus a sum of Gaussian band templates, each
    molecule's peak height set by `cross_sections[name] * 10**log10_amplitude`
    -- `cross_sections` is a REQUIRED caller input (see module `[GAP]`)."""
    total = np.full_like(wavelength_um, CONTINUUM_OPACITY_FLOOR)
    for name, log10_amp in zip(molecules, log10_amplitudes):
        peak = cross_sections.get(name)
        if peak is None or peak <= 0:
            continue
        amplitude = peak * 10.0 ** log10_amp
        for center, width in MOLECULAR_BANDS[name]:
            total += amplitude * np.exp(-0.5 * ((wavelength_um - center) / width) ** 2)
    return total


def transit_depth(wavelength_um, atmosphere: AtmosphereParameters, system: SystemParameters, *,
                  cross_sections: Mapping[str, float]) -> np.ndarray:
    """Isothermal transmission-spectrum transit depth at each wavelength."""
    wave = np.asarray(wavelength_um, dtype=np.float64)
    if np.any(wave <= 0):
        raise BiosignatureError("wavelength_um must be positive")
    molecules = tuple(name for name, _ in atmosphere.abundances)
    log10_amplitudes = np.array([amp for _, amp in atmosphere.abundances], dtype=np.float64)
    band_signal = _band_optical_depth(wave, molecules=molecules, log10_amplitudes=log10_amplitudes,
                                      cross_sections=cross_sections)

    h_m = scale_height_m(atmosphere, system)
    reference_um = float(np.median(wave))
    reference_signal = _band_optical_depth(np.array([reference_um]), molecules=molecules,
                                           log10_amplitudes=log10_amplitudes,
                                           cross_sections=cross_sections)[0]
    # z = H * ln(sigma/sigma_ref); a floor of 1e-30 keeps ln finite where
    # no band contributes (pure Rayleigh-free continuum at that wavelength).
    z_m = h_m * np.log(np.clip(band_signal, 1e-30, None) / max(reference_signal, 1e-30))

    if atmosphere.log10_cloud_pressure_bar is not None:
        # A grey cloud deck floors the apparent altitude: higher (more
        # negative log-pressure, i.e. higher in the atmosphere) clouds
        # clip more of the feature. Expressed here as a floor on z in units
        # of scale heights, since this module carries no pressure grid.
        cloud_floor_m = -atmosphere.log10_cloud_pressure_bar * h_m
        z_m = np.maximum(z_m, cloud_floor_m)

    reference_radius_m = atmosphere.reference_radius_rjup * R_JUPITER_M
    stellar_radius_m = system.stellar_radius_rsun * R_SUN_M
    apparent_radius_m = reference_radius_m + z_m
    return (apparent_radius_m / stellar_radius_m) ** 2


def forward_model(wavelength_um, atmosphere: AtmosphereParameters, system: SystemParameters, *,
                  cross_sections: Mapping[str, float]) -> dict[str, Any]:
    depth = transit_depth(wavelength_um, atmosphere, system, cross_sections=cross_sections)
    h_m = scale_height_m(atmosphere, system)
    stellar_radius_m = system.stellar_radius_rsun * R_SUN_M
    reference_radius_m = atmosphere.reference_radius_rjup * R_JUPITER_M
    # One-scale-height signal amplitude, the standard detectability yardstick
    # (e.g. Stevenson 2016's ~5 scale-height "transmission spectroscopy metric").
    delta_depth_per_h = 2.0 * reference_radius_m * h_m / stellar_radius_m ** 2
    return {"schema_version": SCHEMA_VERSION, "wavelength_um": np.asarray(wavelength_um).tolist(),
           "depth": depth.tolist(), "scale_height_m": h_m,
           "delta_depth_per_scale_height": float(delta_depth_per_h)}


def default_bounds(depth, system: SystemParameters, *,
                   molecules: tuple[str, ...]) -> tuple[tuple[float, float], ...]:
    """Bounds derived from the data's own span, in the `to_array` order
    `[temperature_k, reference_radius_rjup, log10_amp_1, ...]`."""
    depth = np.asarray(depth, dtype=np.float64)
    if np.any(depth < 0):
        raise BiosignatureError("depth must be non-negative")
    depth_span = max(float(np.max(depth) - np.min(depth)), 1e-8)
    stellar_radius_rjup = system.stellar_radius_rsun * R_SUN_M / R_JUPITER_M
    apparent_radius_rjup = np.sqrt(depth) * stellar_radius_rjup
    span_rjup = float(np.max(apparent_radius_rjup) - np.min(apparent_radius_rjup))
    median_rjup = float(np.median(apparent_radius_rjup))
    temperature_bounds = (200.0, 3000.0)
    radius_bounds = (max(1e-3, median_rjup - 4.0 * max(span_rjup, 0.01 * median_rjup)),
                     median_rjup + 4.0 * max(span_rjup, 0.01 * median_rjup))
    amplitude_bounds = (-10.0, np.log10(max(depth_span, 1e-8)) + 2.0)
    return (temperature_bounds, radius_bounds) + tuple(amplitude_bounds for _ in molecules)


__all__ = [
    "BiosignatureError", "SystemParameters", "AtmosphereParameters", "MOLECULAR_BANDS",
    "surface_gravity_ms2", "scale_height_m", "transit_depth", "forward_model",
    "default_bounds", "_finite_arrays",
]
