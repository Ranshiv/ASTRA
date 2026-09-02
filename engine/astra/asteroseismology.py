"""Solar-like oscillation scaling relations and a pure-numpy detection path
(roadmap: astrophysics & extraterrestrial-study feature pass).

Scaling relations: Kjeldsen & Bedding (1995, A&A 293, 87) relate the
frequency of maximum oscillation power (`numax`) and the large frequency
separation (`Delta nu`) to stellar mass, radius, and Teff:

    numax/numax_sun = (M/Msun) * (R/Rsun)^-2 * (Teff/Teff_sun)^-0.5
    Dnu/Dnu_sun     = (M/Msun)^0.5 * (R/Rsun)^-1.5   [ = sqrt(rho/rho_sun) ]

Inverted (the "direct method", e.g. Chaplin & Miglio 2013, ARA&A 51, 353):

    R/Rsun = (numax/numax_sun) * (Dnu/Dnu_sun)^-2 * (Teff/Teff_sun)^0.5
    M/Msun = (numax/numax_sun)^3 * (Dnu/Dnu_sun)^-4 * (Teff/Teff_sun)^1.5

Solar reference values `NUMAX_SUN_UHZ`/`DNU_SUN_UHZ` are the Huber et al.
(2011, ApJ 743, 143) VIRGO-calibrated pair (numax_sun = 3090 +/- 30 uHz,
Dnu_sun = 135.1 +/- 0.1 uHz), confirmed this session against a secondary
source quoting that paper directly; other pipelines quote slightly
different solar references (e.g. 3050-3150 uHz), which is why both
constants are exposed as keyword overrides rather than only hardcoded.

`envelope_window` uses Mosser et al. (2012, A&A 537, A30)'s oscillation-
envelope FWHM scaling `FWHM_uHz = 0.66 * numax_uHz^0.88`, confirmed this
session -- calibrated on red giants (numax below ~100-300 uHz) but used
here only to size an integration/search window around a candidate numax,
not as a reported science quantity, so extrapolating it to main-sequence
numax values is a stated approximation, not a precision claim.

Measurement (`measure`) is a PURE-NUMPY fallback path: `power_spectrum`
computes a Lomb-Scargle periodogram directly via `astropy.timeseries.
LombScargle` (lazy-imported, matching this codebase's astropy-import
convention), converted to a frequency axis in microHz; `estimate_numax`
fits a Gaussian power excess over a two-Harvey-profile-plus-white-noise
background; `estimate_delta_nu` autocorrelates the power spectrum within
the envelope window and disambiguates the peak nearest the
Stello et al. (2009, MNRAS 400, L80) Dnu-numax relation
`Dnu_uHz = 0.263 * numax_uHz^0.772` (coefficients confirmed this
session). `lightkurve.seismology`'s `estimate_numax`/`estimate_deltanu`
(Viani et al. 2019 2-D autocorrelation method) is available as a
strictly optional cross-check of the library's API surface (no network
needed; see `tests/test_asteroseismology_lightkurve.py`) -- it is NOT
the implementation path and its output is NOT asserted to numerically
agree with this module's own estimator (a quick check found it does
not, at default tuning, on the same synthetic input -- see that test
file's docstring), so this module stays offline-testable without it.

[GAP] No individual mode-frequency extraction, no peakbagging, no
Lorentzian-mode MCMC fit, no rotational splitting, no mixed-mode/period-
spacing analysis for red giants (so no RGB-vs-red-clump evolutionary
classification), no fitted granulation (Harvey) background beyond the
crude two-component model `estimate_numax` uses internally to isolate the
excess, and no surface-term or grid-based stellar-evolution-track
correction. Reported masses/radii are DIRECT-METHOD scaling-relation
values; published corrections to the Dnu scaling for evolved/metal-poor
stars (e.g. Sharma et al. 2016) are not applied, and the eval module
reports the raw scaling relation's known failure mode (the ~2x/0.5x Dnu
aliasing rate) rather than assuming it away.

Error propagation in `solve_scaling_relations` is first-order analytic
(partial derivatives of the closed-form inversion), not a Monte Carlo --
cheap, deterministic, and adequate given the inputs are themselves point
estimates with symmetric Gaussian errors; stated here since it differs
from `radio_variability.py`'s Monte Carlo-based uncertainty propagation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

# Huber et al. (2011) solar reference values.
NUMAX_SUN_UHZ = 3090.0
DNU_SUN_UHZ = 135.1
TEFF_SUN_K = 5777.0
LOGG_SUN_CGS = 4.438

# Mosser et al. (2012) envelope FWHM scaling coefficients.
MOSSER_FWHM_COEFF = 0.66
MOSSER_FWHM_EXPONENT = 0.88

# Stello et al. (2009) Dnu-numax relation, used only to disambiguate the
# autocorrelation peak (see module docstring).
STELLO_DNU_COEFF = 0.263
STELLO_DNU_EXPONENT = 0.772

MIN_POINTS_FOR_MEASUREMENT = 200
UHZ_PER_DAY_INVERSE = 1.0e6 / 86400.0  # 1 / day -> microHz


class AsteroseismologyError(ValueError):
    """Raised when asteroseismic inputs are outside the scaling-relation domain."""


@dataclass(frozen=True)
class SeismicParameters:
    numax_uhz: float
    delta_nu_uhz: float
    teff_k: float
    numax_uhz_error: float | None = None
    delta_nu_uhz_error: float | None = None
    teff_k_error: float | None = None

    def __post_init__(self) -> None:
        if self.numax_uhz <= 0 or self.delta_nu_uhz <= 0 or self.teff_k <= 0:
            raise AsteroseismologyError("numax_uhz, delta_nu_uhz, and teff_k must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeismicSolution:
    radius_rsun: float
    mass_msun: float
    logg_cgs: float
    density_rhosun: float
    radius_rsun_error: float | None = None
    mass_msun_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predict_numax(mass_msun: float, radius_rsun: float, teff_k: float, *,
                  numax_sun_uhz: float = NUMAX_SUN_UHZ, teff_sun_k: float = TEFF_SUN_K) -> float:
    if mass_msun <= 0 or radius_rsun <= 0 or teff_k <= 0:
        raise AsteroseismologyError("mass_msun, radius_rsun, and teff_k must be positive")
    return float(numax_sun_uhz * mass_msun * radius_rsun ** -2 * (teff_k / teff_sun_k) ** -0.5)


def predict_delta_nu(mass_msun: float, radius_rsun: float, *,
                     delta_nu_sun_uhz: float = DNU_SUN_UHZ) -> float:
    if mass_msun <= 0 or radius_rsun <= 0:
        raise AsteroseismologyError("mass_msun and radius_rsun must be positive")
    return float(delta_nu_sun_uhz * mass_msun ** 0.5 * radius_rsun ** -1.5)


def solve_scaling_relations(seismic: SeismicParameters, *,
                            numax_sun_uhz: float = NUMAX_SUN_UHZ,
                            delta_nu_sun_uhz: float = DNU_SUN_UHZ,
                            teff_sun_k: float = TEFF_SUN_K) -> SeismicSolution:
    """Direct-method inversion: (numax, Dnu, Teff) -> (R, M, logg, density)."""
    numax_ratio = seismic.numax_uhz / numax_sun_uhz
    dnu_ratio = seismic.delta_nu_uhz / delta_nu_sun_uhz
    teff_ratio = seismic.teff_k / teff_sun_k

    radius_rsun = numax_ratio * dnu_ratio ** -2 * teff_ratio ** 0.5
    mass_msun = numax_ratio ** 3 * dnu_ratio ** -4 * teff_ratio ** 1.5
    logg_cgs = LOGG_SUN_CGS + np.log10(numax_ratio) + 0.5 * np.log10(teff_ratio)
    density_rhosun = mass_msun / radius_rsun ** 3

    radius_error = None
    mass_error = None
    if (seismic.numax_uhz_error is not None and seismic.delta_nu_uhz_error is not None
            and seismic.teff_k_error is not None):
        # First-order analytic propagation: partial derivatives of the
        # closed-form power-law inversion w.r.t. each input.
        frac_numax = seismic.numax_uhz_error / seismic.numax_uhz
        frac_dnu = seismic.delta_nu_uhz_error / seismic.delta_nu_uhz
        frac_teff = seismic.teff_k_error / seismic.teff_k
        radius_error = float(radius_rsun * np.sqrt(
            frac_numax ** 2 + (2.0 * frac_dnu) ** 2 + (0.5 * frac_teff) ** 2))
        mass_error = float(mass_msun * np.sqrt(
            (3.0 * frac_numax) ** 2 + (4.0 * frac_dnu) ** 2 + (1.5 * frac_teff) ** 2))

    return SeismicSolution(
        radius_rsun=float(radius_rsun), mass_msun=float(mass_msun),
        logg_cgs=float(logg_cgs), density_rhosun=float(density_rhosun),
        radius_rsun_error=radius_error, mass_msun_error=mass_error,
    )


def envelope_window(numax_uhz: float) -> tuple[float, float]:
    """Mosser et al. (2012) envelope FWHM window `(numax - FWHM, numax + FWHM)`."""
    if numax_uhz <= 0:
        raise AsteroseismologyError("numax_uhz must be positive")
    fwhm = MOSSER_FWHM_COEFF * numax_uhz ** MOSSER_FWHM_EXPONENT
    return float(numax_uhz - fwhm), float(numax_uhz + fwhm)


def power_spectrum(time_days: np.ndarray, flux: np.ndarray, *,
                   min_frequency_uhz: float = 10.0, max_frequency_uhz: float = 8000.0,
                   samples_per_peak: int = 5) -> dict[str, Any]:
    """Lomb-Scargle power spectrum in microHz, lazy-importing astropy
    (matches this codebase's astropy-import convention)."""
    time_days = np.asarray(time_days, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    finite = np.isfinite(time_days) & np.isfinite(flux)
    time_days, flux = time_days[finite], flux[finite]
    if time_days.shape[0] < MIN_POINTS_FOR_MEASUREMENT:
        return {"frequency_uhz": None, "power": None,
               "warning": f"fewer than {MIN_POINTS_FOR_MEASUREMENT} finite points"}
    if np.std(flux) == 0.0:
        return {"frequency_uhz": None, "power": None, "warning": "flux has zero variance"}

    from astropy.timeseries import LombScargle

    model = LombScargle(time_days, flux - np.mean(flux))
    min_freq_per_day = min_frequency_uhz / UHZ_PER_DAY_INVERSE
    max_freq_per_day = max_frequency_uhz / UHZ_PER_DAY_INVERSE
    frequency_per_day = model.autofrequency(minimum_frequency=min_freq_per_day,
                                            maximum_frequency=max_freq_per_day,
                                            samples_per_peak=samples_per_peak)
    power = model.power(frequency_per_day, method="fast")
    frequency_uhz = frequency_per_day * UHZ_PER_DAY_INVERSE
    return {"frequency_uhz": frequency_uhz, "power": np.asarray(power, dtype=np.float64),
           "warning": None}


def _harvey_plus_white_background(frequency_uhz: np.ndarray, power: np.ndarray, *,
                                  exclude_mask: np.ndarray) -> np.ndarray:
    """Crude two-component Harvey-profile-plus-white-noise background fit
    on the frequency range OUTSIDE the candidate envelope, used only to
    subtract a background before the Gaussian-excess fit -- not a reported
    science quantity (see module `[GAP]`)."""
    from scipy.optimize import curve_fit

    def harvey(f, a1, b1, a2, b2, white):
        return (a1 / (1.0 + (f / b1) ** 4)) + (a2 / (1.0 + (f / b2) ** 4)) + white

    f_bg, p_bg = frequency_uhz[~exclude_mask], power[~exclude_mask]
    white_guess = float(np.median(p_bg[-max(1, len(p_bg) // 10):]))
    p0 = [float(np.median(p_bg)), float(np.median(f_bg) * 0.5),
         float(np.median(p_bg) * 0.3), float(np.median(f_bg) * 2.0), white_guess]
    try:
        popt, _ = curve_fit(harvey, f_bg, p_bg, p0=p0, maxfev=5000)
    except Exception:  # noqa: BLE001 - fall back to a flat background rather than fail
        return np.full_like(frequency_uhz, float(np.median(p_bg)))
    return harvey(frequency_uhz, *popt)


def estimate_numax(frequency_uhz: np.ndarray, power: np.ndarray, *,
                   search_min_uhz: float = 10.0, search_max_uhz: float = 8000.0) -> dict[str, Any]:
    """Fit a Gaussian power excess over a crude Harvey-plus-white background."""
    from scipy.optimize import curve_fit

    search_mask = (frequency_uhz >= search_min_uhz) & (frequency_uhz <= search_max_uhz)
    if not np.any(search_mask):
        return {"numax_uhz": None, "numax_uhz_error": None, "warning": "no data in search range"}

    coarse_peak = float(frequency_uhz[search_mask][np.argmax(power[search_mask])])
    lo, hi = envelope_window(coarse_peak)
    envelope_mask = (frequency_uhz >= lo) & (frequency_uhz <= hi)
    if np.count_nonzero(envelope_mask) < 5:
        return {"numax_uhz": None, "numax_uhz_error": None,
               "warning": "envelope window contains too few frequency samples"}

    background = _harvey_plus_white_background(frequency_uhz, power, exclude_mask=envelope_mask)
    excess = power - background

    def gaussian(f, amplitude, center, sigma):
        return amplitude * np.exp(-0.5 * ((f - center) / sigma) ** 2)

    f_fit, p_fit = frequency_uhz[envelope_mask], excess[envelope_mask]
    p0 = [float(max(p_fit.max(), 1e-6)), coarse_peak, float((hi - lo) / 2.355)]
    try:
        popt, pcov = curve_fit(gaussian, f_fit, p_fit, p0=p0, maxfev=5000)
    except Exception:  # noqa: BLE001 - a failed fit is a non-detection, not a crash
        return {"numax_uhz": None, "numax_uhz_error": None, "warning": "Gaussian excess fit failed"}
    if popt[0] <= 0 or not (search_min_uhz <= popt[1] <= search_max_uhz):
        return {"numax_uhz": None, "numax_uhz_error": None, "warning": "fitted excess is non-physical"}
    numax_error = float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) else None
    return {"numax_uhz": float(popt[1]), "numax_uhz_error": numax_error, "warning": None}


def estimate_delta_nu(frequency_uhz: np.ndarray, power: np.ndarray, numax_uhz: float) -> dict[str, Any]:
    """Autocorrelate the power spectrum within the envelope window;
    disambiguate against the Stello et al. (2009) Dnu-numax relation."""
    lo, hi = envelope_window(numax_uhz)
    mask = (frequency_uhz >= lo) & (frequency_uhz <= hi)
    if np.count_nonzero(mask) < 10:
        return {"delta_nu_uhz": None, "warning": "envelope window contains too few frequency samples"}

    freq_window, power_window = frequency_uhz[mask], power[mask]
    df = float(np.median(np.diff(freq_window)))
    if df <= 0:
        return {"delta_nu_uhz": None, "warning": "non-monotone frequency grid"}
    demeaned = power_window - np.mean(power_window)
    autocorr = np.correlate(demeaned, demeaned, mode="full")[len(demeaned) - 1:]
    lags_uhz = np.arange(autocorr.shape[0]) * df

    predicted_dnu = STELLO_DNU_COEFF * numax_uhz ** STELLO_DNU_EXPONENT
    search_lo, search_hi = 0.5 * predicted_dnu, 1.5 * predicted_dnu
    search_mask = (lags_uhz >= search_lo) & (lags_uhz <= search_hi)
    if not np.any(search_mask):
        return {"delta_nu_uhz": None, "warning": "no autocorrelation lag in the predicted Dnu window"}
    peak_idx = np.where(search_mask)[0][np.argmax(autocorr[search_mask])]
    return {"delta_nu_uhz": float(lags_uhz[peak_idx]), "predicted_dnu_uhz": float(predicted_dnu),
           "warning": None}


def measure(time_days: np.ndarray, flux: np.ndarray, teff_k: float | None = None, *,
           seed: int = 42) -> dict[str, Any]:
    """End-to-end pure-numpy measurement: power spectrum -> numax -> Dnu ->
    (if `teff_k` given) scaling-relation solution."""
    spectrum = power_spectrum(time_days, flux)
    if spectrum["frequency_uhz"] is None:
        return {"schema_version": SCHEMA_VERSION, "numax_uhz": None, "delta_nu_uhz": None,
               "solution": None, "quality": "insufficient", "warnings": [spectrum["warning"]]}

    frequency_uhz, power = spectrum["frequency_uhz"], spectrum["power"]
    numax_result = estimate_numax(frequency_uhz, power)
    warnings = []
    if numax_result["warning"]:
        warnings.append(numax_result["warning"])
    if numax_result["numax_uhz"] is None:
        return {"schema_version": SCHEMA_VERSION, "numax_uhz": None, "delta_nu_uhz": None,
               "solution": None, "quality": "insufficient", "warnings": warnings}

    dnu_result = estimate_delta_nu(frequency_uhz, power, numax_result["numax_uhz"])
    if dnu_result["warning"]:
        warnings.append(dnu_result["warning"])

    solution = None
    if teff_k is not None and dnu_result["delta_nu_uhz"] is not None:
        seismic = SeismicParameters(numax_uhz=numax_result["numax_uhz"],
                                    delta_nu_uhz=dnu_result["delta_nu_uhz"], teff_k=teff_k)
        solution = solve_scaling_relations(seismic).to_dict()

    quality = "usable" if dnu_result["delta_nu_uhz"] is not None else "insufficient"
    return {"schema_version": SCHEMA_VERSION, "numax_uhz": numax_result["numax_uhz"],
           "numax_uhz_error": numax_result["numax_uhz_error"],
           "delta_nu_uhz": dnu_result["delta_nu_uhz"], "solution": solution,
           "quality": quality, "warnings": warnings, "seed": seed}


def echelle(frequency_uhz: np.ndarray, power: np.ndarray, delta_nu_uhz: float) -> dict[str, Any]:
    """Echelle-diagram coordinates: `frequency mod Dnu` vs. frequency."""
    if delta_nu_uhz <= 0:
        raise AsteroseismologyError("delta_nu_uhz must be positive")
    frequency_uhz = np.asarray(frequency_uhz, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    folded_uhz = np.mod(frequency_uhz, delta_nu_uhz)
    return {"frequency_uhz": frequency_uhz.tolist(), "folded_uhz": folded_uhz.tolist(),
           "power": power.tolist(), "delta_nu_uhz": float(delta_nu_uhz)}


__all__ = [
    "AsteroseismologyError", "SeismicParameters", "SeismicSolution",
    "NUMAX_SUN_UHZ", "DNU_SUN_UHZ", "TEFF_SUN_K",
    "predict_numax", "predict_delta_nu", "solve_scaling_relations",
    "envelope_window", "power_spectrum", "estimate_numax", "estimate_delta_nu",
    "measure", "echelle",
]
