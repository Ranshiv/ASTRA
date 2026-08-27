"""Bolometric flare energy and synthetic injection-recovery validation.

Split from `flare.py` purely to keep each file under this project's
500-line guideline (same `stellar_manifold.py`/`stellar_manifold_eval.py`
split rationale, not an independent module).

`flare.equivalent_duration` gives a real, exact time-integral of relative
flux excess from data alone. Converting that into a bolometric energy (erg)
needs the star's quiescent bolometric luminosity
(`L = 4*pi*R^2*sigma*Teff^4`, Stefan-Boltzmann). This module adds the first
absolute (not ratio-only) luminosity computation in this codebase --
`eclipsing_binary.py`'s luminosity weighting only ever needed a RATIO
(`R^2 * Teff^4` between two bodies), never an absolute erg/s value, so no
prior module needed real sigma_SB/R_sun/L_sun constants. `SIGMA_SB_CGS` and
`R_SUN_CM` here are verified this session against the real nominal solar
luminosity: `quiescent_bolometric_luminosity(radius_solar=1.0,
teff_k=5772.0)` returns 3.828e33 erg/s to 4 significant figures, matching
the IAU nominal solar luminosity constant -- a real, checked consistency
test, not an assumed-correct constant.

`bolometric_flare_energy`'s `E = ED * L` relation is the standard
PASSBAND-flux-excess approximation (Hawley et al. 2014, Davenport 2016),
NOT the more sophisticated two-blackbody bandpass-correction method
(comparing a ~9000-10000 K flare blackbody against the star's own cooler
photosphere, integrated against the real TESS/Kepler response function).
That correction needs a real instrument response curve this codebase does
not have and would be a second, separable acquisition task -- an explicit,
stated scope limit, not glossed over, the same restraint `transit_ttv.py`
states for its own "no occultation dimming" limit.

`quiescent_bolometric_luminosity` takes `radius_solar`/`teff_k` directly
rather than an object to anchor itself -- a caller gets those from
`eclipsing_binary_dimensions.anchor_physical_radius`/
`mass_and_radius_at_teff` (item 17's real Gaia-photometry-or-ZAMS-fallback
anchor, reused unchanged rather than reimplemented) before calling this
function, the same "small function, caller assembles the pipeline"
composition `transit_ttv.py`'s own functions use.

The synthetic injection-recovery study below answers this backlog item's
own named metrics (energy bias, flare completeness, duration/amplitude
calibration) on SYNTHETIC ground truth (a flat baseline plus Gaussian
noise, not a real stored light curve) -- the same "mechanism validated on
synthetic data, not yet run at real Stage-B scale" caveat every other eval
module in this family (`stellar_manifold_eval.py`, `multiband_hier.py`)
already carries; running this against real TESS/ZTF baselines remains open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import flare

# Stefan-Boltzmann constant, cgs (erg cm^-2 s^-1 K^-4), CODATA value.
SIGMA_SB_CGS = 5.670374419e-5
# IAU 2015 nominal solar radius, cm (same source `eclipsing_binary_dimensions.
# AU_IN_SOLAR_RADII` derives its own constant from).
R_SUN_CM = 6.957e10


class FlareEnergyError(ValueError):
    """A bolometric energy or injection-recovery computation could not be completed."""


def quiescent_bolometric_luminosity(radius_solar: float, teff_k: float) -> float:
    """Stefan-Boltzmann bolometric luminosity, erg/s."""
    if radius_solar <= 0 or not math.isfinite(radius_solar):
        raise FlareEnergyError("radius_solar must be a positive finite number")
    if teff_k <= 0 or not math.isfinite(teff_k):
        raise FlareEnergyError("teff_k must be a positive finite number")
    radius_cm = radius_solar * R_SUN_CM
    return float(4.0 * math.pi * radius_cm ** 2 * SIGMA_SB_CGS * teff_k ** 4)


def bolometric_flare_energy(ed_seconds: float, quiescent_luminosity_erg_s: float) -> float:
    """`E = ED * L` -- the passband-approximation relation; see the module
    docstring for what this deliberately does not correct for."""
    if ed_seconds < 0:
        raise FlareEnergyError("ed_seconds must be non-negative")
    if quiescent_luminosity_erg_s <= 0:
        raise FlareEnergyError("quiescent_luminosity_erg_s must be positive")
    return float(ed_seconds * quiescent_luminosity_erg_s)


# ---------------------------------------------------------------------------
# Synthetic injection-recovery validation.
# ---------------------------------------------------------------------------

def _summary(values: list[float]) -> dict | None:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not len(finite):
        return None
    return {
        "mean": round(float(np.mean(finite)), 4),
        "std": round(float(np.std(finite, ddof=1)), 4) if len(finite) > 1 else 0.0,
        "ci95": [round(float(np.quantile(finite, 0.025)), 4),
                round(float(np.quantile(finite, 0.975)), 4)],
    }


@dataclass(frozen=True)
class AmplitudeTrialResult:
    amplitude: float
    n_injected: int
    n_recovered: int
    fwhm_fractional_bias: list[float] = field(default_factory=list)
    amplitude_fractional_bias: list[float] = field(default_factory=list)
    energy_fractional_bias: list[float] = field(default_factory=list)

    @property
    def completeness(self) -> float:
        return self.n_recovered / self.n_injected if self.n_injected else 0.0

    def to_dict(self) -> dict:
        return {
            "amplitude": self.amplitude, "n_injected": self.n_injected,
            "n_recovered": self.n_recovered, "completeness": round(self.completeness, 4),
            "fwhm_fractional_bias": _summary(self.fwhm_fractional_bias),
            "amplitude_fractional_bias": _summary(self.amplitude_fractional_bias),
            "energy_fractional_bias": _summary(self.energy_fractional_bias),
        }


def evaluate_flare_recovery(*, amplitude_grid: list[float], fwhm_days: float = 0.03,
                            radius_solar: float = 1.0, teff_k: float = 5000.0,
                            n_trials_per_amplitude: int = 5, span_days: float = 10.0,
                            cadence_days: float = 0.02, noise_sigma: float = 0.002,
                            baseline_window_days: float = 1.0, seed: int = 0
                            ) -> list[AmplitudeTrialResult]:
    """Injects known Davenport-template flares (fixed `fwhm_days`, a known
    synthetic `radius_solar`/`teff_k`) onto a flat, Gaussian-noise synthetic
    baseline across `amplitude_grid`, runs the real detection+fit+ED+energy
    pipeline, and reports completeness and fractional bias in fwhm/
    amplitude/energy with a multi-seed mean/std/ci95 summary
    (`sweep.TrialResult`-style). Never asserts a winner -- reports only, per
    the module docstring.
    """
    if not amplitude_grid:
        raise FlareEnergyError("amplitude_grid must be non-empty")
    if n_trials_per_amplitude < 1:
        raise FlareEnergyError("n_trials_per_amplitude must be at least 1")

    quiescent_luminosity = quiescent_bolometric_luminosity(radius_solar, teff_k)
    time = np.arange(0.0, span_days, cadence_days)
    results: list[AmplitudeTrialResult] = []

    for amplitude in amplitude_grid:
        n_recovered = 0
        fwhm_bias: list[float] = []
        amplitude_bias: list[float] = []
        energy_bias: list[float] = []
        rng = np.random.default_rng(seed + hash(round(amplitude, 6)) % (2 ** 16))
        for _ in range(n_trials_per_amplitude):
            t_peak = float(rng.uniform(span_days * 0.2, span_days * 0.8))
            true_model = flare.flare_model(time, t_peak, fwhm_days, amplitude)
            true_ed = flare.equivalent_duration(time, true_model)
            true_energy = bolometric_flare_energy(true_ed, quiescent_luminosity)

            noisy_flux = (1.0 + true_model) + rng.normal(0.0, noise_sigma, size=time.size)
            err = np.full_like(time, noise_sigma)

            candidates = flare.detect_flare_candidates(
                time, noisy_flux, err, "flux", baseline_window_days=baseline_window_days)
            matched = [c for c in candidates if abs(c.peak_time - t_peak) <= 3.0 * fwhm_days]
            if not matched:
                continue

            excess, excess_err = flare.relative_flux_excess(
                time, noisy_flux, err, "flux", baseline_window_days=baseline_window_days)
            best = max(matched, key=lambda c: c.peak_excess)
            try:
                fit = flare.fit_flare_template(
                    time, excess, excess_err,
                    {"t_peak": best.peak_time, "fwhm": fwhm_days, "amplitude": best.peak_excess})
            except Exception:  # noqa: BLE001 -- a non-converged fit counts as a miss, not a crash
                continue

            recovered_ed = flare.equivalent_duration(
                time, flare.flare_model(time, fit.t_peak, fit.fwhm, fit.amplitude))
            recovered_energy = bolometric_flare_energy(recovered_ed, quiescent_luminosity)

            n_recovered += 1
            fwhm_bias.append((fit.fwhm - fwhm_days) / fwhm_days)
            amplitude_bias.append((fit.amplitude - amplitude) / amplitude)
            energy_bias.append((recovered_energy - true_energy) / true_energy)
        results.append(AmplitudeTrialResult(
            amplitude=float(amplitude), n_injected=n_trials_per_amplitude, n_recovered=n_recovered,
            fwhm_fractional_bias=fwhm_bias, amplitude_fractional_bias=amplitude_bias,
            energy_fractional_bias=energy_bias))
    return results


__all__ = [
    "FlareEnergyError", "SIGMA_SB_CGS", "R_SUN_CM",
    "quiescent_bolometric_luminosity", "bolometric_flare_energy",
    "AmplitudeTrialResult", "evaluate_flare_recovery",
]
