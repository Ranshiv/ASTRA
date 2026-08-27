"""Validation for `kilonova.py`: counterpart recall at a fixed telescope
budget and distance-conditioned calibration (roadmap item 21's two named
metrics), against a REAL GW event's sky-localization and distance
posteriors.

Reuses `gw.py` UNCHANGED for sky localization (`skymap_path`,
`build_skymap_from_samples`) -- that module's own stated scope is sky
position only, not distance, so distance samples are read here instead, by
this module, from the SAME posterior-samples HDF5 file `gw.skymap_path`
already downloads (mirroring `gw._select_posterior_positions`'s own
approach, not modifying `gw.py`). `posterior_coverage`/`sbc_ranks`/
`CoverageTrial` are reused from `microlensing_eval.py` UNCHANGED, exactly
as this session already proved they generalize cleanly to an unrelated
physical model while building `line_profile_eval.py`.

A key performance choice: flux scales EXACTLY as `1/distance^2` for fixed
intrinsic ejecta parameters (the observer's distance never affects the
photosphere's own physics), so every function below computes the expensive
`kilonova.bolometric_luminosity` quadrature ONCE per (parameters, epoch)
at one reference distance, then rescales algebraically for every other
candidate distance -- not by re-running the model per distance, which
would be needlessly, dramatically slower for a Monte Carlo study over many
distance draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import kilonova as kn
from .microlensing_eval import CoverageTrial, parameter_bias, posterior_coverage, sbc_ranks

REFERENCE_DISTANCE_MPC = 1.0
DEFAULT_BAND_WAVELENGTH_ANGSTROM = 6231.0   # SDSS/DES-like r-band pivot


class KilonovaEvalError(ValueError):
    """A kilonova evaluation study could not be run."""


def _select_posterior_distances(hdf5_path: Path) -> np.ndarray | None:
    """Real `luminosity_distance` posterior samples (Mpc), if present.

    Mirrors `gw._select_posterior_positions`'s "first `_posterior` group"
    convention exactly, kept local to this module since distance is
    outside `gw.py`'s own stated scope.
    """
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        group_name = next((name for name in handle if name.endswith("_posterior")), None)
        if group_name is None:
            return None
        dataset = handle[group_name]
        names = dataset.dtype.names or ()
        for candidate in ("luminosity_distance", "distance"):
            if candidate in names:
                return np.asarray(dataset[:][candidate], dtype=np.float64)
    return None


def reference_flux_at_epochs(components: list[kn.KilonovaParams], epoch_days,
                             band_wavelength_angstrom: float = DEFAULT_BAND_WAVELENGTH_ANGSTROM
                             ) -> np.ndarray:
    """Flux at `REFERENCE_DISTANCE_MPC` for each requested epoch -- the one
    expensive model evaluation every distance-rescaled quantity below
    reuses."""
    epoch_s = np.atleast_1d(np.asarray(epoch_days, dtype=np.float64)) * 86400.0
    return kn.multi_component_light_curve(
        epoch_s, components, band_wavelength_angstrom, distance_mpc=REFERENCE_DISTANCE_MPC)


def flux_at_distance(reference_flux: np.ndarray, distance_mpc) -> np.ndarray:
    """Exact `1/d^2` rescaling from a `REFERENCE_DISTANCE_MPC` evaluation
    -- see module docstring."""
    distance = np.asarray(distance_mpc, dtype=np.float64)
    return reference_flux * (REFERENCE_DISTANCE_MPC / distance) ** 2


def counterpart_recall_at_budget(components: list[kn.KilonovaParams], *,
                                 sky_probability: np.ndarray, distance_samples_mpc: np.ndarray,
                                 n_pointings: int, limiting_ab_mag: float,
                                 epoch_days: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
                                 band_wavelength_angstrom: float = DEFAULT_BAND_WAVELENGTH_ANGSTROM,
                                 n_trials: int = 500, seed: int = 42) -> dict:
    """Counterpart recall at a fixed telescope budget (item 21's first
    named metric): given `n_pointings` HEALPix-pixel pointings -- the
    `n_pointings` highest-probability pixels of `sky_probability`, the
    simplest defensible greedy tiling strategy -- and a survey depth
    (`limiting_ab_mag`), what fraction of synthetic kilonovae with a TRUE
    position drawn from the real sky-probability map and a TRUE distance
    drawn from the real GW distance posterior are (a) inside the tiled
    footprint and (b) bright enough to detect at at least one epoch?

    `sky_probability` and `distance_samples_mpc` come from `gw.
    build_skymap_from_samples` and `_select_posterior_distances`
    respectively, both applied to the SAME event -- this function does not
    fetch them itself, so it can be tested against synthetic maps/samples
    without a live GW download.
    """
    if n_pointings < 1:
        raise KilonovaEvalError("n_pointings must be at least 1")
    if len(distance_samples_mpc) == 0:
        raise KilonovaEvalError("distance_samples_mpc must be non-empty")

    reference_flux = reference_flux_at_epochs(components, epoch_days, band_wavelength_angstrom)
    footprint_pixels = set(np.argsort(-sky_probability)[:n_pointings].tolist())

    rng = np.random.default_rng(seed)
    pixel_indices = np.arange(len(sky_probability))
    normalized_probability = sky_probability / sky_probability.sum()

    n_in_footprint = 0
    n_detected = 0
    for _ in range(n_trials):
        true_pixel = int(rng.choice(pixel_indices, p=normalized_probability))
        in_footprint = true_pixel in footprint_pixels
        n_in_footprint += int(in_footprint)
        if not in_footprint:
            continue
        true_distance = float(rng.choice(distance_samples_mpc))
        flux = flux_at_distance(reference_flux, true_distance)
        mags = kn.flux_density_to_ab_mag(flux, band_wavelength_angstrom)
        if bool(np.any(mags <= limiting_ab_mag)):
            n_detected += 1

    return {
        "n_trials": n_trials, "n_pointings": n_pointings,
        "n_in_footprint": n_in_footprint,
        "footprint_fraction": n_in_footprint / n_trials,
        "n_detected": n_detected,
        "recall_at_budget": n_detected / n_trials,
        "limiting_ab_mag": limiting_ab_mag, "epoch_days": list(epoch_days),
    }


def distance_conditioned_calibration(components: list[kn.KilonovaParams],
                                     distance_samples_mpc: np.ndarray, epoch_days: float, *,
                                     band_wavelength_angstrom: float = DEFAULT_BAND_WAVELENGTH_ANGSTROM,
                                     n_trials: int = 200, n_resamples: int = 500,
                                     levels: tuple[float, ...] = (0.5, 0.68, 0.9),
                                     seed: int = 42) -> dict:
    """Distance-conditioned calibration (item 21's second named metric):
    does the REAL GW distance posterior's spread, propagated through the
    kilonova model, produce a well-calibrated apparent-magnitude credible
    interval?

    Each trial draws a TRUE distance from the real posterior and computes
    the true apparent AB magnitude at `epoch_days`; the credible interval
    for that SAME quantity is built by bootstrap-resampling the same
    posterior `n_resamples` times and propagating each draw through the
    (already-computed-once) reference flux. A calibrated `k%` interval
    should contain the true magnitude in ~`k%` of trials -- reuses
    `microlensing_eval.CoverageTrial`/`posterior_coverage`/`sbc_ranks`
    directly, treating "predicted magnitude" as the one parameter being
    covered.
    """
    if len(distance_samples_mpc) == 0:
        raise KilonovaEvalError("distance_samples_mpc must be non-empty")
    reference_flux = reference_flux_at_epochs(components, [epoch_days], band_wavelength_angstrom)
    rng = np.random.default_rng(seed)

    trials: list[CoverageTrial] = []
    fitted_rows: list[dict] = []
    truth_rows: list[dict] = []

    for _ in range(n_trials):
        true_distance = float(rng.choice(distance_samples_mpc))
        true_mag = float(kn.flux_density_to_ab_mag(
            flux_at_distance(reference_flux, true_distance), band_wavelength_angstrom)[0])

        resampled_distances = rng.choice(distance_samples_mpc, size=n_resamples, replace=True)
        resampled_mags = kn.flux_density_to_ab_mag(
            flux_at_distance(reference_flux, resampled_distances), band_wavelength_angstrom)

        intervals: dict = {}
        for level in levels:
            tail = (1.0 - level) / 2.0
            low, high = np.quantile(resampled_mags, [tail, 1.0 - tail])
            intervals["apparent_mag"] = intervals.get("apparent_mag", {})
            intervals["apparent_mag"][str(level)] = [float(low), float(high)]

        trials.append(CoverageTrial(
            truth={"apparent_mag": true_mag}, intervals=intervals,
            samples=resampled_mags.reshape(-1, 1), names=("apparent_mag",)))
        fitted_rows.append({"apparent_mag": float(np.median(resampled_mags))})
        truth_rows.append({"apparent_mag": true_mag})

    return {
        "n_trials": n_trials, "epoch_days": epoch_days,
        "parameter_bias": parameter_bias(fitted_rows, truth_rows, names=("apparent_mag",)),
        "posterior_coverage": posterior_coverage(trials, levels=levels),
        "sbc": sbc_ranks(trials),
    }
