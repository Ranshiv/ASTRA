"""Roadmap "frontier" domains: habitability scoring/ranking, NEO hazard
assessment, asteroseismology, biosignature synthesis/fit/detection, and
technosignature search/cadence.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

from typing import Any

from .common import Handler

from .. import (asteroseismology, biosignature, biosignature_fit, evidence,
                exoplanet_archive, habitability, neo_hazard, security,
                significance, store, technosignature)

def _handle_habitability_score(params: dict[str, Any]) -> dict[str, Any]:
    """Score a named confirmed planet's HZ position and ESI.

    Diagnostic evidence only -- see `habitability.py`'s module docstring
    -- never folded into `evidence.WEIGHTS`/`scoring.combine()`, matching
    every other read-only interpretation handler in this file
    (`physical.characterize`, `significance.calibrate`).
    """
    return habitability.score_archive_planet(
        str(params["planet_name"]), offline=bool(params.get("offline", False)))


def _handle_habitability_rank(params: dict[str, Any]) -> dict[str, Any]:
    records = exoplanet_archive.query_planets_bounded(
        teff_min=params.get("teff_min"), teff_max=params.get("teff_max"),
        insolation_min=params.get("insolation_min"), insolation_max=params.get("insolation_max"),
        max_rows=int(params.get("max_rows", 500)), offline=bool(params.get("offline", False)))
    ranked = habitability.rank_planets(records, limit=int(params.get("limit", 50)))
    return {"count": len(ranked), "planets": ranked}


def _handle_neo_assess(params: dict[str, Any]) -> dict[str, Any]:
    """Hazard assessment for one object's orbital elements.

    `elements` follows the same shape `moving_objects.state_vector_to_elements`
    already produces. `earth_elements` defaults to a circular 1 AU orbit when
    omitted, so a caller need not compute Earth's own osculating elements
    just to get a MOID -- diagnostic evidence only, never folded into
    `evidence.WEIGHTS`/`scoring.combine()`, per `neo_hazard.py`'s docstring.
    """
    elements = dict(params["elements"])
    earth_elements = params.get("earth_elements")
    if earth_elements is None:
        earth_elements = {"semi_major_axis_au": 1.0, "eccentricity": 0.0167,
                          "inclination_deg": 0.0, "raan_deg": 0.0,
                          "argument_of_perihelion_deg": 0.0, "mean_anomaly_deg": 0.0,
                          "epoch_mjd": elements.get("epoch_mjd", 60000.0)}
    return neo_hazard.assess(
        elements, earth_elements=earth_elements,
        apparent_v=params.get("apparent_v"), heliocentric_au=params.get("heliocentric_au"),
        geocentric_au=params.get("geocentric_au"), phase_angle_deg=params.get("phase_angle_deg"))


def _handle_neo_close_approach(params: dict[str, Any]) -> dict[str, Any]:
    return neo_hazard.close_approach(
        dict(params["elements"]), start_mjd=float(params["start_mjd"]),
        end_mjd=float(params["end_mjd"]), step_days=float(params.get("step_days", 1.0)))


def _handle_asteroseismology_measure(params: dict[str, Any]) -> dict[str, Any]:
    """Measure numax/Dnu (and, if `teff_k` given, R/M) from a stored curve.

    Reads the same `path`-addressed stored-curve convention `curves.get`
    already uses (`store.read_curve`), rather than shipping a raw
    thousands-of-points array through the JSON-lines RPC transport.
    """
    curve = store.read_curve(security.authorized_path(params["path"])).dropna().sorted_by_time()
    teff_k = params.get("teff_k")
    return asteroseismology.measure(curve.time, curve.value,
                                    teff_k=float(teff_k) if teff_k is not None else None)


def _handle_technosignature_search(params: dict[str, Any]) -> dict[str, Any]:
    """Synthesize a demo dynamic spectrum and run the de-Doppler search.

    Every number here is measured on SYNTHETIC data -- see
    `technosignature.py`'s module docstring `[GAP]`: there is no real
    Breakthrough Listen data path. Diagnostic-only, never folded into
    `evidence.WEIGHTS`/`scoring.combine()`.
    """
    synthesis = technosignature.synthesize_waterfall(
        n_time=int(params.get("n_time", 16)), n_freq=int(params.get("n_freq", 1024)),
        f0_hz=float(params.get("f0_hz", 1.4e9)),
        channel_width_hz=float(params.get("channel_width_hz", 2.7939677)),
        dt_s=float(params.get("dt_s", 18.25)),
        drift_rate_hz_s=float(params.get("drift_rate_hz_s", 0.0)),
        snr=float(params.get("snr", 0.0)),
        start_channel=params.get("start_channel"),
        seed=int(params.get("seed", 42)))
    spectrum = synthesis["spectrum"]
    result = technosignature.search(
        spectrum, max_drift_hz_s=float(params.get("max_drift_hz_s", technosignature.DEFAULT_MAX_DRIFT_HZ_S)),
        snr_threshold=float(params.get("snr_threshold", technosignature.DEFAULT_SNR_THRESHOLD)))
    result["truth"] = synthesis["truth"]
    return result


def _handle_technosignature_cadence(params: dict[str, Any]) -> dict[str, Any]:
    def _to_hits(rows: list[dict[str, Any]]) -> list[technosignature.TechnosignatureHit]:
        return [technosignature.TechnosignatureHit(
            frequency_hz=float(row["frequency_hz"]), drift_rate_hz_s=float(row["drift_rate_hz_s"]),
            snr=float(row["snr"]), freq_channel_index=int(row["freq_channel_index"]),
            drift_index=int(row["drift_index"])) for row in rows]

    on_scans = [_to_hits(scan) for scan in params["on_hit_lists"]]
    off_scans = [_to_hits(scan) for scan in params["off_hit_lists"]]
    survivors = technosignature.cadence_filter(
        on_scans, off_scans, frequency_tolerance_hz=float(params.get("frequency_tolerance_hz", 1.0)),
        drift_tolerance_hz_s=float(params.get("drift_tolerance_hz_s", 0.1)))
    return {"survivors": [hit.to_dict() for hit in survivors], "n_survivors": len(survivors)}


def _handle_biosignature_synthesize(params: dict[str, Any]) -> dict[str, Any]:
    """Generate a synthetic transmission spectrum for the panel's demo
    workflow -- pure forward-model evaluation plus Gaussian noise, the
    same "synthetic injection demonstration" shape `digital_twin.sample`
    already uses for a data type this engine has no live ingestion for.
    """
    import numpy as np

    system = biosignature.SystemParameters(
        stellar_radius_rsun=float(params["stellar_radius_rsun"]),
        planet_mass_mjup=float(params["planet_mass_mjup"]))
    atmosphere = biosignature.AtmosphereParameters(
        temperature_k=float(params.get("temperature_k", 1000.0)),
        mean_molecular_weight=float(params.get("mean_molecular_weight", 2.3)),
        reference_radius_rjup=float(params.get("reference_radius_rjup", 1.0)),
        abundances=tuple((str(m), float(a)) for m, a in params.get("abundances", [])))
    cross_sections = {str(k): float(v) for k, v in params.get("cross_sections", {}).items()}
    n_points = int(params.get("n_points", 40))
    wave_min, wave_max = float(params.get("wavelength_min_um", 1.0)), float(params.get("wavelength_max_um", 2.5))
    wavelength_um = np.linspace(wave_min, wave_max, n_points)
    depth = biosignature.transit_depth(wavelength_um, atmosphere, system, cross_sections=cross_sections)
    error_ppm = float(params.get("error_ppm", 50.0))
    rng = np.random.default_rng(int(params.get("seed", 42)))
    noisy_depth = depth + rng.normal(0.0, error_ppm * 1e-6, size=depth.shape)
    error = np.full_like(depth, error_ppm * 1e-6)
    return {"wavelength_um": wavelength_um.tolist(), "depth": noisy_depth.tolist(),
           "error": error.tolist(), "truth_depth": depth.tolist()}


def _handle_biosignature_fit(params: dict[str, Any]) -> dict[str, Any]:
    """Fit a transmission spectrum for one or more molecular bands.

    Diagnostic evidence only -- see `biosignature.py`'s module docstring
    -- fitted amplitudes are band-detection significance, NOT calibrated
    abundances; never folded into `evidence.WEIGHTS`/`scoring.combine()`.
    """
    system = biosignature.SystemParameters(
        stellar_radius_rsun=float(params["stellar_radius_rsun"]),
        planet_mass_mjup=float(params["planet_mass_mjup"]))
    molecules = tuple(str(m) for m in params["molecules"])
    cross_sections = {str(k): float(v) for k, v in params["cross_sections"].items()}
    result = biosignature_fit.fit_transmission_spectrum(
        params["wavelength_um"], params["depth"], params["error"], system,
        molecules=molecules, cross_sections=cross_sections,
        mean_molecular_weight=float(params.get("mean_molecular_weight", 2.3)),
        seed=int(params.get("seed", 42)))
    return result.to_dict()


def _handle_biosignature_detect(params: dict[str, Any]) -> dict[str, Any]:
    system = biosignature.SystemParameters(
        stellar_radius_rsun=float(params["stellar_radius_rsun"]),
        planet_mass_mjup=float(params["planet_mass_mjup"]))
    cross_sections = {str(k): float(v) for k, v in params["cross_sections"].items()}
    molecules = tuple(str(m) for m in params.get("molecules", list(cross_sections)))
    significances = {
        molecule: biosignature_fit.detection_significance(
            params["wavelength_um"], params["depth"], params["error"], system, molecule,
            cross_sections=cross_sections,
            mean_molecular_weight=float(params.get("mean_molecular_weight", 2.3)),
            seed=int(params.get("seed", 42)))
        for molecule in molecules
    }
    disequilibrium = biosignature_fit.disequilibrium_flag(significances)
    return {"significances": significances, "disequilibrium": disequilibrium}


def _handle_asteroseismology_solve(params: dict[str, Any]) -> dict[str, Any]:
    seismic = asteroseismology.SeismicParameters(
        numax_uhz=float(params["numax_uhz"]), delta_nu_uhz=float(params["delta_nu_uhz"]),
        teff_k=float(params["teff_k"]),
        numax_uhz_error=params.get("numax_uhz_error"), delta_nu_uhz_error=params.get("delta_nu_uhz_error"),
        teff_k_error=params.get("teff_k_error"))
    return asteroseismology.solve_scaling_relations(seismic).to_dict()


HANDLERS: dict[str, Handler] = {
    "habitability.score": _handle_habitability_score,
    "habitability.rank": _handle_habitability_rank,
    "neo.assess": _handle_neo_assess,
    "neo.close_approach": _handle_neo_close_approach,
    "asteroseismology.measure": _handle_asteroseismology_measure,
    "asteroseismology.solve": _handle_asteroseismology_solve,
    "biosignature.synthesize": _handle_biosignature_synthesize,
    "biosignature.fit": _handle_biosignature_fit,
    "biosignature.detect": _handle_biosignature_detect,
    "technosignature.search": _handle_technosignature_search,
    "technosignature.cadence": _handle_technosignature_cadence,
}
