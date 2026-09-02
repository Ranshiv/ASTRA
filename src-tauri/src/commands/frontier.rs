//! Roadmap "frontier" domains: habitability scoring/ranking, NEO hazard
//! assessment, asteroseismology, and biosignature/technosignature search.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_habitability_score(
    state: tauri::State<'_, Arc<Engine>>,
    planet_name: String,
    offline: Option<bool>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "habitability.score", json!({
        "planet_name": planet_name,
        "offline": offline.unwrap_or(false),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_habitability_rank(
    state: tauri::State<'_, Arc<Engine>>,
    teff_min: Option<f64>,
    teff_max: Option<f64>,
    insolation_min: Option<f64>,
    insolation_max: Option<f64>,
    max_rows: Option<u32>,
    limit: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "habitability.rank", json!({
        "teff_min": teff_min,
        "teff_max": teff_max,
        "insolation_min": insolation_min,
        "insolation_max": insolation_max,
        "max_rows": max_rows.unwrap_or(500),
        "limit": limit.unwrap_or(50),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_neo_assess(
    state: tauri::State<'_, Arc<Engine>>,
    elements: Value,
    earth_elements: Option<Value>,
    apparent_v: Option<f64>,
    heliocentric_au: Option<f64>,
    geocentric_au: Option<f64>,
    phase_angle_deg: Option<f64>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "neo.assess", json!({
        "elements": elements,
        "earth_elements": earth_elements,
        "apparent_v": apparent_v,
        "heliocentric_au": heliocentric_au,
        "geocentric_au": geocentric_au,
        "phase_angle_deg": phase_angle_deg,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_neo_close_approach(
    state: tauri::State<'_, Arc<Engine>>,
    elements: Value,
    start_mjd: f64,
    end_mjd: f64,
    step_days: Option<f64>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "neo.close_approach", json!({
        "elements": elements,
        "start_mjd": start_mjd,
        "end_mjd": end_mjd,
        "step_days": step_days.unwrap_or(1.0),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_asteroseismology_measure(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
    teff_k: Option<f64>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "asteroseismology.measure", json!({
        "path": path,
        "teff_k": teff_k,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_asteroseismology_solve(
    state: tauri::State<'_, Arc<Engine>>,
    numax_uhz: f64,
    delta_nu_uhz: f64,
    teff_k: f64,
    numax_uhz_error: Option<f64>,
    delta_nu_uhz_error: Option<f64>,
    teff_k_error: Option<f64>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "asteroseismology.solve", json!({
        "numax_uhz": numax_uhz,
        "delta_nu_uhz": delta_nu_uhz,
        "teff_k": teff_k,
        "numax_uhz_error": numax_uhz_error,
        "delta_nu_uhz_error": delta_nu_uhz_error,
        "teff_k_error": teff_k_error,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_technosignature_search(
    state: tauri::State<'_, Arc<Engine>>,
    n_time: Option<u32>,
    n_freq: Option<u32>,
    f0_hz: Option<f64>,
    channel_width_hz: Option<f64>,
    dt_s: Option<f64>,
    drift_rate_hz_s: Option<f64>,
    snr: Option<f64>,
    start_channel: Option<u32>,
    max_drift_hz_s: Option<f64>,
    snr_threshold: Option<f64>,
    seed: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "technosignature.search", json!({
        "n_time": n_time.unwrap_or(16),
        "n_freq": n_freq.unwrap_or(1024),
        "f0_hz": f0_hz.unwrap_or(1.4e9),
        "channel_width_hz": channel_width_hz.unwrap_or(2.7939677),
        "dt_s": dt_s.unwrap_or(18.25),
        "drift_rate_hz_s": drift_rate_hz_s.unwrap_or(0.0),
        "snr": snr.unwrap_or(0.0),
        "start_channel": start_channel,
        "max_drift_hz_s": max_drift_hz_s.unwrap_or(4.0),
        "snr_threshold": snr_threshold.unwrap_or(10.0),
        "seed": seed.unwrap_or(42),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_biosignature_synthesize(
    state: tauri::State<'_, Arc<Engine>>,
    stellar_radius_rsun: f64,
    planet_mass_mjup: f64,
    temperature_k: Option<f64>,
    reference_radius_rjup: Option<f64>,
    abundances: Value,
    cross_sections: Value,
    n_points: Option<u32>,
    wavelength_min_um: Option<f64>,
    wavelength_max_um: Option<f64>,
    error_ppm: Option<f64>,
    seed: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "biosignature.synthesize", json!({
        "stellar_radius_rsun": stellar_radius_rsun,
        "planet_mass_mjup": planet_mass_mjup,
        "temperature_k": temperature_k.unwrap_or(1000.0),
        "reference_radius_rjup": reference_radius_rjup.unwrap_or(1.0),
        "abundances": abundances,
        "cross_sections": cross_sections,
        "n_points": n_points.unwrap_or(40),
        "wavelength_min_um": wavelength_min_um.unwrap_or(1.0),
        "wavelength_max_um": wavelength_max_um.unwrap_or(2.5),
        "error_ppm": error_ppm.unwrap_or(50.0),
        "seed": seed.unwrap_or(42),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_biosignature_fit(
    state: tauri::State<'_, Arc<Engine>>,
    wavelength_um: Vec<f64>,
    depth: Vec<f64>,
    error: Vec<f64>,
    stellar_radius_rsun: f64,
    planet_mass_mjup: f64,
    molecules: Vec<String>,
    cross_sections: Value,
    seed: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "biosignature.fit", json!({
        "wavelength_um": wavelength_um,
        "depth": depth,
        "error": error,
        "stellar_radius_rsun": stellar_radius_rsun,
        "planet_mass_mjup": planet_mass_mjup,
        "molecules": molecules,
        "cross_sections": cross_sections,
        "seed": seed.unwrap_or(42),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_biosignature_detect(
    state: tauri::State<'_, Arc<Engine>>,
    wavelength_um: Vec<f64>,
    depth: Vec<f64>,
    error: Vec<f64>,
    stellar_radius_rsun: f64,
    planet_mass_mjup: f64,
    molecules: Vec<String>,
    cross_sections: Value,
    seed: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "biosignature.detect", json!({
        "wavelength_um": wavelength_um,
        "depth": depth,
        "error": error,
        "stellar_radius_rsun": stellar_radius_rsun,
        "planet_mass_mjup": planet_mass_mjup,
        "molecules": molecules,
        "cross_sections": cross_sections,
        "seed": seed.unwrap_or(42),
    })).await
}


