//! Follow-up planning/tracking, the discard-pile scan, literature search,
//! physical (SED) characterization, and digital-twin transfer scoring.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_followup_plan(
    state: tauri::State<'_, Arc<Engine>>,
    ra_deg: f64,
    dec_deg: f64,
    start_utc: Option<String>,
    duration_hours: Option<f64>,
    latitude_deg: Option<f64>,
    longitude_deg: Option<f64>,
    min_altitude_deg: Option<f64>,
    cadence_minutes: Option<u32>,
    target_id: Option<String>,
    twilight_sun_altitude_deg: Option<f64>,
    min_moon_separation_deg: Option<f64>,
    max_moon_illumination: Option<f64>,
    max_airmass: Option<f64>,
    weather: Option<Value>,
    facility_name: Option<String>,
    facility_constraints: Option<Value>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "followup.plan", json!({
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "start_utc": start_utc,
        "duration_hours": duration_hours.unwrap_or(12.0),
        "latitude_deg": latitude_deg.unwrap_or(43.65),
        "longitude_deg": longitude_deg.unwrap_or(-79.38),
        "min_altitude_deg": min_altitude_deg.unwrap_or(30.0),
        "cadence_minutes": cadence_minutes.unwrap_or(10),
        "target_id": target_id,
        "twilight_sun_altitude_deg": twilight_sun_altitude_deg.unwrap_or(-18.0),
        "min_moon_separation_deg": min_moon_separation_deg.unwrap_or(0.0),
        "max_moon_illumination": max_moon_illumination.unwrap_or(1.0),
        "max_airmass": max_airmass,
        "weather": weather,
        "facility_name": facility_name,
        "facility_constraints": facility_constraints,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_discard_scan(
    state: tauri::State<'_, Arc<Engine>>,
    object_id: String,
    ra_deg: f64,
    dec_deg: f64,
    min_run_length: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "discard.scan", json!({
        "object_id": object_id,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        // Mirrors discard_pile.DEFAULT_MIN_RUN_LENGTH (3); a literal `null`
        // would reach params.get("min_run_length", default) as an explicit
        // None, not fall through to the Python side's own default.
        "min_run_length": min_run_length.unwrap_or(3),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_followup_request(
    state: tauri::State<'_, Arc<Engine>>,
    candidate_id: String,
    facility_name: Option<String>,
    note: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "followup.request", json!({
        "candidate_id": candidate_id,
        "facility_name": facility_name.unwrap_or_default(),
        "note": note.unwrap_or_default(),
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_followup_result(
    state: tauri::State<'_, Arc<Engine>>,
    request_id: String,
    status: String,
    note: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "followup.result", json!({
        "request_id": request_id,
        "status": status,
        "note": note.unwrap_or_default(),
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_followup_history(
    state: tauri::State<'_, Arc<Engine>>,
    candidate_id: String,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "followup.history", json!({
        "candidate_id": candidate_id,
        "project_id": project_id,
    })).await
}

/// Profiling runs the real stages, so it takes as long as they do.

#[tauri::command]
pub(crate) async fn engine_literature_status(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "literature.status", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_literature_search(
    state: tauri::State<'_, Arc<Engine>>,
    object_id: Option<String>,
    terms: Option<Vec<String>>,
    event_ids: Option<Vec<String>>,
    providers: Option<Vec<String>>,
    limit: Option<u32>,
    refresh: Option<bool>,
    offline: Option<bool>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "literature.search", json!({
        "object_id": object_id.unwrap_or_default(),
        "terms": terms.unwrap_or_default(),
        "event_ids": event_ids.unwrap_or_default(),
        "providers": providers.unwrap_or_else(|| vec!["ads".into(), "arxiv".into()]),
        "limit": limit.unwrap_or(20),
        "refresh": refresh.unwrap_or(false),
        "offline": offline.unwrap_or(false),
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_literature_enrich(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    refresh: Option<bool>,
    offline: Option<bool>,
    include_arxiv: Option<bool>,
    limit: Option<u32>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "literature.enrich", json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "refresh": refresh.unwrap_or(false),
        "offline": offline.unwrap_or(false),
        "include_arxiv": include_arxiv.unwrap_or(true),
        "limit": limit.unwrap_or(20),
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_physical_characterize(
    state: tauri::State<'_, Arc<Engine>>,
    photometry: Value,
    extinction: Option<Value>,
    source: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "physical.characterize", json!({
        "photometry": photometry,
        "extinction": extinction,
        "source": source.unwrap_or_else(|| "caller".into()),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_physical_enrich(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    extinction: Option<Value>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "physical.enrich", json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "extinction": extinction,
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_digital_twin_fit_profile(
    state: tauri::State<'_, Arc<Engine>>,
    survey: String,
    limit: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "digital_twin.fit_profile", json!({
        "survey": survey,
        "limit": limit.unwrap_or(500),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_digital_twin_sample(
    state: tauri::State<'_, Arc<Engine>>,
    survey: String,
    limit: Option<u32>,
    n: Option<u32>,
    seed: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "digital_twin.sample", json!({
        "survey": survey,
        "limit": limit.unwrap_or(500),
        "n": n.unwrap_or(50),
        "seed": seed.unwrap_or(42),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_digital_twin_evaluate_distance(
    state: tauri::State<'_, Arc<Engine>>,
    survey: String,
    limit: Option<u32>,
    seed: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "digital_twin.evaluate_distance", json!({
        "survey": survey,
        "limit": limit.unwrap_or(500),
        "seed": seed.unwrap_or(42),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_digital_twin_evaluate_transfer(
    state: tauri::State<'_, Arc<Engine>>,
    survey: String,
    limit: Option<u32>,
    seeds: Option<Vec<u32>>,
    epochs: Option<u32>,
    fraction: Option<f64>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "digital_twin.evaluate_transfer", json!({
        "survey": survey,
        "limit": limit.unwrap_or(500),
        "seeds": seeds.unwrap_or_else(|| vec![17, 29, 43]),
        "epochs": epochs.unwrap_or(15),
        "fraction": fraction.unwrap_or(0.1),
    })).await
}


