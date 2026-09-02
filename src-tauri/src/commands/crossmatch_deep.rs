//! Cross-survey matching/profiling, the frame-offset measurement, deep-model
//! train/compare/sweep, and the background job queue.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_job_submit(
    state: tauri::State<'_, Arc<Engine>>,
    method: String,
    params: Option<Value>,
    project_id: Option<String>,
    idempotency_key: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "job.submit",
        json!({
            "method": method,
            "params": params.unwrap_or(json!({})),
            "project_id": project_id,
            "idempotency_key": idempotency_key,
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_job_status(
    state: tauri::State<'_, Arc<Engine>>,
    job_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "job.status", json!({ "job_id": job_id })).await
}


#[tauri::command]
pub(crate) async fn engine_job_cancel(
    state: tauri::State<'_, Arc<Engine>>,
    job_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "job.cancel", json!({ "job_id": job_id })).await
}


#[tauri::command]
pub(crate) async fn engine_job_retry(
    state: tauri::State<'_, Arc<Engine>>,
    job_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "job.retry", json!({ "job_id": job_id })).await
}


#[tauri::command]
pub(crate) async fn engine_jobs(
    state: tauri::State<'_, Arc<Engine>>,
    statuses: Option<Vec<String>>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "job.list", json!({ "statuses": statuses })).await
}

/// Full candidate generation re-reads every curve and runs every detector.

#[tauri::command]
pub(crate) async fn engine_frame_offset(
    state: tauri::State<'_, Arc<Engine>>,
    ra_deg: f64,
    dec_deg: f64,
    time_system: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "timeframe.offset",
        json!({
            "ra_deg": ra_deg, "dec_deg": dec_deg,
            "time_system": time_system.unwrap_or_else(|| "HJD_UTC".into()),
        }),
    ).await
}

/// Cross-matching re-reads every stored curve, so it goes to the blocking pool.

#[tauri::command]
pub(crate) async fn engine_crossmatch(
    state: tauri::State<'_, Arc<Engine>>,
    radius_arcsec: Option<f64>,
    anchor_survey: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "radius_arcsec": radius_arcsec.unwrap_or(2.0),
        "anchor_survey": anchor_survey,
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "crossmatch.run", params).await
}


#[tauri::command]
pub(crate) async fn engine_profiles(
    state: tauri::State<'_, Arc<Engine>>,
    radius_arcsec: Option<f64>,
    top: Option<u32>,
    anchor_survey: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "radius_arcsec": radius_arcsec.unwrap_or(2.0),
        "top": top.unwrap_or(25),
        "anchor_survey": anchor_survey,
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "crossmatch.profile", params).await
}

/// Training runs for minutes and must never block the UI thread.

#[tauri::command]
pub(crate) async fn engine_deep_train(
    state: tauri::State<'_, Arc<Engine>>,
    name: String,
    kind: Option<String>,
    survey: Option<String>,
    epochs: Option<u32>,
) -> Result<Value, String> {
    let params = json!({
        "name": name,
        "kind": kind.unwrap_or_else(|| "autoencoder".into()),
        "survey": survey,
        "epochs": epochs.unwrap_or(30),
    });
    call_blocking(Arc::clone(&state), "deep.train", params).await
}


#[tauri::command]
pub(crate) async fn engine_deep_compare(
    state: tauri::State<'_, Arc<Engine>>,
    survey: Option<String>,
    fraction: Option<f64>,
    epochs: Option<u32>,
) -> Result<Value, String> {
    let params = json!({
        "survey": survey,
        "fraction": fraction.unwrap_or(0.1),
        "epochs": epochs.unwrap_or(20),
    });
    call_blocking(Arc::clone(&state), "deep.compare", params).await
}


/// A hyperparameter sweep trains one model per configuration per seed, so it is
/// the longest-running action the engine exposes.
#[tauri::command]
pub(crate) async fn engine_deep_sweep(
    state: tauri::State<'_, Arc<Engine>>,
    kind: Option<String>,
    survey: Option<String>,
    seeds: Option<Vec<u32>>,
    epochs: Option<u32>,
    mode: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "kind": kind.unwrap_or_else(|| "autoencoder".into()),
        "survey": survey,
        "seeds": seeds.unwrap_or_else(|| vec![17, 29, 43]),
        "epochs": epochs.unwrap_or(20),
        "mode": mode.unwrap_or_else(|| "time".into()),
    });
    call_blocking(Arc::clone(&state), "deep.sweep", params).await
}

