//! TAP queries, calibrated significance/selection, and the review queue.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_tap_status(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "tap.status", json!({ "project_id": project_id })).await
}


#[tauri::command]
pub(crate) async fn engine_tap_query(
    state: tauri::State<'_, Arc<Engine>>,
    service: String,
    adql: String,
    release: Option<String>,
    max_rows: Option<u32>,
    format: Option<String>,
    refresh: Option<bool>,
    offline: Option<bool>,
    timeout: Option<f64>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "tap.query", json!({
        "service": service,
        "adql": adql,
        "release": release.unwrap_or_else(|| "unknown".into()),
        "max_rows": max_rows.unwrap_or(200),
        "format": format.unwrap_or_else(|| "csv".into()),
        "refresh": refresh.unwrap_or(false),
        "offline": offline.unwrap_or(false),
        "timeout": timeout.unwrap_or(60.0),
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_significance_calibrate(
    state: tauri::State<'_, Arc<Engine>>,
    scores: Vec<f64>,
    reference_scores: Option<Vec<f64>>,
    threshold: Option<f64>,
    strata: Option<Value>,
    name: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "significance.calibrate", json!({
        "scores": scores,
        "reference_scores": reference_scores,
        "threshold": threshold,
        "strata": strata,
        "name": name.unwrap_or_else(|| "default".into()),
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_selection_evaluate(
    state: tauri::State<'_, Arc<Engine>>,
    records: Vec<Value>,
    dimensions: Option<Vec<String>>,
    edges: Option<Value>,
    fit_model: Option<bool>,
    model_features: Option<Vec<String>>,
    bootstrap_samples: Option<u32>,
    seed: Option<u64>,
    name: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "selection.evaluate", json!({
        "records": records,
        "dimensions": dimensions,
        "edges": edges,
        "fit_model": fit_model.unwrap_or(false),
        "model_features": model_features,
        "bootstrap_samples": bootstrap_samples.unwrap_or(0),
        "seed": seed.unwrap_or(42),
        "name": name.unwrap_or_else(|| "default".into()),
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_review_next(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    limit: Option<u32>,
    project_id: Option<String>,
    active: Option<bool>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "review.next", json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "limit": limit.unwrap_or(20),
        "project_id": project_id,
        "active": active.unwrap_or(false),
    })).await
}


/// Supervised-review metrics over the labels recorded so far. Cheap: it reads
/// the stored label file rather than refitting anything.
#[tauri::command]
pub(crate) async fn engine_candidates_evaluate(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.evaluate",
        json!({ "name": name.unwrap_or_else(|| "default".into()), "project_id": project_id }),
    ).await
}

