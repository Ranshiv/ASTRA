//! Core engine commands: hardware/paths/cache/versions, survey discovery,
//! acquisition, product listing, and the baseline feature/anomaly pipeline.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_ping(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "ping", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_hardware(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "hardware", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_paths(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "paths", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_versions(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "versions", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_cache_status(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "cache.status", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_cache_enforce(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "cache.enforce", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_surveys(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "surveys.list", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_readiness(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "readiness.status", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_store_usage(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "store.usage", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_acquire(
    state: tauri::State<'_, Arc<Engine>>,
    ra_deg: f64,
    dec_deg: f64,
    radius_arcsec: f64,
    surveys: Vec<String>,
    limit: u32,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "radius_arcsec": radius_arcsec,
        "surveys": surveys,
        "limit": limit,
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "acquire.cone", params).await
}


#[tauri::command]
pub(crate) async fn engine_products(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
    limit: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "products.list", json!({
        "project_id": project_id,
        "limit": limit.unwrap_or(500),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_product(
    state: tauri::State<'_, Arc<Engine>>,
    product_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "products.get", json!({ "product_id": product_id })).await
}


#[tauri::command]
pub(crate) async fn engine_features_list(state: tauri::State<'_, Arc<Engine>>, project_id: Option<String>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "features.list", json!({ "project_id": project_id })).await
}

/// Feature extraction runs a Lomb-Scargle search per curve, so it is slow
/// enough to need the blocking pool.

#[tauri::command]
pub(crate) async fn engine_features_build(
    state: tauri::State<'_, Arc<Engine>>,
    name: String,
    survey: Option<String>,
    limit: Option<u32>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "name": name, "survey": survey, "limit": limit.unwrap_or(10_000), "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "features.build", params).await
}


#[tauri::command]
pub(crate) async fn engine_features_build_resumable(
    state: tauri::State<'_, Arc<Engine>>,
    name: String,
    survey: Option<String>,
    limit: Option<u32>,
    batch_size: Option<u32>,
    checkpoint: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "name": name,
        "survey": survey,
        "limit": limit.unwrap_or(10_000),
        "batch_size": batch_size.unwrap_or(256),
        "checkpoint": checkpoint,
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "features.build_resumable", params).await
}


#[tauri::command]
pub(crate) async fn engine_detect(
    state: tauri::State<'_, Arc<Engine>>,
    name: String,
    contamination: Option<f64>,
    top: Option<u32>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "name": name,
        "contamination": contamination.unwrap_or(0.05),
        "top": top.unwrap_or(50),
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "anomaly.detect", params).await
}


#[tauri::command]
pub(crate) async fn engine_feature_cache_clear(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "cache.features.clear", json!({})).await
}


/// The ordered feature-column names, so the UI can label a feature vector
/// without hardcoding a list that would silently drift from the engine's.
#[tauri::command]
pub(crate) async fn engine_feature_names(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "features.names", json!({})).await
}


