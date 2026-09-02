//! Public catalogue cross-reference, gravitational-wave and FRB coincidence
//! checks, and Gaia DR4 epoch ingestion.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_catalog_status(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "catalog.status", json!({})).await
}

/// Catalog enrichment performs optional network queries, so it runs off the
/// UI thread and never blocks normal candidate generation.

#[tauri::command]
pub(crate) async fn engine_catalog_enrich(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    radius_arcsec: Option<f64>,
    refresh: Option<bool>,
    offline: Option<bool>,
    include_tns: Option<bool>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "radius_arcsec": radius_arcsec.unwrap_or(2.0),
        "refresh": refresh.unwrap_or(false),
        "offline": offline.unwrap_or(false),
        "include_tns": include_tns.unwrap_or(true),
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "catalog.enrich", params).await
}


#[tauri::command]
pub(crate) async fn engine_gw_events(
    state: tauri::State<'_, Arc<Engine>>,
    catalog: Option<String>,
    refresh: Option<bool>,
    offline: Option<bool>,
) -> Result<Value, String> {
    let params = json!({
        "catalog": catalog.unwrap_or_else(|| "GWTC-1-confident".into()),
        "refresh": refresh.unwrap_or(false),
        "offline": offline.unwrap_or(false),
    });
    call_blocking(Arc::clone(&state), "gw.events", params).await
}


#[tauri::command]
pub(crate) async fn engine_gw_enrich(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    catalog: Option<String>,
    window_days: Option<f64>,
    refresh: Option<bool>,
    offline: Option<bool>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "catalog": catalog.unwrap_or_else(|| "GWTC-1-confident".into()),
        "window_days": window_days.unwrap_or(30.0),
        "refresh": refresh.unwrap_or(false),
        "offline": offline.unwrap_or(false),
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "gw.enrich", params).await
}


#[tauri::command]
pub(crate) async fn engine_frb_events(
    state: tauri::State<'_, Arc<Engine>>,
    refresh: Option<bool>,
    offline: Option<bool>,
) -> Result<Value, String> {
    let params = json!({
        "refresh": refresh.unwrap_or(false),
        "offline": offline.unwrap_or(false),
    });
    call_blocking(Arc::clone(&state), "frb.events", params).await
}


#[tauri::command]
pub(crate) async fn engine_frb_enrich(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    window_days: Option<f64>,
    sigma_threshold: Option<f64>,
    refresh: Option<bool>,
    offline: Option<bool>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "window_days": window_days.unwrap_or(1.0),
        "sigma_threshold": sigma_threshold.unwrap_or(3.0),
        "refresh": refresh.unwrap_or(false),
        "offline": offline.unwrap_or(false),
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "frb.enrich", params).await
}


#[tauri::command]
pub(crate) async fn engine_gaia_epoch_ingest(
    state: tauri::State<'_, Arc<Engine>>,
    fixture_path: String,
    checkpoint: Option<String>,
    name: Option<String>,
    batch_size: Option<u32>,
) -> Result<Value, String> {
    let params = json!({
        "fixture_path": fixture_path,
        "checkpoint": checkpoint,
        "name": name.unwrap_or_else(|| "default".into()),
        "batch_size": batch_size.unwrap_or(256),
    });
    call_blocking(Arc::clone(&state), "gaia.epoch_ingest", params).await
}


#[tauri::command]
pub(crate) async fn engine_gaia_epoch_status(
    state: tauri::State<'_, Arc<Engine>>,
    checkpoint: Option<String>,
    name: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "checkpoint": checkpoint,
        "name": name.unwrap_or_else(|| "default".into()),
    });
    call_blocking(Arc::clone(&state), "gaia.epoch_status", params).await
}


