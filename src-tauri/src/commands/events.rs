//! Event/alert broker commands: transient event ingestion, clustering,
//! cross-survey association, and public alert-stream polling.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_event_providers(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "events.providers", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_event_ingest(
    state: tauri::State<'_, Arc<Engine>>,
    provider: String,
    payload: Value,
    release: Option<String>,
    packet_id: Option<String>,
    packet_version: Option<String>,
    received_utc: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "events.ingest", json!({
        "provider": provider,
        "payload": payload,
        "release": release,
        "packet_id": packet_id,
        "packet_version": packet_version,
        "received_utc": received_utc,
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_events(
    state: tauri::State<'_, Arc<Engine>>,
    provider: Option<String>,
    event_id: Option<String>,
    limit: Option<u32>,
    packets: Option<bool>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "events.list", json!({
        "provider": provider,
        "event_id": event_id,
        "limit": limit.unwrap_or(500),
        "packets": packets.unwrap_or(false),
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_event_packet(
    state: tauri::State<'_, Arc<Engine>>,
    packet_key: String,
    include_raw: Option<bool>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "events.get", json!({
        "packet_key": packet_key,
        "include_raw": include_raw.unwrap_or(false),
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_event_replay(
    state: tauri::State<'_, Arc<Engine>>,
    provider: Option<String>,
    event_id: Option<String>,
    limit: Option<u32>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "events.replay", json!({
        "provider": provider,
        "event_id": event_id,
        "limit": limit.unwrap_or(100),
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_event_associate(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    provider: Option<String>,
    event_id: Option<String>,
    radius_arcsec: Option<f64>,
    window_days: Option<f64>,
    allow_unknown_time: Option<bool>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "events.associate", json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "provider": provider,
        "event_id": event_id,
        "radius_arcsec": radius_arcsec.unwrap_or(30.0),
        "window_days": window_days.unwrap_or(30.0),
        "allow_unknown_time": allow_unknown_time.unwrap_or(false),
        "project_id": project_id,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_alert_providers(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "alerts.providers", json!({})).await
}


#[tauri::command]
pub(crate) async fn engine_alert_status(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "alerts.status", json!({ "project_id": project_id })).await
}


#[tauri::command]
pub(crate) async fn engine_alert_poll(
    state: tauri::State<'_, Arc<Engine>>,
    provider: String,
    endpoint: Option<String>,
    cursor: Option<String>,
    limit: Option<u32>,
    offline: Option<bool>,
    payload: Option<Value>,
    params: Option<Value>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "alerts.poll", json!({
        "provider": provider,
        "endpoint": endpoint,
        "cursor": cursor,
        "limit": limit.unwrap_or(100),
        "offline": offline.unwrap_or(false),
        "payload": payload,
        "params": params,
        "project_id": project_id,
    })).await
}


