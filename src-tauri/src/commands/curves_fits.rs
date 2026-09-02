//! Light-curve retrieval/folding/binning, FITS metadata/pixel access, image
//! and spectral feature extraction, and sidecar-file bookkeeping.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_curves_list(
    state: tauri::State<'_, Arc<Engine>>,
    survey: Option<String>,
    limit: Option<u32>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "curves.list",
        json!({ "survey": survey, "limit": limit.unwrap_or(500), "project_id": project_id }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_curve_get(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
    max_points: Option<u32>,
    frame: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "curves.get",
        json!({ "path": path, "max_points": max_points.unwrap_or(2000), "frame": frame }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_curve_fold(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
    period_days: f64,
    epoch: Option<f64>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "curves.fold",
        json!({ "path": path, "period_days": period_days, "epoch": epoch }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_curve_bin(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
    bin_days: f64,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "curves.bin",
        json!({ "path": path, "bin_days": bin_days }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_fits_describe(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "fits.describe", json!({ "path": path })).await
}


#[tauri::command]
pub(crate) async fn engine_fits_header(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
    hdu: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "fits.header",
        json!({ "path": path, "hdu": hdu.unwrap_or(0) }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_fits_image(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
    hdu: Option<u32>,
    contrast: Option<f64>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "fits.image",
        json!({ "path": path, "hdu": hdu, "contrast": contrast.unwrap_or(0.25) }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_image_features(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
    hdu: Option<u32>,
    target_x: Option<f64>,
    target_y: Option<f64>,
    project_id: Option<String>,
    survey: Option<String>,
    release: Option<String>,
    object_id: Option<String>,
    band: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "image.features", json!({
        "path": path,
        "hdu": hdu,
        "target_x": target_x,
        "target_y": target_y,
        "project_id": project_id,
        "survey": survey, "release": release, "object_id": object_id, "band": band,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_spectral_features(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
    project_id: Option<String>,
    survey: Option<String>,
    release: Option<String>,
    object_id: Option<String>,
    band: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "spectrum.features", json!({
        "path": path,
        "project_id": project_id,
        "survey": survey, "release": release, "object_id": object_id, "band": band,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_sidecars_list(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "sidecars.list", json!({ "project_id": project_id })).await
}


#[tauri::command]
pub(crate) async fn engine_sidecar_save(
    state: tauri::State<'_, Arc<Engine>>,
    kind: String,
    payloads: Vec<Value>,
    name: Option<String>,
    project_id: Option<String>,
    identities: Option<Vec<Value>>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "sidecars.save", json!({
        "kind": kind, "payloads": payloads,
        "name": name.unwrap_or_else(|| "default".into()),
        "project_id": project_id, "identities": identities,
    })).await
}


#[tauri::command]
pub(crate) async fn engine_sidecar_join(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
    kind: String,
    identities: Vec<Value>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "sidecars.join", json!({
        "path": path, "kind": kind, "identities": identities,
    })).await
}


