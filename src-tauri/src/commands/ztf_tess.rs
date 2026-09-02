//! ZTF cutout search/download and TESS target-pixel-file download/photometry.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::{call, call_blocking};

#[tauri::command]
pub(crate) async fn engine_ztf_images_search(
    state: tauri::State<'_, Arc<Engine>>,
    ra_deg: f64,
    dec_deg: f64,
    size_arcsec: Option<f64>,
    product_kind: Option<String>,
    release: Option<String>,
    limit: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "ztf.images.search",
        json!({
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "size_arcsec": size_arcsec.unwrap_or(50.0),
            "product_kind": product_kind.unwrap_or_else(|| "science".into()),
            "release": release.unwrap_or_else(|| "dr".into()),
            "limit": limit.unwrap_or(25),
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_ztf_images_download(
    state: tauri::State<'_, Arc<Engine>>,
    ra_deg: f64,
    dec_deg: f64,
    metadata: Value,
    size_arcsec: Option<f64>,
    product_kind: Option<String>,
    release: Option<String>,
    project_id: Option<String>,
    max_bytes: Option<u64>,
) -> Result<Value, String> {
    let params = json!({
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "metadata": metadata,
        "size_arcsec": size_arcsec.unwrap_or(50.0),
        "product_kind": product_kind.unwrap_or_else(|| "science".into()),
        "release": release.unwrap_or_else(|| "dr".into()),
        "project_id": project_id,
        "max_bytes": max_bytes.unwrap_or(256 * 1024 * 1024),
    });
    let submitted = call(
        &state,
        "job.submit",
        json!({
            "method": "ztf.images.download",
            "params": params,
            "project_id": project_id,
        }),
    )?;
    Ok(submitted)
}

/// TPF acquisition is an explicit candidate-scale transfer; keep it in the
/// persistent job system so a Tauri restart or a transient MAST failure does
/// not turn a long download into an invisible partial result.

#[tauri::command]
pub(crate) async fn engine_tess_tpf_download(
    state: tauri::State<'_, Arc<Engine>>,
    ra_deg: f64,
    dec_deg: f64,
    sector: i64,
    size_pixels: Option<i64>,
    target_id: Option<String>,
    product: Option<String>,
    project_id: Option<String>,
    max_bytes: Option<u64>,
    overwrite: Option<bool>,
) -> Result<Value, String> {
    let params = json!({
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "sector": sector,
        "size_pixels": size_pixels.unwrap_or(20),
        "target_id": target_id,
        "product": product.unwrap_or_else(|| "SPOC".into()),
        "project_id": project_id,
        "max_bytes": max_bytes.unwrap_or(128 * 1024 * 1024),
        "overwrite": overwrite.unwrap_or(false),
    });
    call(
        &state,
        "job.submit",
        json!({
            "method": "tess.tpf.download",
            "params": params,
            "project_id": project_id,
        }),
    )
}


#[tauri::command]
pub(crate) async fn engine_tess_tpf_photometry(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
    ra_deg: f64,
    dec_deg: f64,
    neighbors: Option<Value>,
    target_mag: Option<f64>,
    aperture_radius_pixels: Option<f64>,
    quality_mask: Option<u64>,
    target_id: Option<String>,
    persist: Option<bool>,
    max_points: Option<u32>,
) -> Result<Value, String> {
    let params = json!({
        "path": path,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "neighbors": neighbors.unwrap_or_else(|| json!([])),
        "target_mag": target_mag,
        "aperture_radius_pixels": aperture_radius_pixels.unwrap_or(1.5),
        "quality_mask": quality_mask.unwrap_or(0),
        "target_id": target_id,
        "persist": persist.unwrap_or(true),
        "max_points": max_points.unwrap_or(5000),
    });
    call_blocking(Arc::clone(&state), "tess.tpf.photometry", params).await
}


