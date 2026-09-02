//! Candidate loading, spatial view, per-candidate detail/timeline/explain,
//! export, broadcast, and the review label/vote surface.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_candidates(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    top: Option<u32>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.load",
        json!({
            "name": name.unwrap_or_else(|| "default".into()),
            "top": top.unwrap_or(50),
            "project_id": project_id,
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_candidates_spatial(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    top: Option<u32>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.spatial",
        json!({
            "name": name.unwrap_or_else(|| "default".into()),
            "top": top.unwrap_or(200),
            "project_id": project_id,
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_candidate(
    state: tauri::State<'_, Arc<Engine>>,
    candidate_id: String,
    name: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.get",
        json!({
            "candidate_id": candidate_id,
            "name": name.unwrap_or_else(|| "default".into()),
            "project_id": project_id,
        }),
    ).await
}

/// Feature attribution reruns the anomaly ensemble once per feature, so
/// this is a real, multi-second research action, not a refresh.

#[tauri::command]
pub(crate) async fn engine_candidate_explain(
    state: tauri::State<'_, Arc<Engine>>,
    candidate_id: String,
    name: Option<String>,
    project_id: Option<String>,
    top: Option<u32>,
    stable: Option<bool>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.explain",
        json!({
            "candidate_id": candidate_id,
            "name": name.unwrap_or_else(|| "default".into()),
            "project_id": project_id,
            "top": top.unwrap_or(10),
            "stable": stable.unwrap_or(false),
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_candidate_timeline(
    state: tauri::State<'_, Arc<Engine>>,
    candidate_id: String,
    name: Option<String>,
    project_id: Option<String>,
    radius_arcsec: Option<f64>,
    max_curves: Option<u32>,
    max_points: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "candidates.timeline", json!({
        "candidate_id": candidate_id,
        "name": name.unwrap_or_else(|| "default".into()),
        "project_id": project_id,
        "radius_arcsec": radius_arcsec.unwrap_or(30.0),
        "max_curves": max_curves.unwrap_or(24),
        "max_points": max_points.unwrap_or(180),
    })).await
}


#[tauri::command]
pub(crate) async fn engine_candidates_export(
    state: tauri::State<'_, Arc<Engine>>,
    format: String,
    name: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.export",
        json!({
            "format": format,
            "name": name.unwrap_or_else(|| "default".into()),
            "project_id": project_id,
        }),
    ).await
}

/// Writes a LOCAL feed file (not a network push -- see broadcast.py's
/// module docstring); overwrites the same stable path on each call.

#[tauri::command]
pub(crate) async fn engine_candidate_broadcast(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    threshold: Option<f64>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.broadcast",
        json!({
            "name": name.unwrap_or_else(|| "default".into()),
            "threshold": threshold.unwrap_or(0.5),
            "project_id": project_id,
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_label(
    state: tauri::State<'_, Arc<Engine>>,
    candidate_id: String,
    label: String,
    note: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.label",
        json!({
            "candidate_id": candidate_id,
            "label": label,
            "note": note.unwrap_or_default(),
            "project_id": project_id,
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_label_summary(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "candidates.labels", json!({"project_id": project_id})).await
}

/// Casts one independent reviewer's vote (candidates.vote). Unlike
/// engine_label, this never overwrites a prior vote -- append-only.

#[tauri::command]
pub(crate) async fn engine_candidate_vote(
    state: tauri::State<'_, Arc<Engine>>,
    candidate_id: String,
    reviewer_id: String,
    label: String,
    note: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.vote",
        json!({
            "candidate_id": candidate_id,
            "reviewer_id": reviewer_id,
            "label": label,
            "note": note.unwrap_or_default(),
            "project_id": project_id,
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_candidate_votes(
    state: tauri::State<'_, Arc<Engine>>,
    candidate_id: String,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.votes",
        json!({ "candidate_id": candidate_id, "project_id": project_id }),
    ).await
}

/// Promotes the current vote majority to the authoritative label -- always
/// an explicit action, never triggered automatically by a vote cast.

#[tauri::command]
pub(crate) async fn engine_candidate_vote_promote(
    state: tauri::State<'_, Arc<Engine>>,
    candidate_id: String,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.vote_promote",
        json!({ "candidate_id": candidate_id, "project_id": project_id }),
    ).await
}

