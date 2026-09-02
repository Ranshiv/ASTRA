//! Project lifecycle commands: manifest listing and project create/list/
//! open/update/archive/validate.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_manifests(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "manifest.list", json!({ "project_id": project_id })).await
}


#[tauri::command]
pub(crate) async fn engine_projects(
    state: tauri::State<'_, Arc<Engine>>,
    include_archived: Option<bool>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "project.list",
        json!({ "include_archived": include_archived.unwrap_or(true) }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_project_create(
    state: tauri::State<'_, Arc<Engine>>,
    name: String,
    project_id: Option<String>,
    description: Option<String>,
    selected_surveys: Option<Vec<String>>,
    query_regions: Option<Value>,
    tags: Option<Vec<String>>,
    data_root: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "project.create",
        json!({
            "name": name,
            "project_id": project_id,
            "description": description,
            "selected_surveys": selected_surveys,
            "query_regions": query_regions,
            "tags": tags,
            "data_root": data_root,
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_project_open(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "project.open", json!({ "project_id": project_id })).await
}


#[tauri::command]
pub(crate) async fn engine_project_update(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: String,
    patch: Value,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "project.update",
        json!({ "project_id": project_id, "patch": patch }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_project_archive(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: String,
    archived: Option<bool>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "project.archive",
        json!({ "project_id": project_id, "archived": archived.unwrap_or(true) }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_project_validate(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "project.validate", json!({ "project_id": project_id })).await
}


