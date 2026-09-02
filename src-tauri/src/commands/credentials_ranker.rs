//! TNS credential configuration and the supervised ranker (train/apply/list
//! saved models).
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_tns_credentials_configure(
    state: tauri::State<'_, Arc<Engine>>,
    api_key: String,
    bot_id: Option<String>,
    bot_name: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "credentials.tns.configure",
        json!({
            "api_key": api_key,
            "bot_id": bot_id.unwrap_or_default(),
            "bot_name": bot_name.unwrap_or_else(|| "ASTRA".into()),
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_tns_credentials_clear(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "credentials.tns.clear", json!({})).await
}

/// Grouped evaluation, bootstrapping, and fitting can take noticeable time.

#[tauri::command]
pub(crate) async fn engine_ranker_train(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    model_name: Option<String>,
    seed: Option<u32>,
) -> Result<Value, String> {
    let params = json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "model_name": model_name.unwrap_or_else(|| "calibrated-logistic".into()),
        "seed": seed.unwrap_or(42),
    });
    call_blocking(Arc::clone(&state), "ranker.train", params).await
}


#[tauri::command]
pub(crate) async fn engine_ranker_apply(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    model_name: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "model_name": model_name.unwrap_or_else(|| "calibrated-logistic".into()),
    });
    call_blocking(Arc::clone(&state), "ranker.apply", params).await
}


#[tauri::command]
pub(crate) async fn engine_ranker_list(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "ranker.list", json!({})).await
}


