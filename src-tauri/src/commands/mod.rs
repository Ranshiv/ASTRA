//! Tauri command handlers, grouped by domain.
//!
//! `lib.rs` previously held all 128 `#[tauri::command]` functions in one
//! file; every one of them is a thin wrapper that builds a JSON payload and
//! calls `call_blocking` below, so splitting by domain (mirroring the
//! Python engine's own `rpc_handlers/` split) needed no real restructuring,
//! only relocation.
//!
//! `#[tauri::command]` generates a hidden `__cmd__<name>!` macro alongside
//! each function that `generate_handler!` needs, and that macro only
//! resolves through an explicit, fully-qualified path -- a glob `use
//! module::*;` re-export brings the function itself into scope but silently
//! leaves the hidden macro unresolved. So `lib.rs`'s `generate_handler!`
//! list references each command as `commands::<domain>::<name>` rather than
//! importing them here at all; these submodules are `pub(crate)` for that.

use std::sync::Arc;

use serde_json::Value;

use crate::engine::Engine;

/// Errors are returned to the UI rather than panicking, so a missing or
/// crashed engine renders as a banner instead of a blank window.
pub(crate) fn call(engine: &Engine, method: &str, params: Value) -> Result<Value, String> {
    let response = engine.request(method, params)?;
    if response.ok {
        Ok(response.result.unwrap_or(Value::Null))
    } else {
        Err(response
            .error
            .unwrap_or_else(|| "unknown engine error".into()))
    }
}

/// Archive queries take minutes. Running one on the UI thread would freeze the
/// window, so slow commands hand off to the blocking pool.
pub(crate) async fn call_blocking(
    engine: Arc<Engine>,
    method: &'static str,
    params: Value,
) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call(&engine, method, params))
        .await
        .map_err(|e| format!("engine task failed: {e}"))?
}

pub(crate) mod candidates;
pub(crate) mod catalog_gw_frb;
pub(crate) mod core;
pub(crate) mod credentials_ranker;
pub(crate) mod crossmatch_deep;
pub(crate) mod curves_fits;
pub(crate) mod events;
pub(crate) mod experiments_research;
pub(crate) mod followup_schedule;
pub(crate) mod frontier;
pub(crate) mod projects;
pub(crate) mod tap_review;
pub(crate) mod ztf_tess;
