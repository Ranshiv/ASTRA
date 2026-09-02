//! Experiment records, signed reproducibility bundles, research benchmarks,
//! ablation studies, the Stage-B scale comparison, and the pipeline command.
//!
//! Split out of lib.rs (see commands/mod.rs for why); nothing here changed
//! behavior, only location.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::Engine;

use super::call_blocking;

#[tauri::command]
pub(crate) async fn engine_profile(
    state: tauri::State<'_, Arc<Engine>>,
    limit: Option<u32>,
) -> Result<Value, String> {
    let params = json!({ "limit": limit.unwrap_or(100) });
    call_blocking(Arc::clone(&state), "profile.run", params).await
}


#[tauri::command]
pub(crate) async fn engine_experiments(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "experiment.list", json!({ "project_id": project_id })).await
}


#[tauri::command]
pub(crate) async fn engine_experiment(
    state: tauri::State<'_, Arc<Engine>>,
    experiment_id: String,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "experiment.get",
        json!({ "experiment_id": experiment_id, "project_id": project_id }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_experiment_verify(
    state: tauri::State<'_, Arc<Engine>>,
    experiment_id: String,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "experiment.verify",
        json!({ "experiment_id": experiment_id, "project_id": project_id }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_research_bundle_build(
    state: tauri::State<'_, Arc<Engine>>,
    dataset_id: String,
    experiment_ids: Option<Vec<String>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "research.bundle.build",
        json!({
            "dataset_id": dataset_id,
            "experiment_ids": experiment_ids.unwrap_or_default(),
            "project_id": project_id,
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_research_bundle_verify(
    state: tauri::State<'_, Arc<Engine>>,
    dataset_id: String,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "research.bundle.verify",
        json!({ "dataset_id": dataset_id, "project_id": project_id }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_research_benchmark_run(
    state: tauri::State<'_, Arc<Engine>>,
    matrix_name: String,
    benchmark_id: String,
    split_id: String,
    dataset_id: String,
    injection_fraction: Option<f64>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "research.benchmark.run",
        json!({
            "matrix_name": matrix_name,
            "benchmark_id": benchmark_id,
            "split_id": split_id,
            "dataset_id": dataset_id,
            "injection_fraction": injection_fraction.unwrap_or(0.1),
            "project_id": project_id,
        }),
    ).await
}


#[tauri::command]
pub(crate) async fn engine_experiment_compare(
    state: tauri::State<'_, Arc<Engine>>,
    experiment_ids: Vec<String>,
    metric: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "experiment.compare",
        json!({
            "experiment_ids": experiment_ids,
            "metric": metric.unwrap_or_else(|| "roc_auc".into()),
            "project_id": project_id,
        }),
    ).await
}

/// The ablation suite retrains and rescores repeatedly; minutes, not seconds.

#[tauri::command]
pub(crate) async fn engine_ablation(
    state: tauri::State<'_, Arc<Engine>>,
    fraction: Option<f64>,
    seed: Option<u32>,
    survey: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "fraction": fraction.unwrap_or(0.1),
        "seed": seed.unwrap_or(42),
        "survey": survey,
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "ablation.run", params).await
}

/// Independent seeds turn an injection-recovery snapshot into an uncertainty
/// estimate. This is intentionally a separate, long-running research action.

#[tauri::command]
pub(crate) async fn engine_ablation_repeated(
    state: tauri::State<'_, Arc<Engine>>,
    fraction: Option<f64>,
    seeds: Option<Vec<u32>>,
    survey: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "fraction": fraction.unwrap_or(0.1),
        "seeds": seeds.unwrap_or_else(|| vec![17, 29, 43, 59, 71]),
        "survey": survey,
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "ablation.repeated", params).await
}

/// Stage-B-scale comparison is a deliberate, resumable research action. It
/// defaults to the packaged CPU baseline; deep models require a dev engine.

#[tauri::command]
pub(crate) async fn engine_stageb_compare(
    state: tauri::State<'_, Arc<Engine>>,
    survey: Option<String>,
    seeds: Option<Vec<u32>>,
    fraction: Option<f64>,
    strength: Option<f64>,
    limit: Option<u32>,
    mode: Option<String>,
    include_deep: Option<bool>,
    epochs: Option<u32>,
    checkpoint: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "survey": survey,
        "seeds": seeds.unwrap_or_else(|| vec![17, 29, 43, 59, 71]),
        "fraction": fraction.unwrap_or(0.1),
        "strength": strength.unwrap_or(6.0),
        "limit": limit.unwrap_or(10_000),
        "mode": mode.unwrap_or_else(|| "time".into()),
        "include_deep": include_deep.unwrap_or(false),
        "epochs": epochs.unwrap_or(20),
        "checkpoint": checkpoint,
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "stageb.compare", params).await
}


#[tauri::command]
pub(crate) async fn engine_pipeline(
    state: tauri::State<'_, Arc<Engine>>,
    name: String,
    radius_arcsec: Option<f64>,
    top: Option<u32>,
    anchor_survey: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "name": name,
        "radius_arcsec": radius_arcsec.unwrap_or(15.0),
        "top": top.unwrap_or(200),
        "anchor_survey": anchor_survey,
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "pipeline.run", params).await
}


