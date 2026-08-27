mod engine;

use std::sync::Arc;

use serde_json::{json, Value};
use tauri::Manager;

use engine::Engine;

/// Errors are returned to the UI rather than panicking, so a missing or
/// crashed engine renders as a banner instead of a blank window.
fn call(engine: &Engine, method: &str, params: Value) -> Result<Value, String> {
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
async fn call_blocking(
    engine: Arc<Engine>,
    method: &'static str,
    params: Value,
) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call(&engine, method, params))
        .await
        .map_err(|e| format!("engine task failed: {e}"))?
}

#[tauri::command]
async fn engine_ping(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "ping", json!({})).await
}

#[tauri::command]
async fn engine_hardware(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "hardware", json!({})).await
}

#[tauri::command]
async fn engine_paths(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "paths", json!({})).await
}

#[tauri::command]
async fn engine_versions(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "versions", json!({})).await
}

#[tauri::command]
async fn engine_cache_status(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "cache.status", json!({})).await
}

#[tauri::command]
async fn engine_cache_enforce(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "cache.enforce", json!({})).await
}

#[tauri::command]
async fn engine_surveys(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "surveys.list", json!({})).await
}

#[tauri::command]
async fn engine_event_providers(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "events.providers", json!({})).await
}

#[tauri::command]
async fn engine_event_ingest(
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
async fn engine_events(
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
async fn engine_event_packet(
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
async fn engine_event_replay(
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
async fn engine_event_associate(
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
async fn engine_alert_providers(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "alerts.providers", json!({})).await
}

#[tauri::command]
async fn engine_alert_status(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "alerts.status", json!({ "project_id": project_id })).await
}

#[tauri::command]
async fn engine_alert_poll(
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

#[tauri::command]
async fn engine_tap_status(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "tap.status", json!({ "project_id": project_id })).await
}

#[tauri::command]
async fn engine_tap_query(
    state: tauri::State<'_, Arc<Engine>>,
    service: String,
    adql: String,
    release: Option<String>,
    max_rows: Option<u32>,
    format: Option<String>,
    refresh: Option<bool>,
    offline: Option<bool>,
    timeout: Option<f64>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "tap.query", json!({
        "service": service,
        "adql": adql,
        "release": release.unwrap_or_else(|| "unknown".into()),
        "max_rows": max_rows.unwrap_or(200),
        "format": format.unwrap_or_else(|| "csv".into()),
        "refresh": refresh.unwrap_or(false),
        "offline": offline.unwrap_or(false),
        "timeout": timeout.unwrap_or(60.0),
        "project_id": project_id,
    })).await
}

#[tauri::command]
async fn engine_readiness(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "readiness.status", json!({})).await
}

#[tauri::command]
async fn engine_store_usage(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "store.usage", json!({})).await
}

#[tauri::command]
async fn engine_manifests(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "manifest.list", json!({ "project_id": project_id })).await
}

#[tauri::command]
async fn engine_projects(
    state: tauri::State<'_, Arc<Engine>>,
    include_archived: Option<bool>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "project.list",
        json!({ "include_archived": include_archived.unwrap_or(true) }),
    ).await
}

#[tauri::command]
async fn engine_project_create(
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
async fn engine_project_open(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "project.open", json!({ "project_id": project_id })).await
}

#[tauri::command]
async fn engine_project_update(
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
async fn engine_project_archive(
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
async fn engine_project_validate(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "project.validate", json!({ "project_id": project_id })).await
}

#[tauri::command]
async fn engine_curves_list(
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
async fn engine_curve_get(
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
async fn engine_curve_fold(
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
async fn engine_curve_bin(
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
async fn engine_fits_describe(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "fits.describe", json!({ "path": path })).await
}

#[tauri::command]
async fn engine_fits_header(
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
async fn engine_fits_image(
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
async fn engine_ztf_images_search(
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
async fn engine_ztf_images_download(
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
async fn engine_tess_tpf_download(
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
async fn engine_tess_tpf_photometry(
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

#[tauri::command]
async fn engine_products(
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
async fn engine_product(
    state: tauri::State<'_, Arc<Engine>>,
    product_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "products.get", json!({ "product_id": product_id })).await
}

#[tauri::command]
async fn engine_acquire(
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
async fn engine_features_list(state: tauri::State<'_, Arc<Engine>>, project_id: Option<String>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "features.list", json!({ "project_id": project_id })).await
}

/// Feature extraction runs a Lomb-Scargle search per curve, so it is slow
/// enough to need the blocking pool.
#[tauri::command]
async fn engine_features_build(
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
async fn engine_features_build_resumable(
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
async fn engine_detect(
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
async fn engine_feature_cache_clear(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "cache.features.clear", json!({})).await
}

#[tauri::command]
async fn engine_significance_calibrate(
    state: tauri::State<'_, Arc<Engine>>,
    scores: Vec<f64>,
    reference_scores: Option<Vec<f64>>,
    threshold: Option<f64>,
    strata: Option<Value>,
    name: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "significance.calibrate", json!({
        "scores": scores,
        "reference_scores": reference_scores,
        "threshold": threshold,
        "strata": strata,
        "name": name.unwrap_or_else(|| "default".into()),
        "project_id": project_id,
    })).await
}

#[tauri::command]
async fn engine_selection_evaluate(
    state: tauri::State<'_, Arc<Engine>>,
    records: Vec<Value>,
    dimensions: Option<Vec<String>>,
    edges: Option<Value>,
    fit_model: Option<bool>,
    model_features: Option<Vec<String>>,
    bootstrap_samples: Option<u32>,
    seed: Option<u64>,
    name: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "selection.evaluate", json!({
        "records": records,
        "dimensions": dimensions,
        "edges": edges,
        "fit_model": fit_model.unwrap_or(false),
        "model_features": model_features,
        "bootstrap_samples": bootstrap_samples.unwrap_or(0),
        "seed": seed.unwrap_or(42),
        "name": name.unwrap_or_else(|| "default".into()),
        "project_id": project_id,
    })).await
}

#[tauri::command]
async fn engine_review_next(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    limit: Option<u32>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "review.next", json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "limit": limit.unwrap_or(20),
        "project_id": project_id,
    })).await
}

#[tauri::command]
async fn engine_followup_plan(
    state: tauri::State<'_, Arc<Engine>>,
    ra_deg: f64,
    dec_deg: f64,
    start_utc: Option<String>,
    duration_hours: Option<f64>,
    latitude_deg: Option<f64>,
    longitude_deg: Option<f64>,
    min_altitude_deg: Option<f64>,
    cadence_minutes: Option<u32>,
    target_id: Option<String>,
    twilight_sun_altitude_deg: Option<f64>,
    min_moon_separation_deg: Option<f64>,
    max_moon_illumination: Option<f64>,
    max_airmass: Option<f64>,
    weather: Option<Value>,
    facility_name: Option<String>,
    facility_constraints: Option<Value>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "followup.plan", json!({
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "start_utc": start_utc,
        "duration_hours": duration_hours.unwrap_or(12.0),
        "latitude_deg": latitude_deg.unwrap_or(43.65),
        "longitude_deg": longitude_deg.unwrap_or(-79.38),
        "min_altitude_deg": min_altitude_deg.unwrap_or(30.0),
        "cadence_minutes": cadence_minutes.unwrap_or(10),
        "target_id": target_id,
        "twilight_sun_altitude_deg": twilight_sun_altitude_deg.unwrap_or(-18.0),
        "min_moon_separation_deg": min_moon_separation_deg.unwrap_or(0.0),
        "max_moon_illumination": max_moon_illumination.unwrap_or(1.0),
        "max_airmass": max_airmass,
        "weather": weather,
        "facility_name": facility_name,
        "facility_constraints": facility_constraints,
    })).await
}

/// Profiling runs the real stages, so it takes as long as they do.
#[tauri::command]
async fn engine_profile(
    state: tauri::State<'_, Arc<Engine>>,
    limit: Option<u32>,
) -> Result<Value, String> {
    let params = json!({ "limit": limit.unwrap_or(100) });
    call_blocking(Arc::clone(&state), "profile.run", params).await
}

#[tauri::command]
async fn engine_experiments(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "experiment.list", json!({ "project_id": project_id })).await
}

#[tauri::command]
async fn engine_experiment(
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
async fn engine_experiment_verify(
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
async fn engine_research_bundle_build(
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
async fn engine_research_bundle_verify(
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
async fn engine_research_benchmark_run(
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
async fn engine_experiment_compare(
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
async fn engine_ablation(
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
async fn engine_ablation_repeated(
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
async fn engine_stageb_compare(
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
async fn engine_candidates(
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
async fn engine_candidates_spatial(
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
async fn engine_candidate(
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

#[tauri::command]
async fn engine_image_features(
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
async fn engine_spectral_features(
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
async fn engine_sidecars_list(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "sidecars.list", json!({ "project_id": project_id })).await
}

#[tauri::command]
async fn engine_sidecar_save(
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
async fn engine_sidecar_join(
    state: tauri::State<'_, Arc<Engine>>,
    path: String,
    kind: String,
    identities: Vec<Value>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "sidecars.join", json!({
        "path": path, "kind": kind, "identities": identities,
    })).await
}

#[tauri::command]
async fn engine_candidate_timeline(
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
async fn engine_candidates_export(
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

#[tauri::command]
async fn engine_label(
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
async fn engine_label_summary(
    state: tauri::State<'_, Arc<Engine>>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "candidates.labels", json!({"project_id": project_id})).await
}

/// Supervised-review metrics over the labels recorded so far. Cheap: it reads
/// the stored label file rather than refitting anything.
#[tauri::command]
async fn engine_candidates_evaluate(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "candidates.evaluate",
        json!({ "name": name.unwrap_or_else(|| "default".into()), "project_id": project_id }),
    ).await
}

/// A hyperparameter sweep trains one model per configuration per seed, so it is
/// the longest-running action the engine exposes.
#[tauri::command]
async fn engine_deep_sweep(
    state: tauri::State<'_, Arc<Engine>>,
    kind: Option<String>,
    survey: Option<String>,
    seeds: Option<Vec<u32>>,
    epochs: Option<u32>,
    mode: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "kind": kind.unwrap_or_else(|| "autoencoder".into()),
        "survey": survey,
        "seeds": seeds.unwrap_or_else(|| vec![17, 29, 43]),
        "epochs": epochs.unwrap_or(20),
        "mode": mode.unwrap_or_else(|| "time".into()),
    });
    call_blocking(Arc::clone(&state), "deep.sweep", params).await
}

/// The ordered feature-column names, so the UI can label a feature vector
/// without hardcoding a list that would silently drift from the engine's.
#[tauri::command]
async fn engine_feature_names(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "features.names", json!({})).await
}

#[tauri::command]
async fn engine_catalog_status(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "catalog.status", json!({})).await
}

/// Catalog enrichment performs optional network queries, so it runs off the
/// UI thread and never blocks normal candidate generation.
#[tauri::command]
async fn engine_catalog_enrich(
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
async fn engine_literature_status(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "literature.status", json!({})).await
}

#[tauri::command]
async fn engine_literature_search(
    state: tauri::State<'_, Arc<Engine>>,
    object_id: Option<String>,
    terms: Option<Vec<String>>,
    event_ids: Option<Vec<String>>,
    providers: Option<Vec<String>>,
    limit: Option<u32>,
    refresh: Option<bool>,
    offline: Option<bool>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "literature.search", json!({
        "object_id": object_id.unwrap_or_default(),
        "terms": terms.unwrap_or_default(),
        "event_ids": event_ids.unwrap_or_default(),
        "providers": providers.unwrap_or_else(|| vec!["ads".into(), "arxiv".into()]),
        "limit": limit.unwrap_or(20),
        "refresh": refresh.unwrap_or(false),
        "offline": offline.unwrap_or(false),
        "project_id": project_id,
    })).await
}

#[tauri::command]
async fn engine_literature_enrich(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    refresh: Option<bool>,
    offline: Option<bool>,
    include_arxiv: Option<bool>,
    limit: Option<u32>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "literature.enrich", json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "refresh": refresh.unwrap_or(false),
        "offline": offline.unwrap_or(false),
        "include_arxiv": include_arxiv.unwrap_or(true),
        "limit": limit.unwrap_or(20),
        "project_id": project_id,
    })).await
}

#[tauri::command]
async fn engine_physical_characterize(
    state: tauri::State<'_, Arc<Engine>>,
    photometry: Value,
    extinction: Option<Value>,
    source: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "physical.characterize", json!({
        "photometry": photometry,
        "extinction": extinction,
        "source": source.unwrap_or_else(|| "caller".into()),
    })).await
}

#[tauri::command]
async fn engine_physical_enrich(
    state: tauri::State<'_, Arc<Engine>>,
    name: Option<String>,
    extinction: Option<Value>,
    project_id: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "physical.enrich", json!({
        "name": name.unwrap_or_else(|| "default".into()),
        "extinction": extinction,
        "project_id": project_id,
    })).await
}

#[tauri::command]
async fn engine_digital_twin_fit_profile(
    state: tauri::State<'_, Arc<Engine>>,
    survey: String,
    limit: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "digital_twin.fit_profile", json!({
        "survey": survey,
        "limit": limit.unwrap_or(500),
    })).await
}

#[tauri::command]
async fn engine_digital_twin_sample(
    state: tauri::State<'_, Arc<Engine>>,
    survey: String,
    limit: Option<u32>,
    n: Option<u32>,
    seed: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "digital_twin.sample", json!({
        "survey": survey,
        "limit": limit.unwrap_or(500),
        "n": n.unwrap_or(50),
        "seed": seed.unwrap_or(42),
    })).await
}

#[tauri::command]
async fn engine_digital_twin_evaluate_distance(
    state: tauri::State<'_, Arc<Engine>>,
    survey: String,
    limit: Option<u32>,
    seed: Option<u32>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "digital_twin.evaluate_distance", json!({
        "survey": survey,
        "limit": limit.unwrap_or(500),
        "seed": seed.unwrap_or(42),
    })).await
}

#[tauri::command]
async fn engine_digital_twin_evaluate_transfer(
    state: tauri::State<'_, Arc<Engine>>,
    survey: String,
    limit: Option<u32>,
    seeds: Option<Vec<u32>>,
    epochs: Option<u32>,
    fraction: Option<f64>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "digital_twin.evaluate_transfer", json!({
        "survey": survey,
        "limit": limit.unwrap_or(500),
        "seeds": seeds.unwrap_or_else(|| vec![17, 29, 43]),
        "epochs": epochs.unwrap_or(15),
        "fraction": fraction.unwrap_or(0.1),
    })).await
}

#[tauri::command]
async fn engine_gw_events(
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
async fn engine_gw_enrich(
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
async fn engine_frb_events(
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
async fn engine_frb_enrich(
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
async fn engine_gaia_epoch_ingest(
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
async fn engine_gaia_epoch_status(
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

#[tauri::command]
async fn engine_tns_credentials_configure(
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
async fn engine_tns_credentials_clear(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "credentials.tns.clear", json!({})).await
}

/// Grouped evaluation, bootstrapping, and fitting can take noticeable time.
#[tauri::command]
async fn engine_ranker_train(
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
async fn engine_ranker_apply(
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
async fn engine_ranker_list(state: tauri::State<'_, Arc<Engine>>) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "ranker.list", json!({})).await
}

#[tauri::command]
async fn engine_job_submit(
    state: tauri::State<'_, Arc<Engine>>,
    method: String,
    params: Option<Value>,
    project_id: Option<String>,
    idempotency_key: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "job.submit",
        json!({
            "method": method,
            "params": params.unwrap_or(json!({})),
            "project_id": project_id,
            "idempotency_key": idempotency_key,
        }),
    ).await
}

#[tauri::command]
async fn engine_job_status(
    state: tauri::State<'_, Arc<Engine>>,
    job_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "job.status", json!({ "job_id": job_id })).await
}

#[tauri::command]
async fn engine_job_cancel(
    state: tauri::State<'_, Arc<Engine>>,
    job_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "job.cancel", json!({ "job_id": job_id })).await
}

#[tauri::command]
async fn engine_job_retry(
    state: tauri::State<'_, Arc<Engine>>,
    job_id: String,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "job.retry", json!({ "job_id": job_id })).await
}

#[tauri::command]
async fn engine_jobs(
    state: tauri::State<'_, Arc<Engine>>,
    statuses: Option<Vec<String>>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state), "job.list", json!({ "statuses": statuses })).await
}

/// Full candidate generation re-reads every curve and runs every detector.
#[tauri::command]
async fn engine_pipeline(
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

#[tauri::command]
async fn engine_frame_offset(
    state: tauri::State<'_, Arc<Engine>>,
    ra_deg: f64,
    dec_deg: f64,
    time_system: Option<String>,
) -> Result<Value, String> {
    call_blocking(Arc::clone(&state),
        "timeframe.offset",
        json!({
            "ra_deg": ra_deg, "dec_deg": dec_deg,
            "time_system": time_system.unwrap_or_else(|| "HJD_UTC".into()),
        }),
    ).await
}

/// Cross-matching re-reads every stored curve, so it goes to the blocking pool.
#[tauri::command]
async fn engine_crossmatch(
    state: tauri::State<'_, Arc<Engine>>,
    radius_arcsec: Option<f64>,
    anchor_survey: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "radius_arcsec": radius_arcsec.unwrap_or(2.0),
        "anchor_survey": anchor_survey,
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "crossmatch.run", params).await
}

#[tauri::command]
async fn engine_profiles(
    state: tauri::State<'_, Arc<Engine>>,
    radius_arcsec: Option<f64>,
    top: Option<u32>,
    anchor_survey: Option<String>,
    project_id: Option<String>,
) -> Result<Value, String> {
    let params = json!({
        "radius_arcsec": radius_arcsec.unwrap_or(2.0),
        "top": top.unwrap_or(25),
        "anchor_survey": anchor_survey,
        "project_id": project_id,
    });
    call_blocking(Arc::clone(&state), "crossmatch.profile", params).await
}

/// Training runs for minutes and must never block the UI thread.
#[tauri::command]
async fn engine_deep_train(
    state: tauri::State<'_, Arc<Engine>>,
    name: String,
    kind: Option<String>,
    survey: Option<String>,
    epochs: Option<u32>,
) -> Result<Value, String> {
    let params = json!({
        "name": name,
        "kind": kind.unwrap_or_else(|| "autoencoder".into()),
        "survey": survey,
        "epochs": epochs.unwrap_or(30),
    });
    call_blocking(Arc::clone(&state), "deep.train", params).await
}

#[tauri::command]
async fn engine_deep_compare(
    state: tauri::State<'_, Arc<Engine>>,
    survey: Option<String>,
    fraction: Option<f64>,
    epochs: Option<u32>,
) -> Result<Value, String> {
    let params = json!({
        "survey": survey,
        "fraction": fraction.unwrap_or(0.1),
        "epochs": epochs.unwrap_or(20),
    });
    call_blocking(Arc::clone(&state), "deep.compare", params).await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Arc::new(Engine::new()))
        .setup(|app| {
            // The window starts hidden (see tauri.conf.json) so the OS never
            // shows a blank frame while the webview is still loading -- the
            // frontend shows it itself once real content has painted. If
            // that signal is ever lost (a crash before mount, a dropped
            // permission), the window must not stay invisible forever, since
            // that would look identical to the app failing to launch at all.
            // This fallback guarantees it appears within a bounded time
            // regardless.
            if let Some(window) = app.get_webview_window("main") {
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(8));
                    let _ = window.show();
                });
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            engine_ping,
            engine_hardware,
            engine_paths,
            engine_versions,
            engine_cache_status,
            engine_cache_enforce,
            engine_surveys,
            engine_event_providers,
            engine_event_ingest,
            engine_events,
            engine_event_packet,
            engine_event_replay,
            engine_event_associate,
            engine_alert_providers,
            engine_alert_status,
            engine_alert_poll,
            engine_tap_status,
            engine_tap_query,
            engine_readiness,
            engine_store_usage,
            engine_manifests,
            engine_projects,
            engine_project_create,
            engine_project_open,
            engine_project_update,
            engine_project_archive,
            engine_project_validate,
            engine_acquire,
            engine_curves_list,
            engine_curve_get,
            engine_curve_fold,
            engine_curve_bin,
            engine_fits_describe,
            engine_fits_header,
            engine_fits_image,
            engine_image_features,
            engine_spectral_features,
            engine_sidecars_list,
            engine_sidecar_save,
            engine_sidecar_join,
            engine_ztf_images_search,
            engine_ztf_images_download,
            engine_tess_tpf_download,
            engine_tess_tpf_photometry,
            engine_products,
            engine_product,
            engine_features_list,
            engine_features_build,
            engine_features_build_resumable,
            engine_detect,
            engine_deep_train,
            engine_deep_compare,
            engine_frame_offset,
            engine_crossmatch,
            engine_profiles,
            engine_pipeline,
            engine_candidates,
            engine_candidates_spatial,
            engine_candidate,
            engine_candidate_timeline,
            engine_candidates_export,
            engine_label,
            engine_label_summary,
            engine_catalog_status,
            engine_catalog_enrich,
            engine_literature_status,
            engine_literature_search,
            engine_literature_enrich,
            engine_physical_characterize,
            engine_physical_enrich,
            engine_digital_twin_fit_profile,
            engine_digital_twin_sample,
            engine_digital_twin_evaluate_distance,
            engine_digital_twin_evaluate_transfer,
            engine_gw_events,
            engine_gw_enrich,
            engine_frb_events,
            engine_frb_enrich,
            engine_gaia_epoch_ingest,
            engine_gaia_epoch_status,
            engine_tns_credentials_configure,
            engine_tns_credentials_clear,
            engine_ranker_train,
            engine_ranker_apply,
            engine_ranker_list,
            engine_job_submit,
            engine_job_status,
            engine_job_cancel,
            engine_job_retry,
            engine_jobs,
            engine_experiments,
            engine_experiment,
            engine_experiment_verify,
            engine_experiment_compare,
            engine_research_bundle_build,
            engine_research_bundle_verify,
            engine_research_benchmark_run,
            engine_ablation,
            engine_ablation_repeated,
            engine_stageb_compare,
            engine_profile,
            engine_feature_cache_clear,
            engine_significance_calibrate,
            engine_selection_evaluate,
            engine_review_next,
            engine_followup_plan,
            engine_candidates_evaluate,
            engine_feature_names,
            engine_deep_sweep,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                window.state::<Arc<Engine>>().shutdown();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
