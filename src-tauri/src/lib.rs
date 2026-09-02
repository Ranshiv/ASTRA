mod engine;
mod commands;

use std::sync::Arc;

use tauri::Manager;

use engine::Engine;

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
            commands::core::engine_ping,
            commands::core::engine_hardware,
            commands::core::engine_paths,
            commands::core::engine_versions,
            commands::core::engine_cache_status,
            commands::core::engine_cache_enforce,
            commands::core::engine_surveys,
            commands::core::engine_readiness,
            commands::core::engine_store_usage,
            commands::core::engine_acquire,
            commands::core::engine_products,
            commands::core::engine_product,
            commands::core::engine_features_list,
            commands::core::engine_features_build,
            commands::core::engine_features_build_resumable,
            commands::core::engine_detect,
            commands::core::engine_feature_cache_clear,
            commands::core::engine_feature_names,
            commands::events::engine_event_providers,
            commands::events::engine_event_ingest,
            commands::events::engine_events,
            commands::events::engine_event_packet,
            commands::events::engine_event_replay,
            commands::events::engine_event_associate,
            commands::events::engine_alert_providers,
            commands::events::engine_alert_status,
            commands::events::engine_alert_poll,
            commands::tap_review::engine_tap_status,
            commands::tap_review::engine_tap_query,
            commands::tap_review::engine_significance_calibrate,
            commands::tap_review::engine_selection_evaluate,
            commands::tap_review::engine_review_next,
            commands::tap_review::engine_candidates_evaluate,
            commands::followup_schedule::engine_followup_plan,
            commands::followup_schedule::engine_discard_scan,
            commands::followup_schedule::engine_followup_request,
            commands::followup_schedule::engine_followup_result,
            commands::followup_schedule::engine_followup_history,
            commands::followup_schedule::engine_literature_status,
            commands::followup_schedule::engine_literature_search,
            commands::followup_schedule::engine_literature_enrich,
            commands::followup_schedule::engine_physical_characterize,
            commands::followup_schedule::engine_physical_enrich,
            commands::followup_schedule::engine_digital_twin_fit_profile,
            commands::followup_schedule::engine_digital_twin_sample,
            commands::followup_schedule::engine_digital_twin_evaluate_distance,
            commands::followup_schedule::engine_digital_twin_evaluate_transfer,
            commands::projects::engine_manifests,
            commands::projects::engine_projects,
            commands::projects::engine_project_create,
            commands::projects::engine_project_open,
            commands::projects::engine_project_update,
            commands::projects::engine_project_archive,
            commands::projects::engine_project_validate,
            commands::curves_fits::engine_curves_list,
            commands::curves_fits::engine_curve_get,
            commands::curves_fits::engine_curve_fold,
            commands::curves_fits::engine_curve_bin,
            commands::curves_fits::engine_fits_describe,
            commands::curves_fits::engine_fits_header,
            commands::curves_fits::engine_fits_image,
            commands::curves_fits::engine_image_features,
            commands::curves_fits::engine_spectral_features,
            commands::curves_fits::engine_sidecars_list,
            commands::curves_fits::engine_sidecar_save,
            commands::curves_fits::engine_sidecar_join,
            commands::ztf_tess::engine_ztf_images_search,
            commands::ztf_tess::engine_ztf_images_download,
            commands::ztf_tess::engine_tess_tpf_download,
            commands::ztf_tess::engine_tess_tpf_photometry,
            commands::experiments_research::engine_profile,
            commands::experiments_research::engine_experiments,
            commands::experiments_research::engine_experiment,
            commands::experiments_research::engine_experiment_verify,
            commands::experiments_research::engine_research_bundle_build,
            commands::experiments_research::engine_research_bundle_verify,
            commands::experiments_research::engine_research_benchmark_run,
            commands::experiments_research::engine_experiment_compare,
            commands::experiments_research::engine_ablation,
            commands::experiments_research::engine_ablation_repeated,
            commands::experiments_research::engine_stageb_compare,
            commands::experiments_research::engine_pipeline,
            commands::candidates::engine_candidates,
            commands::candidates::engine_candidates_spatial,
            commands::candidates::engine_candidate,
            commands::candidates::engine_candidate_explain,
            commands::candidates::engine_candidate_timeline,
            commands::candidates::engine_candidates_export,
            commands::candidates::engine_candidate_broadcast,
            commands::candidates::engine_label,
            commands::candidates::engine_label_summary,
            commands::candidates::engine_candidate_vote,
            commands::candidates::engine_candidate_votes,
            commands::candidates::engine_candidate_vote_promote,
            commands::catalog_gw_frb::engine_catalog_status,
            commands::catalog_gw_frb::engine_catalog_enrich,
            commands::catalog_gw_frb::engine_gw_events,
            commands::catalog_gw_frb::engine_gw_enrich,
            commands::catalog_gw_frb::engine_frb_events,
            commands::catalog_gw_frb::engine_frb_enrich,
            commands::catalog_gw_frb::engine_gaia_epoch_ingest,
            commands::catalog_gw_frb::engine_gaia_epoch_status,
            commands::credentials_ranker::engine_tns_credentials_configure,
            commands::credentials_ranker::engine_tns_credentials_clear,
            commands::credentials_ranker::engine_ranker_train,
            commands::credentials_ranker::engine_ranker_apply,
            commands::credentials_ranker::engine_ranker_list,
            commands::crossmatch_deep::engine_job_submit,
            commands::crossmatch_deep::engine_job_status,
            commands::crossmatch_deep::engine_job_cancel,
            commands::crossmatch_deep::engine_job_retry,
            commands::crossmatch_deep::engine_jobs,
            commands::crossmatch_deep::engine_frame_offset,
            commands::crossmatch_deep::engine_crossmatch,
            commands::crossmatch_deep::engine_profiles,
            commands::crossmatch_deep::engine_deep_train,
            commands::crossmatch_deep::engine_deep_compare,
            commands::crossmatch_deep::engine_deep_sweep,
            commands::frontier::engine_habitability_score,
            commands::frontier::engine_habitability_rank,
            commands::frontier::engine_neo_assess,
            commands::frontier::engine_neo_close_approach,
            commands::frontier::engine_asteroseismology_measure,
            commands::frontier::engine_asteroseismology_solve,
            commands::frontier::engine_technosignature_search,
            commands::frontier::engine_biosignature_synthesize,
            commands::frontier::engine_biosignature_fit,
            commands::frontier::engine_biosignature_detect,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                window.state::<Arc<Engine>>().shutdown();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
