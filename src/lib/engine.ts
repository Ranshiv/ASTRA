import { invoke } from "@tauri-apps/api/core";

export interface GpuInfo {
  name: string;
  total_vram_mb: number;
  free_vram_mb: number;
  compute_capability: string;
  driver_version: string;
}

export interface DeviceReport {
  device: "cpu" | "cuda";
  reason: string;
  torch_available: boolean;
  cuda_available: boolean;
  gpu?: GpuInfo;
}

export interface EnginePaths {
  root: string;
  projects: string;
  datasets: string;
  models: string;
  cache: string;
  logs: string;
  config: string;
}

export interface ProjectRegion {
  ra_deg: number;
  dec_deg: number;
  radius_arcsec: number;
}

export interface ResearchProject {
  project_id: string;
  name: string;
  description: string;
  selected_surveys: string[];
  query_regions: ProjectRegion[];
  tags: string[];
  data_root: string;
  status: "active" | "archived";
  created_utc: string;
  updated_utc: string;
  archived_utc: string | null;
  schema_version: number;
  layout: Record<string, string>;
  manifest_count: number;
}

export interface ProjectValidation {
  project_id: string;
  valid: boolean;
  issues: string[];
  schema_version?: number;
  status?: "active" | "archived";
  layout?: Record<string, string>;
  manifest_count?: number;
}

export interface CreateProjectArgs {
  name: string;
  projectId?: string;
  description?: string;
  selectedSurveys?: string[];
  queryRegions?: ProjectRegion[];
  tags?: string[];
  dataRoot?: string;
}

export interface CacheStatus {
  total_gb: number;
  cap_gb: number;
  usage_fraction: number;
  file_count: number;
  evicted_gb: number;
  evicted_files: number;
}

export interface DatasetStatus {
  used_gb: number;
  cap_gb: number;
  usage_fraction: number;
  available_gb: number;
}

export interface SurveyInfo {
  name: string;
  release: string;
  class: string;
  capabilities?: string[];
  credential_required?: boolean;
  resolution_arcsec?: number | null;
  enabled_by_default?: boolean;
}

export interface EventProviderInfo {
  name: string;
  kind: string;
  online: boolean;
}

export interface EventPacket {
  packet_key: string;
  event_id: string;
  packet_id: string;
  provider: string;
  release: string;
  packet_version: string;
  event_time: string | null;
  received_utc: string;
  localization: Record<string, unknown>;
  classifications: Array<{ label: string; probability: number | null }>;
  related_ids: string[];
  raw_sha256: string;
  raw_path: string;
  status: string;
  project_id?: string | null;
  raw?: string;
}

export interface EventCluster {
  event_id: string;
  provider: string;
  first_seen_utc: string;
  last_seen_utc: string;
  packet_count: number;
  packet_ids: string[];
  localization: Record<string, unknown>;
  classifications: Array<{ label: string; probability: number | null }>;
  project_id?: string | null;
}

export interface AlertProviderInfo {
  name: string;
  endpoint: string;
  mode: string;
  requires_endpoint_override: boolean;
}

export interface AlertLatencySummary {
  median: number;
  p95: number;
  n: number;
}

export interface AlertPollResult {
  schema_version: number;
  provider: string;
  state: string;
  endpoint: string;
  cursor?: string | null;
  packets: EventPacket[];
  ingested: number;
  new_packets?: number;
  duplicate_rate?: number | null;
  latency_summary?: AlertLatencySummary | null;
  errors: Array<{ index: number; error: string }>;
  polled_utc: string;
}

export interface TapResult {
  schema_version: number;
  service: string;
  release: string;
  state: string;
  rows: Array<Record<string, unknown>>;
  format: "csv" | "votable";
  query: Record<string, unknown>;
  error?: string | null;
  fetched_utc?: string | null;
  expires_utc?: string | null;
  cache: { state: string; stale: boolean };
}

export interface SignificanceReport {
  schema_version: number;
  method: string;
  ready: boolean;
  reason?: string;
  reference_kind?: "external_reference" | "batch_relative";
  n_observed: number;
  n_reference: number;
  threshold?: number;
  score?: number;
  tail_probability?: number;
  calibrated_percentile?: number;
  selected?: number;
  reference_exceedances?: number;
  estimated_fdr?: number;
  score_range?: number[];
  tail_probability_summary?: { min: number; median: number; max: number };
  reference_fingerprint?: string;
  strata?: Record<string, unknown>;
  generated_utc?: string;
  path?: string;
}

export interface SelectionCell {
  bins: Record<string, string>;
  detected: number;
  injected: number;
  completeness: number | null;
  ci95: number[] | null;
  weighted_detected?: number | null;
  weighted_injected?: number | null;
  weighted_completeness?: number | null;
  effective_injected?: number | null;
}

export interface SelectionReport {
  schema_version: number;
  ready: boolean;
  injected: number;
  detected: number;
  dimensions: string[];
  edges: Record<string, number[]>;
  cells: SelectionCell[];
  model?: Record<string, unknown>;
  generated_utc?: string;
  path?: string;
}

export interface ReviewSelection {
  candidate_id: string;
  priority: number;
  reasons: string[];
}

export interface LiteratureRecord {
  provider: string;
  bibcode?: string | null;
  arxiv_id?: string | null;
  title?: string | null;
  authors?: string[];
  abstract?: string | null;
  year?: number | null;
  doi?: string | null;
  url?: string | null;
  citation_count?: number | null;
}

export interface LiteratureProviderResult {
  provider: string;
  release: string;
  state: string;
  records: LiteratureRecord[];
  error?: string | null;
  fetched_utc?: string | null;
  cache?: { state: string; stale: boolean };
}

export interface LiteratureSearchResult {
  schema_version: number;
  query: { object_id: string; terms: string[]; event_ids: string[]; limit: number };
  providers: Record<string, LiteratureProviderResult>;
  records: LiteratureRecord[];
  complete: boolean;
  provenance: Array<Record<string, unknown>>;
}

export interface LiteratureStatus {
  ttl_days: number;
  providers: Record<string, string>;
  cache: { entries: Array<Record<string, unknown>>; total: number };
  ads_token_configured: boolean;
}

export interface PhysicalCharacterization {
  schema_version: number;
  source: string;
  bands_used: string[];
  photometry: Record<string, number>;
  colors: Record<string, number>;
  temperature_k: number | null;
  temperature_spread_k: number | null;
  blackbody_temperature_k: number | null;
  sed_residual_rms: number | null;
  extinction_applied: Record<string, number>;
  quality: "usable" | "insufficient";
  warnings: string[];
}

export interface SurveyProfileSummary {
  survey: string;
  n_curves_used: number;
  mean_coverage: number | null;
  n_gap_runs_sampled: number;
  noise_std: number | null;
  length: number;
  note: string;
}

export interface DigitalTwinSample {
  profile: SurveyProfileSummary;
  batch: {
    rows: number;
    length: number;
    channels: number;
    mode: string;
    mean_coverage: number;
  };
}

export interface DigitalTwinDistance {
  profile: SurveyProfileSummary;
  per_feature: Record<string, number | null>;
  mean_ks_statistic: number;
  real_rows: number;
  synthetic_rows: number;
  note?: string;
  error?: string;
}

export interface DigitalTwinTransferArm {
  mean: number;
  std: number;
  ci95: [number, number];
  n: number;
}

export interface DigitalTwinTransferResult {
  profile: SurveyProfileSummary;
  trained_on_real: DigitalTwinTransferArm | null;
  trained_on_synthetic: DigitalTwinTransferArm | null;
  held_out_test_injection?: Record<string, unknown>;
  error?: string;
}

export interface FollowupPlan {
  schema_version: number;
  target_id?: string | null;
  target: { ra_deg: number; dec_deg: number };
  site: { latitude_deg: number; longitude_deg: number };
  constraints: {
    min_altitude_deg: number; cadence_minutes: number;
    twilight_sun_altitude_deg?: number; min_moon_separation_deg?: number;
    max_moon_illumination?: number; max_airmass?: number | null;
    facility?: Record<string, unknown> | null; weather_supplied?: boolean;
  };
  start_utc: string;
  duration_hours: number;
  visible: boolean;
  windows: Array<{ start_utc: string; end_utc: string; slots: number }>;
  best_slot?: { utc: string; altitude_deg: number; azimuth_deg: number; airmass: number } | null;
  rejected_slots?: Record<string, number>;
  samples?: Array<Record<string, unknown>>;
  mode: "draft_only";
  caveats: string[];
}

export interface ReadinessStatus {
  gaia_epoch: { status: string; expected_release: string; enabled: boolean; code_ready: boolean; reason: string };
  multimodal: { status: string; free_vram_mb?: number; required_min_free_vram_mb: number; enabled: boolean };
  release: { status: string; authenticode_certificate: boolean; timestamp_url: boolean; updater_key: boolean; publication_configured: boolean };
  connectors: SurveyInfo[];
}

export interface SurveyOutcome {
  survey: string;
  release: string;
  sources_found: number;
  curves_stored: number;
  points_stored: number;
  mb_stored: number;
  skipped_existing: number;
  error?: string;
}

export interface AcquisitionResult {
  dataset_id: string;
  project_id?: string | null;
  query: { ra_deg: number; dec_deg: number; radius_arcsec: number };
  surveys: SurveyOutcome[];
  totals: { curves: number; points: number; mb: number };
  manifest_path: string | null;
  content_hash: string | null;
}

export interface PipelineResult {
  candidates: Candidate[];
  candidates_built: number;
  output_path: string;
  anchor_survey?: string | null;
  anchor_policy?: "empty" | "largest_catalogue" | "explicit";
  cross_survey_groups?: number;
  resolved_multi_survey?: number;
}

/** Result of `acquire.project`: one AcquisitionResult per region in the
 *  project's `query_regions`, run sequentially, plus the totals summed
 *  across every region and survey. */
export interface ProjectAcquisitionResult {
  project_id: string;
  regions: AcquisitionResult[];
  totals: { curves: number; points: number; mb: number };
}

export interface ArtifactIndicator {
  name: string;
  weight: number;
  detail: string;
}

/** Why the engine does or does not think a signal is instrumental.
 *
 *  `indicators` and `clearing_evidence` ship in every candidate record and are
 *  the whole substance of the assessment; `likelihood` alone is a number with
 *  no argument behind it.
 */
export interface ArtifactAssessment {
  likelihood?: number;
  verdict?: string;
  indicators?: ArtifactIndicator[];
  clearing_evidence?: string[];
}

export interface Candidate {
  candidate_id: string;
  rank: number;
  object_id: string;
  survey: string;
  release: string;
  band: string;
  path: string;
  ra_deg: number;
  dec_deg: number;
  score: {
    total: number;
    supervised_probability?: number;
    ranking_method?: string;
    components?: Record<string, number | null>;
    /** Per-component contribution (weight x value) and the weight actually
     *  available. The total renormalises over `weight_used`, so a component
     *  score means little without both. */
    weighted?: Record<string, number | null>;
    weight_used?: number;
    weight_version?: number;
    reasons?: string[];
  };
  artifact: ArtifactAssessment;
  features: Record<string, number | null>;
  explanation: {
    what_happened?: string;
    why_flagged?: string[];
    supporting_observations?: {
      epochs?: number;
      surveys_resolving?: number;
      blended_in?: string[];
    };
    could_be_artifact?: ArtifactAssessment;
    resembles?: string[];
    recommended_actions?: string[];
    coverage?: { tier: string; status: string };
  };
  catalog?: {
    summary?: { states?: Record<string, string>; known_variable?: boolean; known_object?: boolean };
    providers?: Record<string, { state: string; matches?: unknown[] }>;
  };
  gw?: GwEvidence;
  frb?: FrbEvidence;
  event_ids?: string[];
  significance?: SignificanceReport;
  evidence_completeness?: Record<string, unknown>;
  source_attribution?: Record<string, unknown>;
  physical_characterization?: Record<string, unknown>;
  follow_up_plan?: Record<string, unknown>;
  literature?: LiteratureSearchResult;
  provenance_refs?: Array<Record<string, unknown>>;
  label?: string;
  review?: { label: string; note: string; recorded_utc: string };
}

export interface CandidateTimelineCurve {
  survey: string;
  release: string;
  object_id: string;
  band: string;
  path: string;
  value_kind: "mag" | "flux";
  time_system: string;
  points: number;
  time_start: number;
  time_end: number;
  separation_arcsec: number;
  resolved: boolean;
  times: number[];
  values: number[];
}

export interface CandidateTimeline {
  candidate_id: string;
  radius_arcsec: number;
  events: Array<Omit<CandidateTimelineCurve, "path" | "times" | "values">>;
  curves: CandidateTimelineCurve[];
  truncated?: boolean;
  warning?: string | null;
}

export interface SpatialCandidatePoint {
  candidate_id: string;
  ra_deg: number;
  dec_deg: number;
  gaia_distance_pc: number | null;
  gaia_abs_g_mag: number | null;
  gaia_parallax_snr: number | null;
  /** SNR >= 5 (the same threshold scoring.py uses for luminosity checks).
   *  A distance can be present and still unreliable -- check this, not just
   *  whether gaia_distance_pc is non-null. */
  distance_reliable: boolean;
  /** Gaia's stored ra_deg/dec_deg is fixed at J2016.0; these two are the
   *  same object propagated to today by its proper motion, for a "where is
   *  it now" overlay. null when there's no Gaia counterpart or no proper
   *  motion to propagate. The candidate's own ra_deg/dec_deg above is left
   *  untouched -- it is the detecting survey's immutable observation. */
  gaia_ra_now_deg: number | null;
  gaia_dec_now_deg: number | null;
  score_total: number | null;
}

export interface SpatialResult {
  points: SpatialCandidatePoint[];
  total: number;
  reliable: number;
  snr_threshold: number;
  gaia_matched: number;
  gaia_match_rate: number | null;
}

export interface CatalogStatus {
  ttl_days: number;
  cache: {
    total: number;
    entries: Array<{ provider: string; status: string; count: number; earliest_expiry?: string }>;
  };
  tns_credentials: { configured: boolean; usable?: boolean; backend: string; bot_name?: string };
}

export interface GwEventSummary {
  name: string;
  catalog: string;
  gps_time: number;
}

export interface GwEventsResult {
  catalog: string;
  events: GwEventSummary[];
}

export interface GwCoincidentEvent {
  event: string;
  catalog: string;
  gps_time: number;
  probability_density: number;
  credible_level: number;
  /** False for a fixed/degenerate posterior (EM-counterpart position) even
   *  at an exact match -- check position_source, not just this flag. */
  in_90pct_region: boolean;
  position_source: "gw_posterior" | "em_counterpart_fixed";
}

/** Never moves a candidate's score -- see gw.py's module docstring for why:
 *  new, unvalidated evidence should not silently reweight the ranking. */
export interface GwEvidence {
  checked_events: number;
  temporally_coincident?: number;
  coincident: GwCoincidentEvent[];
  state: "match" | "no_match" | "unavailable";
  window_days?: number;
  reason?: string;
}

export interface GwEnrichmentResult {
  catalog: string;
  events_checked: number;
  candidates: number;
  counts: { match: number; no_match: number; unavailable: number };
}

export interface FrbBurstSummary {
  tns_name: string;
  repeater_name: string;
  ra_deg: number;
  ra_err_deg: number;
  dec_deg: number;
  dec_err_deg: number;
  mjd_400: number;
  localization_id: string | null;
}

export interface FrbEventsResult {
  bursts: FrbBurstSummary[];
}

export interface FrbCoincidentBurst {
  burst: string;
  repeater_name: string;
  mjd_400: number;
  sigma_offset: number;
  sigma_threshold: number;
  /** "ellipse" (ra_err/dec_err) for most bursts; "healpix" only for the
   *  minority with a precomputed baseband localization map. */
  position_source: "ellipse" | "healpix";
  confidence_level?: number;
  in_90pct_region?: boolean;
}

/** Never moves a candidate's score -- see frb.py's module docstring. */
export interface FrbEvidence {
  checked_bursts: number;
  temporally_coincident?: number;
  coincident: FrbCoincidentBurst[];
  state: "match" | "no_match" | "unavailable";
  window_days?: number;
  sigma_threshold?: number;
  reason?: string;
}

export interface FrbEnrichmentResult {
  bursts_checked: number;
  candidates: number;
  counts: { match: number; no_match: number; unavailable: number };
}

export interface CatalogEnrichmentResult {
  name: string;
  candidates: number;
  state_counts: Record<string, number>;
  output_path: string;
  offline: boolean;
  refresh: boolean;
}

export interface RankerResult {
  ready: boolean;
  model_name?: string;
  model_path?: string;
  manifest_path?: string;
  evaluation?: Record<string, unknown>;
  reason?: string;
  gate?: Record<string, unknown>;
}

export interface AblationResult {
  experiment_id: string;
  seeds?: number[];
  survey?: string | null;
  survey_groups?: Record<string, unknown>;
  feature_groups?: Record<string, unknown>;
  detectors?: Record<string, unknown>;
}

export interface EngineJob {
  job_id: string;
  method: string;
  status: "queued" | "running" | "paused" | "retrying" | "partial" | "completed" | "failed" | "cancelled";
  project_id?: string | null;
  params?: Record<string, unknown>;
  progress?: {
    fraction?: number;
    phase?: string;
    message?: string;
    items_done?: number;
    items_total?: number;
    bytes_downloaded?: number;
    bytes_total?: number;
  } | null;
  checkpoint?: unknown;
  retry_count?: number;
  byte_count?: number;
  cancel_requested?: boolean;
  result?: unknown;
  error?: string;
}

export interface AcquireArgs {
  raDeg: number;
  decDeg: number;
  radiusArcsec: number;
  surveys: string[];
  limit: number;
  projectId?: string;
}

export interface CurveSummary {
  path: string;
  survey: string;
  release: string;
  object_id: string;
  band: string;
  value_kind: "mag" | "flux";
  time_system: string;
  points: number;
  time_span_days: number;
  mean_value: number;
  std_value: number;
}

export interface CurvePayload extends CurveSummary {
  time: number[];
  value: number[];
  value_err: number[];
  downsampled: boolean;
  shown_points: number;
}

export interface FoldedCurve {
  points: number;
  shown_points: number;
  period_days: number;
  epoch: number;
  phase: number[];
  value: number[];
  value_kind: "mag" | "flux";
  band: string;
}

export interface ZtfImageMetadata {
  filefracday: string;
  field: string | number;
  filtercode: string;
  ccdid: string | number;
  imgtypecode: string;
  qid: string | number;
  product_url: string;
  cutout_url: string;
  [key: string]: unknown;
}

export interface ImageProduct {
  product_id: string;
  provider: string;
  product_kind: "science";
  path: string;
  bytes: number;
  sha256: string;
  downloaded_utc: string;
  project_id?: string | null;
  request: { ra_deg: number; dec_deg: number; size_arcsec: number };
  fits: Record<string, unknown>;
  reused?: boolean;
}

export interface ZtfCutoutArgs {
  raDeg: number;
  decDeg: number;
  metadata: ZtfImageMetadata;
  sizeArcsec?: number;
  productKind?: "science";
  release?: string;
  projectId?: string;
  maxBytes?: number;
}

export interface TessTpfDownloadArgs {
  raDeg: number;
  decDeg: number;
  sector: number;
  sizePixels?: number;
  targetId?: string;
  product?: "SPOC";
  projectId?: string;
  maxBytes?: number;
  overwrite?: boolean;
}

export interface TessBlendAssessment {
  resolved: false;
  risk: "high" | "moderate" | "low" | "unknown";
  pixel_scale_arcsec: number;
  target_pixel: [number, number];
  neighbors_considered: number;
  neighbors_in_cutout: number;
  neighbors_in_aperture: number;
  contamination_fraction: number | null;
  source_attribution?: Array<{ object_id?: string; flux_fraction: number; kind: string }>;
  attribution_method?: string;
  attribution_diagnostics?: {
    prior_sum: number | null;
    concentration_index: number | null;
    target_fraction_sensitivity: number | null;
    quality: string;
    neighbor_flux_perturbation: number[];
  };
  neighbors: Array<Record<string, unknown>>;
  warning: string;
}

export interface TessTpfProduct {
  schema_version: number;
  product_id: string;
  provider: string;
  product_kind: "target_pixel_file";
  project_id?: string | null;
  request: {
    ra_deg: number;
    dec_deg: number;
    sector: number;
    size_pixels: number;
    target_id?: string | null;
    product: string;
  };
  path: string;
  fits_bytes: number;
  fits_sha256: string;
  fits: Record<string, unknown>;
  downloaded_utc: string;
  reused?: boolean;
}

export interface TessPhotometryArgs {
  path: string;
  raDeg: number;
  decDeg: number;
  neighbors?: Array<Record<string, unknown>>;
  targetMag?: number;
  apertureRadiusPixels?: number;
  qualityMask?: number;
  targetId?: string;
  persist?: boolean;
  maxPoints?: number;
}

export interface TessPhotometryPayload {
  path: string;
  time: number[];
  flux: number[];
  flux_err: Array<number | null>;
  background: Array<number | null>;
  quality: number[];
  points: number;
  total_cadences: number;
  shown_points: number;
  downsampled: boolean;
  aperture_mask: boolean[][];
  pixel_shape: [number, number];
  position_source: "wcs" | "cutout_center";
  sector: number | null;
  blend: TessBlendAssessment;
  curve_path?: string;
  curve_bytes?: number;
  source?: string;
}

// ---------------------------------------------------------------------------
// Research framework (plan sections 19 and 20)
// ---------------------------------------------------------------------------

export interface ExperimentSummary {
  experiment_id: string;
  kind: string;
  created_utc: string;
  code_version: string;
  code_revision?: string | null;
  runtime_seconds: number;
  failed: boolean;
}

export interface ExperimentProvenance {
  experiment_id: string;
  created_utc: string;
  code_version: string;
  feature_version: number;
  preprocessing_version: number;
  feature_schema_hash: string;
  preprocessing_schema_hash: string;
  dataset_hash: string | null;
  dataset_id: string | null;
  model_version: string | null;
  seed: number;
  hardware: Record<string, unknown>;
  environment: Record<string, string>;
}

export interface ExperimentRecord {
  schema_version: number;
  kind: string;
  provenance: ExperimentProvenance;
  configuration: Record<string, unknown>;
  results: Record<string, unknown>;
  runtime_seconds: number;
  notes: string;
}

/** Recorded-versus-current, per field. An empty `drift` is the reproducible case. */
export interface ExperimentVerification {
  experiment_id: string;
  reproducible: boolean;
  drift: Record<string, { recorded?: unknown; current?: unknown } | Record<string, unknown>>;
  seed: number;
  note: string;
}

export interface ExperimentComparisonRow {
  experiment_id: string;
  kind: string;
  value: number | null;
  feature_version: number;
  runtime_seconds: number;
}

export interface ExperimentComparison {
  metric: string;
  rows: ExperimentComparisonRow[];
  best: ExperimentComparisonRow | null;
  /** False when the experiments span different feature or preprocessing
   *  versions, in which case the metric is not the same metric. */
  comparable: boolean;
  warning: string | null;
}

// ---------------------------------------------------------------------------
// Features, detection and deep models
// ---------------------------------------------------------------------------

export interface FeatureMatrixInfo {
  name: string;
  path: string;
  rows: number;
  mb: number;
}

export interface FeatureMatrixBuild {
  name: string;
  path: string;
  rows: number;
  features: number;
  feature_version: number;
  feature_schema_hash: string;
  usable_rows: number;
  feature_names: string[];
}

export interface FeatureBatchReport {
  checkpoint: string;
  source_count: number;
  completed: number;
  failed: number;
  resumed: boolean;
  batches: number;
}

export interface FeatureMatrixBatchBuild extends FeatureMatrixBuild, FeatureBatchReport {}

export interface StageBResult {
  experiment_id: string;
  ready: boolean;
  reason?: string;
  checkpoint: string;
  dataset_fingerprint: string;
  resumed?: boolean;
  seeds?: number[];
  aggregate?: Array<Record<string, unknown>>;
  caveat?: string;
}

export interface DetectorSummary {
  name: string;
  flagged: number;
  score_mean: number;
  score_max: number;
}

export interface DetectionResult {
  rows_scored: number;
  rows_skipped: number;
  contamination: number;
  detectors: DetectorSummary[];
  ranking_path: string;
  candidates: Array<Record<string, unknown>>;
}

export interface DeepTrainReport {
  kind: string;
  device: string;
  device_reason: string;
  epochs_run: number;
  best_epoch: number;
  best_val_loss: number;
  train_losses: number[];
  val_losses: number[];
  batch_size: number;
  accumulation_steps: number;
  parameters: number;
  amp_enabled: boolean;
  seconds: number;
  checkpoint: string | null;
  seed: number;
  sequences?: { rows: number; length: number; channels: number; mean_coverage: number };
  /** Set instead of a report when there were too few usable sequences. */
  error?: string;
  rows?: number;
}

export interface SweepInterval {
  mean: number;
  std: number;
  ci95: [number, number];
}

export interface SweepTrial {
  parameters: Record<string, number | number[]>;
  model_parameters: number;
  scored_seeds: number;
  roc_auc: SweepInterval | null;
  average_precision: SweepInterval | null;
  seconds: number;
  note: string;
}

export interface SweepResult {
  experiment_id: string;
  kind: string;
  seeds: number[];
  rows: number;
  trials: SweepTrial[];
  best: Record<string, number | number[]> | null;
  /** False when the leading configuration's seed interval overlaps the
   *  runner-up's — the ranking is then not evidence of a better setting. */
  separated: boolean;
  note: string;
}

export interface MethodScore {
  name: string;
  roc_auc: number;
  average_precision: number;
  precision_at_k: number;
  recall_at_k: number;
  seconds: number;
  note: string;
}

export interface DeepComparison {
  injection: { rows: number; injected: number; kinds: Record<string, number> };
  methods: MethodScore[];
  best_method: string | null;
  error?: string;
  rows?: number;
}

// ---------------------------------------------------------------------------
// Cross-survey engine (plan section 15)
// ---------------------------------------------------------------------------

export interface CrossmatchGroup {
  surveys: string[];
  independent_surveys: number;
  resolved_surveys: number;
  members: Record<string, string>;
  separations_arcsec: Record<string, number>;
  ambiguous: string[];
  blended: string[];
  match_radius_arcsec: number;
}

export interface CrossmatchResult {
  summary: {
    groups: number;
    multi_survey: number;
    ambiguous: number;
    resolved_multi_survey: number;
    by_survey_count: Record<string, number>;
    grouping_bias?: {
      anchor_survey: string | null;
      anchor_policy?: "empty" | "largest_catalogue" | "explicit";
      requested_anchor_survey?: string | null;
      survey_counts: Record<string, number>;
      groups: number;
      anchor_share: number | null;
      matched_share: Record<string, number>;
      warning: string;
    };
  };
  groups: CrossmatchGroup[];
}

export interface SurveyView {
  survey: string;
  object_id: string;
  band: string;
  value_kind: string;
  points: number;
  reduced_chi2: number | null;
  best_period_days: number | null;
  period_snr: number | null;
  robust_amplitude: number | null;
  fractional_amplitude: number | null;
  baseline_days: number | null;
}

export interface CrossSurveyProfile {
  independent_surveys: number;
  resolved_surveys: number;
  views: SurveyView[];
  separations_arcsec: Record<string, number>;
  ambiguous: string[];
  blended: string[];
  consistency: number;
  weight_version: number;
  /** Fraction of the total weight that was available. A consistency computed
   *  from 0.90 of the weight is not comparable with one computed from all. */
  weight_used: number;
  /** How often two unrelated periods pass the alias-tolerant agreement test. */
  period_fap: number | null;
  components: Record<string, number>;
  /** What each component is worth, and what it actually contributed. Raw
   *  component scores alone cannot be ranked against each other. */
  weights: Record<string, number>;
  weighted: Record<string, number>;
  notes: string[];
}

export interface ProfilesResult {
  profiled: number;
  profiles: CrossSurveyProfile[];
}

export interface FrameOffset {
  scale_seconds: number;
  reference_seconds: number;
  total_seconds: number;
}

// ---------------------------------------------------------------------------
// Curves, FITS, manifests, labels, profiling
// ---------------------------------------------------------------------------

export interface BinnedCurve {
  bins: number;
  bin_days: number;
  time: number[];
  value: number[];
  value_err: number[];
  value_kind: "mag" | "flux";
  band: string;
}

export interface FitsDescription {
  path: string;
  size_mb: number;
  hdus: Array<{
    index: number;
    name: string;
    type: string;
    shape: number[];
    is_image: boolean;
  }>;
}

export interface FitsHeader {
  hdu: number;
  summary: Record<string, string | number | boolean>;
  cards: Record<string, string | number | boolean>;
}

export interface ImageFeaturePayload {
  schema_version: number;
  source: { path: string; sha256: string; hdu: number; shape: number[]; filter: string };
  features: Record<string, number | null>;
  quality: Record<string, number | string>;
  output_path?: string;
}

export interface SpectralFeaturePayload {
  schema_version: number;
  source: Record<string, unknown>;
  frame: string;
  units: string;
  features: Record<string, number>;
  quality: Record<string, number | string | boolean>;
  output_path?: string;
}

export interface SidecarInfo {
  kind: "image" | "spectral";
  path: string;
  rows: number;
  columns: string[];
  schema_version: number;
}

export interface SidecarJoinReport {
  kind: "image" | "spectral";
  base_rows: number;
  sidecar_rows: number;
  matched_rows: number;
  missing_rows: number;
  match_rate: number | null;
  feature_columns: string[];
  schema_version: number;
}

export interface ManifestSummary {
  dataset_id: string;
  created_utc: string;
  surveys: string[];
  objects: number;
  content_hash: string;
}

export interface LabelSummary {
  total: number;
  by_label: Record<string, number>;
}

/** Gated: reports `ready: false` with the shortfall rather than a metric
 *  computed from too few human labels to mean anything. */
export interface ReviewEvaluation {
  ready: boolean;
  reason?: string;
  minimum_labels: number;
  minimum_per_class: number;
  labels: number;
  positives: number;
  negatives: number;
  precision?: number;
  recall?: number;
  f1?: number;
  roc_auc?: number;
  average_precision?: number;
  threshold?: number;
}

export interface ProfileReport {
  feature_extraction: Record<string, unknown>;
  pipeline_stages: Record<string, unknown>;
  array_ops: Record<string, unknown>;
  gpu: Record<string, unknown>;
}

export interface FeatureNames {
  names: string[];
  feature_version: number;
}

export interface RankerModel {
  model_name: string;
  kind: string;
  created_utc: string;
  model_sha256: string;
  label_snapshot_hash: string;
}

export const engine = {
  projects: (includeArchived = true) =>
    invoke<ResearchProject[]>("engine_projects", { includeArchived }),
  projectCreate: (args: CreateProjectArgs) =>
    invoke<ResearchProject>("engine_project_create", {
      name: args.name,
      projectId: args.projectId,
      description: args.description,
      selectedSurveys: args.selectedSurveys,
      queryRegions: args.queryRegions,
      tags: args.tags,
      dataRoot: args.dataRoot,
    }),
  projectOpen: (projectId: string) =>
    invoke<ResearchProject>("engine_project_open", { projectId }),
  projectUpdate: (projectId: string, patch: Partial<Omit<CreateProjectArgs, "projectId">>) =>
    invoke<ResearchProject>("engine_project_update", {
      projectId,
      patch: {
        name: patch.name,
        description: patch.description,
        selected_surveys: patch.selectedSurveys,
        query_regions: patch.queryRegions,
        tags: patch.tags,
        data_root: patch.dataRoot,
      },
    }),
  projectArchive: (projectId: string, archived = true) =>
    invoke<ResearchProject>("engine_project_archive", { projectId, archived }),
  projectValidate: (projectId: string) =>
    invoke<ProjectValidation>("engine_project_validate", { projectId }),
  surveys: () => invoke<SurveyInfo[]>("engine_surveys"),
  eventProviders: () => invoke<EventProviderInfo[]>("engine_event_providers"),
  eventIngest: (args: {
    provider: string; payload: unknown; release?: string; packetId?: string;
    packetVersion?: string; receivedUtc?: string; projectId?: string;
  }) => invoke<EventPacket>("engine_event_ingest", {
    provider: args.provider,
    payload: args.payload,
    release: args.release,
    packetId: args.packetId,
    packetVersion: args.packetVersion,
    receivedUtc: args.receivedUtc,
    projectId: args.projectId,
  }),
  events: (args: { provider?: string; eventId?: string; limit?: number; packets?: boolean; projectId?: string } = {}) =>
    invoke<Array<EventCluster | EventPacket>>("engine_events", {
      provider: args.provider,
      eventId: args.eventId,
      limit: args.limit ?? 500,
      packets: args.packets ?? false,
      projectId: args.projectId,
    }),
  eventPacket: (packetKey: string, includeRaw = false, projectId?: string) =>
    invoke<EventPacket>("engine_event_packet", { packetKey, includeRaw, projectId }),
  eventReplay: (args: { provider?: string; eventId?: string; limit?: number; projectId?: string } = {}) =>
    invoke<EventPacket[]>("engine_event_replay", {
      provider: args.provider, eventId: args.eventId, limit: args.limit ?? 100,
      projectId: args.projectId,
    }),
  eventAssociate: (args: {
    name?: string; provider?: string; eventId?: string; radiusArcsec?: number;
    windowDays?: number; allowUnknownTime?: boolean; projectId?: string;
  } = {}) => invoke<Record<string, unknown>>("engine_event_associate", {
    name: args.name ?? "default", provider: args.provider, eventId: args.eventId,
    radiusArcsec: args.radiusArcsec ?? 30, windowDays: args.windowDays ?? 30,
    allowUnknownTime: args.allowUnknownTime ?? false, projectId: args.projectId,
  }),
  alertProviders: () => invoke<AlertProviderInfo[]>("engine_alert_providers"),
  alertStatus: (projectId?: string) => invoke<Record<string, unknown>>("engine_alert_status", { projectId }),
  alertPoll: (args: {
    provider: string; endpoint?: string; cursor?: string; limit?: number;
    offline?: boolean; payload?: unknown; params?: Record<string, unknown>; projectId?: string;
  }) => invoke<AlertPollResult>("engine_alert_poll", {
    provider: args.provider, endpoint: args.endpoint, cursor: args.cursor,
    limit: args.limit ?? 100, offline: args.offline ?? false,
    payload: args.payload, params: args.params, projectId: args.projectId,
  }),
  tapStatus: (projectId?: string) => invoke<Record<string, unknown>>("engine_tap_status", { projectId }),
  tapQuery: (args: {
    service: string; adql: string; release?: string; maxRows?: number;
    format?: "csv" | "votable"; refresh?: boolean; offline?: boolean;
    timeout?: number; projectId?: string;
  }) => invoke<TapResult>("engine_tap_query", {
    service: args.service, adql: args.adql, release: args.release ?? "unknown",
    maxRows: args.maxRows ?? 200, format: args.format ?? "csv",
    refresh: args.refresh ?? false, offline: args.offline ?? false,
    timeout: args.timeout ?? 60, projectId: args.projectId,
  }),
  readiness: () => invoke<ReadinessStatus>("engine_readiness"),
  curvesList: (survey?: string, limit = 500, projectId?: string) =>
    invoke<CurveSummary[]>("engine_curves_list", { survey, limit, projectId }),
  curveGet: (path: string, maxPoints = 2000, frame?: "BJD_TDB") =>
    invoke<CurvePayload>("engine_curve_get", { path, maxPoints, frame }),
  curveFold: (path: string, periodDays: number, epoch?: number) =>
    invoke<FoldedCurve>("engine_curve_fold", { path, periodDays, epoch }),
  storeUsage: () =>
    invoke<{ surveys: Record<string, { curves: number; gb: number }>; dataset: DatasetStatus }>(
      "engine_store_usage",
    ),
  acquire: (args: AcquireArgs) =>
    invoke<AcquisitionResult>("engine_acquire", {
      raDeg: args.raDeg,
      decDeg: args.decDeg,
      radiusArcsec: args.radiusArcsec,
      surveys: args.surveys,
      limit: args.limit,
      projectId: args.projectId,
    }),
  ping: () => invoke<{ pong: boolean; protocol: number }>("engine_ping"),
  hardware: () => invoke<DeviceReport>("engine_hardware"),
  paths: () => invoke<EnginePaths>("engine_paths"),
  versions: () => invoke<Record<string, string>>("engine_versions"),
  cacheStatus: () => invoke<CacheStatus>("engine_cache_status"),
  cacheEnforce: () => invoke<CacheStatus>("engine_cache_enforce"),
  pipeline: (name = "default", top = 200, projectId?: string, anchorSurvey?: string) =>
    invoke<PipelineResult>("engine_pipeline", { name, top, projectId, anchorSurvey }),
  candidates: (name = "default", top = 50, projectId?: string) =>
    invoke<{ count: number; candidates: Candidate[] }>("engine_candidates", { name, top, projectId }),
  candidatesSpatial: (name = "default", top = 200, projectId?: string) =>
    invoke<SpatialResult>("engine_candidates_spatial", { name, top, projectId }),
  candidate: (candidateId: string, name = "default", projectId?: string) =>
    invoke<Candidate>("engine_candidate", { candidateId, name, projectId }),
  candidateTimeline: (candidateId: string, name = "default", projectId?: string) =>
    invoke<CandidateTimeline>("engine_candidate_timeline", {
      candidateId, name, projectId, radiusArcsec: 30, maxCurves: 24, maxPoints: 180,
    }),
  label: (candidateId: string, label: string, note = "", projectId?: string) =>
    invoke<{ candidate_id: string; label: string; note: string }>("engine_label", {
      candidateId, label, note, projectId,
    }),
  exportCandidates: (format: "csv" | "fits" | "pdf", name = "default", projectId?: string) =>
    invoke<{ path: string; count: number }>("engine_candidates_export", { format, name, projectId }),
  catalogStatus: () => invoke<CatalogStatus>("engine_catalog_status"),
  catalogEnrich: (name = "default", offline = false, refresh = false, projectId?: string) =>
    invoke<CatalogEnrichmentResult>("engine_catalog_enrich", {
      name, offline, refresh, includeTns: true, radiusArcsec: 2, projectId,
    }),
  literatureStatus: () => invoke<LiteratureStatus>("engine_literature_status"),
  literatureSearch: (args: {
    objectId?: string; terms?: string[]; eventIds?: string[]; providers?: string[];
    limit?: number; refresh?: boolean; offline?: boolean; projectId?: string;
  } = {}) => invoke<LiteratureSearchResult>("engine_literature_search", {
    objectId: args.objectId ?? "", terms: args.terms ?? [], eventIds: args.eventIds ?? [],
    providers: args.providers ?? ["ads", "arxiv"], limit: args.limit ?? 20,
    refresh: args.refresh ?? false, offline: args.offline ?? false, projectId: args.projectId,
  }),
  literatureEnrich: (name = "default", offline = false, refresh = false,
                     includeArxiv = true, limit = 20, projectId?: string) =>
    invoke<Record<string, unknown>>("engine_literature_enrich", {
      name, offline, refresh, includeArxiv, limit, projectId,
    }),
  physicalCharacterize: (photometry: Record<string, unknown>, extinction?: Record<string, unknown>) =>
    invoke<PhysicalCharacterization>("engine_physical_characterize", {
      photometry, extinction, source: "caller",
    }),
  physicalEnrich: (name = "default", extinction?: Record<string, unknown>, projectId?: string) =>
    invoke<Record<string, unknown>>("engine_physical_enrich", { name, extinction, projectId }),
  digitalTwinFitProfile: (survey: string, limit = 500) =>
    invoke<SurveyProfileSummary>("engine_digital_twin_fit_profile", { survey, limit }),
  digitalTwinSample: (survey: string, limit = 500, n = 50, seed = 42) =>
    invoke<DigitalTwinSample>("engine_digital_twin_sample", { survey, limit, n, seed }),
  digitalTwinEvaluateDistance: (survey: string, limit = 500, seed = 42) =>
    invoke<DigitalTwinDistance>("engine_digital_twin_evaluate_distance", { survey, limit, seed }),
  digitalTwinEvaluateTransfer: (survey: string, limit = 500, seeds = [17, 29, 43],
                                epochs = 15, fraction = 0.1) =>
    invoke<DigitalTwinTransferResult>("engine_digital_twin_evaluate_transfer", {
      survey, limit, seeds, epochs, fraction,
    }),
  gwEvents: (catalog = "GWTC-1-confident", refresh = false, offline = false) =>
    invoke<GwEventsResult>("engine_gw_events", { catalog, refresh, offline }),
  gwEnrich: (name = "default", catalog = "GWTC-1-confident", windowDays = 30,
            refresh = false, offline = false, projectId?: string) =>
    invoke<GwEnrichmentResult>("engine_gw_enrich", {
      name, catalog, windowDays, refresh, offline, projectId,
    }),
  frbEvents: (refresh = false, offline = false) =>
    invoke<FrbEventsResult>("engine_frb_events", { refresh, offline }),
  frbEnrich: (name = "default", windowDays = 1, sigmaThreshold = 3,
             refresh = false, offline = false, projectId?: string) =>
    invoke<FrbEnrichmentResult>("engine_frb_enrich", {
      name, windowDays, sigmaThreshold, refresh, offline, projectId,
    }),
  tnsCredentialsConfigure: (apiKey: string, botId = "", botName = "ASTRA") =>
    invoke<{ configured: boolean; backend: string }>("engine_tns_credentials_configure", {
      apiKey, botId, botName,
    }),
  tnsCredentialsClear: () => invoke<{ cleared: boolean }>("engine_tns_credentials_clear"),
  rankerTrain: (name = "default", modelName = "calibrated-logistic", seed = 42) =>
    invoke<RankerResult>("engine_ranker_train", { name, modelName, seed }),
  rankerApply: (name = "default", modelName = "calibrated-logistic") =>
    invoke<RankerResult>("engine_ranker_apply", { name, modelName }),
  rankerList: () => invoke<Array<Record<string, unknown>>>("engine_ranker_list"),
  /** `survey` stratifies the feature and detector ablations. Leaving it unset
   *  pools ZTF and TESS, which measures the mixture as much as the method. */
  ablation: (fraction = 0.1, seed = 42, survey?: string, projectId?: string) =>
    invoke<AblationResult>("engine_ablation", { fraction, seed, survey, projectId }),
  ablationRepeated: (fraction = 0.1, seeds = [17, 29, 43, 59, 71], survey?: string, projectId?: string) =>
    invoke<AblationResult>("engine_ablation_repeated", { fraction, seeds, survey, projectId }),
  stageBCompare: (survey?: string, projectId?: string, options: {
    seeds?: number[]; fraction?: number; strength?: number; limit?: number;
    mode?: "time" | "season" | "phase"; includeDeep?: boolean; epochs?: number; checkpoint?: string;
  } = {}) => invoke<StageBResult>("engine_stageb_compare", {
    survey, projectId,
    seeds: options.seeds ?? [17, 29, 43, 59, 71],
    fraction: options.fraction ?? 0.1,
    strength: options.strength ?? 6,
    limit: options.limit ?? 10_000,
    mode: options.mode ?? "time",
    includeDeep: options.includeDeep ?? false,
    epochs: options.epochs ?? 20,
    checkpoint: options.checkpoint,
  }),
  fitsImage: (path: string, hdu?: number, contrast = 0.25) =>
    invoke<{ shape: [number, number]; pixels: number[]; stats: Record<string, number> }>(
      "engine_fits_image", { path, hdu, contrast }),
  imageFeatures: (path: string, projectId?: string, hdu?: number, targetX?: number, targetY?: number,
                  identity?: Partial<Record<"survey" | "release" | "object_id" | "band", string>>) =>
    invoke<ImageFeaturePayload>("engine_image_features", {
      path, projectId, hdu, targetX, targetY,
      ...identity,
    }),
  spectralFeatures: (path: string, projectId?: string,
                     identity?: Partial<Record<"survey" | "release" | "object_id" | "band", string>>) =>
    invoke<SpectralFeaturePayload>("engine_spectral_features", { path, projectId, ...identity }),
  sidecarsList: (projectId?: string) => invoke<SidecarInfo[]>("engine_sidecars_list", { projectId }),
  sidecarSave: (kind: "image" | "spectral", payloads: unknown[], projectId?: string,
                name = "default", identities?: unknown[]) =>
    invoke<SidecarInfo>("engine_sidecar_save", { kind, payloads, projectId, name, identities }),
  sidecarJoin: (path: string, kind: "image" | "spectral", identities: unknown[]) =>
    invoke<{ rows: Array<Record<string, unknown>>; report: SidecarJoinReport }>(
      "engine_sidecar_join", { path, kind, identities }),
  ztfImageSearch: (raDeg: number, decDeg: number, sizeArcsec = 50, limit = 25) =>
    invoke<ZtfImageMetadata[]>("engine_ztf_images_search", {
      raDeg, decDeg, sizeArcsec, limit, productKind: "science", release: "dr",
    }),
  ztfImageDownload: (args: ZtfCutoutArgs) =>
    invoke<Pick<EngineJob, "job_id" | "method" | "status">>("engine_ztf_images_download", {
      raDeg: args.raDeg,
      decDeg: args.decDeg,
      metadata: args.metadata,
      sizeArcsec: args.sizeArcsec ?? 50,
      productKind: args.productKind ?? "science",
      release: args.release ?? "dr",
      projectId: args.projectId,
      maxBytes: args.maxBytes,
    }),
  tessTpfDownload: (args: TessTpfDownloadArgs) =>
    invoke<Pick<EngineJob, "job_id" | "method" | "status">>("engine_tess_tpf_download", {
      raDeg: args.raDeg,
      decDeg: args.decDeg,
      sector: args.sector,
      sizePixels: args.sizePixels,
      targetId: args.targetId,
      product: args.product ?? "SPOC",
      projectId: args.projectId,
      maxBytes: args.maxBytes,
      overwrite: args.overwrite ?? false,
    }),
  tessTpfPhotometry: (args: TessPhotometryArgs) =>
    invoke<TessPhotometryPayload>("engine_tess_tpf_photometry", {
      path: args.path,
      raDeg: args.raDeg,
      decDeg: args.decDeg,
      neighbors: args.neighbors ?? [],
      targetMag: args.targetMag,
      apertureRadiusPixels: args.apertureRadiusPixels ?? 1.5,
      qualityMask: args.qualityMask ?? 0,
      targetId: args.targetId,
      persist: args.persist ?? true,
      maxPoints: args.maxPoints ?? 5000,
    }),
  products: (projectId?: string, limit = 500) =>
    invoke<ImageProduct[]>("engine_products", { projectId, limit }),
  product: (productId: string) => invoke<ImageProduct>("engine_product", { productId }),
  jobSubmit: (method: string, params: Record<string, unknown> = {}, projectId?: string, idempotencyKey?: string) =>
    invoke<Pick<EngineJob, "job_id" | "method" | "status">>("engine_job_submit", {
      method, params, projectId, idempotencyKey,
    }),
  jobStatus: (jobId: string) => invoke<EngineJob>("engine_job_status", { jobId }),
  jobCancel: (jobId: string) => invoke<EngineJob>("engine_job_cancel", { jobId }),
  jobRetry: (jobId: string) => invoke<EngineJob>("engine_job_retry", { jobId }),
  jobs: (statuses?: EngineJob["status"][]) => invoke<EngineJob[]>("engine_jobs", { statuses }),

  // --- Research framework (plan sections 19-20) ----------------------------
  experiments: (projectId?: string) => invoke<ExperimentSummary[]>("engine_experiments", { projectId }),
  experiment: (experimentId: string, projectId?: string) =>
    invoke<ExperimentRecord>("engine_experiment", { experimentId, projectId }),
  experimentVerify: (experimentId: string, projectId?: string) =>
    invoke<ExperimentVerification>("engine_experiment_verify", { experimentId, projectId }),
  experimentCompare: (experimentIds: string[], metric = "roc_auc", projectId?: string) =>
    invoke<ExperimentComparison>("engine_experiment_compare", { experimentIds, metric, projectId }),

  // --- Features, detection, deep models ------------------------------------
  featuresList: (projectId?: string) => invoke<FeatureMatrixInfo[]>("engine_features_list", { projectId }),
  featuresBuild: (name = "default", survey?: string, limit?: number, projectId?: string) =>
    invoke<FeatureMatrixBuild>("engine_features_build", { name, survey, limit, projectId }),
  featuresBuildResumable: (name = "default", survey?: string, limit?: number,
                           batchSize = 256, checkpoint?: string, projectId?: string) =>
    invoke<FeatureMatrixBatchBuild>("engine_features_build_resumable", {
      name, survey, limit, batchSize, checkpoint, projectId,
    }),
  featureNames: () => invoke<FeatureNames>("engine_feature_names"),
  featureCacheClear: () => invoke<{ cleared: number }>("engine_feature_cache_clear"),
  significanceCalibrate: (args: {
    scores: number[]; referenceScores?: number[]; threshold?: number;
    strata?: Record<string, unknown>; name?: string; projectId?: string;
  }) => invoke<SignificanceReport>("engine_significance_calibrate", {
    scores: args.scores,
    referenceScores: args.referenceScores,
    threshold: args.threshold,
    strata: args.strata,
    name: args.name ?? "default",
    projectId: args.projectId,
  }),
  selectionEvaluate: (args: {
    records: Array<Record<string, unknown>>; dimensions?: string[];
    edges?: Record<string, number[]>; name?: string; projectId?: string;
    fitModel?: boolean; modelFeatures?: string[]; bootstrapSamples?: number; seed?: number;
  }) => invoke<SelectionReport>("engine_selection_evaluate", {
    records: args.records,
    dimensions: args.dimensions,
    edges: args.edges,
    fitModel: args.fitModel ?? false,
    modelFeatures: args.modelFeatures,
    bootstrapSamples: args.bootstrapSamples ?? 0,
    seed: args.seed ?? 42,
    name: args.name ?? "default",
    projectId: args.projectId,
  }),
  reviewNext: (name = "default", limit = 20, projectId?: string) =>
    invoke<ReviewSelection[]>("engine_review_next", { name, limit, projectId }),
  followupPlan: (args: {
    raDeg: number; decDeg: number; startUtc?: string; durationHours?: number;
    latitudeDeg?: number; longitudeDeg?: number; minAltitudeDeg?: number;
    cadenceMinutes?: number; targetId?: string; twilightSunAltitudeDeg?: number;
    minMoonSeparationDeg?: number; maxMoonIllumination?: number; maxAirmass?: number;
    weather?: Array<Record<string, unknown>> | Record<string, unknown>;
    facilityName?: string; facilityConstraints?: Record<string, unknown>;
  }) => invoke<FollowupPlan>("engine_followup_plan", {
    raDeg: args.raDeg,
    decDeg: args.decDeg,
    startUtc: args.startUtc,
    durationHours: args.durationHours ?? 12,
    latitudeDeg: args.latitudeDeg ?? 43.65,
    longitudeDeg: args.longitudeDeg ?? -79.38,
    minAltitudeDeg: args.minAltitudeDeg ?? 30,
    cadenceMinutes: args.cadenceMinutes ?? 10,
    targetId: args.targetId,
    twilightSunAltitudeDeg: args.twilightSunAltitudeDeg ?? -18,
    minMoonSeparationDeg: args.minMoonSeparationDeg ?? 0,
    maxMoonIllumination: args.maxMoonIllumination ?? 1,
    maxAirmass: args.maxAirmass,
    weather: args.weather,
    facilityName: args.facilityName,
    facilityConstraints: args.facilityConstraints,
  }),
  detect: (name = "default", contamination?: number, top = 50, projectId?: string) =>
    invoke<DetectionResult>("engine_detect", { name, contamination, top, projectId }),
  deepTrain: (name = "default", kind: "autoencoder" | "vae" | "transformer" = "autoencoder",
              survey?: string, epochs?: number) =>
    invoke<DeepTrainReport>("engine_deep_train", { name, kind, survey, epochs }),
  deepCompare: (survey?: string, fraction?: number, epochs?: number) =>
    invoke<DeepComparison>("engine_deep_compare", { survey, fraction, epochs }),
  deepSweep: (kind: "autoencoder" | "vae" | "transformer" = "autoencoder", survey?: string,
              seeds = [17, 29, 43], epochs = 20, mode = "time") =>
    invoke<SweepResult>("engine_deep_sweep", { kind, survey, seeds, epochs, mode }),

  // --- Cross-survey engine (plan section 15) -------------------------------
  crossmatch: (radiusArcsec?: number, projectId?: string, anchorSurvey?: string) =>
    invoke<CrossmatchResult>("engine_crossmatch", { radiusArcsec, projectId, anchorSurvey }),
  profiles: (radiusArcsec?: number, top?: number, projectId?: string, anchorSurvey?: string) =>
    invoke<ProfilesResult>("engine_profiles", { radiusArcsec, top, projectId, anchorSurvey }),
  frameOffset: (raDeg: number, decDeg: number, timeSystem = "HJD_UTC") =>
    invoke<FrameOffset>("engine_frame_offset", { raDeg, decDeg, timeSystem }),

  // --- Curves, FITS, manifests, labels, profiling --------------------------
  curveBin: (path: string, binDays: number) =>
    invoke<BinnedCurve>("engine_curve_bin", { path, binDays }),
  fitsDescribe: (path: string) => invoke<FitsDescription>("engine_fits_describe", { path }),
  fitsHeader: (path: string, hdu = 0) =>
    invoke<FitsHeader>("engine_fits_header", { path, hdu }),
  manifests: (projectId?: string) =>
    invoke<ManifestSummary[]>("engine_manifests", { projectId }),
  labelSummary: (projectId?: string) => invoke<LabelSummary>("engine_label_summary", { projectId }),
  candidatesEvaluate: (name = "default", projectId?: string) =>
    invoke<ReviewEvaluation>("engine_candidates_evaluate", { name, projectId }),
  profileRun: (limit = 100) => invoke<ProfileReport>("engine_profile", { limit }),
};
