/** Wire types for every `engine.*` RPC call in engine.ts.
 *
 * Pulled out of engine.ts (which had grown to 2000+ lines split roughly
 * evenly between these declarations and the RPC methods themselves) purely
 * to keep each file a manageable size; nothing here has any behavior.
 */

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

export interface HabitableZone {
  recent_venus_au: number;
  runaway_greenhouse_au: number;
  moist_greenhouse_au: number;
  maximum_greenhouse_au: number;
  early_mars_au: number;
  conservative_inner_au: number;
  conservative_outer_au: number;
  optimistic_inner_au: number;
  optimistic_outer_au: number;
  extrapolated: boolean;
}

export interface HabitabilityScore {
  habitable_zone: HabitableZone;
  hz_position: number | null;
  in_conservative_hz: boolean | null;
  in_optimistic_hz: boolean | null;
  esi_interior: number | null;
  esi_surface_from_teq: number | null;
  esi_global: number | null;
  equilibrium_temp_k: number | null;
  warnings: string[];
  quality: "usable" | "insufficient";
  planet_name: string;
  host_name: string;
}

export interface HabitabilityRanking {
  count: number;
  planets: HabitabilityScore[];
}

export interface OrbitalElements {
  semi_major_axis_au: number;
  eccentricity: number;
  inclination_deg: number;
  raan_deg: number;
  argument_of_perihelion_deg: number;
  mean_anomaly_deg: number;
  epoch_mjd: number;
}

export interface NeoHazardAssessment {
  moid_au: number | null;
  tisserand_jupiter: number | null;
  dynamical_class: "asteroidal" | "comet-like" | "unknown";
  absolute_magnitude: number | null;
  diameter_km: number | null;
  diameter_km_range: [number, number] | null;
  is_pha: boolean;
  moid_detail: { moid_au: number; method: string } | null;
}

export interface NeoCloseApproach {
  close_approach_mjd: number;
  distance_au: number;
  distance_lunar_distances: number;
  window_start_mjd: number;
  window_end_mjd: number;
  step_days: number;
}

export interface SeismicSolution {
  radius_rsun: number;
  mass_msun: number;
  logg_cgs: number;
  density_rhosun: number;
  radius_rsun_error: number | null;
  mass_msun_error: number | null;
}

export interface AsteroseismologyMeasurement {
  numax_uhz: number | null;
  numax_uhz_error: number | null;
  delta_nu_uhz: number | null;
  solution: SeismicSolution | null;
  quality: "usable" | "insufficient";
  warnings: string[];
}

export interface BiosignatureFitResult {
  params: {
    temperature_k: number;
    mean_molecular_weight: number;
    reference_radius_rjup: number;
    log10_cloud_pressure_bar: number | null;
    abundances: [string, number][];
  };
  chi2: number;
  reduced_chi2: number;
  n_points: number;
  converged: boolean;
  note: string;
}

export interface BiosignatureDetection {
  molecule: string;
  delta_bic: number;
  delta_chi2: number;
  full_chi2: number;
  null_chi2: number;
  n_points: number;
  log10_amplitude: number | null;
  detected: boolean;
}

export interface BiosignatureDetectResult {
  significances: Record<string, BiosignatureDetection>;
  disequilibrium: {
    ch4_detected: boolean;
    oxidant_detected: boolean;
    co_detection_flag: boolean;
    caveat: string;
  };
}

export interface BiosignatureSynthesis {
  wavelength_um: number[];
  depth: number[];
  error: number[];
  truth_depth: number[];
}

export interface TechnosignatureHit {
  frequency_hz: number;
  drift_rate_hz_s: number;
  snr: number;
  freq_channel_index: number;
  drift_index: number;
}

export interface TechnosignatureSearchResult {
  n_drift_trials: number;
  max_drift_hz_s: number;
  snr_threshold: number;
  hits: TechnosignatureHit[];
  truth: { drift_rate_hz_s: number; snr: number; start_channel: number | null };
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

/** discard_pile.DiscardRecord — a coherent run of epochs ZTF's own real
 * per-epoch catflags discarded before this candidate was ever assembled,
 * found via discard.scan (surveys.ztf.ZTFConnector.
 * fetch_light_curves_with_quality). ZTF-only: no other connector exposes
 * an equivalent unfiltered-epoch fetch. */
export interface DiscardRecord {
  object_id: string;
  survey: string;
  band: string;
  flag_category: string;
  epoch_count: number;
  time_start: number;
  time_end: number;
  magnitude_offset: number;
  max_step: number;
  coherent: boolean;
}

export interface DiscardScanResult {
  object_id: string;
  records: DiscardRecord[];
}

/** followup.request/result/history — tracking layer on top of the stateless
 * `followup.plan` visibility calculation. A candidate can accumulate several
 * requests over its lifetime, so history is a list, not a single record. */
export interface FollowupRequestEntry {
  request_id: string;
  candidate_key: string;
  facility_name: string;
  note: string;
  status: "requested" | "observed" | "no_show" | "cancelled";
  requested_utc: string;
  result_note: string | null;
  result_utc: string | null;
}

/** candidates.vote/votes/vote_promote — multi-reviewer citizen-science
 * voting (broadcast.py's sibling: this one is append-only per candidate,
 * unlike the single-row `labels` table `label`/`labelSummary` use). Vote
 * promotion is always an explicit, gated action, never automatic. */
export interface LabelVote {
  vote_id: string;
  candidate_key: string;
  reviewer_id: string;
  label: string;
  note: string;
  recorded_utc: string;
}

export interface VoteTally {
  total: number;
  by_label: Record<string, number>;
  majority_label: string | null;
  agreement_fraction: number | null;
}

export interface VotePromotion {
  promoted: boolean;
  reason?: string;
  label?: string;
  minimum_votes: number;
  minimum_agreement: number;
  votes: number;
  agreement_fraction: number | null;
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

/** candidates.explain — occlusion-based per-candidate feature attribution
 * (attribution.py). Reruns the unsupervised anomaly ensemble once per
 * feature, so this is a real, explicit research action, not a refresh. */
export interface AttributionResult {
  candidate_id: string;
  explainable: boolean;
  reason?: string;
  path?: string;
  baseline_score?: number;
  narrative?: string;
  components?: Array<{
    feature: string; value: number; typical: number; impact: number;
    label?: string; unit?: string | null; description?: string;
    impact_mean?: number; impact_std?: number; impact_min?: number; impact_max?: number;
    stable?: boolean;
  }>;
}

/** candidates.broadcast — writes a LOCAL, stable-path feed file of
 * high-confidence candidates (broadcast.py). Not a network push; there is
 * no publish/hosting infrastructure in this app. */
export interface BroadcastFeedResult {
  path: string;
  count: number;
  threshold: number;
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

/** research.bundle.build/verify — a signed, content-hashed evidence bundle
 * over a sealed dataset manifest plus the experiments run against it. */
export interface ReproducibilityBundle {
  dataset_id: string;
  manifest_content_hash: string;
  environment: Record<string, unknown>;
  experiment_record_refs: string[];
  bundle_hash: string;
  signature_hex: string | null;
  public_key_hex: string | null;
  path?: string;
}

export interface ReproducibilityBundleVerification {
  dataset_id: string;
  valid: boolean;
  note?: string;
}

/** research.benchmark.run — one ResultRecord per (method, seed) scored. */
export interface ResearchResultRecord {
  experiment_id: string;
  benchmark_id: string;
  split_id: string;
  dataset_manifest_hash: string;
  metric: string;
  value: number;
  sample_count: number;
  confidence_interval: [number, number];
  seed: number;
  synthetic: boolean;
  artifact_refs: string[];
  notes: string;
}

export interface ResearchBenchmarkRunResult {
  benchmark_id: string;
  split_id: string;
  experiment_id: string;
  results: ResearchResultRecord[];
  /** Rows the loaded feature matrix actually held after being filtered down
   *  to the dataset manifest's own object IDs (`scope_to_manifest`). */
  matrix_rows_scored: number;
  /** Rows the loaded matrix had that were NOT in the manifest and so were
   *  dropped before scoring -- a saved matrix can carry every stored curve
   *  for a survey, not only the ones this manifest's cone matched. */
  dropped_out_of_manifest_rows: number;
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
