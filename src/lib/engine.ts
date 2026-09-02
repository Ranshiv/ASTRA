import { invoke } from "@tauri-apps/api/core";
import type {
  AblationResult, AcquireArgs, AcquisitionResult, AlertPollResult,
  AlertProviderInfo, AsteroseismologyMeasurement, AttributionResult, BinnedCurve,
  BiosignatureDetectResult, BiosignatureFitResult, BiosignatureSynthesis, BroadcastFeedResult,
  CacheStatus, Candidate, CandidateTimeline, CatalogEnrichmentResult,
  CatalogStatus, CreateProjectArgs, CrossmatchResult, CurvePayload,
  CurveSummary, DatasetStatus, DeepComparison, DeepTrainReport,
  DetectionResult, DeviceReport, DigitalTwinDistance, DigitalTwinSample,
  DigitalTwinTransferResult, DiscardScanResult, EngineJob, EnginePaths,
  EventCluster, EventPacket, EventProviderInfo, ExperimentArmResolution, ExperimentComparison,
  ExperimentRecord, ExperimentSummary, ExperimentVerification, ExperimentVote, FeatureMatrixBatchBuild,
  FeatureMatrixBuild, FeatureMatrixInfo, FeatureNames, FitsDescription,
  FitsHeader, FoldedCurve, FollowupPlan, FollowupRequestEntry,
  FrameOffset, FrbEnrichmentResult, FrbEventsResult, GwEnrichmentResult,
  GwEventsResult, HabitabilityRanking, HabitabilityScore, ImageFeaturePayload,
  ImageProduct, LabelSummary, LabelVote, LiteratureSearchResult,
  LiteratureStatus, ManifestSummary, NeoCloseApproach, NeoHazardAssessment,
  OrbitalElements, PhysicalCharacterization, PipelineResult, ProfileReport,
  ProfilesResult, ProjectValidation, RankerResult, ReadinessStatus,
  ReproducibilityBundle, ReproducibilityBundleVerification, ResearchBenchmarkRunResult, ResearchProject,
  ReviewEvaluation, ReviewSelection, SeismicSolution, SelectionReport,
  SidecarInfo, SidecarJoinReport, SignificanceReport, SpatialResult,
  SpectralFeaturePayload, StageBResult, SurveyInfo, SurveyProfileSummary,
  SweepResult, TapResult, TechnosignatureSearchResult, TessPhotometryArgs,
  TessPhotometryPayload, TessTpfDownloadArgs, VotePromotion, VoteTally,
  ZtfCutoutArgs, ZtfImageMetadata,
} from "./engine-types";

export * from "./engine-types";

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
  candidateExplain: (candidateId: string, name = "default", projectId?: string, top = 10, stable = false) =>
    invoke<AttributionResult>("engine_candidate_explain", { candidateId, name, projectId, top, stable }),
  candidateTimeline: (candidateId: string, name = "default", projectId?: string) =>
    invoke<CandidateTimeline>("engine_candidate_timeline", {
      candidateId, name, projectId, radiusArcsec: 30, maxCurves: 24, maxPoints: 180,
    }),
  label: (candidateId: string, label: string, note = "", projectId?: string) =>
    invoke<{ candidate_id: string; label: string; note: string }>("engine_label", {
      candidateId, label, note, projectId,
    }),
  castLabelVote: (candidateId: string, reviewerId: string, label: string, note = "", projectId?: string) =>
    invoke<LabelVote>("engine_candidate_vote", { candidateId, reviewerId, label, note, projectId }),
  labelVotes: (candidateId: string, projectId?: string) =>
    invoke<{ votes: LabelVote[]; tally: VoteTally }>("engine_candidate_votes", { candidateId, projectId }),
  promoteVoteConsensus: (candidateId: string, projectId?: string) =>
    invoke<VotePromotion>("engine_candidate_vote_promote", { candidateId, projectId }),
  // Reviewer human-factors experiment (Direction 6, "the review UI as a
  // controlled experiment"): resolve a reviewer's arm before rendering the
  // score, then cast a vote that records the same arm.
  experimentArm: (candidateId: string, reviewerId: string, scoreLookup?: Record<string, number>) =>
    invoke<ExperimentArmResolution>("engine_review_experiment_arm", {
      candidateId, reviewerId, scoreLookup,
    }),
  castExperimentalVote: (
    candidateId: string, reviewerId: string, label: string,
    scoreLookup?: Record<string, number>, note = "", projectId?: string,
  ) =>
    invoke<ExperimentVote>("engine_review_experiment_vote", {
      candidateId, reviewerId, label, scoreLookup, note, projectId,
    }),
  exportCandidates: (format: "csv" | "fits" | "pdf", name = "default", projectId?: string) =>
    invoke<{ path: string; count: number }>("engine_candidates_export", { format, name, projectId }),
  broadcastFeed: (name = "default", threshold = 0.5, projectId?: string) =>
    invoke<BroadcastFeedResult>("engine_candidate_broadcast", { name, threshold, projectId }),
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
  habitabilityScore: (planetName: string, offline = false) =>
    invoke<HabitabilityScore>("engine_habitability_score", { planetName, offline }),
  habitabilityRank: (opts: { teffMin?: number; teffMax?: number; insolationMin?: number;
                            insolationMax?: number; maxRows?: number; limit?: number } = {}) =>
    invoke<HabitabilityRanking>("engine_habitability_rank", {
      teffMin: opts.teffMin, teffMax: opts.teffMax,
      insolationMin: opts.insolationMin, insolationMax: opts.insolationMax,
      maxRows: opts.maxRows ?? 500, limit: opts.limit ?? 50,
    }),
  neoAssess: (elements: OrbitalElements, opts: { earthElements?: OrbitalElements; apparentV?: number;
                                                  heliocentricAu?: number; geocentricAu?: number;
                                                  phaseAngleDeg?: number } = {}) =>
    invoke<NeoHazardAssessment>("engine_neo_assess", {
      elements, earthElements: opts.earthElements, apparentV: opts.apparentV,
      heliocentricAu: opts.heliocentricAu, geocentricAu: opts.geocentricAu,
      phaseAngleDeg: opts.phaseAngleDeg,
    }),
  neoCloseApproach: (elements: OrbitalElements, startMjd: number, endMjd: number, stepDays = 1.0) =>
    invoke<NeoCloseApproach>("engine_neo_close_approach", {
      elements, startMjd, endMjd, stepDays,
    }),
  asteroseismologyMeasure: (path: string, teffK?: number) =>
    invoke<AsteroseismologyMeasurement>("engine_asteroseismology_measure", { path, teffK }),
  asteroseismologySolve: (numaxUhz: number, deltaNuUhz: number, teffK: number,
                          errors: { numaxUhzError?: number; deltaNuUhzError?: number;
                                   teffKError?: number } = {}) =>
    invoke<SeismicSolution>("engine_asteroseismology_solve", {
      numaxUhz, deltaNuUhz, teffK,
      numaxUhzError: errors.numaxUhzError, deltaNuUhzError: errors.deltaNuUhzError,
      teffKError: errors.teffKError,
    }),
  technosignatureSearch: (opts: { driftRateHzS?: number; snr?: number; startChannel?: number;
                                  maxDriftHzS?: number; snrThreshold?: number; seed?: number } = {}) =>
    invoke<TechnosignatureSearchResult>("engine_technosignature_search", {
      driftRateHzS: opts.driftRateHzS, snr: opts.snr, startChannel: opts.startChannel,
      maxDriftHzS: opts.maxDriftHzS, snrThreshold: opts.snrThreshold, seed: opts.seed,
    }),
  biosignatureSynthesize: (opts: { stellarRadiusRsun: number; planetMassMjup: number;
                                   temperatureK?: number; referenceRadiusRjup?: number;
                                   abundances?: [string, number][];
                                   crossSections: Record<string, number>; nPoints?: number;
                                   wavelengthMinUm?: number; wavelengthMaxUm?: number;
                                   errorPpm?: number; seed?: number }) =>
    invoke<BiosignatureSynthesis>("engine_biosignature_synthesize", {
      stellarRadiusRsun: opts.stellarRadiusRsun, planetMassMjup: opts.planetMassMjup,
      temperatureK: opts.temperatureK, referenceRadiusRjup: opts.referenceRadiusRjup,
      abundances: opts.abundances ?? [], crossSections: opts.crossSections,
      nPoints: opts.nPoints, wavelengthMinUm: opts.wavelengthMinUm,
      wavelengthMaxUm: opts.wavelengthMaxUm, errorPpm: opts.errorPpm, seed: opts.seed,
    }),
  biosignatureFit: (wavelengthUm: number[], depth: number[], error: number[],
                    stellarRadiusRsun: number, planetMassMjup: number, molecules: string[],
                    crossSections: Record<string, number>, seed = 42) =>
    invoke<BiosignatureFitResult>("engine_biosignature_fit", {
      wavelengthUm, depth, error, stellarRadiusRsun, planetMassMjup, molecules,
      crossSections, seed,
    }),
  biosignatureDetect: (wavelengthUm: number[], depth: number[], error: number[],
                       stellarRadiusRsun: number, planetMassMjup: number, molecules: string[],
                       crossSections: Record<string, number>, seed = 42) =>
    invoke<BiosignatureDetectResult>("engine_biosignature_detect", {
      wavelengthUm, depth, error, stellarRadiusRsun, planetMassMjup, molecules,
      crossSections, seed,
    }),
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
  researchBundleBuild: (datasetId: string, experimentIds?: string[], projectId?: string) =>
    invoke<ReproducibilityBundle>("engine_research_bundle_build", {
      datasetId, experimentIds, projectId,
    }),
  researchBundleVerify: (datasetId: string, projectId?: string) =>
    invoke<ReproducibilityBundleVerification>("engine_research_bundle_verify", {
      datasetId, projectId,
    }),
  researchBenchmarkRun: (
    matrixName: string, benchmarkId: string, splitId: string, datasetId: string,
    injectionFraction = 0.1, projectId?: string,
  ) =>
    invoke<ResearchBenchmarkRunResult>("engine_research_benchmark_run", {
      matrixName, benchmarkId, splitId, datasetId, injectionFraction, projectId,
    }),

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
  reviewNext: (name = "default", limit = 20, projectId?: string, active = false) =>
    invoke<ReviewSelection[]>("engine_review_next", { name, limit, projectId, active }),
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
  followupRequest: (candidateId: string, facilityName = "", note = "", projectId?: string) =>
    invoke<FollowupRequestEntry>("engine_followup_request", {
      candidateId, facilityName, note, projectId,
    }),
  followupResult: (requestId: string, status: "observed" | "no_show" | "cancelled",
                   note = "", projectId?: string) =>
    invoke<FollowupRequestEntry>("engine_followup_result", {
      requestId, status, note, projectId,
    }),
  followupHistory: (candidateId: string, projectId?: string) =>
    invoke<FollowupRequestEntry[]>("engine_followup_history", { candidateId, projectId }),
  discardScan: (args: { objectId: string; raDeg: number; decDeg: number; minRunLength?: number }) =>
    invoke<DiscardScanResult>("engine_discard_scan", {
      objectId: args.objectId,
      raDeg: args.raDeg,
      decDeg: args.decDeg,
      minRunLength: args.minRunLength,
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
