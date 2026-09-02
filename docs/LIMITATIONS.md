# Limitations

Synthetic-only modules, dormant connectors, unvalidated assumptions, and
negative results for ASTRA's research evidence package. This document is
generated/maintained alongside `research/sources/module_status.yaml` and
`research/sources/connector_status.yaml`.

`docs/DEFERRED.txt` (a 79-entry, 6000-line limitations log going back to
the project's early phases) was retired on 2026-09-02 after an audit
verified that nearly every module and test file it named already existed
— what remained were genuine residual gaps, now folded into this document
below by category. Every item that could be closed with offline code work
was closed in that same pass (habitability's Kopparapu Eq. 4 eccentricity
correction, the real Barnes et al. 2016 Table 1 thermalization grid, SDSS
photometry columns, HEALPix survey-coverage footprints, the SIGPROC `.fil`
reader, the two-cadence-representation comparison driver, and more — see
git history around that date for the full list). What is recorded below
is what remains genuinely blocked on something outside this codebase, or
was explicitly, deliberately left out of scope.

## Scope decisions made explicitly this session

- **Acquisition scale**: the research corpus acquired this session is a
  small real demonstration (~hundreds of objects, one cone query), not the
  "tens of thousands of objects" first-release target the roadmap
  describes. `scripts/acquire-core-corpus.ps1` and
  `engine/astra/research/acquire.py: acquire_core_corpus()` are the same
  code path a larger, unattended run would use — scaling up is a
  follow-on run, not new code.
- **Artifact rejection benchmark**: now executed, at real but small scale.
  Three real TESS TPFs were pulled live from MAST and
  `artifact_bank_eval.evaluate_cross_group_auprc` run over the resulting
  30-patch, 2-camera-group bank (mean AUPRC 0.9872) — the project's first
  `synthetic: false` row in `research/results/metrics.jsonl`. See
  docs/BENCHMARKS.md and docs/RESULTS.md. A release-scale run (many more
  TPFs, more camera/CCD/night diversity, a larger balanced patch count) is
  still a follow-on, not new code — the same acquire-more-and-rerun pattern
  as the acquisition-scale note above. `research/sources/module_status.yaml`
  should be updated to reflect `artifact_bank`/`artifact_bank_eval` moving
  from `candidate-only` to `benchmarked`.
- **P1/P2 tracks** (transient classification, exoplanet/transit,
  spectroscopy, SED/dust, microlensing, host association, X-ray/radio,
  GW event association, federated training, polarization, lensing,
  advanced active review): unchanged this session — remain candidate-only,
  opt-in, and out of the production score, per the roadmap's promotion
  order.

## Known infrastructure gaps

- **UI panels do not yet render research evidence.** `src/lib/engine.ts`
  has typed client wrappers (`researchBundleBuild`, `researchBundleVerify`,
  `researchBenchmarkRun`) and `src-tauri/src/lib.rs` has the matching Tauri
  commands, both wired through to the `research.bundle.*`/
  `research.benchmark.run` RPC methods and type-checked. `src/components/
  ReportsView.tsx` and `ExplainPanel.tsx` do not yet call them or display
  benchmark ID/split/CI/provenance — that visual wiring, and a dev-server
  walkthrough of it, was out of scope for this session.

- **Cassette layer now covers `netclient.download` as well as `get`, real
  evidence included.** `download()` records/replays through the same
  `research/cassettes.py` identity/checksum path as `get`, keyed by
  `(provider, "DOWNLOAD", url, params)`; replay writes the cassette's
  checksummed body through the same atomic temp-file-then-`os.replace` path
  the live download uses, and still enforces `max_bytes`. This is no longer
  only unit-tested: a real 28.5 MB TESS TPF was pulled live from MAST
  through `tess_pixels.download_tpf`, cassette-recorded, and replayed
  offline with an identical `fits_sha256` and zero network access (see
  docs/BENCHMARKS.md's artifact-rejection track). Both the infrastructure
  and one real evidence exercise of it now exist; a benchmark-ready bank of
  several such TPFs across camera/CCD/night groups does not yet.
- **SIMBAD labels are now positionally object-matched, not field lookups.**
  `research/acquire.py:_pull_simbad_labels` cross-matches each acquired
  object's real discovery-time position (`_acquired_sources`, from
  `metadata.list_sources`) against SIMBAD's returned counterparts via
  `crossmatch.match_catalogs`, with a confidence score that falls with
  separation and local crowding. A `LabelRecord.object_id` is now
  guaranteed to be one of the acquisition's own object IDs; an object with
  no real nearby counterpart gets no label, rather than every SIMBAD row in
  the field being attached to every object. Still small by construction (one
  cone lookup for the whole field, not a batched join) — see
  docs/DATA_CARD.md's "Known biases" section, which should be read as
  describing the pre-cross-match behaviour historically.
- **`*_eval.py` local `_summary()` duplication: the 8 modules with an
  exact-shape (mean/std/ci95[/n]) `_summary()` are migrated; the rest are
  not.** `research/stats.py` provides the shared `summary()`/
  `paired_bootstrap_ci()`/`benjamini_hochberg()` implementations the
  roadmap calls for. `agn_changepoint_eval.py`, `flare_energy.py`,
  `open_world_eval.py`, `photo_z.py`, `pretrain_probe.py`,
  `sn_classification_eval.py`, `stellar_manifold_eval.py`, and `sweep.py`
  now delegate their `_summary()` to `research.stats.summary()` rather than
  each carrying their own copy. Of the 32 `*_eval.py` modules, the other 24
  (`artifact_bank_eval.py` among them) either use a different local
  statistical helper entirely or none at all — those were not touched,
  since forcing an import into a module with no actual duplication would
  not be a fix. A module's own imports are the source of truth for whether
  it still needs this.
- **Cross-survey ablations (single-survey/pairwise/full) are not run.**
  `benchmark.py: run_cross_survey_anomaly` accepts whatever `FeatureMatrix`
  it is given, so ablations are possible by construction (pass a
  single-survey matrix vs. a joint one), but this session's demonstration
  run only exercised the full joint matrix.
- **No Benjamini-Hochberg pass was run.** `stats.benjamini_hochberg` exists
  and is tested, but the demonstration run scored a small, fixed set of
  methods rather than ranking many candidates, so there was nothing to
  apply FDR control to yet.

## Blocked on external inputs (migrated from docs/DEFERRED.txt)

Each item names precisely what would unblock it. None of these are missing
code; all are waiting on something this codebase cannot supply itself.

**Future data releases**
- Gaia epoch (time-series) photometry stays a DR3 static-astrometry-only
  connector (`surveys/gaia.py`) plus an offline DR4-fixture validator
  (`surveys/gaia_epoch.py`) until Gaia DR4 ships (expected 2026-12-02,
  ~400 TB of real epoch photometry/spectra). `GaiaEpochAdapter` already
  validates offline fixtures and records rejection reasons; live ingestion
  is gated on the real archive schema matching those fixtures.
- Rubin/LSST direct TAP access (`https://data.lsst.cloud/api/tap`) needs a
  data-rights account token this codebase has no path to obtain; the
  ALeRCE broker connector (`surveys/alerce.py`) is the credential-free
  alternative already wired in, not a full substitute.
- KMTNet microlensing survey data stays proprietary for one year after
  observation — `microlensing.py`'s binary-lens fitting is validated on
  OGLE (public) and synthetic data only.

**Live-contract validation gaps** (the endpoint's exact field names/schema
have not been confirmed against a real response this session; each
connector degrades to "zero sources returned" rather than crashing on a
mismatch, which a fixture test cannot distinguish from a correct empty
cone): Chandra CSC's `cone_search` (`cda.cfa.harvard.edu/csccli/browse`
needs an undocumented `packageset` parameter — confirmed live to 400/404
on every value tried), Swift/XMM/DES/Hubble/JWST/ALeRCE's `/lsst/v1/`
path. Each needs one real network session against its live endpoint to
close, not new code.

**Real telescope-produced data files**
- Breakthrough Listen HDF5 filterbank format: `technosignature.py` reads
  SIGPROC `.fil` files (fixed, published binary layout, implemented and
  tested against a self-constructed file). BL's own HDF5 `/data` layout
  and header-attribute schema has not been checked against a real
  archive file.
- Real ZTF subtraction-artifact patches: IRSA's `nph_light_curves`
  endpoint returns photometry and `catflags` but never per-epoch
  camera/CCD/quadrant identifiers, so `artifact_bank.py`'s cross-survey
  domain-adaptation arm has no real metadata to group ZTF patches by —
  confirmed by reading the connector, not assumed. The TESS arm of the
  same module is fully real (camera/CCD/night from real downloaded TPF
  FITS headers).
- A real per-system multi-image astrometry catalogue for strong lensing
  (`strong_lens.py`/`strong_lens_imaging.py`) and stacked real galaxy
  clusters for weak-lensing NFW halo-mass fitting (`weak_lensing.py`):
  neither is reachable via VizieR from this session.
- Real nightly instrument-telemetry/observing-condition streams for
  `quality_drift.py` and `ccd_attribution.py`'s causal-attribution study
  (both currently synthetic-only for this reason).
- A published Sharma et al. (2016, arXiv:1603.05661) asteroseismic Δν
  correction is NOT a closed-form formula — verified by reading the paper
  itself this session: it is a grid-interpolation product (`asfgrid`,
  a 357 MB downloadable package) built from a `13×19×2×200`
  `(log Z, M, evolutionary state, Teff)` MESA/GYRE model grid. Fabricating
  an approximate formula was rejected rather than shipped; the real fix is
  downloading and integrating that package, not new arithmetic.

**Credentials this codebase cannot supply**
- TNS (Transient Name Server) requires a user-specific API key;
  `credentials.py` already has the generic DPAPI-backed storage
  (`save_credentials`/`load_credentials`) a real key would use.
- Production installer/updater code-signing needs a real AuthentiCode
  certificate and timestamp URL; the packaged build fails closed without
  one by design (`security.py`).

**Statistical power / real human data**
- The Stage-B injection-recovery method comparison (`stageb.compare`) has
  only run on 54 ZTF sequences; a population of thousands is needed before
  drawing a paper-level conclusion about deep vs. classical anomaly
  detection.
- The reviewer human-factors experiment's (`review_experiment.py`)
  primary analysis (`review_experiment_eval.anchoring_effect_size`) now
  enforces its own preregistered stopping rule (refuses a verdict below
  30 votes/arm) — it has never actually reached that threshold with real
  reviewer votes.
- `active_review.true_inter_rater_agreement` computes real Cohen's kappa
  from `metadata.label_votes` once two reviewers have 5+ overlapping
  votes; no real campaign has produced that data yet.
- Federated-training's Bonawitz et al. secure-aggregation key-agreement
  layer is not implemented — there is no second real training party to
  secure a protocol against yet, so building one would have nothing real
  to protect.

## Deliberately out of scope (migrated from docs/DEFERRED.txt)

Refused by design, stated in the relevant module's own docstring, not an
oversight:

- **No impact probability** in `neo_hazard.py` — needs covariance
  propagation through a perturbed force model this codebase does not
  have; reports close-approach-distance sensitivity instead, never
  reframed as a probability.
- **No individual-mode peakbagging or red-giant mixed-mode analysis** in
  `asteroseismology.py` — scaling-relation-level only.
- **No multiple-scattering radiative transfer** in `biosignature.py` —
  isothermal/isobaric single-slab only; Rackham et al. (2018)
  stellar-contamination correction (likely the single largest real-world
  systematic in transmission spectra) is not modelled.
- **The O(N log N) Taylor (1974) dedrift tree** in `technosignature.py`
  is deliberately not implemented: this codebase's standard is that a
  fast path must be checked bit-for-bit against its brute-force reference
  before shipping, and the tree's edge-of-band combination rule could not
  be verified bit-exact against `dedrift_bruteforce` in the time available
  for this codebase's build sessions. Shipping an unverified fast path
  that might silently disagree with the reference was judged worse than
  not having one.
- **Conformal anomaly p-values are not wired into `review.select_next`'s
  ranking weights** (`conformal.py`) — a real, separate adoption decision
  left deliberately unmade, the same restraint every calibration study in
  this codebase applies before touching `evidence.WEIGHTS`.
- **No third (seismic) domain** for `corroborate/` — two real domains
  (astronomy, GW) validate the mechanism; a third was judged to add
  complexity without new evidence about genericity.
- **No Bonawitz secure-aggregation protocol** in `federated_training.py`
  (see above — no real second party to protect yet).
- **No live ToO (target-of-opportunity) submission** from `schedule.py` —
  scheduling stops at producing an ordered sequence; nothing here submits
  to a real facility.
- **TESS's 21-arcsec pixel scale is a hardware fact**, not a software
  gap: `crossmatch.py` marks blended TESS matches as `blended`, correctly
  excluding manufactured cross-survey agreement, rather than pretending
  to resolve individual stars in crowded fields.
- **The largest-catalogue anchor default in `crossmatch.group_sources`**
  is a documented, real selection bias (the object population is defined
  by whichever survey anchors the match), not removable by more code —
  `anchor_survey` lets a caller override it and the response reports
  `anchor_policy` explicitly, but the bias itself is inherent to any
  choice of anchor.
- **A reviewer's timeline overlay curves stay unfolded** in the candidate
  investigation UI — a deliberate scope limit on `CandidateWorkspace.tsx`.

## Acceptance-gate status

See docs/REPRODUCIBILITY.md for the full walk of the 10 acceptance gates
from the roadmap; gates not yet passable are recorded there explicitly
rather than omitted.
