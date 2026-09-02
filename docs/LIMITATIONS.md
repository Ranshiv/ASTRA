# Limitations

Synthetic-only modules, dormant connectors, unvalidated assumptions, and
negative results for ASTRA's research evidence package. This document is
generated/maintained alongside `research/sources/module_status.yaml` and
`research/sources/connector_status.yaml`, and is meant to be read next to
`docs/DEFERRED.txt`'s existing 50 `[PARTIAL]`/13 `[KNOWN]` entries, not as a
replacement for them.

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

## Acceptance-gate status

See docs/REPRODUCIBILITY.md for the full walk of the 10 acceptance gates
from the roadmap; gates not yet passable are recorded there explicitly
rather than omitted.
