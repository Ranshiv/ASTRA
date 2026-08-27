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
- **Artifact rejection benchmark**: infrastructure (`benchmark.py` can call
  `artifact_bank_eval.evaluate_cross_group_auprc`) exists, but was not
  executed against a fresh acquisition this session — TESS TPF pixel
  acquisition is a separate step from the light-curve acquisition this
  session ran, and was out of scope for the demonstration run. Status:
  `candidate-only` in `research/sources/module_status.yaml`.
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

- **Cassette layer covers `netclient.get`, not `download`.** Bulk product
  downloads (`netclient.download`) are not cassette-wrapped; only
  metadata/query-shaped GET requests are. A connector whose evidence
  depends on a downloaded FITS/TPF file is not yet cassette-verifiable.
- **SIMBAD labels are field lookups, not object-matched labels** — see
  docs/DATA_CARD.md's "Known biases" section. Do not read a
  `LabelRecord` from `_pull_simbad_labels` as "this ASTRA object has this
  label" without an explicit positional cross-match step, which does not
  exist yet.
- **`*_eval.py` local `_summary()` duplication is not yet removed.**
  `research/stats.py` provides the shared `summary()`/
  `paired_bootstrap_ci()`/`benjamini_hochberg()` implementations the
  roadmap calls for, but the ~30 existing `*_eval.py` modules (
  `agn_changepoint_eval.py`, `artifact_bank_eval.py`, etc.) still carry
  their own local reimplementations. Migrating them is mechanical
  (import `research.stats` instead of redefining `_summary`) but was not
  done this session to keep the diff to the modules this session's work
  actually depends on.
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
