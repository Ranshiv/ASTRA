# Results

Generated from `research/results/`. The cross-survey anomaly leaderboard
below is read from `research/results/metrics_synthetic.jsonl` (25 records:
5 baselines x 5 seeds, `synthetic: true`), produced by `EXP-0010`. The
artifact-rejection result further down is read from `research/results/
metrics.jsonl` (`synthetic: false`) — the project's first real, non-
injected result; see that section for why the two files exist and what the
separation means.

## Run provenance

- **Dataset**: `core-demo-2026` — one real cone acquisition (RA=180.122,
  Dec=22.411, radius=90″), acquired 2026-08-27 with
  `ASTRA_CASSETTE_MODE=record` (150 real cassettes captured under
  `research/fixtures/cassettes/`, credential-shaped headers redacted).
  Manifest content hash: `c5d4c375553b06a619a8af68c2bf467495d312e3a3ce6999b737169be2689c26`.
  154 total objects: 150 ZTF (dr24, 62 with stored light curves — the rest
  matched by position but returned no usable photometry), 4 Gaia (dr3), 0
  TESS (no SPOC light curve at this position).
- **Labels**: 1 real SIMBAD label pulled live (`FIRST J120023.9+222418`,
  type `AG?`) — small by construction, this is a demonstration-scale cone,
  not a labeled benchmark corpus.
- **Split**: `core-demo-2026_object_split` (object-grouped, 92/31/31
  train/val/test) — `detect_leakage` confirms `clean: true`, 0 leaked
  objects. A parallel `core-demo-2026_sky_time_split` was also built and
  is also leakage-clean.
- **Feature matrix**: `core-demo-2026-ztf` — 467 rows (built by scanning
  the full local ZTF store, not filtered to exactly this cone's 62 curves;
  see the caveat below), 286 usable (finite-feature) rows, feature schema
  hash `eb1da88160a9899e`.
- **Benchmark**: `cross-survey-anomaly-demo` — cross-survey anomaly track,
  5 seeds (0-4), primary metric `average_precision`, synthetic injected
  labels (fraction=0.1) per docs/BENCHMARKS.md.

## Leaderboard (mean over 5 seeds, real ZTF feature data, synthetic injected labels)

| Method | Mean AUPRC | Seed range | Notes |
|---|---|---|---|
| logistic_regression | 0.929 | 0.862 – 1.000 | Highest mean, but supervised on the injected labels themselves — an optimistic upper bound, not a fair baseline comparison. |
| astra_ensemble | 0.948 | 0.897 – 0.983 | ASTRA's production rank-consensus ensemble (`anomaly.detect`), unsupervised. |
| isolation_forest | 0.970 | 0.939 – 0.998 | Unsupervised baseline. |
| one_class_svm | 0.839 | 0.766 – 0.918 | Unsupervised baseline. |
| robust_zscore | 0.386 | 0.301 – 0.480 | Simple per-feature MAD z-score baseline; clearly weakest. |

Full per-seed values with bootstrap 95% CIs are in
`research/results/metrics_synthetic.jsonl`; every row there carries
`experiment_id`, `benchmark_id="cross-survey-anomaly-demo"`,
`split_id="core-demo-2026_object_split"`, and the dataset manifest hash
above.

## Reading these numbers correctly

- **This is a demonstration run, not the P0 release benchmark.** 154
  objects, one cone, is far below the roadmap's tens-of-thousands target
  (see docs/LIMITATIONS.md's scope note) — the point of this run is to
  prove the evidence pipeline end-to-end (real acquisition → sealed
  manifest → leakage-checked split → bound result records), not to make a
  performance claim.
- **The label is synthetic.** Every value above scores recovery of an
  injected feature-space perturbation, not a verified real anomaly — see
  docs/BENCHMARKS.md's injection methodology note. `logistic_regression`'s
  high score in particular is expected and not meaningful for ranking
  methods: it is trained directly on the same injected labels it is then
  scored against, which the other four methods never see.
- **The feature matrix behind these specific numbers was not strictly
  scoped to this cone.** `featurematrix.build(survey="ZTF")` scans every
  ZTF light curve under the local store, which can include curves from
  earlier sessions/runs — 467 rows were scored for this run, not exactly
  62. The dataset manifest hash above correctly documents *this session's
  acquisition query*; it does not certify that every one of the 467 scored
  rows came from that query. This was a known scoping gap
  (docs/LIMITATIONS.md) at the time this run was produced.
  **Fixed since**: `research.benchmark.scope_to_manifest()` now filters a
  loaded matrix to the manifest's own object ID list before scoring, wired
  into `research.benchmark.run` — a rerun of this benchmark would score
  exactly the manifest's rows and report the drop count. The table above
  has not been regenerated against that fix; treat it as a historical
  record of the run as it actually happened, not as current-code output.
- **Artifact rejection track**: not run this session (docs/LIMITATIONS.md,
  docs/BENCHMARKS.md) — no TESS TPF pixel data was acquired at this
  position (0 TESS sources found in the cone).

## `p0-validation-2026`: a real end-to-end validation of the scoping/split fixes

A second, small real acquisition (RA=210.5, Dec=-5.2, radius=90″; 71 objects:
60 ZTF, 11 Gaia, 0 TESS; `ASTRA_CASSETTE_MODE=record`) was run specifically
to validate three fixes against live data rather than only against unit
tests: the degenerate sky/time split (`_object_time_records`/
`_assert_not_degenerate`), the manifest-scoped feature matrix
(`scope_to_manifest`), and the positional SIMBAD label cross-match
(`_pull_simbad_labels`). This is a pipeline-validation run, not a new
leaderboard entry — do not read its numbers as a performance claim.

- **Split**: `detect_leakage` reports `clean: true` on a *non-degenerate*
  input this time — 37 objects with real, distinct positions/epochs pulled
  from their own stored curves, not the cone centre repeated 71 times.
  `dropped_no_photometry: 34` (objects matched by position with no readable
  curve, honestly excluded rather than assigned a fake epoch).
- **Labels**: 0. `_pull_simbad_labels` executed a real, positional
  cross-match against live SIMBAD for this field and found no counterpart
  within the match radius (`NoResultsWarning` from the live query) — a
  legitimate real outcome, not a failure of the cross-match code. It is the
  expected behaviour change from before: the old field-lookup version would
  have returned every SIMBAD row anywhere near the cone regardless of match
  quality; the new version correctly returns nothing when nothing matches.
- **Feature matrix scoping, proven live**: `featurematrix.build(survey="ZTF")`
  scanned the whole local store (610 rows, both this dataset's and
  `core-demo-2026`'s curves together) — the exact cross-run contamination
  the old `core-demo-2026` run above suffered from. `scope_to_manifest`
  filtered it to `matrix_rows_scored: 37` (this manifest's own curves only)
  with `dropped_out_of_manifest_rows: 573` reported explicitly, confirmed
  via `research.benchmark.run` over the real RPC dispatch path (`EXP-0023`
  in `research/results/metrics_synthetic.jsonl`, alongside the original
  `EXP-DEMO-0001` demonstration rows, both preserved). Labels are still
  synthetic-injected (0 real labels landed for this cone), so this
  validates the scoping fix, not a real-label benchmark.
- **A real bug found and fixed in the process**: the first two attempts at
  this validation run (`EXP-0011`, `EXP-0012`) used `research.store.
  save_result_records`, which opened `metrics_synthetic.jsonl` in `"w"`
  mode — each call silently erased every prior row, including this file's
  original `core-demo-2026` 25-row leaderboard. The original content was
  recovered from git history (the file is committed), `save_result_records`
  was fixed to load-then-append rather than overwrite, a regression test
  was added (`test_result_records_accumulate_across_calls`), and this
  validation run was redone cleanly as `EXP-0023` once the fix was in
  place. Both `EXP-DEMO-0001`'s original rows and `EXP-0023`'s now coexist
  in the file (50 rows total).

## `p0-pleiades-2026`: real SIMBAD labels at nonzero density

A third real acquisition (Pleiades open cluster field, RA=56.75, Dec=24.12,
radius=300″; `ASTRA_CASSETTE_MODE=record`), deliberately chosen for real
SIMBAD label density rather than an arbitrary cone — `core-demo-2026` and
`p0-validation-2026` returned 1 and 0 real object-matched labels
respectively at arbitrary sky positions.

- **508 total objects**: 300 ZTF (dr24, 167 with stored curves), 200 Gaia
  (dr3), 8 TESS (dr with 30 SPOC curves stored — the first acquisition in
  this project's evidence package with any real TESS light curves at all).
- **29 real, positionally cross-matched SIMBAD labels** — by far the
  largest real label set acquired to date, up from 1 and 0. Class
  distribution: `LM*` (7), `*` (6), `Er*` (4), `PM*` (4), `X` (3), `dS*`
  (2), `Y*O` (2), `SB*` (1) — 8 distinct SIMBAD object types, none reaching
  10 members.
- **Split**: `detect_leakage` reports `clean: true`; `dropped_no_photometry:
  341` (matched by position, no readable curve — mostly the 200 Gaia
  sources, which this connector still does not fetch light curves for, and
  the 133 ZTF sources with no usable photometry).
- **Why this is not yet a real-label benchmark run**: 29 labels across 8
  classes is below `ranker.CalibratedLogisticRanker`'s own gate (≥50
  labels, ≥10 per class, ≥2 groups per class) — and would be statistically
  meaningless to force through the unsupervised cross-survey benchmark's
  binary track regardless of that gate. There is also a real selection
  effect worth naming honestly: nearly every object well-matched to a
  bright, well-studied open cluster core is *some* kind of "interesting"
  SIMBAD type by construction (young stellar object, spectroscopic binary,
  flare star) — this field has almost no clean "boring negative" class to
  contrast against, unlike a discovery-oriented cone. A real-label
  benchmark needs either a much larger acquisition (to clear the ranker's
  gate with room for a real negative class) or a deliberately mixed
  selection (cluster core + surrounding field), not a bigger cluster-only
  cone. This remains the honest state of Step 4 in the P0 research plan.

## Step 0: the five staged scale studies, actually run

The five "ready-to-run" job payloads staged in `docs/LIMITATIONS.md` (line
3748) were executed directly (bypassing the `job.submit` queue, which only
adds progress-survivability that this session's direct background execution
already provided). The roadmap's `limit=10000` was requested in every case;
the real local ZTF store held ~610 curves at time of running, so every study
naturally scored the whole store rather than a 10,000-curve sample — no
artificial scale reduction was applied.

**1–2. Stage-B method comparison, ZTF, 5 seeds (17,29,43,59,71), deep
models ON, 20 epochs — both resampling modes.** Replaces the original n=54
finding ("PCA beat both deep models") that `docs/LIMITATIONS.md` line 2412
explicitly flagged as not yet evidence.

| Mode | n (rows) | Winner (mean ROC-AUC) | Runner-up | Separated? |
|---|---|---|---|---|
| time | 373 | `deep_vae` (0.7792, CI [0.739, 0.831]) | `baseline_isolation_forest` (0.7771, CI [0.746, 0.809]) | No — every method's CI overlaps `baseline_pca_reconstruction`'s (0.7624). |
| season | 373 | `deep_vae` (0.8446, CI [0.793, 0.891]) | `deep_autoencoder` (0.8263) / `baseline_isolation_forest` (0.8259) | No — CIs overlap, but `baseline_pca_reconstruction` now ranks 7th of 8 (0.7696), the clearest reversal of the original n=54 claim. |

Experiment IDs `EXP-0015`ff (time), season run ff — both bound to
`research/results/` via `experiment.run`'s provenance record; full per-seed
values are in each run's checkpoint (`results/stage-b/comparison.json` under
the project root) and experiment record.

**Reading this correctly**: at real scale (373 vs. 54), the original
headline ("PCA beats deep learning") does not hold up as a general
conclusion — no method separates from PCA within-seed noise in time mode,
and PCA ranks near the bottom in season mode. This is not "deep learning
wins" either: every comparison's confidence intervals overlap too much to
name a winner. The honest finding is that classical and deep methods are
statistically indistinguishable on this population at this scale — genuine
new evidence, not the same inconclusive single-seed result restated.

**3. Autoencoder capacity sub-grid** (`latent_dim` ∈ {8, 32} × the default
channel/learning-rate grid, 3 seeds, per the roadmap's own "start with a
sub-grid" note) — `separated: false`; best mean ROC-AUC 0.7834
(`latent_dim=32, channels=(16,32), lr=3e-4`) vs. worst-shown 0.7562, CIs
overlapping throughout. **Capacity does not measurably matter within noise
at this scale** — the answer `docs/LIMITATIONS.md`'s "model capacity is far
below what the GPU allows" note said was blocked on this exact study.

**4. Repeated ablation, ZTF, 10 seeds** (17,29,43,59,71,83,97,101,103,107)
— replaces the original 5-seed run whose feature-family removals moved AUC
by ≤±0.018 from noise alone. At 10 seeds: `all_features` mean ROC-AUC 0.7577
(CI [0.685, 0.818]); every `without_<family>` variant's CI still overlaps
it (`without_shape` lowest at 0.7397, CI [0.661, 0.791]). **No feature
family is clearly load-bearing — now a 10-seed-backed finding, not a
single-seed noise artifact.** Detector comparison in the same run:
`isolation_forest` (0.7757) ≈ `lof` (0.774) ≈ `ensemble` (0.7577) ≈
`pca_reconstruction` (0.7556) all overlap; `one_class_svm` (0.5702) is the
clear laggard. `survey_groups` is `null` for this run — the local store has
no meaningful non-ZTF population to ablate against yet (Gaia has essentially
no stored light curves; see the connector status table in
`research/sources/connector_status.yaml`).

## The artifact-rejection track: real TPFs, and the project's first real (non-synthetic) result

Three real TESS TPFs were pulled live from MAST via `tess_pixels.download_tpf`
(`TIC 125736995` sector 42, `TIC 950029959` sector 23, `TIC 63003344`
sector 14 — the latter two chosen far from the first on-sky to raise the
odds of a different camera/CCD), cassette-recorded through this session's
new `netclient.download` cassette layer. The first (28.5 MB, 3534 cadences)
was also used to confirm, live, that `artifact_bank.extract_camera_ccd`'s
FITS-header assumption is correct (`CAMERA=4`/`CCD=4` read straight from
the real primary header), closing a previously-flagged "unverified
assumption" gap; an offline replay of that download reproduced the
identical `fits_sha256` in ~1.4 s with zero network access.

`artifact_bank.build_patch_bank` over all three real files yielded 30 real
patches across 2 real camera groups, with real TESS-quality-flag-derived
labels (15 clean, 15 defect-flagged) — no synthetic injection anywhere in
this track. `evaluate_cross_group_auprc(group_by="camera")`: **mean AUPRC
0.9872** (held-out camera 2: 1.0, n_test=2; held-out camera 4: 0.9745,
n_test=28). Sealed as `BenchmarkSpec artifact-rejection-p0` before running
(per docs/BENCHMARKS.md's declare-before-run discipline), bound to a real
`camera_grouped` `Split` (`artifact-rejection-p0_camera_loco`) and a
`complete()`-passing experiment record (`EXP-0022`).

**This is the first row `research/results/metrics.jsonl` has ever
contained.** Every prior result in this project's history — the entire
`cross-survey-anomaly-demo` leaderboard, both `p0-validation-2026` runs —
was written to `metrics_synthetic.jsonl` (`synthetic: true`). This one is
`synthetic: false`: real feature data *and* a real, non-injected label.

**Expanded to a real 6-TPF bank** (3 more sectors of the same 3 targets,
chosen for epoch diversity): 38 patches across 2 cameras, 3 CCDs, 4 nights.
`evaluate_cross_group_auprc` on all three axes (sealed as
`BenchmarkSpec artifact-rejection-p0-v2`, `EXP-0024`):

| Grouping | Mean AUPRC |
|---|---|
| camera | 0.6411 |
| ccd | 0.7974 |
| night | 0.7091 |

This is a materially more honest number than the first run's 0.9872 — with
real diversity, cross-group generalization is real but incomplete, and at
least one held-out fold per axis scores at or below chance. A CORAL
domain-adaptation rerun on the camera axis scored **worse** (0.5834 vs.
0.6411) — a genuine negative result, most likely because a 6-patch
training fold makes any alignment estimate too noisy to trust, reported
honestly rather than only publishing the run that looked better.

Read the scale honestly either way: even 38 patches across up to 4 groups,
several with single-digit test counts, is a real proof-of-concept the
pipeline works end-to-end on real data, not a release-scale claim — no
bootstrap CI is reported at this sample size rather than fabricating one.
See docs/BENCHMARKS.md for the full methodology and the honest next step
(more TPFs, more camera/CCD/night diversity, a larger and better-balanced
patch count).

**5. Artifact weight calibration**, 150 synthetic patches/class, 5 seeds,
`hard_real_fraction=0.4` — proposes new indicator weights (e.g.
`sampling_period` 0.35→0.85, `consistent_with_constant` 0.25→0.85);
AUC improves from 0.9689 (current hand-set `artifact.WEIGHTS`) to 0.9815
under the proposed weights (`auc_delta=0.0126`). **Not adopted** — per this
module's own stated discipline, a calibration result is a proposal a human
reads and decides on, never auto-applied, and it is measured on synthetic
defect shapes only (`docs/LIMITATIONS.md`'s "no real artifact label store"
gap still applies).
