# Benchmarks

Task definitions, split policy, metrics, baselines, and statistical tests
for ASTRA's research evidence package. A benchmark is only "run" once its
`BenchmarkSpec` (`research/benchmarks/benchmark_specs/`) is sealed and its
results are written to `research/results/`.

## Split policy

Two split kinds, both leakage-checked before use
(`engine/astra/research/splits.py: detect_leakage`):

- **Object-grouped** (`object_grouped_split`): every distinct object ID
  assigned to exactly one fold. Replaces `tensors.train_test_split`'s random
  row split for benchmark paths, where a row-level split can put the same
  object's rows in both train and test.
- **Sky/time** (`sky_time_split`): groups by (coarse sky cell, ~quarterly
  observing season) so a field or season cannot straddle folds even when an
  object-grouped split alone would allow it.

Both are frozen JSON under `research/splits/`, named by `split_id`, and
referenced by ID from every `BenchmarkSpec` and `ResultRecord` — never
re-derived ad hoc per run.

## P0 tracks

### Cross-survey anomaly discovery

- **Positive definition**: real object, synthetic injected feature-space
  perturbation (`engine/astra/research/benchmark.py: _inject_synthetic_anomalies`),
  following the same label-by-construction principle
  `engine/astra/evaluate.py` already documents and uses for its own
  injection studies. Reported under `synthetic=True` in every `ResultRecord`
  — the *feature data* is real (checksummed via a sealed
  `DatasetManifest`), the *label* is not.
- **Primary metric**: AUPRC. **Secondary**: recall@k, precision@k, discovery
  rate, FDR (`engine/astra/research/stats.py: benjamini_hochberg` when
  ranking many candidates).
- **Baselines**: robust MAD z-score, Isolation Forest, One-Class SVM,
  logistic regression, and the existing ASTRA ensemble
  (`engine/astra/anomaly.py: detect`). All five see the identical injected
  matrix per seed (`benchmark.py: _perturbed_matrix`), so a baseline's
  score is comparable to the ensemble's, not scored against a different
  perturbation.
- **Ablations**: single-survey, pairwise-survey, full multimodal —
  controlled by which survey's rows populate the feature matrix passed in.

### Artifact rejection

- **Positive definition**: a real TESS instrumental-defect patch
  (`engine/astra/artifact_bank.py: PatchRecord`, built from real downloaded
  TPFs) vs. a real clean-contrast patch.
- **Primary metric**: FPR at fixed TPR. **Secondary**: AUROC/AUPRC,
  calibration, confusion matrix.
- Reuses `engine/astra/artifact_bank_eval.py: evaluate_cross_group_auprc`
  (real, leave-one-group-out by camera/CCD/night) directly rather than a
  new evaluator.
- **Status**: infrastructure wired, and one real TPF has now been pulled —
  the benchmark itself has still not been executed. `p0-pleiades-2026` is
  the first acquisition in this project's evidence package to return real
  TESS *light curves* (8 sources, 30 SPOC curves), up from every prior
  dataset's 0. Using one of those real TESS sources' position
  (`TIC 125736995`, RA=56.6639, Dec=24.1032), `tess_pixels.download_tpf`
  pulled a real 20×20-pixel, 3534-cadence SPOC TESScut TPF from MAST
  (sector 42; `tess_pixels.find_sectors` also confirmed sectors 43, 44, 70,
  71 cover the same position) — 28.5 MB, cassette-recorded through the
  `netclient.download` cassette layer this session added, and verified: an
  offline replay reproduces the identical `fits_sha256` in ~1.4 s with zero
  network access. This is the *first* real download-cassette exercise
  against a real product in this project, not only the unit-test fixtures
  in `test_research_cassettes.py`.
  **Now benchmarked at real, small-but-growing scale, across all three
  grouping axes.** Six real TPFs have been pulled from MAST (3 targets x
  2 sectors each, chosen specifically for sky/epoch diversity):
  `TIC 125736995` sectors 42 & 70, `TIC 950029959` sectors 23 & 49,
  `TIC 63003344` sectors 14 & 55. `extract_camera_ccd`'s FITS-header
  assumption is confirmed live (`CAMERA=4`/`CCD=4` read correctly from the
  first TPF's real header). `artifact_bank.build_patch_bank` over all six
  real files yields 38 real patches spanning 2 real cameras, 3 real CCDs,
  and 4 real nights, with real TESS-quality-flag-derived labels (19 clean,
  19 defect-flagged — `stray_light`/`excluded` categories, not synthetic
  injection). `evaluate_cross_group_auprc` run on all three real axes,
  sealed as `BenchmarkSpec artifact-rejection-p0-v2` (`EXP-0024`):

  | Grouping | Mean AUPRC | Worst fold |
  |---|---|---|
  | camera (2 groups) | 0.6411 | held-out camera 2: 0.6333 (n_test=6) |
  | ccd (3 groups) | 0.7974 | held-out ccd 3: 0.5 — chance level (n_test=2) |
  | night (4 groups) | 0.7091 | held-out 2022-02-26: 0.4167 — below chance (n_test=4) |

  **This is a materially different, more honest finding than the earlier
  3-TPF run's 0.9872**: with genuine camera/CCD/night diversity, cross-
  group generalization is real but far from perfect, and at least one held-
  out fold per axis performs at or below chance. A `use_coral=True` rerun
  on the camera axis (domain-adapting training features to the held-out
  camera before training) scored **mean AUPRC 0.5834 — worse, not
  better** (one fold improved 0.6333→0.8333; the other, trained on only 6
  patches, got worse 0.649→0.3335). This is reported as a genuine negative
  result: CORAL alignment does not clearly help at this sample size, most
  plausibly because a 6-patch training set makes any alignment estimate
  too noisy to trust, not because domain adaptation is unhelpful in
  principle. The `metrics.jsonl` row for `artifact-rejection-p0` (3-TPF,
  0.9872) is preserved rather than deleted — both are real, both are
  honestly small-sample, and the newer, larger, more diverse run is the
  one to trust.
  **Read this correctly**: 38 patches across up to 4 groups is still a
  real proof-of-concept, not a release-scale claim — several folds have
  single-digit test counts, and no bootstrap CI is reported at this sample
  size. A real release-scale run needs TPFs spanning many more cameras/
  CCDs/nights with a much larger, more balanced patch count per group.

## Experimental standards

Applied by construction, not by convention:

- Splits are object-disjoint; `detect_leakage` is run and its `clean` flag
  checked before a split is used.
- Five fixed seeds (`BenchmarkSpec.seeds`, default `[0, 1, 2, 3, 4]`) —
  declared in the spec before execution, not chosen per run.
- Confidence intervals are paired bootstrap over object groups
  (`research/stats.py: paired_bootstrap_ci`, or the AUPRC-specific grouped
  bootstrap in `benchmark.py` for the anomaly track), not the seed-quantile
  convention `sweep.py` uses for hyperparameter sweeps — those measure a
  different kind of uncertainty (see `stats.py`'s module docstring).
- Real and synthetic results are written to separate files
  (`research/results/metrics.jsonl` vs `metrics_synthetic.jsonl`) by
  construction: `store.save_result_records` raises if a caller's
  `synthetic` flag disagrees with a record's own `synthetic` field.
- Every `ResultRecord` carries `experiment_id`, `benchmark_id`, `split_id`,
  and `dataset_manifest_hash` — a metric with no experiment behind it
  cannot be constructed with this schema.
