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
- **Status this release**: infrastructure wired (the runner can call it),
  but not executed against a fresh acquisition this session — see
  docs/LIMITATIONS.md. TESS TPF pixel acquisition is a separate step from
  the light-curve acquisition this session's demonstration corpus used.

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
