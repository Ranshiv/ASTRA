# Methods

Preprocessing contracts, model versions, score semantics, and calibration
for ASTRA's production and research paths.

## Preprocessing contract

`engine/astra/experiment.py: PREPROCESSING_CONTRACT`/`RESAMPLING_CONTRACTS`
define resampling, normalization, channel, and time conventions as data, so
a change to any of them changes `preprocessing_schema_hash()` by
construction — see that module's docstring. `PREPROCESSING_VERSION` is
currently `2`.

## Feature schema

`engine/astra/features.py: FEATURE_VERSION`/`schema_hash()` version the
photometric, variability, temporal, and periodic feature set. A
`FeatureMatrix` (`engine/astra/featurematrix.py`) records
`feature_version` alongside its values; `research/records.py: ResultRecord`
does not duplicate this — it is reachable via the experiment record's
`Provenance.feature_schema_hash`.

## Score semantics

- **Production candidate score**: `engine/astra/anomaly.py: detect()`'s
  rank-consensus ensemble (`EnsembleResult.consensus`), calibrated via
  `calibrate_scores`/`calibration_report`. This score is never modified by
  a research/interpretation layer (see `docs/RESEARCH-INTEGRATION.md`'s
  "recurring discipline" note, reaffirmed here).
- **Benchmark score**: a `ResultRecord.value` for a declared metric
  (`BenchmarkSpec.primary_metric`/`secondary_metrics`), bound to a split
  and dataset manifest. Distinct from the production score even when both
  come from `anomaly.detect` — a benchmark result is a *measurement about*
  the detector, not an input to it.
- **Significance/calibration layer**: `engine/astra/significance.py`
  (`calibrate`, `evaluate_selection`) is a separately versioned
  interpretation layer over scores, kept visibly distinct in the UI
  (Reports/Explain panels show it alongside, not merged into, the
  candidate score) per existing project discipline.

## Model versioning

`engine/astra/experiment.py: model_version()` — content hash of a
checkpoint file, `None` when there is no checkpoint (e.g. a classical
baseline). A `research_benchmark`-kind experiment
(`engine/astra/research/benchmark.py`, via `rpc.py:
_handle_research_benchmark_run`) sets `model_version` explicitly even
without a checkpoint (`"ensemble+baselines"`), since `Experiment.complete()`
requires a non-empty value — a benchmark run that scores five methods at
once does not have one checkpoint to hash, but it must still record *that*
fact rather than leaving the field silently empty.

## Calibration

`research/stats.py: reliability_table`/`expected_calibration_error`/
`brier_score` report a probabilistic output's calibration alongside its
discrimination metrics (AUPRC/AUROC), per docs/BENCHMARKS.md's
experimental standards. These are separate from
`engine/astra/significance.py: calibrate`, which recalibrates a raw score
distribution for downstream display — `research/stats.py`'s functions
instead *measure* how well-calibrated a set of already-emitted
probabilities was, for benchmark reporting.
