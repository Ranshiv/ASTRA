# research/

Research artefact tree for ASTRA's evidence package: dataset manifests, label
records, benchmark specs, results, and reproducibility bundles. See
`docs/DATA_SOURCES.md`, `docs/BENCHMARKS.md`, and `docs/REPRODUCIBILITY.md`
for the full picture; this file states only the storage policy.

## Size policy

This tree lives inside the repo, so only small, text-diffable artefacts are
committed here:

- `sources/` — YAML/BibTeX source registry and citations. Small, always committed.
- `datasets/manifests/`, `checksums/`, `licenses/` — manifest JSON, SHA-256
  checksum listings, and license text/pointers. Small, always committed.
- `labels/` — label tables (`object_labels.parquet`, `artifact_labels.parquet`,
  `review_history.parquet`). Target **under 50 MB per file**; if a label
  table would exceed that, split it by survey/release rather than committing
  a single oversized file.
- `splits/` — split definitions as JSON (object IDs or IDs + fold assignment,
  not raw data). Small, always committed.
- `benchmarks/` — benchmark specs and baseline configs (YAML). Small, always committed.
- `experiments/` — experiment manifests and **signed** reproducibility bundles
  (hashes + signatures, not raw model weights). Small, always committed.
- `results/` — metrics, leaderboard, calibration tables, selection functions,
  failure-case indices. Tabular and small; figures as vector/PNG under a few MB.
- `figures/`, `reports/` — generated report assets, regenerated from
  `results/`, not hand-edited.
- `fixtures/cassettes/` — recorded HTTP responses for connector record/replay
  tests. Redacted of credentials; kept small per-cassette.

**Bulk data never goes here.** Raw light curves, pixel cutouts/cutout stacks,
spectra, and model checkpoints are acquired into `$ASTRA_ROOT/Datasets` and
`$ASTRA_ROOT/Models` (see `engine/astra/config.py: Paths`), governed by the
existing `DEFAULT_DATASET_CAP_GB = 45.0` cap and `astra.cache.enforce_cap()`.
Everything under `research/` references that bulk data by manifest ID and
SHA-256 checksum — it never embeds it.

`.gitignore` enforces this: parquet/fits/duckdb/checkpoint files are ignored
repo-wide except for explicit negations under `research/labels/` and
`research/results/`, and even those are expected to stay small.
