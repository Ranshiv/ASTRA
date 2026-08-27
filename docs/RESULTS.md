# Results

Generated from `research/results/`. Values below are read from
`research/results/metrics_synthetic.jsonl` (25 records: 5 baselines x 5
seeds), produced by `EXP-0010` — a real, `complete()`-passing experiment
record binding this benchmark to its dataset manifest and split.

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
- **The feature matrix is not strictly scoped to this cone.**
  `featurematrix.build(survey="ZTF")` scans every ZTF light curve under
  the local store, which can include curves from earlier sessions/runs —
  467 rows were scored, not exactly 62. The dataset manifest hash above
  correctly documents *this session's acquisition query*; it does not
  certify that every one of the 467 scored rows came from that query.
  Treat this as a known scoping gap (docs/LIMITATIONS.md), not a
  misrepresentation — a stricter run would filter the matrix to the
  manifest's own object ID list before scoring.
- **Artifact rejection track**: not run this session (docs/LIMITATIONS.md,
  docs/BENCHMARKS.md) — no TESS TPF pixel data was acquired at this
  position (0 TESS sources found in the cone).
