# ASTRA P0 research evidence report

Generated from `research/results/`, `research/experiments/manifests/`, and
`docs/{RESULTS,BENCHMARKS,DATA_CARD,LIMITATIONS,REPRODUCIBILITY}.md` — every
number below traces to a sealed `BenchmarkSpec`, a `Split`, and a
`complete()`-passing `Experiment` record checked into this repository, or is
explicitly marked as not yet available. This is the acceptance-gate-10
consolidated report docs/REPRODUCIBILITY.md's gate table flags as
"partial" — assembled once enough real evidence existed to make it worth
writing, not on a fixed schedule.

## Executive summary

ASTRA's evidence pipeline (acquire → seal manifest → leakage-checked split →
bound `ResultRecord`) is now exercised end-to-end against real archive data,
not only against unit-test fixtures. Two research tracks have real,
`complete()`-passing results: the cross-survey anomaly track (synthetic
injected labels over real feature data) and the artifact-rejection track
(real, non-synthetic TESS quality-flag labels — the first `synthetic: false`
row this project has ever produced). Five methodology studies the roadmap
had staged but never run were executed at the real scale the local corpus
actually supports. One real bug in the evidence-storage layer was found and
fixed during this work, with the damage recovered from git history rather
than silently accepted.

What this report is *not*: a release-scale scientific claim. Every dataset
here is small (71–610 objects; 6 TPFs), chosen to validate the pipeline and
produce genuine — if statistically modest — findings, not to support a
publication-grade discovery claim. Every section below says so explicitly
where it applies.

## Datasets

| Dataset ID | Surveys | Objects | Selection | Real labels |
|---|---|---|---|---|
| `core-demo-2026` | ZTF (150), Gaia (4), TESS (0) | 154 | Arbitrary cone | 1 (unmatched field lookup, pre-crossmatch-fix) |
| `p0-validation-2026` | ZTF (60), Gaia (11), TESS (0) | 71 | Arbitrary cone, pipeline-fix validation | 0 (real cross-match ran, found nothing) |
| `p0-pleiades-2026` | ZTF (300), Gaia (200), TESS (8) | 508 | Pleiades cluster, chosen for label density | 29 (real, positionally cross-matched) |

Plus 6 real TESS TPFs (3 targets × 2 sectors), pulled directly from MAST,
independent of the light-curve acquisitions above. See docs/DATA_CARD.md for
full selection-effect and missingness detail per dataset.

## Results

### Cross-survey anomaly discovery (synthetic labels, real feature data)

Full leaderboard in docs/RESULTS.md. Headline: on `core-demo-2026`
(`EXP-DEMO-0001`, 286 usable rows), `isolation_forest` (mean AUPRC 0.970)
edges out ASTRA's own production ensemble (0.948); `logistic_regression`'s
0.929 is not a fair comparison (trained on the labels it's scored against).
This leaderboard's known caveat, disclosed in docs/RESULTS.md and not yet
fixed retroactively: its 467-row feature matrix was not scoped to the
manifest's 62 curves at the time it ran (fixed in code since; the table is
kept as a historical record rather than silently edited).

A second, code-fixed validation run (`EXP-0023`, `p0-validation-2026`)
confirms the fix: `scope_to_manifest` reduced a 610-row matrix to the
manifest's own 37 rows, `dropped_out_of_manifest_rows: 573` explicit in the
result. Both `EXP-DEMO-0001`'s and `EXP-0023`'s rows coexist in
`metrics_synthetic.jsonl` (50 rows total) — a second real bug (silent
file-overwrite on every save) was found and fixed to make this possible;
see "A real bug found and fixed" below.

### Artifact rejection (real TESS quality-flag labels — no synthetic injection)

The project's first non-synthetic result. `research/results/metrics.jsonl`
now holds two real runs:

- `artifact-rejection-p0` (`EXP-0022`, 3 TPFs, 30 patches, 2 camera groups):
  mean AUPRC 0.9872 — since superseded by a larger, more diverse run;
  preserved rather than deleted.
- `artifact-rejection-p0-v2` (`EXP-0024`, 6 TPFs, 38 patches, spanning 2
  cameras / 3 CCDs / 4 nights): mean AUPRC 0.6411 (camera), 0.7974 (ccd),
  0.7091 (night) — the more honest number. At least one held-out fold per
  axis scores at or below chance (worst: `ccd=3`, AUPRC 0.5, n_test=2). A
  CORAL domain-adaptation rerun on the camera axis scored *worse*
  (0.5834), a genuine negative result attributed to a 6-patch training
  fold being too small to estimate a reliable alignment.

Both are real, small-sample proofs of concept — not release-scale claims.
See docs/BENCHMARKS.md for the full per-fold breakdown.

### Five staged scale studies, actually run

All five job payloads staged in `docs/DEFERRED.txt` (line 3748) as
"ready-to-run" were executed against the real local ZTF corpus (~610
curves; the roadmap's `limit=10000` was requested unmodified and simply
never binds). Full detail in docs/RESULTS.md; headline findings:

1. **Stage-B method comparison (deep vs. classical), 5 real seeds, both
   resampling modes** — the original n=54 finding ("PCA beats deep
   learning") does not hold at n=373: no method separates from PCA within
   noise in time mode; PCA ranks 7th of 8 in season mode.
2. **Autoencoder capacity sub-grid** — no configuration separates; capacity
   does not measurably matter at this scale.
3. **10-seed repeated ablation** — no feature family is clearly
   load-bearing (now a 10-seed finding, not single-seed noise).
4. **Artifact weight calibration** — proposes real weight changes
   (AUC 0.9689→0.9815); not auto-adopted, per this module's own stated
   discipline.

## Negative results

Reported because the project's own standard requires it, not because they
are flattering:

- No deep model clearly beats classical baselines (PCA, Isolation Forest)
  at real scale on this corpus — the opposite of what the original n=54
  measurement suggested.
- No single feature family is load-bearing for anomaly detection, at
  10-seed confidence.
- Model capacity (autoencoder latent dimension, channel width) does not
  measurably affect injection-recovery AUC at this scale.
- CORAL domain adaptation made cross-camera artifact-rejection
  generalization *worse*, not better, in a small-sample regime.
- The real-label SIMBAD cross-match found genuinely nothing in an
  arbitrary field (`p0-validation-2026`) — a legitimate negative result
  about that field's SIMBAD coverage, not a bug.

## Selection effects

See docs/DATA_CARD.md for the full accounting. The load-bearing one for
this report: `p0-pleiades-2026`'s 29 real labels are drawn from a
deliberately dense, well-studied cluster field. Nearly every object well-
matched there is astrophysically "interesting" by construction (young
stellar objects, flare stars, spectroscopic binaries) — there is almost no
clean "boring negative" class in that dataset to contrast against. This is
the concrete reason a real-label cross-survey anomaly benchmark was not
attempted this pass: forcing a binary positive/negative split from 29
labels across 8 fine-grained SIMBAD types, with no principled negative
class, would require inventing a taxonomy this report is not qualified to
assert authoritatively. A real-label anomaly benchmark needs either a
domain-expert-defined taxonomy or actual human-reviewed candidate labels
via the existing review workflow (`ranker.labelled_examples` reads from
`candidates.load_labels`, not from SIMBAD records directly) — this remains
open.

## Compute cost

Measured on this session's real hardware (NVIDIA GeForce GTX 1650, 4 GB) via
`profiling.run_all`, over the current local store (100-curve sample):

| Stage | Share of feature-extraction time |
|---|---|
| `bocpd` (change-point detection) | 57.3% |
| `temporal_features` | 23.4% |
| `periodic_features` (Lomb-Scargle) | 15.1% |
| `read_parquet` | 3.8% |
| everything else | <1% |

This is a real, fresh measurement, and it differs from a previously-recorded
figure elsewhere in this codebase's history (Lomb-Scargle ~76%, bocpd ~17%,
on an earlier/different curve population) — recorded as a genuine finding
that cost distribution shifts with the population measured, not silently
reconciled to match the older number. Full pipeline: `featurematrix_build`
dominates end-to-end wall time (90.5% of a 59 s, 940-object run). GPU
periodogram: 9.74× speedup over the CPU-approximate path (127.5 ms vs.
1241.3 ms on a 350-point curve, 273,144 frequencies). GPU array ops beat CPU
by ~4× including transfer overhead, consistent with prior Amdahl's-law
findings that periodogram/bocpd, not generic array ops, are the real
targets for GPU acceleration.

## A real bug found and fixed

`research.store.save_result_records` opened its output file in `"w"`
(overwrite) mode. Every second real call — exactly what a second benchmark
run does — silently erased every prior row. This session's own
`p0-validation-2026` re-validation run tripped it, erasing the original
`core-demo-2026` 25-row demonstration leaderboard. The content was fully
recovered from git history (the file is committed), the function was fixed
to load-then-append, a regression test was added
(`test_result_records_accumulate_across_calls`), and the validation run was
redone cleanly. Both the original and new rows now coexist correctly. This
is disclosed here in full rather than quietly repaired, per the same
discipline `docs/DEFERRED.txt`'s "GAP CLOSED, REAL BUG FOUND" entries use
throughout this project's history.

## Reproducibility

- Every experiment referenced above is exported to
  `research/experiments/manifests/`, checkable from this repository without
  access to the machine that produced it — closing acceptance gate 4 for
  every row except one pre-existing exception: `metrics_synthetic.jsonl`'s
  original `EXP-DEMO-0001` row has no experiment record anywhere (it
  predates this session and appears to have been written without one).
  This is disclosed rather than silently left unresolved.
- `netclient.download` is now cassette-recorded and replay-verified: a real
  28.5 MB TESS TPF pull was replayed offline with an identical
  `fits_sha256` in ~1.4 s.
- A real signed reproducibility bundle was built for `p0-validation-2026`
  (`research.bundle.build`, referencing `EXP-0023`) and verified
  (`research.bundle.verify` → `valid: true`). Exported to
  `research/experiments/signed_bundles/p0-validation-2026.json` (public key
  and Ed25519 signature only — the per-project signing private key stays
  local, per `reproducibility_bundle.py`'s own design, and was not copied).

## Failure modes and known limitations (unchanged or newly found this pass)

- Real-label cross-survey anomaly benchmark: blocked on methodology
  (see "Selection effects" above), not merely on data volume.
- Artifact-rejection bank remains small (38 patches); several cross-group
  folds have single-digit test counts and should not be over-read.
- `p0-pleiades-2026`'s Gaia connector still returns 0 stored light curves
  (metadata-only, per `research/sources/connector_status.yaml`) — 200 Gaia
  sources contributed 0 usable photometry.
- Gaia DR4 (expected 2026-12-02) remains the hard blocker on the
  `gaia_only`/`all_three` cross-survey ablation arms.
- Full detail: docs/LIMITATIONS.md, docs/DATA_CARD.md's "Known biases".

## Index of real evidence produced this pass

| Experiment | What | Real/synthetic |
|---|---|---|
| `EXP-0022` | Artifact rejection, 3 TPFs, camera axis | real |
| `EXP-0023` | Cross-survey anomaly, `p0-validation-2026`, scoping-fix validation | synthetic labels, real features |
| `EXP-0024` | Artifact rejection, 6 TPFs, camera/ccd/night axes | real |
| Stage-B time mode | Deep-vs-classical, n=373, 5 seeds | synthetic labels, real features |
| Stage-B season mode | Deep-vs-classical, n=373, 5 seeds | synthetic labels, real features |
| Autoencoder sub-grid sweep | Capacity study, n=373, 3 seeds | synthetic labels, real features |
| 10-seed repeated ablation | Feature-family + detector comparison | synthetic labels, real features |
| Artifact weight calibration | Proposed indicator weights | synthetic defect patches |

All bound to sealed `BenchmarkSpec`/`Split`/`Experiment` records under
`research/`. See docs/RESULTS.md for full per-seed values and confidence
intervals.
