# Reproducibility

One-command acquisition/evaluation, checksums, environment lock, and signed
bundle verification for ASTRA's research evidence package.

## One-command flow

```powershell
# Acquire (or extend) the research corpus, cassette-recording live calls.
$env:ASTRA_CASSETTE_MODE = "record"
./scripts/acquire-core-corpus.ps1

# Rerun offline from the recorded cassettes and sealed manifests.
$env:ASTRA_CASSETTE_MODE = "replay"
./scripts/acquire-core-corpus.ps1
```

Both invocations call the same `engine/astra/research/acquire.py:
acquire_core_corpus()`; the only difference is whether `netclient`/`tap`
requests hit the network (`record`) or replay a checksummed cassette
(`replay`). `ASTRA_CASSETTE_MODE` unset means `"off"`, i.e. today's
unmodified live-request behaviour — see `engine/astra/research/
cassettes.py`.

## Checksums

- **Dataset manifests**: `manifest.Manifest.content_hash` (query identity)
  and `manifest.Manifest.checksum` (materialized artefact, set by
  `record_artifact()`) are separate hashes with separate meanings — see
  `manifest.py`'s v2 docstring.
- **Cassettes**: each cassette file stores its own SHA-256 of the recorded
  response body; `cassettes.load()` raises `CassetteChecksumError` if the
  stored checksum does not match the file's own content.
- **Experiment records**: `experiment.record_hash()` hashes the saved
  JSON (excluding the hash field itself); `experiment.verify_record_hash()`
  detects if a record file was hand-edited after saving. This is separate
  from `experiment.verify()`, which detects code/environment *drift*
  rather than *tampering*.

## Environment lock

`engine/requirements.lock` pins every engine dependency exactly (including
`pyyaml`, added for `research/store.py`'s YAML records).
`experiment.capture_environment()`/`manifest.capture_environment()` record
interpreter, platform, and key library versions into every provenance
record, so `experiment.verify()` can report which library changed since a
run.

## Signed reproducibility bundles

`engine/astra/reproducibility_bundle.py` (Ed25519 signing, previously
standalone) is wired into `rpc.py` as of this work:

- `research.bundle.build` — seals a bundle from a project's sealed manifest
  plus a list of experiment IDs, signs it with a per-project-root signing
  key (generated on first use, stored at `<project>/signed_bundles/
  signing_key.hex`), and saves it.
- `research.bundle.verify` — `True` iff the bundle's recorded hash still
  matches recomputing it from its own fields (tamper detection) AND its
  Ed25519 signature verifies against its recorded public key.
- `research.bundle.rerun` — checks a freshly-built manifest's query-identity
  hash against the bundle's recorded one (the query-provenance half of
  rerun verification; the codebase has no live archive access in this
  environment to also check the *data* half bitwise/tolerance — see
  `reproducibility_bundle.py`'s own docstring for that stated gap).

A byte-flipped bundle or a bundle verified against the wrong public key
both return `False` from `verify_bundle`, never raise — confirmed by the
existing `test_reproducibility_bundle.py` suite.

## Acceptance-gate walk

| # | Gate | Status |
|---|---|---|
| 1 | Fresh machine reconstructs the local benchmark from manifests within 45 GB | Infrastructure present (`cache.enforce_cap`, sealed manifests); not exercised on a second machine this session. |
| 2 | Offline reruns distinguish offline / provider failure / no match | `tap.py` already distinguishes `offline` from `no_match`; cassette replay adds a third explicit failure mode (`CassetteMissError`) for connectors using `netclient.get`. |
| 3 | No train/val/test object or field overlap | `research.splits.detect_leakage` implemented and tested (object-grouped and sky/time); run against the demonstration corpus's splits — see docs/RESULTS.md. |
| 4 | Every reported metric resolves to experiment/benchmark/split/checksum/result file | `ResultRecord` schema enforces this by construction; `Experiment.complete()`/`require_complete()` gate leaderboard eligibility. Experiment records are now exported into `research/experiments/manifests/` so this is checkable from the repo, not only from the machine that ran them. One exception, disclosed rather than hidden: `metrics_synthetic.jsonl`'s original `EXP-DEMO-0001` row has no experiment record anywhere (not on disk, not in this export) — it predates this session and was apparently written without one; every row added since (`EXP-0011` onward) does resolve. |
| 5 | Synthetic and real evaluations separate in storage and reports | `store.save_result_records` raises on a `synthetic` flag mismatch; enforced at the file-name level (`metrics.jsonl` vs `metrics_synthetic.jsonl`). Both files are now real and distinct in practice, not just by construction: `metrics.jsonl` holds the artifact-rejection track's real, non-injected result (`EXP-0022`); `metrics_synthetic.jsonl` holds the cross-survey anomaly track's injected-label results. |
| 6 | Signed bundles verify valid records, reject tampered content/keys | Implemented and tested (`test_reproducibility_bundle.py`); wired into RPC. Now also exercised for real: a signed bundle was built for `p0-validation-2026` (referencing `EXP-0023`) and verified `valid: true`; exported to `research/experiments/signed_bundles/`. |
| 7 | Live connectors tested against recorded fixtures; network excluded from default CI | Cassette layer implemented for both `netclient.get` and `netclient.download` (same identity/checksum path, `download` keyed by `(provider, "DOWNLOAD", url, params)`), and now exercised against a real TPF pull: a 28.5 MB TESS TPF from MAST, cassette-recorded, replayed offline with an identical `fits_sha256` (docs/BENCHMARKS.md's artifact-rejection track). Default `pytest tests` still never touches the network (unchanged `live` marker behaviour). |
| 8 | Reports reproduce the same leaderboard/figures from stored result files | `docs/RESULTS.md` is generated from `research/results/`, not hand-written; UI wiring for a leaderboard view is not implemented this session (docs/LIMITATIONS.md). |
| 9 | Each partial module has an explicit status | `research/sources/module_status.yaml` implemented, reconciled against `docs/LIMITATIONS.md`. |
| 10 | Publication report includes negative results, calibration, selection effects, compute cost, failure modes | Assembled: `research/reports/p0-evidence-report.md` covers negative results, selection effects, real compute-cost profiling, and failure modes. Calibration remains the one open piece — `research/results/calibration/` is still empty, since no probabilistic output has real labels to calibrate against yet (the same methodology gap gate 10's own report names for the real-label benchmark). |
