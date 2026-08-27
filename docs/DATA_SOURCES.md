# Data sources

Authoritative external sources ASTRA's evidence package draws on, and how
each is captured. The machine-readable registry is
`research/sources/source_registry.yaml`; this document explains the
retrieval procedure and how to read the registry's `status` field.

## Status values

- `registered` — locator recorded, no data acquired yet.
- `acquired` — a sealed `DatasetManifest` exists under
  `research/datasets/manifests/`, with checksum, row/byte count, license,
  and citation filled in.
- `blocked` — a credential, access, or terms-of-use issue prevented
  acquisition. Recorded explicitly; never presented as "no data found".

## Retrieval procedure

1. `engine/astra/research/acquire.py: acquire_core_corpus()` runs one cone
   search across the requested survey connectors (reusing
   `engine/astra/acquire.py`'s existing extract-and-discard acquisition
   pipeline, not a new one), then:
   - seals a `manifest.Manifest` (query identity, content hash) via the
     existing acquisition path;
   - promotes that manifest into `research/datasets/manifests/` with
     `license`, `citation`, `calibration_version`, and `selection_rule` set
     (`manifest.py` v2 fields);
   - records `row_count` from the manifest's own object count.
2. Every HTTP request a connector makes goes through
   `engine/astra/netclient.py`, which is cassette-aware
   (`engine/astra/research/cassettes.py`): setting `ASTRA_CASSETTE_MODE=record`
   captures real responses into `research/fixtures/cassettes/`, redacting
   credential-shaped headers before writing.
3. `scripts/acquire-core-corpus.ps1` drives this at whatever scale the
   caller chooses; the 45 GB dataset cap
   (`config.DEFAULT_DATASET_CAP_GB`) is enforced by the existing
   `astra.cache.enforce_cap()`, which the acquisition pipeline already calls.

## Sources by evidence area

See `research/sources/source_registry.yaml` for the full, current-status
table (time-domain observations, object/class labels, alert streams,
spectroscopy/broad-band context, X-ray/radio, microlensing/variables/moving
objects, gravitational-wave, and interoperability/preservation standards).
The table in the roadmap brief is a locator list to resolve, not a claim
that this checkout already holds the data — `source_registry.yaml`'s
`status` field is the actual, current truth.

## Credentials and access

A source marked `access: registration_required` (e.g. TNS) needs a
credential ASTRA does not ship. `engine/astra/credentials.py` already
handles per-provider credential storage; missing credentials are recorded
in `source_registry.yaml` as `status: blocked` with a note, never silently
skipped or reported as a null scientific result (see docs/LIMITATIONS.md).
