# Research integration layer

ASTRA keeps archive observations and event notices as two related, explicit
data types.

## Event packets

`events.ingest` accepts a JSON object or VOEvent XML and writes the original
payload under `events/packets/` using its SHA-256 as the immutable filename.
SQLite indexes packet identity, revisions, event time, classifications, and
localization. `events.list` returns event clusters; `events.list` with
`packets: true` returns packet revisions. `events.replay` is read-only and
returns packets in deterministic receive order.

The event inbox is optional and offline-safe. Providers are descriptive
metadata until a live connector is configured; a missing provider is never
treated as a negative scientific result.

## Scientific interpretation

`significance.calibrate` adds an empirical tail probability and estimated FDR
from a held-out/reference score population. It does not change the existing
candidate score or historical ranking semantics. `selection.evaluate` turns
injection/recovery rows into completeness cells with Wilson 95% intervals,
stratified by dimensions such as amplitude, duration, and magnitude.

Pipeline-generated candidates now carry additive `significance`,
`evidence_completeness`, and `provenance_refs` fields. Older candidate files
remain readable because all fields have defaults.

## Source attribution

TESS target-pixel blend reports now include catalog-relative flux priors under
`blend.source_attribution`. These priors are explicitly labelled as
`catalog_relative_flux_prior`; they are not promoted to resolved stellar
confirmation or folded into the production score.

## Review and follow-up

`review.next` returns a deterministic, explainable active-review queue based
on detector disagreement, artifact uncertainty, significance-boundary
uncertainty, and feature diversity. It does not change labels or ranking.

`followup.plan` produces a draft-only visibility calculation from target
coordinates and an observer site. It reports visible windows, altitude,
azimuth, approximate airmass, solar twilight, lunar separation/illumination,
and explicit rejected-slot counts. Caller-supplied weather samples and
facility constraints can be applied without making a provider request; no
request is submitted automatically.

## Literature and event association

`literature.search` and `literature.enrich` add cached ADS/arXiv records with
provider release, cache state, and provenance. ADS uses `ADS_API_TOKEN` when
configured; arXiv is public. Missing credentials, offline cache misses, and
provider failures remain distinct from a true no-match. Literature is context
only and never changes candidate scores.

`events.associate` conservatively links the latest packet revision for each
event to candidates only when both localization and time-window tests pass.
Point localizations handle RA wrap-around; HEALPix packet localizations report
pixel probability/credible context when available. Unknown timing or
localization is not treated as a match unless explicitly allowed.

Selection diagnostics now optionally fit a regularized, interpretable recovery
model and report weighted completeness/effective injection counts. The model
is a diagnostic of the supplied injection population, not a replacement for
the production ranking score.

Cross-survey grouping defaults to the largest catalogue, but callers can set
an explicit `anchor_survey` (for example, `Gaia` or `ZTF`) through the
crossmatch/profile RPCs. The selected policy and anchor are returned in
`grouping_bias`; default ties are resolved lexically so reruns do not depend
on mapping insertion order. Changing the anchor intentionally changes the
population denominator and must be recorded with the analysis.

`physical.characterize` and `physical.enrich` provide a bounded broadband SED
diagnostic from available Gaia/survey magnitudes. They report colors, a coarse
blackbody temperature grid fit, residuals, extinction actually supplied by the
caller, and quality warnings. They do not claim a spectral type, invent
reddening, or alter ranking.

## TAP and alert polling

`tap.query` provides a read-only, bounded IVOA TAP/ADQL path. Queries are
validated as single SELECT statements, capped with `TOP`, parsed from CSV or
VOTable responses, and cached with service/release/query provenance. Offline
cache misses remain `offline`, not `no_match`.

`alerts.poll` is an explicit bounded poll for GCN, VOEvent, ALeRCE, or Fink
endpoints. It stores a resumable cursor, ingests each packet through the same
content-addressed event inbox, and reports malformed packets individually.
There is no background network worker or automatic polling loop.

TESS blend diagnostics now include a prior concentration index and a target
flux-fraction sensitivity range under `blend.attribution_diagnostics`. These
quantify how dependent the catalog-relative attribution is on neighbor flux
assumptions; they are not PSF posteriors and do not restore resolved-survey
credit.

## Cross-messenger event graph and PSF deblending

`events.graph.correlate` and `events.graph.calibrate` expose
`association.event_to_event_correlation`/`calibrate_event_graph`: a
pairwise cross-messenger Bayes-factor statistic (Rayleigh spatial ratio ×
uniform-window temporal ratio) between distinct-provider events, calibrated
against a scrambled-time-slide null population for an estimated
false-coincidence rate. This is separate from `events.associate` (event-to-
candidate); neither changes candidate scores.

`surveys/rubin_tap.py` (`RubinTAPConnector`) is a dormant, credential-gated
direct Rubin/LSST TAP connector, registered but not enabled by default and
tested only against mocked responses (no real data-rights token exists yet).
`credentials.rubin.configure/status/clear` manage the stored token via the
same Windows-DPAPI backend TNS uses. ALeRCE (`surveys/alerce.py`) remains the
credential-free route to real LSST alerts/photometry today.

`tess_psf.py` fits a forward PSF scene model (fixed source positions, joint
per-cadence flux) for TESS target-pixel cutouts, producing a fitted flux
*posterior* per source alongside `flux_rmse`/`blend_attribution_accuracy`/
`injected_source_recovery` validation. This is new evidence alongside (not a
replacement for) `extract_photometry`'s aperture curve and catalog-relative
flux prior, and is not folded into ranking.

## Bridge and UI

The JSON-lines engine, Rust/Tauri commands, and typed TypeScript client expose
the event and diagnostics methods. The Events workspace supports local JSON
packet ingestion, cluster browsing, and replay status. Candidate Explain shows
calibration and evidence-completeness context while retaining the warning that
these are interpretation layers rather than re-ranked scores.
