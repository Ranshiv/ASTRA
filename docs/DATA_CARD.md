# Data card

Population, selection effects, missingness, known biases, and TESS blending
limitations for ASTRA's acquired research corpus. Counts below are read
from sealed manifests under `research/datasets/manifests/`, never claimed
in advance — a manifest with no matching row here has not been acquired.

## Population

| Dataset ID | Surveys | Objects | Selection rule | License | Status |
|---|---|---|---|---|---|
| `core-demo-2026` | ZTF (150), Gaia (4), TESS (0) | 154 | Cone: RA=180.122°, Dec=22.411°, radius=90″ | per-survey, see manifest | acquired (demonstration scale) |
| `p0-validation-2026` | ZTF (60), Gaia (11), TESS (0) | 71 | Cone: RA=210.5°, Dec=-5.2°, radius=90″ | per-survey, see manifest | acquired (P0 plan pipeline-validation scale, not a leaderboard claim — see docs/RESULTS.md) |
| `p0-pleiades-2026` | ZTF (300), Gaia (200), TESS (8) | 508 | Cone: RA=56.75°, Dec=24.12°, radius=300″ (Pleiades, chosen for real SIMBAD label density) | per-survey, see manifest | acquired (29 real SIMBAD labels — see docs/RESULTS.md for class breakdown and why this is not yet a real-label benchmark) |

Update this table by reading `research/datasets/manifests/*.json` — each
file's `dataset_id`, `total_objects()`-equivalent `row_count`,
`selection_rule`, and `license` fields — not by hand-estimating counts.
See docs/RESULTS.md for this dataset's full acquisition provenance
(cassette count, manifest hash, and the caveats on what actually got
scored).

## Selection effects

- **Cone-search selection**: the core demonstration corpus is one cone
  query (RA/Dec/radius) per acquisition, not a magnitude-limited or
  volume-limited sample — it inherits whatever ZTF/Gaia/TESS's own survey
  footprints and depth put in that cone. A discovery-rate or completeness
  claim from this corpus describes that cone, not the sky.
- **Cross-survey overlap requirement**: an object only appears in a
  cross-survey feature matrix if the connectors it came through all
  returned a match; objects only one survey observes are undercounted by
  construction until a per-survey (rather than joint) accounting is added.
- **Label selection**: `_pull_simbad_labels` now positionally cross-matches
  each real acquired object against SIMBAD's field results, so a
  `LabelRecord` is tied to a specific acquired object ID (see the "Known
  biases" section below) — but the underlying SIMBAD *query* is still one
  cone lookup per field, so label yield depends entirely on how densely
  SIMBAD has catalogued that field. `core-demo-2026` and
  `p0-validation-2026`'s arbitrary cones returned 1 and 0 real labels;
  `p0-pleiades-2026`, chosen deliberately for a dense, well-studied cluster
  field, returned 29. **This is itself a selection effect, not a fixed
  yield rate**: choosing fields for label density selects for exactly the
  kind of already-catalogued object a discovery pipeline cares least about
  (see the next bullet).
- **Cluster-field label bias**: `p0-pleiades-2026`'s 29 real labels span 8
  SIMBAD object types, all astrophysically "interesting" by construction
  (young stellar objects, flare stars, spectroscopic binaries — the normal
  population of a well-studied open cluster core). A field chosen this way
  has almost no clean "boring negative" class to contrast against, unlike
  an arbitrary discovery-oriented cone. Do not read this dataset's label
  set as representative of the sky's true interesting/boring ratio.

## Missingness

- A connector marked `metadata-only` or `dormant` in
  `research/sources/connector_status.yaml` contributes identity/position
  rows but no light curve; features derived from it are absent, not zero,
  and `FeatureMatrix.finite_mask()` already excludes those rows from
  detector scoring rather than imputing them.
- TESS pixel-level (TPF) data is a separate acquisition step from TESS
  light-curve acquisition; a dataset manifest recording light curves does
  not imply pixel data is also present (see docs/BENCHMARKS.md's artifact
  rejection track status).

## Known biases

- **SIMBAD labels are now object-matched.** `_pull_simbad_labels` positionally
  cross-matches each acquired object's real discovery-time position against
  SIMBAD's field results via `crossmatch.match_catalogs`, with a confidence
  score derived from separation and local crowding (competing counterparts
  within the match radius lower it). Every `LabelRecord.object_id` is one of
  the acquisition's own object IDs; an object with no real counterpart
  within the radius gets no label. Still a single cone lookup per field, not
  a batched join — the thousands-of-objects release needs a batched
  VizieR/TAP cross-match instead, per that function's own docstring.
- **Injected-anomaly benchmarks measure injection recovery, not discovery.**
  Per docs/BENCHMARKS.md, the cross-survey anomaly track's positive class
  is a synthetic feature-space perturbation of a real object, not a
  verified real anomaly. A method's AUPRC here says it can recover the
  *injected* perturbation shape; it does not certify sensitivity to
  phenomena nobody thought to inject (the same caveat `evaluate.py`'s
  module docstring already states for its own injection studies).

## TESS blending

TESS's large pixel scale means a single light curve can blend flux from
multiple physical sources. `engine/astra/tess_psf.py`'s forward PSF
scene-model fitting and `host_association.py`'s source-attribution work
address this for candidate-scale deblending; a feature matrix built
without going through that path (e.g. directly from `featurematrix.build`
on raw TESS light curves) does not correct for blending, and any
per-object claim from such a matrix should be read as "the blended
aperture's" signal, not necessarily the nominal target's alone.
