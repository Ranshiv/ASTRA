# Data card

Population, selection effects, missingness, known biases, and TESS blending
limitations for ASTRA's acquired research corpus. Counts below are read
from sealed manifests under `research/datasets/manifests/`, never claimed
in advance — a manifest with no matching row here has not been acquired.

## Population

| Dataset ID | Surveys | Objects | Selection rule | License | Status |
|---|---|---|---|---|---|
| `core-demo-2026` | ZTF (150), Gaia (4), TESS (0) | 154 | Cone: RA=180.122°, Dec=22.411°, radius=90″ | per-survey, see manifest | acquired (demonstration scale) |

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
- **Label selection**: SIMBAD region labels (`research/acquire.py:
  _pull_simbad_labels`) are a positional cone lookup, not a matched
  cross-match to specific acquired object IDs — see the "Known biases"
  section below.

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

- **SIMBAD labels are not object-matched.** `_pull_simbad_labels` returns
  every SIMBAD entry in the queried field, keyed by SIMBAD's own
  `MAIN_ID` — not matched positionally to the ASTRA `object_id`s the
  acquisition returned. Treat these labels as "known objects present in
  this field", not "this ASTRA object has this label", until a positional
  cross-match is added.
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
