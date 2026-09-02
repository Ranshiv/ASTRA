"""Per-candidate feature attribution via occlusion (roadmap: explainability).

The 4 live detectors (`anomaly.py`) are heterogeneous: `run_isolation_forest`
is tree-based, but `run_lof`/`run_one_class_svm`/`run_pca_reconstruction` are
not, and `LocalOutlierFactor` isn't even fit in novelty mode -- there is no
`decision_function` for a modified row without a full refit. Using the real
`shap` package would mean mixing `TreeExplainer` and `KernelExplainer` per
detector (two different theoretical footings), a new dependency, and real
per-candidate latency from `KernelExplainer`'s sampling. That reasoning is
unchanged by this module's `explain_candidate_stable` addition below --
still occlusion, still no new dependency, still one uniform method across
all 4 detectors.

This module uses occlusion instead: model-agnostic, one uniform method
across all 4 detectors, no new dependency, and it reuses `anomaly.detect()`
completely unchanged rather than reaching into any detector's internals.
For one candidate row, each raw feature in turn is replaced with the
population's median for that feature (excluding the row's own value), the
whole ensemble is rerun, and the resulting consensus-score change is the
feature's impact. A positive impact means that feature's actual value is
what is raising the anomaly score above what a typical value would produce.

This reruns the full ensemble once per feature -- an explicit, on-demand,
per-candidate action (matching the cost class of `ExperimentsView`'s
existing "Run ablation" button), never something computed automatically for
every candidate in a batch.

`explain_candidate` reports a single occlusion impact per feature, using the
population median as the sole "typical" reference value. That is a real
approximation-quality gap (unlike SHAP, it never samples the marginal/joint
distribution): a feature whose "other" population is skewed or bimodal has
no single value that is honestly "typical," and the median-only impact can
look confidently wrong. `explain_candidate_stable` adds a second, cheap
check for exactly this: only for the handful of features `explain_candidate`
already found most impactful, it reruns occlusion at the 25th and 75th
percentile too (reusing the already-computed median result rather than a
third fresh rerun) and reports whether the impact's sign and magnitude hold
up across all three reference points. A full bootstrap over every raw
feature was rejected for cost -- it would multiply the already-expensive
per-feature full-ensemble rerun by the number of extra reference points
across ~28-43 columns, not just the top few -- so this stability check is
deliberately scoped to the features an observer would actually be shown.
"""

from __future__ import annotations

import numpy as np

from . import anomaly, feature_glossary
from .featurematrix import FeatureMatrix

DEFAULT_STABILITY_TOP = 5
DEFAULT_REFERENCE_QUANTILES = (0.25, 0.5, 0.75)
# An impact is "stable" only if its sign agrees at every reference quantile
# AND its spread is small relative to its own size -- a large but consistent
# std relative to a large mean is still a confidently-large impact.
STABILITY_RELATIVE_STD_THRESHOLD = 0.5


def _locate(result: anomaly.EnsembleResult, path: str) -> int | None:
    """Find a row by its identity's `path`, not by position -- `prepare()`
    drops non-finite rows but preserves order and identity-dict identity, so
    matching by path survives however many rows a rerun happens to keep."""
    for index, identity in enumerate(result.identities):
        if identity.get("path") == path:
            return index
    return None


def _occlude_and_score(matrix: FeatureMatrix, candidate_index: int, feature_index: int,
                       reference_value: float, path: str, baseline_score: float,
                       contamination: float, seed: int) -> float:
    """Rerun the ensemble with one feature of one row swapped to `reference_value`.

    Shared by `explain_candidate` and `explain_candidate_stable` so the two
    never duplicate the rerun/relocate logic -- the ~28-43x-per-candidate
    cost this incurs is the same either way, only the number of reference
    points per feature differs.
    """
    occluded_values = matrix.values.copy()
    occluded_values[candidate_index, feature_index] = reference_value
    occluded_matrix = FeatureMatrix(values=occluded_values, identities=matrix.identities,
                                    feature_names=matrix.feature_names,
                                    feature_version=matrix.feature_version)
    occluded_result = anomaly.detect(occluded_matrix, contamination=contamination, seed=seed)
    occluded_row = _locate(occluded_result, path)
    occluded_score = (float(occluded_result.consensus[occluded_row])
                      if occluded_row is not None else baseline_score)
    return baseline_score - occluded_score


def _baseline(matrix: FeatureMatrix, candidate_index: int,
             contamination: float, seed: int) -> tuple[str, float] | dict:
    """Common baseline-scoring step for both public functions.

    Returns `(path, baseline_score)` on success, or the `explainable: False`
    dict to return verbatim when the candidate can't be scored at all.
    """
    path = matrix.identities[candidate_index].get("path")
    if path is None:
        raise ValueError("candidate identity has no 'path' to track through re-scoring")

    baseline = anomaly.detect(matrix, contamination=contamination, seed=seed)
    baseline_row = _locate(baseline, path)
    if baseline_row is None:
        return {"path": path, "explainable": False,
               "reason": "candidate was skipped (non-finite features)"}
    return path, float(baseline.consensus[baseline_row])


def explain_candidate(matrix: FeatureMatrix, candidate_index: int,
                      contamination: float = anomaly.DEFAULT_CONTAMINATION,
                      seed: int = 42, top: int = 10) -> dict:
    """Explain one candidate's consensus score by occluding each feature.

    Returns ``{"path", "explainable", "baseline_score"?, "components"?,
    "reason"?}``. ``components`` is a list of ``{"feature", "value",
    "typical", "impact"}`` sorted by ``|impact|`` descending, truncated to
    ``top``. A candidate dropped by `anomaly.prepare()`'s finite-mask (any
    non-finite feature) returns ``explainable: False`` with a reason rather
    than raising.
    """
    baselined = _baseline(matrix, candidate_index, contamination, seed)
    if isinstance(baselined, dict):
        return baselined
    path, baseline_score = baselined

    values = matrix.values
    target = values[candidate_index]
    components: list[dict] = []

    for feature_index, name in enumerate(matrix.feature_names):
        original = float(target[feature_index])
        if not np.isfinite(original):
            continue

        column = values[:, feature_index]
        others = column[np.isfinite(column)]
        others = others[others != original]
        median = float(np.median(others)) if others.size else original
        if median == original:
            components.append({"feature": name, "value": original,
                               "typical": median, "impact": 0.0})
            continue

        impact = _occlude_and_score(matrix, candidate_index, feature_index, median,
                                    path, baseline_score, contamination, seed)
        components.append({"feature": name, "value": original, "typical": median,
                           "impact": impact})

    components.sort(key=lambda component: abs(component["impact"]), reverse=True)
    return {"path": path, "explainable": True, "baseline_score": baseline_score,
           "components": components[:top]}


def explain_candidate_stable(matrix: FeatureMatrix, candidate_index: int,
                             contamination: float = anomaly.DEFAULT_CONTAMINATION,
                             seed: int = 42, top: int = 10,
                             stability_top: int = DEFAULT_STABILITY_TOP,
                             reference_quantiles: tuple[float, ...] = DEFAULT_REFERENCE_QUANTILES,
                             ) -> dict:
    """Like `explain_candidate`, plus a confidence signal and a narrative.

    For the `stability_top` highest-|impact| features from a first,
    unchanged median-only occlusion pass, additionally occludes at the 25th
    and 75th percentile of the "other" population (the median result is
    reused, not recomputed) and reports whether the impact holds up:
    `impact_mean`, `impact_std`, `impact_min`, `impact_max`, and a boolean
    `stable`. Every component also carries `label`/`unit`/`description` from
    `feature_glossary`. A deterministic, template-built `narrative` (no LLM
    call) summarizes the top stable drivers in physical terms, and says so
    explicitly when the top driver is NOT stable rather than presenting a
    reference-sensitive number as confident.

    Non-finite/not-explainable behaviour is identical to `explain_candidate`.
    """
    base = explain_candidate(matrix, candidate_index, contamination=contamination,
                             seed=seed, top=max(top, stability_top))
    if not base.get("explainable", False):
        return base

    path = base["path"]
    baseline_score = base["baseline_score"]
    values = matrix.values
    feature_index_by_name = {name: index for index, name in enumerate(matrix.feature_names)}
    quantiles = sorted(reference_quantiles)
    median_quantile = 0.5

    components = [dict(component) for component in base["components"][:top]]
    for rank, component in enumerate(components):
        if rank >= stability_top:
            continue
        name = component["feature"]
        feature_index = feature_index_by_name[name]
        column = values[:, feature_index]
        others = column[np.isfinite(column)]
        others = others[others != component["value"]]
        if others.size == 0:
            continue

        impacts: list[float] = []
        for quantile in quantiles:
            if quantile == median_quantile:
                impacts.append(component["impact"])
                continue
            reference = float(np.quantile(others, quantile))
            impacts.append(_occlude_and_score(
                matrix, candidate_index, feature_index, reference,
                path, baseline_score, contamination, seed))

        impact_mean = float(np.mean(impacts))
        impact_std = float(np.std(impacts))
        signs = {1 if value > 0 else (-1 if value < 0 else 0) for value in impacts}
        sign_consistent = len(signs) == 1
        relative_spread = (impact_std / abs(impact_mean)) if impact_mean != 0 else float("inf")
        stable = sign_consistent and relative_spread <= STABILITY_RELATIVE_STD_THRESHOLD

        component.update({
            "impact_mean": impact_mean, "impact_std": impact_std,
            "impact_min": float(min(impacts)), "impact_max": float(max(impacts)),
            "stable": stable,
        })

    for component in components:
        info = feature_glossary.describe(component["feature"])
        component.update({"label": info["label"], "unit": info["unit"],
                          "description": info["description"]})

    narrative = _narrative(components)
    result = dict(base)
    result["components"] = components
    result["narrative"] = narrative
    return result


def _narrative(components: list[dict], max_drivers: int = 3) -> str:
    """Deterministic, template-built summary of the top stable drivers.

    No LLM call: this is a formatting function over already-computed
    numbers, so its output is exactly reproducible from the same input.
    """
    stable_drivers = [c for c in components if c.get("stable")]
    if not stable_drivers:
        if components and "stable" in components[0]:
            top_label = components[0].get("label", components[0]["feature"])
            return (f"The strongest driver, {top_label}, is reference-sensitive "
                    "(its impact changes depending on which typical value is "
                    "used) -- treat this candidate's ranking with caution "
                    "before committing follow-up time.")
        return "No stability-checked drivers are available for this candidate."

    phrases = []
    for component in stable_drivers[:max_drivers]:
        label = component.get("label", component["feature"])
        value_text = feature_glossary.format_value(component["feature"], component["value"])
        direction = "raises" if component["impact"] > 0 else "suppresses"
        phrases.append(f"{label} ({value_text}, {direction} the anomaly score)")

    lead = "This candidate's flag is most reliably explained by "
    if len(phrases) == 1:
        body = phrases[0]
    elif len(phrases) == 2:
        body = f"{phrases[0]} and {phrases[1]}"
    else:
        body = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    unstable_top = components and not components[0].get("stable", True) \
        and components[0] not in stable_drivers[:max_drivers]
    caveat = ""
    if unstable_top:
        top_label = components[0].get("label", components[0]["feature"])
        caveat = (f" (the single strongest raw impact, {top_label}, is "
                  "reference-sensitive and is excluded from this summary).")

    return f"{lead}{body}.{caveat}"


__all__ = ["explain_candidate", "explain_candidate_stable"]
