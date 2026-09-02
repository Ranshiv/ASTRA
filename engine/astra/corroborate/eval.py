"""Evaluation for `corroborate` (Direction 3: "corroboration as a general
multi-instrument anomaly library").

Three claims, each with its own function:

1. `evaluate_astronomy_equivalence` -- the domain-general core reproduces
   `crossmatch.group_sources`'s own behaviour, not merely a different
   implementation that happens to look similar. Randomised trials, not one
   hand-picked example.
2. `evaluate_domain_transfer` -- corroboration ("declare a candidate only
   when >=2 independent instruments resolve it") measurably reduces the
   false-positive rate in BOTH astronomy and the synthetic GW-style domain,
   using the LITERAL SAME `core.group_records` function for both. This is
   the actual "domain transfer" result.
3. `evaluate_scaling_with_systematics_correlation` -- the scaling claim:
   corroboration's false-positive reduction degrades as the two
   instruments' systematics become more correlated (`gw_adapter.py`'s
   `systematics_correlation` parameter), swept from independent to fully
   shared. This is the finding that is expected to generalise beyond
   either domain studied here.

`significance._ci_binomial` (already reused across this codebase --
`microlensing_eval.py` imports it the same way) gives every reported rate a
binomial confidence interval rather than a bare point estimate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import astronomy_adapter, core, gw_adapter
from ..significance import _ci_binomial
from ..surveys.base import SourceRef

DEFAULT_ASTRONOMY_RADIUS_ARCSEC = 2.0


def _rate_with_ci(successes: int, trials: int) -> dict[str, Any]:
    ci = _ci_binomial(successes, trials)
    return {"rate": round(successes / trials, 4) if trials else None,
           "successes": successes, "trials": trials,
           "ci": (None if ci is None else [round(v, 4) for v in ci])}


# --- 1. Astronomy equivalence -----------------------------------------------

def _random_by_survey(rng: np.random.Generator, *, n_ztf: int, n_gaia: int,
                      field_size_deg: float) -> dict[str, list[SourceRef]]:
    def _sources(survey: str, n: int) -> list[SourceRef]:
        return [SourceRef(survey=survey, object_id=f"{survey}_{i}",
                          ra_deg=float(rng.uniform(0, field_size_deg)),
                          dec_deg=float(rng.uniform(-field_size_deg / 2, field_size_deg / 2)))
               for i in range(n)]
    return {"ZTF": _sources("ZTF", n_ztf), "Gaia": _sources("Gaia", n_gaia)}


def evaluate_astronomy_equivalence(*, n_trials: int = 50, n_ztf: int = 15, n_gaia: int = 15,
                                   field_size_deg: float = 0.01, seed: int = 0) -> dict[str, Any]:
    """`astronomy_adapter.group_sources_via_core` vs `crossmatch.
    group_sources` on many random synthetic fields -- a stronger check than
    one hand-picked example, since a subtle tie-break or ordering
    difference would show up as an occasional mismatch across trials
    rather than a fixed failure.
    """
    from .. import crossmatch

    rng = np.random.default_rng(seed)
    mismatches = 0
    for trial in range(n_trials):
        by_survey = _random_by_survey(rng, n_ztf=n_ztf, n_gaia=n_gaia,
                                      field_size_deg=field_size_deg)
        legacy_groups = crossmatch.group_sources(by_survey, epoch=2024.5)
        core_groups = astronomy_adapter.group_sources_via_core(by_survey, epoch=2024.5)

        legacy_membership = sorted(
            tuple(sorted(group.to_dict()["members"].items())) for group in legacy_groups)
        core_membership = sorted(
            tuple(sorted(astronomy_adapter.group_to_source_membership(group).items()))
            for group in core_groups)
        if legacy_membership != core_membership:
            mismatches += 1

    return {"n_trials": n_trials, "n_mismatches": mismatches,
           "agreement_rate": round(1.0 - mismatches / n_trials, 4) if n_trials else None}


# --- 2. Domain transfer ------------------------------------------------------

def _synthetic_astronomy_population(rng: np.random.Generator, *, n_real: int, n_artifact: int,
                                    n_ambient_gaia: int, field_size_deg: float
                                    ) -> tuple[dict[str, list[SourceRef]], dict[str, bool]]:
    """A synthetic ZTF/Gaia field with known truth: `n_real` objects have a
    genuine, co-located Gaia counterpart; `n_artifact` are ZTF-only spurious
    detections (a bad-pixel-shaped false source, no real counterpart);
    `n_ambient_gaia` unrelated Gaia sources scattered across the same field
    give an artifact a real, if usually small, CHANCE of an accidental
    positional coincidence -- the honest astronomy analogue of `gw_adapter.
    py`'s correlated-glitch mechanism.
    """
    counter = 0
    ztf_sources: list[SourceRef] = []
    gaia_sources: list[SourceRef] = []
    truth: dict[str, bool] = {}

    def _position() -> tuple[float, float]:
        return (float(rng.uniform(0, field_size_deg)),
                float(rng.uniform(-field_size_deg / 2, field_size_deg / 2)))

    for _ in range(n_real):
        ra, dec = _position()
        object_id = f"ztf_{counter}"
        counter += 1
        ztf_sources.append(SourceRef(survey="ZTF", object_id=object_id, ra_deg=ra, dec_deg=dec))
        gaia_sources.append(SourceRef(survey="Gaia", object_id=f"gaia_of_{object_id}",
                                      ra_deg=ra, dec_deg=dec))
        truth[object_id] = True

    for _ in range(n_artifact):
        ra, dec = _position()
        object_id = f"ztf_{counter}"
        counter += 1
        ztf_sources.append(SourceRef(survey="ZTF", object_id=object_id, ra_deg=ra, dec_deg=dec))
        truth[object_id] = False

    for index in range(n_ambient_gaia):
        ra, dec = _position()
        gaia_sources.append(SourceRef(survey="Gaia", object_id=f"ambient_{index}",
                                      ra_deg=ra, dec_deg=dec))

    return {"ZTF": ztf_sources, "Gaia": gaia_sources}, truth


def _corroboration_fp_reduction(groups: list[core.Group], truth: dict[str, bool],
                                anchor: str) -> dict[str, Any]:
    """Compares the false-positive rate of two candidate-declaration
    policies over the SAME grouped population: "declare every anchor
    detection" (no corroboration) vs "declare only when >=2 independent
    instruments resolve it" (corroboration)."""
    baseline_ids = [group.members[anchor].identifier for group in groups if anchor in group.members]
    corroborated_ids = [group.members[anchor].identifier for group in groups
                        if anchor in group.members and group.resolved_instruments >= 2]

    baseline_fp = sum(1 for identifier in baseline_ids if not truth.get(identifier, False))
    corroborated_fp = sum(1 for identifier in corroborated_ids if not truth.get(identifier, False))

    return {
        "baseline": {**_rate_with_ci(baseline_fp, len(baseline_ids)), "n_declared": len(baseline_ids)},
        "corroborated": {**_rate_with_ci(corroborated_fp, len(corroborated_ids)),
                         "n_declared": len(corroborated_ids)},
    }


def evaluate_domain_transfer(*, astronomy_seed: int = 0, gw_seed: int = 0) -> dict[str, Any]:
    """False-positive reduction from corroboration in astronomy AND the
    synthetic GW-style domain, using the identical `core.group_records`
    function for both -- the actual domain-transfer result.
    """
    rng = np.random.default_rng(astronomy_seed)
    by_survey, astronomy_truth = _synthetic_astronomy_population(
        rng, n_real=40, n_artifact=120, n_ambient_gaia=60, field_size_deg=0.01)
    astronomy_groups = astronomy_adapter.group_sources_via_core(
        by_survey, radius_arcsec=DEFAULT_ASTRONOMY_RADIUS_ARCSEC, epoch=2024.5, anchor_survey="ZTF")
    astronomy_result = _corroboration_fp_reduction(astronomy_groups, astronomy_truth, "ZTF")

    gw_population = gw_adapter.generate_synthetic_detector_pair(
        n_real_events=40, n_glitches_a=120, n_glitches_b=120,
        systematics_correlation=0.0, seed=gw_seed)
    gw_groups = gw_adapter.group_population(gw_population)
    gw_result = _corroboration_fp_reduction(gw_groups, gw_population.truth, gw_adapter.DETECTOR_A)

    return {"astronomy": astronomy_result, "gw": gw_result}


# --- 3. Scaling with systematics correlation --------------------------------

def evaluate_scaling_with_systematics_correlation(
    *, correlations: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    n_glitches: int = 150, window_seconds: float = gw_adapter.DEFAULT_COINCIDENCE_WINDOW_SECONDS,
    seed: int = 0,
) -> dict[str, Any]:
    """Corroboration's false-positive rate among CORROBORATED (2-detector)
    declarations, as a function of `systematics_correlation`. The scaling
    claim: this should rise from near zero (independent systematics, real
    corroborating power) toward the uncorroborated baseline (fully shared
    systematics, corroboration provides no information) as correlation
    rises toward 1.
    """
    points = []
    for correlation in correlations:
        population = gw_adapter.generate_synthetic_detector_pair(
            n_real_events=0, n_glitches_a=n_glitches, n_glitches_b=n_glitches,
            systematics_correlation=correlation, window_seconds=window_seconds, seed=seed)
        groups = gw_adapter.group_population(population, window_seconds=window_seconds)
        corroborated_ids = [group.members[gw_adapter.DETECTOR_A].identifier for group in groups
                            if gw_adapter.DETECTOR_A in group.members
                            and group.resolved_instruments >= 2]
        # Every glitch here is false by construction (n_real_events=0), so
        # every corroborated declaration at this correlation IS a false
        # positive -- the count itself, not a separate truth lookup, is the
        # metric.
        points.append({"systematics_correlation": correlation,
                       "n_corroborated_false_positives": len(corroborated_ids),
                       "corroboration_fp_rate": _rate_with_ci(len(corroborated_ids), n_glitches)})

    return {"window_seconds": window_seconds, "n_glitches": n_glitches, "points": points}
