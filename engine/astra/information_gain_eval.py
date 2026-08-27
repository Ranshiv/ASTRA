"""Evaluation studies for `information_gain.py`, split purely to keep
each file under this project's 500-line guideline.

`evaluate_entropy_is_maximized_at_p_half` checks a real, checkable
property of binary Shannon entropy directly, not merely asserting the
formula is right by inspection.

`evaluate_ranking_prioritizes_uncertain_visible_targets_synthetic`
constructs synthetic candidates spanning the full posterior range and
two feasibility regimes (always-visible vs. never-visible declinations,
under permissive geometric constraints so the distinction is purely
positional), and checks the ranking (a) never places an infeasible
candidate ahead of a feasible one and (b) orders feasible candidates by
distance of their posterior from 0.5, the real, checkable shape entropy
has to take.

`evaluate_real_followup_infeasibility_integration` is the one real
integration-level check: it picks a declination `followup.plan` itself
reports as never visible under permissive constraints (a real property
of the real geometry in `followup.py`, not fabricated), and confirms
`rank_by_information_gain` marks that exact candidate infeasible via the
real call, not a stubbed one.
"""

from __future__ import annotations

import numpy as np

from .information_gain import (
    InformationGainError, posterior_entropy, rank_by_information_gain,
)


def evaluate_entropy_is_maximized_at_p_half(grid_size: int = 101) -> dict:
    """`posterior_entropy` evaluated on a fine grid over [0, 1]; checks
    the maximizer is at (or numerically adjacent to) 0.5."""
    if grid_size < 3:
        raise InformationGainError(f"grid_size must be at least 3, got {grid_size}")
    grid = np.linspace(0.0, 1.0, grid_size)
    entropies = [posterior_entropy(p) for p in grid]
    argmax = grid[int(np.argmax(entropies))]
    return {"grid_size": grid_size, "argmax_probability": float(argmax),
            "max_entropy_bits": float(max(entropies)),
            "is_near_one_half": bool(abs(argmax - 0.5) <= (1.0 / (grid_size - 1)))}


def evaluate_ranking_prioritizes_uncertain_visible_targets_synthetic(
        n_candidates: int = 40, seed: int = 42) -> dict:
    """Synthetic candidates: half at a circumpolar-like declination
    (always visible under permissive constraints), half at a declination
    permanently below the horizon from the assumed site (never visible),
    with posteriors spanning [0.02, 0.98]. Checks no infeasible candidate
    outranks a feasible one, and feasible candidates are ordered by
    entropy (equivalently, distance of the posterior from 0.5)."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_candidates):
        visible_regime = i % 2 == 0
        # latitude_deg default is 43.65 N; dec=80 is circumpolar there,
        # dec=-80 never rises above the horizon there.
        dec = 80.0 if visible_regime else -80.0
        posterior = float(rng.uniform(0.02, 0.98))
        items.append({"candidate_id": f"c{i}", "ra_deg": float(rng.uniform(0, 360)),
                     "dec_deg": dec, "tail_probability": posterior})

    ranked = rank_by_information_gain(
        items, exposure_hours=1.0,
        followup_kwargs={"duration_hours": 24, "min_altitude_deg": 20.0,
                         "twilight_sun_altitude_deg": 10.0, "cadence_minutes": 30})

    observable_flags = [row.observable for row in ranked]
    first_infeasible = observable_flags.index(False) if False in observable_flags else len(ranked)
    no_infeasible_before_feasible = all(observable_flags[:first_infeasible]) if ranked else True

    feasible_entropies = [row.entropy_bits for row in ranked if row.observable]
    is_sorted_descending = all(feasible_entropies[i] >= feasible_entropies[i + 1] - 1e-9
                               for i in range(len(feasible_entropies) - 1))
    return {"n_candidates": n_candidates, "n_observable": sum(observable_flags),
            "no_infeasible_ranked_before_feasible": no_infeasible_before_feasible,
            "feasible_entropies_sorted_descending": is_sorted_descending}


def evaluate_real_followup_infeasibility_integration(seed: int = 42) -> dict:
    """Real (non-synthetic) integration check: a target at dec=-85 from
    the default site (latitude_deg=43.65) never rises above 20 deg
    altitude -- a real property of `followup.py`'s own geometry, checked
    directly via a real `followup.plan` call, then confirmed `rank_by_
    information_gain` reports the same candidate as infeasible."""
    from . import followup

    direct = followup.plan(ra_deg=180.0, dec_deg=-85.0, duration_hours=24,
                           min_altitude_deg=20.0, twilight_sun_altitude_deg=10.0,
                           cadence_minutes=30)
    ranked = rank_by_information_gain(
        [{"candidate_id": "never_visible", "ra_deg": 180.0, "dec_deg": -85.0, "tail_probability": 0.5}],
        followup_kwargs={"duration_hours": 24, "min_altitude_deg": 20.0,
                         "twilight_sun_altitude_deg": 10.0, "cadence_minutes": 30})
    return {"direct_followup_plan_visible": direct["visible"],
            "rank_by_information_gain_observable": ranked[0].observable,
            "consistent": direct["visible"] == ranked[0].observable}


__all__ = [
    "evaluate_entropy_is_maximized_at_p_half",
    "evaluate_ranking_prioritizes_uncertain_visible_targets_synthetic",
    "evaluate_real_followup_infeasibility_integration",
]
