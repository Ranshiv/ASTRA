"""Retrospective scheduling-policy simulator (Direction 1 evaluation
harness: "closed-loop decision-theoretic scheduling").

Publishing a scheduling result without a telescope means the PRIMARY claim
has to be a simulation, not a live campaign. This replays several nights
against a synthetic candidate population with a KNOWN hidden ground truth
("real" vs "artifact") -- the same "labels true by construction" discipline
`discard_pile_eval.py`/`review_experiment_eval.py` already use for this
codebase's other simulated-truth studies -- and compares three observing
PRIORITY policies through the exact same real sequencer
(`schedule.build_night_schedule`'s packing, windows, and local search),
via that function's `priority_fn` override, so the comparison isolates
priority order rather than confounding it with three different scheduler
implementations:

- `information_gain`: `information_gain.posterior_entropy` -- observe the
  most AMBIGUOUS candidates first (this codebase's actual active-learning
  claim: an observation of an uncertain candidate teaches the most).
- `rank_order`: `1 - posterior` -- observe the most CONFIDENT-looking
  candidates first, the naive "work through your best candidates" baseline
  every un-optimised review queue already does without an information-
  theoretic prioritiser.
- `random`: a seeded shuffle each night.

Each night, every SCHEDULED candidate's posterior moves toward the hidden
truth by `RESOLUTION_FRACTION` of the remaining gap (`posterior +=
RESOLUTION_FRACTION * (truth - posterior)`) -- a synthetic, explicitly
idealised resolution model (repeated observation converges exponentially
toward certainty), not a real per-instrument noise model, the same
documented idealisation `information_gain.py`'s own docstring already
uses for "a noiseless follow-up... fully resolve[s] the current
uncertainty." A candidate is RESOLVED once its entropy drops below
`RESOLVED_ENTROPY_THRESHOLD`; resolved candidates drop out of future
nights' pools -- there is nothing left to schedule them for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np

from . import information_gain
from . import schedule as sch

Policy = Literal["information_gain", "rank_order", "random"]
POLICIES: tuple[Policy, ...] = ("information_gain", "rank_order", "random")

RESOLVED_ENTROPY_THRESHOLD = 0.1  # bits
RESOLUTION_FRACTION = 0.6
DEFAULT_START_UTC = "2026-12-01T00:00:00Z"  # a long winter night at the default site


def _priority_fn(policy: Policy, rng: np.random.Generator):
    if policy == "information_gain":
        return lambda item: information_gain.posterior_entropy(item["tail_probability"])
    if policy == "rank_order":
        return lambda item: 1.0 - float(item["tail_probability"])
    if policy == "random":
        draws = {}
        def _random_priority(item):
            key = item["candidate_id"]
            if key not in draws:
                draws[key] = float(rng.uniform())
            return draws[key]
        return _random_priority
    raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")


def _synthetic_population(rng: np.random.Generator, *, n_candidates: int,
                          real_fraction: float, field_size_deg: float
                          ) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """`n_candidates` synthetic candidates scattered across a circumpolar
    patch of sky (dec near 80 deg -- always above the default site's
    minimum altitude regardless of the hour, matching `tests/test_schedule.
    py`'s own "always visible" convention), each with a hidden truth
    probability (1.0 real, 0.0 artifact) and a starting posterior that has
    NOT yet converged to it -- otherwise there would be nothing left for a
    follow-up campaign to resolve.
    """
    n_real = int(round(n_candidates * real_fraction))
    truths = [1.0] * n_real + [0.0] * (n_candidates - n_real)
    rng.shuffle(truths)

    candidates = []
    truth_by_id = {}
    for index, truth in enumerate(truths):
        candidate_id = f"cand{index}"
        # Starting posterior: correlated with truth (a real detector signal,
        # not pure noise) but with substantial overlap -- ambiguous, the
        # premise a follow-up campaign needs to be worth running at all.
        posterior = float(np.clip(truth + rng.normal(0, 0.35), 0.02, 0.98))
        candidates.append({
            "candidate_id": candidate_id,
            "ra_deg": float(rng.uniform(0, field_size_deg)) + (index % 12) * 30.0,
            "dec_deg": 80.0 + float(rng.uniform(-2.0, 2.0)),
            "tail_probability": posterior,
        })
        truth_by_id[candidate_id] = truth
    return candidates, truth_by_id


def _advance_utc(start_utc: str, nights: int) -> str:
    parsed = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    return (parsed + timedelta(days=nights)).isoformat()


@dataclass
class SimulationResult:
    policy: Policy
    n_nights: int
    resolved_by_night: list[int] = field(default_factory=list)          # cumulative
    telescope_hours_by_night: list[float] = field(default_factory=list)  # cumulative
    entropy_reduced_by_night: list[float] = field(default_factory=list)  # cumulative, bits

    def to_dict(self) -> dict:
        return {"policy": self.policy, "n_nights": self.n_nights,
               "resolved_by_night": self.resolved_by_night,
               "telescope_hours_by_night": [round(h, 4) for h in self.telescope_hours_by_night],
               "entropy_reduced_by_night": [round(v, 4) for v in self.entropy_reduced_by_night]}


def resolved_at_hour_budget(result: SimulationResult, hour_budget: float) -> int:
    """The resolved count as of the LAST night whose cumulative telescope-
    hours did not exceed `hour_budget` -- the resolution-vs-time-budget
    curve's value at one point on the x-axis."""
    resolved = 0
    for hours, count in zip(result.telescope_hours_by_night, result.resolved_by_night):
        if hours <= hour_budget:
            resolved = count
        else:
            break
    return resolved


def entropy_reduced_at_hour_budget(result: SimulationResult, hour_budget: float) -> float:
    reduced = 0.0
    for hours, value in zip(result.telescope_hours_by_night, result.entropy_reduced_by_night):
        if hours <= hour_budget:
            reduced = value
        else:
            break
    return reduced


def simulate_policy(candidates: list[dict[str, Any]], truth: dict[str, float], *,
                    policy: Policy, n_nights: int = 20, duration_hours: float = 8.0,
                    exposure_hours: float = 0.5, start_utc: str = DEFAULT_START_UTC,
                    seed: int = 0) -> SimulationResult:
    """Run one policy over `n_nights`, tracking cumulative resolved
    candidates, cumulative telescope-hours, and cumulative entropy reduced
    after every night -- the raw material for a resolution-vs-time-budget
    curve under EITHER metric. Both matter and are reported separately
    (`evaluate_scheduling_policies`'s own docstring): "candidates resolved"
    and "entropy reduced" are not the same objective, and a policy that
    wins on one need not win on the other -- see that function's docstring
    for the measured relationship between them.
    """
    rng = np.random.default_rng(seed)
    pool = {item["candidate_id"]: dict(item) for item in candidates}
    initial_entropy = {cid: information_gain.posterior_entropy(item["tail_probability"])
                       for cid, item in pool.items()}
    resolved_ids: set[str] = set()
    resolved_by_night: list[int] = []
    telescope_hours_by_night: list[float] = []
    entropy_reduced_by_night: list[float] = []
    cumulative_hours = 0.0

    for night in range(n_nights):
        active = [item for cid, item in pool.items() if cid not in resolved_ids]
        night_start = _advance_utc(start_utc, night)
        if active:
            night_schedule = sch.build_night_schedule(
                active, start_utc=night_start, duration_hours=duration_hours,
                exposure_hours=exposure_hours, priority_fn=_priority_fn(policy, rng))
            cumulative_hours += night_schedule.total_exposure_hours
            for observation in night_schedule.observations:
                item = pool[observation.candidate_id]
                current = float(item["tail_probability"])
                item["tail_probability"] = current + RESOLUTION_FRACTION * (
                    truth[observation.candidate_id] - current)
                if information_gain.posterior_entropy(item["tail_probability"]) < RESOLVED_ENTROPY_THRESHOLD:
                    resolved_ids.add(observation.candidate_id)
        resolved_by_night.append(len(resolved_ids))
        telescope_hours_by_night.append(cumulative_hours)
        current_total_entropy = sum(information_gain.posterior_entropy(item["tail_probability"])
                                    for item in pool.values())
        entropy_reduced_by_night.append(sum(initial_entropy.values()) - current_total_entropy)

    return SimulationResult(policy=policy, n_nights=n_nights, resolved_by_night=resolved_by_night,
                            telescope_hours_by_night=telescope_hours_by_night,
                            entropy_reduced_by_night=entropy_reduced_by_night)


def evaluate_scheduling_policies(*, n_candidates: int = 60, real_fraction: float = 0.4,
                                 n_nights: int = 20, n_runs: int = 8,
                                 hour_budgets: tuple[float, ...] = (16.0, 40.0, 80.0, 160.0),
                                 seed: int = 0) -> dict[str, Any]:
    """Head-to-head resolution-vs-time-budget curve for all three policies,
    over `n_runs` independently seeded synthetic populations, under BOTH of
    two genuinely different metrics at each `hour_budgets` checkpoint:

    - `resolved_candidates`: the raw count of candidates whose entropy has
      crossed `RESOLVED_ENTROPY_THRESHOLD`.
    - `entropy_reduced_bits`: total posterior-entropy reduction across the
      WHOLE population, `information_gain`'s own actual optimisation
      target.

    Measured, not assumed: these two metrics do NOT agree on which policy
    is ahead, especially at small time budgets -- `information_gain`
    (entropy-first) invests repeatedly in the FEW most ambiguous
    candidates, which wins decisively on `entropy_reduced_bits` early in a
    campaign but LAGS `rank_order`/`random` on raw `resolved_candidates`
    at the same checkpoint, because a candidate that starts near-certain
    needs far fewer observations to cross the resolved threshold than one
    that starts near-ambiguous. Both metrics converge once the campaign
    runs long enough to resolve the whole population -- this is why a
    result reported only at the END of a long campaign would have hidden
    the actual difference between the policies; see `hour_budgets`'
    smaller values for where it is visible.
    """
    rng = np.random.default_rng(seed)
    runs: dict[str, list[SimulationResult]] = {policy: [] for policy in POLICIES}

    for run in range(n_runs):
        run_seed = int(rng.integers(0, 2**31 - 1))
        population, truth = _synthetic_population(
            np.random.default_rng(run_seed), n_candidates=n_candidates,
            real_fraction=real_fraction, field_size_deg=1.0)
        for policy in POLICIES:
            runs[policy].append(simulate_policy(population, truth, policy=policy,
                                                n_nights=n_nights, seed=run_seed))

    summary: dict[str, Any] = {}
    for policy in POLICIES:
        curve = []
        for budget in hour_budgets:
            resolved = [resolved_at_hour_budget(result, budget) for result in runs[policy]]
            entropy = [entropy_reduced_at_hour_budget(result, budget) for result in runs[policy]]
            curve.append({
                "hour_budget": budget,
                "resolved_candidates": {"mean": round(float(np.mean(resolved)), 3),
                                        "ci": [round(float(np.quantile(resolved, 0.025)), 3),
                                              round(float(np.quantile(resolved, 0.975)), 3)]},
                "entropy_reduced_bits": {"mean": round(float(np.mean(entropy)), 3),
                                         "ci": [round(float(np.quantile(entropy, 0.025)), 3),
                                               round(float(np.quantile(entropy, 0.975)), 3)]},
            })
        summary[policy] = {"curve": curve}
    return {"n_candidates": n_candidates, "n_nights": n_nights, "n_runs": n_runs,
           "hour_budgets": list(hour_budgets), "by_policy": summary}


def evaluate_robustness(*, n_candidates: int = 60, real_fraction: float = 0.4, n_nights: int = 8,
                        early_hour_budget: float = 40.0, n_runs: int = 6,
                        noise_levels: tuple[float, ...] = (0.0, 0.1, 0.3, 0.6),
                        seed: int = 0) -> dict[str, Any]:
    """Degrades the information-gain policy's OWN input -- Gaussian noise
    added to the posterior it prioritises by -- and reports where its
    EARLY-campaign advantage over `rank_order` on `entropy_reduced_bits`
    (the metric it is actually designed to optimise -- see `evaluate_
    scheduling_policies`'s docstring for why `resolved_candidates` is the
    WRONG metric to check this against: `rank_order` leads on it early
    regardless of noise) narrows or disappears. A scheduling result
    without this check is not credible: a prioritiser is only as good as
    the probability estimate it prioritises by, and this makes that
    dependency visible rather than assumed away.
    """
    rng = np.random.default_rng(seed)
    points = []
    for noise in noise_levels:
        info_gain_totals, rank_order_totals = [], []
        for run in range(n_runs):
            run_seed = int(rng.integers(0, 2**31 - 1))
            population, truth = _synthetic_population(
                np.random.default_rng(run_seed), n_candidates=n_candidates,
                real_fraction=real_fraction, field_size_deg=1.0)
            noisy_population = [
                {**item, "tail_probability": float(np.clip(
                    item["tail_probability"] + np.random.default_rng(run_seed + 1).normal(0, noise),
                    0.01, 0.99))}
                for item in population
            ]
            info_gain = simulate_policy(noisy_population, truth, policy="information_gain",
                                        n_nights=n_nights, seed=run_seed)
            rank_order = simulate_policy(population, truth, policy="rank_order",
                                         n_nights=n_nights, seed=run_seed)
            info_gain_totals.append(entropy_reduced_at_hour_budget(info_gain, early_hour_budget))
            rank_order_totals.append(entropy_reduced_at_hour_budget(rank_order, early_hour_budget))
        points.append({
            "noise_std": noise,
            "information_gain_entropy_reduced": round(float(np.mean(info_gain_totals)), 3),
            "rank_order_entropy_reduced": round(float(np.mean(rank_order_totals)), 3),
            "information_gain_advantage": round(
                float(np.mean(info_gain_totals) - np.mean(rank_order_totals)), 3),
        })
    return {"n_nights": n_nights, "early_hour_budget": early_hour_budget, "n_runs": n_runs,
           "points": points}
