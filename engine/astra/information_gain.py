"""Information-gain follow-up scheduler (roadmap item 37, P1).

`followup.py`'s `plan()` already answers "when, or whether at all, CAN
this target be observed" -- real airmass/twilight/lunar-separation/
weather/facility-constraint geometry, confirmed by reading it in full.
Confirmed genuinely missing by grep (zero hits for "information_gain"/
"entropy"/"expected_value" anywhere in `engine/astra` before this
session) and by `followup.plan`'s own single-target signature (`plan(*,
ra_deg, dec_deg, ...)`, `followup.py` line ~134): it never ranks or
compares MULTIPLE candidates against each other, and has no concept of
which target is most WORTH observing. This module adds exactly that
ranking layer, calling `followup.plan` UNCHANGED for the
visibility/feasibility side rather than re-deriving any of its
sidereal-time, solar, or lunar geometry.

Framing: Bayesian experimental design measures the value of a proposed
experiment by the EXPECTED REDUCTION in posterior entropy it would
produce (Lindley 1956, "On a Measure of the Information Provided by an
Experiment," Annals of Mathematical Statistics; MacKay 1992,
"Information-Based Objective Functions for Active Data Selection,"
Neural Computation). `posterior_entropy` is the standard binary Shannon
entropy of a candidate's current "is this real?" posterior probability.
`entropy_from_tail_probability` accepts either `significance.
calibrate`/`annotate`'s `tail_probability` or `conformal.
conformal_p_values`'s p-value UNCHANGED -- item 34's own docstring
already establishes these are literally the same empirical-tail
quantity, so both are valid posterior-probability inputs here without
new arithmetic.

Confirmed UNREACHABLE, stated up front: a real per-instrument
measurement-noise model for what a follow-up observation would actually
resolve. No survey connector in `engine/astra/surveys/` records a
per-exposure signal-to-noise/depth model this codebase could use to
compute an exact myopic expected posterior update. This module therefore
uses the standard, citable IDEALIZED upper bound used throughout the
active-learning/Bayesian-design literature when no noise model is
available: a noiseless follow-up is assumed to fully resolve the current
uncertainty, so expected information gain is approximated by the CURRENT
entropy itself (Houlsby, Huszar, Lengyel & Ghahramani 2011, "Bayesian
Active Learning for Classification and Preference Learning," uses the
same current-entropy term as the dominant component of their BALD
acquisition function). This is an upper bound, not an exact expected
gain under a real noise model -- stated as a limitation in `rank_by_
information_gain`'s own docstring, not glossed over.

`rank_by_information_gain` divides that entropy by `exposure_hours` (the
assumed cost, in observing hours, of taking one follow-up observation of
a target) to produce the roadmap item's own "expected posterior entropy
reduction per observing hour" metric, and excludes -- ranks last, with a
`None` value -- any candidate `followup.plan` itself reports as not
currently visible, reusing its real `visible`/`windows` output rather
than re-implementing feasibility.

Explicitly NOT done: does not modify `followup.py` in any way -- `plan`
is called once per candidate, never forked. Does not import `anomaly.py`
or `conformal.py`; takes plain posterior-probability floats, staying
detector-agnostic like `conformal.py`. Like every other opt-in research
module in this codebase, NOT wired into `rpc.py`, `scoring.WEIGHTS`, or
`evidence.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from . import followup


class InformationGainError(ValueError):
    """A posterior probability, exposure cost, or candidate input was invalid."""


def posterior_entropy(probability: float) -> float:
    """Binary Shannon entropy, in bits, of a probability in [0, 1].
    Maximal (1.0 bit) at `probability=0.5`; exactly 0 at either extreme."""
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise InformationGainError(f"probability must be in [0, 1], got {p}")
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def entropy_from_tail_probability(tail_probability: float) -> float:
    """`posterior_entropy` applied to a `significance.calibrate`/
    `annotate` tail probability or a `conformal.conformal_p_values`
    p-value -- both are the same empirical-tail quantity (established by
    `conformal.py`'s own docstring), reused here unchanged as an
    "is this real?" posterior-probability proxy."""
    return posterior_entropy(tail_probability)


@dataclass(frozen=True)
class FollowUpValue:
    candidate_id: str
    entropy_bits: float
    exposure_hours: float
    observable: bool
    observable_hours_in_window: float
    value_per_hour: float | None

    def to_dict(self) -> dict:
        return {"candidate_id": self.candidate_id, "entropy_bits": round(self.entropy_bits, 6),
                "exposure_hours": self.exposure_hours, "observable": self.observable,
                "observable_hours_in_window": round(self.observable_hours_in_window, 4),
                "value_per_hour": round(self.value_per_hour, 6) if self.value_per_hour is not None else None}


def rank_by_information_gain(items: list[dict[str, Any]], *, exposure_hours: float = 1.0,
                             followup_kwargs: dict[str, Any] | None = None) -> list[FollowUpValue]:
    """Rank candidates by entropy-per-`exposure_hours`, calling
    `followup.plan` once per candidate (UNCHANGED) for feasibility. Each
    item needs `candidate_id`, `ra_deg`, `dec_deg`, and a posterior
    probability under `tail_probability` or `p_value`. Infeasible
    candidates (`followup.plan`'s own `visible=False`) sort last with
    `value_per_hour=None`. THIS IS AN UPPER-BOUND APPROXIMATION -- see
    the module docstring for why an exact expected-gain-under-noise
    computation is not available in this codebase."""
    if exposure_hours <= 0:
        raise InformationGainError(f"exposure_hours must be positive, got {exposure_hours}")
    if not items:
        return []

    kwargs = dict(followup_kwargs or {})
    cadence_minutes = kwargs.get("cadence_minutes", 10)
    rows: list[FollowUpValue] = []
    for item in items:
        candidate_id = item.get("candidate_id")
        if not candidate_id:
            raise InformationGainError("every item needs a candidate_id")
        posterior = item.get("tail_probability", item.get("p_value"))
        if posterior is None:
            raise InformationGainError(f"{candidate_id!r} has neither tail_probability nor p_value")
        entropy = posterior_entropy(posterior)

        plan_result = followup.plan(ra_deg=item["ra_deg"], dec_deg=item["dec_deg"], **kwargs)
        observable_hours = sum(window["slots"] for window in plan_result["windows"]) \
            * cadence_minutes / 60.0
        observable = bool(plan_result["visible"])
        value = (entropy / exposure_hours) if observable else None
        rows.append(FollowUpValue(candidate_id=str(candidate_id), entropy_bits=entropy,
                                  exposure_hours=float(exposure_hours), observable=observable,
                                  observable_hours_in_window=observable_hours, value_per_hour=value))

    rows.sort(key=lambda row: (row.value_per_hour is not None, row.value_per_hour or 0.0), reverse=True)
    return rows


__all__ = [
    "InformationGainError", "posterior_entropy", "entropy_from_tail_probability",
    "FollowUpValue", "rank_by_information_gain",
]
