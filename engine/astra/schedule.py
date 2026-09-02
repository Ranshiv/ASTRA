"""Decision-theoretic observation scheduling (Direction 1 of the research
plan adopted 2026-08-29: "closed-loop decision-theoretic scheduling").

`followup.plan` already answers "when can this ONE target be observed", and
`information_gain.rank_by_information_gain` already ranks MULTIPLE
candidates by expected value per observing hour -- but nothing before this
module turns that ranking into an actual ORDERED SEQUENCE for one night,
subject to each candidate's own visibility windows and to how much time is
actually available. This module is that sequencer, calling both UNCHANGED:
`followup.plan` for feasibility windows, `information_gain.posterior_entropy`
for the per-candidate value.

Scope, stated up front rather than glossed over: `exposure_hours` is a
single, uniform value across every scheduled candidate (matching
`rank_by_information_gain`'s own uniform-`exposure_hours` design), so a
scheduled slot is always the same length and a local-search SWAP between
two already-scheduled slots is well-defined. Slew cost (`followup.
angular_separation_deg` between consecutive targets, divided by
`DEFAULT_SLEW_RATE_DEG_PER_SEC`) is a REPORTED diagnostic and the
local-search pass's optimisation objective, not a hard feasibility
constraint during greedy insertion -- an exact slew-feasible interval
scheduler needs a mount-specific slew-rate table this codebase does not
have, the same class of approximation `followup.py`'s own low-precision
sidereal/solar/lunar ephemeris already uses and states plainly in its own
`caveats` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from . import followup, information_gain

# A reasonable medium-telescope approximation, not a facility-specific slew
# table -- see the module docstring's stated scope.
DEFAULT_SLEW_RATE_DEG_PER_SEC = 2.0
DEFAULT_EXPOSURE_HOURS = 0.5


def _parse_utc(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _candidate_entropy(item: dict[str, Any]) -> float:
    posterior = item.get("tail_probability", item.get("p_value"))
    if posterior is None:
        return 0.0
    return information_gain.posterior_entropy(posterior)


@dataclass(frozen=True)
class ScheduledObservation:
    """One slot in a night's sequence."""

    candidate_id: str
    ra_deg: float
    dec_deg: float
    start_utc: str
    end_utc: str
    exposure_hours: float
    entropy_bits: float
    slew_deg_from_previous: float | None

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id, "ra_deg": self.ra_deg, "dec_deg": self.dec_deg,
            "start_utc": self.start_utc, "end_utc": self.end_utc,
            "exposure_hours": self.exposure_hours, "entropy_bits": round(self.entropy_bits, 6),
            "slew_deg_from_previous": (None if self.slew_deg_from_previous is None
                                       else round(self.slew_deg_from_previous, 4)),
        }


@dataclass
class NightSchedule:
    start_utc: str
    duration_hours: float
    exposure_hours: float
    observations: list[ScheduledObservation] = field(default_factory=list)
    unscheduled_candidate_ids: list[str] = field(default_factory=list)
    # Count of local-search slew-reducing swaps that were REJECTED because
    # they would have placed a candidate outside its own true visibility
    # window -- a measured count, not an assumption the swap pass is always
    # safe (see `_local_search_reduce_slew`'s docstring).
    window_violations_avoided: int = 0

    @property
    def total_exposure_hours(self) -> float:
        return len(self.observations) * self.exposure_hours

    @property
    def total_slew_deg(self) -> float:
        return sum(o.slew_deg_from_previous or 0.0 for o in self.observations)

    @property
    def total_entropy_captured_bits(self) -> float:
        return sum(o.entropy_bits for o in self.observations)

    def to_dict(self) -> dict:
        return {
            "start_utc": self.start_utc, "duration_hours": self.duration_hours,
            "exposure_hours": self.exposure_hours,
            "observations": [o.to_dict() for o in self.observations],
            "unscheduled_candidate_ids": list(self.unscheduled_candidate_ids),
            "total_exposure_hours": self.total_exposure_hours,
            "total_slew_deg": round(self.total_slew_deg, 4),
            "total_entropy_captured_bits": round(self.total_entropy_captured_bits, 6),
            "window_violations_avoided": self.window_violations_avoided,
        }


def _windows_as_datetimes(plan_result: dict[str, Any]) -> list[tuple[datetime, datetime]]:
    return [(_parse_utc(w["start_utc"]), _parse_utc(w["end_utc"]))
           for w in plan_result["windows"]]


def _free_gaps(window: tuple[datetime, datetime], busy: list[tuple[datetime, datetime]]
              ) -> list[tuple[datetime, datetime]]:
    """`window` minus every overlapping interval in `busy` (already sorted
    by start), as the ordered list of free sub-intervals within it."""
    start, end = window
    gaps: list[tuple[datetime, datetime]] = []
    cursor = start
    for busy_start, busy_end in busy:
        if busy_end <= cursor or busy_start >= end:
            continue
        if busy_start > cursor:
            gaps.append((cursor, min(busy_start, end)))
        cursor = max(cursor, busy_end)
        if cursor >= end:
            break
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def _slew_deg(a: dict[str, Any], b: dict[str, Any]) -> float:
    return followup.angular_separation_deg(a["ra_deg"], a["dec_deg"], b["ra_deg"], b["dec_deg"])


def build_night_schedule(candidates: list[dict[str, Any]], *, start_utc: str,
                         duration_hours: float = 12.0, exposure_hours: float = DEFAULT_EXPOSURE_HOURS,
                         local_search_passes: int = 3,
                         followup_kwargs: dict[str, Any] | None = None,
                         priority_fn: Any = None) -> NightSchedule:
    """Greedy-insertion sequencer with a slew-reducing local-search pass.

    Each `candidates` item needs `candidate_id`, `ra_deg`, `dec_deg`, and a
    posterior probability under `tail_probability` or `p_value` (the same
    contract `information_gain.rank_by_information_gain` uses). Priority is
    entropy-per-`exposure_hours` by default, computed identically to
    `rank_by_information_gain`'s own `value_per_hour`; candidates are
    attempted for insertion highest-priority first, each placed in the
    EARLIEST feasible free gap across its own `followup.plan` windows. A
    candidate with no feasible gap is reported in
    `unscheduled_candidate_ids`, never dropped silently.

    `priority_fn`, if given, replaces the default `entropy/exposure_hours`
    ranking with `priority_fn(item) -> float` -- everything else (windows,
    packing, local search) stays identical. This exists so `schedule_eval.
    py` can compare the information-gain policy against alternative
    observing-priority policies through the exact same sequencer, isolating
    the comparison to priority order alone rather than confounding it with
    a second, differently-implemented scheduler.
    """
    if exposure_hours <= 0:
        raise ValueError(f"exposure_hours must be positive, got {exposure_hours}")
    night_start = _parse_utc(start_utc)
    night_end = night_start + timedelta(hours=float(duration_hours))
    kwargs = dict(followup_kwargs or {})
    kwargs.setdefault("duration_hours", duration_hours)

    scored = []
    for item in candidates:
        entropy = _candidate_entropy(item)
        priority = priority_fn(item) if priority_fn is not None else entropy / exposure_hours
        plan_result = followup.plan(ra_deg=item["ra_deg"], dec_deg=item["dec_deg"],
                                    start_utc=start_utc, **kwargs)
        scored.append((priority, entropy, item, plan_result))
    scored.sort(key=lambda row: row[0], reverse=True)

    # Each candidate's own clamped-to-night feasibility windows, keyed by
    # id, so the local-search swap pass below can verify a swap keeps both
    # candidates inside their TRUE visibility windows rather than only
    # their slot's already-scheduled time (see `_local_search_reduce_slew`).
    windows_by_candidate_id: dict[str, list[tuple[datetime, datetime]]] = {}
    for _, _, item, plan_result in scored:
        clamped = []
        for window in _windows_as_datetimes(plan_result):
            window = (max(window[0], night_start), min(window[1], night_end))
            if window[0] < window[1]:
                clamped.append(window)
        windows_by_candidate_id[str(item["candidate_id"])] = clamped

    timeline: list[ScheduledObservation] = []  # kept sorted by start_utc
    unscheduled: list[str] = []
    exposure_delta = timedelta(hours=float(exposure_hours))

    for priority, entropy, item, plan_result in scored:
        if not plan_result["visible"]:
            unscheduled.append(str(item["candidate_id"]))
            continue
        busy = [(_parse_utc(o.start_utc), _parse_utc(o.end_utc)) for o in timeline]
        placed = False
        for window in _windows_as_datetimes(plan_result):
            window = (max(window[0], night_start), min(window[1], night_end))
            if window[0] >= window[1]:
                continue
            for gap_start, gap_end in _free_gaps(window, busy):
                if gap_end - gap_start < exposure_delta:
                    continue
                start = gap_start
                end = start + exposure_delta
                slew = None
                predecessors = [o for o in timeline if _parse_utc(o.end_utc) <= start]
                if predecessors:
                    previous = max(predecessors, key=lambda o: o.end_utc)
                    slew = _slew_deg(item, {"ra_deg": previous.ra_deg, "dec_deg": previous.dec_deg})
                observation = ScheduledObservation(
                    candidate_id=str(item["candidate_id"]), ra_deg=float(item["ra_deg"]),
                    dec_deg=float(item["dec_deg"]), start_utc=start.isoformat(),
                    end_utc=end.isoformat(), exposure_hours=float(exposure_hours),
                    entropy_bits=entropy, slew_deg_from_previous=slew)
                timeline.append(observation)
                timeline.sort(key=lambda o: o.start_utc)
                placed = True
                break
            if placed:
                break
        if not placed:
            unscheduled.append(str(item["candidate_id"]))

    window_violations = _local_search_reduce_slew(
        timeline, passes=local_search_passes, windows_by_candidate_id=windows_by_candidate_id)
    _recompute_slew(timeline)

    return NightSchedule(start_utc=night_start.isoformat(), duration_hours=float(duration_hours),
                         exposure_hours=float(exposure_hours), observations=timeline,
                         unscheduled_candidate_ids=unscheduled,
                         window_violations_avoided=window_violations)


def _total_slew(timeline: list[ScheduledObservation]) -> float:
    total = 0.0
    for previous, current in zip(timeline, timeline[1:]):
        total += _slew_deg(
            {"ra_deg": previous.ra_deg, "dec_deg": previous.dec_deg},
            {"ra_deg": current.ra_deg, "dec_deg": current.dec_deg})
    return total


def _within_any_window(start: datetime, end: datetime,
                       windows: list[tuple[datetime, datetime]]) -> bool:
    return any(w_start <= start and end <= w_end for w_start, w_end in windows)


def _local_search_reduce_slew(timeline: list[ScheduledObservation], *, passes: int,
                              windows_by_candidate_id: dict[str, list[tuple[datetime, datetime]]]
                              | None = None) -> int:
    """Adjacent-pair slot swaps that reduce total slew distance.

    A swap exchanges which candidate occupies which already-scheduled slot
    (same slot length for every observation), so it is well-defined
    without touching `followup.plan` -- but a candidate's own feasibility
    window can differ from the slot's, so a slew-reducing swap is only
    ACCEPTED when both candidates' new slot times fall within their own
    true visibility windows (`windows_by_candidate_id`, built by
    `build_night_schedule` from each candidate's own clamped `followup.
    plan` windows). A swap that would violate either candidate's window is
    rejected and counted, not silently applied -- `window_violations_
    avoided` on `NightSchedule` reports the count rather than assuming a
    swap is always safe (previously an acknowledged, unmeasured gap).
    When `windows_by_candidate_id` is omitted, the check is skipped
    entirely (no windows to check against), preserving prior behavior for
    any direct caller that does not have per-candidate windows to pass.
    """
    violations_avoided = 0
    if len(timeline) < 3:
        return violations_avoided
    for _ in range(max(0, passes)):
        improved = False
        for index in range(len(timeline) - 1):
            first, second = timeline[index], timeline[index + 1]
            before = _total_slew(timeline)
            swapped = list(timeline)
            swapped[index], swapped[index + 1] = _swap_positions(first, second)
            after = _total_slew(swapped)
            if not (after < before - 1e-9):
                continue
            new_first, new_second = swapped[index], swapped[index + 1]
            if windows_by_candidate_id is not None:
                first_windows = windows_by_candidate_id.get(new_first.candidate_id, [])
                second_windows = windows_by_candidate_id.get(new_second.candidate_id, [])
                first_ok = _within_any_window(_parse_utc(new_first.start_utc),
                                              _parse_utc(new_first.end_utc), first_windows)
                second_ok = _within_any_window(_parse_utc(new_second.start_utc),
                                               _parse_utc(new_second.end_utc), second_windows)
                if not (first_ok and second_ok):
                    violations_avoided += 1
                    continue
            timeline[index], timeline[index + 1] = new_first, new_second
            improved = True
        if not improved:
            break
    return violations_avoided


def _swap_positions(first: ScheduledObservation, second: ScheduledObservation
                    ) -> tuple[ScheduledObservation, ScheduledObservation]:
    """Exchange which candidate occupies `first`'s slot and which occupies
    `second`'s -- times/exposure stay with the SLOT, identity/coordinates/
    entropy move with the CANDIDATE."""
    new_first = ScheduledObservation(
        candidate_id=second.candidate_id, ra_deg=second.ra_deg, dec_deg=second.dec_deg,
        start_utc=first.start_utc, end_utc=first.end_utc, exposure_hours=first.exposure_hours,
        entropy_bits=second.entropy_bits, slew_deg_from_previous=None)
    new_second = ScheduledObservation(
        candidate_id=first.candidate_id, ra_deg=first.ra_deg, dec_deg=first.dec_deg,
        start_utc=second.start_utc, end_utc=second.end_utc, exposure_hours=second.exposure_hours,
        entropy_bits=first.entropy_bits, slew_deg_from_previous=None)
    return new_first, new_second


def _recompute_slew(timeline: list[ScheduledObservation]) -> None:
    """Final pass: `slew_deg_from_previous` reflects each observation's
    ACTUAL chronological predecessor after local search, so the reported
    number is always accurate even though it was advisory during
    insertion (see the module docstring's stated scope)."""
    for index, observation in enumerate(timeline):
        if index == 0:
            slew = None
        else:
            previous = timeline[index - 1]
            slew = _slew_deg({"ra_deg": previous.ra_deg, "dec_deg": previous.dec_deg},
                             {"ra_deg": observation.ra_deg, "dec_deg": observation.dec_deg})
        timeline[index] = ScheduledObservation(
            candidate_id=observation.candidate_id, ra_deg=observation.ra_deg,
            dec_deg=observation.dec_deg, start_utc=observation.start_utc,
            end_utc=observation.end_utc, exposure_hours=observation.exposure_hours,
            entropy_bits=observation.entropy_bits, slew_deg_from_previous=slew)


def replan(schedule: NightSchedule, *, executed_candidate_ids: list[str],
          remaining_candidates: list[dict[str, Any]], from_utc: str,
          exposure_hours: float | None = None, local_search_passes: int = 3,
          followup_kwargs: dict[str, Any] | None = None) -> NightSchedule:
    """Re-solve the REST of the night from `from_utc` onward: a mid-night
    trigger (weather change, a new alert) invalidates the remaining plan,
    but every observation already executed stays exactly as recorded --
    this never rewrites history, only what has not happened yet.

    `remaining_candidates` should be the caller's current candidate set for
    the rest of the night (typically the original set minus anything
    already executed, plus any newly arrived candidate from `alerts.py`).
    """
    executed = {str(cid) for cid in executed_candidate_ids}
    kept = [o for o in schedule.observations if o.candidate_id in executed]
    cutoff = _parse_utc(from_utc)
    remaining_hours = max(0.0, (_parse_utc(schedule.start_utc)
                                + timedelta(hours=schedule.duration_hours) - cutoff)
                          .total_seconds() / 3600.0)

    fresh = build_night_schedule(
        remaining_candidates, start_utc=from_utc, duration_hours=remaining_hours,
        exposure_hours=exposure_hours or schedule.exposure_hours,
        local_search_passes=local_search_passes, followup_kwargs=followup_kwargs)

    combined = sorted(kept + fresh.observations, key=lambda o: o.start_utc)
    return NightSchedule(start_utc=schedule.start_utc, duration_hours=schedule.duration_hours,
                         exposure_hours=schedule.exposure_hours, observations=combined,
                         unscheduled_candidate_ids=fresh.unscheduled_candidate_ids)
