"""Domain-agnostic association and agreement scoring.

Generalises `crossmatch.py`'s `match_catalogs`/`group_sources`/
`_resolve_anchor`/`_flag_blends` and `scoring.py`'s `combine`/
`ScoreBreakdown`, with every sky-coordinate-specific piece replaced by an
injected function or parameter:

- `crossmatch.angular_separation_arcsec` -> an injected `DistanceFn`, over
  an abstract `position: tuple[float, ...]` that need not be (ra, dec) --
  see `gw_adapter.py`, where it is a 1-tuple of time.
- `crossmatch.epoch_corrected`/`propagate_position` (proper motion) stay OUT
  of this module entirely: motion correction is astronomy-specific and
  belongs in `astronomy_adapter.py`, which corrects positions BEFORE
  building `InstrumentRecord`s, exactly as `match_catalogs` corrects
  positions before computing separations.
- `crossmatch.PIXEL_SCALE_ARCSEC`/`COARSE_BEAM_ARCSEC` (beam-width
  blending) also stay OUT: "this instrument's resolution is too coarse to
  isolate one source" is a real, but domain-specific, refinement layered on
  top of the domain-general "one counterpart claimed by several groups"
  blend rule this module DOES generalise.
- `scoring.WEIGHTS` (a fixed module-level dict) becomes a `weights`
  parameter to `combine_components`, so the same weighted-mean-over-
  available-components logic serves any domain's component set, not just
  ASTRA's five astronomy score components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeVar

T = TypeVar("T")

DistanceFn = Callable[[tuple[float, ...], tuple[float, ...]], float]


@dataclass(frozen=True)
class InstrumentRecord:
    """One detection from one instrument, in a domain-agnostic shape.

    `position` is whatever metric space this domain's `DistanceFn` operates
    over -- (ra_deg, dec_deg) for astronomy, (time_seconds,) for the GW
    adapter's coincidence matching. `quality_flags` is an opaque per-record
    integer (or 0), left for the adapter to interpret; this module never
    reads it -- flag-driven filtering (`ztf_artifact_patches.
    categorize_catflags`'s astronomy equivalent) is upstream of building a
    record, not this module's job.
    """

    instrument: str
    identifier: str
    position: tuple[float, ...]
    measurement: float | None = None
    quality_flags: int = 0
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MatchResult:
    """One record's best within-radius counterpart in another instrument's
    catalogue -- generalises `crossmatch.Match`."""

    record: InstrumentRecord
    counterpart: InstrumentRecord
    distance: float
    competitors: int  # other counterparts within the radius, besides the best

    def to_dict(self) -> dict:
        return {
            "instrument": self.record.instrument, "identifier": self.record.identifier,
            "counterpart_instrument": self.counterpart.instrument,
            "counterpart_identifier": self.counterpart.identifier,
            "distance": round(self.distance, 6), "competitors": self.competitors,
        }


@dataclass
class Group:
    """One physical event/object as seen by several instruments --
    generalises `crossmatch.MatchGroup`."""

    members: dict[str, InstrumentRecord] = field(default_factory=dict)
    distances: dict[str, float] = field(default_factory=dict)
    ambiguous: set[str] = field(default_factory=set)
    blended: set[str] = field(default_factory=set)
    match_radius: float = 0.0

    @property
    def instruments(self) -> list[str]:
        return sorted(self.members)

    @property
    def resolved_instruments(self) -> int:
        return len([name for name in self.members if name not in self.blended])

    def to_dict(self) -> dict:
        return {
            "instruments": self.instruments,
            "independent_instruments": len(self.members),
            "resolved_instruments": self.resolved_instruments,
            "members": {name: record.identifier for name, record in self.members.items()},
            "distances": {name: round(value, 6) for name, value in self.distances.items()},
            "ambiguous": sorted(self.ambiguous), "blended": sorted(self.blended),
            "match_radius": self.match_radius,
        }


def match_records(records: list[InstrumentRecord], counterparts: list[InstrumentRecord],
                  distance_fn: DistanceFn, radius: float) -> list[MatchResult]:
    """Match each record to its nearest counterpart within `radius`, under
    `distance_fn`. Generalises `crossmatch.match_catalogs`; no vectorised
    fast path -- this is the general-purpose reference algorithm, not the
    already-optimised astronomy hot path (`crossmatch.
    _angular_separation_grid_arcsec` stays astronomy's own, faster,
    numerically-identical-by-construction implementation).
    """
    matches: list[MatchResult] = []
    for record in records:
        distances = [(distance_fn(record.position, counterpart.position), counterpart)
                    for counterpart in counterparts]
        within = [pair for pair in distances if pair[0] <= radius]
        if not within:
            continue
        best_distance, best_counterpart = min(within, key=lambda pair: pair[0])
        matches.append(MatchResult(
            record=record, counterpart=best_counterpart, distance=best_distance,
            competitors=len(within) - 1))
    return matches


def _resolve_anchor(by_instrument: dict[str, list[InstrumentRecord]],
                    anchor: str | None = None) -> tuple[str | None, str]:
    """Generalises `crossmatch._resolve_anchor`: an explicit anchor if
    named, else the instrument with the most records, lexically tie-broken
    for reproducibility regardless of input dict ordering."""
    if not by_instrument:
        return None, "empty"
    if anchor is not None and str(anchor).strip():
        requested = str(anchor).strip()
        matching = next((name for name in by_instrument
                        if name.casefold() == requested.casefold()), None)
        if matching is None:
            available = ", ".join(sorted(str(name) for name in by_instrument)) or "none"
            raise ValueError(f"anchor instrument {requested!r} is not available; "
                             f"choose one of: {available}")
        if not by_instrument[matching]:
            raise ValueError(f"anchor instrument {matching!r} has no records to anchor grouping")
        return matching, "explicit"
    anchor_name = min(by_instrument,
                      key=lambda name: (-len(by_instrument[name]), str(name).casefold(), str(name)))
    return anchor_name, "largest_catalogue"


def _flag_shared_counterparts(groups: list[Group], anchor: str) -> None:
    """Mark a counterpart claimed by more than one group as blended for
    that instrument: a definitive structural failure to resolve, the one
    part of `crossmatch._flag_blends` that has nothing to do with sky
    coordinates or beam width."""
    usage: dict[tuple[str, str], int] = {}
    for group in groups:
        for instrument, record in group.members.items():
            if instrument == anchor:
                continue
            key = (instrument, record.identifier)
            usage[key] = usage.get(key, 0) + 1

    for group in groups:
        for instrument, record in group.members.items():
            if instrument == anchor:
                continue
            if usage.get((instrument, record.identifier), 0) > 1:
                group.blended.add(instrument)


def group_records(by_instrument: dict[str, list[InstrumentRecord]], distance_fn: DistanceFn,
                  radius: float, anchor: str | None = None) -> list[Group]:
    """Cluster records from several instruments into per-event/object
    groups. Generalises `crossmatch.group_sources`.

    The instrument with the most records anchors the grouping by default
    (matching `crossmatch.group_sources`'s own "the largest catalogue
    defines the object list" default); every other instrument's records
    attach to it within `radius`. A group with only one instrument is still
    returned -- "seen by only one instrument" is itself the negative case
    corroboration exists to distinguish from the positive one.
    """
    if not by_instrument:
        return []
    resolved_anchor, _ = _resolve_anchor(by_instrument, anchor)
    if resolved_anchor is None:
        return []
    anchor_records = by_instrument[resolved_anchor]

    groups: list[Group] = []
    for record in anchor_records:
        group = Group(members={resolved_anchor: record}, distances={resolved_anchor: 0.0},
                      match_radius=radius)
        for instrument, candidates in by_instrument.items():
            if instrument == resolved_anchor:
                continue
            found = match_records([record], candidates, distance_fn, radius)
            if not found:
                continue
            match = found[0]
            group.members[instrument] = match.counterpart
            group.distances[instrument] = match.distance
            if match.competitors > 0:
                group.ambiguous.add(instrument)
        groups.append(group)

    _flag_shared_counterparts(groups, resolved_anchor)
    return groups


@dataclass
class AgreementBreakdown:
    """Every component of one event's agreement score, with the total --
    generalises `scoring.ScoreBreakdown`, parameterised by `weights`
    instead of reading the fixed module-level `scoring.WEIGHTS`."""

    components: dict[str, float | None] = field(default_factory=dict)
    total: float = 0.0
    weight_used: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"total": round(self.total, 4), "weight_used": round(self.weight_used, 4),
                "components": {k: (None if v is None else round(v, 4))
                              for k, v in self.components.items()},
                "reasons": self.reasons}


def combine_components(components: dict[str, float | None], weights: dict[str, float],
                       reasons: list[str] | None = None) -> AgreementBreakdown:
    """Weighted mean over the components that could actually be computed.
    Generalises `scoring.combine`: identical algorithm, `weights` supplied
    by the caller rather than a fixed module constant, so the same function
    serves any domain's component set."""
    usable = {name: value for name, value in components.items()
             if value is not None and value == value and name in weights}  # NaN != NaN
    weight_used = sum(weights[name] for name in usable)
    total = (sum(weights[name] * usable[name] for name in usable) / weight_used
            if weight_used > 0 else 0.0)
    return AgreementBreakdown(
        components={name: components.get(name) for name in weights},
        total=float(total), weight_used=float(weight_used), reasons=list(reasons or []))
