"""Astronomy, re-expressed in terms of `corroborate.core`.

Deliberately does NOT modify `crossmatch.py`/`scoring.py`: those modules
already drive every existing candidate's score, and the research plan's own
non-negotiable constraint is that the existing 690+-test suite stays green
unchanged through this work. Rewriting `group_sources`'s internals to route
through a brand-new generic package is exactly the kind of change that is
easy to get subtly wrong in ways existing tests, tuned to the current
implementation's exact behaviour, would not catch (a different ambiguous-
match tie-break, a different blend-detection order). This adapter instead
builds an INDEPENDENT path -- `crossmatch.py` untouched, still the one and
only path production code uses -- and is verified for AGREEMENT against
`crossmatch.group_sources` on the same input by
`corroborate.eval.evaluate_astronomy_equivalence`, which is a stronger
check than merely calling the same function through a new name would be.

Position correction (proper motion via `crossmatch.epoch_corrected`) is
applied HERE, before building `InstrumentRecord`s -- exactly where
`crossmatch.match_catalogs` applies it, before computing separations. The
core package never sees a raw, uncorrected position.

Beam-width blending (`crossmatch.PIXEL_SCALE_ARCSEC`/`COARSE_BEAM_ARCSEC`)
is astronomy-specific and layered on AFTER calling the generic
`core.group_records`, mirroring `crossmatch._flag_blends`'s own two-part
structure (shared-counterpart blending, which IS generalised in
`core.py`, plus beam-width blending, which is not).
"""

from __future__ import annotations

from . import core
from ..crossmatch import (COARSE_BEAM_ARCSEC, DEFAULT_RADIUS_ARCSEC, PIXEL_SCALE_ARCSEC,
                          angular_separation_arcsec, current_epoch, epoch_corrected)
from ..surveys.base import SourceRef


def _distance_fn(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return angular_separation_arcsec(a[0], a[1], b[0], b[1])


def to_instrument_record(source: SourceRef, epoch: float) -> core.InstrumentRecord:
    ra, dec, _ = epoch_corrected(source, epoch)
    return core.InstrumentRecord(
        instrument=source.survey, identifier=source.object_id, position=(ra, dec),
        extra={"source": source})


def _apply_beam_width_blending(groups: list[core.Group]) -> None:
    """The half of `crossmatch._flag_blends` that IS astronomy-specific:
    an instrument whose beam is wider than `COARSE_BEAM_ARCSEC` cannot
    isolate one star from its neighbours regardless of match uniqueness."""
    for group in groups:
        if len(group.members) <= 1:
            continue
        for instrument in group.members:
            if PIXEL_SCALE_ARCSEC.get(instrument.upper(), 1.0) >= COARSE_BEAM_ARCSEC:
                group.blended.add(instrument)


def group_sources_via_core(by_survey: dict[str, list[SourceRef]],
                           radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
                           epoch: float | None = None,
                           anchor_survey: str | None = None) -> list[core.Group]:
    """`crossmatch.group_sources`'s behaviour, produced via the
    domain-general core instead of astronomy-specific code."""
    if epoch is None:
        epoch = current_epoch()
    by_instrument = {survey: [to_instrument_record(source, epoch) for source in sources]
                     for survey, sources in by_survey.items()}
    groups = core.group_records(by_instrument, _distance_fn, radius_arcsec, anchor_survey)
    _apply_beam_width_blending(groups)
    return groups


def group_to_source_membership(group: core.Group) -> dict[str, str]:
    """`{survey: object_id}`, for comparing against `MatchGroup.to_dict()
    ["members"]` in the equivalence check."""
    return {instrument: record.identifier for instrument, record in group.members.items()}
