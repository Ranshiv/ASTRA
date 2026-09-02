"""Cross-survey corroboration for discard-pile events (Direction 2, step 2
of the research plan adopted 2026-08-29).

`discard_pile.py` finds coherent runs of epochs one survey's own quality
flags discarded. A coherent run is not yet evidence of anything real: a
satellite trail or a persistent hot pixel can also produce several
consecutive flagged epochs that move together. The discriminator this
module adds is exactly ASTRA's founding thesis (see `README.md`): a
survey-local effect has no reason to appear in an *independent* survey's own
photometry over the same real time window; a genuine astrophysical event
does.

This module deliberately does not fetch data itself -- `acquire.py` and the
survey connectors already own that, and `crossmatch.group_sources` already
owns locating an object's counterparts across surveys. This module is a pure
function over already-fetched `LightCurve`s from those counterparts, the
same layering `scoring.py` and `crossmatch.py` use.

The output is intentionally NOT injected into `scoring.WEIGHTS`/`combine()`:
those drive every existing candidate's score, and this is a new, separate
signal under active development. `CorroborationResult` mirrors
`scoring.ScoreBreakdown`'s shape (components, a summary total, reasons) so
it reads the same way once it is promoted, without touching the weights
every other candidate is already scored and compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import timeframe
from .discard_pile import DiscardRecord
from .surveys.base import LightCurve

# An in-window deviation must exceed this many combined-error sigma to count
# as support -- matching the spirit of `scoring.py`'s significance-driven
# components rather than a bare "any change at all" threshold.
DEFAULT_SIGNIFICANCE_SIGMA = 3.0

# Widens the window slightly for a counterpart survey's own, independently
# sampled cadence -- an event straddling the discard record's exact edges in
# one survey should not be missed by one epoch in another survey's coarser
# sampling.
DEFAULT_PAD_DAYS = 1.0


@dataclass(frozen=True)
class SurveySupport:
    """One counterpart survey's evidence for or against one discard record."""

    survey: str
    band: str
    in_window_points: int
    z_score: float | None
    supports: bool

    def to_dict(self) -> dict:
        return {
            "survey": self.survey,
            "band": self.band,
            "in_window_points": self.in_window_points,
            "z_score": (None if self.z_score is None else round(self.z_score, 3)),
            "supports": self.supports,
        }


@dataclass(frozen=True)
class CorroborationResult:
    """Whether independent surveys corroborate one discard-pile event."""

    record: DiscardRecord
    components: tuple[SurveySupport, ...]
    corroborated: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "record": self.record.to_dict(),
            "components": [c.to_dict() for c in self.components],
            "corroborated": self.corroborated,
            "supporting_surveys": [c.survey for c in self.components if c.supports],
            "reasons": self.reasons,
        }


def _window_deviation(curve: LightCurve, time_start: float, time_end: float,
                      pad_days: float) -> tuple[int, float | None]:
    """In-window point count and a z-score of the in-window mean against the
    curve's own out-of-window baseline.

    Uses whichever points fall outside the (padded) window as the baseline;
    if too few remain outside it (a short curve, or a window spanning nearly
    all of it), the curve's own overall error-weighted mean stands in
    instead -- a coarser baseline, not a missing one.
    """
    in_window = (curve.time >= time_start - pad_days) & (curve.time <= time_end + pad_days)
    n_in = int(np.sum(in_window))
    if n_in == 0:
        return 0, None

    err = np.clip(curve.value_err, 1e-12, None)
    outside = ~in_window
    if np.sum(outside) >= 2:
        weights = 1.0 / err[outside] ** 2
        baseline = float(np.sum(weights * curve.value[outside]) / np.sum(weights))
        baseline_err = float(np.sqrt(1.0 / np.sum(weights)))
    elif len(curve.value) >= 2:
        weights = 1.0 / err ** 2
        baseline = float(np.sum(weights * curve.value) / np.sum(weights))
        baseline_err = float(np.sqrt(1.0 / np.sum(weights)))
    else:
        return n_in, None

    in_weights = 1.0 / err[in_window] ** 2
    in_mean = float(np.sum(in_weights * curve.value[in_window]) / np.sum(in_weights))
    in_mean_err = float(np.sqrt(1.0 / np.sum(in_weights)))

    combined_err = float(np.sqrt(baseline_err ** 2 + in_mean_err ** 2))
    if combined_err <= 0:
        return n_in, None
    return n_in, float((in_mean - baseline) / combined_err)


def corroborate(record: DiscardRecord, other_curves: list[LightCurve], *,
                record_curve: LightCurve | None = None,
                significance_sigma: float = DEFAULT_SIGNIFICANCE_SIGMA,
                pad_days: float = DEFAULT_PAD_DAYS,
                min_supporting_surveys: int = 1) -> CorroborationResult:
    """Test one discard record against independent surveys' own photometry.

    `other_curves` must already be restricted to counterparts of the same
    physical object -- typically via `crossmatch.group_sources`, excluding
    `record.survey` itself, since a survey cannot corroborate its own
    discarded epochs. Curves from `record.survey` passed in by mistake are
    not filtered out defensively here: the caller owns that grouping
    decision, matching `crossmatch.group_sources`'s own "the caller resolves
    the anchor" discipline.

    Time-system reconciliation: `record.time_start`/`time_end` are in
    whatever time system the survey `record.survey` was extracted from
    natively uses (e.g. ZTF's HJD_UTC), while `other_curves` may be in a
    DIFFERENT native time system (e.g. TESS's BJD_TDB) -- comparing them
    directly, as this function previously did, silently mixes two clocks
    that disagree by up to about a minute (`timeframe.py`'s own
    TDB-vs-UTC + heliocentric-vs-barycentric terms), not epsilon. Passing
    `record_curve` -- the curve `record` was itself extracted from, so its
    `time_system`/`source.ra_deg`/`source.dec_deg` are known -- converts
    both `record`'s window boundaries and every `other_curves` entry to the
    common `timeframe.TARGET_SYSTEM` (BJD_TDB) via `timeframe.to_bjd_tdb`/
    `timeframe.align` before comparison, reusing that module's conversion
    unchanged rather than a second implementation. `record_curve` is
    optional (default `None`, preserving the prior unconverted comparison)
    because not every caller has retained the originating curve -- but any
    caller that HAS it should pass it; the window-day-scale `pad_days`
    tolerance usually absorbs the sub-minute offset, so this was rarely
    visible as a wrong verdict, but "usually absorbed by padding" is not
    the same claim as "reconciled."
    """
    aligned_curves = other_curves
    time_start, time_end = record.time_start, record.time_end
    if record_curve is not None:
        aligned_curves = [timeframe.align(curve) for curve in other_curves]
        converted_bounds = timeframe.to_bjd_tdb(
            np.array([record.time_start, record.time_end]), record_curve.time_system,
            record_curve.source.ra_deg, record_curve.source.dec_deg,
            record_curve.source.survey)
        time_start, time_end = float(converted_bounds[0]), float(converted_bounds[1])

    components: list[SurveySupport] = []
    for curve in aligned_curves:
        n_in, z_score = _window_deviation(curve, time_start, time_end, pad_days)
        supports = bool(z_score is not None and abs(z_score) >= significance_sigma)
        components.append(SurveySupport(
            survey=curve.source.survey, band=curve.band,
            in_window_points=n_in, z_score=z_score, supports=supports,
        ))

    supporting = [c.survey for c in components if c.supports]
    corroborated = len(set(supporting)) >= min_supporting_surveys

    reasons = []
    if corroborated:
        reasons.append(
            f"independent evidence from {sorted(set(supporting))} "
            f"during the discarded epoch window"
        )
    elif not components:
        reasons.append("no independent-survey coverage in this time window")
    else:
        reasons.append("no independent survey shows a coincident deviation")

    return CorroborationResult(
        record=record, components=tuple(components),
        corroborated=corroborated, reasons=reasons,
    )
