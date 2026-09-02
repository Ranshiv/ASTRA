"""Discard-pile extraction: real epochs a survey's own quality flags strip
before a candidate is ever assembled (Direction 2 of the research plan
adopted 2026-08-29: "anomalies in the discard pile").

`surveys.ztf.ZTFConnector.fetch_light_curves_with_quality` already recovers
every real epoch IRSA's own `DEFAULT_CATFLAGS_MASK` filter strips, paired
1:1 with the real per-epoch `catflags` word (see that method's docstring).
`ztf_artifact_patches.py` is the first consumer of that pairing, and uses it
to train an artifact classifier on short windows. This module is the second
consumer, and asks a different question of the same real data: does a run of
discarded epochs look like a coherent astrophysical event -- a burst, a
fade, a sustained excursion -- rather than the isolated single-epoch scatter
a bad pixel or cosmic ray produces?

A single discarded epoch is never itself evidence. `min_run_length`
consecutive epochs in the same flag category is the bar, mirroring
`ztf_artifact_patches.MIN_RUN_LENGTH`'s own discipline for the same catflags
data. Whether a coherent run is real or a genuine instrumental effect that
merely persists across several epochs (e.g. a satellite trail) is NOT
decided here -- that is `crossmatch`/`scoring`'s job (cross-survey
corroboration) and `ztf_forced_photometry`'s job (independent pixel-level
flux), both downstream of this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surveys.base import LightCurve
from .ztf_artifact_patches import MIN_RUN_LENGTH, categorize_catflags

DEFAULT_MIN_RUN_LENGTH = MIN_RUN_LENGTH  # 3, the same bar artifact patches use


@dataclass(frozen=True)
class DiscardRecord:
    """One coherent run of discarded epochs, for one (object, band) curve."""

    object_id: str
    survey: str
    band: str
    flag_category: str
    epoch_count: int
    time_start: float
    time_end: float
    magnitude_offset: float
    max_step: float
    coherent: bool

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "survey": self.survey,
            "band": self.band,
            "flag_category": self.flag_category,
            "epoch_count": self.epoch_count,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "magnitude_offset": round(self.magnitude_offset, 4),
            "max_step": round(self.max_step, 4),
            "coherent": self.coherent,
        }


def _weighted_baseline(value: np.ndarray, value_err: np.ndarray) -> float:
    """Error-weighted mean, matching `features.variability_indices`'s own
    weighting so a discarded run is compared against the same kind of
    baseline the accepted curve's own variability statistics use."""
    err = np.clip(value_err, 1e-12, None)
    weights = 1.0 / err ** 2
    return float(np.sum(weights * value) / np.sum(weights))


def _is_coherent(run_time: np.ndarray, run_value: np.ndarray, offset: float) -> tuple[bool, float]:
    """A run is coherent when it moves together toward one excursion rather
    than jittering around the baseline it was cut from: a monotonic fade or
    a sustained brightness change has a largest single step much smaller
    than its total offset from baseline. Pure noise straddling the flag
    threshold does not have that property -- its steps are comparable in
    size to the offset itself.
    """
    max_step = float(np.max(np.abs(np.diff(run_value)))) if len(run_value) > 1 else 0.0
    coherent = bool(abs(offset) > 0 and max_step < abs(offset) * 1.5)
    return coherent, max_step


def extract_discard_records(
    curve: LightCurve, catflags: np.ndarray, *,
    min_run_length: int = DEFAULT_MIN_RUN_LENGTH,
) -> list[DiscardRecord]:
    """Coherent discarded-epoch runs for one (curve, catflags) pair.

    `curve` and `catflags` are exactly the shape
    `surveys.ztf.ZTFConnector.fetch_light_curves_with_quality` returns per
    band: every real epoch, with `catflags` aligned 1:1 to `curve.time`/
    `curve.value`. A curve with no accepted (catflags == 0) epochs at all
    has no baseline to compare a discarded run against and yields no
    records -- this deliberately never happens for real ZTF data, since
    `fetch_light_curves` (the default, filtered path) is what every other
    module already trusts to have data at all.
    """
    n = len(curve.value)
    if n == 0 or len(catflags) != n:
        return []

    order = np.argsort(curve.time)
    time = curve.time[order]
    value = curve.value[order]
    value_err = curve.value_err[order]
    flags = np.asarray(catflags)[order]

    categories = [categorize_catflags(int(word)) for word in flags]
    accepted_mask = np.array([category is None for category in categories])
    if not np.any(accepted_mask):
        return []
    baseline = _weighted_baseline(value[accepted_mask], value_err[accepted_mask])

    records: list[DiscardRecord] = []
    index = 0
    while index < n:
        category = categories[index]
        run_end = index
        while run_end < n and categories[run_end] == category:
            run_end += 1
        run_length = run_end - index

        if category is not None and run_length >= min_run_length:
            run_time = time[index:run_end]
            run_value = value[index:run_end]
            offset = float(np.mean(run_value) - baseline)
            coherent, max_step = _is_coherent(run_time, run_value, offset)
            records.append(DiscardRecord(
                object_id=curve.source.object_id,
                survey=curve.source.survey,
                band=curve.band,
                flag_category=category,
                epoch_count=run_length,
                time_start=float(run_time[0]),
                time_end=float(run_time[-1]),
                magnitude_offset=offset,
                max_step=max_step,
                coherent=coherent,
            ))
        index = run_end

    return records


def scan_source(connector, source, *,
                min_run_length: int = DEFAULT_MIN_RUN_LENGTH) -> list[DiscardRecord]:
    """Fetch one source's full (unfiltered) photometry and extract discard
    records from every band.

    `connector` must implement `fetch_light_curves_with_quality`, matching
    `surveys.ztf.ZTFConnector`'s contract; a fake is accepted for tests, the
    same injection convention `ztf_artifact_patches.fetch_and_extract` uses.
    A source whose fetch fails is skipped rather than aborting a batch scan,
    matching that same module's discipline.
    """
    try:
        pairs = connector.fetch_light_curves_with_quality(source)
    except Exception:  # noqa: BLE001 - one bad source must not abort a batch
        return []

    records: list[DiscardRecord] = []
    for curve, catflags in pairs:
        records.extend(extract_discard_records(
            curve, catflags, min_run_length=min_run_length))
    return records


def scan_sources(connector, sources, *,
                 min_run_length: int = DEFAULT_MIN_RUN_LENGTH) -> list[DiscardRecord]:
    """`scan_source` over many sources, flattened into one list."""
    records: list[DiscardRecord] = []
    for source in sources:
        records.extend(scan_source(connector, source, min_run_length=min_run_length))
    return records
