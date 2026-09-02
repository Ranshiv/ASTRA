"""Gravitational-wave-detector-style coincidence vetting, on clearly-
labelled SYNTHETIC data (Direction 3's second domain).

Real GWOSC/Gravity Spy ingestion is explicitly NOT attempted here -- the
same honest scoping `ztf_forced_photometry.py`'s own docstring already uses
for real ZTF difference-image acquisition ("genuinely open, not attempted
here"). What this module demonstrates is structural, not a real-data
result: two independent detectors, each producing its own local
"triggers" -- some real (a shared astrophysical signal both detectors see
within a short time window), most false (instrument-local glitches) -- is
exactly the shape `corroborate.core`'s abstract association/scoring was
generalised for, with time standing in for sky position and a coincidence
window standing in for an angular match radius.

The one deliberately controllable, scientifically interesting parameter is
`systematics_correlation`: the fraction of one detector's glitches that
have a coincident glitch in the OTHER detector (a shared environmental
noise source -- e.g. a mains-power transient both detectors' auxiliary
channels see). At `systematics_correlation=0` the two detectors' false
triggers are independent, and time-coincidence alone should cleanly
separate real events from glitches. As it rises toward 1, coincidence
vetting degrades toward useless, because the very thing that made
"two instruments agree" strong evidence -- INDEPENDENT systematics -- stops
being true. This is the parameter `corroborate.eval.
evaluate_scaling_with_systematics_correlation` sweeps for the scaling
claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import core

DEFAULT_COINCIDENCE_WINDOW_SECONDS = 0.05
DETECTOR_A, DETECTOR_B = "detector_a", "detector_b"


def _distance_fn(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return abs(a[0] - b[0])


@dataclass(frozen=True)
class SyntheticGWPopulation:
    """One synthetic two-detector run, with known ground truth per record."""

    by_instrument: dict[str, list[core.InstrumentRecord]]
    truth: dict[str, bool]  # identifier -> True iff this trigger is a real event


def generate_synthetic_detector_pair(
    *, n_real_events: int = 30, n_glitches_a: int = 100, n_glitches_b: int = 100,
    systematics_correlation: float = 0.0,
    window_seconds: float = DEFAULT_COINCIDENCE_WINDOW_SECONDS,
    real_event_jitter_seconds: float = 0.005, duration_seconds: float = 10_000.0,
    seed: int = 0,
) -> SyntheticGWPopulation:
    """A synthetic two-detector trigger population.

    Real events: one shared true time, each detector's own trigger jittered
    by up to `real_event_jitter_seconds` (real physical arrival-time/
    timing-noise scatter, well inside `window_seconds`).
    Glitches: independent random times in each detector, EXCEPT a
    `systematics_correlation` fraction of detector A's glitches get a
    coincident detector-B glitch injected at (roughly) the same time --
    the shared-systematic failure mode coincidence vetting cannot see.
    """
    rng = np.random.default_rng(seed)
    records: dict[str, list[core.InstrumentRecord]] = {DETECTOR_A: [], DETECTOR_B: []}
    truth: dict[str, bool] = {}
    counter = 0

    def _add(instrument: str, time_seconds: float, is_real: bool) -> None:
        nonlocal counter
        identifier = f"{instrument}_{counter}"
        counter += 1
        records[instrument].append(core.InstrumentRecord(
            instrument=instrument, identifier=identifier, position=(float(time_seconds),)))
        truth[identifier] = is_real

    for _ in range(n_real_events):
        true_time = float(rng.uniform(0, duration_seconds))
        _add(DETECTOR_A, true_time + rng.normal(0, real_event_jitter_seconds), True)
        _add(DETECTOR_B, true_time + rng.normal(0, real_event_jitter_seconds), True)

    n_correlated = int(round(n_glitches_a * systematics_correlation))
    for index in range(n_glitches_a):
        glitch_time = float(rng.uniform(0, duration_seconds))
        _add(DETECTOR_A, glitch_time, False)
        if index < n_correlated:
            _add(DETECTOR_B, glitch_time + rng.normal(0, window_seconds / 3), False)

    remaining_b = max(0, n_glitches_b - n_correlated)
    for _ in range(remaining_b):
        _add(DETECTOR_B, float(rng.uniform(0, duration_seconds)), False)

    return SyntheticGWPopulation(by_instrument=records, truth=truth)


def group_population(population: SyntheticGWPopulation, *,
                     window_seconds: float = DEFAULT_COINCIDENCE_WINDOW_SECONDS,
                     anchor: str = DETECTOR_A) -> list[core.Group]:
    return core.group_records(population.by_instrument, _distance_fn, window_seconds, anchor)
