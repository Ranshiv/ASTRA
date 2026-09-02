"""Injection-recovery study for `discard_pile.py` (Direction 2 evaluation
harness -- research plan adopted 2026-08-29, "anomalies in the discard
pile").

Real catflags data has no "this discarded run was a real astrophysical
event, not an instrumental effect" label store -- the same gap
`research/benchmark.py`'s own docstring already states for cross-survey
anomaly injection ("known anomalies of known shape and amplitude are
injected... labels are then true by construction"). Following that already-
established approach, this harness builds synthetic (curve, catflags) pairs
with a KNOWN, injected ground truth: a coherent excursion really is injected
into some flagged runs (the shape a real fading/brightening event
produces), and independent per-epoch noise into others (the shape a cosmic
ray or a satellite-trail-like burst of bad pixels produces -- several
flagged epochs in a row with no underlying coherent signal). It then
measures how well `discard_pile`'s own discriminator recovers that known
label.

AUPRC with a bootstrap CI, grouped by (synthetic) object, reuses
`research.benchmark._bootstrap_auprc_by_object` rather than re-deriving the
same resampling logic a second time (`flare.py` importing
`transit_ttv._finite_arrays` and `microlensing_eval.py` importing
`significance._ci_binomial` are this project's precedent for reusing a
sibling module's private helper instead of duplicating it).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import discard_pile as dp
from .research.benchmark import _bootstrap_auprc_by_object
from .surveys.base import LightCurve, SourceRef

DEFAULT_N_EPOCHS = 60
# IRSA's real `DEFAULT_CATFLAGS_MASK` value, matching `ztf_artifact_patches.py`.
FLAG_WORD = 32768


def _synthetic_curve_and_catflags(rng: np.random.Generator, *, object_id: str, n_epochs: int,
                                  run_start: int, run_length: int, real: bool,
                                  amplitude: float = 0.5, noise_scale: float = 0.05
                                  ) -> tuple[LightCurve, np.ndarray]:
    """One synthetic (curve, catflags) pair with a known-truth flagged run.

    `real=True` injects a smooth half-sine bump inside the flagged run --
    single-signed and centred, exactly the "moves together toward one
    excursion" shape `discard_pile._is_coherent` is designed to recognise.
    `real=False` fills the SAME run with independent per-epoch noise of
    comparable amplitude instead: several flagged epochs in a row, but no
    underlying coherent signal.
    """
    time = np.arange(n_epochs, dtype=float)
    value = 18.0 + rng.normal(0, noise_scale, n_epochs)
    catflags = np.zeros(n_epochs, dtype=np.uint32)
    run_slice = slice(run_start, run_start + run_length)
    catflags[run_slice] = FLAG_WORD

    if real:
        phase = np.linspace(0, np.pi, run_length)
        value[run_slice] += amplitude * np.sin(phase)
    else:
        value[run_slice] += rng.normal(0, amplitude, run_length)

    source = SourceRef(survey="ZTF", object_id=object_id, ra_deg=0.0, dec_deg=0.0)
    curve = LightCurve(source=source, release="dr24", band="g", value_kind="mag",
                       time=time, value=value, value_err=np.full(n_epochs, noise_scale),
                       time_system="HJD_UTC")
    return curve, catflags


def _discriminator_score(record: dp.DiscardRecord) -> float:
    """Continuous stand-in for `coherent`'s boolean, so AUPRC has more than
    two operating points to rank against: a larger offset relative to the
    largest single step within the run is stronger evidence of one smooth
    excursion rather than several independent jumps.
    """
    if record.max_step <= 0:
        return abs(record.magnitude_offset) * 10.0  # a perfectly smooth run
    return abs(record.magnitude_offset) / record.max_step


def _generate_objects(rng: np.random.Generator, *, n_objects: int, n_epochs: int,
                      run_length: int, real_fraction: float
                      ) -> list[tuple[str, bool, LightCurve, np.ndarray]]:
    n_real = int(round(n_objects * real_fraction))
    truths = [True] * n_real + [False] * (n_objects - n_real)
    rng.shuffle(truths)

    objects = []
    for index, real in enumerate(truths):
        object_id = f"obj{index}"
        run_start = int(rng.integers(10, max(11, n_epochs - run_length - 10)))
        curve, catflags = _synthetic_curve_and_catflags(
            rng, object_id=object_id, n_epochs=n_epochs, run_start=run_start,
            run_length=run_length, real=real)
        objects.append((object_id, real, curve, catflags))
    return objects


def evaluate_injection_recovery(*, n_objects: int = 200, n_epochs: int = DEFAULT_N_EPOCHS,
                                run_length: int = 5, real_fraction: float = 0.5,
                                min_run_length: int = 3, seed: int = 0) -> dict[str, Any]:
    """AUPRC (with a bootstrap CI) for ranking real injected discard-pile
    events above artifact-shaped ones. The headline "recovery with truth"
    evaluation the research plan calls for.
    """
    rng = np.random.default_rng(seed)
    objects = _generate_objects(rng, n_objects=n_objects, n_epochs=n_epochs,
                                run_length=run_length, real_fraction=real_fraction)

    object_ids: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    n_runs_found = 0

    for object_id, real, curve, catflags in objects:
        records = dp.extract_discard_records(curve, catflags, min_run_length=min_run_length)
        object_ids.append(object_id)
        labels.append(int(real))
        if not records:
            # No run survived the min-run-length bar at all -- scored as the
            # weakest possible detection, so a discriminator that misses
            # real events is penalised by AUPRC rather than by a shrinking
            # denominator.
            scores.append(0.0)
            continue
        n_runs_found += 1
        best = max(records, key=_discriminator_score)
        scores.append(_discriminator_score(best))

    labels_array = np.array(labels)
    scores_array = np.array(scores)
    ci = _bootstrap_auprc_by_object(labels_array, scores_array, object_ids, seed=seed)

    return {
        "n_objects": n_objects, "n_real": int(labels_array.sum()),
        "n_artifact": int(n_objects - labels_array.sum()),
        "runs_surviving_min_run_length": n_runs_found,
        "auprc": ci["point"], "auprc_ci": ci["ci"],
    }


def evaluate_coherence_precision(*, n_objects: int = 200, n_epochs: int = DEFAULT_N_EPOCHS,
                                 run_length: int = 5, real_fraction: float = 0.5,
                                 min_run_length: int = 3, seed: int = 1) -> dict[str, Any]:
    """Precision/recall of `DiscardRecord.coherent`'s boolean verdict alone,
    the simplest possible headline number: of the runs `discard_pile` calls
    coherent, what fraction were really injected excursions, and what
    fraction of real excursions did it find?

    All synthetic runs here carry the same flag word (`FLAG_WORD`), so a
    per-flag-category breakdown -- which the plan calls for on real,
    multi-category data -- is not meaningful on this single-category
    synthetic set; grouping by category is a real-data reporting step, not
    part of this synthetic study.
    """
    rng = np.random.default_rng(seed)
    objects = _generate_objects(rng, n_objects=n_objects, n_epochs=n_epochs,
                                run_length=run_length, real_fraction=real_fraction)

    true_positive = false_positive = false_negative = true_negative = 0
    for _, real, curve, catflags in objects:
        records = dp.extract_discard_records(curve, catflags, min_run_length=min_run_length)
        predicted_real = any(record.coherent for record in records)
        if predicted_real and real:
            true_positive += 1
        elif predicted_real and not real:
            false_positive += 1
        elif not predicted_real and real:
            false_negative += 1
        else:
            true_negative += 1

    precision = (true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) else float("nan"))
    recall = (true_positive / (true_positive + false_negative)
             if (true_positive + false_negative) else float("nan"))

    return {
        "n_objects": n_objects,
        "true_positive": true_positive, "false_positive": false_positive,
        "false_negative": false_negative, "true_negative": true_negative,
        "precision": round(precision, 4) if precision == precision else precision,
        "recall": round(recall, 4) if recall == recall else recall,
    }
