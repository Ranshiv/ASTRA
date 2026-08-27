"""Causal CCD-artifact attribution (roadmap item 35, P2, exploratory).

A feasibility scan before writing any code this session confirmed this
item is more constrained than its own data list ("pixel-level cutouts,
observing conditions, detector telemetry") suggests, and this module is
scoped around exactly what was confirmed real and reachable, not what the
item names in the abstract:

- Real DETECTOR TELEMETRY (CCD temperature/voltage/gain, gyro/attitude
  housekeeping) is UNREACHABLE: `tess_pixels.py`'s TPF reader parses only
  `TIME`/`FLUX`/`FLUX_ERR`/`QUALITY`/`BJDREFI`/`BJDREFF`/`TIMEUNIT`/
  `TIMESYS` from the FITS header (confirmed by reading it) -- true
  detector engineering telemetry is a separate MAST product this codebase
  does not download. Not attempted here, the same restraint
  `artifact_patches.py` already applies to ZTF artifacts.
- Real "OBSERVING CONDITIONS" (seeing, airmass, moon illumination) do not
  exist anywhere in this codebase for either survey (grepped, zero hits).
- Real CAUSAL-INFERENCE machinery (a structural causal model, do-calculus,
  propensity scores) does not exist anywhere in this codebase either
  (grepped, zero hits) -- and no randomized/natural-experiment structure
  is available to identify one from real TESS data.

What IS real and reachable: `artifact_bank.py`'s (item #33) real
`camera`/`ccd`/`night` per real TESS artifact patch, reused UNCHANGED
here (`PatchRecord`, `build_patch_bank`, `patch_features`,
`train_hard_negative_classifier`), plus one new covariate this module
adds -- a real per-cadence local background level, `extract_background_
level`, computed from the same real pixel cube those functions already
read (`tess_pixels.read_tpf_cube`), the same whole-frame-median
background `artifact_patches._cadence_flux` already computes internally
as an intermediate, exposed here as its own reusable value. `night` is
already a per-FILE (not per-patch) approximation in `artifact_bank.py`,
documented there; `background_level` follows the identical convention,
for the identical reason (the file's per-patch cadence windows are not
exposed by the reused extraction function).

Given that, this module does NOT claim genuine causal identification. It
implements the standard, citable alternative when a full causal graph or
a randomized intervention is unavailable: REGRESSION/COVARIATE
ADJUSTMENT (Rubin 1974's potential-outcomes framing of a "counterfactual"
outcome under a different covariate assignment; Pearl 2009, *Causality:
Models, Reasoning, and Inference*, 2nd ed., the "backdoor adjustment"/
g-computation formula in its simplest linear form). `fit_covariate_
adjustment` fits an outcome model of `is_artifact` on the real covariates
(one-hot camera/CCD, standardized background level);
`counterfactual_probability` evaluates that model at an arbitrary
covariate assignment -- the g-computation step. `adjusted_scores`
subtracts the covariate-attributable portion of a raw classifier's
predicted probability from its score, a standard "partial out a
covariate" construction -- `ccd_attribution_eval.py` then checks whether
this measurably reduces false positives attributable to a real or
constructed camera/CCD/background bias, the item's own named metric.

Like every other opt-in research module in this codebase, NOT wired into
`rpc.py`, `scoring.WEIGHTS`, or `evidence.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import artifact_bank
from .artifact_bank import PatchRecord


class CCDAttributionError(ValueError):
    """A covariate patch, design matrix, or adjustment-model input was invalid."""


@dataclass(frozen=True)
class CovariatePatch:
    record: PatchRecord
    background_level: float | None


def extract_background_level(tpf_path: str | Path) -> float | None:
    """Median, across real cadences, of each cadence's real whole-frame
    background level (`np.median` of the finite pixels in that frame) --
    the same quantity `artifact_patches._cadence_flux` already computes
    internally and discards, exposed here as its own reusable value.
    `None` on a corrupt/unreadable file or an empty cube, never
    fabricated."""
    from . import tess_pixels

    try:
        data = tess_pixels.read_tpf_cube(Path(tpf_path))
    except Exception:  # noqa: BLE001 - a corrupt TPF yields unknown, not a crash
        return None

    cube = data.get("flux")
    if cube is None or len(cube) == 0:
        return None

    backgrounds = []
    for frame in cube:
        finite = frame[np.isfinite(frame)]
        if finite.size:
            backgrounds.append(float(np.median(finite)))
    return float(np.median(backgrounds)) if backgrounds else None


def build_covariate_patch_bank(tpf_paths: list[str | Path],
                               **build_patch_bank_kwargs) -> list[CovariatePatch]:
    """Real artifact patches (`artifact_bank.build_patch_bank`, reused
    unchanged, called once per file for the same reason that function
    itself calls `artifact_patches.extract_artifact_patches` once per
    file) plus one real per-FILE background-level covariate attached to
    every patch drawn from that file."""
    patches: list[CovariatePatch] = []
    for raw_path in tpf_paths:
        path = Path(raw_path)
        records = artifact_bank.build_patch_bank([path], **build_patch_bank_kwargs)
        background = extract_background_level(path)
        patches.extend(CovariatePatch(record=record, background_level=background)
                       for record in records)
    return patches


def covariate_design_matrix(patches: list[CovariatePatch]) -> tuple[np.ndarray, list[str], dict]:
    """One-hot camera/CCD columns plus a standardized background column.
    A patch with a missing `background_level` is imputed with the
    population median -- a stated, documented limitation, not a silent
    one. Returns `(X, column_names, stats)`, where `stats` (cameras,
    ccds, background_median, background_std) is what `counterfactual_
    probability` needs to score an arbitrary covariate assignment on the
    SAME column layout later."""
    if not patches:
        raise CCDAttributionError("patches must be non-empty")

    cameras = tuple(sorted({p.record.camera for p in patches if p.record.camera is not None}))
    ccds = tuple(sorted({p.record.ccd for p in patches if p.record.ccd is not None}))
    backgrounds = [p.background_level for p in patches if p.background_level is not None]
    if not backgrounds:
        raise CCDAttributionError("at least one patch must have a known background_level")
    background_median = float(np.median(backgrounds))
    background_std = float(np.std(backgrounds)) or 1.0

    columns = [f"camera_{c}" for c in cameras] + [f"ccd_{c}" for c in ccds] + ["background_z"]
    rows = []
    for patch in patches:
        row = [1.0 if patch.record.camera == c else 0.0 for c in cameras]
        row += [1.0 if patch.record.ccd == c else 0.0 for c in ccds]
        bg = patch.background_level if patch.background_level is not None else background_median
        row.append((bg - background_median) / background_std)
        rows.append(row)

    stats = {"cameras": cameras, "ccds": ccds,
             "background_median": background_median, "background_std": background_std}
    return np.array(rows, dtype=np.float64), columns, stats


@dataclass(frozen=True)
class AdjustmentModel:
    model: object  # fitted sklearn.linear_model.LogisticRegression
    cameras: tuple[int, ...]
    ccds: tuple[int, ...]
    background_median: float
    background_std: float
    column_names: tuple[str, ...]


def fit_covariate_adjustment(patches: list[CovariatePatch], *, seed: int = 42) -> AdjustmentModel:
    """The covariate-outcome model `counterfactual_probability`/
    `adjusted_scores` need: `is_artifact ~ camera + ccd + background`."""
    from sklearn.linear_model import LogisticRegression

    design, columns, stats = covariate_design_matrix(patches)
    labels = np.array([0 if p.record.category == "clean" else 1 for p in patches])
    if len(np.unique(labels)) < 2:
        raise CCDAttributionError("patches must contain both clean and artifact examples")

    model = LogisticRegression(max_iter=1000, random_state=seed).fit(design, labels)
    return AdjustmentModel(model=model, column_names=tuple(columns), **stats)


def counterfactual_probability(model: AdjustmentModel, *, camera: int | None, ccd: int | None,
                               background_level: float | None) -> float:
    """`P(is_artifact | camera, ccd, background)` under `model` -- the
    g-computation step: evaluating the fitted outcome model at an
    arbitrary, possibly counterfactual, covariate assignment."""
    row = [1.0 if camera == c else 0.0 for c in model.cameras]
    row += [1.0 if ccd == c else 0.0 for c in model.ccds]
    bg = background_level if background_level is not None else model.background_median
    row.append((bg - model.background_median) / model.background_std)
    design = np.array([row], dtype=np.float64)
    return float(model.model.predict_proba(design)[0, 1])


def adjusted_scores(patches: list[CovariatePatch], feature_model, adjustment_model: AdjustmentModel, *,
                    reference_camera: int | None, reference_ccd: int | None,
                    reference_background: float) -> np.ndarray:
    """`raw_score - (P(artifact | real covariates) - P(artifact |
    reference covariates))` per patch -- partials the covariate-
    attributable portion of a raw classifier's score out, clipped back
    into `[0, 1]`. `feature_model` is any fitted classifier exposing
    `predict_proba` over `artifact_bank.patch_features` (e.g. `artifact_
    bank.train_hard_negative_classifier`'s output), reused unchanged."""
    if not patches:
        raise CCDAttributionError("patches must be non-empty")

    from .artifact_bank import patch_features

    features = np.array([patch_features(p.record.patch) for p in patches])
    raw = feature_model.predict_proba(features)[:, 1]
    reference_probability = counterfactual_probability(
        adjustment_model, camera=reference_camera, ccd=reference_ccd,
        background_level=reference_background)
    covariate_effect = np.array([
        counterfactual_probability(adjustment_model, camera=p.record.camera, ccd=p.record.ccd,
                                   background_level=p.background_level) - reference_probability
        for p in patches
    ])
    return np.clip(raw - covariate_effect, 0.0, 1.0)


__all__ = [
    "CCDAttributionError", "CovariatePatch", "extract_background_level",
    "build_covariate_patch_bank", "covariate_design_matrix", "AdjustmentModel",
    "fit_covariate_adjustment", "counterfactual_probability", "adjusted_scores",
]
