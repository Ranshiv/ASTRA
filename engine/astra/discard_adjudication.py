"""Pixel-level adjudication of discard-pile events (Direction 2, step 3 of
the research plan adopted 2026-08-29).

`discard_pile.py` finds a coherent run of discarded epochs in catalog
photometry; `discard_corroboration.py` asks whether an independent survey's
own catalog photometry agrees. This module asks a third, more fundamental
question of the SAME survey's own pixels: does an independent forced flux
measurement at the source's fixed position -- bypassing that survey's own
catalog pipeline entirely -- actually show the excursion the catalog
photometry (and its flag) claims?

This is the scoped pixel step the research plan calls for: forced PSF scene
photometry via `ztf_forced_photometry.build_scene_model`, not a from-scratch
difference-imaging implementation. The scene model already gives an
independent flux at each epoch; comparing that flux during the flagged
epochs against the same target's flux outside them is enough to adjudicate
without a new subtraction pipeline. Full image subtraction remains open
(`ztf_forced_photometry.py`'s own docstring already states real ZTF
difference-image acquisition is not attempted anywhere in this codebase)
and is deferred here for the same reason.

Like `discard_corroboration.py`, this module fetches nothing. The caller
supplies an already-assembled cutout cube (typically via `products.py`) and
tells this module which cube indices correspond to the discarded epochs;
everything else in the cube is treated as the target's own baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from . import ztf_forced_photometry as zfp
from .discard_pile import DiscardRecord

Verdict = Literal["likely_real", "likely_artifact", "inconclusive"]

DEFAULT_SIGNIFICANCE_SIGMA = 3.0
DEFAULT_MIN_VALID_EPOCHS = 2


@dataclass(frozen=True)
class AdjudicationResult:
    """The independent-pixel verdict on one discard-pile event."""

    record: DiscardRecord
    verdict: Verdict
    flux_z_score: float | None
    baseline_flux_mean: float | None
    flagged_flux_mean: float | None
    valid_baseline_epochs: int
    valid_flagged_epochs: int
    fit_failures: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "record": self.record.to_dict(),
            "verdict": self.verdict,
            "flux_z_score": (None if self.flux_z_score is None
                            else round(self.flux_z_score, 3)),
            "baseline_flux_mean": self.baseline_flux_mean,
            "flagged_flux_mean": self.flagged_flux_mean,
            "valid_baseline_epochs": self.valid_baseline_epochs,
            "valid_flagged_epochs": self.valid_flagged_epochs,
            "fit_failures": self.fit_failures,
            "reasons": self.reasons,
        }


def _weighted_mean_and_error(values: list[float], errors: list[float | None]
                             ) -> tuple[float | None, float | None]:
    """Error-weighted mean and its standard error; falls back to an
    unweighted mean and the sample standard error when per-point errors are
    unavailable, mirroring `discard_corroboration._window_deviation`'s same
    graceful degradation.
    """
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    if all(error is not None and error > 0 for error in errors):
        err = np.asarray(errors, dtype=np.float64)
        weights = 1.0 / err ** 2
        mean = float(np.sum(weights * array) / np.sum(weights))
        return mean, float(np.sqrt(1.0 / np.sum(weights)))
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, None
    return mean, float(np.std(array, ddof=1) / np.sqrt(len(array)))


def adjudicate(record: DiscardRecord, cube: np.ndarray,
              positions: list[zfp.ScenePosition], flagged_indices: list[int], *,
              error_cube: np.ndarray | None = None,
              fwhm_pixels: float = zfp.DEFAULT_FWHM_PIXELS,
              significance_sigma: float = DEFAULT_SIGNIFICANCE_SIGMA,
              min_valid_epochs: int = DEFAULT_MIN_VALID_EPOCHS) -> AdjudicationResult:
    """Independent forced-photometry verdict on one discard-pile event.

    `cube` is an `(epoch, y, x)` image cube covering both the discarded
    epochs and enough surrounding baseline to compare against, in the same
    time order as `record`'s survey epochs. `flagged_indices` are the cube
    indices that fall inside `record`'s discarded run; every other index is
    treated as baseline. `positions` must include a `zfp.TARGET_LABEL`
    position (see `zfp.build_scene_positions`) -- the source this record is
    about.
    """
    scene = zfp.build_scene_model(cube, positions, fwhm_pixels=fwhm_pixels,
                                  error_cube=error_cube)
    if zfp.TARGET_LABEL not in scene["flux_by_label"]:
        return AdjudicationResult(
            record=record, verdict="inconclusive", flux_z_score=None,
            baseline_flux_mean=None, flagged_flux_mean=None,
            valid_baseline_epochs=0, valid_flagged_epochs=0,
            fit_failures=scene["fit_failures"],
            reasons=["no target position in the scene model"],
        )

    target_flux = scene["flux_by_label"][zfp.TARGET_LABEL]
    target_flux_err = [
        None if epoch is None else epoch["flux_errors"].get(zfp.TARGET_LABEL)
        for epoch in scene["per_epoch"]
    ]
    flagged_set = set(flagged_indices)

    baseline_values, baseline_errors = [], []
    flagged_values, flagged_errors = [], []
    for index, flux in enumerate(target_flux):
        if flux is None or not np.isfinite(flux):
            continue
        target = (flagged_values, flagged_errors) if index in flagged_set \
            else (baseline_values, baseline_errors)
        target[0].append(flux)
        target[1].append(target_flux_err[index])

    if len(baseline_values) < min_valid_epochs or len(flagged_values) < min_valid_epochs:
        return AdjudicationResult(
            record=record, verdict="inconclusive", flux_z_score=None,
            baseline_flux_mean=(float(np.mean(baseline_values)) if baseline_values else None),
            flagged_flux_mean=(float(np.mean(flagged_values)) if flagged_values else None),
            valid_baseline_epochs=len(baseline_values), valid_flagged_epochs=len(flagged_values),
            fit_failures=scene["fit_failures"],
            reasons=["insufficient independently-fit epochs to adjudicate"],
        )

    baseline_mean, baseline_se = _weighted_mean_and_error(baseline_values, baseline_errors)
    flagged_mean, flagged_se = _weighted_mean_and_error(flagged_values, flagged_errors)

    z_score = None
    if baseline_se is not None and flagged_se is not None:
        combined_se = float(np.sqrt(baseline_se ** 2 + flagged_se ** 2))
        if combined_se > 0:
            z_score = float((flagged_mean - baseline_mean) / combined_se)

    if z_score is None:
        verdict: Verdict = "inconclusive"
        reasons = ["could not estimate a flux uncertainty for either window"]
    elif abs(z_score) < significance_sigma:
        verdict = "likely_artifact"
        reasons = ["independent forced photometry shows no significant flux "
                  "change during the flagged epochs"]
    else:
        # magnitude_offset > 0 means fainter in the catalog's own convention
        # (higher magnitude); an independent flux DECREASE during the
        # flagged epochs is the pixel-level equivalent of that. Agreement in
        # direction is what promotes the record; a significant but
        # OPPOSITE-signed pixel flux change means the catalog's own
        # photometry moved the wrong way -- itself evidence of a catalog
        # artifact (e.g. blending), not a real source event.
        catalog_fainter = record.magnitude_offset > 0
        pixel_fainter = flagged_mean < baseline_mean
        if catalog_fainter == pixel_fainter:
            verdict = "likely_real"
            reasons = ["independent forced photometry confirms a significant, "
                      "correctly-signed flux change during the flagged epochs"]
        else:
            verdict = "likely_artifact"
            reasons = ["independent forced photometry disagrees in direction "
                      "with the catalog-reported change"]

    return AdjudicationResult(
        record=record, verdict=verdict, flux_z_score=z_score,
        baseline_flux_mean=baseline_mean, flagged_flux_mean=flagged_mean,
        valid_baseline_epochs=len(baseline_values), valid_flagged_epochs=len(flagged_values),
        fit_failures=scene["fit_failures"], reasons=reasons,
    )
