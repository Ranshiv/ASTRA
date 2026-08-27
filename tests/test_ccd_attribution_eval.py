"""Leave-one-camera-out false-positive reduction and the controlled
base-rate-miscalibration demonstration for `ccd_attribution_eval.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import ccd_attribution as ca
from astra import ccd_attribution_eval as cae
from astra.artifact_bank import PatchRecord


def _camera_patches(rng, n, prevalence, camera, background, patch_length=32):
    patches = []
    for _ in range(n):
        is_artifact = rng.random() < prevalence
        loc = 1.5 if is_artifact else 0.0
        value = rng.normal(loc=loc, scale=1.5, size=patch_length)
        record = PatchRecord(category="cosmic_ray" if is_artifact else "clean", sector=1,
                             camera=camera, ccd=1, night="2026-01-01",
                             patch=np.stack([value, np.ones(patch_length)]).astype(np.float32))
        patches.append(ca.CovariatePatch(record=record, background_level=background))
    return patches


# ---------------------------------------------------------------------------
# evaluate_counterfactual_false_positive_reduction
# ---------------------------------------------------------------------------

def test_evaluate_counterfactual_false_positive_reduction_runs_end_to_end():
    rng = np.random.default_rng(5)
    patches = (_camera_patches(rng, 60, 0.3, 1, 0.0)
              + _camera_patches(rng, 60, 0.6, 2, 3.0))
    result = cae.evaluate_counterfactual_false_positive_reduction(patches, group_by="camera", seed=1)
    assert result["group_by"] == "camera"
    scored_folds = [f for f in result["folds"] if "raw_auprc" in f]
    assert scored_folds
    for fold in scored_folds:
        assert 0.0 <= fold["raw_auprc"] <= 1.0
        assert 0.0 <= fold["adjusted_auprc"] <= 1.0


def test_evaluate_counterfactual_false_positive_reduction_rejects_empty_patches():
    with pytest.raises(ca.CCDAttributionError):
        cae.evaluate_counterfactual_false_positive_reduction([])


def test_evaluate_counterfactual_false_positive_reduction_rejects_a_single_group():
    rng = np.random.default_rng(1)
    patches = _camera_patches(rng, 20, 0.3, 1, 0.0)
    with pytest.raises(ca.CCDAttributionError):
        cae.evaluate_counterfactual_false_positive_reduction(patches)


def test_evaluate_counterfactual_false_positive_reduction_rejects_bad_group_by():
    rng = np.random.default_rng(1)
    patches = (_camera_patches(rng, 20, 0.3, 1, 0.0) + _camera_patches(rng, 20, 0.3, 2, 0.0))
    with pytest.raises(ca.CCDAttributionError):
        cae.evaluate_counterfactual_false_positive_reduction(patches, group_by="not_a_field")


# ---------------------------------------------------------------------------
# evaluate_counterfactual_removal_synthetic
# ---------------------------------------------------------------------------

def test_evaluate_counterfactual_removal_synthetic_leaves_reference_camera_unchanged():
    result = cae.evaluate_counterfactual_removal_synthetic(seed=7)
    assert result["raw_fpr"]["camera1"] == pytest.approx(result["adjusted_fpr"]["camera1"])


@pytest.mark.parametrize("seed", range(1, 6))
def test_evaluate_counterfactual_removal_synthetic_reduces_camera_2_false_positives(seed):
    result = cae.evaluate_counterfactual_removal_synthetic(seed=seed)
    # Camera 2 has genuinely higher true prevalence than the reference
    # (camera 1), so its learned covariate effect is positive and
    # subtracting it from camera 2's raw scores can only ever lower them
    # -- a robust, mechanistically guaranteed reduction in camera 2's OWN
    # false-positive rate (verified non-trivial: raw_fpr > 0 here).
    assert result["raw_fpr"]["camera2"] > 0.0
    assert result["adjusted_fpr"]["camera2"] <= result["raw_fpr"]["camera2"] + 1e-9


def test_evaluate_counterfactual_removal_synthetic_rejects_bad_prevalence():
    with pytest.raises(ca.CCDAttributionError):
        cae.evaluate_counterfactual_removal_synthetic(prevalence_camera1=0.0)


def test_ccd_attribution_eval_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "ccd_attribution" not in rpc_source
