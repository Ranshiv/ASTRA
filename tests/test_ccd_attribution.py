"""Background extraction, covariate design matrix, and the covariate-
adjustment model's correctness for `ccd_attribution.py`. No `research`
extra needed (no new optional dependency).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import ccd_attribution as ca
from astra.artifact_bank import PatchRecord


def _patch(camera, category="clean", background=0.0, patch_length=16):
    value = np.zeros(patch_length, dtype=np.float32)
    mask = np.ones(patch_length, dtype=np.float32)
    record = PatchRecord(category=category, sector=1, camera=camera, ccd=1,
                         night="2026-01-01", patch=np.stack([value, mask]))
    return ca.CovariatePatch(record=record, background_level=background)


# ---------------------------------------------------------------------------
# extract_background_level
# ---------------------------------------------------------------------------

def test_extract_background_level_returns_none_for_a_missing_file(tmp_path):
    assert ca.extract_background_level(tmp_path / "does_not_exist.fits") is None


# ---------------------------------------------------------------------------
# covariate_design_matrix
# ---------------------------------------------------------------------------

def test_covariate_design_matrix_shape_and_columns():
    patches = [_patch(1, background=0.0), _patch(2, background=5.0)]
    design, columns, stats = ca.covariate_design_matrix(patches)
    assert design.shape == (2, len(columns))
    assert "background_z" in columns
    assert stats["cameras"] == (1, 2)


def test_covariate_design_matrix_imputes_missing_background_with_the_median():
    patches = [_patch(1, background=0.0), _patch(1, background=10.0),
              ca.CovariatePatch(record=_patch(1).record, background_level=None)]
    design, columns, stats = ca.covariate_design_matrix(patches)
    background_col = columns.index("background_z")
    # The imputed row's z-score should be exactly 0 (median minus median).
    assert design[2, background_col] == pytest.approx(0.0)


def test_covariate_design_matrix_rejects_empty_patches():
    with pytest.raises(ca.CCDAttributionError):
        ca.covariate_design_matrix([])


def test_covariate_design_matrix_rejects_all_missing_background():
    patches = [ca.CovariatePatch(record=_patch(1).record, background_level=None)]
    with pytest.raises(ca.CCDAttributionError):
        ca.covariate_design_matrix(patches)


# ---------------------------------------------------------------------------
# fit_covariate_adjustment / counterfactual_probability
# ---------------------------------------------------------------------------

def test_fit_covariate_adjustment_recovers_a_known_camera_prevalence_effect():
    rng = np.random.default_rng(0)
    patches = []
    # Camera 1: 10% artifact prevalence, background 0. Camera 2: 90%
    # artifact prevalence, background 5 -- a real, known covariate-label
    # association for the model to recover.
    for camera, prevalence, background in ((1, 0.1, 0.0), (2, 0.9, 5.0)):
        for _ in range(200):
            category = "cosmic_ray" if rng.random() < prevalence else "clean"
            patches.append(ca.CovariatePatch(
                record=PatchRecord(category=category, sector=1, camera=camera, ccd=1,
                                   night="2026-01-01",
                                   patch=np.stack([np.zeros(8), np.ones(8)])),
                background_level=background))

    model = ca.fit_covariate_adjustment(patches, seed=1)
    p_camera1 = ca.counterfactual_probability(model, camera=1, ccd=1, background_level=0.0)
    p_camera2 = ca.counterfactual_probability(model, camera=2, ccd=1, background_level=5.0)
    assert p_camera1 < 0.5 < p_camera2


def test_fit_covariate_adjustment_rejects_a_single_class():
    patches = [_patch(1, category="clean") for _ in range(5)]
    with pytest.raises(ca.CCDAttributionError):
        ca.fit_covariate_adjustment(patches)


# ---------------------------------------------------------------------------
# adjusted_scores
# ---------------------------------------------------------------------------

class _ConstantModel:
    def predict_proba(self, features):
        return np.tile([0.5, 0.5], (len(features), 1))


def test_adjusted_scores_leaves_the_reference_camera_unchanged():
    rng = np.random.default_rng(2)
    patches = []
    for camera, prevalence, background in ((1, 0.1, 0.0), (2, 0.9, 5.0)):
        for _ in range(100):
            category = "cosmic_ray" if rng.random() < prevalence else "clean"
            patches.append(ca.CovariatePatch(
                record=PatchRecord(category=category, sector=1, camera=camera, ccd=1,
                                   night="2026-01-01",
                                   patch=np.stack([np.zeros(8), np.ones(8)])),
                background_level=background))
    model = ca.fit_covariate_adjustment(patches, seed=1)
    camera1_patches = [p for p in patches if p.record.camera == 1]

    adjusted = ca.adjusted_scores(camera1_patches, _ConstantModel(), model,
                                  reference_camera=1, reference_ccd=1, reference_background=0.0)
    # Reference == actual covariates for every camera-1 patch, so the
    # covariate effect term is exactly zero and the score is unchanged.
    assert np.allclose(adjusted, 0.5)


def test_adjusted_scores_rejects_empty_patches():
    with pytest.raises(ca.CCDAttributionError):
        ca.adjusted_scores([], _ConstantModel(), object(),
                           reference_camera=1, reference_ccd=1, reference_background=0.0)


def test_ccd_attribution_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "ccd_attribution" not in rpc_source
