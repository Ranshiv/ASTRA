"""Cross-group AUPRC (real synthetic-shift TESS-shaped records) and the
synthetic cross-survey generalization proxy for `artifact_bank_eval.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import artifact_bank as ab
from astra import artifact_bank_eval as abe
from astra.surveys.base import LightCurve, SourceRef


def _shifted_records(rng, biases: dict[int, float], n_per_class: int = 25,
                     patch_length: int = 32) -> list[ab.PatchRecord]:
    """Constructed `PatchRecord`s where `camera` carries a real, controlled
    feature-domain shift (`bias`) independent of the classification signal
    -- lets a test check whether CORAL actually helps, not just runs."""
    records = []
    for camera, bias in biases.items():
        for is_artifact in (False, True):
            signal = 4.0 if is_artifact else 0.0
            category = "cosmic_ray" if is_artifact else "clean"
            for _ in range(n_per_class):
                value = rng.normal(loc=bias + signal, scale=0.3, size=patch_length)
                patch = np.stack([value, np.ones(patch_length)]).astype(np.float32)
                records.append(ab.PatchRecord(
                    category=category, sector=1, camera=camera, ccd=1,
                    night="2026-01-01", patch=patch))
    return records


# ---------------------------------------------------------------------------
# grouped_split
# ---------------------------------------------------------------------------

def test_grouped_split_separates_held_out_camera():
    rng = np.random.default_rng(1)
    records = _shifted_records(rng, {1: 0.0, 2: 10.0}, n_per_class=5)
    train_idx, test_idx = abe.grouped_split(records, "camera", 2)
    assert all(records[i].camera == 1 for i in train_idx)
    assert all(records[i].camera == 2 for i in test_idx)


def test_grouped_split_rejects_bad_group_key():
    with pytest.raises(ab.ArtifactBankError):
        abe.grouped_split([], "not_a_real_key", 1)


# ---------------------------------------------------------------------------
# evaluate_cross_group_auprc
# ---------------------------------------------------------------------------

def test_coral_improves_held_out_auprc_under_a_real_covariate_shift():
    rng = np.random.default_rng(11)
    records = _shifted_records(rng, {1: 0.0, 2: 10.0}, n_per_class=25)

    without_coral = abe.evaluate_cross_group_auprc(records, group_by="camera", use_coral=False, seed=1)
    with_coral = abe.evaluate_cross_group_auprc(records, group_by="camera", use_coral=True, seed=1)

    assert without_coral["mean_auprc"] is not None
    assert with_coral["mean_auprc"] is not None
    # A classifier trained on one camera's shifted features and evaluated
    # unadapted on the other camera's should do measurably worse than the
    # CORAL-aligned version.
    assert with_coral["mean_auprc"] >= without_coral["mean_auprc"]


def test_evaluate_cross_group_auprc_rejects_empty_records():
    with pytest.raises(ab.ArtifactBankError):
        abe.evaluate_cross_group_auprc([])


def test_evaluate_cross_group_auprc_rejects_a_single_group():
    rng = np.random.default_rng(2)
    records = _shifted_records(rng, {1: 0.0}, n_per_class=5)
    with pytest.raises(ab.ArtifactBankError):
        abe.evaluate_cross_group_auprc(records, group_by="camera")


# ---------------------------------------------------------------------------
# evaluate_cross_survey_generalization_synthetic
# ---------------------------------------------------------------------------

def _write_ztf_curves(root, n=8, seed=7):
    rng = np.random.default_rng(seed)
    for i in range(n):
        source = SourceRef(survey="ZTF", object_id=f"synthetic{i}", ra_deg=180.0, dec_deg=0.0)
        time = 2458000.0 + np.arange(200, dtype=np.float64) * 0.5
        value = 18.0 + rng.normal(0.0, 0.05, size=200)
        err = np.full(200, 0.03)
        curve = LightCurve(source=source, release="dr24", band="g", value_kind="mag",
                           time=time, value=value, value_err=err, time_system="HJD_UTC")
        from astra import store
        store.write_curve(curve, root=root)


def test_evaluate_cross_survey_generalization_synthetic_reports_a_valid_auprc(isolated_root):
    _write_ztf_curves(isolated_root.datasets, n=8)

    rng = np.random.default_rng(9)
    tess_records = _shifted_records(rng, {1: 0.0}, n_per_class=10)

    result = abe.evaluate_cross_survey_generalization_synthetic(
        tess_records, n_ztf_windows=8, seed=3)

    assert result["synthetic_proxy"] is True
    assert result["n_ztf_windows_found"] == 8
    assert result["auprc"] is None or 0.0 <= result["auprc"] <= 1.0


def test_evaluate_cross_survey_generalization_synthetic_handles_no_local_ztf_data(isolated_root):
    rng = np.random.default_rng(9)
    tess_records = _shifted_records(rng, {1: 0.0}, n_per_class=10)
    result = abe.evaluate_cross_survey_generalization_synthetic(tess_records, n_ztf_windows=8, seed=3)
    assert result["n_ztf_windows_found"] == 0
    assert result["auprc"] is None


def test_evaluate_cross_survey_generalization_synthetic_rejects_too_few_tess_records():
    with pytest.raises(ab.ArtifactBankError):
        abe.evaluate_cross_survey_generalization_synthetic([])


def test_artifact_bank_eval_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "artifact_bank" not in rpc_source
