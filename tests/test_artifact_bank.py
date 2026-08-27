"""Camera/CCD/night parsing, patch-feature extraction, CORAL correctness,
and classifier training for `artifact_bank.py`. No `research` extra
needed (no new optional dependency).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astra import artifact_bank as ab


# ---------------------------------------------------------------------------
# extract_camera_ccd / night_bucket
# ---------------------------------------------------------------------------

def test_extract_camera_ccd_parses_a_constructed_fits_header(tmp_path):
    from astropy.io import fits

    hdu = fits.PrimaryHDU()
    hdu.header["CAMERA"] = 2
    hdu.header["CCD"] = 3
    path = tmp_path / "fake.fits"
    hdu.writeto(path)

    camera, ccd = ab.extract_camera_ccd(path)
    assert camera == 2
    assert ccd == 3


def test_extract_camera_ccd_returns_none_for_missing_keywords(tmp_path):
    from astropy.io import fits

    path = tmp_path / "bare.fits"
    fits.PrimaryHDU().writeto(path)
    assert ab.extract_camera_ccd(path) == (None, None)


def test_extract_camera_ccd_returns_none_for_a_missing_file(tmp_path):
    assert ab.extract_camera_ccd(tmp_path / "does_not_exist.fits") == (None, None)


def test_night_bucket_returns_a_calendar_date_string():
    # 2459000.5 JD (TDB) is a real, unremarkable date in TESS's operating era.
    night = ab.night_bucket(2459000.5)
    assert night is not None
    assert len(night) == 10  # YYYY-MM-DD
    assert night.count("-") == 2


def test_night_bucket_returns_none_for_non_finite_input():
    assert ab.night_bucket(float("nan")) is None


# ---------------------------------------------------------------------------
# patch_features
# ---------------------------------------------------------------------------

def test_patch_features_returns_six_statistics():
    rng = np.random.default_rng(1)
    value = rng.normal(size=32)
    mask = np.ones(32)
    patch = np.stack([value, mask])
    features = ab.patch_features(patch)
    assert features.shape == (6,)
    assert np.all(np.isfinite(features))


def test_patch_features_returns_zeros_when_fully_masked():
    patch = np.stack([np.zeros(32), np.zeros(32)])
    assert np.array_equal(ab.patch_features(patch), np.zeros(6))


def test_patch_features_rejects_wrong_shape():
    with pytest.raises(ab.ArtifactBankError):
        ab.patch_features(np.zeros((3, 32)))


# ---------------------------------------------------------------------------
# coral_align
# ---------------------------------------------------------------------------

def test_coral_align_moves_source_covariance_toward_target_covariance():
    rng = np.random.default_rng(3)
    # Source: tight, near-zero-mean cluster. Target: wide, shifted cluster.
    source = rng.normal(loc=0.0, scale=0.2, size=(200, 3))
    target = rng.normal(loc=5.0, scale=2.0, size=(200, 3))

    aligned = ab.coral_align(source, target)

    cov_target = np.cov(target, rowvar=False)
    cov_aligned = np.cov(aligned, rowvar=False)
    cov_source = np.cov(source, rowvar=False)

    # The aligned source's covariance should be a much closer match to the
    # target's than the raw (unaligned) source's covariance was.
    assert np.linalg.norm(cov_aligned - cov_target) < np.linalg.norm(cov_source - cov_target)
    # And its mean should land close to the target's mean.
    assert np.allclose(aligned.mean(axis=0), target.mean(axis=0), atol=0.5)


def test_coral_align_rejects_mismatched_feature_dimensions():
    with pytest.raises(ab.ArtifactBankError):
        ab.coral_align(np.zeros((10, 3)), np.zeros((10, 4)))


def test_coral_align_rejects_too_few_samples():
    with pytest.raises(ab.ArtifactBankError):
        ab.coral_align(np.zeros((1, 3)), np.zeros((10, 3)))


# ---------------------------------------------------------------------------
# train_hard_negative_classifier
# ---------------------------------------------------------------------------

def test_train_hard_negative_classifier_fits_a_separable_toy_set():
    rng = np.random.default_rng(5)
    clean = rng.normal(loc=0.0, scale=0.1, size=(30, 6))
    artifact = rng.normal(loc=5.0, scale=0.1, size=(30, 6))
    features = np.vstack([clean, artifact])
    labels = np.array([0] * 30 + [1] * 30)

    model = ab.train_hard_negative_classifier(features, labels)
    predictions = model.predict(features)
    assert (predictions == labels).mean() > 0.9


def test_train_hard_negative_classifier_rejects_a_single_class():
    with pytest.raises(ab.ArtifactBankError):
        ab.train_hard_negative_classifier(np.zeros((5, 3)), np.zeros(5))


def test_train_hard_negative_classifier_rejects_mismatched_lengths():
    with pytest.raises(ab.ArtifactBankError):
        ab.train_hard_negative_classifier(np.zeros((5, 3)), np.array([0, 1]))


def test_artifact_bank_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "artifact_bank" not in rpc_source
