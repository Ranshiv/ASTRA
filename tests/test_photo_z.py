"""photo_z_nmad arithmetic, kNN photo-z recovery on a known synthetic
colour-redshift relation, and calibration-sample cross-match wiring."""

from pathlib import Path

import numpy as np
import pytest

from astra import photo_z as pz
from astra.surveys.base import SourceRef


def _synthetic_calibration_sample(n=200, seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    mags = rng.uniform(18, 22, size=(n, 4))
    color1 = mags[:, 0] - mags[:, 1]
    color2 = mags[:, 1] - mags[:, 2]
    z_true = np.clip(0.1 * color1 + 0.05 * color2 + 0.3 + rng.normal(0.0, noise, n), 0.001, None)
    return [{"magnitudes": list(mags[i]), "z_true": float(z_true[i])} for i in range(n)]


# ---------------------------------------------------------------------------
# NMAD arithmetic
# ---------------------------------------------------------------------------

def test_photo_z_nmad_matches_hand_computed_value():
    z_true = np.array([1.0, 2.0])
    z_pred = np.array([1.1, 1.9])
    nmad = pz.photo_z_nmad(z_true, z_pred)
    expected = 1.4826 * np.median(np.abs((z_pred - z_true) / (1.0 + z_true)))
    assert nmad == pytest.approx(expected)


def test_photo_z_nmad_is_zero_for_perfect_prediction():
    z_true = np.array([0.1, 0.5, 1.2])
    assert pz.photo_z_nmad(z_true, z_true) == pytest.approx(0.0)


def test_photo_z_nmad_rejects_invalid_input():
    with pytest.raises(pz.PhotoZError):
        pz.photo_z_nmad([], [])
    with pytest.raises(pz.PhotoZError):
        pz.photo_z_nmad([-1.5], [0.0])


# ---------------------------------------------------------------------------
# kNN fit / evaluation
# ---------------------------------------------------------------------------

def test_fit_photo_z_knn_recovers_a_known_deterministic_relation():
    sample = _synthetic_calibration_sample(n=300, seed=1, noise=0.005)
    magnitudes = np.array([row["magnitudes"] for row in sample])
    redshifts = np.array([row["z_true"] for row in sample])
    model = pz.fit_photo_z_knn(magnitudes[:250], redshifts[:250], k=8)
    predicted = model.predict(magnitudes[250:])
    nmad = pz.photo_z_nmad(redshifts[250:], predicted)
    assert nmad < 0.05


def test_fit_photo_z_knn_rejects_too_few_rows():
    with pytest.raises(pz.PhotoZError):
        pz.fit_photo_z_knn(np.ones((3, 2)), np.array([0.1, 0.2, 0.3]))


def test_fit_photo_z_knn_rejects_mismatched_lengths():
    with pytest.raises(pz.PhotoZError):
        pz.fit_photo_z_knn(np.ones((20, 2)), np.ones(19))


def test_evaluate_photo_z_reports_small_nmad_on_a_clean_relation():
    sample = _synthetic_calibration_sample(n=200, seed=2, noise=0.01)
    result = pz.evaluate_photo_z(sample, k=8, n_seeds=5, seed=3)
    assert result["nmad"] is not None
    assert result["nmad"]["mean"] < 0.05
    assert result["n_seeds_used"] > 0


def test_evaluate_photo_z_rejects_too_few_rows():
    with pytest.raises(pz.PhotoZError):
        pz.evaluate_photo_z([{"magnitudes": [1, 2], "z_true": 0.1}])


# ---------------------------------------------------------------------------
# Calibration-sample cross-match wiring
# ---------------------------------------------------------------------------

def test_build_calibration_sample_pairs_photometry_with_redshift():
    photometry = [SourceRef(survey="DES", object_id="1", ra_deg=10.0, dec_deg=20.0,
                            extra={"g_mean": 20.0, "r_mean": 19.5, "i_mean": 19.2, "z_mean": 19.0})]
    redshifts = [SourceRef(survey="SDSS", object_id="2", ra_deg=10.0001, dec_deg=20.0001,
                           extra={"z": 0.42, "z_err": 0.001})]
    rows = pz.build_calibration_sample(photometry, redshifts,
                                       band_keys=("g_mean", "r_mean", "i_mean", "z_mean"),
                                       radius_arcsec=5.0)
    assert len(rows) == 1
    assert rows[0]["z_true"] == pytest.approx(0.42)
    assert rows[0]["magnitudes"] == [20.0, 19.5, 19.2, 19.0]


def test_build_calibration_sample_skips_unmatched_or_missing_redshift():
    photometry = [SourceRef(survey="DES", object_id="1", ra_deg=10.0, dec_deg=20.0,
                            extra={"g_mean": 20.0})]
    far_redshift = [SourceRef(survey="SDSS", object_id="2", ra_deg=50.0, dec_deg=-30.0, extra={"z": 0.42})]
    rows = pz.build_calibration_sample(photometry, far_redshift, band_keys=("g_mean",),
                                       radius_arcsec=2.0)
    assert rows == []


def test_build_calibration_sample_rejects_empty_band_keys():
    with pytest.raises(pz.PhotoZError):
        pz.build_calibration_sample([], [], band_keys=())


def test_photo_z_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "photo_z" not in rpc_source
