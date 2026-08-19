"""Bounded image-derived feature extraction and provenance."""

from __future__ import annotations

import json

import numpy as np

from astra import image_features


def _fits(tmp_path):
    from astropy.io import fits

    data = np.full((32, 32), 100.0, dtype=np.float32)
    data[16, 16] = 500.0
    data[15, 16] = 300.0
    data[16, 15] = 250.0
    path = tmp_path / "feature.fits"
    fits.PrimaryHDU(data).writeto(path)
    return path


def test_extract_reports_morphology_and_source_checksum(tmp_path):
    path = _fits(tmp_path)
    payload = image_features.extract(path, target_xy=(16.0, 16.0))

    assert payload["schema_version"] == image_features.FEATURE_SCHEMA_VERSION
    assert payload["source"]["sha256"]
    assert payload["features"]["peak_snr"] > 5
    assert payload["features"]["detected_pixel_count"] >= 1
    assert payload["features"]["target_centroid_distance_pixels"] < 2
    json.dumps(payload)


def test_extract_rejects_oversized_pixel_arrays(monkeypatch, tmp_path):
    path = _fits(tmp_path)
    monkeypatch.setattr(image_features, "MAX_PIXELS", 10)
    try:
        image_features.extract(path)
    except ValueError as exc:
        assert "pixels" in str(exc)
    else:
        raise AssertionError("oversized image was accepted")


def test_save_is_atomic_and_uses_checksum_name(tmp_path):
    payload = {"source": {"sha256": "a" * 64}, "features": {"x": 1}}
    path = image_features.save(payload, tmp_path)
    assert path.exists()
    assert path.name.startswith("image_features_" + "a" * 32)
