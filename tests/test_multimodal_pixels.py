"""multimodal_pixels.py: bounded, fixed-size pixel arrays for the image
branch of the multimodal MoCo encoder (backlog item 11)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import multimodal_pixels as mp


class TestToFixedSize:
    def test_crops_a_larger_array_to_the_target_size(self):
        array = np.arange(64, dtype=np.float64).reshape(8, 8)
        out = mp.to_fixed_size(array, size=4)
        assert out.shape == (4, 4)

    def test_pads_a_smaller_array_to_the_target_size(self):
        array = np.ones((3, 3))
        out = mp.to_fixed_size(array, size=8)
        assert out.shape == (8, 8)

    def test_padding_uses_the_arrays_own_median_by_default(self):
        array = np.full((3, 3), 5.0)
        out = mp.to_fixed_size(array, size=7)
        # Corners are pure padding (median of an all-5.0 array is 5.0).
        assert out[0, 0] == pytest.approx(5.0)

    def test_explicit_fill_overrides_the_default(self):
        array = np.full((3, 3), 5.0)
        out = mp.to_fixed_size(array, size=7, fill=-1.0)
        assert out[0, 0] == pytest.approx(-1.0)

    def test_crop_keeps_the_centered_content_intact(self):
        array = np.zeros((6, 6))
        array[2:4, 2:4] = 9.0  # a 2x2 bright block dead center
        out = mp.to_fixed_size(array, size=4)
        # A 4x4 centered crop of a 6x6 array is array[1:5, 1:5]; the 9.0
        # block (rows/cols 2:4) lands at rows/cols 1:3 in the crop.
        assert np.all(out[1:3, 1:3] == 9.0)
        assert out[0, 0] == 0.0

    def test_output_has_no_nan_given_all_finite_input(self):
        array = np.random.default_rng(0).normal(size=(10, 10))
        out = mp.to_fixed_size(array, size=16)
        assert np.all(np.isfinite(out))


class TestPreprocessImage:
    def test_output_is_finite_given_finite_input(self):
        array = np.random.default_rng(1).normal(100, 20, size=(16, 16))
        out = mp.preprocess_image(array)
        assert np.all(np.isfinite(out))

    def test_non_finite_pixels_do_not_propagate_nan(self):
        array = np.ones((8, 8))
        array[3, 3] = np.nan
        array[4, 4] = np.inf
        out = mp.preprocess_image(array)
        assert np.all(np.isfinite(out))

    def test_background_maps_close_to_zero(self):
        # A uniform frame's own median is its background; arcsinh(0) == 0.
        array = np.full((10, 10), 42.0)
        out = mp.preprocess_image(array)
        assert np.allclose(out, 0.0, atol=1e-5)

    def test_brighter_source_produces_a_larger_stretched_value(self):
        base = np.zeros((10, 10))
        dim = base.copy()
        dim[5, 5] = 10.0
        bright = base.copy()
        bright[5, 5] = 1000.0
        assert mp.preprocess_image(bright)[5, 5] > mp.preprocess_image(dim)[5, 5]


class TestTessReferenceFrame:
    def test_excludes_cadences_failing_the_quality_mask(self, monkeypatch):
        cube = np.stack([np.full((4, 4), 1.0), np.full((4, 4), 100.0)])
        quality = np.array([0, 1], dtype=np.uint64)  # second cadence flagged

        import astra.tess_pixels as real_tess_pixels

        def fake_read(path):
            return {"flux": cube, "quality": quality}

        monkeypatch.setattr(real_tess_pixels, "read_tpf_cube", fake_read)

        frame = mp.tess_reference_frame("fake-path.fits", quality_mask=1)
        assert np.allclose(frame, 1.0)  # only the good cadence contributes

    def test_raises_when_no_cadence_passes(self, monkeypatch):
        cube = np.stack([np.full((4, 4), 1.0)])
        quality = np.array([1], dtype=np.uint64)

        import astra.tess_pixels as real_tess_pixels

        def fake_read(path):
            return {"flux": cube, "quality": quality}

        monkeypatch.setattr(real_tess_pixels, "read_tpf_cube", fake_read)

        with pytest.raises(ValueError, match="no cadence"):
            mp.tess_reference_frame("fake-path.fits", quality_mask=1)
