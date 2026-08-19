"""FITS inspection and display scaling."""

from __future__ import annotations

import numpy as np
import pytest

from astra import fitsio


@pytest.fixture
def sample_fits(tmp_path):
    """A frame with realistic astronomical dynamic range."""
    from astropy.io import fits

    rng = np.random.default_rng(7)
    data = rng.normal(100.0, 5.0, size=(64, 64))
    data[32, 32] = 50_000.0  # a saturated star
    data[10, 10] = 900.0     # a faint source

    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdu.header["OBJECT"] = "TEST FIELD"
    hdu.header["TELESCOP"] = "ASTRA-TEST"
    hdu.header["FILTER"] = "g"
    hdu.header["EXPTIME"] = 30.0

    path = tmp_path / "sample.fits"
    hdu.writeto(path)
    return path


@pytest.fixture
def large_fits(tmp_path):
    from astropy.io import fits

    data = np.random.default_rng(3).normal(100.0, 5.0, size=(2048, 2048))
    path = tmp_path / "large.fits"
    fits.PrimaryHDU(data.astype(np.float32)).writeto(path)
    return path


class TestDescribe:
    def test_lists_hdus_and_shape(self, sample_fits):
        described = fitsio.describe(sample_fits)
        assert described["hdus"][0]["is_image"] is True
        assert described["hdus"][0]["shape"] == [64, 64]

    def test_reports_file_size(self, sample_fits):
        assert fitsio.describe(sample_fits)["size_mb"] > 0


class TestHeader:
    def test_summary_picks_the_interesting_cards(self, sample_fits):
        header = fitsio.read_header(sample_fits)
        assert header["summary"]["OBJECT"] == "TEST FIELD"
        assert header["summary"]["FILTER"] == "g"

    def test_full_cards_are_available(self, sample_fits):
        header = fitsio.read_header(sample_fits)
        assert "NAXIS" in header["cards"]

    def test_values_are_json_safe(self, sample_fits):
        import json

        json.dumps(fitsio.read_header(sample_fits))  # must not raise


class TestZScale:
    def test_limits_exclude_the_saturated_pixel(self, sample_fits):
        """A linear min-max stretch would be dominated by the 50,000 spike."""
        from astropy.io import fits

        data = fits.getdata(sample_fits).astype(np.float64)
        low, high = fitsio.zscale(data)

        assert high < 50_000.0
        assert low < high

    def test_uniform_frame_does_not_produce_an_empty_range(self):
        low, high = fitsio.zscale(np.full((32, 32), 5.0))
        assert high > low

    def test_all_nan_frame_is_handled(self):
        low, high = fitsio.zscale(np.full((8, 8), np.nan))
        assert (low, high) == (0.0, 1.0)


class TestDecimate:
    def test_small_frame_is_untouched(self):
        data = np.zeros((100, 100))
        assert fitsio.decimate(data, 512).shape == (100, 100)

    def test_large_frame_is_reduced_below_the_limit(self):
        data = np.zeros((2048, 2048))
        reduced = fitsio.decimate(data, 512)
        assert max(reduced.shape) <= 512

    def test_aspect_ratio_is_preserved(self):
        reduced = fitsio.decimate(np.zeros((1000, 500)), 250)
        assert reduced.shape[0] > reduced.shape[1]


class TestImagePayload:
    def test_pixels_are_eight_bit(self, sample_fits):
        payload = fitsio.image_payload(sample_fits)
        assert min(payload["pixels"]) >= 0
        assert max(payload["pixels"]) <= 255

    def test_pixel_count_matches_shape(self, sample_fits):
        payload = fitsio.image_payload(sample_fits)
        height, width = payload["shape"]
        assert len(payload["pixels"]) == height * width

    def test_large_frame_is_decimated_before_transport(self, large_fits):
        payload = fitsio.image_payload(large_fits, max_dimension=256)
        assert payload["decimated"] is True
        assert payload["original_shape"] == [2048, 2048]
        assert len(payload["pixels"]) <= 256 * 256

    def test_statistics_describe_the_frame(self, sample_fits):
        stats = fitsio.image_payload(sample_fits)["stats"]
        assert stats["max"] == pytest.approx(50_000.0, rel=0.01)
        assert 90 < stats["median"] < 110

    def test_payload_is_json_safe(self, sample_fits):
        import json

        json.dumps(fitsio.image_payload(sample_fits))  # must not raise
