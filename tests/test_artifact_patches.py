"""artifact_patches.py: real TESS instrumental-artifact patches from real
per-cadence QUALITY bitmasks (backlog item 14, gap 1)."""

from __future__ import annotations

import io

import numpy as np
import pytest

from astra import artifact_patches as ap

pytest.importorskip("lightkurve", reason="lightkurve not installed")


def _fake_tpf(tmp_path, n=60, ny=5, nx=5, quality=None, sector=7):
    """Mirrors tests/test_tess_pixels.py's own synthetic-TPF fixture style."""
    from astropy.io import fits

    time = 1000.0 + np.arange(n, dtype=np.float64)
    flux = np.full((n, ny, nx), 100.0, dtype=np.float32)
    flux[:, ny // 2, nx // 2] += np.sin(np.linspace(0, 6, n)).astype(np.float32) * 5.0
    if quality is None:
        quality = np.zeros(n, dtype=np.uint32)

    columns = [
        fits.Column(name="TIME", format="D", array=time),
        fits.Column(name="FLUX", format=f"{ny * nx}E", dim=f"({nx},{ny})", array=flux),
        fits.Column(name="QUALITY", format="J", array=quality),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    table.header["BJDREFI"] = 2457000
    primary = fits.PrimaryHDU()
    primary.header["SECTOR"] = sector

    path = tmp_path / "fake_tpf.fits"
    fits.HDUList([primary, table]).writeto(path, overwrite=True)
    return path


class TestDownloadReferenceTpfs:
    def test_downloads_one_tpf_per_target_with_coverage(self, monkeypatch, tmp_path):
        from astra import tess_pixels

        def fake_find_sectors(ra_deg, dec_deg):
            return [11] if ra_deg == 10.0 else []  # only the first target has coverage

        calls = []

        def fake_download_tpf(request, root=None, max_bytes=None):
            calls.append(request)
            path = tmp_path / f"{request.ra_deg}.fits"
            path.write_bytes(b"fake")
            return {"path": str(path)}

        monkeypatch.setattr(tess_pixels, "find_sectors", fake_find_sectors)
        monkeypatch.setattr(tess_pixels, "download_tpf", fake_download_tpf)

        paths = ap.download_reference_tpfs([(10.0, 20.0), (30.0, 40.0)], root=tmp_path)

        assert len(paths) == 1  # the second target had no sector coverage
        assert len(calls) == 1

    def test_a_failing_download_does_not_abort_the_batch(self, monkeypatch, tmp_path):
        from astra import tess_pixels

        monkeypatch.setattr(tess_pixels, "find_sectors", lambda ra, dec: [11])

        def fake_download_tpf(request, root=None, max_bytes=None):
            raise RuntimeError("simulated network failure")

        monkeypatch.setattr(tess_pixels, "download_tpf", fake_download_tpf)

        paths = ap.download_reference_tpfs([(10.0, 20.0)], root=tmp_path)
        assert paths == []

    def test_no_targets_returns_an_empty_list(self):
        assert ap.download_reference_tpfs([]) == []


class TestCategorizeQuality:
    def test_zero_is_clean(self):
        assert ap.categorize_quality(0) is None

    def test_real_cosmic_ray_bit(self):
        from lightkurve.utils import TessQualityFlags as Q
        assert ap.categorize_quality(Q.ApertureCosmic) == "cosmic_ray"

    def test_real_stray_light_bit(self):
        from lightkurve.utils import TessQualityFlags as Q
        assert ap.categorize_quality(Q.Straylight) == "stray_light"
        assert ap.categorize_quality(Q.Straylight2) == "stray_light"

    def test_real_pointing_bits(self):
        from lightkurve.utils import TessQualityFlags as Q
        assert ap.categorize_quality(Q.CoarsePoint) == "pointing"
        assert ap.categorize_quality(Q.EarthPoint) == "pointing"

    def test_an_ungrouped_nonzero_bit_falls_back_to_excluded(self):
        assert ap.categorize_quality(1 << 20) == "excluded"

    def test_a_combined_word_picks_the_first_matching_category(self):
        from lightkurve.utils import TessQualityFlags as Q
        combined = Q.ApertureCosmic | Q.Straylight
        assert ap.categorize_quality(combined) == "cosmic_ray"

    def test_category_names_are_a_stable_fixed_tuple(self):
        assert ap.CATEGORY_NAMES[0] == "clean"
        assert set(ap.CATEGORY_NAMES) == {
            "clean", "cosmic_ray", "stray_light", "pointing", "systematic", "excluded"}


class TestExtractArtifactPatches:
    def test_finds_a_flagged_run_and_labels_it(self, tmp_path):
        from lightkurve.utils import TessQualityFlags as Q

        quality = np.zeros(60, dtype=np.uint32)
        quality[10:15] = Q.ApertureCosmic
        path = _fake_tpf(tmp_path, quality=quality)

        patches, labels = ap.extract_artifact_patches([path], patch_length=8, min_run_length=3)

        assert len(patches) > 0
        assert patches.shape[1:] == (2, 8)
        assert ap.CATEGORY_NAMES.index("cosmic_ray") in labels

    def test_finds_clean_contrast_windows_too(self, tmp_path):
        from lightkurve.utils import TessQualityFlags as Q

        quality = np.zeros(60, dtype=np.uint32)
        quality[10:15] = Q.ApertureCosmic
        path = _fake_tpf(tmp_path, quality=quality)

        _, labels = ap.extract_artifact_patches([path], patch_length=8, min_run_length=3)

        assert ap.CATEGORY_NAMES.index("clean") in labels

    def test_a_run_shorter_than_min_run_length_is_ignored(self, tmp_path):
        from lightkurve.utils import TessQualityFlags as Q

        quality = np.zeros(60, dtype=np.uint32)
        quality[10:11] = Q.ApertureCosmic  # a single flagged cadence
        path = _fake_tpf(tmp_path, quality=quality)

        _, labels = ap.extract_artifact_patches([path], patch_length=8, min_run_length=5)

        assert ap.CATEGORY_NAMES.index("cosmic_ray") not in labels

    def test_fully_clean_tpf_yields_only_clean_or_no_patches(self, tmp_path):
        path = _fake_tpf(tmp_path)  # all-zero quality
        _, labels = ap.extract_artifact_patches([path], patch_length=8, min_run_length=3)
        assert all(label == ap.CATEGORY_NAMES.index("clean") for label in labels)

    def test_a_corrupt_path_is_skipped_not_raised(self, tmp_path):
        bad_path = tmp_path / "not_a_fits_file.fits"
        bad_path.write_bytes(b"not a fits file")
        patches, labels = ap.extract_artifact_patches([bad_path], patch_length=8)
        assert len(patches) == 0
        assert len(labels) == 0

    def test_no_paths_returns_an_empty_array_not_an_error(self):
        patches, labels = ap.extract_artifact_patches([], patch_length=16)
        assert patches.shape == (0, 2, 16)
        assert labels.shape == (0,)

    def test_output_values_are_finite(self, tmp_path):
        from lightkurve.utils import TessQualityFlags as Q

        quality = np.zeros(60, dtype=np.uint32)
        quality[20:25] = Q.Straylight
        path = _fake_tpf(tmp_path, quality=quality)

        patches, _ = ap.extract_artifact_patches([path], patch_length=8, min_run_length=3)
        assert np.all(np.isfinite(patches))

    def test_respects_max_patches_per_category(self, tmp_path):
        from lightkurve.utils import TessQualityFlags as Q

        # Several short, separated cosmic-ray runs.
        quality = np.zeros(200, dtype=np.uint32)
        for start in range(10, 190, 10):
            quality[start:start + 3] = Q.ApertureCosmic
        path = _fake_tpf(tmp_path, n=200, quality=quality)

        patches, labels = ap.extract_artifact_patches(
            [path], patch_length=8, min_run_length=3, max_patches_per_category=2)

        cosmic_count = int(np.sum(labels == ap.CATEGORY_NAMES.index("cosmic_ray")))
        assert cosmic_count <= 2
