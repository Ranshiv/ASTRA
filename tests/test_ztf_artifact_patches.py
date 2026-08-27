"""ztf_artifact_patches.py: real ZTF instrumental-artifact patches from
real per-epoch `catflags` (the digital twin's ZTF-artifact follow-up)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import ztf_artifact_patches as zap
from astra.surveys.base import LightCurve, SourceRef


class TestCategorizeCatflags:
    def test_zero_is_clean(self):
        assert zap.categorize_catflags(0) is None

    def test_any_nonzero_word_is_flagged(self):
        assert zap.categorize_catflags(32768) == "flagged"
        assert zap.categorize_catflags(1) == "flagged"

    def test_category_names_are_coarse_and_stable(self):
        assert zap.CATEGORY_NAMES == ("clean", "flagged")


class TestExtractZtfArtifactPatches:
    def _curve(self, n=60, flagged_slice=None):
        value = np.zeros(n, dtype=np.float32)
        mask = np.ones(n, dtype=np.float32)
        catflags = np.zeros(n, dtype=np.uint32)
        if flagged_slice is not None:
            catflags[flagged_slice] = 32768
        return value, mask, catflags

    def test_finds_a_flagged_run_and_labels_it(self):
        value, mask, catflags = self._curve(flagged_slice=slice(10, 15))
        patches, labels = zap.extract_ztf_artifact_patches(
            [(value, mask, catflags)], patch_length=8, min_run_length=3)

        assert len(patches) > 0
        assert patches.shape[1:] == (2, 8)
        assert zap.CATEGORY_NAMES.index("flagged") in labels

    def test_finds_clean_contrast_windows_too(self):
        value, mask, catflags = self._curve(flagged_slice=slice(10, 15))
        _, labels = zap.extract_ztf_artifact_patches(
            [(value, mask, catflags)], patch_length=8, min_run_length=3)

        assert zap.CATEGORY_NAMES.index("clean") in labels

    def test_a_run_shorter_than_min_run_length_is_ignored(self):
        value, mask, catflags = self._curve(flagged_slice=slice(10, 11))
        _, labels = zap.extract_ztf_artifact_patches(
            [(value, mask, catflags)], patch_length=8, min_run_length=5)

        assert zap.CATEGORY_NAMES.index("flagged") not in labels

    def test_fully_clean_curve_yields_only_clean_or_no_patches(self):
        value, mask, catflags = self._curve()
        _, labels = zap.extract_ztf_artifact_patches(
            [(value, mask, catflags)], patch_length=8, min_run_length=3)
        assert all(label == zap.CATEGORY_NAMES.index("clean") for label in labels)

    def test_a_curve_shorter_than_patch_length_is_skipped(self):
        value, mask, catflags = self._curve(n=4)
        patches, labels = zap.extract_ztf_artifact_patches(
            [(value, mask, catflags)], patch_length=8)
        assert len(patches) == 0
        assert len(labels) == 0

    def test_mismatched_catflags_length_is_skipped(self):
        value, mask, _ = self._curve(n=60)
        catflags = np.zeros(30, dtype=np.uint32)
        patches, labels = zap.extract_ztf_artifact_patches(
            [(value, mask, catflags)], patch_length=8)
        assert len(patches) == 0
        assert len(labels) == 0

    def test_no_curves_returns_an_empty_array_not_an_error(self):
        patches, labels = zap.extract_ztf_artifact_patches([], patch_length=16)
        assert patches.shape == (0, 2, 16)
        assert labels.shape == (0,)

    def test_output_values_are_finite(self):
        value, mask, catflags = self._curve(flagged_slice=slice(20, 25))
        patches, _ = zap.extract_ztf_artifact_patches(
            [(value, mask, catflags)], patch_length=8, min_run_length=3)
        assert np.all(np.isfinite(patches))

    def test_respects_max_patches_per_category(self):
        n = 200
        value = np.zeros(n, dtype=np.float32)
        mask = np.ones(n, dtype=np.float32)
        catflags = np.zeros(n, dtype=np.uint32)
        for start in range(10, 190, 10):
            catflags[start:start + 3] = 32768

        patches, labels = zap.extract_ztf_artifact_patches(
            [(value, mask, catflags)], patch_length=8, min_run_length=3,
            max_patches_per_category=2)

        flagged_count = int(np.sum(labels == zap.CATEGORY_NAMES.index("flagged")))
        assert flagged_count <= 2


class _FakeConnector:
    """Duck-types `fetch_light_curves_with_quality` -- no network, mirrors
    `test_open_world_eval.py`'s fake-connector convention for a two-call
    composed function."""

    def __init__(self, n_points=60, flagged_slice=None, fail_object_id=None):
        self.n_points = n_points
        self.flagged_slice = flagged_slice
        self.fail_object_id = fail_object_id

    def fetch_light_curves_with_quality(self, source: SourceRef):
        if source.object_id == self.fail_object_id:
            raise RuntimeError("simulated fetch failure")
        n = self.n_points
        time = 58000.0 + np.arange(n, dtype=float)
        value = 18.0 + np.sin(np.linspace(0, 6, n))
        catflags = np.zeros(n, dtype=np.uint32)
        if self.flagged_slice is not None:
            catflags[self.flagged_slice] = 32768
        curve = LightCurve(
            source=source, release="dr24", band="g", value_kind="mag",
            time=time, value=value, value_err=np.full(n, 0.02),
            time_system="HJD_UTC",
        )
        return [(curve, catflags)]


class TestFetchAndExtract:
    def test_recovers_real_flagged_patches_via_the_connector(self):
        sources = [SourceRef(survey="ZTF", object_id="1", ra_deg=0.0, dec_deg=0.0)]
        connector = _FakeConnector(flagged_slice=slice(20, 25))

        patches, labels = zap.fetch_and_extract(
            sources, patch_length=8, min_run_length=3, connector=connector)

        assert len(patches) > 0
        assert zap.CATEGORY_NAMES.index("flagged") in labels

    def test_a_failing_source_does_not_abort_the_batch(self):
        sources = [
            SourceRef(survey="ZTF", object_id="bad", ra_deg=0.0, dec_deg=0.0),
            SourceRef(survey="ZTF", object_id="good", ra_deg=0.0, dec_deg=0.0),
        ]
        connector = _FakeConnector(flagged_slice=slice(20, 25), fail_object_id="bad")

        patches, labels = zap.fetch_and_extract(
            sources, patch_length=8, min_run_length=3, connector=connector)

        assert len(patches) > 0  # "good" source still contributed patches

    def test_no_sources_returns_an_empty_result(self):
        patches, labels = zap.fetch_and_extract([], connector=_FakeConnector())
        assert len(patches) == 0
        assert len(labels) == 0
