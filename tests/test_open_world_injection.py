"""open_world_injection.py: real-curve splicing of diffusion-sampled
morphology patches (backlog item 14)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import diffusion_train as diff
from astra import evaluate
from astra import open_world_injection as owi

torch = pytest.importorskip("torch", reason="PyTorch not installed")


def fake_sequences(n=64, length=64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 4 * np.pi, length)
    values = np.stack([
        np.sin(time * rng.uniform(0.5, 2.0) + rng.uniform(0, np.pi))
        + rng.normal(0, 0.05, length)
        for _ in range(n)
    ]).astype(np.float32)
    mask = np.ones((n, length), dtype=np.float32)
    return np.stack([values, mask], axis=1)


@pytest.fixture(scope="module")
def tiny_generator(tmp_path_factory):
    """A tiny trained denoiser, reused across this file's tests."""
    patches = fake_sequences(64, 16)  # (2, length) patches, length=16
    cfg = diff.DiffusionConfig(patch_length=16, channels=(16, 32), time_embed_dim=16,
                               timesteps=15, epochs=6, batch_size=16,
                               effective_batch_size=16, patience=20)
    report = diff.train_diffusion(patches[:48], patches[48:], cfg,
                                  tmp_path_factory.mktemp("diff"), "t")
    model, _ = diff.load_diffusion_model(report.checkpoint)
    return model, cfg


class TestExtractRealPatches:
    def _write_curve(self, object_id: str, n=200, seed=0):
        from astra import store
        from astra.surveys.base import LightCurve, SourceRef

        rng = np.random.default_rng(seed)
        source = SourceRef(survey="ZTF", object_id=object_id, ra_deg=180.0, dec_deg=20.0)
        time = 2458000.0 + np.cumsum(rng.uniform(0.5, 2.0, n))
        value = 18.0 + rng.normal(0, 0.2, n)
        curve = LightCurve(source=source, release="dr24", band="g", value_kind="mag",
                           time=time, value=value, value_err=np.full(n, 0.02))
        store.write_curve(curve)
        return curve

    def test_extracts_real_patches_of_the_requested_shape(self, isolated_root):
        for i in range(5):
            self._write_curve(f"obj{i}", seed=i)

        patches = owi.extract_real_patches(survey="ZTF", patch_length=16,
                                           sequence_length=64, limit=10)

        assert patches.shape[1:] == (2, 16)
        assert patches.shape[0] > 0
        assert np.all(np.isfinite(patches))

    def test_excluded_object_ids_are_never_used(self, isolated_root):
        for i in range(3):
            self._write_curve(f"obj{i}", seed=i)

        patches = owi.extract_real_patches(
            survey="ZTF", patch_length=16, sequence_length=64, limit=10,
            exclude_object_ids={"obj0", "obj1", "obj2"})

        assert len(patches) == 0

    def test_empty_store_returns_an_empty_array_not_an_error(self, isolated_root):
        patches = owi.extract_real_patches(survey="ZTF", patch_length=16,
                                           sequence_length=64, limit=10)
        assert patches.shape == (0, 2, 16)


class TestSplicePatch:
    def test_adds_the_patch_within_its_span(self):
        sequence = np.zeros((2, 20), dtype=np.float32)
        sequence[1, :] = 1.0  # fully valid
        patch = np.full(5, 3.0, dtype=np.float32)
        out = owi._splice_patch(sequence, patch, start=10, strength=1.0)
        assert np.allclose(out[0, 10:15], 3.0)
        assert np.allclose(out[0, :10], 0.0)

    def test_respects_the_validity_mask(self):
        sequence = np.zeros((2, 20), dtype=np.float32)
        sequence[1, :10] = 0.0  # first half unobserved
        sequence[1, 10:] = 1.0
        patch = np.full(20, 5.0, dtype=np.float32)
        out = owi._splice_patch(sequence, patch, start=0, strength=1.0)
        assert np.all(out[0, :10] == 0.0)

    def test_patch_extending_past_the_sequence_end_is_truncated(self):
        sequence = np.zeros((2, 10), dtype=np.float32)
        sequence[1, :] = 1.0
        patch = np.full(8, 2.0, dtype=np.float32)
        out = owi._splice_patch(sequence, patch, start=7, strength=1.0)
        assert out.shape == (2, 10)
        assert np.all(np.isfinite(out))


class TestInjectGenerative:
    def test_respects_the_validity_mask(self, tiny_generator):
        model, cfg = tiny_generator
        rng = np.random.default_rng(0)
        sequence = fake_sequences(1, 64)[0]
        sequence[1, :32] = 0.0  # first half unobserved

        modified = owi.inject_generative(sequence, model, cfg, rng, strength=1.0)

        assert np.all(modified[0, :32] == 0.0)

    def test_output_shape_matches_input(self, tiny_generator):
        model, cfg = tiny_generator
        rng = np.random.default_rng(1)
        sequence = fake_sequences(1, 64)[0]
        modified = owi.inject_generative(sequence, model, cfg, rng)
        assert modified.shape == sequence.shape

    def test_changes_the_sequence(self, tiny_generator):
        model, cfg = tiny_generator
        rng = np.random.default_rng(2)
        sequence = fake_sequences(1, 64)[0]
        modified = owi.inject_generative(sequence, model, cfg, rng, strength=3.0)
        assert not np.allclose(modified[0], sequence[0])


class TestBuildInjectedOpenWorld:
    def test_labels_the_right_number_of_rows(self, tiny_generator):
        model, cfg = tiny_generator
        data = fake_sequences(100, 64)
        result = owi.build_injected_open_world(data, [{}] * 100, model, cfg, fraction=0.1, seed=3)
        assert result.labels.sum() == 10
        assert len(result) == 100

    def test_untouched_rows_are_unchanged(self, tiny_generator):
        model, cfg = tiny_generator
        data = fake_sequences(50, 64)
        result = owi.build_injected_open_world(data, [{}] * 50, model, cfg, fraction=0.1, seed=3)
        for index in np.where(result.labels == 0)[0]:
            np.testing.assert_array_equal(result.values[index], data[index])

    def test_injected_rows_are_marked_generative(self, tiny_generator):
        model, cfg = tiny_generator
        data = fake_sequences(30, 64)
        result = owi.build_injected_open_world(data, [{}] * 30, model, cfg, fraction=0.2, seed=4)
        for index in np.where(result.labels == 1)[0]:
            assert result.kinds[index] == "generative"

    def test_empty_input_does_not_crash(self, tiny_generator):
        model, cfg = tiny_generator
        result = owi.build_injected_open_world(
            np.empty((0, 2, 64)), [], model, cfg, fraction=0.1)
        assert len(result) == 0

    def test_result_is_accepted_unmodified_by_compare_on_sequences(self, tiny_generator):
        """Confirms the InjectionResult this module builds is exactly what
        evaluate.compare_on_sequences already expects -- no fork needed."""
        model, cfg = tiny_generator
        data = fake_sequences(40, 64)
        injection = owi.build_injected_open_world(
            data, [{}] * 40, model, cfg, fraction=0.2, seed=5)

        comparison = evaluate.compare_on_sequences(injection, include_deep=False)

        assert len(comparison.methods) > 0
        assert comparison.injection["injected"] == 8
