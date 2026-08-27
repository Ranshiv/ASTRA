"""diffusion.py: minimal 1-D DDPM for the open-world morphology generator
(backlog item 14)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import diffusion as diff
from astra import diffusion_train as dtrain

torch = pytest.importorskip("torch", reason="PyTorch not installed")


def fake_patches(n=64, length=32, seed=0) -> np.ndarray:
    """Smooth normalised patches with a full validity mask -- same
    convention `tests/test_pretrain.py`'s `fake_sequences` already uses."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 4 * np.pi, length)
    values = np.stack([
        np.sin(time * rng.uniform(0.5, 2.0) + rng.uniform(0, np.pi))
        + rng.normal(0, 0.05, length)
        for _ in range(n)
    ]).astype(np.float32)
    mask = np.ones((n, length), dtype=np.float32)
    return np.stack([values, mask], axis=1)


def _tiny_config(**overrides) -> diff.DiffusionConfig:
    defaults = dict(patch_length=32, channels=(16, 32), time_embed_dim=16,
                    timesteps=20, epochs=8, batch_size=16,
                    effective_batch_size=16, patience=20)
    defaults.update(overrides)
    return diff.DiffusionConfig(**defaults)


class TestBetaSchedule:
    def test_is_monotonically_increasing(self):
        betas = diff.linear_beta_schedule(50)
        assert np.all(np.diff(betas) > 0)

    def test_starts_and_ends_at_the_documented_bounds(self):
        betas = diff.linear_beta_schedule(50)
        assert betas[0] == pytest.approx(1e-4)
        assert betas[-1] == pytest.approx(0.02)

    def test_alphas_cumprod_is_monotonically_decreasing(self):
        constants = diff.diffusion_constants(50)
        assert np.all(np.diff(constants["alphas_cumprod"]) < 0)
        assert constants["alphas_cumprod"][0] < 1.0
        assert constants["alphas_cumprod"][-1] > 0.0


class TestForwardDiffusionSample:
    def test_output_shape_matches_input(self):
        constants = diff.diffusion_constants(50)
        x0 = np.random.default_rng(0).normal(size=(8, 32)).astype(np.float32)
        noise = np.random.default_rng(1).normal(size=(8, 32)).astype(np.float32)
        t = np.zeros(8, dtype=int)
        x_t = diff.forward_diffusion_sample(x0, t, noise, constants)
        assert x_t.shape == (8, 32)

    def test_at_t_zero_output_is_close_to_x0(self):
        """Near t=0, alpha_bar is close to 1 -- x_t should stay close to x0."""
        constants = diff.diffusion_constants(50)
        x0 = np.full((4, 16), 5.0, dtype=np.float32)
        noise = np.zeros((4, 16), dtype=np.float32)
        t = np.zeros(4, dtype=int)
        x_t = diff.forward_diffusion_sample(x0, t, noise, constants)
        np.testing.assert_allclose(x_t, x0, rtol=0.05)

    def test_at_large_t_signal_is_mostly_replaced_by_noise(self):
        constants = diff.diffusion_constants(50)
        x0 = np.full((4, 16), 5.0, dtype=np.float32)
        noise = np.zeros((4, 16), dtype=np.float32)  # zero noise isolates the signal term
        t = np.full(4, 49, dtype=int)
        x_t = diff.forward_diffusion_sample(x0, t, noise, constants)
        # At the final timestep, sqrt(alpha_bar) has shrunk close to 0, so
        # the signal contribution should be far smaller than x0 itself.
        assert np.all(np.abs(x_t) < np.abs(x0))


class TestMakeDenoiser:
    def test_output_shape(self):
        cfg = _tiny_config()
        model = diff.make_denoiser(cfg)
        x = torch.randn(4, 2, 32)
        t = torch.randint(0, cfg.timesteps, (4,))
        out = model(x, t)
        assert out.shape == (4, 1, 32)

    def test_gradients_reach_every_parameter(self):
        cfg = _tiny_config()
        model = diff.make_denoiser(cfg)
        x = torch.randn(2, 2, 32)
        t = torch.randint(0, cfg.timesteps, (2,))
        out = model(x, t)
        out.sum().backward()
        assert all(p.grad is not None for p in model.parameters())

    def test_different_timesteps_produce_different_output(self):
        """The timestep embedding must actually influence the forward pass."""
        cfg = _tiny_config()
        model = diff.make_denoiser(cfg)
        model.eval()
        x = torch.randn(2, 2, 32)
        with torch.no_grad():
            out_early = model(x, torch.zeros(2, dtype=torch.long))
            out_late = model(x, torch.full((2,), cfg.timesteps - 1, dtype=torch.long))
        assert not torch.allclose(out_early, out_late)


class TestConditioning:
    def test_unconditional_model_has_no_embedding_tables(self):
        model = diff.make_denoiser(_tiny_config())
        assert model.artifact_embed is None
        assert model.transient_embed is None

    def test_enabling_a_channel_creates_its_embedding_table(self):
        cfg = _tiny_config(n_artifact_classes=6, n_transient_classes=3)
        model = diff.make_denoiser(cfg)
        assert model.artifact_embed is not None
        assert model.artifact_embed.num_embeddings == 7  # +1 for "unspecified"
        assert model.transient_embed is not None
        assert model.transient_embed.num_embeddings == 4

    def test_artifact_class_changes_the_output(self):
        cfg = _tiny_config(n_artifact_classes=6)
        model = diff.make_denoiser(cfg)
        model.eval()
        x = torch.randn(4, 2, 32)
        t = torch.zeros(4, dtype=torch.long)
        with torch.no_grad():
            unspecified = model(x, t)
            specified = model(x, t, torch.full((4,), 1, dtype=torch.long))
        assert not torch.allclose(unspecified, specified)

    def test_transient_class_changes_the_output(self):
        cfg = _tiny_config(n_transient_classes=3)
        model = diff.make_denoiser(cfg)
        model.eval()
        x = torch.randn(4, 2, 32)
        t = torch.zeros(4, dtype=torch.long)
        with torch.no_grad():
            unspecified = model(x, t)
            specified = model(x, t, None, torch.full((4,), 2, dtype=torch.long))
        assert not torch.allclose(unspecified, specified)

    def test_the_two_channels_are_independent(self):
        cfg = _tiny_config(n_artifact_classes=6, n_transient_classes=3)
        model = diff.make_denoiser(cfg)
        model.eval()
        x = torch.randn(4, 2, 32)
        t = torch.zeros(4, dtype=torch.long)
        artifact_only = torch.full((4,), 2, dtype=torch.long)
        transient_only = torch.full((4,), 1, dtype=torch.long)
        with torch.no_grad():
            out_a = model(x, t, artifact_only, None)
            out_b = model(x, t, None, transient_only)
            out_both = model(x, t, artifact_only, transient_only)
        # Changing one channel while holding the other fixed must not
        # collapse to the same output as changing the other channel alone.
        assert not torch.allclose(out_a, out_b)
        assert not torch.allclose(out_both, out_a)
        assert not torch.allclose(out_both, out_b)

    def test_train_diffusion_with_conditioning_labels_reduces_the_loss(self, tmp_path):
        patches = fake_patches(96, 32)
        artifact_labels = np.random.default_rng(0).integers(-1, 6, size=96)
        cfg = _tiny_config(epochs=10, n_artifact_classes=6)

        report = dtrain.train_diffusion(
            patches[:80], patches[80:], cfg, tmp_path, "cond",
            train_artifact_labels=artifact_labels[:80],
            val_artifact_labels=artifact_labels[80:])

        assert report.train_losses[-1] < report.train_losses[0]

    def test_conditioned_checkpoint_round_trips(self, tmp_path):
        patches = fake_patches(64, 32)
        labels = np.random.default_rng(1).integers(0, 6, size=64)
        cfg = _tiny_config(epochs=4, n_artifact_classes=6)

        report = dtrain.train_diffusion(
            patches[:48], patches[48:], cfg, tmp_path, "cond",
            train_artifact_labels=labels[:48], val_artifact_labels=labels[48:])
        model, saved = dtrain.load_diffusion_model(report.checkpoint)

        assert model.artifact_embed is not None
        assert saved["n_artifact_classes"] == 6

    def test_sample_with_disabled_channel_raises_a_clear_error(self, tmp_path):
        patches = fake_patches(64, 32)
        cfg = _tiny_config(epochs=3)
        report = dtrain.train_diffusion(patches[:48], patches[48:], cfg, tmp_path, "t")
        model, _ = dtrain.load_diffusion_model(report.checkpoint)

        with pytest.raises(ValueError, match="artifact conditioning"):
            dtrain.sample(model, 2, cfg, artifact_class=1)

    def test_sample_conditioned_on_artifact_class_matches_shape(self, tmp_path):
        patches = fake_patches(64, 32)
        labels = np.random.default_rng(2).integers(0, 6, size=64)
        cfg = _tiny_config(epochs=4, n_artifact_classes=6)
        report = dtrain.train_diffusion(
            patches[:48], patches[48:], cfg, tmp_path, "cond",
            train_artifact_labels=labels[:48], val_artifact_labels=labels[48:])
        model, _ = dtrain.load_diffusion_model(report.checkpoint)

        samples = dtrain.sample(model, 5, cfg, seed=3, artifact_class=2)
        assert samples.shape == (5, 32)
        assert np.all(np.isfinite(samples))


class TestDiffusionLoss:
    def test_ignores_points_outside_the_mask(self):
        predicted = torch.zeros(2, 1, 10)
        target = torch.ones(2, 10)
        mask = torch.zeros(2, 10)
        mask[:, :5] = 1.0
        loss = diff.diffusion_loss(predicted, target, mask)
        assert float(loss) == pytest.approx(1.0)

    def test_perfect_prediction_is_zero_loss(self):
        target = torch.randn(3, 10)
        mask = torch.ones(3, 10)
        loss = diff.diffusion_loss(target.unsqueeze(1), target, mask)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_fully_masked_row_does_not_divide_by_zero(self):
        loss = diff.diffusion_loss(torch.zeros(1, 1, 10), torch.ones(1, 10), torch.zeros(1, 10))
        assert torch.isfinite(loss)


class TestTrainDiffusion:
    def test_reduces_the_loss(self, tmp_path):
        patches = fake_patches(96, 32)
        cfg = _tiny_config(epochs=15)

        report = dtrain.train_diffusion(patches[:80], patches[80:], cfg, tmp_path, "t")

        assert report.train_losses[-1] < report.train_losses[0]
        assert report.best_epoch >= 0

    def test_no_nan_losses_with_amp_enabled(self, tmp_path):
        patches = fake_patches(64, 32)
        cfg = _tiny_config(epochs=6, amp=True)

        report = dtrain.train_diffusion(patches[:48], patches[48:], cfg, tmp_path, "t")

        assert all(np.isfinite(v) for v in report.train_losses)
        assert all(np.isfinite(v) for v in report.val_losses)

    def test_checkpoint_round_trips(self, tmp_path):
        patches = fake_patches(64, 32)
        cfg = _tiny_config(epochs=5)

        report = dtrain.train_diffusion(patches[:48], patches[48:], cfg, tmp_path, "ckpt")
        model, saved = dtrain.load_diffusion_model(report.checkpoint)

        x = torch.randn(2, 2, 32)
        t = torch.randint(0, cfg.timesteps, (2,))
        out = model(x, t)
        assert out.shape == (2, 1, 32)
        assert saved["patch_length"] == cfg.patch_length

    def test_early_stopping_halts_a_stalled_run(self, tmp_path):
        patches = np.repeat(fake_patches(1, 32), 40, axis=0)
        cfg = _tiny_config(epochs=100, patience=2)

        report = dtrain.train_diffusion(patches[:32], patches[32:], cfg, tmp_path, "t")

        assert report.epochs_run < 100

    def test_training_is_reproducible(self, tmp_path):
        patches = fake_patches(48, 32)
        cfg = _tiny_config(epochs=3, seed=13, amp=False)

        first = dtrain.train_diffusion(patches[:40], patches[40:], cfg, tmp_path, "a")
        second = dtrain.train_diffusion(patches[:40], patches[40:], cfg, tmp_path, "b")

        assert first.train_losses[0] == pytest.approx(second.train_losses[0], rel=1e-3)

    def test_effective_batch_is_held_by_accumulation(self, tmp_path):
        patches = fake_patches(32, 32)
        cfg = _tiny_config(epochs=1, batch_size=8, effective_batch_size=32)

        report = dtrain.train_diffusion(patches[:24], patches[24:], cfg, tmp_path, "t")

        assert report.accumulation_steps == 4


class TestSample:
    def test_returns_the_requested_shape(self, tmp_path):
        patches = fake_patches(64, 32)
        cfg = _tiny_config(epochs=5)
        report = dtrain.train_diffusion(patches[:48], patches[48:], cfg, tmp_path, "t")
        model, _ = dtrain.load_diffusion_model(report.checkpoint)

        samples = dtrain.sample(model, 6, cfg, seed=1)

        assert samples.shape == (6, 32)
        assert np.all(np.isfinite(samples))

    def test_reproducible_given_the_same_seed(self, tmp_path):
        patches = fake_patches(64, 32)
        cfg = _tiny_config(epochs=5)
        report = dtrain.train_diffusion(patches[:48], patches[48:], cfg, tmp_path, "t")
        model, _ = dtrain.load_diffusion_model(report.checkpoint)

        first = dtrain.sample(model, 4, cfg, seed=7)
        second = dtrain.sample(model, 4, cfg, seed=7)

        np.testing.assert_array_equal(first, second)

    def test_different_seeds_produce_different_samples(self, tmp_path):
        patches = fake_patches(64, 32)
        cfg = _tiny_config(epochs=5)
        report = dtrain.train_diffusion(patches[:48], patches[48:], cfg, tmp_path, "t")
        model, _ = dtrain.load_diffusion_model(report.checkpoint)

        first = dtrain.sample(model, 4, cfg, seed=7)
        second = dtrain.sample(model, 4, cfg, seed=8)

        assert not np.allclose(first, second)

    def test_custom_mask_is_honoured_in_shape(self, tmp_path):
        patches = fake_patches(64, 32)
        cfg = _tiny_config(epochs=3)
        report = dtrain.train_diffusion(patches[:48], patches[48:], cfg, tmp_path, "t")
        model, _ = dtrain.load_diffusion_model(report.checkpoint)

        mask = np.ones((3, 32), dtype=np.float32)
        mask[:, 16:] = 0.0
        samples = dtrain.sample(model, 3, cfg, mask=mask, seed=2)
        assert samples.shape == (3, 32)
        assert np.all(np.isfinite(samples))
