"""Masked self-supervised light-curve pretraining (pretrain.py).

These tests run on whatever device is available; kept deliberately tiny and
fast, the same discipline test_deep.py already documents for this machine's
4 GB GTX 1650.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import evaluate, pretrain, pretrain_probe

torch = pytest.importorskip("torch", reason="PyTorch not installed")


def fake_sequences(n=64, length=64, seed=0) -> np.ndarray:
    """Smooth normalised curves with full validity masks."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 4 * np.pi, length)
    values = np.stack([
        np.sin(time * rng.uniform(0.5, 2.0) + rng.uniform(0, np.pi))
        + rng.normal(0, 0.05, length)
        for _ in range(n)
    ]).astype(np.float32)
    mask = np.ones((n, length), dtype=np.float32)
    return np.stack([values, mask], axis=1)


class TestSpanMasking:
    def test_masked_positions_were_previously_valid(self):
        data = fake_sequences(16, 64)
        rng = np.random.default_rng(0)
        _, pretrain_mask = pretrain.apply_span_mask(data, rng, mask_ratio=0.2)
        assert np.all(pretrain_mask[data[:, 1, :] == 0] == 0)

    def test_never_hides_an_already_invalid_gap_point(self):
        data = fake_sequences(8, 64)
        data[:, 1, :32] = 0.0  # first half unobserved (interpolated gap)
        rng = np.random.default_rng(1)
        _, pretrain_mask = pretrain.apply_span_mask(data, rng, mask_ratio=0.3)
        assert np.all(pretrain_mask[:, :32] == 0)

    def test_hidden_fraction_approximates_the_requested_ratio(self):
        data = fake_sequences(32, 256)
        rng = np.random.default_rng(2)
        _, pretrain_mask = pretrain.apply_span_mask(
            data, rng, mask_ratio=0.15, mean_span_length=8)
        fraction = pretrain_mask.sum() / (data.shape[0] * data.shape[-1])
        assert 0.07 < fraction < 0.25

    def test_masked_value_is_zeroed_at_hidden_positions(self):
        data = fake_sequences(16, 64)
        rng = np.random.default_rng(3)
        masked, pretrain_mask = pretrain.apply_span_mask(data, rng, mask_ratio=0.2)
        assert np.all(masked[:, 0, :][pretrain_mask == 1] == 0.0)

    def test_row_with_too_few_valid_points_is_skipped_not_degraded(self):
        data = fake_sequences(4, 64)
        data[:, 1, :] = 0.0
        data[:, 1, :3] = 1.0  # fewer than MIN_VALID_FOR_MASK
        rng = np.random.default_rng(4)
        _, pretrain_mask = pretrain.apply_span_mask(data, rng, mask_ratio=0.5)
        assert np.all(pretrain_mask == 0)

    def test_gap_mask_channel_is_left_untouched_by_span_masking(self):
        data = fake_sequences(16, 64)
        rng = np.random.default_rng(5)
        masked, _ = pretrain.apply_span_mask(data, rng, mask_ratio=0.2)
        np.testing.assert_array_equal(masked[:, 1, :], data[:, 1, :])


class TestOrderTask:
    def test_shuffled_rows_are_labelled_one_unshuffled_labelled_zero(self):
        data = fake_sequences(20, 64)
        rng = np.random.default_rng(0)
        _, labels = pretrain.make_order_task(
            data, rng, patch=8, shuffle_fraction=0.5, num_swaps=2)
        assert set(np.unique(labels)).issubset({0.0, 1.0})
        assert labels.sum() > 0

    def test_shuffle_fraction_is_approximately_honoured(self):
        data = fake_sequences(100, 64)
        rng = np.random.default_rng(1)
        _, labels = pretrain.make_order_task(
            data, rng, patch=8, shuffle_fraction=0.5, num_swaps=2)
        assert 30 <= labels.sum() <= 70

    def test_value_and_gap_mask_channels_are_swapped_together(self):
        data = fake_sequences(20, 64)
        rng = np.random.default_rng(2)
        patch = 8
        out, labels = pretrain.make_order_task(
            data, rng, patch=patch, shuffle_fraction=1.0, num_swaps=2)
        token_count = data.shape[-1] // patch

        for row in np.where(labels == 1)[0]:
            for token in range(token_count):
                block = out[row, :, token * patch:(token + 1) * patch]
                matches = [
                    np.array_equal(block, data[row, :, j * patch:(j + 1) * patch])
                    for j in range(token_count)
                ]
                assert any(matches), "every patch must come intact from some original patch"

    def test_adjacent_patches_are_never_swapped(self):
        data = fake_sequences(30, 64)
        rng = np.random.default_rng(3)
        patch = 8
        out, labels = pretrain.make_order_task(
            data, rng, patch=patch, shuffle_fraction=1.0, num_swaps=1)
        token_count = data.shape[-1] // patch

        for row in np.where(labels == 1)[0]:
            changed = [
                token for token in range(token_count)
                if not np.array_equal(
                    out[row, :, token * patch:(token + 1) * patch],
                    data[row, :, token * patch:(token + 1) * patch])
            ]
            assert len(changed) == 2
            assert abs(changed[0] - changed[1]) >= 2

    def test_unshuffled_rows_are_returned_bit_identical(self):
        data = fake_sequences(10, 64)
        rng = np.random.default_rng(4)
        out, labels = pretrain.make_order_task(
            data, rng, patch=8, shuffle_fraction=0.0, num_swaps=2)
        np.testing.assert_array_equal(out, data)
        assert np.all(labels == 0)


class TestEncoderAndModel:
    def test_encoder_output_token_count_matches_patchified_length(self):
        config = pretrain.PretrainConfig(length=64, patch_size=8, transformer_dim=32)
        encoder = pretrain.make_encoder(config)
        tokens = encoder(torch.from_numpy(fake_sequences(4, 64)))
        assert tokens.shape[1] == 8

    def test_pooled_embedding_has_transformer_dim_width(self):
        config = pretrain.PretrainConfig(length=64, patch_size=8, transformer_dim=32)
        encoder = pretrain.make_encoder(config)
        pooled = encoder.pooled(torch.from_numpy(fake_sequences(4, 64)))
        assert pooled.shape == (4, 32)

    def test_pretrain_model_reconstruction_preserves_sequence_length(self):
        config = pretrain.PretrainConfig(length=64, patch_size=8, transformer_dim=32)
        model = pretrain.make_pretrain_model(config)
        reconstruction, _ = model(torch.from_numpy(fake_sequences(4, 64)))
        assert reconstruction.shape == (4, 64)

    def test_pretrain_model_order_logit_is_one_per_row(self):
        config = pretrain.PretrainConfig(length=64, patch_size=8, transformer_dim=32)
        model = pretrain.make_pretrain_model(config)
        _, order_logit = model(torch.from_numpy(fake_sequences(4, 64)))
        assert order_logit.shape == (4,)

    def test_transformer_dim_not_divisible_by_heads_is_rejected(self):
        with pytest.raises(ValueError):
            pretrain.make_encoder(
                pretrain.PretrainConfig(transformer_dim=10, transformer_heads=4))


class TestSpanReconstructionLoss:
    def test_loss_ignores_points_outside_pretrain_mask(self):
        prediction = torch.zeros(2, 10)
        target = torch.ones(2, 10)
        gap_mask = torch.ones(2, 10)
        pretrain_mask = torch.zeros(2, 10)
        pretrain_mask[:, :5] = 1.0

        loss = pretrain.span_reconstruction_loss(prediction, target, pretrain_mask, gap_mask)

        assert float(loss) == pytest.approx(1.0)

    def test_loss_ignores_points_where_gap_mask_is_zero_even_if_pretrain_mask_is_one(self):
        prediction = torch.zeros(1, 10)
        target = torch.ones(1, 10)
        pretrain_mask = torch.ones(1, 10)
        gap_mask = torch.zeros(1, 10)
        gap_mask[:, :4] = 1.0

        loss = pretrain.span_reconstruction_loss(prediction, target, pretrain_mask, gap_mask)

        assert float(loss) == pytest.approx(1.0)

    def test_perfect_reconstruction_is_zero_loss(self):
        target = torch.randn(3, 10)
        mask = torch.ones(3, 10)
        loss = pretrain.span_reconstruction_loss(target, target, mask, mask)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_fully_masked_row_does_not_divide_by_zero(self):
        loss = pretrain.span_reconstruction_loss(
            torch.zeros(1, 10), torch.ones(1, 10), torch.zeros(1, 10), torch.zeros(1, 10))
        assert torch.isfinite(loss)


class TestPretrainLoop:
    def _config(self, **overrides) -> pretrain.PretrainConfig:
        defaults = dict(length=64, patch_size=8, transformer_dim=32,
                        transformer_heads=4, transformer_layers=1,
                        epochs=15, batch_size=16, span_mask_ratio=0.2)
        defaults.update(overrides)
        return pretrain.PretrainConfig(**defaults)

    def test_pretraining_reduces_the_combined_loss(self, tmp_path):
        data = fake_sequences(96, 64)
        cfg = self._config()

        report = pretrain.pretrain(data[:80], data[80:], cfg, tmp_path, "t")

        assert report.train_losses[-1] < report.train_losses[0]
        assert report.best_epoch >= 0

    def test_report_tracks_reconstruction_and_order_losses_separately(self, tmp_path):
        data = fake_sequences(64, 64)
        cfg = self._config(epochs=4)

        report = pretrain.pretrain(data[:48], data[48:], cfg, tmp_path, "t")

        assert len(report.reconstruction_losses) == report.epochs_run
        assert len(report.order_losses) == report.epochs_run
        assert all(np.isfinite(v) for v in report.reconstruction_losses)
        assert all(np.isfinite(v) for v in report.order_losses)

    def test_checkpoint_round_trips_via_load_pretrained_encoder(self, tmp_path):
        data = fake_sequences(64, 64)
        cfg = self._config(epochs=4)

        report = pretrain.pretrain(data[:48], data[48:], cfg, tmp_path, "ckpt")
        encoder, saved = pretrain.load_pretrained_encoder(report.checkpoint)

        pooled = encoder.pooled(torch.from_numpy(data[:4]))
        assert pooled.shape == (4, cfg.transformer_dim)
        assert saved["length"] == cfg.length

    def test_early_stopping_halts_a_stalled_run(self, tmp_path):
        data = np.repeat(fake_sequences(1, 64), 40, axis=0)
        cfg = self._config(epochs=100, patience=2)

        report = pretrain.pretrain(data[:32], data[32:], cfg, tmp_path, "t")

        assert report.epochs_run < 100

    def test_training_is_reproducible(self, tmp_path):
        data = fake_sequences(48, 64)
        cfg = self._config(epochs=3, seed=13)

        first = pretrain.pretrain(data[:40], data[40:], cfg, tmp_path, "a")
        second = pretrain.pretrain(data[:40], data[40:], cfg, tmp_path, "b")

        assert first.train_losses[0] == pytest.approx(second.train_losses[0], rel=1e-4)

    def test_effective_batch_is_held_by_accumulation(self, tmp_path):
        data = fake_sequences(32, 64)
        cfg = self._config(epochs=1, batch_size=8, effective_batch_size=64)

        report = pretrain.pretrain(data[:24], data[24:], cfg, tmp_path, "t")

        assert report.accumulation_steps == 8


class TestProbeTransfer:
    def _encoder(self):
        config = pretrain.PretrainConfig(length=64, patch_size=8,
                                         transformer_dim=16, transformer_heads=2,
                                         transformer_layers=1)
        return pretrain.make_encoder(config)

    def test_returns_all_three_requested_fractions(self):
        values = fake_sequences(200, 64)
        injection = evaluate.build_injected(values, [{}] * 200, fraction=0.3, seed=1)

        results = pretrain_probe.probe_transfer(
            self._encoder(), injection.values, injection.labels,
            fractions=(0.01, 0.10, 1.0), seed=2, n_repeats=3)

        assert set(results.keys()) == {0.01, 0.10, 1.0}

    def test_baseline_and_pretrained_arms_are_both_present_at_every_fraction(self):
        values = fake_sequences(200, 64)
        injection = evaluate.build_injected(values, [{}] * 200, fraction=0.3, seed=1)

        results = pretrain_probe.probe_transfer(
            self._encoder(), injection.values, injection.labels,
            fractions=(0.01, 0.10, 1.0), seed=2, n_repeats=3)

        for fraction in (0.01, 0.10, 1.0):
            assert results[fraction]["pretrained"] is not None
            assert results[fraction]["baseline"] is not None

    def test_larger_label_fraction_does_not_degrade_reported_auc_on_average(self):
        values = fake_sequences(300, 64)
        injection = evaluate.build_injected(values, [{}] * 300, fraction=0.3, seed=7)

        results = pretrain_probe.probe_transfer(
            self._encoder(), injection.values, injection.labels,
            fractions=(0.01, 1.0), seed=3, n_repeats=5)

        small = results[0.01]["baseline"]["mean"]
        full = results[1.0]["baseline"]["mean"]
        assert full >= small - 0.15

    def test_fraction_too_small_to_include_a_positive_is_skipped_not_crashed(self):
        values = fake_sequences(50, 64)
        labels = np.zeros(50, dtype=int)  # no positive class present at all

        results = pretrain_probe.probe_transfer(
            self._encoder(), values, labels, fractions=(0.1,), seed=0, n_repeats=3)

        assert results[0.1]["pretrained"] is None
        assert results[0.1]["baseline"] is None
        assert results[0.1]["skipped_repeats"] == 3

    def test_probe_uses_a_fixed_held_out_test_set_across_fractions(self):
        first_train, first_test = pretrain_probe._train_test_split_indices(
            200, seed=5, test_fraction=0.3)
        second_train, second_test = pretrain_probe._train_test_split_indices(
            200, seed=5, test_fraction=0.3)

        np.testing.assert_array_equal(first_test, second_test)
        np.testing.assert_array_equal(first_train, second_train)


class TestUnwiredFromProduction:
    def test_pretrain_modules_are_not_imported_by_rpc(self):
        from pathlib import Path

        rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
        assert "pretrain" not in rpc_source
