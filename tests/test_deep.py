"""Models, training loop and the method comparison.

These tests run on whatever device is available; on this machine that is a
4 GB GTX 1650, so everything here is deliberately tiny and fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import evaluate, models, train

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


def fake_irregular_sequences(n=64, length=64, seed=0) -> np.ndarray:
    """3-channel (value, mask, scaled time-delta) batch, tensors.py's
    "irregular" mode shape -- what make_neural_ode expects."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 4 * np.pi, length)
    values = np.stack([
        np.sin(time * rng.uniform(0.5, 2.0) + rng.uniform(0, np.pi))
        + rng.normal(0, 0.05, length)
        for _ in range(n)
    ]).astype(np.float32)
    mask = np.ones((n, length), dtype=np.float32)
    dt = np.tile(np.diff(time, prepend=time[0]).astype(np.float32), (n, 1))
    return np.stack([values, mask, dt], axis=1)


class TestModels:
    @pytest.mark.parametrize("kind", ["autoencoder", "vae"])
    def test_forward_preserves_sequence_length(self, kind):
        config = models.ModelConfig(length=64)
        model = models.make(kind, config)
        out, _, _ = model(torch.from_numpy(fake_sequences(4, 64)))
        assert out.shape[-1] == 64

    def test_vae_returns_distribution_parameters(self):
        model = models.make("vae", models.ModelConfig(length=64))
        _, mu, logvar = model(torch.from_numpy(fake_sequences(4, 64)))
        assert mu is not None and logvar is not None
        assert mu.shape[1] == 16

    def test_autoencoder_returns_no_distribution(self):
        model = models.make("autoencoder", models.ModelConfig(length=64))
        _, mu, logvar = model(torch.from_numpy(fake_sequences(4, 64)))
        assert mu is None and logvar is None

    def test_patch_transformer_preserves_sequence_length(self):
        model = models.make("transformer", models.ModelConfig(length=64, patch_size=8))
        out, _, _ = model(torch.from_numpy(fake_sequences(4, 64)))
        assert out.shape[-1] == 64

    def test_neural_ode_preserves_sequence_length(self):
        config = models.ModelConfig(length=32, ode_hidden_dim=16, ode_steps=2)
        model = models.make("neural_ode", config)
        out, mu, logvar = model(torch.from_numpy(fake_irregular_sequences(4, 32)))
        assert out.shape == (4, 1, 32)
        assert mu is None and logvar is None

    def test_neural_ode_gradients_reach_every_parameter(self):
        config = models.ModelConfig(length=16, ode_hidden_dim=8, ode_steps=1)
        model = models.make("neural_ode", config)
        batch = torch.from_numpy(fake_irregular_sequences(2, 16))
        out, _, _ = model(batch)
        loss = models.masked_reconstruction_loss(out, batch[:, 0, :], batch[:, 1, :])
        loss.backward()
        assert all(p.grad is not None and torch.any(p.grad != 0)
                  for p in model.parameters())

    def test_unknown_model_is_rejected(self):
        with pytest.raises(KeyError, match="available"):
            models.make("unknown")

    def test_default_model_fits_the_vram_budget(self):
        """Sized for ~2.2 GB usable, so keep the parameter count modest."""
        count = models.count_parameters(models.make("autoencoder"))
        assert count < 2_000_000

    def test_logvar_is_clamped_against_float16_overflow(self):
        """Turing has no bfloat16, so float16 overflow is a real failure mode."""
        model = models.make("vae", models.ModelConfig(length=64))
        extreme = torch.from_numpy(fake_sequences(4, 64)) * 1000.0
        _, _, logvar = model(extreme)
        assert torch.all(logvar <= 10.0) and torch.all(logvar >= -10.0)


class TestLosses:
    def test_masked_loss_ignores_unobserved_points(self):
        prediction = torch.zeros(2, 1, 10)
        target = torch.ones(2, 10)
        mask = torch.zeros(2, 10)
        mask[:, :5] = 1.0

        loss = models.masked_reconstruction_loss(prediction, target, mask)

        assert float(loss) == pytest.approx(1.0)  # only the observed half counts

    def test_fully_masked_row_does_not_divide_by_zero(self):
        loss = models.masked_reconstruction_loss(
            torch.zeros(1, 1, 10), torch.ones(1, 10), torch.zeros(1, 10))
        assert torch.isfinite(loss)

    def test_perfect_reconstruction_is_zero_loss(self):
        target = torch.randn(3, 10)
        loss = models.masked_reconstruction_loss(
            target.unsqueeze(1), target, torch.ones(3, 10))
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_kl_of_standard_normal_is_zero(self):
        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)
        assert float(models.kl_divergence(mu, logvar)) == pytest.approx(0.0)

    def test_kl_grows_as_the_posterior_moves(self):
        near = models.kl_divergence(torch.full((4, 8), 0.1), torch.zeros(4, 8))
        far = models.kl_divergence(torch.full((4, 8), 3.0), torch.zeros(4, 8))
        assert float(far) > float(near)


class TestBatchSizing:
    def test_explicit_request_is_honoured(self):
        assert train.choose_batch_size(256, requested=32) == 32

    def test_chosen_size_is_within_sane_bounds(self):
        size = train.choose_batch_size(256)
        assert 8 <= size <= 256

    def test_longer_sequences_do_not_increase_the_batch(self):
        assert train.choose_batch_size(1024) <= train.choose_batch_size(64)


class TestTraining:
    @pytest.mark.parametrize("kind", ["autoencoder", "vae"])
    def test_training_reduces_the_loss(self, kind, tmp_path):
        data = fake_sequences(96, 64)
        cfg = train.TrainConfig(kind=kind, epochs=12, batch_size=16,
                                model=models.ModelConfig(length=64))

        report = train.train(data[:80], data[80:], cfg, tmp_path, "t")

        assert report.train_losses[-1] < report.train_losses[0]
        assert report.best_epoch >= 0

    def test_neural_ode_trains_via_the_shared_loop(self, tmp_path):
        """train.py needs no kind-specific branch for this model: it returns
        (prediction, None, None) exactly like the autoencoder does."""
        data = fake_irregular_sequences(48, 16, seed=1)
        cfg = train.TrainConfig(
            kind="neural_ode", epochs=8, batch_size=16,
            model=models.ModelConfig(length=16, ode_hidden_dim=12, ode_steps=1))

        report = train.train(data[:40], data[40:], cfg, tmp_path, "ode")

        assert report.train_losses[-1] < report.train_losses[0]
        assert report.best_epoch >= 0

    def test_report_records_the_device_and_reason(self, tmp_path):
        data = fake_sequences(32, 64)
        cfg = train.TrainConfig(epochs=2, batch_size=8,
                                model=models.ModelConfig(length=64))

        report = train.train(data[:24], data[24:], cfg, tmp_path, "t")

        assert report.device in {"cpu", "cuda"}
        assert report.device_reason
        assert report.parameters > 0

    def test_effective_batch_is_held_by_accumulation(self, tmp_path):
        """Results must stay comparable between this GPU and a rented one."""
        data = fake_sequences(32, 64)
        cfg = train.TrainConfig(epochs=1, batch_size=8,
                                effective_batch_size=64,
                                model=models.ModelConfig(length=64))

        report = train.train(data[:24], data[24:], cfg, tmp_path, "t")

        assert report.accumulation_steps == 8

    def test_checkpoint_round_trips(self, tmp_path):
        data = fake_sequences(48, 64)
        cfg = train.TrainConfig(epochs=3, batch_size=8,
                                model=models.ModelConfig(length=64))

        report = train.train(data[:40], data[40:], cfg, tmp_path, "ckpt")
        model, saved = train.load_model(report.checkpoint)

        assert saved["kind"] == "autoencoder"
        out, _, _ = model(torch.from_numpy(data[:2]))
        assert out.shape[-1] == 64

    def test_early_stopping_halts_a_stalled_run(self, tmp_path):
        # Identical rows: the model converges almost immediately.
        data = np.repeat(fake_sequences(1, 64), 40, axis=0)
        cfg = train.TrainConfig(epochs=100, batch_size=8, patience=2,
                                model=models.ModelConfig(length=64))

        report = train.train(data[:32], data[32:], cfg, tmp_path, "t")

        assert report.epochs_run < 100

    def test_training_is_reproducible(self, tmp_path):
        data = fake_sequences(48, 64)
        cfg = train.TrainConfig(epochs=3, batch_size=8, seed=13,
                                model=models.ModelConfig(length=64))

        first = train.train(data[:40], data[40:], cfg, tmp_path, "a")
        second = train.train(data[:40], data[40:], cfg, tmp_path, "b")

        assert first.train_losses[0] == pytest.approx(second.train_losses[0],
                                                      rel=1e-5)

    def test_reconstruction_scores_flag_an_odd_row(self, tmp_path):
        data = fake_sequences(64, 64)
        cfg = train.TrainConfig(epochs=15, batch_size=16,
                                model=models.ModelConfig(length=64))
        report = train.train(data[:56], data[56:], cfg, tmp_path, "t")
        model, _ = train.load_model(report.checkpoint)

        odd = data.copy()
        odd[0, 0, :] = np.random.default_rng(9).normal(0, 5, 64)  # pure noise

        scores = train.reconstruction_scores(model, odd)

        assert scores[0] > np.median(scores)


class TestInjection:
    @pytest.mark.parametrize("kind", evaluate.ANOMALY_KINDS)
    def test_each_anomaly_kind_changes_the_curve(self, kind):
        rng = np.random.default_rng(0)
        original = fake_sequences(1, 128)[0]
        modified = evaluate.inject(original, kind, rng)

        assert not np.allclose(original[0], modified[0])

    def test_injection_respects_the_validity_mask(self):
        rng = np.random.default_rng(0)
        sequence = fake_sequences(1, 128)[0]
        sequence[1, :64] = 0.0  # first half unobserved

        modified = evaluate.inject(sequence, "flare", rng)

        assert np.all(modified[0, :64] == 0.0)

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown anomaly"):
            evaluate.inject(fake_sequences(1, 64)[0], "supernova",
                            np.random.default_rng(0))

    def test_build_injected_labels_the_right_number(self):
        data = fake_sequences(100, 64)
        result = evaluate.build_injected(data, [{}] * 100, fraction=0.1)

        assert result.labels.sum() == 10
        assert len(result) == 100

    def test_untouched_rows_are_unchanged(self):
        data = fake_sequences(50, 64)
        result = evaluate.build_injected(data, [{}] * 50, fraction=0.1, seed=3)

        for index in np.where(result.labels == 0)[0]:
            np.testing.assert_array_equal(result.values[index], data[index])


class TestDeepMethodsDispatch:
    def test_two_channel_input_runs_the_three_original_kinds(self):
        values = fake_sequences(24, 16)
        labels = np.zeros(24, dtype=int)
        results = evaluate._deep_methods(values, labels, epochs=1, seed=0)
        assert {r.name for r in results} == {
            "deep_autoencoder", "deep_vae", "deep_transformer"}

    def test_three_channel_input_runs_only_neural_ode(self):
        values = fake_irregular_sequences(24, 16)
        labels = np.zeros(24, dtype=int)
        results = evaluate._deep_methods(values, labels, epochs=1, seed=0)
        assert {r.name for r in results} == {"deep_neural_ode"}


class TestScoring:
    def test_perfect_ranking_scores_one(self):
        labels = np.array([0] * 90 + [1] * 10)
        scores = labels.astype(float)

        result = evaluate.score_method("perfect", scores, labels)

        assert result.roc_auc == pytest.approx(1.0)
        assert result.precision_at_k == pytest.approx(1.0)

    def test_random_ranking_scores_near_a_half(self):
        rng = np.random.default_rng(0)
        labels = np.array([0] * 900 + [1] * 100)
        result = evaluate.score_method("random", rng.normal(size=1000), labels)

        assert 0.4 < result.roc_auc < 0.6

    def test_degenerate_labels_are_reported_not_raised(self):
        result = evaluate.score_method("x", np.zeros(10), np.zeros(10, dtype=int))
        assert np.isnan(result.roc_auc)
        assert "degenerate" in result.note

    def test_comparison_picks_the_best_by_auc(self):
        comparison = evaluate.Comparison(methods=[
            evaluate.MethodScore("a", 0.7, 0.5, 0.5, 0.5),
            evaluate.MethodScore("b", 0.9, 0.6, 0.6, 0.6),
        ])
        assert comparison.best().name == "b"

    def test_comparison_ignores_failed_methods(self):
        comparison = evaluate.Comparison(methods=[
            evaluate.MethodScore("ok", 0.7, 0.5, 0.5, 0.5),
            evaluate.MethodScore("failed", float("nan"), float("nan"),
                                 float("nan"), float("nan")),
        ])
        assert comparison.best().name == "ok"


class TestCudaOomGuard:
    """`train._cuda_oom_guard()` -- empty the allocator, add VRAM context.

    `choose_batch_size()` sizes batches from measured free VRAM, but that is
    an estimate: another process can consume VRAM between the check and the
    actual allocation. On the 4 GB card this suite runs on (about 2.2 GB
    genuinely free per this file's own docstring) that is a real, expected
    failure, not a hypothetical one. Before this guard existed, a caught OOM
    left the allocator fragmented for the rest of the long-lived engine
    session -- contained (not a process crash, dispatch()/jobs.py already
    catch any exception), but presenting as repeated failures after the
    first one.
    """

    def test_a_genuine_oom_empties_the_cache_and_adds_vram_context(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("emptied"))
        if torch.cuda.is_available():
            monkeypatch.setattr(torch.cuda, "mem_get_info",
                                lambda: (512 * 1024 ** 2, 4096 * 1024 ** 2))
        else:
            monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
            monkeypatch.setattr(torch.cuda, "mem_get_info",
                                lambda: (512 * 1024 ** 2, 4096 * 1024 ** 2))

        with pytest.raises(RuntimeError, match="CUDA ran out of memory"):
            with train._cuda_oom_guard():
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        assert calls == ["emptied"]

    def test_the_reraised_message_reports_free_and_total_vram(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info",
                            lambda: (512 * 1024 ** 2, 4096 * 1024 ** 2))

        with pytest.raises(RuntimeError) as excinfo:
            with train._cuda_oom_guard():
                raise RuntimeError("CUDA out of memory")

        assert "512 MB free" in str(excinfo.value)
        assert "4096 MB" in str(excinfo.value)

    def test_a_diagnostic_failure_does_not_hide_the_original_error(self, monkeypatch):
        """The VRAM report is a nicety; it must not become a NEW crash."""
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def broken():
            raise RuntimeError("driver not responding")

        monkeypatch.setattr(torch.cuda, "mem_get_info", broken)

        with pytest.raises(RuntimeError, match="CUDA ran out of memory"):
            with train._cuda_oom_guard():
                raise RuntimeError("CUDA out of memory")

    def test_an_unrelated_runtime_error_passes_through_unchanged(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("emptied"))

        with pytest.raises(RuntimeError, match="dimension mismatch"):
            with train._cuda_oom_guard():
                raise RuntimeError("dimension mismatch: expected 64, got 32")

        assert calls == []

    def test_an_oom_shaped_message_without_cuda_available_passes_through(self, monkeypatch):
        """The guard only applies its recovery when there is a CUDA device to
        recover -- on a CPU-only run the same wording means something else."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(RuntimeError, match="out of memory"):
            with train._cuda_oom_guard():
                raise RuntimeError("out of memory")

    def test_a_real_torch_out_of_memory_error_is_recognised_by_type(self, monkeypatch):
        """Not every OOM message contains the literal words "out of memory";
        the real exception type must be recognised on its own."""
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 1))
        error_cls = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)

        with pytest.raises(RuntimeError, match="CUDA ran out of memory"):
            with train._cuda_oom_guard():
                raise error_cls("allocator exhausted")

    def test_reconstruction_scores_is_guarded(self, tmp_path, monkeypatch):
        """The inference path is a second, independent OOM risk from
        training -- scoring a large batch after training already succeeded."""
        data = fake_sequences(8, 64)
        cfg = train.TrainConfig(epochs=1, batch_size=4,
                                model=models.ModelConfig(length=64))
        report = train.train(data[:6], data[6:], cfg, tmp_path, "t")
        model, _ = train.load_model(report.checkpoint)

        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("emptied"))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 1))

        real_forward = model.forward

        def oom_forward(*args, **kwargs):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(model, "forward", oom_forward)

        with pytest.raises(RuntimeError, match="CUDA ran out of memory"):
            train.reconstruction_scores(model, data)

        assert calls == ["emptied"]
        monkeypatch.setattr(model, "forward", real_forward)

    def test_cuda_initialization_oom_falls_back_to_cpu(self, tmp_path, monkeypatch):
        """A headroom check is only advisory: allocation can still OOM before
        the epoch guard is entered.  That failure must recover as CPU training.
        """
        class FakeDeviceReport:
            device = "cuda"
            reason = "CUDA device available"
            torch_available = True
            cuda_available = True
            gpu = None

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def to(self, device):
                if str(device) == "cuda":
                    raise RuntimeError("CUDA out of memory during initialization")
                return super().to(device)

            def forward(self, batch):
                return batch[:, :1, :] * self.weight, None, None

        monkeypatch.setattr(train.hardware, "select_device",
                            lambda: FakeDeviceReport())
        monkeypatch.setattr(train.models, "make", lambda *args, **kwargs: TinyModel())

        data = fake_sequences(8, 16)
        cfg = train.TrainConfig(epochs=1, batch_size=4,
                                model=models.ModelConfig(length=16))
        report = train.train(data[:6], data[6:], cfg, tmp_path, "fallback")

        assert report.device == "cpu"
        assert "initialization ran out of memory" in report.device_reason
        assert report.amp_enabled is False
