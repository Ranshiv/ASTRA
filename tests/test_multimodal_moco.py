"""multimodal_moco.py: MoCo momentum contrast across four modalities
(backlog item 11) -- the mechanism that resolves docs/DEFERRED.txt's
`[BLOCKED] Multimodal encoder ... and contrastive learning` entry.

Deliberately tiny configs throughout: this machine's real GPU (GTX 1650)
runs these tests, so every dimension is kept small for speed, matching
`tests/test_pretrain.py`'s discipline.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import multimodal_moco as moco
from astra import multimodal_synthetic as syn

torch = pytest.importorskip("torch", reason="PyTorch not installed")


def _tiny_config(**overrides) -> moco.MultimodalMoCoConfig:
    defaults = dict(
        embedding_dim=16, projection_dim=8, queue_size=32,
        lc_length=64, spectrum_length=64, image_input_size=16,
        lc_patch_size=8, spectrum_patch_size=8, catalog_hidden=32,
        epochs=8, batch_size=16, effective_batch_size=16, patience=20,
    )
    defaults.update(overrides)
    return moco.MultimodalMoCoConfig(**defaults)


def _tiny_batch(n=48, seed=1) -> syn.SyntheticMultimodalBatch:
    return syn.build_synthetic_pairs(n=n, seed=seed, lc_length=64,
                                     image_size=16, spectrum_length=64)


class TestEmbeddingQueue:
    def test_fills_up_to_capacity(self):
        queue = moco.EmbeddingQueue(dim=4, size=10)
        queue.enqueue(torch.randn(6, 4))
        assert queue.filled == 6
        queue.enqueue(torch.randn(6, 4))
        assert queue.filled == 10  # capped at size, not 12

    def test_wraparound_overwrites_the_oldest_entries(self):
        queue = moco.EmbeddingQueue(dim=2, size=4)
        queue.enqueue(torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
        queue.enqueue(torch.tensor([[4.0, 4.0], [5.0, 5.0]]))
        negatives = queue.negatives()
        assert negatives.shape[0] == 4
        # The two oldest (1,1) rows should have been overwritten; (4,4) and
        # (5,5) must be present.
        values = {tuple(row.tolist()) for row in negatives}
        assert (4.0, 4.0) in values and (5.0, 5.0) in values

    def test_negatives_returns_a_clone_not_a_view(self):
        """A view would corrupt an in-flight autograd graph the next time
        enqueue() mutates the buffer in place -- see the fix this exact bug
        required during development."""
        queue = moco.EmbeddingQueue(dim=2, size=4)
        queue.enqueue(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
        negatives = queue.negatives()
        queue.enqueue(torch.tensor([[9.0, 9.0]]))
        assert negatives[0, 0].item() == 1.0  # unaffected by the later enqueue

    def test_empty_queue_returns_zero_rows(self):
        queue = moco.EmbeddingQueue(dim=4, size=10)
        assert queue.negatives().shape == (0, 4)

    def test_enqueue_larger_than_capacity_keeps_the_most_recent(self):
        queue = moco.EmbeddingQueue(dim=1, size=3)
        queue.enqueue(torch.arange(10.0).unsqueeze(-1))
        negatives = queue.negatives()
        assert negatives.shape[0] == 3
        assert sorted(v.item() for v in negatives) == [7.0, 8.0, 9.0]


class TestUpdateMomentum:
    def test_ema_moves_toward_online_by_the_expected_amount(self):
        online = torch.nn.Linear(2, 2, bias=False)
        momentum = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            online.weight.fill_(1.0)
            momentum.weight.fill_(0.0)

        moco.update_momentum(online, momentum, m=0.9)
        # momentum <- 0.9*0 + 0.1*1 = 0.1
        assert torch.allclose(momentum.weight, torch.full((2, 2), 0.1))

    def test_online_parameters_are_unaffected(self):
        online = torch.nn.Linear(2, 2, bias=False)
        momentum = torch.nn.Linear(2, 2, bias=False)
        original = online.weight.clone()
        moco.update_momentum(online, momentum, m=0.5)
        assert torch.equal(online.weight, original)

    def test_momentum_parameters_never_require_grad(self):
        online = torch.nn.Linear(2, 2)
        momentum = torch.nn.Linear(2, 2)
        for p in momentum.parameters():
            p.requires_grad_(False)
        moco.update_momentum(online, momentum, m=0.9)
        assert all(not p.requires_grad for p in momentum.parameters())


class TestInfoNceLoss:
    def test_near_zero_when_query_equals_positive_and_negatives_are_orthogonal(self):
        query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
        positive = query.clone()
        negatives = torch.eye(3)[1:]  # orthogonal to the query
        loss = moco.info_nce_loss(query, positive, negatives, temperature=0.07)
        assert loss.item() < 0.01

    def test_larger_when_a_wrong_key_is_substituted(self):
        # With ZERO negatives, cross-entropy over a single-column logit is
        # always exactly 0 regardless of similarity (correctly -- there is
        # no alternative to be wrong about) -- real negatives are needed
        # for the loss to depend on query/key similarity at all.
        query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
        right_positive = query.clone()
        wrong_positive = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=-1)
        negatives = torch.eye(3)[1:]  # two orthogonal negatives

        right_loss = moco.info_nce_loss(query, right_positive, negatives, temperature=0.07)
        wrong_loss = moco.info_nce_loss(query, wrong_positive, negatives, temperature=0.07)
        assert wrong_loss.item() > right_loss.item()

    def test_works_with_zero_negatives(self):
        query = torch.randn(4, 8)
        query = torch.nn.functional.normalize(query, dim=-1)
        loss = moco.info_nce_loss(query, query.clone(), torch.zeros(0, 8))
        assert torch.isfinite(loss)


class TestMakeModel:
    def test_online_and_momentum_start_with_identical_weights(self):
        model = moco.make_model(_tiny_config())
        for kind in moco.KINDS:
            online_params = list(model.online[kind].parameters())
            momentum_params = list(model.momentum[kind].parameters())
            for p_online, p_momentum in zip(online_params, momentum_params):
                assert torch.equal(p_online, p_momentum)

    def test_momentum_parameters_are_frozen(self):
        model = moco.make_model(_tiny_config())
        assert all(not p.requires_grad for p in model.momentum.parameters())
        assert any(p.requires_grad for p in model.online.parameters())

    def test_encode_online_and_momentum_output_shapes(self):
        cfg = _tiny_config()
        model = moco.make_model(cfg)
        x = torch.randn(4, 40)
        scale = torch.randn(4)
        fused, projected = model.encode_online("catalog", x, scale)
        assert fused.shape == (4, cfg.embedding_dim)
        assert projected.shape == (4, cfg.projection_dim)
        # L2-normalised.
        assert torch.allclose(projected.norm(dim=-1), torch.ones(4), atol=1e-4)


class TestTrainMoco:
    def test_reduces_the_combined_loss_from_its_early_peak(self, tmp_path):
        """The queue starts empty, so the first few steps have a trivially
        easy (near-zero-negatives) loss; loss RISES as the queue fills and
        the task becomes genuinely hard, then falls as training proceeds --
        so this compares against the run's own peak, not epoch 0."""
        train = _tiny_batch(n=64, seed=1)
        val = _tiny_batch(n=16, seed=2)
        cfg = _tiny_config(epochs=15)

        report = moco.train_moco(train, val, cfg, tmp_path, "t")

        assert report.train_losses[-1] < max(report.train_losses)
        assert report.best_epoch >= 0

    def test_no_nan_losses_with_amp_enabled(self, tmp_path):
        """Regression test: an earlier version produced NaN losses under
        float16 autocast on this machine's real GPU (fixed by computing
        info_nce_loss in float32 and by normalising modality inputs to
        unit scale before the encoder, moving absolute brightness entirely
        into the scale token)."""
        train = _tiny_batch(n=32, seed=3)
        val = _tiny_batch(n=16, seed=4)
        cfg = _tiny_config(epochs=5, amp=True)

        report = moco.train_moco(train, val, cfg, tmp_path, "t")

        assert all(np.isfinite(v) for v in report.train_losses)
        assert all(np.isfinite(v) for v in report.val_losses)

    def test_report_tracks_each_pairs_loss_separately(self, tmp_path):
        train = _tiny_batch(n=32, seed=5)
        val = _tiny_batch(n=16, seed=6)
        cfg = _tiny_config(epochs=3)

        report = moco.train_moco(train, val, cfg, tmp_path, "t")

        assert set(report.pair_losses.keys()) == {
            "lightcurve_image", "lightcurve_spectrum", "lightcurve_catalog"}
        for values in report.pair_losses.values():
            assert len(values) == report.epochs_run

    def test_queue_fills_up_to_its_configured_size(self, tmp_path):
        train = _tiny_batch(n=64, seed=7)
        val = _tiny_batch(n=16, seed=8)
        cfg = _tiny_config(epochs=6, queue_size=20)

        report = moco.train_moco(train, val, cfg, tmp_path, "t")

        for kind in moco.KINDS:
            assert report.queue_fill[kind] <= 20

    def test_checkpoint_round_trips_all_eight_state_dicts(self, tmp_path):
        train = _tiny_batch(n=32, seed=9)
        val = _tiny_batch(n=16, seed=10)
        cfg = _tiny_config(epochs=3)

        report = moco.train_moco(train, val, cfg, tmp_path, "ckpt")
        assert report.checkpoint is not None

        model, saved_cfg = moco.load_multimodal_moco(report.checkpoint)
        x = torch.randn(2, 40)
        scale = torch.randn(2)
        fused, _ = model.encode_online("catalog", x, scale)
        assert fused.shape == (2, cfg.embedding_dim)
        assert saved_cfg["embedding_dim"] == cfg.embedding_dim

    def test_training_is_reproducible(self, tmp_path):
        train = _tiny_batch(n=32, seed=11)
        val = _tiny_batch(n=16, seed=12)
        cfg = _tiny_config(epochs=2, amp=False)

        first = moco.train_moco(train, val, cfg, tmp_path, "a")
        second = moco.train_moco(train, val, cfg, tmp_path, "b")

        assert first.train_losses[0] == pytest.approx(second.train_losses[0], rel=1e-3)


class TestUnwiredFromProduction:
    def test_multimodal_moco_is_not_imported_by_rpc(self):
        from pathlib import Path

        rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
        assert "multimodal" not in rpc_source
