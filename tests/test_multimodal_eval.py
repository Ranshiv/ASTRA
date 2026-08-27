"""multimodal_eval.py: linear-probe macro-F1, cross-survey retrieval
recall, and brightness-preservation error -- the three metrics backlog
item 11 names."""

from __future__ import annotations

import numpy as np
import pytest

from astra import multimodal_eval as ev
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


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """One small trained model, reused across this file's tests -- training
    is the expensive part; each test only needs to probe the result."""
    train = syn.build_synthetic_pairs(n=120, seed=1, lc_length=64,
                                      image_size=16, spectrum_length=64)
    val = syn.build_synthetic_pairs(n=40, seed=2, lc_length=64,
                                    image_size=16, spectrum_length=64)
    cfg = _tiny_config(epochs=10)
    report = moco.train_moco(train, val, cfg, tmp_path_factory.mktemp("moco"), "eval")
    model, _ = moco.load_multimodal_moco(report.checkpoint)
    model = model.to(next(model.online["lightcurve"].parameters()).device)
    return model, val


class TestRetrievalRecall:
    def test_perfect_alignment_gives_recall_one_at_k_one(self):
        embeddings = torch.nn.functional.normalize(torch.randn(20, 8), dim=-1).numpy()
        object_ids = [f"o{i}" for i in range(20)]
        recall = ev.retrieval_recall(embeddings, embeddings, object_ids, ks=(1, 5))
        assert recall[1] == pytest.approx(1.0)

    def test_recall_is_monotonically_non_decreasing_in_k(self, trained_model):
        model, batch = trained_model
        lc_values, lc_scale = moco.kind_batch(batch, "lightcurve")
        img_values, img_scale = moco.kind_batch(batch, "image")
        lc_emb = ev._embed_all(model, "lightcurve", lc_values, lc_scale)
        img_emb = ev._embed_all(model, "image", img_values, img_scale)

        recall = ev.retrieval_recall(lc_emb, img_emb, batch.object_ids, ks=(1, 5, 10, 20))
        values = [recall[k] for k in (1, 5, 10, 20)]
        assert values == sorted(values)

    def test_empty_input_does_not_crash(self):
        empty = np.empty((0, 8))
        recall = ev.retrieval_recall(empty, empty, [], ks=(1, 5))
        assert all(np.isnan(v) for v in recall.values())


class TestLinearProbeMacroF1:
    def test_returns_well_formed_result(self, trained_model):
        model, batch = trained_model
        result = ev.linear_probe_macro_f1(model, batch)
        assert 0.0 <= result["macro_f1"] <= 1.0
        assert result["n_classes"] == len(syn.CLASS_KINDS)
        assert result["chance_level"] == pytest.approx(1.0 / len(syn.CLASS_KINDS))

    def test_works_for_every_modality(self, trained_model):
        model, batch = trained_model
        for kind in moco.KINDS:
            result = ev.linear_probe_macro_f1(model, batch, kind=kind)
            assert 0.0 <= result["macro_f1"] <= 1.0


class TestBrightnessPreservation:
    def test_returns_finite_rmse_in_real_units(self, trained_model):
        model, batch = trained_model
        result = ev.probe_brightness_preservation(model, batch, epochs=20)
        assert np.isfinite(result["rmse_real_units"])
        assert result["rmse_real_units"] >= 0.0
        assert result["n_test"] > 0

    def test_reported_range_matches_the_batchs_real_scale(self, trained_model):
        model, batch = trained_model
        result = ev.probe_brightness_preservation(model, batch, kind="catalog", epochs=20)
        low, high = result["true_scale_range"]
        assert low <= high
        # catalog_scale is the raw magnitude (10-20 by construction).
        assert 9.0 <= low and high <= 21.0

    def test_works_for_every_modality(self, trained_model):
        model, batch = trained_model
        for kind in moco.KINDS:
            result = ev.probe_brightness_preservation(model, batch, kind=kind, epochs=10)
            assert np.isfinite(result["rmse_real_units"])
