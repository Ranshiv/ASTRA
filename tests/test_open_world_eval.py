"""open_world_eval.py: the held-out real-transient recovery metric backlog
item 14 actually names."""

from __future__ import annotations

import numpy as np
import pytest

from astra import diffusion_train as diff
from astra import open_world_eval as owe
from astra.surveys.base import LightCurve, SourceRef

torch = pytest.importorskip("torch", reason="PyTorch not installed")


def _design_population(n=60, length=64, seed=0):
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 1, length)
    values = np.stack([
        np.sin(time * rng.uniform(2, 6)) + rng.normal(0, 0.05, length)
        for _ in range(n)
    ]).astype(np.float32)
    mask = np.ones((n, length), dtype=np.float32)
    design_values = np.stack([values, mask], axis=1)
    identities = [{"object_id": f"design{i}"} for i in range(n)]
    return design_values, identities


class _FakeConnector:
    """Duck-types the two `ALeRCEConnector` methods `assemble_held_out_set`
    calls -- no network, mirrors this codebase's existing connector test
    convention of injecting a fake rather than hitting `netclient` for a
    two-call composed function."""

    name = "ALeRCE"

    def __init__(self, n_objects=5, n_points=60, fail_object_id=None):
        self.n_objects = n_objects
        self.n_points = n_points
        self.fail_object_id = fail_object_id

    def query_classified_objects(self, class_name, min_probability=0.5, limit=100):
        return [
            SourceRef(survey="ALeRCE", object_id=f"real{i}", ra_deg=0.0, dec_deg=0.0,
                     extra={"class_name": class_name})
            for i in range(min(self.n_objects, limit))
        ]

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        if source.object_id == self.fail_object_id:
            raise RuntimeError("simulated fetch failure")
        rng = np.random.default_rng(abs(hash(source.object_id)) % 1000)
        n = self.n_points
        time = 58000.0 + np.cumsum(rng.uniform(0.5, 2.0, n))
        value = 18.0 + rng.normal(0, 0.3, n)
        return [LightCurve(source=source, release="ztf", band="g", value_kind="mag",
                           time=time, value=value, value_err=np.full(n, 0.05))]


class TestAssembleHeldOutSet:
    def test_labels_positives_and_negatives_correctly(self):
        design_values, design_identities = _design_population()
        held_out = owe.assemble_held_out_set(
            ("SNIa",), design_values, design_identities, length=64,
            limit_per_class=5, connector=_FakeConnector(n_objects=5))
        assert held_out.labels.sum() == 5
        assert len(held_out) == 10  # 5 positive + 5 negative (default balanced)

    def test_negative_count_can_be_overridden(self):
        design_values, design_identities = _design_population()
        held_out = owe.assemble_held_out_set(
            ("SNIa",), design_values, design_identities, length=64,
            limit_per_class=3, negative_count=8, connector=_FakeConnector(n_objects=3))
        assert held_out.labels.sum() == 3
        assert (held_out.labels == 0).sum() == 8

    def test_carries_an_honest_caveat_note(self):
        design_values, design_identities = _design_population()
        held_out = owe.assemble_held_out_set(
            ("SNIa",), design_values, design_identities, length=64,
            limit_per_class=2, connector=_FakeConnector(n_objects=2))
        assert "not individually verified" in held_out.note

    def test_a_failing_object_fetch_does_not_abort_assembly(self):
        design_values, design_identities = _design_population()
        held_out = owe.assemble_held_out_set(
            ("SNIa",), design_values, design_identities, length=64,
            limit_per_class=5, connector=_FakeConnector(n_objects=5, fail_object_id="real2"))
        assert held_out.labels.sum() == 4  # one object's fetch failed, rest survive

    def test_empty_design_population_does_not_crash(self):
        held_out = owe.assemble_held_out_set(
            ("SNIa",), np.empty((0, 2, 64)), [], length=64,
            limit_per_class=3, connector=_FakeConnector(n_objects=3))
        assert held_out.labels.sum() == 3
        assert (held_out.labels == 0).sum() == 0


class TestHeldOutRecovery:
    def test_matches_evaluate_score_method(self):
        from astra import evaluate

        labels = np.array([0] * 90 + [1] * 10)
        scores = labels.astype(float)
        result = owe.held_out_recovery(scores, labels)
        reference = evaluate.score_method("x", scores, labels).to_dict()
        assert result["roc_auc"] == reference["roc_auc"]


class TestEvaluateOpenWorldGeneralization:
    def _generator(self, tmp_path, design_values):
        cfg = diff.DiffusionConfig(patch_length=16, channels=(16, 32), time_embed_dim=16,
                                   timesteps=15, epochs=5, batch_size=16,
                                   effective_batch_size=16, patience=20)
        patches = design_values[:, :, :16].copy()
        report = diff.train_diffusion(patches[:48], patches[48:], cfg, tmp_path, "gen")
        model, _ = diff.load_diffusion_model(report.checkpoint)
        return model, cfg

    def test_runs_both_arms_and_returns_well_formed_summaries(self, tmp_path):
        design_values, design_identities = _design_population(n=60)
        held_out = owe.assemble_held_out_set(
            ("SNIa",), design_values, design_identities, length=64,
            limit_per_class=5, connector=_FakeConnector(n_objects=5))
        generator, gen_cfg = self._generator(tmp_path, design_values)

        result = owe.evaluate_open_world_generalization(
            design_values, design_identities, held_out,
            diffusion_generator=generator, diffusion_cfg=gen_cfg,
            fraction=0.15, seeds=(1, 2), epochs=5)

        assert set(result.keys()) == {"closed_world", "open_world", "held_out"}
        # Not asserting which arm wins -- only that both ran and produced a
        # well-formed summary shape.
        for arm in ("closed_world", "open_world"):
            if result[arm] is not None:
                assert {"mean", "std", "ci95", "n"} <= result[arm].keys()

    def test_requires_at_least_two_seeds(self, tmp_path):
        design_values, design_identities = _design_population(n=60)
        held_out = owe.assemble_held_out_set(
            ("SNIa",), design_values, design_identities, length=64,
            limit_per_class=5, connector=_FakeConnector(n_objects=5))
        generator, gen_cfg = self._generator(tmp_path, design_values)

        with pytest.raises(ValueError, match="at least two seeds"):
            owe.evaluate_open_world_generalization(
                design_values, design_identities, held_out,
                diffusion_generator=generator, diffusion_cfg=gen_cfg, seeds=(1,))

    def test_requires_both_classes_in_held_out_set(self, tmp_path):
        design_values, design_identities = _design_population(n=60)
        generator, gen_cfg = self._generator(tmp_path, design_values)
        all_positive = owe.HeldOutSet(
            values=design_values[:5], labels=np.ones(5, dtype=int), identities=[{}] * 5)

        with pytest.raises(ValueError, match="positive and negative"):
            owe.evaluate_open_world_generalization(
                design_values, design_identities, all_positive,
                diffusion_generator=generator, diffusion_cfg=gen_cfg, seeds=(1, 2))
