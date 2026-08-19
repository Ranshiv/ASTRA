"""The supervised loop is gated, grouped, calibrated, and reproducible."""
from __future__ import annotations

import json

import numpy as np

from astra import candidates, features, ranker


def _candidate(index: int, positive: bool) -> candidates.Candidate:
    rng = np.random.default_rng(index)
    values = {name: float(rng.normal()) for name in features.FEATURE_NAMES}
    # A learnable but not completely deterministic reviewed pattern.
    values["robust_amplitude"] = (2.0 if positive else -2.0) + rng.normal(0, 0.2)
    values["period_snr"] = (25.0 if positive else 3.0) + rng.normal(0, 0.3)
    return candidates.Candidate(
        candidate_id=f"cand-{index:03d}", object_id=f"object-{index:03d}",
        survey="ZTF", band="g", ra_deg=180.0 + index * 0.001, dec_deg=20.0,
        release="dr24", score={"total": 1.0 - index / 100},
        artifact={"likelihood": 0.0}, features=values,
        explanation={"recommended_actions": []},
    )


def _labelled_run(tmp_path, count: int = 60) -> list[candidates.Candidate]:
    built = [_candidate(index, index < count // 2) for index in range(count)]
    candidates.save(built, "reviewed", tmp_path)
    for candidate in built:
        label = "interesting" if int(candidate.candidate_id[-3:]) < count // 2 else "artifact"
        candidates.record_label(candidate.candidate_id, label, root=tmp_path)
    return built


def test_gate_requires_50_usable_labels_and_10_per_class(tmp_path):
    _labelled_run(tmp_path, 48)
    dataset = ranker.labelled_examples("reviewed", tmp_path)
    result = ranker.gate(dataset)

    assert result["ready"] is False
    assert result["usable_labels"] == 48
    assert "50 usable labels" in result["reason"]


def test_train_persists_auditable_grouped_calibrated_model(tmp_path):
    _labelled_run(tmp_path)

    result = ranker.train("reviewed", root=tmp_path, seed=123, bootstrap_samples=20)
    manifest = json.loads((tmp_path / "models" / "rankers" / "calibrated-logistic.json").read_text())

    assert result["ready"] is True
    assert (tmp_path / "models" / "rankers" / "calibrated-logistic.pkl").exists()
    assert {"feature_schema_hash", "preprocessing_hash", "evidence_hash",
            "label_snapshot_hash", "label_snapshot", "evaluation_split", "seed"} <= manifest.keys()
    train_groups = set(manifest["evaluation_split"]["train_groups"])
    test_groups = set(manifest["evaluation_split"]["test_groups"])
    assert train_groups.isdisjoint(test_groups)
    supervised = manifest["evaluation"]["supervised"]
    assert {"brier_score", "expected_calibration_error", "roc_auc", "calibration_bins"} <= supervised.keys()
    assert manifest["evaluation"]["confidence_intervals"]["roc_auc"]["samples"] > 0

    model, loaded = ranker.load(root=tmp_path)
    assert loaded["model_sha256"] == result["model_sha256"]
    assert model.predict_proba(np.zeros((2, len(model.feature_names)))).shape == (2,)


def test_apply_uses_the_persisted_probability_as_an_explicit_ranking(tmp_path):
    _labelled_run(tmp_path)
    ranker.train("reviewed", root=tmp_path, seed=8, bootstrap_samples=10)

    applied = ranker.apply("reviewed", root=tmp_path)
    ranked = candidates.load("reviewed", tmp_path)

    assert applied["ranking_method"] == "calibrated_logistic"
    assert all("supervised_probability" in candidate.score for candidate in ranked)
    assert ranked[0].score["ranking_method"] == "calibrated_logistic"
    assert ranked[0].rank == 1
