from __future__ import annotations

from astra import stageb


def test_dataset_fingerprint_is_ordered_and_mode_aware():
    rows = [
        {"survey": "ZTF", "release": "dr24", "object_id": "1", "band": "g", "path": "a"},
        {"survey": "ZTF", "release": "dr24", "object_id": "2", "band": "r", "path": "b"},
    ]
    assert stageb.dataset_fingerprint(rows, mode="time", length=256) == \
        stageb.dataset_fingerprint(rows, mode="time", length=256)
    assert stageb.dataset_fingerprint(rows, mode="time", length=256) != \
        stageb.dataset_fingerprint(rows, mode="season", length=256)


def test_aggregate_reports_seed_intervals_without_missing_as_zero():
    result = stageb._aggregate([
        {"seed": 17, "methods": [{"name": "baseline", "roc_auc": 0.6,
                                     "average_precision": 0.5,
                                     "precision_at_k": 0.4, "recall_at_k": 0.4}]},
        {"seed": 29, "methods": [{"name": "baseline", "roc_auc": None,
                                     "average_precision": None,
                                     "precision_at_k": None, "recall_at_k": None}]},
    ])
    row = result[0]
    assert row["runs"] == 2
    assert row["roc_auc"]["mean"] == 0.6
    assert row["roc_auc"]["ci95"][0] == 0.6
