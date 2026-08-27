"""Object-grouped and sky/time splits stay leakage-free and deterministic."""

from __future__ import annotations

from astra.research import splits


def test_object_grouped_split_has_no_leakage():
    ids = [f"obj{i}" for i in range(100)]
    split = splits.object_grouped_split(ids, split_id="test-split", seed=42)
    report = splits.detect_leakage(split)
    assert report["clean"]
    assert sum(split.fold_counts().values()) == 100


def test_object_grouped_split_deterministic_for_same_seed():
    ids = [f"obj{i}" for i in range(50)]
    a = splits.object_grouped_split(ids, split_id="s", seed=7)
    b = splits.object_grouped_split(ids, split_id="s", seed=7)
    assert a.folds == b.folds
    assert a.content_hash() == b.content_hash()


def test_object_grouped_split_dedupes_repeated_ids():
    ids = ["a", "a", "b", "c", "c", "c"]
    split = splits.object_grouped_split(ids, split_id="dedup", seed=1)
    assert sum(split.fold_counts().values()) == 3


def test_object_grouped_split_rejects_bad_fractions():
    import pytest
    with pytest.raises(ValueError):
        splits.object_grouped_split(["a"], split_id="s", fractions={"train": 0.5})


def test_sky_time_split_groups_by_cell_and_season_no_leakage():
    records = []
    for i in range(60):
        records.append({
            "object_id": f"obj{i}",
            "ra_deg": float((i * 37) % 360),
            "dec_deg": float((i * 13) % 180 - 90),
            "mjd": 59000.0 + i * 45,
        })
    split = splits.sky_time_split(records, split_id="sky-time", seed=3)
    report = splits.detect_leakage(split)
    assert report["clean"]
    assert sum(split.fold_counts().values()) == 60


def test_sky_time_split_keeps_same_object_in_one_fold_when_repeated():
    """An object observed on two nights in the same season/cell must not
    straddle folds -- this is the leakage a random row split would allow."""
    records = [
        {"object_id": "obj1", "ra_deg": 10.0, "dec_deg": 5.0, "mjd": 59000.0},
        {"object_id": "obj1", "ra_deg": 10.0, "dec_deg": 5.0, "mjd": 59005.0},
        {"object_id": "obj2", "ra_deg": 200.0, "dec_deg": -40.0, "mjd": 59100.0},
    ]
    split = splits.sky_time_split(records, split_id="repeat", seed=1)
    report = splits.detect_leakage(split)
    assert report["clean"]


def test_detect_leakage_flags_object_in_two_folds():
    split = splits.Split(split_id="broken", kind="object_grouped",
                         grouping_key="object_id", seed=0,
                         folds={"train": ["a", "b"], "test": ["b", "c"]})
    report = splits.detect_leakage(split)
    assert not report["clean"]
    assert report["n_leaked"] == 1
    assert report["leaked"][0]["object_id"] == "b"


def test_save_and_load_round_trip(tmp_path):
    split = splits.object_grouped_split(["a", "b", "c", "d"], split_id="rt", seed=5)
    splits.save(split, tmp_path)
    loaded = splits.load("rt", tmp_path)
    assert loaded.folds == split.folds
