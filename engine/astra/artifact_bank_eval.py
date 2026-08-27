"""AUPRC across unseen nights, cameras, and surveys -- the metric roadmap
item 33 names -- split from `artifact_bank.py` purely to keep each file
under this project's 500-line guideline (same `stellar_manifold.py`/
`stellar_manifold_eval.py` split rationale, not an independent module).

`evaluate_cross_group_auprc` is a real leave-one-group-out study over
REAL TESS `PatchRecord`s, grouped by camera, CCD, or the per-file `night`
bucket `artifact_bank.build_patch_bank` attaches. `evaluate_cross_survey_
generalization_synthetic` is the "unseen survey" arm, and is NOT real
data: `artifact_bank.py`'s own docstring already states ZTF has no real
subtraction-artifact data this codebase can reach, so this function uses
real ZTF light-curve windows (`open_world_injection.extract_real_patches`)
corrupted by `evaluate.py`'s existing synthetic anomaly kinds as a
labelled stand-in. Every return dict from this function says so in a
`"synthetic_proxy"` field -- never presented as real ZTF artifact
performance.

Both studies validated on real (TESS) or explicitly-labelled-synthetic
(ZTF) data respectively -- the same "mechanism validated, not yet run at
real Stage-B scale" caveat every eval module in this family states, here
sharpened by an honest data-availability constraint rather than only a
scale one.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .artifact_bank import (
    ArtifactBankError, PatchRecord, coral_align, patch_features, train_hard_negative_classifier,
)
from .evaluate import ANOMALY_KINDS, inject


def grouped_split(records: list[PatchRecord], group_key: str,
                  held_out_value: object) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays `(train_idx, test_idx)`: every record whose `group_key`
    attribute equals `held_out_value` goes to test, everything else (with a
    non-`None` value for that attribute) goes to train."""
    if group_key not in ("camera", "ccd", "night", "sector"):
        raise ArtifactBankError(f"group_key must be one of camera/ccd/night/sector, got {group_key!r}")

    train_idx, test_idx = [], []
    for index, record in enumerate(records):
        value = getattr(record, group_key)
        if value is None:
            continue
        (test_idx if value == held_out_value else train_idx).append(index)
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def _labels_and_features(records: list[PatchRecord], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.array([patch_features(records[i].patch) for i in indices])
    labels = np.array([0 if records[i].category == "clean" else 1 for i in indices])
    return features, labels


def evaluate_cross_group_auprc(records: list[PatchRecord], *, group_by: str = "camera",
                               use_coral: bool = False, seed: int = 42) -> dict:
    """Leave-one-group-out AUPRC: for each real `group_by` value present in
    `records`, train on every other group's real patches and test on the
    held-out group's. `use_coral=True` aligns the training features to the
    held-out group's feature statistics (`artifact_bank.coral_align`)
    before training, rather than after -- so the SAME classifier consumes
    already-aligned features, matching CORAL's intended use."""
    if not records:
        raise ArtifactBankError("records must be non-empty")

    groups = sorted({getattr(r, group_by) for r in records if getattr(r, group_by) is not None},
                    key=str)
    if len(groups) < 2:
        raise ArtifactBankError(
            f"need at least 2 distinct real {group_by!r} values to run a held-out study, got {len(groups)}")

    folds: list[dict] = []
    for held_out in groups:
        train_idx, test_idx = grouped_split(records, group_by, held_out)
        if len(train_idx) == 0 or len(test_idx) == 0:
            folds.append({"held_out": held_out, "skipped": "empty train or test split"})
            continue
        train_features, train_labels = _labels_and_features(records, train_idx)
        test_features, test_labels = _labels_and_features(records, test_idx)
        if len(set(train_labels.tolist())) < 2 or len(set(test_labels.tolist())) < 2:
            folds.append({"held_out": held_out, "skipped": "fold has only one class present"})
            continue

        if use_coral:
            train_features = coral_align(train_features, test_features)

        from sklearn.metrics import average_precision_score

        model = train_hard_negative_classifier(train_features, train_labels, seed=seed)
        scores = model.predict_proba(test_features)[:, 1]
        auprc = float(average_precision_score(test_labels, scores))
        folds.append({"held_out": held_out, "n_train": int(len(train_idx)),
                     "n_test": int(len(test_idx)), "auprc": round(auprc, 4)})

    scored = [f["auprc"] for f in folds if "auprc" in f]
    return {
        "group_by": group_by, "use_coral": use_coral, "groups": [str(g) for g in groups],
        "folds": folds, "n_scored_folds": len(scored),
        "mean_auprc": round(float(np.mean(scored)), 4) if scored else None,
    }


def evaluate_cross_survey_generalization_synthetic(tess_records: list[PatchRecord], *,
                                                    survey: str = "ZTF", n_ztf_windows: int = 200,
                                                    patch_length: int = 32,
                                                    corruption_kinds: tuple[str, ...] = ANOMALY_KINDS,
                                                    seed: int = 42) -> dict:
    """SYNTHETIC proxy for "unseen survey" generalization -- see module
    docstring. Trains on real TESS `tess_records` only; evaluates on real
    ZTF baseline windows (`open_world_injection.extract_real_patches`)
    where roughly half are synthetically corrupted (`evaluate.inject`)
    into artifact-shaped positives. Returns an empty-scored result
    (`"n_ztf_windows_found": 0`) rather than a fabricated AUPRC when no
    real ZTF light curves are available in the local store."""
    if len(tess_records) < 2:
        raise ArtifactBankError("tess_records must contain at least 2 records to train a classifier")

    from .open_world_injection import extract_real_patches

    train_features, train_labels = _labels_and_features(tess_records, np.arange(len(tess_records)))
    if len(set(train_labels.tolist())) < 2:
        raise ArtifactBankError("tess_records must contain both clean and artifact patches")
    model = train_hard_negative_classifier(train_features, train_labels, seed=seed)

    baseline = extract_real_patches(survey=survey, limit=n_ztf_windows, patch_length=patch_length, seed=seed)
    result: dict = {
        "synthetic_proxy": True,
        "note": ("evaluates the TESS-trained classifier against real ZTF light-curve "
                "windows with SYNTHETICALLY injected corruption standing in for real "
                "ZTF subtraction artifacts, which this codebase cannot currently reach "
                "(see artifact_bank.py's module docstring) -- not a measurement of real "
                "ZTF artifact-detection performance"),
        "n_ztf_windows_found": int(len(baseline)),
    }
    if len(baseline) < 2:
        result["auprc"] = None
        return result

    rng = np.random.default_rng(seed)
    corrupt_mask = rng.random(len(baseline)) < 0.5
    sequences = baseline.copy()
    kind_counts: Counter = Counter()
    for index in np.flatnonzero(corrupt_mask):
        kind = corruption_kinds[index % len(corruption_kinds)]
        sequences[index] = inject(baseline[index], kind, rng)
        kind_counts[kind] += 1

    features = np.array([patch_features(sequence) for sequence in sequences])
    labels = corrupt_mask.astype(np.int64)
    if len(set(labels.tolist())) < 2:
        result["auprc"] = None
        return result

    from sklearn.metrics import average_precision_score

    scores = model.predict_proba(features)[:, 1]
    result["auprc"] = round(float(average_precision_score(labels, scores)), 4)
    result["n_corrupted"] = int(corrupt_mask.sum())
    result["kinds_injected"] = dict(kind_counts)
    return result


__all__ = [
    "grouped_split", "evaluate_cross_group_auprc", "evaluate_cross_survey_generalization_synthetic",
]
