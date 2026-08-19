"""Auditable, calibrated supervised ranking from reviewed candidates.

This is deliberately a conservative baseline, not a black-box replacement for
the evidence score.  It trains only after a meaningful label gate passes,
keeps objects together across bands/surveys during every split, calibrates on
a held-out group set, and writes the label snapshot plus every schema hash
beside the serialized model.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import candidates, config, features, scoring

POSITIVE = {"interesting", "needs_follow_up"}
NEGATIVE = {"artifact", "known_object"}
MIN_LABELS = 50
MIN_PER_CLASS = 10
RANKER_VERSION = 1
CALIBRATION_FRACTION = 0.25
TEST_FRACTION = 0.20
BOOTSTRAP_SAMPLES = 500


@dataclass
class CalibratedLogisticRanker:
    """Median-imputed/scaled logistic model with group-held-out Platt scaling."""

    feature_names: tuple[str, ...]
    preprocessor: Any
    estimator: Any
    calibrator: Any

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        transformed = self.preprocessor.transform(values)
        logits = self.estimator.decision_function(transformed).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]


def _hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def ranker_feature_names() -> tuple[str, ...]:
    """Raw measurement features only; composite scores are never training input."""
    return tuple(features.FEATURE_NAMES)


def feature_schema_hash() -> str:
    return _hash({"feature_version": features.FEATURE_VERSION,
                  "names": ranker_feature_names()})


def preprocessing_hash() -> str:
    return _hash({
        "imputer": "median", "scaler": "standard", "estimator": "logistic_regression",
        "calibration": "platt_logistic_held_out_groups",
        "calibration_fraction": CALIBRATION_FRACTION,
    })


def evidence_hash() -> str:
    """Pin the evidence-weight context in which labels were made."""
    return _hash({"weight_version": scoring.WEIGHT_VERSION,
                  "weights": scoring.WEIGHTS})


def _stable_group(candidate: candidates.Candidate) -> str:
    """Keep repeated observations of one sky object in a single split group."""
    if np.isfinite(candidate.ra_deg) and np.isfinite(candidate.dec_deg):
        # 1e-5 degrees is ~0.036 arcsec: finer than the normal match radius,
        # yet stable under JSON serialization and survey metadata round-off.
        return f"sky:{candidate.ra_deg:.5f}:{candidate.dec_deg:.5f}"
    return f"source:{candidate.survey}:{candidate.release}:{candidate.object_id}"


def _row(candidate: candidates.Candidate, names: tuple[str, ...]) -> np.ndarray:
    values: list[float] = []
    for name in names:
        value = candidate.features.get(name)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=float)


def labelled_examples(name: str = "default", root: Path | None = None) -> dict:
    """Build a frozen, usable label table and account for every exclusion."""
    built = candidates.load(name, root)
    labels = candidates.load_labels(root)
    by_id = {candidate.candidate_id: candidate for candidate in built}
    names = ranker_feature_names()
    rows: list[np.ndarray] = []
    y: list[int] = []
    groups: list[str] = []
    snapshot: list[dict] = []
    baseline_scores: list[float] = []
    excluded = {"uncertain_or_unknown": 0, "candidate_missing": 0,
                "non_finite_features": 0}

    for candidate_id, entry in sorted(labels.items()):
        label = entry.get("label")
        if label not in POSITIVE | NEGATIVE:
            excluded["uncertain_or_unknown"] += 1
            continue
        candidate = by_id.get(candidate_id)
        if candidate is None:
            excluded["candidate_missing"] += 1
            continue
        values = _row(candidate, names)
        if not np.all(np.isfinite(values)):
            excluded["non_finite_features"] += 1
            continue
        target = int(label in POSITIVE)
        group = _stable_group(candidate)
        rows.append(values)
        y.append(target)
        groups.append(group)
        baseline_scores.append(float(candidate.score.get("total", 0.0)))
        snapshot.append({
            "candidate_id": candidate.candidate_id, "label": label,
            "target": target, "group": group,
            "recorded_utc": entry.get("recorded_utc"),
        })

    values = np.vstack(rows) if rows else np.empty((0, len(names)), dtype=float)
    targets = np.asarray(y, dtype=int)
    return {
        "values": values, "targets": targets, "groups": np.asarray(groups, dtype=object),
        "feature_names": names, "snapshot": snapshot,
        "snapshot_hash": _hash(snapshot), "baseline_scores": np.asarray(baseline_scores, dtype=float),
        "excluded": excluded,
    }


def gate(dataset: dict) -> dict:
    y = dataset["targets"]
    groups = dataset["groups"]
    positives = int(np.count_nonzero(y == 1))
    negatives = int(np.count_nonzero(y == 0))
    positive_groups = len(set(groups[y == 1]))
    negative_groups = len(set(groups[y == 0]))
    result = {
        "minimum_labels": MIN_LABELS, "minimum_per_class": MIN_PER_CLASS,
        "usable_labels": int(len(y)), "positives": positives, "negatives": negatives,
        "groups": len(set(groups)), "positive_groups": positive_groups,
        "negative_groups": negative_groups, "excluded": dataset["excluded"],
    }
    if len(y) < MIN_LABELS or min(positives, negatives) < MIN_PER_CLASS:
        return {"ready": False, "reason": "need at least 50 usable labels and 10 per class", **result}
    if min(positive_groups, negative_groups) < 2:
        return {"ready": False, "reason": "each class needs at least two independent object groups", **result}
    return {"ready": True, **result}


def _group_split(y: np.ndarray, groups: np.ndarray, fraction: float,
                 seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Find a deterministic grouped holdout with both classes on both sides."""
    from sklearn.model_selection import GroupShuffleSplit

    best: tuple[float, np.ndarray, np.ndarray] | None = None
    splitter = GroupShuffleSplit(n_splits=100, test_size=fraction, random_state=seed)
    global_rate = float(np.mean(y))
    for train, held_out in splitter.split(np.zeros(len(y)), y, groups):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[held_out])) < 2:
            continue
        balance = abs(float(np.mean(y[held_out])) - global_rate)
        size = abs(len(held_out) / len(y) - fraction)
        score = balance + size
        if best is None or score < best[0]:
            best = score, train, held_out
    if best is None:
        raise ValueError("labels cannot form a class-balanced grouped holdout")
    return best[1], best[2]


def _fit(values: np.ndarray, y: np.ndarray, groups: np.ndarray,
         names: tuple[str, ...], seed: int) -> tuple[CalibratedLogisticRanker, dict]:
    """Fit base and calibration models on disjoint object groups."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    base_indices, calibration_indices = _group_split(y, groups, CALIBRATION_FRACTION, seed)
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformed = preprocessor.fit_transform(values[base_indices])
    estimator = LogisticRegression(
        class_weight="balanced", solver="lbfgs", max_iter=2_000, random_state=seed,
    ).fit(transformed, y[base_indices])
    calibration_logits = estimator.decision_function(
        preprocessor.transform(values[calibration_indices])).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1e6, solver="lbfgs", max_iter=2_000, random_state=seed + 1,
    ).fit(calibration_logits, y[calibration_indices])
    split = {
        "base_candidate_rows": base_indices.tolist(),
        "calibration_candidate_rows": calibration_indices.tolist(),
        "base_groups": sorted(set(groups[base_indices])),
        "calibration_groups": sorted(set(groups[calibration_indices])),
    }
    return CalibratedLogisticRanker(names, preprocessor, estimator, calibrator), split


def _calibration_bins(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    rows: list[dict] = []
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = ((probabilities >= low) & (probabilities < high)
                if index < bins - 1 else
                (probabilities >= low) & (probabilities <= high))
        if not np.any(mask):
            continue
        observed = float(np.mean(y[mask]))
        predicted = float(np.mean(probabilities[mask]))
        fraction = float(np.mean(mask))
        ece += fraction * abs(observed - predicted)
        rows.append({"lower": low, "upper": high, "count": int(np.count_nonzero(mask)),
                     "observed_rate": observed, "mean_probability": predicted})
    return float(ece), rows


def metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    """Discrimination and calibration metrics with a predeclared threshold."""
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)

    prediction = probabilities >= threshold
    ece, bins = _calibration_bins(y, probabilities)
    return {
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "expected_calibration_error": ece,
        "threshold": threshold,
        "calibration_bins": bins,
    }


def confidence_intervals(y: np.ndarray, probabilities: np.ndarray,
                         groups: np.ndarray, seed: int,
                         samples: int = BOOTSTRAP_SAMPLES) -> dict:
    """Grouped bootstrap CIs, never resampling duplicate bands independently."""
    rng = np.random.default_rng(seed)
    unique_groups = np.asarray(sorted(set(groups)), dtype=object)
    values: dict[str, list[float]] = {key: [] for key in
        ("precision", "recall", "f1", "roc_auc", "average_precision",
         "brier_score", "expected_calibration_error")}
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for _ in range(samples):
        chosen = rng.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[group] for group in chosen])
        if len(np.unique(y[indices])) < 2:
            continue
        result = metrics(y[indices], probabilities[indices])
        for key in values:
            values[key].append(float(result[key]))
    return {
        key: ({"low": float(np.quantile(series, 0.025)),
               "high": float(np.quantile(series, 0.975)),
               "samples": len(series)} if series else None)
        for key, series in values.items()
    }


def _model_paths(model_name: str, root: Path | None = None) -> tuple[Path, Path]:
    directory = (root / "models" / "rankers" if root is not None
                 else config.PATHS.models / "rankers")
    directory.mkdir(parents=True, exist_ok=True)
    safe = "".join(char for char in model_name if char.isalnum() or char in "-_" )
    if not safe:
        raise ValueError("model name must contain a letter or number")
    return directory / f"{safe}.pkl", directory / f"{safe}.json"


def _write_artifact(model: CalibratedLogisticRanker, manifest: dict,
                    model_name: str, root: Path | None) -> tuple[Path, Path]:
    model_path, manifest_path = _model_paths(model_name, root)
    temporary = model_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(model_path)
    manifest["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return model_path, manifest_path


def train(name: str = "default", *, model_name: str = "calibrated-logistic",
          root: Path | None = None, seed: int = 42,
          bootstrap_samples: int = BOOTSTRAP_SAMPLES) -> dict:
    """Train, evaluate, and persist the supervised baseline after its gate."""
    dataset = labelled_examples(name, root)
    readiness = gate(dataset)
    if not readiness["ready"]:
        return readiness

    values, y, groups = dataset["values"], dataset["targets"], dataset["groups"]
    train_indices, test_indices = _group_split(y, groups, TEST_FRACTION, seed)
    evaluation_model, calibration_split = _fit(
        values[train_indices], y[train_indices], groups[train_indices],
        dataset["feature_names"], seed + 10)
    probabilities = evaluation_model.predict_proba(values[test_indices])
    evaluated = metrics(y[test_indices], probabilities)
    intervals = confidence_intervals(y[test_indices], probabilities, groups[test_indices],
                                     seed + 20, bootstrap_samples)
    baseline = metrics(y[test_indices], dataset["baseline_scores"][test_indices])
    baseline_intervals = confidence_intervals(
        y[test_indices], dataset["baseline_scores"][test_indices], groups[test_indices],
        seed + 30, bootstrap_samples)

    deployment_model, deployment_split = _fit(
        values, y, groups, dataset["feature_names"], seed + 100)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "ranker_version": RANKER_VERSION, "kind": "calibrated_logistic",
        "created_utc": created, "candidate_run": name, "seed": seed,
        "feature_names": list(dataset["feature_names"]),
        "feature_schema_hash": feature_schema_hash(),
        "preprocessing_hash": preprocessing_hash(), "evidence_hash": evidence_hash(),
        "label_snapshot_hash": dataset["snapshot_hash"], "label_snapshot": dataset["snapshot"],
        "gate": readiness,
        "evaluation_split": {
            "train_candidate_ids": [dataset["snapshot"][i]["candidate_id"] for i in train_indices],
            "test_candidate_ids": [dataset["snapshot"][i]["candidate_id"] for i in test_indices],
            "train_groups": sorted(set(groups[train_indices])),
            "test_groups": sorted(set(groups[test_indices])),
            "calibration_within_train": calibration_split,
        },
        "deployment_split": deployment_split,
        "evaluation": {"supervised": evaluated, "composite_baseline": baseline,
                       "confidence_intervals": intervals,
                       "baseline_confidence_intervals": baseline_intervals},
    }
    model_path, manifest_path = _write_artifact(deployment_model, manifest, model_name, root)
    return {
        "ready": True, "model_name": model_name, "model_path": str(model_path),
        "manifest_path": str(manifest_path), "model_sha256": manifest["model_sha256"],
        "gate": readiness, "evaluation": manifest["evaluation"],
    }


def load(model_name: str = "calibrated-logistic", root: Path | None = None
         ) -> tuple[CalibratedLogisticRanker, dict]:
    model_path, manifest_path = _model_paths(model_name, root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("ranker_version") != RANKER_VERSION:
        raise ValueError("unsupported ranker artifact version")
    if manifest.get("feature_schema_hash") != feature_schema_hash():
        raise ValueError("ranker feature schema differs from the current engine")
    if manifest.get("preprocessing_hash") != preprocessing_hash():
        raise ValueError("ranker preprocessing definition differs from the current engine")
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != manifest.get("model_sha256"):
        raise ValueError("ranker model checksum does not match its manifest")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, CalibratedLogisticRanker):
        raise ValueError("ranker artifact has an unexpected type")
    return model, manifest


def apply(name: str = "default", *, model_name: str = "calibrated-logistic",
          root: Path | None = None) -> dict:
    """Attach auditable supervised probabilities and explicitly re-rank a run."""
    model, manifest = load(model_name, root)
    built = candidates.load(name, root)
    matrix = np.vstack([_row(candidate, model.feature_names) for candidate in built]) \
        if built else np.empty((0, len(model.feature_names)))
    probabilities = model.predict_proba(matrix) if len(built) else np.empty(0)
    for candidate, probability in zip(built, probabilities):
        candidate.score["supervised_probability"] = round(float(probability), 6)
        candidate.score["ranking_method"] = "calibrated_logistic"
        candidate.score["ranker_model_sha256"] = manifest["model_sha256"][:16]
        candidate.explanation["supervised_ranker"] = {
            "model": model_name, "probability_interesting": round(float(probability), 6),
            "label_snapshot_hash": manifest["label_snapshot_hash"],
            "feature_schema_hash": manifest["feature_schema_hash"],
        }
    ranked = candidates.rank(built, ranking_field="supervised_probability")
    path = candidates.save(ranked, name, root)
    return {"name": name, "model_name": model_name, "candidates": len(ranked),
            "output_path": str(path), "ranking_method": "calibrated_logistic"}


def list_models(root: Path | None = None) -> list[dict]:
    directory = (root / "models" / "rankers" if root is not None
                 else config.PATHS.models / "rankers")
    if not directory.exists():
        return []
    results = []
    for path in sorted(directory.glob("*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results.append({"model_name": path.stem, "kind": manifest.get("kind"),
                        "created_utc": manifest.get("created_utc"),
                        "model_sha256": manifest.get("model_sha256"),
                        "label_snapshot_hash": manifest.get("label_snapshot_hash")})
    return results
