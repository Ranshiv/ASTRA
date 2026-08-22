"""Calibration and selection-function diagnostics.

Anomaly scores are intentionally left unchanged.  This module adds a
separately versioned interpretation layer: empirical tail probabilities,
threshold FDR estimates, and injection-recovery completeness summaries.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite(values: Iterable[object]) -> np.ndarray:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=np.float64)


def _fingerprint(values: np.ndarray) -> str:
    canonical = ",".join(f"{value:.12g}" for value in np.sort(values))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ci_binomial(successes: int, trials: int) -> list[float] | None:
    if trials <= 0:
        return None
    # Wilson interval is stable for rare events and requires no scipy.
    p = successes / trials
    z = 1.959963984540054
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def calibrate(scores: Iterable[object], *, reference_scores: Iterable[object] | None = None,
              threshold: float | None = None, strata: dict[str, Any] | None = None,
              method: str = "empirical_tail") -> dict[str, Any]:
    """Return a batch-relative or reference-backed calibration report.

    ``reference_scores`` must be a held-out/null population when supplied.
    The function never mutates scores and does not alter production ranking.
    """
    observed = _finite(scores)
    reference = _finite(reference_scores if reference_scores is not None else scores)
    if observed.size == 0 or reference.size == 0:
        return {
            "schema_version": SCHEMA_VERSION, "method": method,
            "ready": False, "reason": "no finite scores",
            "n_observed": int(observed.size), "n_reference": int(reference.size),
        }
    if threshold is None:
        threshold = float(np.quantile(observed, 0.95))
    threshold = float(threshold)
    # Empirical upper-tail probability with a +1 correction.  Keep this as a
    # scalar loop: a reference population can be millions of rows and a full
    # observed-by-reference broadcast would unnecessarily allocate gigabytes.
    p_values = np.asarray([
        (float(np.sum(reference >= score)) + 1.0) / (reference.size + 1.0)
        for score in observed
    ], dtype=np.float64)
    exceedances = int(np.sum(reference >= threshold))
    selected = int(np.sum(observed >= threshold))
    # Estimate the expected number of null selections at this threshold.
    expected_false = selected * (exceedances / reference.size)
    fdr = expected_false / selected if selected else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "ready": True,
        "reference_kind": "external_reference" if reference_scores is not None else "batch_relative",
        "n_observed": int(observed.size),
        "n_reference": int(reference.size),
        "threshold": threshold,
        "selected": selected,
        "reference_exceedances": exceedances,
        "estimated_fdr": round(float(min(1.0, max(0.0, fdr))), 8),
        "score_range": [float(np.min(observed)), float(np.max(observed))],
        "tail_probability_summary": {
            "min": float(np.min(p_values)),
            "median": float(np.median(p_values)),
            "max": float(np.max(p_values)),
        },
        "reference_fingerprint": _fingerprint(reference),
        "strata": dict(strata or {}),
        "generated_utc": _now(),
    }


def annotate(scores: Iterable[object], calibration: dict[str, Any],
             reference_scores: Iterable[object] | None = None) -> list[dict[str, Any]]:
    """Attach empirical p-values and selection flags to score rows."""
    observed = _finite(scores)
    reference = _finite(reference_scores if reference_scores is not None else scores)
    if not observed.size or not reference.size:
        return []
    threshold = float(calibration.get("threshold", np.quantile(observed, 0.95)))
    result = []
    for score in observed:
        exceedances = int(np.sum(reference >= score))
        p_value = (exceedances + 1.0) / (reference.size + 1.0)
        result.append({"score": float(score), "tail_probability": float(p_value),
                       "selected": bool(score >= threshold),
                       "calibration_schema_version": SCHEMA_VERSION})
    return result


def _bin_label(value: object, edges: list[float] | None) -> str:
    if edges is None:
        return "all"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(number):
        return "unknown"
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= number < high or (number == high == edges[-1]):
            return f"[{low:g},{high:g})"
    return "out_of_range"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic(x: np.ndarray, y: np.ndarray, regularization: float = 1.0,
                  max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small deterministic ridge-logistic solver for selection diagnostics."""
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    for _ in range(max(1, int(max_iter))):
        probability = _sigmoid(design @ beta)
        gradient = design.T @ (probability - y) + float(regularization) * penalty @ beta
        weights = np.maximum(probability * (1.0 - probability), 1e-6)
        hessian = design.T @ (design * weights[:, None]) + float(regularization) * penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        candidate = beta - step
        if float(np.max(np.abs(candidate - beta))) < 1e-7:
            beta = candidate
            break
        beta = candidate
    return beta, _sigmoid(design @ beta), design


def _auc(y: np.ndarray, probability: np.ndarray) -> float | None:
    positives = probability[y >= 0.5]
    negatives = probability[y < 0.5]
    if not len(positives) or not len(negatives):
        return None
    comparisons = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positives) * len(negatives)))


def fit_selection_model(records: Iterable[dict[str, Any]], *,
                        features: tuple[str, ...] = ("amplitude", "duration_days", "magnitude",
                                                      "cadence_days", "crowding"),
                        regularization: float = 1.0, max_iter: int = 100,
                        bootstrap_samples: int = 0, seed: int = 42) -> dict[str, Any]:
    """Fit an interpretable recovery-probability model.

    This is intentionally a diagnostic layer: it estimates detection
    probability as a function of injection properties, but it never rewrites
    candidate scores or claims causal completeness outside the supplied
    injection population.
    """
    rows = [row for row in records if isinstance(row, dict)]
    usable = []
    for row in rows:
        values = [_number(row.get(name)) for name in features]
        detected = row.get("detected")
        if any(value is None for value in values) or detected is None:
            continue
        usable.append((values, bool(detected)))
    if not usable:
        return {"schema_version": SCHEMA_VERSION, "ready": False,
                "reason": "no complete model rows", "features": list(features),
                "injected": len(rows)}
    x = np.asarray([row[0] for row in usable], dtype=np.float64)
    y = np.asarray([row[1] for row in usable], dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {"schema_version": SCHEMA_VERSION, "ready": False,
                "reason": "model requires detected and missed rows", "features": list(features),
                "injected": len(rows), "usable": len(usable)}
    centre = np.nanmedian(x, axis=0)
    scale = np.nanmedian(np.abs(x - centre), axis=0) * 1.4826
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    standardized = (x - centre) / scale
    beta, probability, _ = _fit_logistic(standardized, y, regularization, max_iter)
    logloss = float(-np.mean(y * np.log(np.clip(probability, 1e-12, 1.0))
                           + (1.0 - y) * np.log(np.clip(1.0 - probability, 1e-12, 1.0))))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "ready": True,
        "features": list(features), "injected": len(rows), "usable": len(usable),
        "detected": int(y.sum()), "intercept": float(beta[0]),
        "coefficients": {name: float(value) for name, value in zip(features, beta[1:])},
        "centre": {name: float(value) for name, value in zip(features, centre)},
        "scale": {name: float(value) for name, value in zip(features, scale)},
        "metrics": {"roc_auc": _auc(y, probability),
                    "brier": float(np.mean((probability - y) ** 2)),
                    "logloss": logloss},
        "regularization": float(regularization), "generated_utc": _now(),
    }
    if bootstrap_samples > 0:
        rng = np.random.default_rng(int(seed))
        coefficients = []
        for _ in range(min(int(bootstrap_samples), 500)):
            indices = rng.integers(0, len(y), size=len(y))
            sampled_beta, _, _ = _fit_logistic(standardized[indices], y[indices],
                                               regularization, max_iter)
            coefficients.append(sampled_beta)
        if coefficients:
            matrix = np.asarray(coefficients, dtype=np.float64)
            result["bootstrap"] = {
                "samples": len(matrix), "seed": int(seed),
                "ci95": {"intercept": [float(np.quantile(matrix[:, 0], 0.025)),
                                         float(np.quantile(matrix[:, 0], 0.975))],
                         **{name: [float(np.quantile(matrix[:, index + 1], 0.025)),
                                   float(np.quantile(matrix[:, index + 1], 0.975))]
                            for index, name in enumerate(features)}},
            }
    return result


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate_selection(records: Iterable[dict[str, Any]], *, dimensions: tuple[str, ...] =
                       ("amplitude", "duration_days", "magnitude"),
                       edges: dict[str, list[float]] | None = None,
                       fit_model: bool = False,
                       model_features: tuple[str, ...] | None = None,
                       bootstrap_samples: int = 0,
                       seed: int = 42) -> dict[str, Any]:
    """Summarize injection-recovery completeness by physical dimensions.

    Each record requires ``detected`` and may carry the named dimensions.
    Missing dimensions are kept in an explicit ``unknown`` bin rather than
    silently dropped.
    """
    rows = [row for row in records if isinstance(row, dict)]
    grouped: dict[tuple[str, ...], list[bool]] = defaultdict(list)
    for row in rows:
        key = tuple(_bin_label(row.get(name), (edges or {}).get(name))
                    for name in dimensions)
        grouped[key].append(bool(row.get("detected", False)))
    cells = []
    for key in sorted(grouped):
        values = grouped[key]
        successes = sum(values)
        trials = len(values)
        weights = []
        for row in rows:
            row_key = tuple(_bin_label(row.get(name), (edges or {}).get(name))
                            for name in dimensions)
            if row_key == key:
                weight = _number(row.get("sampling_weight", row.get("weight", 1.0)))
                weights.append(max(0.0, float(weight or 0.0)))
        weight_total = float(sum(weights))
        weighted_detected = float(sum(weight for weight, detected in zip(weights, values) if detected))
        effective_n = (weight_total * weight_total / sum(weight * weight for weight in weights)
                       if weights and sum(weight * weight for weight in weights) > 0 else 0.0)
        cells.append({
            "bins": dict(zip(dimensions, key)),
            "detected": successes,
            "injected": trials,
            "completeness": successes / trials if trials else None,
            "ci95": _ci_binomial(successes, trials),
            "weighted_detected": weighted_detected if weights else None,
            "weighted_injected": weight_total if weights else None,
            "weighted_completeness": weighted_detected / weight_total if weight_total else None,
            "effective_injected": effective_n if weights else None,
        })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ready": bool(rows),
        "injected": len(rows),
        "detected": sum(bool(row.get("detected", False)) for row in rows),
        "dimensions": list(dimensions),
        "edges": edges or {},
        "cells": cells,
        "generated_utc": _now(),
    }
    if fit_model:
        payload["model"] = fit_selection_model(
            rows, features=model_features or tuple(dimensions),
            bootstrap_samples=bootstrap_samples, seed=seed,
        )
    return payload


def save(payload: dict[str, Any], *, root: Path | None = None,
         kind: str = "calibration", name: str = "default") -> Path:
    base = (root or Path.cwd()).resolve() / "results" / kind
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
