"""Shared summary statistics for benchmark results.

Before this module, each `*_eval.py` (agn_changepoint_eval.py,
artifact_bank_eval.py, ...) carried its own local `summary()`/`_summary()`
reimplementation of mean/std/ci95 -- fine for one module in isolation, but
it meant every evaluator's confidence interval used a slightly different
convention with no single place to fix or extend it. `sweep._summary` is
the closest existing example: a quantile-over-seeds interval, adequate for
"how much does this seed vary" but not for "how much would this metric
vary if the object sample had been different", which is what a benchmark
leaderboard actually needs.

This module provides one shared implementation for both:

- `summary()` -- the seed-quantile convention `sweep.py` already uses,
  unchanged, so existing callers migrate without a numeric behaviour change.
- `paired_bootstrap_ci()` -- resampling *object groups* (not rows), which is
  the object-disjoint-split-consistent way to get a confidence interval for
  a benchmark metric: an object's rows should be resampled together or not
  at all, or the interval understates its own uncertainty.
- `benjamini_hochberg()` -- standard step-up FDR control (Benjamini &
  Hochberg 1995) for ranking many candidates. This is a different quantity
  from `significance.calibrate`'s empirical-tail FDR *estimate* on a score
  population; this function controls FDR across a set of p-values, which
  nothing in the codebase did before.
- `reliability_table()` / `expected_calibration_error()` / `brier_score()`
  for probabilistic-output calibration reporting (docs/BENCHMARKS.md).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def summary(values: Sequence[float]) -> dict | None:
    """Mean/std/ci95 over `values`, quantile-based (matches `sweep._summary`).

    Appropriate for summarizing across seeds, where "how much does this
    metric move as the seed changes" is the question. For "how much would
    this metric move under a different object sample", use
    `paired_bootstrap_ci` instead.
    """
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(finite):
        return None
    return {
        "mean": round(float(np.mean(finite)), 4),
        "std": round(float(np.std(finite, ddof=1)), 4) if len(finite) > 1 else 0.0,
        "ci95": [round(float(np.quantile(finite, 0.025)), 4),
                 round(float(np.quantile(finite, 0.975)), 4)],
        "n": int(len(finite)),
    }


def paired_bootstrap_ci(
    group_values: dict[str, Sequence[float]],
    metric_fn: Callable[[np.ndarray], float],
    *, n_resamples: int = 2000, seed: int = 42, alpha: float = 0.05,
) -> dict:
    """Bootstrap CI for a metric, resampling *object groups* with replacement.

    `group_values` maps an object-group ID (e.g. the object ID itself, or a
    field/season group for a sky/time split) to that group's per-row values
    (e.g. its prediction errors, or a 1-length list for a single per-object
    score). Resampling groups rather than rows is what keeps this interval
    consistent with an object-disjoint split: two rows from the same object
    are correlated, and resampling rows independently would understate the
    interval's true width.

    `metric_fn` receives one flat array (the resample's pooled values) and
    returns a scalar; e.g. `np.mean` for a rate, or a closure computing
    AUPRC against paired labels.
    """
    keys = sorted(group_values)
    if not keys:
        return {"point": float("nan"), "ci": [float("nan"), float("nan")],
                "n_groups": 0, "n_resamples": n_resamples}

    arrays = [np.asarray(group_values[k], dtype=float) for k in keys]
    pooled = np.concatenate(arrays) if arrays else np.asarray([])
    point = float(metric_fn(pooled))

    rng = np.random.default_rng(seed)
    n_groups = len(keys)
    boot_values = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        chosen = rng.integers(0, n_groups, size=n_groups)
        resample = np.concatenate([arrays[j] for j in chosen]) if n_groups else pooled
        boot_values[i] = metric_fn(resample)

    lower = float(np.quantile(boot_values, alpha / 2))
    upper = float(np.quantile(boot_values, 1 - alpha / 2))
    return {
        "point": round(point, 4),
        "ci": [round(lower, 4), round(upper, 4)],
        "n_groups": n_groups,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def benjamini_hochberg(p_values: Sequence[float], *, alpha: float = 0.05) -> dict:
    """Step-up FDR control across many candidate p-values.

    Returns which indices (into the original `p_values` order) are rejected
    at the given FDR level, plus each hypothesis's BH-adjusted q-value.
    Standard reference: Benjamini & Hochberg (1995), JRSS-B 57(1):289-300.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return {"rejected": [], "q_values": [], "alpha": alpha}

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)
    raw_q = ranked * m / ranks
    # Enforce monotonicity: q-values must not decrease as rank decreases.
    q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_values = np.empty(m, dtype=float)
    q_values[order] = q_sorted
    rejected = [bool(q <= alpha) for q in q_values]

    return {
        "rejected": rejected,
        "q_values": [round(float(q), 6) for q in q_values],
        "n_rejected": int(sum(rejected)),
        "alpha": alpha,
    }


def reliability_table(
    probabilities: Sequence[float], outcomes: Sequence[int], *, n_bins: int = 10,
) -> list[dict]:
    """Per-bin (predicted probability vs. observed frequency) for a
    reliability diagram."""
    probs = np.asarray(probabilities, dtype=float)
    outs = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        n = int(mask.sum())
        rows.append({
            "bin_lo": round(float(lo), 3), "bin_hi": round(float(hi), 3),
            "n": n,
            "mean_predicted": round(float(probs[mask].mean()), 4) if n else None,
            "observed_frequency": round(float(outs[mask].mean()), 4) if n else None,
        })
    return rows


def expected_calibration_error(
    probabilities: Sequence[float], outcomes: Sequence[int], *, n_bins: int = 10,
) -> float:
    probs = np.asarray(probabilities, dtype=float)
    table = reliability_table(probabilities, outcomes, n_bins=n_bins)
    total = len(probs)
    if total == 0:
        return float("nan")
    ece = 0.0
    for row in table:
        if row["n"] == 0:
            continue
        ece += (row["n"] / total) * abs(row["mean_predicted"] - row["observed_frequency"])
    return round(float(ece), 4)


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    probs = np.asarray(probabilities, dtype=float)
    outs = np.asarray(outcomes, dtype=float)
    if len(probs) == 0:
        return float("nan")
    return round(float(np.mean((probs - outs) ** 2)), 4)


__all__ = [
    "summary", "paired_bootstrap_ci", "benjamini_hochberg",
    "reliability_table", "expected_calibration_error", "brier_score",
]
