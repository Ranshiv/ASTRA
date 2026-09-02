"""Linear-probe transfer evaluation for `pretrain.py`'s encoder.

Split out from `pretrain.py` (which builds and trains the encoder) purely to
keep each file under this project's 500-line guideline: pretraining and
transfer-evaluation are genuinely separable concerns, and this module only
ever consumes an already-trained encoder (e.g. from
`pretrain.load_pretrained_encoder`), never `pretrain.py`'s internals.

No real human-labelled data exists yet (Phase 7's human-in-the-loop
labelling system is not built). `probe_transfer`'s "1%/10%/100% labels"
metric is therefore measured against `evaluate.build_injected()`'s
true-by-construction injection labels, the same honest stand-in
`evaluate.py` already uses everywhere else in this codebase.
"""

from __future__ import annotations

import numpy as np

from .research import stats as research_stats


def _pooled_embeddings(encoder, values: np.ndarray, batch_size: int = 64) -> np.ndarray:
    import torch

    encoder.eval()
    device = next(encoder.parameters()).device
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            chunk = torch.from_numpy(values[start:start + batch_size]).to(device)
            chunks.append(encoder.pooled(chunk).float().cpu().numpy())
    dim = encoder.embed.out_channels
    return np.concatenate(chunks) if chunks else np.empty((0, dim), dtype=np.float32)


def _stratified_subsample(train_idx: np.ndarray, labels: np.ndarray,
                          fraction: float, seed: int) -> np.ndarray | None:
    """A `fraction` of each class from `train_idx`, at least one row per class
    when that class has any rows at all -- so a very small `fraction` does
    not silently drop the positive class entirely. Returns None when a class
    is absent from `train_idx`, since no meaningful fit is possible then."""
    rng = np.random.default_rng(seed)
    sub_labels = labels[train_idx]
    positives = train_idx[sub_labels == 1]
    negatives = train_idx[sub_labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None

    n_pos = max(1, int(round(len(positives) * fraction)))
    n_neg = max(1, int(round(len(negatives) * fraction)))
    chosen_pos = rng.choice(positives, size=min(n_pos, len(positives)), replace=False)
    chosen_neg = rng.choice(negatives, size=min(n_neg, len(negatives)), replace=False)
    return np.concatenate([chosen_pos, chosen_neg])


def _summary(values: list[float]) -> dict | None:
    """Delegates to `research.stats.summary` -- see that module's docstring
    for why this shape (mean/std/ci95 over repeated seeds, not object-group
    bootstrap) is the right one here. Was this module's own local
    reimplementation; migrated per docs/LIMITATIONS.md's tracked debt."""
    return research_stats.summary(values)


def _train_test_split_indices(n: int, seed: int,
                              test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic index split, independent of anything fraction-related --
    `probe_transfer` calls this ONCE and reuses the same test indices across
    every label fraction, so a result cannot be an artefact of an easier
    test split at one fraction than another."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = max(1, int(round(n * (1.0 - test_fraction)))) if n > 1 else 1
    train_idx, test_idx = order[:cut], order[cut:]
    if len(test_idx) == 0:
        test_idx, train_idx = train_idx[-1:], train_idx[:-1]
    return train_idx, test_idx


def probe_transfer(encoder, values: np.ndarray, labels: np.ndarray,
                   fractions: tuple[float, ...] = (0.01, 0.10, 1.0),
                   seed: int = 42, n_repeats: int = 5,
                   test_fraction: float = 0.3) -> dict:
    """Frozen-encoder linear-probe transfer at each label fraction.

    This project's concrete stand-in for "transfer performance with 1%/10%/
    100% labels" until Phase 7's real human labels exist: `labels` is
    expected to come from `evaluate.build_injected()`. The SAME held-out
    test partition (drawn once, from `seed`) is reused across every
    fraction, so a result cannot be an artefact of an easier test split.
    Compares a `pretrained` arm (LogisticRegression on frozen pooled
    embeddings) against a `baseline` arm (LogisticRegression on
    `evaluate.sequence_summary` features) at the same fractions, so the
    reported numbers answer "does pretraining help", not just "can a linear
    probe do anything". Each fraction is repeated `n_repeats` times with an
    independent label subsample, reported as mean/std/ci95 -- a single draw
    at a small fraction is a noisy measurement, the same reasoning
    `sweep.py`'s CI95-gated `best()` already applies elsewhere in this
    codebase. A repeat with too few labelled rows of one class is skipped
    and counted, never silently scored as NaN.
    """
    from sklearn.linear_model import LogisticRegression

    from . import evaluate

    n = len(values)
    if n == 0:
        return {}

    train_idx, test_idx = _train_test_split_indices(n, seed, test_fraction)

    embeddings = _pooled_embeddings(encoder, values)
    baseline_features = evaluate.sequence_summary(values)
    test_labels = labels[test_idx]

    results: dict[float, dict] = {}
    for fraction in fractions:
        pretrained_scores: list[float] = []
        baseline_scores: list[float] = []
        skipped = 0

        for repeat in range(n_repeats):
            repeat_seed = seed + repeat + int(round(fraction * 100_000))
            sub_idx = _stratified_subsample(train_idx, labels, fraction, repeat_seed)
            if sub_idx is None or len(np.unique(labels[sub_idx])) < 2:
                skipped += 1
                continue
            sub_labels = labels[sub_idx]

            pretrained_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            pretrained_clf.fit(embeddings[sub_idx], sub_labels)
            pretrained_probs = pretrained_clf.predict_proba(embeddings[test_idx])[:, 1]
            pretrained_scores.append(
                evaluate.score_method("pretrained", pretrained_probs, test_labels).roc_auc)

            baseline_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            baseline_clf.fit(baseline_features[sub_idx], sub_labels)
            baseline_probs = baseline_clf.predict_proba(baseline_features[test_idx])[:, 1]
            baseline_scores.append(
                evaluate.score_method("baseline", baseline_probs, test_labels).roc_auc)

        results[fraction] = {
            "pretrained": _summary(pretrained_scores),
            "baseline": _summary(baseline_scores),
            "skipped_repeats": skipped,
        }

    return results
