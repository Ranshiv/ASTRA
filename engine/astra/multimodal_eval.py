"""Evaluation for the multimodal MoCo encoder (backlog item 11): linear-probe
macro-F1, cross-survey retrieval recall, and brightness-preservation error --
the three metrics this backlog item names.

Same frozen-encoder-plus-sklearn-probe discipline `pretrain_probe.
probe_transfer` already established for backlog item 13: freeze the trained
encoder, extract embeddings, and fit a small supervised probe on top,
reporting what that probe can and cannot recover. No real human labels
exist yet (Phase 7 not built), so `linear_probe_macro_f1` uses
`multimodal_synthetic.CLASS_KINDS`' true-by-construction labels, with the
same caveat every other synthetic-label metric in this codebase carries:
it measures transfer to the classes actually generated, not to whatever a
human would eventually flag.
"""

from __future__ import annotations

import numpy as np


def _embed_all(model, kind: str, values: np.ndarray, raw_scale: np.ndarray,
               batch_size: int = 64) -> np.ndarray:
    """Frozen ONLINE-branch projection embeddings (L2-normalised), batched."""
    import torch

    from . import multimodal_encoders as enc

    device = next(model.online[kind].parameters()).device
    model.eval()
    chunks: list[np.ndarray] = []
    scaled = enc.signed_log_scale(np.asarray(raw_scale, dtype=np.float64))
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            x = torch.from_numpy(
                np.asarray(values[start:start + batch_size], dtype=np.float32)).to(device)
            s = torch.from_numpy(
                scaled[start:start + batch_size].astype(np.float32)).to(device)
            _, projected = model.encode_online(kind, x, s)
            chunks.append(projected.float().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, 0), dtype=np.float32)


def _fused_embeddings(model, batch, kind: str) -> np.ndarray:
    """Frozen FUSED (pre-projection) embeddings for one modality -- used by
    the brightness-preservation probe, since that probe asks whether the
    fused embedding (not the projection head's contrastive-specific output)
    retains the real scale scalar."""
    import torch

    from . import multimodal_encoders as enc
    from .multimodal_moco import kind_batch

    values, raw_scale = kind_batch(batch, kind)
    device = next(model.online[kind].parameters()).device
    model.eval()
    scaled = enc.signed_log_scale(np.asarray(raw_scale, dtype=np.float64))
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
        s = torch.from_numpy(scaled.astype(np.float32)).to(device)
        fused = enc.encode_and_fuse(model.online[kind], x, s)
    return fused.float().cpu().numpy()


def linear_probe_macro_f1(model, batch, kind: str = "lightcurve",
                          test_fraction: float = 0.3, seed: int = 42) -> dict:
    """Frozen `kind` projection embeddings -> `LogisticRegression` on
    `batch.class_labels` -> `f1_score(average="macro")`.

    Reports both the probe's macro-F1 and the chance level
    (`1/len(CLASS_KINDS)`), since a bare F1 number means little without
    knowing how many classes it had to separate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    from .multimodal_moco import kind_batch

    values, raw_scale = kind_batch(batch, kind)
    embeddings = _embed_all(model, kind, values, raw_scale)
    labels = np.asarray(batch.class_labels)

    n = len(labels)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = max(1, int(round(n * (1.0 - test_fraction))))
    train_idx, test_idx = order[:cut], order[cut:]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings[train_idx], labels[train_idx])
    predictions = clf.predict(embeddings[test_idx])

    n_classes = len(np.unique(labels))
    return {
        "macro_f1": float(f1_score(labels[test_idx], predictions, average="macro")),
        "chance_level": round(1.0 / max(n_classes, 1), 4),
        "n_classes": n_classes,
        "n_test": len(test_idx),
    }


def retrieval_recall(query_embeddings: np.ndarray, key_embeddings: np.ndarray,
                     object_ids: list[str],
                     ks: tuple[int, ...] = (1, 5, 10)) -> dict[int, float]:
    """Cross-modal retrieval recall@K: for each query, is its true paired
    object among the K nearest keys by cosine similarity?

    Embeddings are assumed already L2-normalised (the projection head's own
    output), so cosine similarity is a plain dot product. `object_ids` gives
    the true pairing -- query[i] and key[i] are the same real object.
    """
    n = len(object_ids)
    if n == 0:
        return {k: float("nan") for k in ks}

    similarity = query_embeddings @ key_embeddings.T  # (n, n)
    ranking = np.argsort(-similarity, axis=1)

    recalls: dict[int, float] = {}
    for k in ks:
        top_k = ranking[:, :min(k, n)]
        hit = np.array([i in top_k[i] for i in range(n)])
        recalls[k] = float(np.mean(hit))
    return recalls


def probe_brightness_preservation(model, batch, kind: str = "lightcurve",
                                  epochs: int = 50, seed: int = 42,
                                  test_fraction: float = 0.3) -> dict:
    """Does the FUSED embedding actually retain the real absolute-scale
    scalar? A frozen 2-layer MLP probe regresses
    `signed_log_scale(true_scale)` back out of the fused embedding; RMSE is
    reported in REAL physical units (inverting signed_log_scale), not
    normalised-space MSE, since a person needs a unit to judge the number
    by.
    """
    import torch
    from torch import nn

    from . import multimodal_encoders as enc
    from .multimodal_moco import kind_batch

    _, raw_scale = kind_batch(batch, kind)
    embeddings = _fused_embeddings(model, batch, kind)
    targets = enc.signed_log_scale(np.asarray(raw_scale, dtype=np.float64))

    n = len(targets)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = max(1, int(round(n * (1.0 - test_fraction))))
    train_idx, test_idx = order[:cut], order[cut:]

    embedding_dim = embeddings.shape[1]
    probe = nn.Sequential(nn.Linear(embedding_dim, 32), nn.GELU(), nn.Linear(32, 1))
    optimiser = torch.optim.Adam(probe.parameters(), lr=1e-2)

    train_x = torch.from_numpy(embeddings[train_idx].astype(np.float32))
    train_y = torch.from_numpy(targets[train_idx].astype(np.float32)).unsqueeze(-1)
    for _ in range(epochs):
        optimiser.zero_grad(set_to_none=True)
        prediction = probe(train_x)
        loss = torch.nn.functional.mse_loss(prediction, train_y)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        test_x = torch.from_numpy(embeddings[test_idx].astype(np.float32))
        predicted_log_scale = probe(test_x).squeeze(-1).numpy()

    predicted_real = enc.inverse_signed_log_scale(predicted_log_scale)
    true_real = np.asarray(raw_scale, dtype=np.float64)[test_idx]
    rmse_real_units = float(np.sqrt(np.mean((predicted_real - true_real) ** 2)))

    return {
        "rmse_real_units": rmse_real_units,
        "n_test": len(test_idx),
        "true_scale_range": [float(np.min(true_real)), float(np.max(true_real))],
    }
