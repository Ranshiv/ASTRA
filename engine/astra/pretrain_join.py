"""Join frozen pretrained-encoder embeddings onto a FeatureMatrix, by row.

`pretrain.py` trains a self-supervised light-curve encoder; `pretrain_probe.py`
measures whether its embeddings help a linear probe at small label budgets.
Neither has ever produced a checkpoint or a favorable, real-data result in
this codebase (docs/LIMITATIONS.md marks the item `[PARTIAL]` with no reported
numbers), and both are explicit that this is "new evidence about a training
strategy, not a production score change." Wiring embeddings into
`anomaly.detect()`'s default path without that evidence would promote an
unvalidated method into live scoring, against this codebase's own stated
discipline (`anomaly.py`'s module docstring: "a newer method is not
automatically better ... the comparison has to be quantitative").

This module deliberately does NOT close that gap. It only adds the join
capability so a researcher CAN append embeddings to a `FeatureMatrix` and
separately measure whether detection improves (e.g. via `research/
benchmark.py` or `ablation.py`, both of which already work off any
`FeatureMatrix`) -- `anomaly.detect()`, `rpc.py`, and the UI are unchanged.

The join itself follows `featurematrix.join_gaia_columns`'s proven shape:
a column join, not a row union (`len(result) == len(matrix)`), and a row
whose light curve is unreadable or too short for the encoder gets NaN in
every appended column rather than an imputed value -- embedding availability
is itself informative, the same discipline `join_gaia_columns` applies to
catalogue non-matches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import store, tensors
from .featurematrix import FeatureMatrix


def join_pretrained_embeddings(matrix: FeatureMatrix, checkpoint_path: str | Path,
                               length: int | None = None,
                               batch_size: int = 64) -> tuple[FeatureMatrix, dict]:
    """Append frozen pretrained-encoder embeddings as extra columns.

    ``length`` defaults to the checkpoint's own training length
    (``cfg["length"]``) so a caller does not need to know the encoder's
    architecture in advance. Returns ``(FeatureMatrix, diagnostics)`` where
    ``diagnostics`` is ``{"matched", "total", "match_rate"}``, the same shape
    ``join_gaia_columns`` returns, so a caller can see how much of the join
    actually landed rather than trusting it happened at all.
    """
    from . import pretrain

    encoder, cfg = pretrain.load_pretrained_encoder(checkpoint_path)
    dim = int(cfg["transformer_dim"])
    sequence_length = int(length if length is not None else cfg["length"])
    embed_names = tuple(f"pretrain_embed_{i}" for i in range(dim))
    joined_names = matrix.feature_names + embed_names

    if len(matrix) == 0:
        empty = np.empty((0, len(joined_names)))
        return (FeatureMatrix(values=empty, identities=[], feature_names=joined_names,
                              feature_version=matrix.feature_version),
               {"matched": 0, "total": 0, "match_rate": None})

    sequences: list[np.ndarray | None] = []
    for identity in matrix.identities:
        try:
            curve = store.read_curve(Path(identity["path"]))
            sequences.append(tensors.resample(curve, length=sequence_length))
        except Exception:  # noqa: BLE001 - a missing/bad curve just stays unmatched
            sequences.append(None)

    valid = [(index, sequence) for index, sequence in enumerate(sequences)
             if sequence is not None]

    extra = np.full((len(matrix), dim), np.nan, dtype=np.float64)
    if valid:
        stacked = np.stack([sequence for _, sequence in valid]).astype(np.float32)
        embeddings = _pooled(encoder, stacked, batch_size)
        for (row_index, _), embedding in zip(valid, embeddings):
            extra[row_index] = embedding

    matched = len(valid)
    total = len(matrix)
    values = np.hstack([matrix.values, extra])
    new_identities = [dict(identity) for identity in matrix.identities]
    return (FeatureMatrix(values=values, identities=new_identities,
                          feature_names=joined_names, feature_version=matrix.feature_version),
           {"matched": matched, "total": total,
            "match_rate": (matched / total) if total else None})


def _pooled(encoder, values: np.ndarray, batch_size: int) -> np.ndarray:
    """Batch ``values`` (n, 2, length) through the frozen encoder's mean-pool
    head. A small local loop rather than importing `pretrain_probe`'s
    private `_pooled_embeddings` across a module boundary."""
    import torch

    encoder.eval()
    device = next(encoder.parameters()).device
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            chunk = torch.from_numpy(values[start:start + batch_size]).to(device)
            chunks.append(encoder.pooled(chunk).float().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, encoder.embed.out_channels))


__all__ = ["join_pretrained_embeddings"]
