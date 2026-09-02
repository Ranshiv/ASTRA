"""Join capability for pretrained-encoder embeddings (pretrain_join.py).

Deliberately not testing that embeddings improve detection -- no favorable
result exists yet (see the module docstring). These tests only verify the
join mechanics: row count preserved, NaN for unmatched rows, and that the
appended columns flow through the existing anomaly pipeline unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra import anomaly, featurematrix, pretrain, pretrain_join, store
from astra.surveys.base import LightCurve, SourceRef

torch = pytest.importorskip("torch", reason="PyTorch not installed")


def _config(**overrides) -> pretrain.PretrainConfig:
    defaults = dict(length=64, patch_size=8, transformer_dim=16,
                    transformer_heads=4, transformer_layers=1,
                    epochs=3, batch_size=16, span_mask_ratio=0.2)
    defaults.update(overrides)
    return pretrain.PretrainConfig(**defaults)


def _fake_sequences(n=64, length=64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 4 * np.pi, length)
    values = np.stack([
        np.sin(time * rng.uniform(0.5, 2.0) + rng.uniform(0, np.pi))
        + rng.normal(0, 0.05, length)
        for _ in range(n)
    ]).astype(np.float32)
    mask = np.ones((n, length), dtype=np.float32)
    return np.stack([values, mask], axis=1)


def _write_curve(root, object_id: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    source = SourceRef(survey="ZTF", object_id=object_id, ra_deg=180.122, dec_deg=22.411)
    time = 2458000.123456 + np.arange(200, dtype=np.float64) * 0.5
    value = 18.0 + rng.normal(0.0, 0.05, size=200)
    err = np.full(200, 0.03)
    curve = LightCurve(source=source, release="dr24", band="g", value_kind="mag",
                       time=time, value=value, value_err=err, time_system="HJD_UTC")
    store.write_curve(curve, root)


@pytest.fixture
def checkpoint(tmp_path) -> str:
    data = _fake_sequences(64, 64)
    cfg = _config()
    checkpoint_dir = tmp_path / "checkpoints"
    report = pretrain.pretrain(data[:48], data[48:], cfg, checkpoint_dir, "t")
    return report.checkpoint


class TestJoinPretrainedEmbeddings:
    def test_row_count_and_column_names_are_preserved(self, tmp_path, checkpoint):
        curves_root = tmp_path / "curves"
        _write_curve(curves_root, "obj0", seed=0)
        _write_curve(curves_root, "obj1", seed=1)
        matrix = featurematrix.build(root=curves_root)

        joined, diagnostics = pretrain_join.join_pretrained_embeddings(matrix, checkpoint)

        assert len(joined) == len(matrix)
        assert joined.feature_names[:len(matrix.feature_names)] == matrix.feature_names
        assert joined.feature_names[len(matrix.feature_names):] == tuple(
            f"pretrain_embed_{i}" for i in range(16))
        assert joined.values.shape[1] == matrix.values.shape[1] + 16

    def test_all_valid_curves_all_match(self, tmp_path, checkpoint):
        curves_root = tmp_path / "curves"
        _write_curve(curves_root, "obj0", seed=0)
        _write_curve(curves_root, "obj1", seed=1)
        matrix = featurematrix.build(root=curves_root)

        _, diagnostics = pretrain_join.join_pretrained_embeddings(matrix, checkpoint)

        assert diagnostics == {"matched": 2, "total": 2, "match_rate": 1.0}

    def test_unreadable_curve_gets_nan_not_dropped(self, tmp_path, checkpoint):
        curves_root = tmp_path / "curves"
        _write_curve(curves_root, "obj0", seed=0)
        _write_curve(curves_root, "obj1", seed=1)
        matrix = featurematrix.build(root=curves_root)
        matrix.identities[1]["path"] = str(tmp_path / "does-not-exist.parquet")

        joined, diagnostics = pretrain_join.join_pretrained_embeddings(matrix, checkpoint)

        assert len(joined) == 2
        assert diagnostics == {"matched": 1, "total": 2, "match_rate": 0.5}
        n_original = len(matrix.feature_names)
        assert np.all(np.isnan(joined.values[1, n_original:]))
        assert not np.any(np.isnan(joined.values[0, n_original:]))

    def test_empty_matrix_returns_empty_with_joined_names(self, checkpoint):
        empty = featurematrix.FeatureMatrix(
            values=np.empty((0, 0)), identities=[], feature_names=(), feature_version=1)

        joined, diagnostics = pretrain_join.join_pretrained_embeddings(empty, checkpoint)

        assert len(joined) == 0
        assert joined.feature_names == tuple(f"pretrain_embed_{i}" for i in range(16))
        assert diagnostics == {"matched": 0, "total": 0, "match_rate": None}

    def test_joined_columns_flow_through_anomaly_detect(self, tmp_path, checkpoint):
        """Confirms the column-generic claim end-to-end, not just by
        inspection of anomaly.py."""
        curves_root = tmp_path / "curves"
        for i in range(12):
            _write_curve(curves_root, f"obj{i}", seed=i)
        matrix = featurematrix.build(root=curves_root)

        joined, _ = pretrain_join.join_pretrained_embeddings(matrix, checkpoint)
        result = anomaly.detect(joined)

        assert result.consensus.shape[0] <= len(joined)
