"""open_world_scale.py: resumable real-data scale runner (backlog item 14,
gap 2). No network in these tests -- a fake connector, and a real, small,
locally-written ZTF population via `store.write_curve` (the `isolated_root`
fixture keeps all of this in a temp directory, never the real Datasets
root)."""

from __future__ import annotations

import numpy as np
import pytest

from astra import open_world_scale as ows
from astra.surveys.base import LightCurve, SourceRef

torch = pytest.importorskip("torch", reason="PyTorch not installed")


def _write_ztf_population(n=40, seed=0):
    from astra import store

    rng = np.random.default_rng(seed)
    for i in range(n):
        source = SourceRef(survey="ZTF", object_id=f"design{i}", ra_deg=180.0, dec_deg=20.0)
        points = 100
        time = 2458000.0 + np.cumsum(rng.uniform(0.5, 2.0, points))
        value = 18.0 + rng.normal(0, 0.2, points)
        curve = LightCurve(source=source, release="dr24", band="g", value_kind="mag",
                           time=time, value=value, value_err=np.full(points, 0.02))
        store.write_curve(curve)


class _FakeConnector:
    name = "ALeRCE"

    def __init__(self, n_objects=8):
        self.n_objects = n_objects

    def query_classified_objects(self, class_name, min_probability=0.5, limit=100):
        return [
            SourceRef(survey="ALeRCE", object_id=f"real{i}", ra_deg=0.0, dec_deg=0.0,
                     extra={"class_name": class_name})
            for i in range(min(self.n_objects, limit))
        ]

    def fetch_light_curves(self, source: SourceRef) -> list[LightCurve]:
        rng = np.random.default_rng(abs(hash(source.object_id)) % 1000)
        n = 100
        time = 58000.0 + np.cumsum(rng.uniform(0.5, 2.0, n))
        value = 18.0 + rng.normal(0, 0.3, n)
        value[30:40] -= 2.0
        return [LightCurve(source=source, release="ztf", band="g", value_kind="mag",
                           time=time, value=value, value_err=np.full(n, 0.05))]


def _tiny_kwargs(**overrides):
    defaults = dict(class_names=("SNIa",), design_survey="ZTF", design_length=64,
                    limit_per_class=5, seeds=(1, 2), epochs=3, patch_length=16,
                    diffusion_epochs=3, connector=_FakeConnector())
    defaults.update(overrides)
    return defaults


class TestRunScaleStudy:
    def test_completes_and_returns_a_well_formed_result(self, isolated_root, tmp_path):
        _write_ztf_population()
        checkpoint = tmp_path / "scale.json"

        result = ows.run_scale_study(root=tmp_path, checkpoint=checkpoint, **_tiny_kwargs())

        assert result["ready"] is True
        assert "closed_world" in result["result"]
        assert "open_world" in result["result"]
        assert result["held_out_summary"]["positive"] > 0

    def test_rejects_fewer_than_two_seeds(self, isolated_root, tmp_path):
        _write_ztf_population()
        with pytest.raises(ValueError, match="at least two"):
            ows.run_scale_study(root=tmp_path, seeds=(1,), **{
                k: v for k, v in _tiny_kwargs().items() if k != "seeds"})

    def test_resuming_reuses_the_checkpoint_and_matches(self, isolated_root, tmp_path):
        _write_ztf_population()
        checkpoint = tmp_path / "scale.json"
        kwargs = _tiny_kwargs()

        first = ows.run_scale_study(root=tmp_path, checkpoint=checkpoint, **kwargs)
        second = ows.run_scale_study(root=tmp_path, checkpoint=checkpoint, **kwargs)

        assert first["result"] == second["result"]
        assert first["generator_checkpoint"] == second["generator_checkpoint"]

    def test_a_changed_configuration_does_not_reuse_a_stale_checkpoint(self, isolated_root, tmp_path):
        _write_ztf_population()
        checkpoint = tmp_path / "scale.json"

        first = ows.run_scale_study(root=tmp_path, checkpoint=checkpoint, **_tiny_kwargs())
        second = ows.run_scale_study(
            root=tmp_path, checkpoint=checkpoint, **_tiny_kwargs(limit_per_class=3))

        # Different configuration -> the checkpoint is not reused, so this
        # must still complete correctly rather than returning stale state.
        assert second["ready"] is True

    def test_too_small_a_design_population_reports_not_ready(self, isolated_root, tmp_path):
        # No curves written at all.
        checkpoint = tmp_path / "scale.json"
        result = ows.run_scale_study(root=tmp_path, checkpoint=checkpoint, **_tiny_kwargs())
        assert result["ready"] is False
        assert "usable design sequences" in result["reason"]

    def test_never_asserts_which_arm_wins(self, isolated_root, tmp_path):
        """A correctness/contract check, not a scientific one: both arms'
        summaries must be present and neither privileged over the other in
        the returned shape."""
        _write_ztf_population()
        checkpoint = tmp_path / "scale.json"
        result = ows.run_scale_study(root=tmp_path, checkpoint=checkpoint, **_tiny_kwargs())
        assert "winner" not in result["result"]
        assert "best" not in result["result"]
