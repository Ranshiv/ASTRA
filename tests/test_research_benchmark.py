"""`research.benchmark`'s manifest-scoping and injection mechanics.

Before `scope_to_manifest` existed, `research.benchmark.run_cross_survey_anomaly`
was handed whatever `featurematrix.build(survey=...)` returned -- every stored
curve for that survey, not only the objects one dataset manifest's cone
actually matched. A `ResultRecord`'s `dataset_manifest_hash` then certified
the *query*, not the rows scored (docs/RESULTS.md's "Reading these numbers
correctly"). See the P0 research plan.
"""

from __future__ import annotations

import numpy as np
import pytest

from astra.featurematrix import FeatureMatrix
from astra.research import benchmark


def _matrix(object_ids: list[str], n_features: int = 3, seed: int = 0) -> FeatureMatrix:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(object_ids), n_features))
    identities = [{"object_id": oid, "survey": "ZTF"} for oid in object_ids]
    return FeatureMatrix(values=values, identities=identities,
                         feature_names=tuple(f"f{i}" for i in range(n_features)))


def test_scope_to_manifest_keeps_only_wanted_rows():
    matrix = _matrix(["a", "b", "c", "d"])
    scoped, dropped = benchmark.scope_to_manifest(matrix, {"a", "c"})

    assert dropped == 2
    assert len(scoped) == 2
    assert {i["object_id"] for i in scoped.identities} == {"a", "c"}


def test_scope_to_manifest_keeps_all_rows_when_fully_covered():
    matrix = _matrix(["a", "b"])
    scoped, dropped = benchmark.scope_to_manifest(matrix, {"a", "b"})

    assert dropped == 0
    assert len(scoped) == 2


def test_scope_to_manifest_empty_result_when_no_overlap():
    matrix = _matrix(["a", "b"])
    scoped, dropped = benchmark.scope_to_manifest(matrix, {"x", "y"})

    assert dropped == 2
    assert len(scoped) == 0


def test_scope_to_manifest_accepts_list_or_set():
    matrix = _matrix(["a", "b", "c"])
    scoped_from_list, _ = benchmark.scope_to_manifest(matrix, ["a", "b"])
    scoped_from_set, _ = benchmark.scope_to_manifest(matrix, {"a", "b"})

    assert {i["object_id"] for i in scoped_from_list.identities} == \
        {i["object_id"] for i in scoped_from_set.identities} == {"a", "b"}


def test_perturbed_matrix_injects_in_raw_pre_scaler_space():
    """Every scored method must see the identical injected perturbation --
    `_perturbed_matrix` must not standardise before injecting, or a
    baseline that standardises differently would see a different shape."""
    matrix = _matrix([f"obj{i}" for i in range(50)], n_features=6, seed=1)
    identities = [{"object_id": i["object_id"]} for i in matrix.identities]

    perturbed, labels = benchmark._perturbed_matrix(
        matrix, identities, fraction=0.2, seed=3)

    assert perturbed.values.shape == matrix.values.shape
    assert labels.sum() == pytest.approx(round(0.2 * 50), abs=1)
    # Injected rows must actually differ from the originals.
    injected_rows = np.where(labels == 1)[0]
    assert not np.allclose(perturbed.values[injected_rows], matrix.values[injected_rows])
