"""Evaluation-study correctness for `reproducibility_bundle_eval.py`.
Requires the `research` extra (`cryptography`); skips cleanly otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from astra import reproducibility_bundle_eval as rbe


def test_evaluate_signed_bundle_round_trip():
    result = rbe.evaluate_signed_bundle_round_trip()
    assert result["valid_signature_verifies"] is True
    assert result["unsigned_bundle_is_rejected"] is True
    assert result["tampered_hash_is_rejected"] is True
    assert result["tampered_environment_is_rejected"] is True
    assert result["tampered_experiment_refs_rejected"] is True


def test_evaluate_rerun_bitwise_and_tolerance_synthetic():
    result = rbe.evaluate_rerun_bitwise_and_tolerance_synthetic()
    assert result["bitwise_identical_ok"] is True
    assert result["bitwise_changed_rejected"] is True
    assert result["tolerance_correctly_distinguishes"] is True


def test_reproducibility_bundle_eval_is_not_wired_into_rpc():
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "reproducibility_bundle" not in rpc_source
