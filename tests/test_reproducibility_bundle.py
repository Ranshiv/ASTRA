"""Bundle-hash, Ed25519 signing/verification, and rerun-check
correctness for `reproducibility_bundle.py`. Requires the `research`
extra (`pip install .[research]`) for the `cryptography` dependency;
skips cleanly otherwise.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cryptography")

from astra import reproducibility_bundle as rb
from astra.manifest import Manifest, SurveyQuery


def _sealed_manifest(dataset_id="ds1"):
    manifest = Manifest.create(dataset_id)
    manifest.add(SurveyQuery(survey="ZTF", release="dr24", ra_deg=180.0, dec_deg=22.0,
                            radius_arcsec=10.0, limit=50, object_ids=["x", "y"]))
    return manifest.seal()


# ---------------------------------------------------------------------------
# generate_keypair
# ---------------------------------------------------------------------------

def test_generate_keypair_is_deterministic_for_a_fixed_seed():
    first = rb.generate_keypair(seed=b"fixed")
    second = rb.generate_keypair(seed=b"fixed")
    assert first.public_key_hex() == second.public_key_hex()


def test_generate_keypair_differs_for_different_seeds():
    first = rb.generate_keypair(seed=b"a")
    second = rb.generate_keypair(seed=b"b")
    assert first.public_key_hex() != second.public_key_hex()


# ---------------------------------------------------------------------------
# build_bundle / sign_bundle / verify_bundle
# ---------------------------------------------------------------------------

def test_build_bundle_rejects_an_unsealed_manifest():
    manifest = Manifest.create("unsealed")
    with pytest.raises(rb.ReproducibilityBundleError):
        rb.build_bundle(manifest)


def test_sign_and_verify_round_trip_succeeds():
    bundle = rb.build_bundle(_sealed_manifest(), experiment_record_refs=["exp-1"])
    keypair = rb.generate_keypair(seed=b"k")
    signed = rb.sign_bundle(bundle, keypair)
    assert rb.verify_bundle(signed) is True


def test_unsigned_bundle_fails_verification():
    bundle = rb.build_bundle(_sealed_manifest())
    assert rb.verify_bundle(bundle) is False


def test_tampered_bundle_hash_fails_verification():
    bundle = rb.build_bundle(_sealed_manifest())
    signed = rb.sign_bundle(bundle, rb.generate_keypair(seed=b"k"))
    tampered = dataclasses.replace(signed, bundle_hash="0" * 64)
    assert rb.verify_bundle(tampered) is False


def test_tampered_environment_fails_verification():
    bundle = rb.build_bundle(_sealed_manifest())
    signed = rb.sign_bundle(bundle, rb.generate_keypair(seed=b"k"))
    tampered = dataclasses.replace(signed, environment={**signed.environment, "extra": "x"})
    assert rb.verify_bundle(tampered) is False


def test_verification_fails_under_the_wrong_public_key():
    bundle = rb.build_bundle(_sealed_manifest())
    signed = rb.sign_bundle(bundle, rb.generate_keypair(seed=b"key-a"))
    wrong_public = rb.generate_keypair(seed=b"key-b").public_key_hex()
    tampered = dataclasses.replace(signed, public_key_hex=wrong_public)
    assert rb.verify_bundle(tampered) is False


# ---------------------------------------------------------------------------
# verify_manifest_rerun
# ---------------------------------------------------------------------------

def test_verify_manifest_rerun_matches_an_identical_rerun():
    manifest = _sealed_manifest()
    bundle = rb.build_bundle(manifest)
    fresh = _sealed_manifest()
    result = rb.verify_manifest_rerun(bundle, fresh)
    assert result["matches"] is True


def test_verify_manifest_rerun_detects_a_changed_query():
    manifest = _sealed_manifest()
    bundle = rb.build_bundle(manifest)
    changed = Manifest.create("ds1")
    changed.add(SurveyQuery(survey="ZTF", release="dr24", ra_deg=180.0, dec_deg=22.0,
                            radius_arcsec=10.0, limit=50, object_ids=["x", "y", "z"]))
    changed.seal()
    result = rb.verify_manifest_rerun(bundle, changed)
    assert result["matches"] is False


# ---------------------------------------------------------------------------
# hash_array_bitwise / verify_rerun_bitwise / verify_rerun_tolerance
# ---------------------------------------------------------------------------

def test_verify_rerun_bitwise_accepts_identical_and_rejects_changed():
    reference = np.array([1.0, 2.0, 3.0])
    reference_hash = rb.hash_array_bitwise(reference)
    assert rb.verify_rerun_bitwise(reference_hash, reference.copy()) is True
    assert rb.verify_rerun_bitwise(reference_hash, reference + 1e-12) is False


def test_verify_rerun_tolerance_distinguishes_small_and_large_perturbations():
    reference = np.array([1.0, 2.0, 3.0])
    small = rb.verify_rerun_tolerance(reference, reference + 1e-10, rtol=1e-6, atol=1e-8)
    large = rb.verify_rerun_tolerance(reference, reference + 1.0, rtol=1e-6, atol=1e-8)
    assert small["matches"] is True
    assert large["matches"] is False
    assert large["max_absolute_difference"] == pytest.approx(1.0)


def test_verify_rerun_tolerance_rejects_mismatched_shapes():
    with pytest.raises(rb.ReproducibilityBundleError):
        rb.verify_rerun_tolerance(np.array([1.0, 2.0]), np.array([1.0]))


def test_reproducibility_bundle_is_wired_into_rpc():
    """As of the research-evidence-package work, `reproducibility_bundle.py`
    IS wired into rpc.py (`research.bundle.build/verify/rerun`) -- this test
    previously asserted the opposite, when the module was still standalone."""
    rpc_source = (Path(__file__).parent.parent / "engine" / "astra" / "rpc.py").read_text()
    assert "reproducibility_bundle" in rpc_source
    assert "research.bundle.build" in rpc_source
    assert "research.bundle.verify" in rpc_source
