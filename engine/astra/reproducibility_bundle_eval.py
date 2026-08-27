"""Evaluation studies for `reproducibility_bundle.py`, split purely to
keep each file under this project's 500-line guideline.

`evaluate_signed_bundle_round_trip` is the concrete round-trip the
roadmap item names: build a bundle from a real sealed `manifest.
Manifest`, sign it, verify it succeeds, then mutate one field and
confirm verification FAILS -- tamper detection and provenance
verification checked directly via the module's own functions, not
merely asserted.

`evaluate_rerun_bitwise_and_tolerance_synthetic` is the "bitwise or
tolerance-bounded rerun" metric: SYNTHETIC arrays stand in for a real
rerun's output (no live archive access exists in this offline session to
produce a genuine rerun -- honestly labelled, not glossed over), checking
`verify_rerun_bitwise` accepts an identical array and rejects a changed
one, and `verify_rerun_tolerance` accepts a within-tolerance perturbation
while reporting the real deviation magnitude, and rejects one outside it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .manifest import Manifest, SurveyQuery
from .reproducibility_bundle import (
    ReproducibilityBundleError, build_bundle, generate_keypair, hash_array_bitwise,
    sign_bundle, verify_bundle, verify_rerun_bitwise, verify_rerun_tolerance,
)


def _sealed_manifest(dataset_id: str = "eval_dataset") -> Manifest:
    manifest = Manifest.create(dataset_id)
    manifest.add(SurveyQuery(survey="ZTF", release="dr24", ra_deg=180.0, dec_deg=22.0,
                            radius_arcsec=10.0, limit=100, object_ids=["a", "b", "c"]))
    return manifest.seal()


def evaluate_signed_bundle_round_trip(seed: bytes = b"eval-seed") -> dict:
    """Build, sign, verify -- then mutate one field and confirm
    verification fails, for each of: the bundle hash, the environment
    dict, and the experiment-record refs."""
    manifest = _sealed_manifest()
    keypair = generate_keypair(seed=seed)
    bundle = build_bundle(manifest, experiment_record_refs=["exp-001", "exp-002"])
    signed = sign_bundle(bundle, keypair)

    valid = verify_bundle(signed)
    unsigned_rejected = not verify_bundle(bundle)

    tampered_hash = dataclasses.replace(signed, bundle_hash="0" * 64)
    tampered_env = dataclasses.replace(signed, environment={**signed.environment, "extra": "injected"})
    tampered_refs = dataclasses.replace(signed, experiment_record_refs=("exp-999",))

    return {
        "valid_signature_verifies": valid,
        "unsigned_bundle_is_rejected": unsigned_rejected,
        "tampered_hash_is_rejected": not verify_bundle(tampered_hash),
        "tampered_environment_is_rejected": not verify_bundle(tampered_env),
        "tampered_experiment_refs_rejected": not verify_bundle(tampered_refs),
    }


def evaluate_rerun_bitwise_and_tolerance_synthetic(seed: int = 42) -> dict:
    """SYNTHETIC "reference" and "rerun" arrays -- see module docstring
    for why real rerun data is unreachable in this offline session."""
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=64)
    identical_rerun = reference.copy()
    changed_rerun = reference.copy()
    changed_rerun[0] += 1.0

    reference_hash = hash_array_bitwise(reference)
    bitwise_identical_ok = verify_rerun_bitwise(reference_hash, identical_rerun)
    bitwise_changed_rejected = not verify_rerun_bitwise(reference_hash, changed_rerun)

    tiny_perturbation = reference + rng.normal(scale=1e-10, size=64)
    large_perturbation = reference + 1.0
    within_tolerance = verify_rerun_tolerance(reference, tiny_perturbation, rtol=1e-6, atol=1e-8)
    outside_tolerance = verify_rerun_tolerance(reference, large_perturbation, rtol=1e-6, atol=1e-8)

    return {
        "bitwise_identical_ok": bitwise_identical_ok,
        "bitwise_changed_rejected": bitwise_changed_rejected,
        "tolerance_check_within": within_tolerance,
        "tolerance_check_outside": outside_tolerance,
        "tolerance_correctly_distinguishes": within_tolerance["matches"] and not outside_tolerance["matches"],
    }


__all__ = [
    "evaluate_signed_bundle_round_trip", "evaluate_rerun_bitwise_and_tolerance_synthetic",
]
