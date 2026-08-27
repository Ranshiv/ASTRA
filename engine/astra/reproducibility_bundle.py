"""Signed reproducibility bundles (roadmap item 40, P0).

`manifest.py` already implements most of this item's non-signing half,
confirmed by reading it in full: `Manifest`/`SurveyQuery` record what a
query asked for, `compute_content_hash`/`seal`/`verify` hash the query
(not the timestamp) so identical requests hash identically, and
`capture_environment` records the software/hardware context. This
module does NOT reimplement any of that -- `build_bundle` takes an
already-SEALED `Manifest` and reuses its `content_hash`/`environment`
UNCHANGED.

What `manifest.py` genuinely does not have, confirmed by grep for
"sign"/"signature"/"ed25519"/"public_key" (zero hits) across
`engine/astra` before this session: cryptographic SIGNING (a
verifiable claim of WHO produced a manifest, not just whether it
matches itself) and a BUNDLE concept (packaging a manifest's hash with
environment and experiment-record references into one distributable,
independently-verifiable unit).

**Signing**: Ed25519 (Bernstein, Duif, Lange, Schwabe & Yang 2012,
"High-speed high-security signatures," J. Cryptographic Engineering),
via the `cryptography` package -- newly added to this project's
`research` extra (`engine/pyproject.toml`), confirmed genuinely absent
from every existing dependency list before this session (no pure-stdlib
asymmetric-signature primitive exists in CPython). `_require_
cryptography` lazy-imports it, matching `agn_changepoint._require_
celerite2`'s existing house pattern for an optional research
dependency, and raises `ReproducibilityBundleError` with the same
install instruction shape when absent.

**Bundle vs. manifest, precisely**: `build_bundle` packages a sealed
manifest's `content_hash` (query provenance) with `capture_environment`'s
dict (code/environment identity) and a list of experiment-record
references into one payload, then `compute_bundle_hash` hashes that
whole payload -- a SEPARATE hash from the manifest's own, since a
bundle can carry the same manifest into different experiment contexts.
`sign_bundle` signs `bundle_hash` with an Ed25519 private key;
`verify_bundle` checks BOTH that the recorded `bundle_hash` still
matches recomputing it from the bundle's own fields (tamper detection
on content) AND that the signature verifies against the recorded public
key (provenance verification) -- returning `False`, never raising, on
any failure so a caller can treat "not ready" and "tampered" uniformly.

**Bitwise/tolerance-bounded rerun**: `manifest.Manifest.compute_content_
hash` only hashes the QUERY (survey/release/cone/limit/object-id list),
by design -- its own docstring states a manifest is "a few hundred
kilobytes where a snapshot is tens of gigabytes," so it deliberately
does NOT hash the retrieved DATA itself. `verify_manifest_rerun` reuses
that query-identity check unchanged (a fresh manifest's re-derived hash
should match the bundle's recorded one). For the DATA side of "rerun,"
which needs the actual numeric result to check bitwise or within
tolerance -- genuinely new, since nothing in this codebase hashes or
tolerance-compares result arrays today (confirmed by the same grep) --
`hash_array_bitwise`/`verify_rerun_bitwise` do an exact byte-for-byte
check, and `verify_rerun_tolerance` does a `numpy.allclose`-based
check, reporting the actual max absolute/relative deviation rather than
only a boolean.

Confirmed UNREACHABLE, stated up front: a real live re-fetch of survey
data to check bitwise/tolerance rerun equality against. No network
access exists in this offline codebase/test session to re-query ZTF,
Gaia, or TESS; `reproducibility_bundle_eval.py`'s rerun studies
therefore use synthetic arrays standing in for a "rerun," honestly
labelled as such.

Explicitly NOT done: does not modify `manifest.py` -- `Manifest`,
`SurveyQuery`, `capture_environment`, and `compute_content_hash` are all
called unchanged. Like every other opt-in research module in this
codebase, NOT wired into `rpc.py`, `scoring.WEIGHTS`, or `evidence.py`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .manifest import Manifest


class ReproducibilityBundleError(ValueError):
    """A bundle, signing key, or rerun-verification input was invalid."""


def _require_cryptography():
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey, Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ReproducibilityBundleError(
            "cryptography is not installed; install the 'research' extra "
            "(pip install .[research]) to use reproducibility_bundle.py"
        ) from exc
    return Ed25519PrivateKey, Ed25519PublicKey, InvalidSignature


@dataclass(frozen=True)
class KeyPair:
    private_key: object
    public_key: object

    def public_key_hex(self) -> str:
        from cryptography.hazmat.primitives import serialization
        raw = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
        return raw.hex()


def generate_keypair(*, seed: bytes | None = None) -> KeyPair:
    """A real Ed25519 keypair. `seed` (any bytes) deterministically
    derives the 32-byte private key via SHA-256 -- for reproducible
    TESTS only; a real deployment should omit `seed` and let
    `Ed25519PrivateKey.generate()` draw from OS randomness."""
    Ed25519PrivateKey, _, _ = _require_cryptography()
    if seed is not None:
        private_key = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed).digest())
    else:
        private_key = Ed25519PrivateKey.generate()
    return KeyPair(private_key=private_key, public_key=private_key.public_key())


@dataclass(frozen=True)
class ReproducibilityBundle:
    dataset_id: str
    manifest_content_hash: str
    environment: dict
    experiment_record_refs: tuple[str, ...] = field(default_factory=tuple)
    bundle_hash: str = ""
    signature_hex: str | None = None
    public_key_hex: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def compute_bundle_hash(dataset_id: str, manifest_content_hash: str, environment: dict,
                        experiment_record_refs: tuple[str, ...]) -> str:
    """SHA-256 over a canonical JSON payload -- the whole bundle's
    identity, separate from the manifest's own `content_hash`."""
    payload = {"dataset_id": dataset_id, "manifest_content_hash": manifest_content_hash,
              "environment": environment, "experiment_record_refs": sorted(experiment_record_refs)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_bundle(manifest: Manifest, *, experiment_record_refs: list[str] | None = None) -> ReproducibilityBundle:
    """Packages a SEALED manifest's `content_hash`/`environment`
    (`manifest.py`, UNCHANGED) plus experiment-record references into
    one unsigned bundle."""
    if manifest.content_hash is None:
        raise ReproducibilityBundleError("manifest must be sealed (Manifest.seal()) before bundling")
    refs = tuple(experiment_record_refs or [])
    bundle_hash = compute_bundle_hash(manifest.dataset_id, manifest.content_hash, manifest.environment, refs)
    return ReproducibilityBundle(dataset_id=manifest.dataset_id, manifest_content_hash=manifest.content_hash,
                                 environment=manifest.environment, experiment_record_refs=refs,
                                 bundle_hash=bundle_hash)


def sign_bundle(bundle: ReproducibilityBundle, keypair: KeyPair) -> ReproducibilityBundle:
    """Ed25519-signs `bundle.bundle_hash`, returning a NEW bundle
    (frozen dataclass) with `signature_hex`/`public_key_hex` populated."""
    signature = keypair.private_key.sign(bytes.fromhex(bundle.bundle_hash))
    return dataclasses.replace(bundle, signature_hex=signature.hex(),
                               public_key_hex=keypair.public_key_hex())


def verify_bundle(bundle: ReproducibilityBundle) -> bool:
    """`True` iff the recorded `bundle_hash` still matches recomputing
    it from the bundle's own fields (TAMPER DETECTION on content) AND
    the Ed25519 signature verifies against the recorded public key
    (PROVENANCE VERIFICATION). Never raises -- any failure (missing
    signature, corrupted field, bad signature bytes) returns `False`."""
    if not bundle.signature_hex or not bundle.public_key_hex:
        return False
    recomputed = compute_bundle_hash(bundle.dataset_id, bundle.manifest_content_hash,
                                     bundle.environment, bundle.experiment_record_refs)
    if recomputed != bundle.bundle_hash:
        return False
    try:
        _, Ed25519PublicKey, InvalidSignature = _require_cryptography()
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(bundle.public_key_hex))
        public_key.verify(bytes.fromhex(bundle.signature_hex), bytes.fromhex(bundle.bundle_hash))
        return True
    except Exception:  # noqa: BLE001 - any malformed signature/key/hex is "not verified", not a crash
        return False


def verify_manifest_rerun(bundle: ReproducibilityBundle, fresh_manifest: Manifest) -> dict:
    """Query-provenance half of "rerun verification": a FRESH manifest
    built from re-running the same queries should re-derive the
    identical `compute_content_hash()` (`manifest.py`, UNCHANGED)."""
    fresh_hash = fresh_manifest.compute_content_hash()
    return {"matches": fresh_hash == bundle.manifest_content_hash,
            "bundle_hash": bundle.manifest_content_hash, "fresh_hash": fresh_hash}


def hash_array_bitwise(array: np.ndarray) -> str:
    """SHA-256 over an array's raw bytes -- a cheap, exact fingerprint
    of the ACTUAL DATA a rerun should reproduce bit-for-bit."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_rerun_bitwise(reference_hash: str, rerun_array: np.ndarray) -> bool:
    return hash_array_bitwise(rerun_array) == reference_hash


def verify_rerun_tolerance(reference_array: np.ndarray, rerun_array: np.ndarray, *,
                           rtol: float = 1e-7, atol: float = 1e-9) -> dict:
    """`numpy.allclose`-based rerun check, reporting the actual maximum
    absolute/relative deviation rather than only a boolean -- for
    numeric pipelines where exact bitwise reproduction is not
    guaranteed (e.g. across BLAS/platform floating-point variation)."""
    reference_array = np.asarray(reference_array, dtype=np.float64)
    rerun_array = np.asarray(rerun_array, dtype=np.float64)
    if reference_array.shape != rerun_array.shape:
        raise ReproducibilityBundleError(
            f"reference_array and rerun_array must share a shape, got {reference_array.shape} vs {rerun_array.shape}")
    diff = np.abs(reference_array - rerun_array)
    max_abs_diff = float(np.max(diff)) if diff.size else 0.0
    denom = np.abs(reference_array)
    max_rel_diff = float(np.max(diff / np.where(denom > 0, denom, 1.0))) if diff.size else 0.0
    return {"matches": bool(np.allclose(reference_array, rerun_array, rtol=rtol, atol=atol)),
            "max_absolute_difference": max_abs_diff, "max_relative_difference": max_rel_diff,
            "rtol": rtol, "atol": atol}


__all__ = [
    "ReproducibilityBundleError", "KeyPair", "generate_keypair", "ReproducibilityBundle",
    "compute_bundle_hash", "build_bundle", "sign_bundle", "verify_bundle",
    "verify_manifest_rerun", "hash_array_bitwise", "verify_rerun_bitwise", "verify_rerun_tolerance",
]
