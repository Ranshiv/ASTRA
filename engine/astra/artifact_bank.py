"""Real-artifact hard-negative bank and domain adaptation (roadmap item
33, P0).

This is a THIRD, distinct sense of "artifact" in this codebase's
vocabulary, deliberately named to avoid colliding with the other two:
`artifact.py` is a heuristic real/artifact verdict system operating on
light-curve-DERIVED features (`ArtifactAssessment`, calibrated via
synthetic injection); `artifact_patches.py` is a real (non-synthetic)
TESS instrumental-defect PATCH extractor, keyed to the real per-cadence
SPOC `QUALITY` bitmask via `lightkurve.utils.TessQualityFlags`. This
module builds directly on `artifact_patches.py` -- reusing
`extract_artifact_patches`/`CATEGORY_NAMES`/`categorize_quality`
unchanged -- and adds the two things that module does not have: real
per-patch camera/CCD/night metadata, and a bank-plus-domain-adaptation
layer so a classifier trained on some cameras/nights generalizes to
others.

UPDATE 2026-08-26: the flagged-EPOCH half of the constraint below is no
longer true -- `surveys.ztf.ZTFConnector.fetch_light_curves_with_quality`
(additive, `fetch_light_curves` unchanged) plus `ztf_artifact_patches.py`
now recover real ZTF `catflags`-flagged epochs, at a coarse `("clean",
"flagged")` granularity (see that module's docstring for why no richer
category set is invented). What ZTF still lacks, and what this specific
module's cross-survey domain-adaptation arm actually needs beyond
`artifact_patches.py`'s plain patch/label pairs, is real per-patch
CAMERA/CCD/NIGHT metadata -- IRSA's `nph_light_curves` endpoint returns
photometry and `catflags`, not per-epoch camera/CCD/quadrant identifiers,
so `coral_align`'s domain grouping still has nothing real to group ZTF
patches BY. The ZTF/cross-survey arm below therefore remains explicitly
synthetic, for that narrower, still-real reason -- not the broader "no
real data at all" claim the original entry made.

Original entry, left for the history: one real, already-documented
constraint carries forward from `artifact_patches.py` and is not silently
worked around here: **ZTF has no real subtraction-artifact data reachable
by this codebase.** `surveys/ztf.py` sends `BAD_CATFLAGS_MASK=32768` on
every IRSA request, so bad-subtraction epochs are stripped server-side
before ASTRA ever sees them, and `artifact.py` already investigated and
rejected the one candidate real dataset (SNAD's ZTF DR3 artifact set: image
cutouts only, no object IDs, no light-curve data) for the same reason.
Recovering real ZTF artifacts means changing ZTF acquisition codebase-wide,
out of scope for one research module. This module is therefore genuinely
real for TESS (camera/CCD/night all come from a real downloaded TPF's own
FITS header and cadence timestamps) and explicitly synthetic for the ZTF/
cross-survey arm, which lives in `artifact_bank_eval.py` and is labelled
as such throughout, not glossed over.

**Domain adaptation** is CORAL (CORrelation ALignment; Sun & Saenko 2016,
"Return of Frustratingly Easy Domain Adaptation," AAAI): a closed-form,
unsupervised feature-alignment technique needing no new dependency --
whiten the source domain's feature covariance, then re-color with the
target domain's covariance. `coral_align` implements the textbook
transform via eigendecomposition (`numpy.linalg.eigh`, valid since a
covariance matrix is symmetric positive semi-definite): `source_aligned =
(source - mean_s) @ Cs^(-1/2) @ Ct^(1/2) + mean_t`, with a small ridge
term on each covariance for numerical stability.

**Features** are a small, bounded, hand-rolled statistics vector per
`(value, mask)` patch (mean, std, MAD, max absolute deviation, a
standardized third moment, and the longest run of masked-out points) fed
into `sklearn.linear_model.LogisticRegression` -- matching this
codebase's established "simple features into `LogisticRegression`" house
style for a classifier (`multimodal_eval.linear_probe_macro_f1`,
`sn_classification_eval._macro_f1`, `significance.fit_selection_model`),
not a new torch deep-net architecture.

Like every other opt-in research module in this codebase, NOT wired into
`rpc.py`, `scoring.WEIGHTS`, or `evidence.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import artifact_patches


class ArtifactBankError(ValueError):
    """A patch, feature, or domain-adaptation input/computation was invalid."""


@dataclass(frozen=True)
class PatchRecord:
    """One real TESS artifact (or clean-contrast) patch plus its real
    provenance metadata. `camera`/`ccd`/`night` stay `None` -- never
    fabricated -- when a header field could not be parsed."""

    category: str
    sector: int | None
    camera: int | None
    ccd: int | None
    night: str | None
    patch: np.ndarray  # shape (2, patch_length): value, mask


def extract_camera_ccd(tpf_path: str | Path) -> tuple[int | None, int | None]:
    """Real `CAMERA`/`CCD` values read straight from a downloaded TPF's
    primary FITS header, independent of `tess_pixels.find_sectors` (which
    only keeps `sector` from the live TESScut lookup, dropping camera/ccd
    even though the same lookup response carries them). SPOC TPF primary
    headers document `CAMERA`/`CCD` integer keywords; this has NOT yet
    been confirmed against a live downloaded file this session (no live
    network access at implementation time) -- treat the keyword names as
    an unverified assumption until checked live (see the module's live
    smoke test), the same "verify before trusting" discipline every prior
    connector-column addition in this codebase follows. A missing or
    malformed value degrades to `None`, never a fabricated number.
    """
    from astropy.io import fits

    try:
        header = fits.getheader(Path(tpf_path), ext=0)
    except Exception:  # noqa: BLE001 - a corrupt/unreadable file yields unknown metadata
        return (None, None)

    def _parse(key: str) -> int | None:
        raw = header.get(key)
        if raw in (None, ""):
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    return (_parse("CAMERA"), _parse("CCD"))


def night_bucket(time_bjd: float) -> str | None:
    """A calendar-date label for one real BJD timestamp.

    Takes an already-fully-referenced BJD (the shape `tess_pixels.
    read_tpf_cube`'s own `time` array is in -- `_time_to_bjd` adds the
    `BJDREFI`/`BJDREFF` header reference, or the standard `2457000.0`
    fallback, before this function ever sees a value; checked against that
    function's real behaviour before writing this one, so this does NOT
    re-add the offset itself). Uses TDB, the timescale TESS timestamps are
    published in, then buckets by the UTC calendar date.
    """
    if not np.isfinite(time_bjd):
        return None
    from astropy.time import Time

    try:
        moment = Time(float(time_bjd), format="jd", scale="tdb")
        return moment.utc.datetime.date().isoformat()
    except Exception:  # noqa: BLE001 - an out-of-range/malformed time yields unknown, not a crash
        return None


def build_patch_bank(tpf_paths: list[str | Path], *, patch_length: int = 32,
                     min_run_length: int = artifact_patches.MIN_RUN_LENGTH,
                     max_patches_per_category: int =
                     artifact_patches.DEFAULT_MAX_PATCHES_PER_CATEGORY,
                     seed: int = 42) -> list[PatchRecord]:
    """Real patches plus real per-file provenance metadata.

    Calls `artifact_patches.extract_artifact_patches` UNCHANGED once PER
    FILE (rather than once over the whole list, which is how that
    function is normally called) specifically so sector/camera/ccd/night
    can be attached per patch. `night` is one value per TPF FILE (from its
    first finite cadence timestamp), not a true per-patch timestamp --
    `extract_artifact_patches` does not expose each patch's cadence
    window, and reimplementing its windowing loop here just to recover
    that would duplicate real, already-tested logic. A stated, bounded
    approximation: patches from one TPF file span at most one TESS
    sector's ~27-day baseline, so a per-file night is coarser than a
    per-patch one, not wrong in kind.
    """
    from . import tess_pixels

    records: list[PatchRecord] = []
    for raw_path in tpf_paths:
        path = Path(raw_path)
        try:
            cube_data = tess_pixels.read_tpf_cube(path)
        except Exception:  # noqa: BLE001 - a corrupt TPF must not stop the whole bank build
            continue

        sector = cube_data.get("summary", {}).get("sector")
        camera, ccd = extract_camera_ccd(path)
        time = np.asarray(cube_data.get("time", []), dtype=np.float64)
        finite_time = time[np.isfinite(time)]
        night = night_bucket(float(finite_time[0])) if finite_time.size else None

        patches, labels = artifact_patches.extract_artifact_patches(
            [path], patch_length=patch_length, min_run_length=min_run_length,
            max_patches_per_category=max_patches_per_category, seed=seed)
        for patch, label in zip(patches, labels):
            records.append(PatchRecord(
                category=artifact_patches.CATEGORY_NAMES[int(label)],
                sector=sector, camera=camera, ccd=ccd, night=night, patch=patch))
    return records


def patch_features(patch: np.ndarray) -> np.ndarray:
    """A 6-statistic feature vector: mean, std, MAD, max absolute
    deviation, a standardized third moment, and the longest run of
    masked-out points -- computed over only the valid (`mask > 0.5`)
    points. All-invalid patches return an all-zero vector rather than
    NaN, a neutral fallback a downstream classifier can still consume."""
    patch = np.asarray(patch, dtype=np.float64)
    if patch.shape[0] != 2:
        raise ArtifactBankError(f"patch must have shape (2, length), got {patch.shape}")

    values, mask = patch[0], patch[1]
    valid = mask > 0.5
    if not np.any(valid):
        return np.zeros(6, dtype=np.float64)

    v = values[valid]
    mean = float(np.mean(v))
    std = float(np.std(v))
    mad = float(np.median(np.abs(v - np.median(v))))
    max_abs_dev = float(np.max(np.abs(v - mean)))
    skew = float(np.mean(((v - mean) / std) ** 3)) if std > 1e-12 else 0.0

    invalid = ~valid
    longest_run = 0
    current = 0
    for flag in invalid:
        current = current + 1 if flag else 0
        longest_run = max(longest_run, current)

    return np.array([mean, std, mad, max_abs_dev, skew, float(longest_run)], dtype=np.float64)


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def _matrix_inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


def coral_align(source_features: np.ndarray, target_features: np.ndarray, *,
                eps: float = 1e-6) -> np.ndarray:
    """CORrelation ALignment (Sun & Saenko 2016): whitens `source_features`
    by its own covariance, then re-colors with `target_features`'s
    covariance, so a classifier trained on the aligned source generalizes
    better to the target domain's feature statistics."""
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ArtifactBankError("source_features and target_features must be 2-D")
    if source_features.shape[1] != target_features.shape[1]:
        raise ArtifactBankError(
            "source_features and target_features must share the same feature dimension")
    if len(source_features) < 2 or len(target_features) < 2:
        raise ArtifactBankError("CORAL needs at least 2 samples per domain to estimate a covariance")

    n_features = source_features.shape[1]
    source_mean = source_features.mean(axis=0)
    target_mean = target_features.mean(axis=0)

    cov_source = np.atleast_2d(np.cov(source_features, rowvar=False)) + eps * np.eye(n_features)
    cov_target = np.atleast_2d(np.cov(target_features, rowvar=False)) + eps * np.eye(n_features)

    whitened = (source_features - source_mean) @ _matrix_inv_sqrt(cov_source)
    return whitened @ _matrix_sqrt(cov_target) + target_mean


def train_hard_negative_classifier(features: np.ndarray, labels: np.ndarray, *, seed: int = 42):
    """Thin wrapper over `sklearn.linear_model.LogisticRegression`,
    matching `multimodal_eval.linear_probe_macro_f1`'s house convention of
    a simple classifier over extracted features."""
    from sklearn.linear_model import LogisticRegression

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if len(features) != len(labels):
        raise ArtifactBankError("features and labels must have the same length")
    if len(np.unique(labels)) < 2:
        raise ArtifactBankError("labels must contain at least two classes to train a classifier")

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(features, labels)
    return model


__all__ = [
    "ArtifactBankError", "PatchRecord", "extract_camera_ccd", "night_bucket",
    "build_patch_bank", "patch_features", "coral_align", "train_hard_negative_classifier",
]
