"""Instrument-aware survey digital twin (backlog item 42).

This module's job is deliberately narrow: cadence, noise, and artifact
realism, not astrophysical variability-shape realism. Generating realistic
transient/variability SHAPES already has real machinery in this codebase
(`open_world_injection.py`'s real-patch splicing, `diffusion.py`'s learned
generator); duplicating that here would not be a "digital twin," it would be
a second, worse copy of the same thing. What is genuinely missing is a
model of HOW a survey samples and corrupts whatever signal is underneath:
its cadence (how much of a fixed grid ends up real vs. interpolated, and how
the gaps are shaped), its noise floor, and -- where real data exists -- its
instrumental artifacts.

Every distribution here is FIT from real, already-locally-stored curves via
`tensors.build`/`tensors.resample` (unchanged), never invented. A survey
with too few locally stored curves gets an explicit, stated degradation
(`SurveyProfile.note`), not a silently-fabricated profile -- the same
discipline `docs/DEFERRED.txt` documents throughout this codebase for
missing/insufficient real data.

Real per-cadence artifact patches (`artifact_patches.extract_artifact_patches`,
TESS-only -- ZTF artifacts are still unavailable, see that module's
docstring) are accepted as an OPTIONAL caller-supplied argument rather than
fetched here: this module never downloads anything on its own, matching
`followup.plan`'s "no request is submitted automatically" convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import tensors

DEFAULT_LENGTH = tensors.DEFAULT_LENGTH

# Below this a survey's real local sample is too small to fit a cadence/
# noise profile from; every value is derived from real curves, so there must
# be enough of them for the fit to mean anything.
MIN_CURVES_FOR_PROFILE = 5

# Fraction of synthetic curves that receive a spliced real artifact patch,
# when the caller supplies any. Not calibrated to a measured real artifact
# rate (no survey publishes one directly comparable to this); chosen to be
# common enough that `summary_statistic_distance` can detect its effect.
ARTIFACT_INJECTION_RATE = 0.15


@dataclass
class SurveyProfile:
    """Cadence and noise distributions fit from real stored curves."""

    survey: str
    n_curves_used: int
    mean_coverage: float                  # fraction of grid points backed by a real observation
    gap_run_lengths: tuple[int, ...] = field(default_factory=tuple)
    noise_std: float = float("nan")       # robust per-point noise on the normalised value channel
    length: int = DEFAULT_LENGTH
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "survey": self.survey,
            "n_curves_used": self.n_curves_used,
            "mean_coverage": (round(self.mean_coverage, 4)
                              if np.isfinite(self.mean_coverage) else None),
            "n_gap_runs_sampled": len(self.gap_run_lengths),
            "noise_std": round(self.noise_std, 4) if np.isfinite(self.noise_std) else None,
            "length": self.length,
            "note": self.note,
        }


def _mask_gap_runs(mask_row: np.ndarray) -> list[int]:
    """Lengths of contiguous zero (interpolated-gap) runs in one mask row."""
    runs: list[int] = []
    run_length = 0
    for value in mask_row:
        if value <= 0:
            run_length += 1
        elif run_length:
            runs.append(run_length)
            run_length = 0
    if run_length:
        runs.append(run_length)
    return runs


def fit_survey_profile(survey: str, root=None, limit: int = 500,
                       length: int = DEFAULT_LENGTH) -> SurveyProfile:
    """Fit a `SurveyProfile` from real, already-locally-stored curves.

    Reuses `tensors.build` unchanged: the same uniform-grid resampling every
    sequence model in this codebase already trains on, so a synthetic curve
    sampled from the resulting profile is directly comparable to a real one
    produced the same way.
    """
    batch = tensors.build(survey=survey, length=length, root=root, limit=limit)

    if len(batch) < MIN_CURVES_FOR_PROFILE:
        return SurveyProfile(
            survey=survey, n_curves_used=len(batch), mean_coverage=float("nan"),
            gap_run_lengths=(), noise_std=float("nan"), length=length,
            note=(f"fewer than {MIN_CURVES_FOR_PROFILE} real {survey} curves "
                 "locally stored; profile not fit"),
        )

    masks = batch.values[:, 1, :]
    values = batch.values[:, 0, :]
    mean_coverage = float(masks.mean())

    gap_run_lengths: list[int] = []
    for mask_row in masks:
        gap_run_lengths.extend(_mask_gap_runs(mask_row))

    noise_samples: list[float] = []
    for value_row, mask_row in zip(values, masks):
        valid = mask_row > 0
        observed = value_row[valid]
        if len(observed) > 2:
            # Median absolute successive difference, scaled to a Gaussian
            # sigma -- the same robust noise estimator `features.bocpd`
            # already uses for exactly the same reason: a first difference
            # cancels slow smooth trend, leaving point-to-point noise.
            diffs = np.diff(observed)
            noise_samples.append(1.4826 * float(np.median(np.abs(diffs))) / np.sqrt(2.0))

    noise_std = float(np.median(noise_samples)) if noise_samples else float("nan")

    return SurveyProfile(
        survey=survey, n_curves_used=len(batch), mean_coverage=mean_coverage,
        gap_run_lengths=tuple(gap_run_lengths), noise_std=noise_std, length=length,
    )


def _sample_baseline(length: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth, low-order "quiet source" signal -- shape only.

    Deliberately simple: this module measures cadence/noise/artifact
    realism, not variability-shape realism (see module docstring). A fixed
    small number of random sinusoidal terms gives a smoothly-varying but
    non-trivial baseline without claiming to model any real astrophysical
    process.
    """
    grid = np.linspace(0.0, 2.0 * np.pi, length)
    signal = np.zeros(length, dtype=np.float64)
    for term in range(1, 4):
        amplitude = rng.uniform(0.1, 0.4) / term
        frequency = rng.uniform(0.5, 1.5) * term
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += amplitude * np.sin(frequency * grid + phase)
    return tensors.normalise(signal)


def _sample_mask(length: int, profile: SurveyProfile,
                 rng: np.random.Generator) -> np.ndarray:
    """A validity mask matching the fitted mean coverage and gap-run sizes.

    Gaps of empirically-sampled length are punched into an otherwise-full
    grid at random positions until the target zeroed fraction is reached.
    Without a usable profile (too little real data, or no gaps observed at
    all), the grid stays fully valid -- an unrealistically dense but never
    fabricated fallback.
    """
    mask = np.ones(length, dtype=np.float32)
    if not profile.gap_run_lengths or not np.isfinite(profile.mean_coverage):
        return mask

    target_zero = int(round((1.0 - profile.mean_coverage) * length))
    if target_zero <= 0:
        return mask

    guard = 0
    while int((mask == 0).sum()) < target_zero and guard < 20 * length:
        guard += 1
        run = int(rng.choice(profile.gap_run_lengths))
        run = max(1, min(run, length))
        start = int(rng.integers(0, max(1, length - run + 1)))
        mask[start:start + run] = 0.0
    return mask


def sample_synthetic_curve(profile: SurveyProfile, length: int | None = None,
                           rng: np.random.Generator | None = None,
                           artifact_patches: np.ndarray | None = None) -> np.ndarray:
    """One synthetic (2, length) curve: fitted cadence + noise, plus an
    optional spliced real artifact patch.

    Same (value, mask) shape `tensors.resample` produces, so the result is a
    drop-in row for `tensors.SequenceBatch`/`evaluate.build_injected`/
    `evaluate.compare_on_sequences` -- no separate representation invented.
    """
    length = length or profile.length
    rng = rng if rng is not None else np.random.default_rng()

    baseline = _sample_baseline(length, rng)
    mask = _sample_mask(length, profile, rng)
    noise_std = profile.noise_std if np.isfinite(profile.noise_std) else 0.1
    value = (baseline + rng.normal(0.0, noise_std, length)).astype(np.float32)
    value = value * mask

    if artifact_patches is not None and len(artifact_patches) \
            and rng.random() < ARTIFACT_INJECTION_RATE:
        patch = artifact_patches[int(rng.integers(0, len(artifact_patches)))]
        patch_value, patch_mask = patch[0], patch[1]
        patch_length = patch_value.shape[-1]
        if patch_length <= length:
            start = int(rng.integers(0, length - patch_length + 1))
            value[start:start + patch_length] = patch_value
            mask[start:start + patch_length] = np.maximum(
                mask[start:start + patch_length], patch_mask)

    return np.stack([value, mask.astype(np.float32)], axis=0)


def sample_synthetic_batch(profile: SurveyProfile, n: int, length: int | None = None,
                           seed: int = 42, artifact_patches: np.ndarray | None = None
                           ) -> tensors.SequenceBatch:
    """A batch of synthetic curves as a `tensors.SequenceBatch`, directly
    comparable to `tensors.build(survey=profile.survey, ...)`'s real output.
    """
    length = length or profile.length
    rng = np.random.default_rng(seed)
    rows = [sample_synthetic_curve(profile, length, rng, artifact_patches)
           for _ in range(n)]
    values = (np.stack(rows).astype(np.float32) if rows
             else np.empty((0, 2, length), dtype=np.float32))
    identities = [
        {"object_id": f"synthetic_{profile.survey}_{i}", "survey": profile.survey,
         "synthetic": "1"}
        for i in range(n)
    ]
    return tensors.SequenceBatch(values=values, identities=identities,
                                 length=length, mode="synthetic")
