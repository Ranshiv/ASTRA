"""Open-world transient injection: real-curve splicing of diffusion-sampled
morphology patches (backlog item 14).

`evaluate.py`'s injection-recovery harness only ever injects 4 hand-coded
shapes (`evaluate.ANOMALY_KINDS`): a fast-rise/exponential-decay flare, a
flat-bottomed eclipse dip, a persistent step, and literal Gaussian noise.
Winning there proves sensitivity to those specific shapes -- this module is
this project's answer to `docs/DEFERRED.txt`'s own
`[KNOWN] Injection-recovery measures the anomalies that were injected` entry:
instead of a hand-coded formula, `inject_generative` splices a patch SAMPLED
from a trained `diffusion.py` denoiser -- a shape nobody wrote a formula for.

Two things keep this honestly "real, not synthetic," matching the backlog
item's "real survey noise/artifacts" half:
1. The diffusion generator is trained on REAL light-curve windows
   (`extract_real_patches`), not synthetic curves -- it learns what real
   variability shape looks like, broader than "anomaly-only," so a sampled
   patch is not a remix of the four hand-designed kinds.
2. Generated patches are spliced onto REAL stored baseline sequences (real
   cadence, real gap-validity mask, real per-point noise already present in
   the baseline) -- never onto a synthetic baseline.

What stays explicitly out of scope: genuine instrumental-artifact material
(cosmic rays, satellite trails, bad columns). `artifact.py` already
investigated and rejected the one candidate real dataset for this
(SNAD's ZTF DR3 artifact set -- image cutouts only, no object IDs, no
light-curve data) for the same reason it would fail here; that gap is not
re-investigated, just inherited honestly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .evaluate import InjectionResult

# Same "not too close to either edge" range evaluate.inject() uses for its
# own hand-designed shapes, reused here for consistency.
MIN_START_FRACTION = 0.1
MAX_START_FRACTION = 0.8


def extract_real_patches(survey: str | None = None, limit: int = 2000,
                         patch_length: int = 32, sequence_length: int = 256,
                         root: Path | None = None,
                         exclude_object_ids: set[str] | None = None,
                         seed: int = 42) -> np.ndarray:
    """Random contiguous windows of real, already-resampled light curves.

    Reuses `tensors.resample()` (MAD-normalised value + gap-validity mask,
    already tested) rather than writing a second resampling path -- each
    patch is a real contiguous slice of a real resampled curve, so it
    carries real relative sampling/mask structure within that window, not a
    synthetic construction.

    `exclude_object_ids` removes any object reserved for the held-out
    real-transient evaluation set (`open_world_eval.py`), so the generator
    never trains on data it will later be judged on recovering.
    """
    from . import config, store, tensors

    root = root or config.PATHS.datasets
    exclude = exclude_object_ids or set()
    rng = np.random.default_rng(seed)
    search_root = root / survey.upper() if survey else root

    if not search_root.exists():
        return np.empty((0, 2, patch_length), dtype=np.float32)

    patches: list[np.ndarray] = []
    for path in sorted(search_root.rglob("*.parquet")):
        if len(patches) >= limit:
            break
        try:
            curve = store.read_curve(path)
        except Exception:  # noqa: BLE001 - a corrupt file must not stop extraction
            continue
        if curve.source.object_id in exclude:
            continue

        sequence = tensors.resample(curve, length=sequence_length)
        if sequence is None:
            continue

        max_start = sequence_length - patch_length
        if max_start < 1:
            continue
        start = int(rng.integers(0, max_start))
        patches.append(sequence[:, start:start + patch_length])

    if not patches:
        return np.empty((0, 2, patch_length), dtype=np.float32)
    return np.stack(patches).astype(np.float32)


def _splice_patch(sequence: np.ndarray, patch_value: np.ndarray,
                  start: int, strength: float) -> np.ndarray:
    """Add a generated patch into a real sequence at `start`, respecting the
    sequence's own gap-validity mask -- same "signal only counts where the
    curve was actually observed" convention `evaluate.inject()` uses."""
    out = sequence.copy()
    values, mask = out[0], out[1]
    length = len(values)

    end = min(start + len(patch_value), length)
    span = end - start
    if span > 0:
        values[start:end] += strength * patch_value[:span]

    out[0] = values * mask
    return out


def inject_generative(sequence: np.ndarray, generator, diffusion_cfg,
                      rng: np.random.Generator, strength: float = 1.0,
                      device=None) -> np.ndarray:
    """Splice one diffusion-sampled patch into a real sequence.

    `strength` defaults to 1.0, not `evaluate.inject()`'s 6.0: a sampled
    patch already carries a realistic amplitude (it was trained on real,
    MAD-normalised curves), unlike the hand-designed shapes which needed an
    explicit strength multiplier to reach a "clear but not absurd" scale.
    """
    from . import diffusion_train as diff

    length = len(sequence[0])
    low = int(length * MIN_START_FRACTION)
    high = max(low + 1, int(length * MAX_START_FRACTION))
    start = int(rng.integers(low, high))

    seed = int(rng.integers(0, 2**31 - 1))
    patch = diff.sample(generator, 1, diffusion_cfg, device=device, seed=seed)[0]
    return _splice_patch(sequence, patch, start, strength)


def build_injected_open_world(values: np.ndarray, identities: list[dict],
                              generator, diffusion_cfg,
                              fraction: float = 0.1, strength: float = 1.0,
                              seed: int = 42, device=None) -> InjectionResult:
    """Generative-injection analogue of `evaluate.build_injected()`.

    Produces the exact same `InjectionResult` shape, so it can be handed to
    `evaluate.compare_on_sequences()` unmodified -- that function only ever
    consumes an already-built `InjectionResult`, it never re-derives the
    injection itself.

    Patches for every chosen row are sampled in ONE batched call (rather
    than once per row) since diffusion sampling is the genuinely expensive
    part of this module (`cfg.timesteps` forward passes per batch, not per
    row) -- batching keeps that cost from scaling with the injected
    fraction's row count on top of its own iteration count.
    """
    from . import diffusion_train as diff

    rng = np.random.default_rng(seed)
    n = len(values)

    out = values.copy()
    labels = np.zeros(n, dtype=int)
    kinds = [""] * n

    if n == 0:
        return InjectionResult(out, labels, kinds, identities)

    count = max(1, int(round(n * fraction)))
    chosen = rng.choice(n, size=min(count, n), replace=False)

    length = values.shape[-1]
    low = int(length * MIN_START_FRACTION)
    high = max(low + 1, int(length * MAX_START_FRACTION))
    starts = rng.integers(low, high, size=len(chosen))

    sample_seed = int(rng.integers(0, 2**31 - 1))
    patches = diff.sample(generator, len(chosen), diffusion_cfg, device=device,
                          seed=sample_seed)

    for position, index in enumerate(chosen):
        out[index] = _splice_patch(values[index], patches[position],
                                   int(starts[position]), strength)
        labels[index] = 1
        kinds[index] = "generative"

    return InjectionResult(out, labels, kinds, identities)
