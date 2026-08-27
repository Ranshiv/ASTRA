"""Real ZTF instrumental-artifact patches, from real per-epoch `catflags`
(backlog item 42 follow-up: the digital twin's stated ZTF-artifact gap).

`artifact_patches.py` reads TESS's own real per-cadence `QUALITY` bitmask,
categorised via `lightkurve.utils.TessQualityFlags` -- a real, authoritative
table from a bundled library. ZTF has no equivalent bundled dependency (no
`ztfquery` in this project's requirements), so this module deliberately
does NOT invent a TESS-style multi-category breakdown for ZTF. It uses only
the ONE real, already-authoritative value this codebase already sends to
IRSA on every other ZTF request: `surveys/ztf.py`'s `DEFAULT_CATFLAGS_MASK`
(32768, "a suspect or contaminated epoch", the exact value IRSA's own
service contract defines). `CATEGORY_NAMES` is therefore coarser than
TESS's five categories -- `("clean", "flagged")` -- stated as coarser, not
glossed over as equivalent.

This closes the gap `docs/DEFERRED.txt`/`artifact_bank.py` previously
described as requiring "changing ZTF acquisition codebase-wide": that
was true for the DEFAULT acquisition path, but recovering the epochs only
needed a new, additive method
(`surveys.ztf.ZTFConnector.fetch_light_curves_with_quality`) that requests
the SAME endpoint with a different `BAD_CATFLAGS_MASK` value.
`fetch_light_curves` itself -- and every existing caller of it -- is
completely unchanged.
"""

from __future__ import annotations

import numpy as np

CATEGORY_NAMES: tuple[str, ...] = ("clean", "flagged")

MIN_RUN_LENGTH = 3
DEFAULT_MAX_PATCHES_PER_CATEGORY = 50


def categorize_catflags(catflags: int) -> str | None:
    """One real ZTF `catflags` word -> a category, or None when clean.

    Matches `artifact_patches.categorize_quality`'s `None`-means-clean
    convention exactly, so callers of either module handle "no flag" the
    same way.
    """
    return "flagged" if int(catflags) != 0 else None


def extract_ztf_artifact_patches(
    quality_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    patch_length: int = 32, min_run_length: int = MIN_RUN_LENGTH,
    max_patches_per_category: int = DEFAULT_MAX_PATCHES_PER_CATEGORY,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Real (value, mask) patches centred on real flagged-epoch runs, plus
    real clean contrast windows, from real ZTF photometry.

    `quality_curves` is a list of `(value, mask, catflags)` triples, one per
    curve -- the shape
    `surveys.ztf.ZTFConnector.fetch_light_curves_with_quality` plus
    `tensors.resample`-style normalisation naturally produces (the caller
    resamples/normalises the light curve itself; this function only slices
    windows and reads `catflags`, mirroring
    `artifact_patches.extract_artifact_patches`'s split between "the
    caller reads the real product" and "this function extracts patches").

    Returns `(patches, labels)`: `patches` is `(n, 2, patch_length)`, the
    same MAD-normalised value/validity-mask shape every sequence model in
    this codebase already consumes; `labels` is `(n,)` integer indices into
    `CATEGORY_NAMES`.
    """
    rng = np.random.default_rng(seed)
    patches: list[np.ndarray] = []
    labels: list[int] = []
    counts = {name: 0 for name in CATEGORY_NAMES}

    for value, mask, catflags in quality_curves:
        n_points = len(value)
        if n_points < patch_length or len(catflags) != n_points:
            continue

        categories = [categorize_catflags(word) for word in catflags]

        def _take(start: int, category_index: int) -> None:
            end = start + patch_length
            patches.append(np.stack([value[start:end], mask[start:end]]))
            labels.append(category_index)

        index = 0
        while index < n_points:
            category = categories[index]
            run_end = index
            while run_end < n_points and categories[run_end] == category:
                run_end += 1
            run_length = run_end - index
            if (category is not None and run_length >= min_run_length
                    and counts[category] < max_patches_per_category):
                center = (index + run_end) // 2
                start = max(0, min(n_points - patch_length, center - patch_length // 2))
                _take(start, CATEGORY_NAMES.index(category))
                counts[category] += 1
            index = run_end

        target_clean = sum(counts[name] for name in CATEGORY_NAMES if name != "clean")
        clean_starts = [
            start for start in range(0, n_points - patch_length + 1)
            if all(c is None for c in categories[start:start + patch_length])
        ]
        if clean_starts and target_clean:
            chosen = rng.choice(clean_starts, size=min(target_clean, len(clean_starts)),
                                replace=False)
            for start in chosen:
                if counts["clean"] >= max_patches_per_category:
                    break
                _take(int(start), CATEGORY_NAMES.index("clean"))
                counts["clean"] += 1

    if not patches:
        return (np.empty((0, 2, patch_length), dtype=np.float32),
               np.empty((0,), dtype=np.int64))
    return np.stack(patches).astype(np.float32), np.array(labels, dtype=np.int64)


def fetch_and_extract(sources, patch_length: int = 32,
                      min_run_length: int = MIN_RUN_LENGTH,
                      max_patches_per_category: int = DEFAULT_MAX_PATCHES_PER_CATEGORY,
                      seed: int = 42, connector=None) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: real ZTF sources in, real artifact patches out.

    `connector` defaults to a real `ZTFConnector` but accepts an injected
    fake for tests, mirroring `open_world_eval.assemble_held_out_set`'s
    connector-injection convention. A source whose fetch fails or yields no
    usable curve is skipped rather than aborting the whole batch, matching
    `download_reference_tpfs`'s same discipline in `artifact_patches.py`.
    """
    from . import tensors
    from .surveys.ztf import ZTFConnector

    connector = connector or ZTFConnector()
    quality_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for source in sources:
        try:
            pairs = connector.fetch_light_curves_with_quality(source)
        except Exception:  # noqa: BLE001 - one bad source must not abort the batch
            continue
        for curve, catflags in pairs:
            # NOT `tensors.resample`: resampling onto a uniform grid would
            # break the 1:1 index alignment `catflags` needs with the
            # curve's own real epochs (unlike TESS's per-cadence QUALITY
            # array, which already lines up with a raw, un-resampled cube).
            # `tensors.normalise` alone keeps every real point at its own
            # index, mask=1 throughout since every point here is a real
            # observation, not an interpolated gap.
            if len(curve.value) != len(catflags) or len(curve.value) < patch_length:
                continue
            normalised = tensors.normalise(curve.value)
            mask = np.ones(len(curve.value), dtype=np.float32)
            quality_curves.append((normalised, mask, catflags))

    return extract_ztf_artifact_patches(
        quality_curves, patch_length=patch_length, min_run_length=min_run_length,
        max_patches_per_category=max_patches_per_category, seed=seed)
