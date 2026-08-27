"""Real TESS instrumental-artifact patches (backlog item 14, gap 1).

`open_world_injection.py` trains its diffusion generator on ordinary,
unflagged real light-curve windows -- genuine variability shape, but not
genuine INSTRUMENTAL DEFECTS (cosmic rays, stray light, bad pointing).
`artifact.py` already investigated and rejected the one candidate real
defect dataset (SNAD's ZTF DR3 artifact set: image cutouts only, no object
IDs, no light-curve data) for exactly this purpose. This module finds a
different real source: TESS's own real per-cadence `QUALITY` bitmask,
carried on every target-pixel file MAST serves.

Two things make this genuinely real, not another synthetic construction:
category names and bit values come from `lightkurve.utils.TessQualityFlags`
(already a core dependency), the same authoritative table SPOC-pipeline
documentation defines -- not hand-recalled constants; and the flagged
CADENCES themselves are real telescope data, read straight from a real
downloaded TPF (`tess_pixels.read_tpf_cube`).

The reason this reads TPFs directly instead of a stored light curve:
`tess_pixels.persist_photometry` drops the quality array when it builds the
`LightCurve` it persists (`store.SCHEMA` has no per-point flag column, and
changing it would touch the most load-bearing files in this codebase for a
research module's sake) -- confirmed by reading both files before writing
this one. The quality signal only ever exists in the raw TPF and in
`extract_photometry`'s transient return value, so this module reads the
raw TPF, the same real product `read_tpf_cube` already exposes.

UPDATE 2026-08-26: the claim below (ZTF artifacts out of reach) turned out
to assume more than was true -- it conflated "the DEFAULT acquisition path
strips them" with "recovering them needs a codebase-wide change." Only the
former is real. `surveys/ztf.py`'s `ZTFConnector.fetch_light_curves_with_quality`
is a new, ADDITIVE method requesting the same IRSA endpoint with
`BAD_CATFLAGS_MASK=0` instead of the default 32768, recovering the real
`catflags` IRSA already sends on every response; `fetch_light_curves`
itself is completely unchanged. `ztf_artifact_patches.py` is the ZTF sibling
of this module, with a deliberately coarser `("clean", "flagged")` category
set (no bundled library publishes a ZTF catflags bit-name table the way
`lightkurve.utils.TessQualityFlags` does for TESS, so no multi-category
breakdown is invented). See that module's docstring for the rest.

Original entry, left for the history: explicitly still out of scope: ZTF
artifacts. `surveys/ztf.py` sends `BAD_CATFLAGS_MASK=32768` on every
request, so IRSA strips flagged epochs server-side before they ever reach
this codebase -- recovering them would mean changing acquisition behaviour
for the whole ZTF pipeline, not a change scoped to this research module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# "clean" is index 0 so it lines up with diffusion.py's convention of
# reserving index 0 of an unused conditioning channel for "no label" --
# NOT the same as diffusion's own "unspecified" index (n_classes itself);
# clean is a real, meaningful class here, not an absence of one.
CATEGORY_NAMES: tuple[str, ...] = (
    "clean", "cosmic_ray", "stray_light", "pointing", "systematic", "excluded",
)

MIN_RUN_LENGTH = 3
DEFAULT_MAX_PATCHES_PER_CATEGORY = 50


def _quality_category_bits() -> dict[str, int]:
    """Real bit groupings from `lightkurve.utils.TessQualityFlags` --
    values are read from the library, not retyped by hand."""
    from lightkurve.utils import TessQualityFlags as Q

    return {
        "cosmic_ray": Q.ApertureCosmic | Q.CollateralCosmic,
        "stray_light": Q.Straylight | Q.Straylight2,
        "pointing": Q.CoarsePoint | Q.EarthPoint | Q.AttitudeTweak | Q.SafeMode,
        "systematic": Q.Desat | Q.Argabrightening | Q.Discontinuity | Q.ImpulsiveOutlier,
        "excluded": Q.ManualExclude | Q.BadCalibrationExclude
                   | Q.PlanetSearchExclude | Q.InsufficientTargets,
    }


def categorize_quality(quality_word: int) -> str | None:
    """One real TESS quality word -> a category name, or None when clean.

    Checked in a fixed, most-physically-specific-first order so a word with
    several bits set still gets one deterministic label. A nonzero word
    that matches none of the named groups (a real but ungrouped bit) still
    counts as an artifact -- falls through to "excluded" rather than being
    silently treated as clean.
    """
    word = int(quality_word)
    if word == 0:
        return None
    bits = _quality_category_bits()
    for name in ("cosmic_ray", "stray_light", "pointing", "systematic"):
        if word & bits[name]:
            return name
    return "excluded"


def download_reference_tpfs(targets: list[tuple[float, float]], root: Path | None = None,
                            size_pixels: int = 11, max_bytes: int = 32 * 1024 * 1024
                            ) -> list[Path]:
    """Download a small number of real TPFs for real (ra, dec) targets.

    Thin wrapper over the existing, real `tess_pixels.find_sectors`/
    `download_tpf` (MAST TESScut, confirmed live) -- no new acquisition
    machinery, just a loop over real targets. A target with no covering
    sector, or a download failure, is skipped rather than aborting the
    whole batch.
    """
    from . import tess_pixels

    paths: list[Path] = []
    for ra_deg, dec_deg in targets:
        try:
            sectors = tess_pixels.find_sectors(ra_deg, dec_deg)
            if not sectors:
                continue
            request = tess_pixels.TPFRequest(
                ra_deg=ra_deg, dec_deg=dec_deg, sector=sectors[0], size_pixels=size_pixels)
            result = tess_pixels.download_tpf(request, root=root, max_bytes=max_bytes)
            paths.append(Path(result["path"]))
        except Exception:  # noqa: BLE001 - one bad target must not abort the batch
            continue
    return paths


def _cadence_flux(frame: np.ndarray) -> float:
    """A coarse whole-frame, background-subtracted flux proxy for one
    cadence -- not resolved photometry, only a scalar series to build a
    time-ordered artifact patch from."""
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return float("nan")
    background = float(np.median(finite))
    return float(np.nansum(frame - background))


def extract_artifact_patches(tpf_paths: list[str | Path], patch_length: int = 32,
                             min_run_length: int = MIN_RUN_LENGTH,
                             max_patches_per_category: int = DEFAULT_MAX_PATCHES_PER_CATEGORY,
                             seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Real (value, mask) patches centred on real flagged-cadence runs, plus
    real clean contrast windows, from real downloaded TPFs.

    Returns `(patches, labels)`: `patches` is `(n, 2, patch_length)` in the
    same MAD-normalised value / validity-mask shape every sequence model in
    this codebase already consumes; `labels` is `(n,)` integer indices into
    `CATEGORY_NAMES`.
    """
    from . import tensors, tess_pixels

    rng = np.random.default_rng(seed)
    patches: list[np.ndarray] = []
    labels: list[int] = []
    counts = {name: 0 for name in CATEGORY_NAMES}

    for path in tpf_paths:
        try:
            data = tess_pixels.read_tpf_cube(path)
        except Exception:  # noqa: BLE001 - a corrupt TPF must not stop extraction
            continue

        cube = data["flux"]
        quality = np.asarray(data["quality"], dtype=np.uint64)
        n_cadences = cube.shape[0]
        if n_cadences < max(patch_length, tensors.MIN_POINTS):
            continue

        flux = np.array([_cadence_flux(cube[i]) for i in range(n_cadences)])
        valid = np.isfinite(flux)
        if int(valid.sum()) < tensors.MIN_POINTS:
            continue

        normalised = np.zeros(n_cadences, dtype=np.float32)
        normalised[valid] = tensors.normalise(flux[valid])
        mask = valid.astype(np.float32)
        categories = [categorize_quality(q) for q in quality]

        def _take(start: int, category_index: int) -> None:
            end = start + patch_length
            patches.append(np.stack([normalised[start:end], mask[start:end]]))
            labels.append(category_index)

        # Runs of the same flagged category.
        index = 0
        while index < n_cadences:
            category = categories[index]
            run_end = index
            while run_end < n_cadences and categories[run_end] == category:
                run_end += 1
            run_length = run_end - index
            if (category is not None and run_length >= min_run_length
                    and counts[category] < max_patches_per_category):
                center = (index + run_end) // 2
                start = max(0, min(n_cadences - patch_length, center - patch_length // 2))
                _take(start, CATEGORY_NAMES.index(category))
                counts[category] += 1
            index = run_end

        # Clean contrast windows: entirely-unflagged spans, sampled up to
        # roughly the number of flagged patches this TPF contributed.
        target_clean = sum(counts[name] for name in CATEGORY_NAMES if name != "clean")
        clean_starts = [
            start for start in range(0, n_cadences - patch_length + 1)
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
