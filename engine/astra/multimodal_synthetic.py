"""Synthetic paired multimodal data for validating `multimodal_moco.py`
(backlog item 11).

No connector in this codebase assembles a real object with an image cutout
AND a spectrum AND a light curve AND a Gaia match all downloaded at once
(checked `surveys/alerce.py` specifically -- it brokers detections only, no
images/spectra/catalog bundling). Training and evaluation therefore use
synthetic paired data here, the same "mechanism validated on synthetic
ground truth, not yet run at real Stage-B scale" discipline
`multiband_hier.py`/`moving_objects.py` already carry.

Every object's four modality-scale scalars are derived from ONE shared
magnitude via the real photometric relation
`flux = 10**(-0.4*(mag-zeropoint))`, so they are genuinely correlated the
way a real object's would be -- randomising them independently would make
`multimodal_eval.probe_brightness_preservation` measure nothing real.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .multimodal_encoders import CATALOG_FEATURE_COUNT

CLASS_KINDS: tuple[str, ...] = (
    "quiet_dwarf", "flare_star", "eclipsing_binary", "hot_subdwarf", "agn_like",
)
# Each class's light-curve injection kind, reusing evaluate.inject()'s real
# anomaly shapes rather than inventing new ones. "quiet_dwarf" gets none.
_INJECTION_KIND = {
    "flare_star": "flare", "eclipsing_binary": "eclipse",
    "hot_subdwarf": "step", "agn_like": "noise_burst",
}
ZEROPOINT = 25.0


@dataclass
class SyntheticMultimodalBatch:
    object_ids: list[str]
    class_labels: np.ndarray
    lightcurve_values: np.ndarray   # (n, 2, lc_length)
    lightcurve_scale: np.ndarray    # real flux-like scalar
    image_arrays: np.ndarray        # (n, 1, image_size, image_size)
    image_scale: np.ndarray
    spectrum_arrays: np.ndarray     # (n, 3, spectrum_length)
    spectrum_scale: np.ndarray
    catalog_features: np.ndarray    # (n, CATALOG_FEATURE_COUNT)
    catalog_scale: np.ndarray       # real magnitude scalar (gaia_phot_g_mean_mag-like)

    def __len__(self) -> int:
        return len(self.object_ids)


def _base_lightcurve(length: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth normalised curve with a full validity mask -- same
    convention `tests/test_pretrain.py`'s `fake_sequences` already uses."""
    time = np.linspace(0, 4 * np.pi, length)
    value = (np.sin(time * rng.uniform(0.5, 2.0) + rng.uniform(0, np.pi))
            + rng.normal(0, 0.05, length)).astype(np.float32)
    mask = np.ones(length, dtype=np.float32)
    return np.stack([value, mask], axis=0)


def _synthetic_spectrum(length: int, class_kind: str,
                        rng: np.random.Generator) -> np.ndarray:
    """(3, length): log-wavelength grid, flux, error -- a UNIT-SCALE flat
    continuum (~1.0) plus a class-dependent line feature. Absolute
    brightness deliberately does NOT enter this array: like `tensors.py`'s
    MAD-normalised light-curve channel, the encoder input carries shape
    only, and the real flux-like scalar lives entirely in
    `spectrum_scale`/the scale token instead. Feeding the raw,
    orders-of-magnitude-varying flux directly into an untrained transformer
    was tried first and produced NaN activations under float16 autocast on
    this machine's GPU -- confirming this codebase's own light-curve
    precedent (`tensors.normalise()` discarding absolute brightness by
    design) is the right convention here too, not an optional nicety.
    """
    log_wave = np.linspace(3.55, 3.75, length).astype(np.float32)
    flux = np.ones(length, dtype=np.float32)
    flux += rng.normal(0, 0.02, length).astype(np.float32)

    center = length // 2
    width = max(3, length // 20)
    profile = np.exp(-0.5 * ((np.arange(length) - center) / width) ** 2)
    if class_kind == "agn_like":
        flux += (0.6 * profile).astype(np.float32)   # emission
    elif class_kind in ("flare_star", "eclipsing_binary"):
        flux -= (0.3 * profile).astype(np.float32)   # absorption

    error = np.full(length, 0.01, dtype=np.float32)
    return np.stack([log_wave, flux, error], axis=0)


def _synthetic_image(size: int, rng: np.random.Generator) -> np.ndarray:
    """A single Gaussian source of UNIT-SCALE amplitude on a noisy
    background -- see `_synthetic_spectrum`'s docstring for why absolute
    brightness is deliberately excluded here too (the real scalar lives in
    `image_scale`); the raw flux-scaled version produced NaN activations
    under float16 autocast."""
    yy, xx = np.indices((size, size), dtype=np.float64)
    center = (size - 1) / 2.0
    sigma = size / 8.0
    source = 5.0 * np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))
    background = rng.normal(0, 0.1, size=(size, size))
    return (source + background).astype(np.float32)[None, :, :]  # (1, size, size)


def _synthetic_catalog_row(class_loading: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A class-correlated feature row -- same "structured signal plus
    noise" convention `tests/test_stellar_manifold_eval.py`'s
    `_gaia_manifold_matrix` fixture already uses, so different classes
    occupy different regions of catalog-feature space (needed for the
    linear-probe macro-F1 metric to be answerable at all).

    `class_loading` is FIXED per class (drawn once in `build_synthetic_pairs`,
    not re-drawn per row): every object of the same class shares the same
    signal DIRECTION in feature space, differing only by per-row noise --
    otherwise classes would only differ in signal MAGNITUDE, not position,
    and would not actually separate in feature space at all.
    """
    return class_loading + rng.normal(0, 0.3, class_loading.shape[0])


def build_synthetic_pairs(n: int = 200, seed: int = 42,
                          lc_length: int = 256, image_size: int = 32,
                          spectrum_length: int = 256) -> SyntheticMultimodalBatch:
    """`n` fully paired synthetic objects, one shared magnitude per object
    driving all four modalities' scale scalars through the real
    `flux = 10**(-0.4*(mag-zeropoint))` relation.
    """
    from . import evaluate

    rng = np.random.default_rng(seed)
    n_classes = len(CLASS_KINDS)
    class_indices = rng.integers(0, n_classes, size=n)
    # One fixed feature-space DIRECTION per class, shared by every object of
    # that class (see _synthetic_catalog_row's docstring for why this must
    # be per-class, not per-row).
    class_loadings = np.random.default_rng(seed + 500).normal(
        0.0, 1.0, size=(n_classes, CATALOG_FEATURE_COUNT))

    object_ids = [f"synthetic-{i}" for i in range(n)]
    magnitudes = rng.uniform(10.0, 20.0, size=n)
    flux = 10.0 ** (-0.4 * (magnitudes - ZEROPOINT))

    lightcurve_values = np.empty((n, 2, lc_length), dtype=np.float32)
    image_arrays = np.empty((n, 1, image_size, image_size), dtype=np.float32)
    spectrum_arrays = np.empty((n, 3, spectrum_length), dtype=np.float32)
    catalog_features = np.empty((n, CATALOG_FEATURE_COUNT), dtype=np.float64)

    for i in range(n):
        class_kind = CLASS_KINDS[class_indices[i]]
        row_rng = np.random.default_rng(seed + 1000 + i)

        curve = _base_lightcurve(lc_length, row_rng)
        injection_kind = _INJECTION_KIND.get(class_kind)
        if injection_kind is not None:
            curve = evaluate.inject(curve, injection_kind, row_rng, strength=6.0)
        lightcurve_values[i] = curve

        image_arrays[i] = _synthetic_image(image_size, row_rng)
        spectrum_arrays[i] = _synthetic_spectrum(spectrum_length, class_kind, row_rng)
        catalog_features[i] = _synthetic_catalog_row(
            class_loadings[class_indices[i]], row_rng)

    return SyntheticMultimodalBatch(
        object_ids=object_ids,
        class_labels=class_indices.astype(int),
        lightcurve_values=lightcurve_values,
        lightcurve_scale=flux.astype(np.float32),
        image_arrays=image_arrays,
        image_scale=(flux * 0.3).astype(np.float32),
        spectrum_arrays=spectrum_arrays,
        spectrum_scale=(flux * 0.1).astype(np.float32),
        catalog_features=catalog_features,
        catalog_scale=magnitudes.astype(np.float32),
    )
