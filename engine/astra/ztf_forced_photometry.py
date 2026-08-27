"""Forward PSF scene-model forced photometry for ZTF cutouts.

`products.py`'s cutout pipeline only ever produces a plain science-image
FITS product; nothing in this codebase today fits flux at a *fixed* source
position the way a forced-photometry pipeline needs to. This module adds
that: source positions are fixed at the target coordinate plus each
in-cutout reference-catalog (Gaia) neighbor's pixel position -- forced
photometry means measuring flux at KNOWN positions, not detecting new
sources -- and a shared circular Gaussian PRF
(`photutils.psf.CircularGaussianPRF`, a reasonable first-order shape for
ZTF's ~1-2 arcsec seeing against 1.01 arcsec/pixel plate scale, not a claim
of a publication-grade ZTF PSF model) is jointly fit per epoch via
`photutils.psf.PSFPhotometry` with positions fixed and only flux free.

Deliberately mirrors `tess_psf.py`'s shape almost exactly -- the underlying
problem (fixed-position joint PSF flux fit, validated by injection-recovery)
is the same one solved there for TESS's much coarser 21-arcsec pixels; only
the default PSF/fit-window scale differs here for ZTF's better-sampled PSF.

This produces a fitted flux *posterior* per source per epoch, alongside
fit-quality diagnostics (residual RMS) and, unlike tess_psf.py, an explicit
detection-recall diagnostic (this feature's stated success metric is
limiting-magnitude/transient recall, not only flux bias). It is new
evidence, not a replacement for any existing catalog-relative photometry,
and is never wired into `scoring.WEIGHTS`, matching every other evidence
module added alongside it (gw.py, frb.py, tess_psf.py).

Genuinely open, not attempted here: real ZTF difference-image acquisition
(`products.py`'s `CutoutRequest` only supports `product_kind="science"`
today, and this module does not guess the IBE difference/reference-image
path contract -- see `docs/DEFERRED.txt`) and per-epoch WCS-to-pixel
projection of reference-catalog positions. This module operates on
already-projected pixel positions and an already-assembled image/error
cube, exactly like `tess_psf.fit_cadence`/`build_scene_model` do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

# ZTF's plate scale is ~1.01 arcsec/pixel; typical seeing of ~2 arcsec puts
# the PSF FWHM at roughly 2 pixels -- far better sampled than TESS's 21
# arcsec pixels, so a slightly wider fit window than tess_psf.py's is used.
DEFAULT_FWHM_PIXELS = 2.0
DEFAULT_FIT_SHAPE = (9, 9)
DEFAULT_APERTURE_RADIUS = 3.5
TARGET_LABEL = "target"


class ZTFForcedPhotometryError(ValueError):
    """A forced-photometry PSF scene fit could not be produced from the given inputs."""


@dataclass(frozen=True)
class ScenePosition:
    """One fixed source position in the scene model, in cutout pixel coordinates."""

    label: str
    x_pixel: float
    y_pixel: float


def build_scene_positions(target_pixel: tuple[float, float],
                          neighbor_pixels: Iterable[tuple[str, float, float]], *,
                          shape: tuple[int, int]) -> list[ScenePosition]:
    """Fixed scene positions: the target plus every reference-catalog neighbor
    inside the cutout.

    A neighbor whose pixel position falls outside the cutout contributes no
    flux to it and would only destabilise the joint fit (an unconstrained
    source with no data to constrain it); it is silently excluded rather
    than passed through and left for the fitter to fail on.
    """
    ny, nx = shape
    positions = [ScenePosition(TARGET_LABEL, float(target_pixel[0]), float(target_pixel[1]))]
    for object_id, x_pixel, y_pixel in neighbor_pixels:
        x_pixel, y_pixel = float(x_pixel), float(y_pixel)
        if -0.5 <= x_pixel < nx - 0.5 and -0.5 <= y_pixel < ny - 0.5:
            positions.append(ScenePosition(str(object_id), x_pixel, y_pixel))
    return positions


def _psf_model(fwhm_pixels: float):
    from photutils.psf import CircularGaussianPRF

    model = CircularGaussianPRF(fwhm=float(fwhm_pixels))
    # Forced photometry means resolving flux at KNOWN positions -- letting
    # x_0/y_0 float would turn this into source detection/astrometry, a
    # different (and much less constrained) problem than the one this
    # module exists to solve.
    model.x_0.fixed = True
    model.y_0.fixed = True
    return model


def fit_epoch(image: np.ndarray, positions: list[ScenePosition], *,
             fwhm_pixels: float = DEFAULT_FWHM_PIXELS,
             fit_shape: tuple[int, int] = DEFAULT_FIT_SHAPE,
             aperture_radius: float = DEFAULT_APERTURE_RADIUS,
             error: np.ndarray | None = None) -> dict[str, Any]:
    """Joint PSF flux fit for one epoch's 2D image, positions held fixed."""
    from astropy.table import Table
    from photutils.psf import PSFPhotometry

    if not positions:
        raise ZTFForcedPhotometryError("at least one source position is required")
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ZTFForcedPhotometryError("fit_epoch expects one 2D image")

    model = _psf_model(fwhm_pixels)
    photometry = PSFPhotometry(model, fit_shape=fit_shape, aperture_radius=aperture_radius)
    init_params = Table()
    init_params["x_init"] = [position.x_pixel for position in positions]
    init_params["y_init"] = [position.y_pixel for position in positions]

    try:
        result = photometry(image, error=error, init_params=init_params)
        residual = photometry.make_residual_image(image)
    except Exception as exc:  # noqa: BLE001 - a fit failure is evidence, not a crash
        raise ZTFForcedPhotometryError(f"PSF scene fit failed: {exc}") from exc

    fluxes: dict[str, float] = {}
    flux_errors: dict[str, float | None] = {}
    for index, position in enumerate(positions):
        flux_value = float(result["flux_fit"][index])
        fluxes[position.label] = flux_value if math.isfinite(flux_value) else float("nan")
        raw_error = result["flux_err"][index]
        error_value = float(raw_error) if raw_error is not None else None
        flux_errors[position.label] = (error_value if error_value is not None
                                       and math.isfinite(error_value) else None)

    if all(not math.isfinite(value) for value in fluxes.values()):
        # photutils treats an all-NaN/unusable image as "nothing to fit" and
        # returns NaN fluxes rather than raising -- surfaced here as a
        # genuine fit failure (matching every other unusable-input path in
        # this module), not a silently accepted "zero flux" result.
        raise ZTFForcedPhotometryError("PSF scene fit produced no finite flux for any source")

    finite_residual = residual[np.isfinite(residual)]
    residual_rms = float(np.sqrt(np.mean(finite_residual ** 2))) if finite_residual.size else None
    return {
        "labels": [position.label for position in positions],
        "fluxes": fluxes, "flux_errors": flux_errors,
        "diagnostics": {"residual_rms": residual_rms,
                        "finite_residual_pixels": int(finite_residual.size)},
    }


def build_scene_model(cube: np.ndarray, positions: list[ScenePosition], *,
                      fwhm_pixels: float = DEFAULT_FWHM_PIXELS,
                      fit_shape: tuple[int, int] = DEFAULT_FIT_SHAPE,
                      aperture_radius: float = DEFAULT_APERTURE_RADIUS,
                      error_cube: np.ndarray | None = None) -> dict[str, Any]:
    """Per-epoch joint PSF flux fit across a whole forced-photometry image cube.

    One independent 2D fit per epoch -- the scene's source positions are
    fixed and known ahead of time, so epochs do not need to share
    information the way a variable-PSF model might; an epoch whose fit
    fails (a NaN-only frame, a degenerate cutout) is recorded as `None` in
    `per_epoch` rather than aborting the whole run, matching this project's
    general "a bad unit fails, not the whole job" discipline
    (`tess_psf.build_scene_model`/`gaia_epoch.ingest_resumable`'s per-chunk
    failure handling are the same shape).
    """
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim != 3:
        raise ZTFForcedPhotometryError("build_scene_model expects an (epoch, y, x) image cube")
    n_epochs = cube.shape[0]
    per_epoch: list[dict[str, Any] | None] = []
    for index in range(n_epochs):
        epoch_error = None if error_cube is None else np.asarray(error_cube[index])
        try:
            per_epoch.append(fit_epoch(
                cube[index], positions, fwhm_pixels=fwhm_pixels, fit_shape=fit_shape,
                aperture_radius=aperture_radius, error=epoch_error))
        except ZTFForcedPhotometryError:
            per_epoch.append(None)

    labels = [position.label for position in positions]
    flux_by_label: dict[str, list[float | None]] = {label: [] for label in labels}
    for epoch_result in per_epoch:
        for label in labels:
            flux_by_label[label].append(
                None if epoch_result is None else epoch_result["fluxes"].get(label))

    return {
        "labels": labels, "n_epochs": n_epochs,
        "per_epoch": per_epoch, "flux_by_label": flux_by_label,
        "fit_failures": sum(1 for entry in per_epoch if entry is None),
    }


# ---------------------------------------------------------------------------
# Falsifiable validation: injection-recovery, mirroring `tess_psf.py`'s shape
# (itself mirroring `ablation.py`/`stageb.py`'s "inject a known-truth
# scenario, run the real pipeline, measure recovery" discipline).
# ---------------------------------------------------------------------------

def flux_rmse(fitted: dict[str, list[float | None]],
             injected: dict[str, list[float]]) -> dict[str, Any]:
    """RMSE between fitted and known-injected per-epoch flux, per source.

    `fitted` is `build_scene_model(...)["flux_by_label"]`-shaped; `injected`
    holds the ground-truth flux used to construct the synthetic scene. An
    epoch with a failed fit (`None`) is excluded from that source's RMSE,
    not treated as a zero-flux error -- a missing measurement and a wrong
    measurement are different failure modes and must not be conflated in
    the reported metric.
    """
    per_source: dict[str, float] = {}
    for label, truth in injected.items():
        estimates = fitted.get(label, [])
        pairs = [(float(estimate), float(truth_value))
                for estimate, truth_value in zip(estimates, truth)
                if estimate is not None and math.isfinite(estimate)]
        per_source[label] = (float(np.sqrt(np.mean([(estimate - truth_value) ** 2
                                                     for estimate, truth_value in pairs])))
                             if pairs else float("nan"))
    finite_values = [value for value in per_source.values() if math.isfinite(value)]
    return {
        "per_source_rmse": per_source,
        "overall_rmse": float(np.mean(finite_values)) if finite_values else float("nan"),
    }


def blend_attribution_accuracy(fitted: dict[str, list[float | None]],
                               injected: dict[str, list[float]]) -> dict[str, Any]:
    """How well the fitted flux-fraction split matches the injected ground truth.

    Computed per epoch (the flux split can shift epoch to epoch if injected
    fluxes vary), then summarised. An epoch is scored only when every source
    has both a finite fitted value and a positive injected total -- an
    all-or-nothing epoch keeps the fraction well-defined and comparable
    across sources, rather than silently renormalising over a partial,
    fit-failure-biased subset.
    """
    labels = list(injected.keys())
    n_epochs = len(next(iter(injected.values()), []))
    per_epoch_error: list[float] = []
    for epoch_index in range(n_epochs):
        fitted_values = [fitted.get(label, [None] * n_epochs)[epoch_index] for label in labels]
        if any(value is None or not math.isfinite(value) for value in fitted_values):
            continue
        injected_values = [float(injected[label][epoch_index]) for label in labels]
        injected_total = sum(injected_values)
        fitted_total = sum(fitted_values)
        if injected_total <= 0 or fitted_total <= 0:
            continue
        injected_fractions = [value / injected_total for value in injected_values]
        fitted_fractions = [value / fitted_total for value in fitted_values]
        error = float(np.mean([abs(a - b) for a, b in zip(fitted_fractions, injected_fractions)]))
        per_epoch_error.append(error)

    return {
        "labels": labels, "epochs_scored": len(per_epoch_error),
        "epochs_total": n_epochs,
        "mean_absolute_fraction_error": (float(np.mean(per_epoch_error))
                                         if per_epoch_error else float("nan")),
    }


def _synthetic_gaussian_image(shape: tuple[int, int], positions: list[ScenePosition],
                              fluxes: dict[str, float], fwhm_pixels: float,
                              *, background: float = 0.0,
                              noise_sigma: float = 0.0,
                              rng: np.random.Generator | None = None) -> np.ndarray:
    """A synthetic scene image built from the exact model class `fit_epoch` fits.

    Deliberately instantiates the SAME `photutils.psf.CircularGaussianPRF`
    class `_psf_model` uses (rather than a hand-written Gaussian formula) so
    the injected image and the fit model agree exactly on pixel-integration
    behaviour, not just on nominal FWHM/flux. This validates the fitting
    MACHINERY (does it correctly separate known, overlapping sources at a
    known flux ratio/separation and recover a faint source near the
    detection floor), not whether a circular Gaussian is itself a good ZTF
    PSF model, which is a separate, real-world question this synthetic test
    cannot answer.
    """
    from photutils.psf import CircularGaussianPRF

    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    image = np.full((ny, nx), float(background), dtype=np.float64)
    for position in positions:
        flux = float(fluxes.get(position.label, 0.0))
        if flux == 0.0:
            continue
        model = CircularGaussianPRF(fwhm=float(fwhm_pixels), flux=flux,
                                    x_0=position.x_pixel, y_0=position.y_pixel)
        image += model(xx, yy)
    if noise_sigma > 0:
        rng = rng or np.random.default_rng()
        image = image + rng.normal(0.0, noise_sigma, size=image.shape)
    return image


def injected_source_recovery(*, n_trials: int, separations_pixels: Iterable[float],
                             flux_ratios: Iterable[float], shape: tuple[int, int] = (21, 21),
                             fwhm_pixels: float = DEFAULT_FWHM_PIXELS,
                             target_flux: float = 5000.0, background: float = 10.0,
                             noise_sigma: float = 5.0, seed: int = 42) -> dict[str, Any]:
    """Monte-Carlo recovery rate/bias as a function of separation and flux ratio.

    For each (separation, flux_ratio) grid point, `n_trials` synthetic
    two-source scenes are built at that separation and ratio (positions and
    sub-pixel jitter randomised per trial, target position jittered by up to
    0.3 pixel to avoid the fit trivially recovering an exact-center
    injection every time), and `fit_epoch` is run to recover both fluxes.
    Reports, per grid point, the mean fractional flux bias/RMSE for the
    target -- the same shape as `tess_psf.injected_source_recovery`.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    rng = np.random.default_rng(seed)
    ny, nx = shape
    center = (nx / 2.0, ny / 2.0)
    grid: list[dict[str, Any]] = []
    for separation in separations_pixels:
        for flux_ratio in flux_ratios:
            neighbor_flux = target_flux * float(flux_ratio)
            biases: list[float] = []
            errors: list[float] = []
            for _ in range(n_trials):
                jitter_x, jitter_y = rng.uniform(-0.3, 0.3, size=2)
                angle = rng.uniform(0.0, 2.0 * math.pi)
                target_position = ScenePosition(TARGET_LABEL, center[0] + jitter_x,
                                                center[1] + jitter_y)
                neighbor_position = ScenePosition(
                    "neighbor",
                    target_position.x_pixel + separation * math.cos(angle),
                    target_position.y_pixel + separation * math.sin(angle),
                )
                positions = [target_position, neighbor_position]
                truth = {TARGET_LABEL: target_flux, "neighbor": neighbor_flux}
                image = _synthetic_gaussian_image(
                    shape, positions, truth, fwhm_pixels, background=background,
                    noise_sigma=noise_sigma, rng=rng)
                try:
                    result = fit_epoch(image, positions, fwhm_pixels=fwhm_pixels)
                except ZTFForcedPhotometryError:
                    continue
                recovered = result["fluxes"][TARGET_LABEL]
                if not math.isfinite(recovered):
                    continue
                biases.append((recovered - target_flux) / target_flux)
                errors.append((recovered - target_flux) ** 2)
            grid.append({
                "separation_pixels": float(separation), "flux_ratio": float(flux_ratio),
                "trials_completed": len(biases),
                "mean_fractional_bias": float(np.mean(biases)) if biases else float("nan"),
                "rmse": float(np.sqrt(np.mean(errors))) if errors else float("nan"),
            })
    return {"n_trials": int(n_trials), "grid": grid}


def limiting_magnitude_recovery(*, n_trials: int, target_fluxes: Iterable[float],
                                shape: tuple[int, int] = (21, 21),
                                fwhm_pixels: float = DEFAULT_FWHM_PIXELS,
                                background: float = 10.0, noise_sigma: float = 5.0,
                                detection_snr: float = 5.0, seed: int = 42) -> dict[str, Any]:
    """Detection-recall Monte Carlo at a range of injected source fluxes.

    This is the item's own stated success metric -- "limiting magnitude and
    transient recall" -- as distinct from `injected_source_recovery`'s flux
    *bias* at fixed separation/ratio. For each injected flux, `n_trials`
    single, isolated synthetic sources (no blending neighbor) are placed at
    a jittered position and fit; a trial counts as "recovered" when the fit
    both succeeds and returns a flux whose S/N (flux / flux_error) clears
    `detection_snr` -- a fit that succeeds numerically but reports a flux
    consistent with noise is not a detection.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    rng = np.random.default_rng(seed)
    ny, nx = shape
    center = (nx / 2.0, ny / 2.0)
    grid: list[dict[str, Any]] = []
    for target_flux in target_fluxes:
        recovered = 0
        biases: list[float] = []
        for _ in range(n_trials):
            jitter_x, jitter_y = rng.uniform(-0.3, 0.3, size=2)
            position = ScenePosition(TARGET_LABEL, center[0] + jitter_x, center[1] + jitter_y)
            image = _synthetic_gaussian_image(
                shape, [position], {TARGET_LABEL: float(target_flux)}, fwhm_pixels,
                background=background, noise_sigma=noise_sigma, rng=rng)
            try:
                result = fit_epoch(image, [position], fwhm_pixels=fwhm_pixels)
            except ZTFForcedPhotometryError:
                continue
            flux = result["fluxes"][TARGET_LABEL]
            error = result["flux_errors"][TARGET_LABEL]
            if not math.isfinite(flux):
                continue
            biases.append((flux - target_flux) / target_flux if target_flux else float("nan"))
            if error is not None and error > 0 and (flux / error) >= detection_snr:
                recovered += 1
        grid.append({
            "target_flux": float(target_flux),
            "trials_completed": n_trials,
            "detection_recall": recovered / n_trials,
            "mean_fractional_bias": float(np.mean(biases)) if biases else float("nan"),
        })
    return {"n_trials": int(n_trials), "detection_snr": float(detection_snr), "grid": grid}


__all__ = [
    "ZTFForcedPhotometryError", "ScenePosition", "build_scene_positions",
    "fit_epoch", "build_scene_model", "flux_rmse", "blend_attribution_accuracy",
    "injected_source_recovery", "limiting_magnitude_recovery",
    "DEFAULT_FWHM_PIXELS", "DEFAULT_FIT_SHAPE", "DEFAULT_APERTURE_RADIUS",
]
