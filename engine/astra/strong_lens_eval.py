"""Strong-lens recovery study: lens AUPRC, image-position residual, and
time-delay error, all against synthetic ground truth (roadmap item 29).

Split from `strong_lens.py` purely to keep each file under this project's
500-line guideline, same `stellar_manifold.py`/`stellar_manifold_eval.py`
rationale, not an independent module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .strong_lens import SIELens, fit_lens_model, solve_image_positions, time_delay_seconds


def _random_sie(rng: np.random.Generator) -> SIELens:
    theta_e = float(rng.uniform(0.5, 1.5))
    axis_ratio = float(rng.uniform(0.5, 0.95))
    position_angle = float(rng.uniform(0.0, np.pi))
    return SIELens(theta_e=theta_e, axis_ratio=axis_ratio, position_angle=position_angle)


def evaluate_lens_auprc(*, n_lensed: int = 100, n_unlensed: int = 100, seed: int = 31
                        ) -> dict[str, Any]:
    """Synthetic lensed systems (source well inside the lens's caustic,
    reliably producing >=2 images) vs. synthetic unlensed/marginal systems
    (source placed far outside the caustic, where a real SIE produces
    exactly one image -- the realistic negative case for a lens-candidate
    vetting pipeline, not an arbitrary null). The score is the solved
    image count; `average_precision_score` against the known synthetic
    labels IS the "lens AUPRC" metric this roadmap item names.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    labels: list[int] = []
    scores: list[float] = []
    for _ in range(n_lensed):
        lens = _random_sie(rng)
        beta = rng.uniform(-0.15, 0.15, size=2) * lens.theta_e
        images = solve_image_positions(lens, beta)
        labels.append(1)
        scores.append(float(len(images)))
    for _ in range(n_unlensed):
        lens = _random_sie(rng)
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        beta = direction * lens.theta_e * rng.uniform(2.5, 4.0)
        images = solve_image_positions(lens, beta)
        labels.append(0)
        scores.append(float(len(images)))
    auprc = float(average_precision_score(labels, scores))
    return {
        "n_lensed": n_lensed, "n_unlensed": n_unlensed,
        "lens_auprc": round(auprc, 4),
        "mean_images_lensed": round(float(np.mean(scores[:n_lensed])), 3),
        "mean_images_unlensed": round(float(np.mean(scores[n_lensed:])), 3),
    }


def evaluate_image_position_residual(*, n_trials: int = 30, astrometric_noise: float = 0.01,
                                     seed: int = 37) -> dict[str, Any]:
    """Inject a known SIE+beta, add Gaussian astrometric noise to the true
    image positions, fit `fit_lens_model` back from the noisy positions,
    then compare the FITTED model's own re-solved image positions against
    the TRUE (noiseless) injected ones -- the "image-position residual"
    metric this roadmap item names."""
    rng = np.random.default_rng(seed)
    residuals_arcsec: list[float] = []
    n_used = 0
    for _ in range(n_trials):
        lens = _random_sie(rng)
        beta = rng.uniform(-0.15, 0.15, size=2) * lens.theta_e
        true_images = solve_image_positions(lens, beta)
        # A two-image ("double") system gives only 4 residuals for the
        # fit's 5 unknowns -- mathematically underdetermined (see
        # `fit_lens_model`'s own comment), so only well-constrained
        # 3+-image ("quad"-like) systems are used for this recovery
        # study, the same restriction real strong-lens modelling applies
        # to doubles without an external constraint (e.g. flux ratios).
        if len(true_images) < 3:
            continue
        noisy_images = [image + rng.normal(0.0, astrometric_noise, size=2) for image in true_images]
        fit = fit_lens_model(noisy_images, initial_theta_e=lens.theta_e)
        if not fit["converged"]:
            continue
        fitted_lens = SIELens(theta_e=fit["theta_e"], axis_ratio=fit["axis_ratio"],
                              position_angle=fit["position_angle_rad"])
        fitted_images = solve_image_positions(fitted_lens, np.array(fit["beta"]))
        if len(fitted_images) != len(true_images):
            continue
        for true_image in true_images:
            nearest = min(fitted_images, key=lambda candidate: np.linalg.norm(candidate - true_image))
            residuals_arcsec.append(float(np.linalg.norm(nearest - true_image)))
        n_used += 1
    if not residuals_arcsec:
        return {"n_trials": n_trials, "n_used": 0, "rms_residual": None, "warnings": ["no trial converged"]}
    residuals = np.asarray(residuals_arcsec)
    return {
        "n_trials": n_trials, "n_used": n_used,
        "rms_residual": round(float(np.sqrt(np.mean(residuals ** 2))), 6),
        "max_residual": round(float(np.max(residuals)), 6),
    }


@dataclass(frozen=True)
class Cosmology:
    """Minimal angular-diameter-distance inputs, so this module does not
    require `astropy.cosmology` for a synthetic-only recovery check --
    the caller supplies real cosmological distances in production use."""

    z_lens: float
    d_l_mpc: float
    d_s_mpc: float
    d_ls_mpc: float


def evaluate_time_delay_error(*, n_trials: int = 30, astrometric_noise: float = 0.01,
                              cosmology: Cosmology | None = None, seed: int = 41
                              ) -> dict[str, Any]:
    """Inject a known SIE+beta+cosmology, compute the TRUE time delay
    between the two brightest-separated images, add astrometric noise,
    refit, and compare the FITTED model's predicted time delay (using the
    same, assumed-known cosmology/redshift) against the true injected
    value -- the "time-delay error" metric this roadmap item names."""
    cosmology = cosmology or Cosmology(z_lens=0.5, d_l_mpc=1200.0, d_s_mpc=1700.0, d_ls_mpc=900.0)
    rng = np.random.default_rng(seed)
    errors_days: list[float] = []
    n_used = 0
    for _ in range(n_trials):
        lens = _random_sie(rng)
        beta = rng.uniform(-0.15, 0.15, size=2) * lens.theta_e
        true_images = solve_image_positions(lens, beta)
        if len(true_images) < 3:
            continue
        pair = sorted(true_images, key=lambda p: -np.linalg.norm(p))[:2]
        true_delay = time_delay_seconds(lens, pair[0], pair[1], beta, z_lens=cosmology.z_lens,
                                        d_l_mpc=cosmology.d_l_mpc, d_s_mpc=cosmology.d_s_mpc,
                                        d_ls_mpc=cosmology.d_ls_mpc)

        noisy_images = [image + rng.normal(0.0, astrometric_noise, size=2) for image in true_images]
        fit = fit_lens_model(noisy_images, initial_theta_e=lens.theta_e)
        if not fit["converged"]:
            continue
        fitted_lens = SIELens(theta_e=fit["theta_e"], axis_ratio=fit["axis_ratio"],
                              position_angle=fit["position_angle_rad"])
        fitted_beta = np.array(fit["beta"])
        fitted_images = solve_image_positions(fitted_lens, fitted_beta)
        if len(fitted_images) < 2:
            continue
        fitted_pair = sorted(fitted_images, key=lambda p: -np.linalg.norm(p))[:2]
        fitted_delay = time_delay_seconds(fitted_lens, fitted_pair[0], fitted_pair[1], fitted_beta,
                                          z_lens=cosmology.z_lens, d_l_mpc=cosmology.d_l_mpc,
                                          d_s_mpc=cosmology.d_s_mpc, d_ls_mpc=cosmology.d_ls_mpc)
        errors_days.append(abs(fitted_delay - true_delay) / 86400.0)
        n_used += 1
    if not errors_days:
        return {"n_trials": n_trials, "n_used": 0, "mean_absolute_error_days": None,
               "warnings": ["no trial converged"]}
    errors = np.asarray(errors_days)
    return {
        "n_trials": n_trials, "n_used": n_used,
        "mean_absolute_error_days": round(float(np.mean(errors)), 4),
        "median_absolute_error_days": round(float(np.median(errors)), 4),
    }
