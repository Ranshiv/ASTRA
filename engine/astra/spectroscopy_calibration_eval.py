"""Validation for `spectroscopy_calibration.py`: redshift residuals against
released catalog values, and line-flux recovery on synthetic spectra
(roadmap item 24's named metric).

Redshift-residual scoring reuses `microlensing_eval.parameter_bias`
UNCHANGED rather than reimplementing median-fractional-bias/MAD-scatter
arithmetic a third time in this codebase -- its `{name: value}` row shape
generalises to a redshift exactly as it does to a microlensing parameter.
"""

from __future__ import annotations

import numpy as np

from . import spectroscopy_calibration as calibration
from .microlensing_eval import parameter_bias

REDSHIFT_NAMES: tuple[str, ...] = ("z",)


def redshift_residuals(spectra: list[dict]) -> dict:
    """Independent-redshift-vs-released-redshift residuals over many spectra.

    Each entry in `spectra` is `{"wavelength", "flux", "error", "released_z"}`.
    A spectrum whose independent fit finds no usable line (`z_best is None`)
    is counted and reported but excluded from the bias statistics -- the
    same None-vs-excluded discipline `evaluate_selection`/`parameter_bias`
    already use elsewhere in this codebase, not silently scored as zero.
    """
    fitted: list[dict] = []
    reference: list[dict] = []
    n_unresolved = 0
    per_spectrum: list[dict] = []

    for entry in spectra:
        report = calibration.independent_redshift_from_lines(
            entry["wavelength"], entry["flux"], entry["error"])
        released_z = float(entry["released_z"])
        z_best = report["z_best"]
        per_spectrum.append({"z_best": z_best, "released_z": released_z,
                             "n_lines_matched": report["n_lines_matched"]})
        if z_best is None:
            n_unresolved += 1
            continue
        fitted.append({"z": z_best})
        reference.append({"z": released_z})

    bias = parameter_bias(fitted, reference, names=REDSHIFT_NAMES) if fitted else {
        "n_events": 0, "parameters": {"z": {"n_compared": 0}}}

    return {"n_spectra": len(spectra), "n_unresolved": n_unresolved,
           "bias": bias, "per_spectrum": per_spectrum}


def line_flux_recovery(trials: list[dict]) -> dict:
    """Line-parameter residuals on synthetic spectra with a KNOWN injected
    line (no real released line-flux catalog is available to this codebase
    yet -- the same "synthetic injection-recovery when no real reference
    exists" fallback `microlensing_fit`/`reverberation.py` both already use).

    Each trial is `{"wavelength", "flux", "error", "true_wavelength_rest",
    "true_z"}`: `find_candidate_lines` must recover a detected line within
    `DEFAULT_VELOCITY_TOLERANCE_KMS` of the true observed-frame wavelength.
    Reports recall (fraction recovered) and, for recovered lines, the
    median/robust wavelength residual in km/s -- the closest velocity-space
    analogue of a "line-flux residual" this bounded peak-finder (not a
    flux-fitting model; that is item 25's job) can honestly report.
    """
    recovered = 0
    offsets_kms: list[float] = []
    for trial in trials:
        true_observed = float(trial["true_wavelength_rest"]) * (1.0 + float(trial["true_z"]))
        candidates = calibration.find_candidate_lines(
            trial["wavelength"], trial["flux"], trial["error"])
        if not candidates:
            continue
        waves = np.array([c["wavelength"] for c in candidates])
        offsets = np.abs(waves - true_observed) / true_observed * calibration.SPEED_OF_LIGHT_KMS
        nearest = int(np.argmin(offsets))
        if offsets[nearest] <= calibration.DEFAULT_VELOCITY_TOLERANCE_KMS:
            recovered += 1
            offsets_kms.append(float(offsets[nearest]))

    result = {"n_trials": len(trials), "n_recovered": recovered,
             "recall": (recovered / len(trials)) if trials else float("nan")}
    if offsets_kms:
        values = np.asarray(offsets_kms)
        result["median_offset_kms"] = float(np.median(values))
        result["robust_scatter_kms"] = float(np.median(np.abs(values - np.median(values))) * 1.4826)
    return result
