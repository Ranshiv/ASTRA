"""X-ray hardness-ratio computation and discrete-state modeling
(roadmap item 23).

`hardness_ratio` is the standard `(H-S)/(H+S)` definition -- the same
convention Swift's own 2SXPS catalogue publishes for its `HR1`/`HR2`
columns (Evans et al. 2020), confirmed this session by reading that
paper's own stated formula rather than assuming a convention.

`fit_hardness_states` classifies an epoch-by-epoch hardness-ratio time
series into a small number of discrete spectral states using
`sklearn.mixture.GaussianMixture` -- `scikit-learn` is already a core
dependency (see `engine/pyproject.toml`), so this needs no new package.
A genuine state carries a fitted mean/std hardness ratio from the data
itself, not an arbitrary threshold cut; states are relabelled in ascending
hardness order so "state 0" is always the softest state, not an arbitrary
`sklearn` cluster index (cluster labels from a fresh fit are otherwise
unordered and can flip between runs on the same data).
"""

from __future__ import annotations

import numpy as np


class XrayHardnessError(ValueError):
    """An X-ray hardness computation or state model could not be evaluated."""


def hardness_ratio(soft_flux, hard_flux) -> np.ndarray:
    """`(H-S)/(H+S)`, dimensionless in `[-1, 1]` -- softer sources have
    negative values, harder sources positive. `None` (via NaN propagation)
    when soft+hard is exactly zero, rather than a fabricated ratio.
    """
    soft = np.asarray(soft_flux, dtype=np.float64)
    hard = np.asarray(hard_flux, dtype=np.float64)
    if soft.shape != hard.shape:
        raise XrayHardnessError("soft_flux and hard_flux must have the same shape")
    if np.any(soft < 0) or np.any(hard < 0):
        raise XrayHardnessError("soft_flux and hard_flux must be non-negative")
    total = soft + hard
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(total > 0, (hard - soft) / np.where(total > 0, total, 1.0), np.nan)
    return ratio


def fit_hardness_states(hardness_ratios, n_states: int = 2, seed: int = 42) -> dict:
    """K-state Gaussian-mixture classification of a hardness-ratio time
    series.

    Needs at least `2 * n_states` finite points -- a mixture with more
    components than the data can meaningfully constrain is not a fit, it
    is overfitting dressed up as one. NaN/missing hardness ratios (see
    `hardness_ratio`'s `total == 0` case) are dropped before fitting, not
    imputed.
    """
    from sklearn.mixture import GaussianMixture

    values = np.asarray(hardness_ratios, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if n_states < 1:
        raise XrayHardnessError("n_states must be at least 1")
    if len(finite) < 2 * n_states:
        raise XrayHardnessError(f"need at least {2 * n_states} finite points to fit {n_states} states")

    model = GaussianMixture(n_components=n_states, random_state=seed, n_init=5)
    raw_labels = model.fit_predict(finite.reshape(-1, 1))

    order = np.argsort(model.means_.flatten())
    remap = {int(old): int(new) for new, old in enumerate(order)}
    ordered_labels = np.array([remap[int(label)] for label in raw_labels])

    # Re-expand to the ORIGINAL length/positions, with a non-finite input
    # epoch labelled -1 (no state, not a fabricated state 0).
    full_labels = np.full(len(values), -1, dtype=np.int64)
    full_labels[np.isfinite(values)] = ordered_labels

    return {
        "n_states": n_states,
        "labels": full_labels.tolist(),
        "state_means": model.means_.flatten()[order].tolist(),
        "state_stds": np.sqrt(model.covariances_.flatten())[order].tolist(),
        "converged": bool(model.converged_),
        "n_points_fit": int(len(finite)),
    }


def detect_state_transitions(labels) -> list[int]:
    """Indices where the assigned state differs from the previous FINITE
    (`label != -1`) epoch -- a transition into or out of a missing-data gap
    is not itself counted as a spectral-state transition.
    """
    labels = np.asarray(labels, dtype=np.int64)
    transitions: list[int] = []
    previous_label = None
    for index, label in enumerate(labels):
        if label == -1:
            continue
        if previous_label is not None and label != previous_label:
            transitions.append(index)
        previous_label = label
    return transitions
