"""Top-k host recall and probability calibration for `host_association.py`
(the two metrics roadmap item 31 names), split from that module purely to
keep each file under this project's 500-line guideline (same
`stellar_manifold.py`/`stellar_manifold_eval.py` split rationale, not an
independent module).

Validated on SYNTHETIC ground truth only -- the same "mechanism validated
on synthetic data, not yet run at real Stage-B scale" caveat every eval
module in this family (`sn_classification_eval.py`, `agn_changepoint_
eval.py`) already states. A real top-k recall/calibration study needs a
real catalog of transients with spectroscopically confirmed hosts (e.g.
cross-matching ALeRCE-classified SNe against their published host
associations), which is not attempted here.

`synthesize_host_population` draws a true host's offset directly from
`numpy.random.Generator.gamma(shape=2.0, scale=r_e)` rather than inverting
the offset CDF by hand: `host_association.exponential_offset_likelihood`'s
`f(r) = (r / r_e**2) * exp(-r / r_e)` IS the Gamma(shape=2, scale=r_e)
density (confirmed by direct comparison to `scipy.stats.gamma`'s pdf
formula), so this reuses a standard library sampler exactly rather than
implementing a redundant inverse-CDF routine. Absolute magnitudes are
drawn from the Schechter prior via bounded rejection sampling over a fixed
finite magnitude range (`_sample_schechter_abs_mag`) -- a real, stated
simplification: the unnormalized Schechter density is not proper (its
faint-end tail is not integrable to a fixed constant without `phi*`), so
sampling is only well-posed over a bounded range, not the true infinite
support. This is adequate for generating a plausible synthetic population,
not a rigorous draw from the true Schechter distribution.

Every synthetic field includes both a true host (real, drawn offset and
magnitude, real matching redshift) and contaminant candidates (larger,
more scattered offsets; unrelated redshifts; the first contaminant tagged
as a Gaia foreground-star false positive by default) so recall/calibration
are measured under real, non-trivial confusion, not a single-candidate toy
case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .host_association import (
    DEFAULT_ALPHA, DEFAULT_M_STAR, HostAssociationError, SourceRef,
    associate_host, default_cosmology, schechter_luminosity_prior,
)


def _offset_position(ra_deg: float, dec_deg: float, offset_arcsec: float, angle_rad: float) -> tuple[float, float]:
    """Small-angle planar offset -- adequate near dec=0 for synthetic
    fields spanning at most a few tens of arcsec, not a general spherical
    projection."""
    offset_deg = offset_arcsec / 3600.0
    d_ra = offset_deg * math.cos(angle_rad) / math.cos(math.radians(dec_deg))
    d_dec = offset_deg * math.sin(angle_rad)
    return ra_deg + d_ra, dec_deg + d_dec


def _sample_schechter_abs_mag(rng: np.random.Generator, m_star: float, alpha: float,
                              mag_range: tuple[float, float] = (-24.0, -16.0),
                              max_tries: int = 10_000) -> float:
    lo, hi = mag_range
    grid = np.linspace(lo, hi, 400)
    envelope = max(schechter_luminosity_prior(float(m), m_star, alpha) for m in grid) * 1.05
    for _ in range(max_tries):
        candidate = float(rng.uniform(lo, hi))
        density = schechter_luminosity_prior(candidate, m_star, alpha)
        if rng.uniform(0.0, envelope) <= density:
            return candidate
    raise HostAssociationError("rejection sampling failed to draw a Schechter-distributed magnitude")


@dataclass(frozen=True)
class SyntheticHostField:
    transient: SourceRef
    candidates: list[SourceRef]
    redshifts: dict[str, tuple[float, str]]
    r_e_arcsec: dict[str, float]
    foreground_flags: dict[str, bool]
    true_host_id: str


def synthesize_host_population(rng: np.random.Generator, *, n_contaminants: int = 4,
                               true_r_e_arcsec: float = 2.0, redshift: float = 0.05,
                               contaminant_offset_scale_arcsec: float = 8.0,
                               contaminant_r_e_arcsec: float = 1.2,
                               include_foreground_star: bool = True,
                               m_star: float = DEFAULT_M_STAR, alpha: float = DEFAULT_ALPHA,
                               cosmology=None) -> SyntheticHostField:
    if true_r_e_arcsec <= 0:
        raise HostAssociationError(f"true_r_e_arcsec must be positive, got {true_r_e_arcsec}")
    if n_contaminants < 0:
        raise HostAssociationError(f"n_contaminants must be non-negative, got {n_contaminants}")

    cosmology = cosmology or default_cosmology()
    transient = SourceRef(survey="synthetic", object_id="transient", ra_deg=180.0, dec_deg=0.0)

    true_offset_arcsec = float(rng.gamma(shape=2.0, scale=true_r_e_arcsec))
    true_angle = float(rng.uniform(0.0, 2.0 * math.pi))
    true_abs_mag = _sample_schechter_abs_mag(rng, m_star, alpha)
    true_apparent_mag = true_abs_mag + cosmology.distmod(redshift).value
    true_ra, true_dec = _offset_position(transient.ra_deg, transient.dec_deg, true_offset_arcsec, true_angle)
    true_host = SourceRef(survey="synthetic", object_id="host_true", ra_deg=true_ra, dec_deg=true_dec,
                          extra={"r_mean": true_apparent_mag})

    candidates = [true_host]
    redshifts: dict[str, tuple[float, str]] = {true_host.object_id: (redshift, "sdss")}
    r_e_arcsec: dict[str, float] = {true_host.object_id: true_r_e_arcsec}
    foreground_flags: dict[str, bool] = {}

    for i in range(n_contaminants):
        offset = float(rng.uniform(0.5, contaminant_offset_scale_arcsec))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        abs_mag = _sample_schechter_abs_mag(rng, m_star, alpha)
        contaminant_z = float(rng.uniform(0.02, 0.5))
        apparent_mag = abs_mag + cosmology.distmod(contaminant_z).value
        ra, dec = _offset_position(transient.ra_deg, transient.dec_deg, offset, angle)
        object_id = f"contaminant_{i}"
        candidates.append(SourceRef(survey="synthetic", object_id=object_id, ra_deg=ra, dec_deg=dec,
                                    extra={"r_mean": apparent_mag}))
        redshifts[object_id] = (contaminant_z, "sdss")
        r_e_arcsec[object_id] = contaminant_r_e_arcsec
        if i == 0 and include_foreground_star:
            foreground_flags[object_id] = True

    return SyntheticHostField(transient=transient, candidates=candidates, redshifts=redshifts,
                              r_e_arcsec=r_e_arcsec, foreground_flags=foreground_flags,
                              true_host_id=true_host.object_id)


@dataclass(frozen=True)
class TopKRecallResult:
    k_values: tuple[int, ...]
    n_trials: int
    recall: dict[int, float]

    def to_dict(self) -> dict:
        return {"k_values": list(self.k_values), "n_trials": self.n_trials,
                "recall": {str(k): round(v, 4) for k, v in self.recall.items()}}


def evaluate_top_k_recall(n_trials: int = 500, k_values: tuple[int, ...] = (1, 2, 3),
                          seed: int = 42, **field_kwargs) -> TopKRecallResult:
    """Fraction of trials where the known true host ranks within the top
    `k` candidates by posterior probability, for each `k`."""
    if n_trials < 1:
        raise HostAssociationError(f"n_trials must be at least 1, got {n_trials}")
    if not k_values:
        raise HostAssociationError("k_values must be non-empty")

    rng = np.random.default_rng(seed)
    hits = {k: 0 for k in k_values}
    for _ in range(n_trials):
        synthetic = synthesize_host_population(rng, **field_kwargs)
        result = associate_host(
            synthetic.transient, synthetic.candidates, redshifts=synthetic.redshifts,
            r_e_arcsec=synthetic.r_e_arcsec, foreground_flags=synthetic.foreground_flags)
        ranked_ids = [candidate.source.object_id for candidate in result.candidates]
        rank = ranked_ids.index(synthetic.true_host_id) if synthetic.true_host_id in ranked_ids else None
        for k in k_values:
            if rank is not None and rank < k:
                hits[k] += 1

    return TopKRecallResult(k_values=tuple(k_values), n_trials=n_trials,
                            recall={k: hits[k] / n_trials for k in k_values})


@dataclass(frozen=True)
class CalibrationResult:
    n_bins: int
    n_trials: int
    n_candidates: int
    bin_edges: list[float]
    bin_predicted: list[float | None]
    bin_empirical: list[float | None]
    bin_counts: list[int]
    expected_calibration_error: float

    def to_dict(self) -> dict:
        return {
            "n_bins": self.n_bins, "n_trials": self.n_trials, "n_candidates": self.n_candidates,
            "bin_edges": self.bin_edges, "bin_predicted": self.bin_predicted,
            "bin_empirical": self.bin_empirical, "bin_counts": self.bin_counts,
            "expected_calibration_error": self.expected_calibration_error,
        }


def evaluate_probability_calibration(n_trials: int = 500, n_bins: int = 10,
                                     seed: int = 43, **field_kwargs) -> CalibrationResult:
    """Reliability-curve binning: within each predicted-probability bin,
    compares the mean predicted posterior against the empirical fraction
    of candidates that were actually the true host, across every candidate
    (true host and contaminants alike) in every trial. `expected_
    calibration_error` is the standard count-weighted mean absolute gap
    between predicted and empirical per bin (Naeini et al. 2015-style
    ECE)."""
    if n_trials < 1:
        raise HostAssociationError(f"n_trials must be at least 1, got {n_trials}")
    if n_bins < 1:
        raise HostAssociationError(f"n_bins must be at least 1, got {n_bins}")

    rng = np.random.default_rng(seed)
    predicted: list[float] = []
    correct: list[float] = []
    for _ in range(n_trials):
        synthetic = synthesize_host_population(rng, **field_kwargs)
        result = associate_host(
            synthetic.transient, synthetic.candidates, redshifts=synthetic.redshifts,
            r_e_arcsec=synthetic.r_e_arcsec, foreground_flags=synthetic.foreground_flags)
        for candidate in result.candidates:
            predicted.append(candidate.posterior_probability)
            correct.append(1.0 if candidate.source.object_id == synthetic.true_host_id else 0.0)

    predicted_arr = np.asarray(predicted, dtype=np.float64)
    correct_arr = np.asarray(correct, dtype=np.float64)
    n = len(predicted_arr)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_predicted: list[float | None] = []
    bin_empirical: list[float | None] = []
    bin_counts: list[int] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (predicted_arr >= lo) & (predicted_arr <= hi if i == n_bins - 1 else predicted_arr < hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count:
            mean_pred = float(predicted_arr[mask].mean())
            mean_emp = float(correct_arr[mask].mean())
            bin_predicted.append(round(mean_pred, 4))
            bin_empirical.append(round(mean_emp, 4))
            ece += (count / n) * abs(mean_pred - mean_emp)
        else:
            bin_predicted.append(None)
            bin_empirical.append(None)

    return CalibrationResult(
        n_bins=n_bins, n_trials=n_trials, n_candidates=n,
        bin_edges=[round(float(edge), 4) for edge in edges],
        bin_predicted=bin_predicted, bin_empirical=bin_empirical, bin_counts=bin_counts,
        expected_calibration_error=round(float(ece), 4),
    )


__all__ = [
    "SyntheticHostField", "synthesize_host_population",
    "TopKRecallResult", "evaluate_top_k_recall",
    "CalibrationResult", "evaluate_probability_calibration",
]
