"""Composite candidate scoring (plan section 16).

A single machine-learning probability is a poor ranking key for research: it
collapses independent kinds of evidence into one number nobody can argue with.
Section 16 instead specifies a weighted composite, and this module implements
it with every component exposed separately so a researcher can see which
evidence actually drove a candidate up the list.

The weights below are the ones the plan proposes. They are deliberately data
rather than constants baked into the arithmetic, because section 16 says they
should eventually become experimental variables.

Components that cannot be computed return None rather than zero. That
distinction matters: zero means "checked and found nothing", None means "not
checked", and a candidate must not be penalised for evidence nobody gathered.
Missing components are renormalised out of the weighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Plan section 16, verbatim.
WEIGHT_VERSION = 1
WEIGHTS: dict[str, float] = {
    "statistical_rarity": 0.25,
    "cross_survey_consistency": 0.20,
    "temporal_uniqueness": 0.15,
    "physical_inconsistency": 0.15,
    "catalog_novelty": 0.10,
    "model_agreement": 0.10,
    "data_quality": 0.05,
}

# Approximate absolute G magnitudes for the classes a period search can
# suggest. These are rough class centres, not fits; they exist to catch
# objects that are wildly inconsistent with their apparent class, not to
# classify. Ranges are (period_low_d, period_high_d, abs_g_centre, tolerance).
PERIOD_LUMINOSITY_CLASSES = (
    ("delta_scuti", 0.02, 0.30, 2.0, 1.5),
    ("rr_lyrae", 0.20, 1.20, 0.6, 1.0),
    ("cepheid", 1.00, 100.0, -3.0, 2.5),
)


@dataclass
class ScoreBreakdown:
    """Every component of one candidate's score, with the total."""

    components: dict[str, float | None] = field(default_factory=dict)
    total: float = 0.0
    weight_used: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "weight_version": WEIGHT_VERSION,
            "total": round(self.total, 4),
            "weight_used": round(self.weight_used, 4),
            "components": {
                k: (None if v is None else round(v, 4))
                for k, v in self.components.items()
            },
            "weighted": {
                k: (None if v is None else round(v * WEIGHTS[k], 4))
                for k, v in self.components.items()
            },
            "reasons": self.reasons,
        }

    def top_drivers(self, count: int = 3) -> list[tuple[str, float]]:
        """The components contributing most to the total, for the explanation."""
        contributions = [
            (name, value * WEIGHTS[name])
            for name, value in self.components.items() if value is not None
        ]
        return sorted(contributions, key=lambda pair: -pair[1])[:count]


def combine(components: dict[str, float | None],
            reasons: list[str] | None = None) -> ScoreBreakdown:
    """Weighted mean over the components that could actually be computed."""
    usable = {k: v for k, v in components.items()
              if v is not None and np.isfinite(v) and k in WEIGHTS}

    weight_used = sum(WEIGHTS[k] for k in usable)
    total = (sum(WEIGHTS[k] * usable[k] for k in usable) / weight_used
             if weight_used > 0 else 0.0)

    return ScoreBreakdown(
        components={k: components.get(k) for k in WEIGHTS},
        total=float(total),
        weight_used=float(weight_used),
        reasons=list(reasons or []),
    )


def temporal_uniqueness(feature_values: dict[str, float]) -> float | None:
    """How unusual the object's behaviour in time is, independent of amplitude.

    Combines periodic coherence, abrupt changes and smoothness. A source can
    be highly variable and completely ordinary — an eclipsing binary is not
    interesting merely for varying — so amplitude is deliberately excluded.
    """
    parts: list[float] = []

    snr = feature_values.get("period_snr")
    if snr is not None and np.isfinite(snr):
        # Saturates around 50: beyond that a stronger peak says little more.
        parts.append(float(min(max(snr, 0.0) / 50.0, 1.0)))

    change = feature_values.get("change_point_score")
    if change is not None and np.isfinite(change):
        # White noise sits near 3; 30 is a decisive level change.
        parts.append(float(min(max(change - 3.0, 0.0) / 27.0, 1.0)))

    eta = feature_values.get("eta")
    if eta is not None and np.isfinite(eta):
        # Near 2 is white noise; well below 2 means correlated structure.
        parts.append(float(min(max(2.0 - eta, 0.0) / 2.0, 1.0)))

    return float(np.mean(parts)) if parts else None


def data_quality(feature_values: dict[str, float]) -> float | None:
    """Confidence in the measurements themselves.

    Low quality does not make a candidate uninteresting, but it does make the
    evidence weak, which is why this carries only 5% in section 16.
    """
    parts: list[float] = []

    points = feature_values.get("n_points")
    if points is not None and np.isfinite(points):
        # 500 epochs is a well-sampled ground-based curve.
        parts.append(float(min(points / 500.0, 1.0)))

    amplitude = feature_values.get("robust_amplitude")
    error = feature_values.get("median_err")
    if all(v is not None and np.isfinite(v) for v in (amplitude, error)) \
            and error > 0:
        # Signal-to-noise of the variation itself; 10 is comfortable.
        parts.append(float(min((amplitude / error) / 10.0, 1.0)))

    return float(np.mean(parts)) if parts else None


def physical_inconsistency(period_days: float | None,
                           gaia_properties: dict | None) -> tuple[float | None, str]:
    """Does the object's luminosity match what its period implies?

    A period alone suggests a variability class, and each class occupies a
    known place on the luminosity scale. A star pulsating like an RR Lyrae but
    a hundred times too faint for one is physically inconsistent, and that
    inconsistency is exactly the kind of thing worth a researcher's attention.

    The class centres used here are approximate. This is a screen for gross
    inconsistency, not a classifier, and it returns None whenever the Gaia
    astrometry needed to compute an absolute magnitude is missing.
    """
    if not gaia_properties or period_days is None or not np.isfinite(period_days):
        return None, "no Gaia astrometry or no period"

    abs_g = gaia_properties.get("abs_g_mag")
    # Prefer a supplied line-of-sight extinction estimate.  Gaia catalogue
    # rows do not always carry one, so the uncorrected value remains the
    # conservative fallback rather than inventing a dust map result.
    extinction = gaia_properties.get("a_g")
    if extinction is None:
        ebv = gaia_properties.get("ebv")
        if ebv is not None and np.isfinite(ebv):
            extinction = 2.74 * float(ebv)
    snr = gaia_properties.get("parallax_snr")
    if abs_g is None or not np.isfinite(abs_g):
        return None, "no absolute magnitude (parallax missing or non-positive)"
    if snr is not None and np.isfinite(snr) and snr < 5:
        return None, f"parallax too noisy to use (SNR {snr:.1f})"

    corrected_abs_g = float(abs_g) - float(extinction or 0.0)
    extinction_note = (f", extinction-corrected by A_G={float(extinction):.2f}"
                       if extinction is not None and np.isfinite(extinction)
                       else "")
    for name, low, high, centre, tolerance in PERIOD_LUMINOSITY_CLASSES:
        if low <= period_days <= high:
            deviation = abs(corrected_abs_g - centre)
            if deviation <= tolerance:
                return 0.0, (f"luminosity consistent with {name} "
                             f"(M_G={corrected_abs_g:.2f}, expected ~{centre:+.1f}"
                             f"{extinction_note})")
            # Scaled so a few magnitudes off saturates the score.
            score = float(min((deviation - tolerance) / 4.0, 1.0))
            return score, (f"M_G={corrected_abs_g:.2f} is {deviation:.1f} mag from the "
                           f"{name} locus (~{centre:+.1f}) at P={period_days:.4f} d")

    return None, f"period {period_days:.4f} d matches no modelled class"


def catalog_novelty(gaia_properties: dict | None = None,
                    known_variable: bool | None = None,
                    catalog_evidence: dict | None = None) -> tuple[float | None, str]:
    """Is this already a known, classified object?

    A catalogue response is preferred when available.  The provider states
    distinguish an actual no-match from an offline, rate-limited, or
    not-configured lookup; only the former is evidence of novelty.  Gaia's
    variability flag remains a weak backwards-compatible fallback for rows
    created before the catalogue enrichment job existed.
    """
    if catalog_evidence is not None:
        summary = catalog_evidence.get("summary", {})
        states = summary.get("states", {})
        known_variable = bool(summary.get("known_variable"))
        if known_variable:
            return 0.0, "already present in SIMBAD/VSX/TNS as a variable or transient"
        if summary.get("known_object"):
            return 0.2, "catalogued counterpart exists, but no variable/transient classification was returned"

        public_states = {states.get(name) for name in ("simbad", "vsx")}
        if not public_states <= {"match", "no_match"}:
            unavailable = ", ".join(sorted(str(state) for state in public_states
                                              if state not in {"match", "no_match"}))
            return None, f"catalogue cross-reference incomplete ({unavailable or 'not queried'})"
        if states.get("tns") in {"match", "no_match"}:
            return 1.0, "SIMBAD, VSX and TNS returned no known variable/transient match"
        return 0.9, "SIMBAD and VSX returned no known match; TNS was not available"

    if known_variable is not None:
        return (0.0 if known_variable else 1.0), (
            "already catalogued as variable" if known_variable
            else "not flagged as a known variable")

    if not gaia_properties:
        return None, "no catalogue cross-reference available"

    flag = gaia_properties.get("phot_variable_flag")
    if flag is None:
        return None, "no catalogue cross-reference available"

    is_known = str(flag).upper().startswith("VARIABLE")
    return (0.0 if is_known else 0.7), (
        "flagged VARIABLE in Gaia DR3" if is_known
        else "not flagged variable in Gaia DR3 (weak evidence only)")


def score_candidate(feature_values: dict[str, float],
                    anomaly_score: float | None = None,
                    model_agreement: int | None = None,
                    detector_count: int = 4,
                    consistency: float | None = None,
                    gaia_properties: dict | None = None,
                    known_variable: bool | None = None,
                    catalog_evidence: dict | None = None) -> ScoreBreakdown:
    """Assemble the full section 16 composite for one candidate."""
    reasons: list[str] = []

    physical, physical_reason = physical_inconsistency(
        feature_values.get("best_period_days"), gaia_properties)
    reasons.append(f"physical: {physical_reason}")

    novelty, novelty_reason = catalog_novelty(
        gaia_properties, known_variable, catalog_evidence)
    reasons.append(f"novelty: {novelty_reason}")

    components: dict[str, float | None] = {
        "statistical_rarity": anomaly_score,
        "cross_survey_consistency": consistency,
        "temporal_uniqueness": temporal_uniqueness(feature_values),
        "physical_inconsistency": physical,
        "catalog_novelty": novelty,
        "model_agreement": (None if model_agreement is None
                            else float(model_agreement) / max(detector_count, 1)),
        "data_quality": data_quality(feature_values),
    }

    return combine(components, reasons)
