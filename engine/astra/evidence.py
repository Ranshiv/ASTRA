"""Cross-survey evidence scoring (plan sections 15 and 16).

This is the module the whole project is arguing for: one survey saying an
object is odd is a claim, and two independent instruments saying the same
thing is evidence. The distinction matters because the most common cause of a
single-survey anomaly is the survey itself — a bad pixel column, a diffraction
spike, an unflagged cosmic ray — and none of those repeat on a different
telescope with a different detector and a different cadence.

The strongest available check is period agreement. Two surveys with different
cadences, different filters and different systematics have no shared reason to
produce the same period unless the star really is pulsating at it. Aliasing is
handled explicitly: a period and its 2x, 1/2x and 1-day-alias relatives are
counted as agreement, because that is what the underlying signal looks like
through two different sampling windows.

Nothing here decides that a candidate is real. It assembles the evidence and
scores its internal consistency, which is what plan section 17 needs in order
to explain a candidate to a researcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import features as features_mod, store, timeframe
from .crossmatch import MatchGroup
from .surveys.base import LightCurve

# Two periods count as agreeing within this fractional tolerance.
PERIOD_TOLERANCE = 0.02

# Harmonic relationships a period search legitimately confuses.
HARMONIC_RATIOS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)

# Fallback long-period end of the searched band, used only when a caller
# cannot supply the real one. The true limit is half the observed baseline
# (features.MAX_PERIOD_FRACTION), which `score_profile` passes in per pair.
DEFAULT_SEARCH_MAX_PERIOD_DAYS = 100.0

# Effective wavelength of each known band, in nanometres. Only the ordering is
# used — to decide which of two bands is the bluer one when checking colour
# consistency — so approximate values are sufficient.
BAND_WAVELENGTH_NM = {
    "u": 355.0, "g": 484.0, "r": 641.0, "i": 810.0, "z": 900.0, "y": 1004.0,
    "TESS": 786.0,
    "G": 639.0, "BP": 518.0, "RP": 782.0,
}

# How far two amplitudes may differ before they stop counting as consistent,
# as a multiplicative factor. Wider across different bands, because a real
# variable genuinely has a different amplitude in each filter — that
# difference is the colour, not an inconsistency.
AMPLITUDE_TOLERANCE_SAME_BAND = 1.5
AMPLITUDE_TOLERANCE_CROSS_BAND = 3.0

# Weights for the consistency score. Deliberately explicit and adjustable —
# plan section 16 says these should eventually become experimental variables.
#
# Version 2 adds amplitude_agreement and discounts period_agreement by its
# false-alarm probability. Both change the numbers a stored profile would get,
# which is exactly what the version is for: a v1 consistency and a v2
# consistency are not the same quantity and must not be compared.
WEIGHT_VERSION = 2
WEIGHTS = {
    "independent_detection": 0.27,
    "period_agreement": 0.27,
    "variability_agreement": 0.18,
    "amplitude_agreement": 0.10,
    "temporal_overlap": 0.09,
    "positional_quality": 0.09,
}


@dataclass
class SurveyView:
    """What one survey saw of one object."""

    survey: str
    object_id: str
    band: str
    points: int
    reduced_chi2: float
    best_period_days: float
    period_snr: float
    robust_amplitude: float
    time_start: float
    time_end: float
    # Last, with a default, so the positional construction used throughout the
    # tests and the pipeline keeps working unchanged.
    value_kind: str = "mag"
    median_value: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "survey": self.survey,
            "object_id": self.object_id,
            "band": self.band,
            "value_kind": self.value_kind,
            "points": self.points,
            "reduced_chi2": _round(self.reduced_chi2),
            "best_period_days": _round(self.best_period_days, 6),
            "period_snr": _round(self.period_snr),
            "robust_amplitude": _round(self.robust_amplitude, 4),
            "fractional_amplitude": _round(fractional_amplitude(self), 5),
            "baseline_days": _round(self.time_end - self.time_start, 3),
        }


@dataclass
class CrossSurveyProfile:
    """All evidence gathered for one object, plus the consistency score."""

    views: list[SurveyView] = field(default_factory=list)
    separations_arcsec: dict[str, float] = field(default_factory=dict)
    ambiguous: list[str] = field(default_factory=list)
    blended: list[str] = field(default_factory=list)
    match_radius_arcsec: float = 2.0
    consistency: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # Fraction of the total weight that was actually available. A consistency
    # of 0.7 computed from 0.90 of the weight is not the same number as 0.7
    # computed from all of it, so callers comparing profiles must check this.
    weight_used: float = 0.0
    # Probability that two unrelated periods would pass the alias-tolerant
    # agreement test anyway. None when no period pair was comparable.
    period_fap: float | None = None

    @property
    def independent_surveys(self) -> int:
        return len({view.survey for view in self.views})

    @property
    def resolved_surveys(self) -> int:
        """Surveys that resolve this object rather than blending it."""
        return len({v.survey for v in self.views if v.survey not in self.blended})

    def to_dict(self) -> dict:
        return {
            "independent_surveys": self.independent_surveys,
            "resolved_surveys": self.resolved_surveys,
            "views": [v.to_dict() for v in self.views],
            "separations_arcsec": {k: round(v, 4)
                                   for k, v in self.separations_arcsec.items()},
            "ambiguous": self.ambiguous,
            "blended": self.blended,
            "consistency": round(self.consistency, 4),
            "weight_version": WEIGHT_VERSION,
            "weight_used": round(self.weight_used, 4),
            "period_fap": (None if self.period_fap is None
                           else round(self.period_fap, 5)),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            # The weight each component carries, and what it actually
            # contributed. A component scoring 0.9 at weight 0.09 moves the
            # result less than one scoring 0.5 at weight 0.27, and a reader
            # given only the raw scores cannot tell those apart.
            "weights": {k: WEIGHTS[k] for k in self.components if k in WEIGHTS},
            "weighted": {k: round(WEIGHTS[k] * v, 4)
                         for k, v in self.components.items() if k in WEIGHTS},
            "notes": self.notes,
        }


def _round(value: float, digits: int = 3) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def periods_agree(first: float, second: float,
                  tolerance: float = PERIOD_TOLERANCE) -> tuple[bool, str]:
    """Do two periods describe the same signal, allowing for harmonics?

    A period search on sparse ground-based data frequently locks onto twice or
    half the true period, and onto the 1-day alias imposed by only observing
    at night. Treating those as disagreement would throw away real matches.
    """
    if not (np.isfinite(first) and np.isfinite(second)) or first <= 0 or second <= 0:
        return False, "unavailable"

    for ratio in HARMONIC_RATIOS:
        if abs(second - first * ratio) <= tolerance * first * max(ratio, 1.0):
            if ratio == 1.0:
                return True, "direct"
            return True, f"harmonic x{ratio:g}"

    # One-day aliasing: 1/P_obs = 1/P_true +- n, from nightly sampling.
    for n in (1, 2):
        for sign in (1, -1):
            denominator = 1.0 / first + sign * n
            if abs(denominator) < 1e-9:
                continue
            alias = 1.0 / denominator
            if alias > 0 and abs(second - alias) <= tolerance * max(alias, first):
                return True, f"1-day alias n={sign * n}"

    return False, "disagree"


def _acceptance_windows(first: float,
                        tolerance: float = PERIOD_TOLERANCE
                        ) -> list[tuple[float, float]]:
    """Every period interval `periods_agree` would accept against `first`.

    Mirrors that function's branches exactly rather than approximating them,
    so the false-alarm probability describes the test actually being applied
    and cannot drift away from it.
    """
    windows: list[tuple[float, float]] = []

    for ratio in HARMONIC_RATIOS:
        target = first * ratio
        half = tolerance * first * max(ratio, 1.0)
        windows.append((target - half, target + half))

    for n in (1, 2):
        for sign in (1, -1):
            denominator = 1.0 / first + sign * n
            if abs(denominator) < 1e-9:
                continue
            alias = 1.0 / denominator
            if alias <= 0:
                continue
            half = tolerance * max(alias, first)
            windows.append((alias - half, alias + half))

    return windows


def period_agreement_fap(first: float,
                         tolerance: float = PERIOD_TOLERANCE,
                         min_period_days: float | None = None,
                         max_period_days: float | None = None) -> float:
    """Probability an unrelated period would pass the agreement test anyway.

    `periods_agree` accepts five harmonic ratios and four one-day aliases, each
    with its own tolerance window. That is roughly eight chances to agree by
    coincidence, and a boolean hides it: an alias-tolerant match on sparse data
    is genuinely weaker evidence than a direct one, and the score should say so
    rather than treating both as certainty.

    Measured in FREQUENCY, not period, because that is where the null lives.
    `features.periodogram` searches a uniform frequency grid, so an unrelated
    peak is uniform in frequency; a uniform-in-period null would badly
    mis-weight the short-period end.

    Returns 1.0 when the searched band is degenerate, so an unusable test is
    reported as carrying no information rather than as certainty.
    """
    if not np.isfinite(first) or first <= 0:
        return 1.0

    min_period = (features_mod.MIN_PERIOD_DAYS if min_period_days is None
                  else float(min_period_days))
    max_period = (DEFAULT_SEARCH_MAX_PERIOD_DAYS if max_period_days is None
                  else float(max_period_days))
    if not (np.isfinite(min_period) and np.isfinite(max_period)):
        return 1.0
    if min_period <= 0 or max_period <= min_period:
        return 1.0

    band_low, band_high = 1.0 / max_period, 1.0 / min_period

    # Period window -> frequency window, clipped to the searched band.
    intervals: list[tuple[float, float]] = []
    for low, high in _acceptance_windows(first, tolerance):
        if high <= 0:
            continue
        low = max(low, 1e-12)
        f_low, f_high = 1.0 / high, 1.0 / low
        f_low, f_high = max(f_low, band_low), min(f_high, band_high)
        if f_high > f_low:
            intervals.append((f_low, f_high))

    if not intervals:
        return 0.0

    # Merge before measuring: the harmonic and alias windows overlap for short
    # periods, and counting an overlap twice would overstate the FAP.
    intervals.sort()
    covered = 0.0
    current_low, current_high = intervals[0]
    for low, high in intervals[1:]:
        if low > current_high:
            covered += current_high - current_low
            current_low, current_high = low, high
        else:
            current_high = max(current_high, high)
    covered += current_high - current_low

    return float(min(covered / (band_high - band_low), 1.0))


def fractional_amplitude(view: SurveyView) -> float:
    """Amplitude as a fraction of the source's own brightness.

    Raw amplitudes cannot be compared across surveys: ZTF reports magnitudes
    and TESS reports electron flux, so a flux amplitude also carries how bright
    the star happens to be in that detector. Dividing flux by the median makes
    it dimensionless, and dividing a magnitude amplitude by 1.0857 converts it
    to the same quantity through dmag = 1.0857 * df/f.

    That conversion is a small-amplitude linearisation. It is good to a few
    percent below about 0.2 mag and degrades above roughly 0.5 mag, which
    `amplitude_agreement` notes rather than silently absorbing.
    """
    amplitude = view.robust_amplitude
    if not np.isfinite(amplitude) or amplitude <= 0:
        return float("nan")

    if view.value_kind == "flux":
        median = view.median_value
        if not np.isfinite(median) or median == 0:
            return float("nan")
        return float(amplitude / abs(median))

    return float(amplitude / 1.0857)


def amplitude_agreement(first: SurveyView,
                        second: SurveyView) -> tuple[float, str] | None:
    """How consistent two surveys' amplitudes are, or None if incomparable.

    Returning None rather than 0.0 matters. A pair that could not be compared
    has produced no evidence either way, and scoring that as total disagreement
    would penalise an object for a measurement nobody made -- the same reason
    `scoring.py` renormalises over the components it actually has.
    """
    left, right = fractional_amplitude(first), fractional_amplitude(second)
    if not (np.isfinite(left) and np.isfinite(right)):
        return None
    if left <= 0 or right <= 0:
        return None

    same_band = first.band == second.band
    tolerance = (AMPLITUDE_TOLERANCE_SAME_BAND if same_band
                 else AMPLITUDE_TOLERANCE_CROSS_BAND)
    ratio = max(left, right) / min(left, right)
    score = max(0.0, 1.0 - float(np.log(ratio)) / float(np.log(tolerance)))
    note = f"amplitude ratio {ratio:.2f}x"

    # Colour consistency: for essentially every pulsating and eclipsing
    # variable the amplitude is larger in the bluer band. A redder band showing
    # markedly MORE variation hints that the two detections are not the same
    # star, which is exactly the blend this component exists to catch.
    first_nm = BAND_WAVELENGTH_NM.get(first.band)
    second_nm = BAND_WAVELENGTH_NM.get(second.band)
    if not same_band and first_nm is not None and second_nm is not None:
        bluer, redder = ((left, right) if first_nm < second_nm
                         else (right, left))
        if redder > bluer * AMPLITUDE_TOLERANCE_SAME_BAND:
            score *= 0.5
            note += "; redder band varies more than the bluer one"

    if first.value_kind != second.value_kind:
        mags = [v.robust_amplitude for v in (first, second)
                if v.value_kind == "mag" and np.isfinite(v.robust_amplitude)]
        if mags and max(mags) > 0.5:
            note += "; large-amplitude mag/flux conversion is approximate"

    return float(min(score, 1.0)), note


def view_key(curve: LightCurve) -> tuple[str, str, str]:
    return (curve.source.survey, curve.source.object_id, curve.band)


def build_view(curve: LightCurve,
               feature_lookup: dict[tuple[str, str, str], dict] | None = None
               ) -> SurveyView:
    """Summarise one curve, after aligning it to the common time frame.

    A caller that already holds extracted features can pass them in. Without
    that, this re-runs the period search, which profiling measured at ~970 ms
    per group — 392 s across 404 groups, and the single largest remaining cost
    in the pipeline once feature caching was in place.
    """
    aligned = timeframe.align(curve)

    if feature_lookup is not None:
        cached = feature_lookup.get(view_key(curve))
        if cached is not None:
            return SurveyView(
                survey=aligned.source.survey,
                object_id=aligned.source.object_id,
                band=aligned.band,
                points=int(cached.get("n_points", len(aligned)) or 0),
                reduced_chi2=cached.get("reduced_chi2", float("nan")),
                best_period_days=cached.get("best_period_days", float("nan")),
                period_snr=cached.get("period_snr", float("nan")),
                robust_amplitude=cached.get("robust_amplitude", float("nan")),
                time_start=float(aligned.time[0]) if len(aligned) else float("nan"),
                time_end=float(aligned.time[-1]) if len(aligned) else float("nan"),
                value_kind=str(aligned.value_kind),
                median_value=cached.get("median", float("nan")),
            )

    extracted = features_mod.extract(aligned).values

    return SurveyView(
        survey=aligned.source.survey,
        object_id=aligned.source.object_id,
        band=aligned.band,
        points=int(extracted["n_points"]),
        reduced_chi2=extracted["reduced_chi2"],
        best_period_days=extracted["best_period_days"],
        period_snr=extracted["period_snr"],
        robust_amplitude=extracted["robust_amplitude"],
        time_start=float(aligned.time[0]) if len(aligned) else float("nan"),
        time_end=float(aligned.time[-1]) if len(aligned) else float("nan"),
        value_kind=str(aligned.value_kind),
        median_value=extracted["median"],
    )


def _pair_fap(first: SurveyView, second: SurveyView) -> float:
    """False-alarm probability for one pair, over the band both could search.

    The long-period end is set by the SHORTER of the two baselines, because
    neither survey could have reported a period longer than its own record
    supports (features.MAX_PERIOD_FRACTION).
    """
    baselines = [v.time_end - v.time_start for v in (first, second)
                 if np.isfinite(v.time_start) and np.isfinite(v.time_end)]
    baselines = [b for b in baselines if b > 0]
    max_period = (min(baselines) * features_mod.MAX_PERIOD_FRACTION
                  if len(baselines) == 2 else None)
    return period_agreement_fap(first.best_period_days,
                                max_period_days=max_period)


def score_profile(profile: CrossSurveyProfile) -> CrossSurveyProfile:
    """Compute the consistency score from the assembled views."""
    components: dict[str, float] = {}
    notes: list[str] = []
    views = profile.views

    # Blended detections are excluded from the count. A TESS source shared
    # with several neighbours corroborates the neighbourhood, not this object,
    # and counting it would manufacture confirmation for every star sitting
    # near a bright variable.
    surveys = {v.survey for v in views}
    resolved = {v.survey for v in views if v.survey not in profile.blended}
    components["independent_detection"] = min(len(resolved) / 3.0, 1.0)

    if len(resolved) < 2:
        notes.append("Fewer than two surveys resolve this object; cannot rule "
                     "out an instrumental artifact from cross-survey evidence "
                     "alone.")
    if profile.blended:
        notes.append(f"Blended in {', '.join(sorted(profile.blended))}: the "
                     f"counterpart is unresolved at that survey's pixel scale, "
                     f"so its agreement is not independent evidence.")

    credible = [v for v in views
                if np.isfinite(v.best_period_days) and v.period_snr > 5]
    if len(credible) >= 2:
        agreements, total, kinds = 0, 0, []
        faps: list[float] = []
        for i in range(len(credible)):
            for j in range(i + 1, len(credible)):
                if credible[i].survey == credible[j].survey:
                    continue
                total += 1
                agreed, kind = periods_agree(credible[i].best_period_days,
                                             credible[j].best_period_days)
                faps.append(_pair_fap(credible[i], credible[j]))
                if agreed:
                    agreements += 1
                    kinds.append(kind)

        fraction = agreements / total if total else 0.0
        # Discount by how often this test agrees on unrelated periods. With
        # eight acceptable values at 2% each, a bare "they agree" overstates
        # the evidence by a few percent, and the alias-heavy short-period
        # cases by considerably more.
        fap = float(np.mean(faps)) if faps else None
        profile.period_fap = fap
        components["period_agreement"] = fraction * (1.0 - (fap or 0.0))

        if agreements:
            notes.append(
                f"Period agrees across surveys ({', '.join(sorted(set(kinds)))}); "
                f"an unrelated period passes this test {(fap or 0.0):.1%} of "
                f"the time.")
        elif total:
            notes.append("Surveys report incompatible periods.")
    else:
        components["period_agreement"] = 0.0
        notes.append("Fewer than two surveys yielded a credible period.")

    # Variability should be detected by every survey that looked, not just one.
    variable = [v for v in views if np.isfinite(v.reduced_chi2)]
    if len(variable) >= 2:
        flags = [v.reduced_chi2 > 2.0 for v in variable]
        components["variability_agreement"] = (
            1.0 if all(flags) else (0.5 if any(flags) else 1.0)
        )
        if any(flags) and not all(flags):
            notes.append("Only some surveys see significant variability.")
    else:
        components["variability_agreement"] = 0.0

    # Amplitude and colour consistency. Omitted rather than zeroed when no
    # pair is comparable -- see `amplitude_agreement`.
    amplitude_scores, amplitude_notes = [], []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            if views[i].survey == views[j].survey:
                continue
            result = amplitude_agreement(views[i], views[j])
            if result is None:
                continue
            score, note = result
            amplitude_scores.append(score)
            amplitude_notes.append(
                f"{views[i].survey} {views[i].band} vs "
                f"{views[j].survey} {views[j].band}: {note}")

    if amplitude_scores:
        components["amplitude_agreement"] = float(np.mean(amplitude_scores))
        if components["amplitude_agreement"] < 0.5:
            notes.append("Amplitudes disagree between surveys "
                         f"({'; '.join(amplitude_notes)}), which is what a "
                         f"contaminating blend looks like.")
    elif len(views) >= 2:
        notes.append("No comparable amplitude pair; that component is "
                     "excluded from the score rather than counted against "
                     "this object.")

    spans = [(v.time_start, v.time_end) for v in views
             if np.isfinite(v.time_start) and np.isfinite(v.time_end)]
    if len(spans) >= 2:
        start = max(s for s, _ in spans)
        end = min(e for _, e in spans)
        overlap = max(0.0, end - start)
        longest = max(e - s for s, e in spans) or 1.0
        components["temporal_overlap"] = float(min(overlap / longest, 1.0))
        if overlap <= 0:
            notes.append("Surveys never observed simultaneously; only "
                         "persistent behaviour can be corroborated.")
    else:
        components["temporal_overlap"] = 0.0

    if profile.separations_arcsec:
        # Scored against the radius actually used, not a fixed one: matching
        # at 15 arcsec and grading against 2 would saturate every group at
        # zero and make the component meaningless.
        worst = max(profile.separations_arcsec.values())
        scale = max(profile.match_radius_arcsec, 1e-6)
        components["positional_quality"] = float(max(0.0, 1.0 - worst / scale))
        if profile.ambiguous:
            notes.append(f"Crowded field: multiple counterparts in "
                         f"{', '.join(profile.ambiguous)}.")
    else:
        components["positional_quality"] = 0.0

    profile.components = components

    # Renormalise over the weight actually available. Missing evidence must
    # not read as bad evidence; `weight_used` is reported so two consistency
    # scores computed from different amounts of evidence are not mistaken for
    # the same measurement.
    available = {k: v for k, v in components.items() if k in WEIGHTS}
    weight_used = sum(WEIGHTS[k] for k in available)
    profile.weight_used = float(weight_used)
    profile.consistency = (
        float(sum(WEIGHTS[k] * available[k] for k in available) / weight_used)
        if weight_used > 0 else 0.0)
    profile.notes = notes
    return profile


def profile_group(group: MatchGroup,
                  curves_by_key: dict[tuple[str, str], list[LightCurve]],
                  feature_lookup: dict[tuple[str, str, str], dict] | None = None
                  ) -> CrossSurveyProfile:
    """Assemble and score the evidence for one matched group."""
    profile = CrossSurveyProfile(
        separations_arcsec=dict(group.separations),
        ambiguous=sorted(group.ambiguous),
        blended=sorted(group.blended),
        match_radius_arcsec=group.match_radius_arcsec,
    )

    for survey, source in group.members.items():
        for curve in curves_by_key.get((survey, source.object_id), []):
            if len(curve) > 0:
                profile.views.append(build_view(curve, feature_lookup))

    return score_profile(profile)


def feature_lookup_from_matrices(survey_names: list[str] | None = None,
                                 matrices: dict[str, object] | None = None
                                 ) -> dict[tuple[str, str, str], dict]:
    """Build a (survey, object_id, band) -> features map from cached matrices.

    Reuses the feature cache, so assembling every cross-survey profile costs
    a dictionary lookup instead of a fresh period search per curve.

    A caller that has already built the per-survey matrices can pass them in.
    `pipeline.run` does, because it needs the same matrices itself: building
    them here as well walked the whole store a second time for no new
    information.
    """
    from . import featurematrix, surveys

    lookup: dict[tuple[str, str, str], dict] = {}
    for survey in (survey_names or surveys.available()):
        matrix = (matrices or {}).get(survey)
        if matrix is None:
            matrix = featurematrix.build(survey=survey)
        for row, identity in enumerate(matrix.identities):
            values = {name: float(matrix.values[row, column])
                      for column, name in enumerate(matrix.feature_names)}
            lookup[(identity["survey"], identity["object_id"],
                    identity["band"])] = values
    return lookup


@dataclass
class CurveIndex:
    """Every stored curve, indexed the two ways the pipeline needs it.

    Both views come from ONE walk of the store. Reading the same Parquet file
    once for the cross-survey index and again to recover a candidate's sky
    position was measurable: position lookup alone was one file read per
    candidate, on top of a full walk that had already opened every one of them.
    """

    by_key: dict[tuple[str, str], list[LightCurve]] = field(default_factory=dict)
    positions_by_path: dict[str, dict] = field(default_factory=dict)


def load_curve_index(root=None) -> CurveIndex:
    """Walk the store once, returning both the key index and positions."""
    from . import config

    root = root or config.PATHS.datasets
    index = CurveIndex()
    if not root.exists():
        return index

    for path in sorted(root.rglob("*.parquet")):
        try:
            curve = store.read_curve(path)
        except Exception:  # noqa: BLE001 - a corrupt file must not stop indexing
            continue
        index.by_key.setdefault((curve.source.survey, curve.source.object_id),
                                []).append(curve)
        index.positions_by_path[str(path)] = {
            "ra_deg": curve.source.ra_deg,
            "dec_deg": curve.source.dec_deg,
        }
    return index


def load_curves_by_key(root=None) -> dict[tuple[str, str], list[LightCurve]]:
    """Index every stored curve by (survey, object_id)."""
    return load_curve_index(root).by_key
