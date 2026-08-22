"""Artifact likelihood assessment (plan sections 4, 17 and 36).

Plan section 4 is unambiguous: being able to conclude "this is probably an
artifact" is as important as finding a real candidate. Section 36 adds that an
anomaly must never be assumed astrophysical by default.

That is not a pessimistic framing, it is the arithmetic of survey astronomy.
Genuinely rare astrophysics is rare; detector defects, cosmic rays, satellite
trails, diffraction spikes, blending and calibration jumps are common. An
unusual measurement is therefore far more likely to be instrumental than
astrophysical unless something specifically argues otherwise.

Each indicator below is a concrete, checkable reason for suspicion, reported
by name so a researcher can agree or disagree with the machine rather than
receive a bare probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Periods that recur because of how the observing works, not because of the
# star. A signal at one of these is suspect until shown otherwise.
SAMPLING_PERIODS_DAYS = {
    "1 sidereal day": 0.99727,
    "1 solar day": 1.0,
    "half day": 0.5,
    "1 lunar month": 29.53,
    "1 year": 365.25,
    "TESS orbit": 13.7,
}
SAMPLING_TOLERANCE = 0.01  # fractional

# Per-indicator probabilities for the noisy-OR combination below. Originally
# all hand-set judgement calls (plan section 17 always described them that
# way). The six FEATURE_INDICATOR_NAMES below are now calibrated against a
# synthetic injection-recovery study (artifact.calibrate_from_injection) --
# see docs/DEFERRED.txt for the measured result. The remaining four are
# structural/categorical (cross-survey resolution and period agreement, not
# a statistical threshold on one curve's own features) and are not
# calibrated by that study; they stay hand-set, documented as such.
WEIGHTS: dict[str, float] = {
    "sampling_period": 0.35,
    "low_significance": 0.30,
    "consistent_with_constant": 0.25,
    "sparse_sampling": 0.20,
    "step_change": 0.20,
    "extreme_outliers": 0.25,
    "single_instrument": 0.30,
    "blended_photometry": 0.20,
    "agreement_not_independent": 0.05,
    "period_disagreement": 0.25,
}

# The subset calibrate_from_injection() can measure: each is a threshold on
# a statistic of one curve's own features. The other four WEIGHTS entries
# depend on cross-survey resolution/blending, which a single synthetic
# curve cannot represent.
FEATURE_INDICATOR_NAMES = (
    "sampling_period", "low_significance", "consistent_with_constant",
    "sparse_sampling", "step_change", "extreme_outliers",
)


@dataclass
class ArtifactIndicator:
    name: str
    weight: float
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "weight": round(self.weight, 3),
                "detail": self.detail}


@dataclass
class ArtifactAssessment:
    """Likelihood the signal is instrumental, with the reasons behind it."""

    likelihood: float = 0.0
    indicators: list[ArtifactIndicator] = field(default_factory=list)
    clearing_evidence: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.likelihood >= 0.6:
            return "probably an artifact"
        if self.likelihood >= 0.3:
            return "possibly an artifact"
        return "no strong artifact indicators"

    def to_dict(self) -> dict:
        return {
            "likelihood": round(self.likelihood, 4),
            "verdict": self.verdict,
            "indicators": [i.to_dict() for i in self.indicators],
            "clearing_evidence": self.clearing_evidence,
        }


def matches_sampling_period(period_days: float | None
                            ) -> tuple[bool, str]:
    """Is the period one the observing cadence imposes rather than the star?"""
    if period_days is None or not np.isfinite(period_days) or period_days <= 0:
        return False, ""

    for name, value in SAMPLING_PERIODS_DAYS.items():
        if abs(period_days - value) <= SAMPLING_TOLERANCE * value:
            return True, name
    return False, ""


def assess(feature_values: dict[str, float],
           resolved_surveys: int = 1,
           blended: list[str] | None = None,
           period_agrees_across_surveys: bool | None = None
           ) -> ArtifactAssessment:
    """Weigh the instrumental explanations for an apparent anomaly."""
    indicators: list[ArtifactIndicator] = []
    clearing: list[str] = []
    blended = blended or []

    period = feature_values.get("best_period_days")
    is_sampling, sampling_name = matches_sampling_period(period)
    if is_sampling:
        indicators.append(ArtifactIndicator(
            "sampling_period", WEIGHTS["sampling_period"],
            f"Period {period:.4f} d matches {sampling_name}; this is what the "
            f"observing cadence produces, not necessarily the star."))

    amplitude = feature_values.get("robust_amplitude")
    error = feature_values.get("median_err")
    if all(v is not None and np.isfinite(v) for v in (amplitude, error)):
        if error > 0 and amplitude < 3.0 * error:
            indicators.append(ArtifactIndicator(
                "low_significance", WEIGHTS["low_significance"],
                f"Variation ({amplitude:.3f}) is only {amplitude / error:.1f}x "
                f"the typical uncertainty ({error:.3f})."))
        elif error > 0 and amplitude > 10.0 * error:
            clearing.append(f"Variation is {amplitude / error:.0f}x the typical "
                            f"uncertainty, far above the noise.")

    chi2 = feature_values.get("reduced_chi2")
    if chi2 is not None and np.isfinite(chi2):
        if chi2 < 2.0:
            indicators.append(ArtifactIndicator(
                "consistent_with_constant", WEIGHTS["consistent_with_constant"],
                f"Reduced chi-square of {chi2:.2f} is consistent with a "
                f"constant source; the scatter is explained by the errors."))
        elif chi2 > 20.0:
            clearing.append(f"Reduced chi-square of {chi2:.1f} rules out a "
                            f"constant source decisively.")

    points = feature_values.get("n_points")
    if points is not None and np.isfinite(points) and points < 50:
        indicators.append(ArtifactIndicator(
            "sparse_sampling", WEIGHTS["sparse_sampling"],
            f"Only {int(points)} usable epochs; a handful of bad measurements "
            f"could produce this entirely."))

    change = feature_values.get("change_point_score")
    if change is not None and np.isfinite(change) and change > 50:
        # A step is physically possible but far more often a recalibration,
        # a filter change or a pipeline version boundary.
        indicators.append(ArtifactIndicator(
            "step_change", WEIGHTS["step_change"],
            f"Abrupt persistent level change (z={change:.0f}); commonly a "
            f"recalibration or pipeline boundary rather than the source."))

    kurtosis = feature_values.get("kurtosis")
    if kurtosis is not None and np.isfinite(kurtosis) and kurtosis > 20:
        indicators.append(ArtifactIndicator(
            "extreme_outliers", WEIGHTS["extreme_outliers"],
            f"Excess kurtosis of {kurtosis:.0f}: the signal is dominated by a "
            f"few extreme points, the signature of cosmic rays or bad pixels."))

    if resolved_surveys < 2:
        indicators.append(ArtifactIndicator(
            "single_instrument", WEIGHTS["single_instrument"],
            "Only one survey resolves this object, so no independent "
            "instrument can corroborate the behaviour."))
    else:
        clearing.append(f"{resolved_surveys} surveys resolve this object "
                        f"independently.")

    if blended:
        indicators.append(ArtifactIndicator(
            "blended_photometry", WEIGHTS["blended_photometry"],
            f"Unresolved in {', '.join(blended)}; the flux there is a sum over "
            f"neighbouring sources."))

    if period_agrees_across_surveys is True:
        # Only counts if a second survey actually resolves the object. A
        # blended counterpart measures the same photons from the same
        # neighbourhood, so its agreement is not an independent check — and
        # crediting it here would contradict the single_instrument indicator
        # raised above.
        if resolved_surveys >= 2:
            clearing.append("Independent instruments recover a consistent "
                            "period, which detector defects do not reproduce.")
        else:
            indicators.append(ArtifactIndicator(
                "agreement_not_independent", WEIGHTS["agreement_not_independent"],
                "A consistent period is reported by another survey, but that "
                "survey does not resolve this object, so the agreement is not "
                "independent corroboration."))
    elif period_agrees_across_surveys is False:
        indicators.append(ArtifactIndicator(
            "period_disagreement", WEIGHTS["period_disagreement"],
            "Surveys recover incompatible periods, suggesting at least one is "
            "instrumental."))

    # Combined as independent probabilities rather than summed, so several
    # weak hints accumulate without any single one saturating the result.
    survival = 1.0
    for indicator in indicators:
        survival *= (1.0 - indicator.weight)
    likelihood = 1.0 - survival

    # Clearing evidence reduces, but never eliminates, the suspicion.
    likelihood *= (0.75 ** len(clearing))

    return ArtifactAssessment(
        likelihood=float(np.clip(likelihood, 0.0, 1.0)),
        indicators=sorted(indicators, key=lambda i: -i.weight),
        clearing_evidence=clearing,
    )


# ---------------------------------------------------------------------------
# Calibration: synthetic injection-recovery study for the six feature-based
# indicators above (plan section 17's weights were hand-set judgement calls
# until this was written; see docs/DEFERRED.txt Phase 7).
#
# No external labelled dataset fits this: SNAD's ZTF DR3 "Dataset of
# artefacts" (arXiv:2504.08053) publishes 28x28 / 63x63 FITS image cutouts
# with no ZTF object IDs or light-curve data at all, and their broader
# anomaly-detection tooling (zwad) ships light-curve features but no bundled
# artifact/real ground truth. artifact.assess operates entirely on one
# curve's own light-curve statistics, so an image-only dataset cannot
# calibrate it regardless of label quality.
#
# Instead this reuses the same idea evaluate.py already applies to the
# anomaly-detection ensemble: inject a KNOWN defect into a REAL feature
# pipeline, so the ground truth is true by construction rather than assumed.
# "Real" curves are clean synthetic variables; "artifact" curves each carry
# exactly one deliberately injected defect, one per FEATURE_INDICATOR_NAMES
# entry. Every synthetic curve goes through the actual features.extract()
# and artifact.assess() code paths -- nothing here re-implements the
# threshold logic those functions already contain, so there is no risk of
# the calibration silently drifting from what assess() actually checks.
#
# The same caveat this codebase applies everywhere else applies here too:
# this measures sensitivity to the defect SHAPES injected, not proof that
# real instrumental defects always look like this.
# ---------------------------------------------------------------------------

SAMPLING_ARTIFACT_PERIODS_DAYS = (1.0, 0.5, 13.7)

# Fraction of the synthetic "real" class drawn from the awkward-but-genuine
# population. Without it every real object is clean and no indicator can be
# shown to be wrong.
DEFAULT_HARD_REAL_FRACTION = 0.4

# Laplace smoothing on the precision estimate, and a hard ceiling below 1.0.
#
# The ceiling is not cosmetic. Indicators combine by noisy-OR, so a weight of
# exactly 1.0 saturates the product: one indicator firing pins the likelihood
# at 100% and every other indicator, including all the clearing evidence,
# stops mattering. A calibration is allowed to say "this indicator is very
# reliable"; it is not allowed to say "this indicator alone is proof".
CALIBRATION_SMOOTHING = 2.0
MAX_CALIBRATED_WEIGHT = 0.85


def smoothed_precision(true_positives: int, support: int,
                       alpha: float = CALIBRATION_SMOOTHING) -> float:
    """Laplace-smoothed precision, capped below the noisy-OR saturation point.

    Raw precision from a handful of firings is happy to return exactly 1.0 off
    three observations. Smoothing pulls small-support estimates toward 0.5 so
    the weight reflects how much was actually seen.
    """
    if support <= 0:
        raise ValueError("smoothed_precision needs at least one observation")
    estimate = (true_positives + alpha) / (support + 2.0 * alpha)
    return min(estimate, MAX_CALIBRATED_WEIGHT)


@dataclass
class IndicatorCalibration:
    """One indicator's data-derived weight, replacing a hand-set guess."""

    name: str
    weight: float
    previous_weight: float
    support: int  # synthetic objects on which this indicator actually fired

    def to_dict(self) -> dict:
        return {"name": self.name, "weight": round(self.weight, 4),
                "previous_weight": self.previous_weight, "support": self.support}


@dataclass
class CalibrationReport:
    """Per-indicator calibrated weights plus held-out validation."""

    indicators: list[IndicatorCalibration]
    auc_old_weights: float
    auc_new_weights: float
    n_train: int
    n_test: int
    seeds: list[int]

    def to_dict(self) -> dict:
        return {
            "indicators": [i.to_dict() for i in self.indicators],
            "auc_old_weights": round(self.auc_old_weights, 4),
            "auc_new_weights": round(self.auc_new_weights, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "seeds": self.seeds,
        }


def _make_curve(object_id: str, time: np.ndarray, value: np.ndarray,
                value_err: np.ndarray):
    from .surveys.base import LightCurve, SourceRef

    return LightCurve(
        source=SourceRef(survey="SYNTH", object_id=object_id,
                         ra_deg=0.0, dec_deg=0.0),
        release="calibration", band="g", value_kind="mag",
        time=time, value=value, value_err=value_err, time_system="JD_UTC",
    )


def _baseline_signal(rng: np.random.Generator, n_points: int,
                     baseline_days: float, period_days: float,
                     amplitude_mag: float, error_mag: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A clean sinusoidal variable, quasi-randomly sampled like a real survey."""
    time = np.sort(rng.uniform(0.0, baseline_days, size=n_points))
    phase = rng.uniform(0.0, 2 * np.pi)
    signal = amplitude_mag * np.sin(2 * np.pi * time / period_days + phase)
    noise = rng.normal(0.0, error_mag, size=n_points)
    value = 18.0 + signal + noise
    value_err = np.full(n_points, error_mag)
    return time, value, value_err


def _synthetic_real(rng: np.random.Generator, index: int,
                    hard: bool = False):
    """A genuine variable: no defect, by construction label 0.

    `hard` draws an awkward but entirely real object -- sparsely sampled, close
    to its noise floor, or carrying a few genuine large excursions. Those are
    the objects on which an indicator can fire without a defect being present,
    so they are the only way a false-positive rate gets measured rather than
    assumed. Calibrating against clean variables alone is what made the first
    calibration return weights of 0.95-1.0: nothing in the population could
    ever contradict an indicator.
    """
    n_points = int(rng.integers(60, 300))
    baseline = float(rng.uniform(120.0, 500.0))
    # Kept away from every SAMPLING_ARTIFACT_PERIODS_DAYS value so a real
    # object never coincidentally trips sampling_period; that indicator's
    # false-positive rate should be measured on sparse/short baselines, not
    # manufactured by construction here.
    period = float(rng.uniform(2.0, 90.0))
    while any(abs(period - p) / p < 0.05 for p in SAMPLING_ARTIFACT_PERIODS_DAYS):
        period = float(rng.uniform(2.0, 90.0))
    error = float(rng.uniform(0.02, 0.06))
    amplitude = error * float(rng.uniform(6.0, 20.0))  # well above the noise

    if not hard:
        time, value, value_err = _baseline_signal(
            rng, n_points, baseline, period, amplitude, error)
        return _make_curve(f"real{index}", time, value, value_err)

    flavour = str(rng.choice(("sparse", "near_noise", "outliers")))
    if flavour == "sparse":
        # A real variable a survey simply did not visit often.
        n_points = int(rng.integers(12, 34))
        baseline = float(rng.uniform(300.0, 900.0))
        time, value, value_err = _baseline_signal(
            rng, n_points, baseline, period, amplitude, error)

    elif flavour == "near_noise":
        # Genuine low-amplitude variability, only just above the error bars.
        amplitude = error * float(rng.uniform(1.2, 2.5))
        time, value, value_err = _baseline_signal(
            rng, n_points, baseline, period, amplitude, error)

    else:  # outliers
        # Real flares/dips on top of real variability. Astrophysical, not
        # instrumental, but they look exactly like the extreme_outliers case.
        time, value, value_err = _baseline_signal(
            rng, n_points, baseline, period, amplitude, error)
        n_excursions = int(rng.integers(2, 6))
        where = rng.choice(n_points, size=min(n_excursions, n_points),
                           replace=False)
        value[where] -= error * rng.uniform(8.0, 18.0, size=len(where))

    return _make_curve(f"real{index}", time, value, value_err)


def _synthetic_artifact(rng: np.random.Generator, kind: str, index: int):
    """One curve carrying exactly one deliberately injected defect."""
    n_points = int(rng.integers(60, 300))
    baseline = float(rng.uniform(120.0, 500.0))
    error = float(rng.uniform(0.02, 0.06))

    if kind == "sampling_period":
        period = float(rng.choice(SAMPLING_ARTIFACT_PERIODS_DAYS))
        amplitude = error * float(rng.uniform(6.0, 20.0))
        time, value, value_err = _baseline_signal(
            rng, n_points, baseline, period, amplitude, error)

    elif kind == "low_significance":
        # robust_amplitude is the P5-P95 spread of the OBSERVED values, and
        # for pure Gaussian noise with sigma == the reported error that
        # spread is already ~3.3x the error (the P5-P95 range of a normal
        # distribution is ~3.29 sigma) -- ABOVE assess()'s 3x threshold on
        # its own, before any real signal is added. This indicator can only
        # fire when the reported per-point error OVERESTIMATES the true
        # scatter, which is a real, if less common, instrumental scenario
        # (a conservative/inflated uncertainty pipeline) rather than a
        # contrived one. No periodic signal is added on top: any real
        # signal would only push the ratio further above the threshold.
        time = np.sort(rng.uniform(0.0, baseline, size=n_points))
        true_sigma = error * float(rng.uniform(0.25, 0.45))
        value = 18.0 + rng.normal(0.0, true_sigma, size=n_points)
        value_err = np.full(n_points, error)

    elif kind == "consistent_with_constant":
        time = np.sort(rng.uniform(0.0, baseline, size=n_points))
        value = 18.0 + rng.normal(0.0, error, size=n_points)
        value_err = np.full(n_points, error)

    elif kind == "sparse_sampling":
        n_points = int(rng.integers(12, 45))
        period = float(rng.uniform(2.0, 90.0))
        amplitude = error * float(rng.uniform(6.0, 20.0))
        time, value, value_err = _baseline_signal(
            rng, n_points, baseline, period, amplitude, error)

    elif kind == "step_change":
        # No periodic signal underneath: it adds its own variance and
        # dilutes the step relative to it (verified empirically -- with a
        # real signal added first, this generator cleared
        # change_point_score's threshold of 50 only ~78% of the time; on
        # pure noise it clears it reliably). A defect can land on any
        # target, variable or not, so this is not a less realistic ground
        # truth, just a cleaner one for isolating this one signature.
        time = np.sort(rng.uniform(0.0, baseline, size=n_points))
        value = 18.0 + rng.normal(0.0, error, size=n_points)
        value_err = np.full(n_points, error)
        split = int(n_points * rng.uniform(0.3, 0.7))
        direction = rng.choice([-1.0, 1.0])
        # change_point_score's standard error is set by the MEDIAN absolute
        # successive difference, which a single jump barely moves; a step
        # has to be large relative to that, not merely a multi-sigma outlier
        # in the ordinary sense.
        value[split:] += direction * float(rng.uniform(80.0, 200.0)) * error

    elif kind == "extreme_outliers":
        # Same reasoning as step_change: no periodic signal underneath, so
        # the spikes are not competing with real variance for the kurtosis
        # statistic's attention.
        time = np.sort(rng.uniform(0.0, baseline, size=n_points))
        value = 18.0 + rng.normal(0.0, error, size=n_points)
        value_err = np.full(n_points, error)
        # Kurtosis is diluted by sample size (more ordinary points lower the
        # excess-kurtosis statistic for the same handful of spikes), so the
        # spike count scales with n_points rather than using a fixed range.
        n_spikes = max(2, int(n_points * 0.03))
        spike_indices = rng.choice(n_points, size=n_spikes, replace=False)
        value[spike_indices] += rng.choice([-1.0, 1.0], size=n_spikes) \
            * rng.uniform(35.0, 55.0, size=n_spikes) * error

    else:
        raise ValueError(f"unknown artifact kind: {kind!r}")

    return _make_curve(f"{kind}{index}", time, value, value_err)


def _fired_indicators(curve) -> set[str]:
    """Which FEATURE_INDICATOR_NAMES actually fire for one synthetic curve.

    Runs the real features.extract() / artifact.assess() code paths, with
    the cross-survey inputs held neutral (two resolving surveys, no blend,
    no period-agreement claim) so none of the four structural indicators
    can fire and contaminate a measurement this study cannot ground-truth.
    """
    from . import features

    values = features.extract(curve).values
    result = assess(values, resolved_surveys=2, blended=[],
                    period_agrees_across_surveys=None)
    return {i.name for i in result.indicators} & set(FEATURE_INDICATOR_NAMES)


def _likelihood_from_fired(fired: set[str], weights: dict[str, float]) -> float:
    """The same noisy-OR combination assess() uses, for a fixed fired-set.

    Reimplemented rather than re-run through assess() for the "old weights"
    side of the comparison, because assess() always reads the live WEIGHTS
    dict -- there is no way to ask it "what would this have scored under a
    different weight set" without either mutating global state mid-study or
    duplicating this one multiply-and-complement step. The set of indicators
    that fired still comes from the real assess() call in _fired_indicators.
    """
    survival = 1.0
    for name in fired:
        survival *= (1.0 - weights.get(name, 0.0))
    return 1.0 - survival


def calibrate_from_injection(n_per_class: int = 150, test_fraction: float = 0.3,
                             seeds: tuple[int, ...] = (17, 29, 43, 59, 71),
                             hard_real_fraction: float = DEFAULT_HARD_REAL_FRACTION
                             ) -> CalibrationReport:
    """Calibrate the six feature-based indicator weights against synthetic
    truth-by-construction labels, and validate old vs. new weights held out.

    One dataset built per seed, weights averaged across seeds -- the same
    "do not trust one injection scheme" discipline ablation.py applies to
    the anomaly-detection ensemble.
    """
    from sklearn.metrics import roc_auc_score

    per_seed_weights: dict[str, list[float]] = {n: [] for n in FEATURE_INDICATOR_NAMES}
    per_seed_support: dict[str, list[int]] = {n: [] for n in FEATURE_INDICATOR_NAMES}
    auc_old_runs: list[float] = []
    auc_new_runs: list[float] = []
    n_train_total = n_test_total = 0

    for seed in seeds:
        rng = np.random.default_rng(seed)

        n_hard = int(round(n_per_class * float(hard_real_fraction)))
        curves, labels = [], []
        for i in range(n_per_class):
            curves.append(_synthetic_real(rng, i, hard=i < n_hard))
            labels.append(0)
        for i in range(n_per_class):
            kind = FEATURE_INDICATOR_NAMES[i % len(FEATURE_INDICATOR_NAMES)]
            curves.append(_synthetic_artifact(rng, kind, i))
            labels.append(1)

        labels = np.array(labels)
        fired = [_fired_indicators(c) for c in curves]

        n = len(curves)
        order = rng.permutation(n)
        split = int(n * (1.0 - test_fraction))
        train_idx, test_idx = order[:split], order[split:]

        new_weights = dict(WEIGHTS)
        for name in FEATURE_INDICATOR_NAMES:
            fired_train = [i for i in train_idx if name in fired[i]]
            support = len(fired_train)
            per_seed_support[name].append(support)
            if support == 0:
                # No evidence this seed's synthetic set can calibrate this
                # indicator; keep the hand-set value rather than inventing
                # one from zero observations.
                per_seed_weights[name].append(WEIGHTS[name])
                continue
            true_positives = int(sum(labels[i] for i in fired_train))
            precision = smoothed_precision(true_positives, support)
            new_weights[name] = precision
            per_seed_weights[name].append(precision)

        old_scores = np.array([_likelihood_from_fired(fired[i], WEIGHTS)
                               for i in test_idx])
        new_scores = np.array([_likelihood_from_fired(fired[i], new_weights)
                               for i in test_idx])
        test_labels = labels[test_idx]

        auc_old_runs.append(float(roc_auc_score(test_labels, old_scores)))
        auc_new_runs.append(float(roc_auc_score(test_labels, new_scores)))
        n_train_total += len(train_idx)
        n_test_total += len(test_idx)

    indicators = [
        IndicatorCalibration(
            name=name,
            weight=round(float(np.mean(per_seed_weights[name])), 4),
            previous_weight=WEIGHTS[name],
            support=int(round(float(np.mean(per_seed_support[name])))),
        )
        for name in FEATURE_INDICATOR_NAMES
    ]

    return CalibrationReport(
        indicators=indicators,
        auc_old_weights=float(np.mean(auc_old_runs)),
        auc_new_weights=float(np.mean(auc_new_runs)),
        n_train=n_train_total // len(seeds),
        n_test=n_test_total // len(seeds),
        seeds=list(seeds),
    )


def calibrate_recorded(n_per_class: int = 150, test_fraction: float = 0.3,
                       seeds: tuple[int, ...] = (17, 29, 43, 59, 71),
                       hard_real_fraction: float = DEFAULT_HARD_REAL_FRACTION,
                       root=None) -> dict:
    """Run the calibration and record it like every other study.

    This was the only study in the codebase with no provenance record, which
    made its result impossible to cite or to re-verify later. The weights it
    proposes are still not adopted automatically -- `WEIGHTS` is edited by hand
    after reading the report, because adopting a weight changes what the
    verdict bands mean.
    """
    from . import experiment

    configuration = {
        "n_per_class": n_per_class,
        "test_fraction": test_fraction,
        "seeds": list(seeds),
        "hard_real_fraction": hard_real_fraction,
        "smoothing": CALIBRATION_SMOOTHING,
        "max_weight": MAX_CALIBRATED_WEIGHT,
    }

    def work() -> dict:
        report = calibrate_from_injection(
            n_per_class=n_per_class, test_fraction=test_fraction,
            seeds=seeds, hard_real_fraction=hard_real_fraction)
        payload = report.to_dict()
        # A top-level headline so `experiment.compare` can line calibration
        # runs up without a dotted path.
        payload["auc_delta"] = round(
            report.auc_new_weights - report.auc_old_weights, 4)
        payload["adopted"] = False
        payload["caveat"] = (
            "Synthetic defect shapes only. Measures sensitivity to the "
            "injected defects, not proof that real instrumental defects look "
            "like these. Weights are proposed, not applied."
        )
        return payload

    record = experiment.run("artifact_weight_calibration", configuration,
                            work, seed=int(seeds[0]), root=root)
    return {"experiment_id": record.provenance.experiment_id, **record.results}
