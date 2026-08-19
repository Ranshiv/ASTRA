"""Controlled experiments and ablations (plan sections 13 and 20).

Section 20 specifies seven experiment groups — each survey alone, each pair,
and all three — to answer the project's primary research question: does
combining surveys improve discovery? This module runs them, plus two ablations
the plan implies but does not enumerate: which feature families carry the
signal, and whether the four-detector ensemble beats its members.

Every study measures the same way, through injection-recovery, because no
human labels exist yet. That makes the comparisons internally consistent and
also bounds what they can claim: a method that wins here is sensitive to the
anomaly shapes injected, not proven sensitive to the unknown.

An important honesty constraint on the survey study: comparing ZTF-only
against ZTF+TESS is only meaningful if the two runs concern the same objects.
Otherwise the "improvement" is just a different object population. Groups
whose surveys share too few objects are reported as not comparable rather
than scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Plan section 20, verbatim.
SURVEY_GROUPS: dict[str, tuple[str, ...]] = {
    "ztf_only": ("ztf",),
    "gaia_only": ("gaia",),
    "tess_only": ("tess",),
    "ztf_gaia": ("ztf", "gaia"),
    "ztf_tess": ("ztf", "tess"),
    "gaia_tess": ("gaia", "tess"),
    "all_three": ("ztf", "gaia", "tess"),
}

# Feature families, for leave-one-family-out ablation.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "photometric": ("mean", "weighted_mean", "median", "std", "mad",
                    "amplitude", "robust_amplitude", "skew", "kurtosis",
                    "beyond_1std", "median_err"),
    "variability": ("reduced_chi2", "stetson_j", "stetson_k", "eta"),
    "temporal": ("time_span_days", "cadence_median_days",
                 "cadence_max_gap_days", "linear_trend_per_day", "max_step",
                 "change_point_score", "bocpd_change_probability",
                 "bocpd_max_probability", "bocpd_change_index",
                 "bocpd_change_time"),
    "periodic": ("best_period_days", "best_power", "period_snr"),
    "sampling": ("n_points",),
}


@dataclass
class AblationRow:
    name: str
    roc_auc: float | None = None
    average_precision: float | None = None
    rows_scored: int = 0
    comparable: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "roc_auc": None if self.roc_auc is None else round(self.roc_auc, 4),
            "average_precision": (None if self.average_precision is None
                                  else round(self.average_precision, 4)),
            "rows_scored": self.rows_scored,
            "comparable": self.comparable,
            "note": self.note,
        }


@dataclass
class AblationStudy:
    kind: str
    rows: list[AblationRow] = field(default_factory=list)
    baseline: str | None = None

    def best(self) -> AblationRow | None:
        scored = [r for r in self.rows
                  if r.roc_auc is not None and np.isfinite(r.roc_auc)
                  and r.comparable]
        return max(scored, key=lambda r: r.roc_auc) if scored else None

    def deltas(self) -> dict[str, float]:
        """Change in ROC-AUC relative to the baseline row."""
        if self.baseline is None:
            return {}
        base = next((r for r in self.rows if r.name == self.baseline), None)
        if base is None or base.roc_auc is None:
            return {}
        return {
            r.name: round(r.roc_auc - base.roc_auc, 4)
            for r in self.rows
            if r.roc_auc is not None and r.name != self.baseline
        }

    def to_dict(self) -> dict:
        winner = self.best()
        return {
            "kind": self.kind,
            "rows": [r.to_dict() for r in self.rows],
            "best": winner.name if winner else None,
            "baseline": self.baseline,
            "deltas_vs_baseline": self.deltas(),
        }


def _score_matrix(matrix, labels: np.ndarray, name: str,
                  seed: int = 42) -> AblationRow:
    """Run the ensemble on a feature matrix and score against known labels."""
    from . import anomaly, evaluate

    if len(matrix) < 10:
        return AblationRow(name, note=f"only {len(matrix)} rows; need 10")

    result = anomaly.detect(matrix, seed=seed)
    if not result.detectors:
        return AblationRow(name, note="no detector produced a score")

    usable = matrix.finite_mask()
    scored = evaluate.score_method(name, result.consensus, labels[usable])

    return AblationRow(
        name=name,
        roc_auc=None if not np.isfinite(scored.roc_auc) else scored.roc_auc,
        average_precision=(None if not np.isfinite(scored.average_precision)
                           else scored.average_precision),
        rows_scored=int(np.count_nonzero(usable)),
        note=scored.note,
    )


def _injected_matrix(survey_names: tuple[str, ...], fraction: float,
                     strength: float, seed: int):
    """Build sequences for a survey group, inject anomalies, summarise.

    Anomalies must be injected into the *data*, not merely labelled. An earlier
    version of this module assigned labels at random to untouched rows, which
    measured nothing: the resulting ROC-AUC of 0.511 was exactly the chance
    level it deserved. Injection has to happen in sequence space and the
    features be recomputed afterwards, so the anomaly is genuinely present in
    what the detectors see.

    Gaia is handled separately from every other entry in `survey_names`.
    `fetch_light_curves` returns nothing for it by design (surveys/gaia.py),
    so it is dropped from the sequence-building loop below and instead joined
    onto the finished summary matrix as extra columns by
    `featurematrix.join_gaia_columns`. This keeps the object population
    (and therefore the injected labels) IDENTICAL to the group with "gaia"
    removed -- only the feature width changes -- which is what makes a
    ztf_only vs ztf_gaia comparison honest rather than a population
    artefact. A group of ("gaia",) alone has no sequence survey left to
    inject into or join onto, so it stays unscorable; that is a structural
    fact about Gaia being catalogue-only, not something a join can fix.
    """
    from . import evaluate, featurematrix, tensors
    from .featurematrix import FeatureMatrix

    sequence_surveys = tuple(s for s in survey_names if s != "gaia")
    join_gaia = "gaia" in survey_names

    values: list[np.ndarray] = []
    identities: list[dict] = []
    length = tensors.DEFAULT_LENGTH

    for survey in sequence_surveys:
        batch = tensors.build(survey=survey)
        if len(batch):
            values.append(batch.values)
            identities.extend(batch.identities)
            length = batch.length

    if not values:
        return None, np.empty(0, dtype=int), {}

    stacked = np.vstack(values)
    injection = evaluate.build_injected(stacked, identities, fraction=fraction,
                                        strength=strength, seed=seed)
    summary = evaluate.sequence_summary(injection.values)

    matrix = FeatureMatrix(
        values=summary, identities=injection.identities,
        feature_names=tuple(f"seq_{i}" for i in range(summary.shape[1])),
    )

    gaia_diagnostics: dict = {}
    if join_gaia:
        matrix, gaia_diagnostics = featurematrix.join_gaia_columns(matrix)

    return matrix, injection.labels, gaia_diagnostics


def feature_ablation(fraction: float = 0.1, seed: int = 42,
                     strength: float = 6.0,
                     survey: str | None = None) -> AblationStudy:
    """Leave one feature family out and measure what it was worth.

    Injection happens in sequence space and features are recomputed, so the
    injected anomaly is visible to whichever families can see it — which is
    the point of the ablation.

    `survey` restricts the population to one survey. Without it this mixes
    ZTF and TESS sequences, which are structurally different — hundreds of
    magnitudes over years against tens of thousands of flux points at
    two-minute cadence — so the detectors partly separate by SURVEY rather
    than by behaviour. That is the survey bias plan section 36 warns about,
    reproduced inside the ablation meant to measure something else.
    """
    from . import evaluate, featurematrix, features as features_mod, tensors
    from .featurematrix import FeatureMatrix

    batch = tensors.build(survey=survey)
    if len(batch) < 20:
        return AblationStudy("feature_groups",
                             [AblationRow("all", note="not enough sequences")])

    injection = evaluate.build_injected(batch.values, batch.identities,
                                        fraction=fraction, strength=strength,
                                        seed=seed)
    summary = evaluate.sequence_summary(injection.values)
    names = tuple(f"seq_{i}" for i in range(summary.shape[1]))

    study = AblationStudy("feature_groups", baseline="all_features")
    study.rows.append(_score_matrix(
        FeatureMatrix(values=summary, identities=injection.identities,
                      feature_names=names),
        injection.labels, "all_features", seed))

    # Sequence-summary columns map onto families by position; drop each in turn.
    families = {
        "scatter": (0, 1, 2),
        "differences": (3, 4),
        "shape": (5, 6),
        "extremes": (7, 8),
        "coverage": (9,),
    }
    for family, columns in families.items():
        keep = [i for i in range(summary.shape[1]) if i not in columns]
        study.rows.append(_score_matrix(
            FeatureMatrix(values=summary[:, keep],
                          identities=injection.identities,
                          feature_names=tuple(names[i] for i in keep)),
            injection.labels, f"without_{family}", seed))

    return study


def detector_ablation(fraction: float = 0.1, seed: int = 42,
                      survey: str | None = None) -> AblationStudy:
    """Does the ensemble beat its individual members?

    Plan section 16 allocates weight to model agreement on the assumption that
    it does. This checks the assumption instead of trusting it.

    `survey` restricts the population to one survey, for the reason given in
    `feature_ablation`. Leaving it unset was measurably consequential: the
    same comparison scores the ensemble at about 0.79 on ZTF alone and about
    0.63 on a mixed ZTF+TESS population, and the difference is the mixture,
    not the detectors.
    """
    from . import anomaly, evaluate, tensors
    from .featurematrix import FeatureMatrix

    batch = tensors.build(survey=survey)
    if len(batch) < 20:
        return AblationStudy("detectors",
                             [AblationRow("ensemble", note="not enough data")])

    injection = evaluate.build_injected(batch.values, batch.identities,
                                        fraction=fraction, seed=seed)
    summary = evaluate.sequence_summary(injection.values)
    matrix = FeatureMatrix(
        values=summary, identities=injection.identities,
        feature_names=tuple(f"seq_{i}" for i in range(summary.shape[1])))

    result = anomaly.detect(matrix, seed=seed)
    usable = matrix.finite_mask()
    labels = injection.labels[usable]

    study = AblationStudy("detectors", baseline="ensemble")
    if not result.detectors:
        return AblationStudy("detectors",
                             [AblationRow("ensemble", note="no detectors ran")])

    scored = evaluate.score_method("ensemble", result.consensus, labels)
    study.rows.append(AblationRow(
        "ensemble", scored.roc_auc, scored.average_precision,
        int(np.count_nonzero(usable))))

    for name, detector in result.detectors.items():
        one = evaluate.score_method(name, detector.scores, labels)
        study.rows.append(AblationRow(
            name, one.roc_auc, one.average_precision,
            int(np.count_nonzero(usable))))

    return study


def _gaia_catalog_count(projects_root=None) -> int:
    """Stored Gaia catalogue rows -- the join source, not a sequence count.

    Gaia never contributes rows to a sequence-based study (its main table
    has no time series), so its "availability" for a group has to be judged
    by catalogue rows persisted from acquisition, not by
    `featurematrix.build(survey="gaia")`, which is always empty by design.
    """
    from . import config, metadata

    root = projects_root or config.PATHS.projects
    return sum(1 for row in metadata.list_sources(root)
              if row["survey"].upper() == "GAIA")


def survey_ablation(fraction: float = 0.1, seed: int = 42,
                    min_shared_fraction: float = 0.5,
                    strength: float = 6.0) -> AblationStudy:
    """Run the seven section 20 groups.

    A group is only scored when its surveys actually overlap on objects. With
    a ZTF-heavy archive and one blended TESS target, most groups do not, and
    saying so is more useful than producing numbers that compare different
    populations and calling the difference an improvement.

    Gaia is judged and joined differently from the other two surveys
    throughout this function -- see `_injected_matrix` and
    `featurematrix.join_gaia_columns` for why. In short: Gaia contributes
    columns to an existing sequence survey's rows, never rows of its own, so
    it is excluded from the row-count comparability check and instead gated
    on whether any catalogue rows exist to join at all.
    """
    from . import featurematrix

    study = AblationStudy("survey_groups", baseline="ztf_only")

    available: dict[str, int] = {}
    for survey in ("ztf", "tess"):
        available[survey] = len(featurematrix.build(survey=survey))
    available["gaia"] = _gaia_catalog_count()

    for group_name, group_surveys in SURVEY_GROUPS.items():
        sequence_surveys = [s for s in group_surveys if s != "gaia"]
        wants_gaia = "gaia" in group_surveys

        if wants_gaia and available["gaia"] == 0:
            study.rows.append(AblationRow(
                group_name, comparable=False,
                note="no stored Gaia catalogue data to join"))
            continue

        if not sequence_surveys:
            # Gaia alone: no sequence exists to inject an anomaly into or to
            # join Gaia's columns onto. This is a structural fact about a
            # catalogue-only connector, not a data-availability gap that
            # acquiring more Gaia rows could ever fix.
            study.rows.append(AblationRow(
                group_name, comparable=False,
                note="Gaia has no light curves of its own (catalogue "
                     "connector); it can only join as columns onto another "
                     "survey's sequences, not stand alone"))
            continue

        counts = [available.get(s, 0) for s in sequence_surveys]
        if any(c == 0 for c in counts):
            missing = [s for s, c in zip(sequence_surveys, counts) if c == 0]
            study.rows.append(AblationRow(
                group_name, comparable=False,
                note=f"no stored data for {', '.join(missing)}"))
            continue

        if len(sequence_surveys) > 1:
            smallest, largest = min(counts), max(counts)
            if smallest / largest < min_shared_fraction:
                study.rows.append(AblationRow(
                    group_name, comparable=False, rows_scored=sum(counts),
                    note=(f"survey sizes differ too much to compare "
                          f"({dict(zip(sequence_surveys, counts))}); the "
                          f"groups would describe different object "
                          f"populations")))
                continue

        matrix, labels, gaia_diagnostics = _injected_matrix(
            group_surveys, fraction, strength, seed)
        if matrix is None or len(matrix) == 0:
            study.rows.append(AblationRow(
                group_name, comparable=False,
                note="no usable sequences after resampling"))
            continue

        row = _score_matrix(matrix, labels, group_name, seed)
        if wants_gaia and gaia_diagnostics:
            rate = gaia_diagnostics.get("match_rate")
            rate_note = (f"gaia join matched {gaia_diagnostics['matched']}/"
                        f"{gaia_diagnostics['total']} objects "
                        f"({rate:.0%})" if rate is not None else
                        "gaia join matched 0 objects")
            row.note = f"{row.note}; {rate_note}" if row.note else rate_note
        study.rows.append(row)

    return study


def _combined_matrix(survey_names: tuple[str, ...]):
    """Stack feature matrices from several surveys into one."""
    from . import featurematrix
    from .featurematrix import FeatureMatrix
    from .features import FEATURE_NAMES

    values: list[np.ndarray] = []
    identities: list[dict] = []
    for survey in survey_names:
        matrix = featurematrix.build(survey=survey)
        if len(matrix):
            values.append(matrix.values)
            identities.extend(matrix.identities)

    stacked = (np.vstack(values) if values
               else np.empty((0, len(FEATURE_NAMES))))
    return FeatureMatrix(values=stacked, identities=identities)


def run_all(fraction: float = 0.1, seed: int = 42,
            survey: str | None = None, root=None) -> dict:
    """Every study, recorded as one experiment.

    `survey` stratifies the feature and detector ablations. It is recorded in
    the experiment configuration either way, so a stratified run and an older
    unstratified one stay distinguishable rather than being silently compared.
    The survey-group study is never stratified: comparing survey groups is
    what it exists to do.
    """
    from . import experiment

    def work() -> dict:
        return {
            "survey_groups": survey_ablation(fraction, seed).to_dict(),
            "feature_groups": feature_ablation(fraction, seed,
                                               survey=survey).to_dict(),
            "detectors": detector_ablation(fraction, seed,
                                           survey=survey).to_dict(),
        }

    record = experiment.run("ablation_suite",
                            {"fraction": fraction, "seed": seed,
                             "survey": survey},
                            work, seed=seed,
                            notes="Plan section 20 experiment groups plus "
                                  "feature and detector ablations."
                                  + ("" if survey is None else
                                     f" Feature and detector ablations "
                                     f"stratified to {survey}."),
                            root=root)
    return {"experiment_id": record.provenance.experiment_id,
            **record.results}


def aggregate_repeated(studies: list[AblationStudy], metric: str = "roc_auc") -> dict:
    """Aggregate independent injection seeds without averaging missing runs.

    A method may be unscorable for a seed (for example, after a survey-group
    comparability guard fires).  That is reported as an absent run, never as a
    zero AUC.  The interval is an empirical 95% seed interval; it quantifies
    injection-seed sensitivity rather than pretending to be a population CI.
    """
    grouped: dict[str, list[AblationRow]] = {}
    for study in studies:
        for row in study.rows:
            grouped.setdefault(row.name, []).append(row)

    rows = []
    for name, entries in sorted(grouped.items()):
        values = [getattr(row, metric) for row in entries]
        finite = np.asarray([value for value in values
                             if value is not None and np.isfinite(value)], dtype=float)
        ap = np.asarray([row.average_precision for row in entries
                         if row.average_precision is not None
                         and np.isfinite(row.average_precision)], dtype=float)
        rows.append({
            "name": name,
            "runs": len(entries),
            "comparable_runs": sum(row.comparable for row in entries),
            "scored_runs": int(len(finite)),
            metric: (None if not len(finite) else {
                "mean": round(float(np.mean(finite)), 4),
                "std": round(float(np.std(finite, ddof=1)), 4) if len(finite) > 1 else 0.0,
                "ci95": [round(float(np.quantile(finite, 0.025)), 4),
                         round(float(np.quantile(finite, 0.975)), 4)],
            }),
            "average_precision": (None if not len(ap) else {
                "mean": round(float(np.mean(ap)), 4),
                "ci95": [round(float(np.quantile(ap, 0.025)), 4),
                         round(float(np.quantile(ap, 0.975)), 4)],
            }),
            "unscored_notes": sorted({row.note for row in entries
                                      if getattr(row, metric) is None and row.note}),
        })
    return {"metric": metric, "rows": rows,
            "interval": "empirical 2.5th–97.5th percentile across independent seeds"}


def run_repeated(fraction: float = 0.1,
                 seeds: tuple[int, ...] = (17, 29, 43, 59, 71),
                 survey: str | None = None, root=None) -> dict:
    """Run the full suite across independent injection seeds and persist it.

    `survey` stratifies the feature and detector ablations; see `run_all`.
    """
    from . import experiment

    if len(seeds) < 2:
        raise ValueError("repeated ablation needs at least two distinct seeds")
    unique = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if len(unique) < 2:
        raise ValueError("repeated ablation needs at least two distinct seeds")

    def work() -> dict:
        groups = [survey_ablation(fraction, seed) for seed in unique]
        feature = [feature_ablation(fraction, seed, survey=survey)
                   for seed in unique]
        detector = [detector_ablation(fraction, seed, survey=survey)
                    for seed in unique]
        return {
            "seeds": list(unique),
            "survey": survey,
            "survey_groups": aggregate_repeated(groups),
            "feature_groups": aggregate_repeated(feature),
            "detectors": aggregate_repeated(detector),
        }

    record = experiment.run(
        "ablation_suite_repeated",
        {"fraction": fraction, "seeds": list(unique), "survey": survey,
         "interval": "empirical seed percentile"},
        work, seed=unique[0],
        notes="Repeated-seed injection-recovery with uncertainty intervals; "
              "only comparable survey groups are scored."
              + ("" if survey is None else
                 f" Feature and detector ablations stratified to {survey}."),
        root=root)
    return {"experiment_id": record.provenance.experiment_id,
            **record.results}
