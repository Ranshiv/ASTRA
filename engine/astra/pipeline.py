"""End-to-end candidate generation (plan section 3).

Ties the stages together: features -> baseline detection -> cross-survey
matching -> composite scoring -> artifact assessment -> ranked, explained
candidates.

Detection runs per survey rather than over a mixed matrix. That is not a
detail: a matrix combining ZTF and TESS makes the detectors separate by
instrument — 18,000 points at 2-minute cadence in flux against 500 points
over years in magnitudes — rather than by behaviour, which is the survey bias
plan section 36 warns about and which was measured directly in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import (anomaly, candidates as candidates_mod, config, crossmatch, evidence,
               featurematrix, features as features_mod, metadata, store, surveys)
from .surveys.base import SourceRef


@dataclass
class PipelineReport:
    surveys_processed: list[str] = field(default_factory=list)
    rows_by_survey: dict[str, int] = field(default_factory=dict)
    rows_by_stratum: dict[str, int] = field(default_factory=dict)
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    candidates_built: int = 0
    likely_artifacts: int = 0
    cross_survey_groups: int = 0
    resolved_multi_survey: int = 0
    output_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "surveys_processed": self.surveys_processed,
            "rows_by_survey": self.rows_by_survey,
            "rows_by_stratum": self.rows_by_stratum,
            "skipped_reasons": self.skipped_reasons,
            "candidates_built": self.candidates_built,
            "likely_artifacts": self.likely_artifacts,
            "cross_survey_groups": self.cross_survey_groups,
            "resolved_multi_survey": self.resolved_multi_survey,
            "output_path": self.output_path,
        }


def _resolution_index(radius_arcsec: float,
                      curve_index: dict[tuple[str, str], list] | None = None,
                      features_by_key: dict | None = None
                      ) -> dict[tuple[str, str], dict]:
    """Map each (survey, object_id) to its cross-survey standing.

    `curve_index` and `features_by_key` are accepted so `run` can share the
    single store walk and the single set of feature matrices it already needs,
    rather than each stage independently rebuilding them.
    """
    if curve_index is None:
        curve_index = evidence.load_curves_by_key()

    by_survey: dict[str, list] = {}
    for (survey, _oid), curves in curve_index.items():
        by_survey.setdefault(survey, []).append(curves[0].source)
    for row in metadata.list_sources(config.PATHS.projects):
        sources = by_survey.setdefault(row["survey"], [])
        if not any(source.object_id == row["object_id"] for source in sources):
            sources.append(SourceRef(survey=row["survey"], object_id=row["object_id"],
                                     ra_deg=row["ra_deg"], dec_deg=row["dec_deg"],
                                     extra=row["extra"]))

    groups = crossmatch.group_sources(by_survey, radius_arcsec=radius_arcsec)

    # Reuse cached features rather than re-running a period search per group.
    if features_by_key is None:
        features_by_key = evidence.feature_lookup_from_matrices()

    lookup: dict[tuple[str, str], dict] = {}
    for group in groups:
        profile = evidence.profile_group(group, curve_index, features_by_key)
        periods = [v.best_period_days for v in profile.views
                   if np.isfinite(v.best_period_days)]
        agrees: bool | None = None
        if len({v.survey for v in profile.views}) > 1 and len(periods) > 1:
            agrees = profile.components.get("period_agreement", 0.0) > 0.5

        for survey, source in group.members.items():
            lookup[(survey, source.object_id)] = {
                "resolved_surveys": group.resolved_surveys,
                "blended": sorted(group.blended),
                "consistency": profile.consistency,
                "period_agrees": agrees,
                "gaia": _gaia_properties(group),
            }

    return lookup, len(groups), sum(1 for g in groups if g.resolved_surveys > 1)


def _gaia_properties(group: crossmatch.MatchGroup) -> dict | None:
    """Derived astrometry, when a Gaia counterpart is present."""
    gaia_source = group.members.get("Gaia")
    if gaia_source is None:
        return None

    from .surveys.gaia import derived_properties

    return {**gaia_source.extra, **derived_properties(gaia_source.extra)}


def run(survey_names: list[str] | None = None,
        radius_arcsec: float = 15.0,
        contamination: float = 0.05,
        top: int = 200,
        name: str = "default",
        seed: int = 42,
        root: Path | None = None) -> tuple[list[candidates_mod.Candidate],
                                           PipelineReport]:
    """Produce a ranked, explained candidate list from the stored data."""
    report = PipelineReport()
    survey_names = survey_names or surveys.available()

    # One walk of the store, and one feature matrix per survey, shared by every
    # stage below. Each of these used to be rebuilt independently by the
    # resolution index, the feature lookup and the per-survey loop.
    curve_index = evidence.load_curve_index()
    matrices = {survey: featurematrix.build(survey=survey)
                for survey in survey_names}
    features_by_key = evidence.feature_lookup_from_matrices(
        survey_names, matrices=matrices)

    lookup, group_count, resolved_multi = _resolution_index(
        radius_arcsec, curve_index.by_key, features_by_key)
    report.cross_survey_groups = group_count
    report.resolved_multi_survey = resolved_multi

    positions = curve_index.positions_by_path
    built: list[candidates_mod.Candidate] = []
    index = 1

    for survey in survey_names:
        matrix = matrices[survey]
        if len(matrix) == 0:
            continue

        report.surveys_processed.append(survey.upper())
        report.rows_by_survey[survey.upper()] = len(matrix)

        # Detection is mandatory within (survey, band, coverage tier).  Tier B
        # omits the period columns it cannot support; Tier C is retained for
        # review with an explicit reason and is never silently imputed.
        scores: dict[str, dict] = {}
        for band in sorted({i.get("band", "unknown") for i in matrix.identities}):
            for tier in ("A", "B", "C"):
                rows = [i for i, identity in enumerate(matrix.identities)
                        if identity.get("band") == band
                        and identity.get("coverage_tier", "A") == tier]
                if not rows:
                    continue
                key = f"{survey.upper()}/{band}/{tier}"
                report.rows_by_stratum[key] = len(rows)
                if tier == "C":
                    report.skipped_reasons["insufficient_data_lt_10_points"] = \
                        report.skipped_reasons.get("insufficient_data_lt_10_points", 0) + len(rows)
                    continue
                names = matrix.feature_names
                if tier == "B":
                    names = tuple(n for n in names
                                  if n not in {"best_period_days", "best_power", "period_snr"})
                stratum = matrix.subset(rows, names)
                result = anomaly.detect(stratum, contamination=contamination, seed=seed)
                if result.skipped_rows:
                    reason = f"non_finite_features_tier_{tier.lower()}"
                    report.skipped_reasons[reason] = \
                        report.skipped_reasons.get(reason, 0) + result.skipped_rows
                scores.update({entry["path"]: entry
                               for entry in result.ranked(top=len(stratum))})

        for row, identity in enumerate(matrix.identities):
            feature_values = {
                name: matrix.values[row, column]
                for column, name in enumerate(matrix.feature_names)
            }
            ranked = scores.get(identity["path"], {})
            standing = lookup.get((identity["survey"], identity["object_id"]), {})

            # Already known from the single store walk above; only a curve
            # written since then needs its own read.
            source_position = positions.get(identity["path"]) \
                or _position_for(identity["path"])
            built.append(candidates_mod.build_candidate(
                index,
                {**identity, **source_position},
                feature_values,
                anomaly_score=ranked.get("consensus_score"),
                model_agreement=ranked.get("model_agreement"),
                consistency=standing.get("consistency"),
                resolved_surveys=standing.get("resolved_surveys", 1),
                blended=standing.get("blended"),
                period_agrees=standing.get("period_agrees"),
                gaia_properties=standing.get("gaia"),
            ))
            tier = identity.get("coverage_tier", "A")
            built[-1].explanation["coverage"] = {
                "tier": tier,
                "periodic_features": tier == "A",
                "status": ("full" if tier == "A" else
                           "non_periodic_only" if tier == "B" else
                           "insufficient_data_lt_10_points"),
            }
            index += 1

    ranked_candidates = candidates_mod.rank(built)[:top]
    report.candidates_built = len(ranked_candidates)
    report.likely_artifacts = sum(
        1 for c in ranked_candidates
        if c.artifact.get("likelihood", 0.0) >= 0.6)

    report.output_path = str(candidates_mod.save(ranked_candidates, name, root))
    return ranked_candidates, report


def _position_for(path: str) -> dict:
    """Recover sky coordinates from the stored curve's metadata.

    Fallback only. `run` resolves positions from the shared curve index; this
    covers a path that appeared after that index was built.
    """
    try:
        curve = store.read_curve(Path(path))
        return {"ra_deg": curve.source.ra_deg, "dec_deg": curve.source.dec_deg}
    except Exception:  # noqa: BLE001 - position is nice to have, not essential
        return {"ra_deg": float("nan"), "dec_deg": float("nan")}
