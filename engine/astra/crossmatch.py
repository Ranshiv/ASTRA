"""Positional matching across surveys (plan section 15).

Matching by coordinates sounds trivial and is not, for two reasons.

First, stars move. Gaia positions are given at epoch J2016.0, ZTF observes
from 2018 onward, and TESS elsewhere again. A star with 100 mas/yr of proper
motion drifts a full arcsecond in ten years — larger than a tight match radius
— so a high-motion star silently fails to match itself unless its position is
propagated to a common epoch first. High-proper-motion stars are exactly the
nearby objects a survey of unusual behaviour cares about, so losing them is
the worst possible failure mode.

Second, a match is not proof. In a crowded field several sources fall inside
any reasonable radius, so this module reports the separation and the number of
competing candidates and lets the evidence scoring decide, rather than
silently returning the nearest one as if it were certain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .surveys.base import SourceRef

# Default tolerance. ZTF's pixel scale is about 1 arcsec, so a couple of
# arcseconds accommodates centroid error without sweeping in the whole field.
DEFAULT_RADIUS_ARCSEC = 2.0

# Reference epoch for Gaia positions, as a Julian year. This one is a genuine
# catalogue constant -- Gaia DR3 astrometry is published at J2016.0 -- and
# must not track the current date.
GAIA_EPOCH = 2016.0

MAS_PER_YEAR_TO_DEG = 1.0 / (3600.0 * 1000.0)


def current_epoch() -> float:
    """Now, as a fractional Julian year.

    Positions with proper motion (chiefly Gaia's) must be propagated to
    *today*, not to a fixed year written into the code -- a hardcoded target
    epoch goes stale the moment the calendar turns, silently drifting every
    cross-survey match by another year of proper motion. Calling this fresh
    at match time keeps the target epoch moving with the clock instead.
    """
    now = datetime.now(timezone.utc)
    start_of_year = datetime(now.year, 1, 1, tzinfo=timezone.utc)
    start_of_next_year = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
    fraction = ((now - start_of_year).total_seconds()
               / (start_of_next_year - start_of_year).total_seconds())
    return now.year + fraction


@dataclass(frozen=True)
class Match:
    """One cross-survey association, with the evidence for it."""

    source: SourceRef
    counterpart: SourceRef
    separation_arcsec: float
    competitors: int          # other sources within the radius
    proper_motion_applied: bool

    def to_dict(self) -> dict:
        return {
            "survey": self.source.survey,
            "object_id": self.source.object_id,
            "counterpart_survey": self.counterpart.survey,
            "counterpart_id": self.counterpart.object_id,
            "separation_arcsec": round(self.separation_arcsec, 4),
            "competitors": self.competitors,
            "proper_motion_applied": self.proper_motion_applied,
        }


# Angular resolution, in arcseconds. TESS pixels are 21 arcsec, so a single
# TESS "source" is a blend of everything in a wide neighbourhood; treating one
# as confirmation of a specific ZTF object would be wrong.
PIXEL_SCALE_ARCSEC = {"ZTF": 1.0, "GAIA": 0.1, "TESS": 21.0}

# A survey whose beam is at least this wide cannot isolate a single star from
# its neighbours; its photometry is a sum over the neighbourhood.
COARSE_BEAM_ARCSEC = 5.0


@dataclass
class MatchGroup:
    """One physical object as seen by several surveys."""

    members: dict[str, SourceRef] = field(default_factory=dict)
    separations: dict[str, float] = field(default_factory=dict)
    ambiguous: set[str] = field(default_factory=set)
    # Surveys whose counterpart is shared with other groups, i.e. that survey
    # cannot resolve this object from its neighbours.
    blended: set[str] = field(default_factory=set)
    match_radius_arcsec: float = DEFAULT_RADIUS_ARCSEC

    @property
    def surveys(self) -> list[str]:
        return sorted(self.members)

    @property
    def independent_surveys(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict:
        return {
            "surveys": self.surveys,
            "independent_surveys": self.independent_surveys,
            "members": {s: r.object_id for s, r in self.members.items()},
            "separations_arcsec": {s: round(v, 4)
                                   for s, v in self.separations.items()},
            "ambiguous": sorted(self.ambiguous),
            "blended": sorted(self.blended),
            "match_radius_arcsec": self.match_radius_arcsec,
            "resolved_surveys": self.resolved_surveys,
        }

    @property
    def resolved_surveys(self) -> int:
        """Surveys that actually resolve this object, ignoring blends.

        This, not `independent_surveys`, is the honest count of independent
        evidence: a blended detection corroborates the neighbourhood, not the
        object.
        """
        return len([s for s in self.members if s not in self.blended])


def propagate_position(ra_deg: float, dec_deg: float,
                       pm_ra_mas_yr: float | None, pm_dec_mas_yr: float | None,
                       from_epoch: float, to_epoch: float) -> tuple[float, float]:
    """Move a position by its proper motion between two epochs.

    pm_ra is the sky-projected motion (pmRA*), so the cos(dec) factor is
    already included and must be divided out to get the change in RA itself.
    Without that division, motion near the pole would be badly wrong.
    """
    if pm_ra_mas_yr is None and pm_dec_mas_yr is None:
        return ra_deg, dec_deg

    years = to_epoch - from_epoch
    pm_ra = (pm_ra_mas_yr or 0.0) * MAS_PER_YEAR_TO_DEG * years
    pm_dec = (pm_dec_mas_yr or 0.0) * MAS_PER_YEAR_TO_DEG * years

    cos_dec = np.cos(np.radians(dec_deg))
    if abs(cos_dec) < 1e-8:
        return ra_deg, dec_deg + pm_dec  # at the pole RA is degenerate

    return ra_deg + pm_ra / cos_dec, dec_deg + pm_dec


def epoch_corrected(source: SourceRef, to_epoch: float | None = None
                    ) -> tuple[float, float, bool]:
    """Position at a common epoch, using Gaia proper motion when present."""
    if to_epoch is None:
        to_epoch = current_epoch()
    pm_ra = source.extra.get("pmra")
    pm_dec = source.extra.get("pmdec")
    if pm_ra is None and pm_dec is None:
        return source.ra_deg, source.dec_deg, False

    from_epoch = GAIA_EPOCH if source.survey.upper() == "GAIA" else to_epoch
    ra, dec = propagate_position(source.ra_deg, source.dec_deg,
                                 pm_ra, pm_dec, from_epoch, to_epoch)
    return ra, dec, from_epoch != to_epoch


def angular_separation_arcsec(ra1: float, dec1: float,
                              ra2: float, dec2: float) -> float:
    """Great-circle separation via the haversine form.

    Haversine rather than the plain spherical cosine rule, which loses
    precision catastrophically at the small separations that matter here.
    """
    phi1, phi2 = np.radians(dec1), np.radians(dec2)
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(ra2 - ra1)

    a = (np.sin(delta_phi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
    return float(np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))) * 3600.0)


def _angular_separation_grid_arcsec(ra1: np.ndarray, dec1: np.ndarray,
                                    ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    """Vectorized haversine separation between every (source, counterpart)
    pair at once, shape (len(ra1), len(ra2)).

    `angular_separation_arcsec` above is correct but scalar; calling it once
    per pair in a Python loop makes `match_catalogs` scale as
    len(sources) * len(counterparts) individual numpy dispatches. Against the
    project-wide Gaia metadata store (thousands of entries, not just the
    handful near one query) that turns a single UI-facing request into
    millions of scalar calls -- multiple minutes of wall time. This computes
    the identical formula over the full pairwise grid in one batch instead.
    """
    phi1 = np.radians(dec1)[:, None]
    phi2 = np.radians(dec2)[None, :]
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(ra2)[None, :] - np.radians(ra1)[:, None]

    a = (np.sin(delta_phi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))) * 3600.0


def match_catalogs(sources: list[SourceRef], counterparts: list[SourceRef],
                   radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
                   epoch: float | None = None) -> list[Match]:
    """Match each source to its nearest counterpart within the radius."""
    if not sources or not counterparts:
        return []
    if epoch is None:
        epoch = current_epoch()

    corrected = [epoch_corrected(c, epoch) for c in counterparts]
    src_corrected = [epoch_corrected(s, epoch) for s in sources]

    separations = _angular_separation_grid_arcsec(
        np.array([ra for ra, _, _ in src_corrected]),
        np.array([dec for _, dec, _ in src_corrected]),
        np.array([ra for ra, _, _ in corrected]),
        np.array([dec for _, dec, _ in corrected]),
    )

    matches: list[Match] = []
    for i, source in enumerate(sources):
        row = separations[i]
        within = np.nonzero(row <= radius_arcsec)[0]
        if within.size == 0:
            continue

        best = int(within[np.argmin(row[within])])
        _, _, moved_source = src_corrected[i]
        matches.append(Match(
            source=source,
            counterpart=counterparts[best],
            separation_arcsec=float(row[best]),
            competitors=int(within.size) - 1,
            proper_motion_applied=moved_source or corrected[best][2],
        ))

    return matches


def _resolve_anchor(by_survey: dict[str, list[SourceRef]],
                    anchor_survey: str | None = None) -> tuple[str | None, str]:
    """Choose a deterministic grouping anchor and record how it was chosen."""
    if not by_survey:
        return None, "empty"
    if anchor_survey is not None and str(anchor_survey).strip():
        requested = str(anchor_survey).strip()
        matching = next((name for name in by_survey if name.casefold() == requested.casefold()), None)
        if matching is None:
            available = ", ".join(sorted(str(name) for name in by_survey)) or "none"
            raise ValueError(f"anchor survey {requested!r} is not available; choose one of: {available}")
        if not by_survey[matching]:
            raise ValueError(f"anchor survey {matching!r} has no sources to anchor grouping")
        return matching, "explicit"
    # A lexical tie-break makes the default reproducible even when callers
    # construct the input mapping in a different insertion order.
    anchor = min(by_survey, key=lambda name: (-len(by_survey[name]), str(name).casefold(), str(name)))
    return anchor, "largest_catalogue"


def group_sources(by_survey: dict[str, list[SourceRef]],
                  radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
                  epoch: float | None = None,
                  anchor_survey: str | None = None) -> list[MatchGroup]:
    """Cluster sources from several surveys into per-object groups.

    By default the survey with the most detections anchors the grouping, so
    the largest catalogue defines the object list and the others attach to it.
    ``anchor_survey`` makes that denominator explicit and reproducible for a
    science run. Groups are returned even when only one survey contributes,
    because "seen by only one instrument" is itself information for the
    artifact assessment.
    """
    if not by_survey:
        return []
    if epoch is None:
        epoch = current_epoch()

    anchor_survey, _ = _resolve_anchor(by_survey, anchor_survey)
    if anchor_survey is None:
        return []
    anchor_sources = by_survey[anchor_survey]

    groups = []
    for source in anchor_sources:
        group = MatchGroup(members={anchor_survey: source},
                           separations={anchor_survey: 0.0},
                           match_radius_arcsec=radius_arcsec)

        for survey, candidates in by_survey.items():
            if survey == anchor_survey:
                continue
            found = match_catalogs([source], candidates, radius_arcsec, epoch)
            if not found:
                continue
            match = found[0]
            group.members[survey] = match.counterpart
            group.separations[survey] = match.separation_arcsec
            if match.competitors > 0:
                group.ambiguous.add(survey)

        groups.append(group)

    _flag_blends(groups, anchor_survey)
    return groups


def _flag_blends(groups: list[MatchGroup], anchor_survey: str) -> None:
    """Mark counterparts shared by several groups as blended.

    A survey matching many distinct anchor objects to one of its own sources
    is telling you its beam is wider than the separations involved. TESS's
    21 arcsec pixels make this routine, and unflagged it would manufacture
    cross-survey "confirmation" for every star near a bright variable.
    """
    usage: dict[tuple[str, str], int] = {}
    for group in groups:
        for survey, source in group.members.items():
            # A coarse-beam survey is blended even when an explicit anchor
            # policy makes it the anchor.  Otherwise reversing the grouping
            # direction would turn the same TESS neighbourhood into resolved
            # evidence merely because TESS had fewer rows in one run.
            if (survey.upper() in PIXEL_SCALE_ARCSEC
                    and PIXEL_SCALE_ARCSEC.get(survey.upper(), 1.0) >= COARSE_BEAM_ARCSEC
                    and len(group.members) > 1):
                group.blended.add(survey)

            if survey == anchor_survey:
                continue
            key = (survey, source.object_id)
            usage[key] = usage.get(key, 0) + 1

    for group in groups:
        for survey, source in group.members.items():
            if survey == anchor_survey:
                continue

            # Definitive: the same counterpart serves several distinct objects.
            if usage.get((survey, source.object_id), 0) > 1:
                group.blended.add(survey)

                continue

            # Structural: a survey with a beam this coarse cannot isolate one
            # star from its neighbours at all without PSF fitting on pixel
            # data, which ASTRA deliberately does not download. Its aperture
            # photometry is a sum over the neighbourhood, so treating it as a
            # measurement of this specific object is unsound even when the
            # match happens to be unique in our catalogue.
            if PIXEL_SCALE_ARCSEC.get(survey.upper(), 1.0) >= COARSE_BEAM_ARCSEC:
                group.blended.add(survey)


def grouping_bias_report(by_survey: dict[str, list[SourceRef]],
                         groups: list[MatchGroup] | None = None,
                         anchor_survey: str | None = None) -> dict:
    """Quantify selection effects from the largest-catalogue anchor.

    Grouping defaults to the survey with the most rows, but a report should
    make that choice visible instead of presenting the result as an unbiased
    union. The returned rates are descriptive diagnostics, not corrected
    probabilities: callers can stratify acquisition or rerun with a chosen
    anchor when a science claim depends on completeness.
    """
    counts = {str(name): len(rows) for name, rows in by_survey.items()}
    if not counts:
        return {"anchor_survey": None, "survey_counts": {}, "groups": 0,
                "anchor_share": None, "matched_share": {},
                "anchor_policy": "empty"}
    anchor, policy = _resolve_anchor(by_survey, anchor_survey)
    result_groups = groups if groups is not None else group_sources(
        by_survey, anchor_survey=anchor_survey)
    matched_share = {
        survey: round(sum(1 for group in result_groups if survey in group.members)
                      / max(len(result_groups), 1), 4)
        for survey in counts
    }
    total = sum(counts.values())
    return {
        "anchor_survey": anchor,
        "anchor_policy": policy,
        "requested_anchor_survey": anchor_survey,
        "survey_counts": counts,
        "groups": len(result_groups),
        "anchor_share": round(counts[anchor] / max(total, 1), 4),
        "matched_share": matched_share,
        "warning": (f"groups are anchored on {anchor}; its selection function "
                     "defines the object population"),
    }
def summarise(groups: list[MatchGroup]) -> dict:
    """Aggregate statistics for the interface and for run records."""
    if not groups:
        return {"groups": 0, "multi_survey": 0, "ambiguous": 0,
                "by_survey_count": {}}

    counts: dict[int, int] = {}
    for group in groups:
        counts[group.independent_surveys] = \
            counts.get(group.independent_surveys, 0) + 1

    return {
        "groups": len(groups),
        "multi_survey": sum(1 for g in groups if g.independent_surveys > 1),
        "ambiguous": sum(1 for g in groups if g.ambiguous),
        "by_survey_count": {str(k): v for k, v in sorted(counts.items())},
    }
