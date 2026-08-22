"""Candidate assembly, explanation and review (plan sections 17, 18 and 22).

Section 17 requires every candidate to answer six questions: what happened,
why it was flagged, which observations support it, whether it could be an
artifact, what known objects resemble it, and what the researcher should do
next. A ranking that cannot answer those is not usable for research, because
nobody can act on a number.

Section 22 adds the human-in-the-loop side: a researcher labels a candidate
and that label becomes training data. Labels are stored separately from the
candidates themselves so a re-run of the pipeline never destroys human
judgement.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import artifact as artifact_mod, config, crossmatch, evidence, metadata, scoring

LABELS = ("interesting", "artifact", "known_object", "uncertain", "needs_follow_up")

# Rough class hints from period alone. Presented as "resembles", never as a
# classification, because a period is nowhere near sufficient to classify.
CLASS_HINTS = (
    ("Delta Scuti pulsator", 0.02, 0.30),
    ("RR Lyrae pulsator", 0.20, 1.20),
    ("W UMa contact binary", 0.20, 1.00),
    ("Classical Cepheid", 1.00, 100.0),
    ("Detached eclipsing binary", 0.50, 100.0),
)


@dataclass
class Candidate:
    """One ranked object with the complete evidence behind it."""

    candidate_id: str
    object_id: str
    survey: str
    band: str
    ra_deg: float
    dec_deg: float
    release: str = "unknown"
    path: str = ""
    score: dict = field(default_factory=dict)
    artifact: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    cross_survey: dict = field(default_factory=dict)
    catalog: dict = field(default_factory=dict)
    gw: dict = field(default_factory=dict)
    frb: dict = field(default_factory=dict)
    # Additive research context.  These fields are deliberately separate from
    # ``score`` so historical ranking semantics remain unchanged.
    event_ids: list[str] = field(default_factory=list)
    significance: dict = field(default_factory=dict)
    evidence_completeness: dict = field(default_factory=dict)
    source_attribution: dict = field(default_factory=dict)
    physical_characterization: dict = field(default_factory=dict)
    follow_up_plan: dict = field(default_factory=dict)
    literature: dict = field(default_factory=dict)
    provenance_refs: list[dict] = field(default_factory=list)
    explanation: dict = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def make_candidate_id(index: int) -> str:
    """Legacy run-order identifier retained for importing old reviews."""
    return f"ASTRA-{index:06d}"


def stable_candidate_id(survey: str, release: str, object_id: str,
                        band: str) -> str:
    """Identity derived only from the immutable observation coordinates."""
    raw = f"{survey}/{release}/{object_id}/{band}".encode("utf-8")
    return "cand_" + hashlib.sha256(raw).hexdigest()[:32]


def resembles(period_days: float | None) -> list[str]:
    """Known classes with a compatible period. Not a classification."""
    if period_days is None or not np.isfinite(period_days) or period_days <= 0:
        return []
    return [name for name, low, high in CLASS_HINTS if low <= period_days <= high]


def describe_behaviour(feature_values: dict[str, float]) -> str:
    """Answer 'what happened?' in plain language from the measurements."""
    parts: list[str] = []

    amplitude = feature_values.get("robust_amplitude")
    kind = "brightness"
    if amplitude is not None and np.isfinite(amplitude):
        parts.append(f"Varies by {amplitude:.3f} in {kind} (5th-95th percentile)")

    period = feature_values.get("best_period_days")
    snr = feature_values.get("period_snr")
    if period is not None and np.isfinite(period):
        confidence = "strong" if (snr or 0) > 20 else \
            ("moderate" if (snr or 0) > 10 else "weak")
        parts.append(f"shows a {confidence} periodicity at {period:.5f} d "
                     f"(peak SNR {snr:.1f})" if snr is not None and np.isfinite(snr)
                     else f"shows a periodicity at {period:.5f} d")

    chi2 = feature_values.get("reduced_chi2")
    if chi2 is not None and np.isfinite(chi2) and chi2 > 2:
        parts.append(f"is inconsistent with a constant source "
                     f"(reduced chi-square {chi2:.1f})")

    change = feature_values.get("change_point_score")
    if change is not None and np.isfinite(change) and change > 30:
        parts.append(f"contains an abrupt level change (z={change:.0f})")

    points = feature_values.get("n_points")
    if points is not None and np.isfinite(points):
        span = feature_values.get("time_span_days")
        if span is not None and np.isfinite(span):
            parts.append(f"measured over {int(points)} epochs spanning "
                         f"{span:.0f} days")

    return "; ".join(parts) + "." if parts else "Insufficient data to describe."


def recommend(breakdown: scoring.ScoreBreakdown,
              assessment: artifact_mod.ArtifactAssessment,
              feature_values: dict[str, float],
              resolved_surveys: int) -> list[str]:
    """Answer 'what should the researcher do next?'

    Recommendations follow from what is missing, not from the score. A high
    score with no corroboration needs a second instrument; a high score with
    an unknown distance needs astrometry.
    """
    actions: list[str] = []

    if assessment.likelihood >= 0.6:
        actions.append("Inspect the individual epochs and image cutouts before "
                       "spending further effort; the instrumental explanation "
                       "is currently the stronger one.")

    if resolved_surveys < 2:
        actions.append("Seek an independent detection from another survey that "
                       "resolves this object; single-instrument evidence cannot "
                       "separate astrophysics from detector behaviour.")

    points = feature_values.get("n_points")
    if points is not None and np.isfinite(points) and points < 100:
        actions.append(f"Obtain more epochs — {int(points)} is too few to "
                       f"characterise the behaviour reliably.")

    if breakdown.components.get("physical_inconsistency") is None:
        actions.append("Obtain a parallax or distance estimate; without it the "
                       "luminosity cannot be checked against the period.")
    elif (breakdown.components.get("physical_inconsistency") or 0) > 0.5:
        actions.append("Investigate the luminosity mismatch: the object does "
                       "not sit where its period implies it should.")

    if breakdown.components.get("catalog_novelty") is None:
        actions.append("Cross-reference against SIMBAD, VSX and the Transient "
                       "Name Server to establish whether this is already known.")

    period = feature_values.get("best_period_days")
    if period is not None and np.isfinite(period) and assessment.likelihood < 0.4:
        actions.append("Consider spectroscopic follow-up to confirm the class "
                       "suggested by the period and colour.")

    if not actions:
        actions.append("Review the folded light curve and confirm the period "
                       "visually before escalating.")
    return actions


def build_candidate(index: int, identity: dict, feature_values: dict[str, float],
                    anomaly_score: float | None = None,
                    model_agreement: int | None = None,
                    consistency: float | None = None,
                    resolved_surveys: int = 1,
                    blended: list[str] | None = None,
                    period_agrees: bool | None = None,
                    gaia_properties: dict | None = None,
                    catalog_evidence: dict | None = None,
                    event_ids: list[str] | None = None,
                    significance: dict | None = None,
                    evidence_completeness: dict | None = None,
                    source_attribution: dict | None = None,
                    physical_characterization: dict | None = None,
                    follow_up_plan: dict | None = None,
                    provenance_refs: list[dict] | None = None) -> Candidate:
    """Assemble one fully explained candidate."""
    breakdown = scoring.score_candidate(
        feature_values, anomaly_score=anomaly_score,
        model_agreement=model_agreement, consistency=consistency,
        gaia_properties=gaia_properties, catalog_evidence=catalog_evidence,
    )
    assessment = artifact_mod.assess(
        feature_values, resolved_surveys=resolved_surveys,
        blended=blended, period_agrees_across_surveys=period_agrees,
    )

    drivers = breakdown.top_drivers()
    explanation = {
        "what_happened": describe_behaviour(feature_values),
        "why_flagged": [
            f"{name.replace('_', ' ')} contributed {value:.3f} to the score"
            for name, value in drivers
        ] or ["No component could be computed."],
        "supporting_observations": {
            "epochs": int(feature_values.get("n_points", 0) or 0),
            "surveys_resolving": resolved_surveys,
            "blended_in": blended or [],
        },
        "could_be_artifact": assessment.to_dict(),
        "resembles": resembles(feature_values.get("best_period_days")),
        "recommended_actions": recommend(breakdown, assessment,
                                         feature_values, resolved_surveys),
    }

    release = identity.get("release", "unknown")
    # Old callers and candidate files did not carry release.  Preserve their
    # identifiers so their tests/reviews remain valid; all new feature rows do.
    candidate_id = (stable_candidate_id(identity.get("survey", "unknown"), release,
                                        identity.get("object_id", "unknown"),
                                        identity.get("band", "unknown"))
                    if release != "unknown" else make_candidate_id(index))
    return Candidate(
        candidate_id=candidate_id,
        object_id=identity.get("object_id", "unknown"),
        survey=identity.get("survey", "unknown"),
        band=identity.get("band", "unknown"),
        ra_deg=float(identity.get("ra_deg", float("nan"))),
        dec_deg=float(identity.get("dec_deg", float("nan"))),
        release=release,
        path=identity.get("path", ""),
        score=breakdown.to_dict(),
        artifact=assessment.to_dict(),
        features={k: (None if v is None or not np.isfinite(v) else round(float(v), 6))
                  for k, v in feature_values.items()},
        cross_survey={"resolved_surveys": resolved_surveys,
                      "blended": blended or [],
                      "consistency": consistency,
                      "source_attribution": source_attribution or {}},
        event_ids=list(event_ids or []),
        significance=dict(significance or {}),
        evidence_completeness=dict(evidence_completeness or {}),
        source_attribution=dict(source_attribution or {}),
        physical_characterization=dict(physical_characterization or {}),
        follow_up_plan=dict(follow_up_plan or {}),
        provenance_refs=list(provenance_refs or []),
        explanation=explanation,
    )


def rank(candidates: list[Candidate],
         demote_artifacts: bool = True,
         ranking_field: str = "total") -> list[Candidate]:
    """Order by score, optionally pushing likely artifacts down the list.

    Artifacts are demoted rather than removed. Plan section 4 treats "this is
    probably an artifact" as a real conclusion, and an artifact catalogue is
    useful for characterising the instrument.
    """
    def key(candidate: Candidate) -> float:
        total = float(candidate.score.get(ranking_field,
                                          candidate.score.get("total", 0.0)))
        if demote_artifacts:
            total *= (1.0 - float(candidate.artifact.get("likelihood", 0.0)))
        return -total

    ordered = sorted(candidates, key=key)
    for position, candidate in enumerate(ordered, start=1):
        candidate.rank = position
    return ordered


def save(candidates: list[Candidate], name: str,
         root: Path | None = None) -> Path:
    root = root or config.PATHS.projects
    path = root / "candidates" / f"{name}_candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_run_order_labels(path, candidates, root)
    path.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
    }, indent=2), encoding="utf-8")
    return path


def _migrate_run_order_labels(path: Path, current: list[Candidate], root: Path) -> None:
    """Move legacy ASTRA-###### reviews by immutable observation identity."""
    if not path.exists():
        return
    _import_legacy_labels(root)
    try:
        previous = json.loads(path.read_text(encoding="utf-8")).get("candidates", [])
    except (OSError, json.JSONDecodeError):
        return
    old_by_identity = {(item.get("survey"), item.get("object_id"), item.get("band")):
                       item.get("candidate_id") for item in previous}
    for candidate in current:
        old = old_by_identity.get((candidate.survey, candidate.object_id, candidate.band))
        if old and old != candidate.candidate_id:
            metadata.move_label(root, old, candidate.candidate_id)


def load(name: str, root: Path | None = None) -> list[Candidate]:
    root = root or config.PATHS.projects
    path = root / "candidates" / f"{name}_candidates.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Candidate(**entry) for entry in payload.get("candidates", [])]


def timeline(candidate_id: str, name: str = "default", root: Path | None = None,
             radius_arcsec: float = 30.0, max_curves: int = 24,
             max_points: int = 180) -> dict:
    """Return a bounded, normalized event timeline for one candidate.

    The canonical store remains global and immutable; the selected candidate
    and its review stay project-scoped through ``root``.  Matching is explicit
    and conservative: the candidate's own survey/object is always included,
    while other surveys must fall within ``radius_arcsec``.  TESS blend status
    is carried through as metadata and never promoted to resolved evidence.
    """
    built = load(name, root)
    selected = next((item for item in built if item.candidate_id == candidate_id), None)
    if selected is None:
        raise KeyError(f"candidate not found: {candidate_id}")

    if not np.isfinite(selected.ra_deg) or not np.isfinite(selected.dec_deg):
        return {"candidate_id": candidate_id, "radius_arcsec": radius_arcsec,
                "events": [], "curves": [], "warning": "candidate has no finite sky position"}

    curves_by_key = evidence.load_curves_by_key()
    blended = set(selected.cross_survey.get("blended", []))
    curves: list[dict] = []
    for (survey, object_id), curves_for_source in curves_by_key.items():
        for curve in curves_for_source:
            separation = crossmatch.angular_separation_arcsec(
                selected.ra_deg, selected.dec_deg,
                curve.source.ra_deg, curve.source.dec_deg,
            )
            same_object = survey == selected.survey and object_id == selected.object_id
            if not same_object and separation > radius_arcsec:
                continue
            finite = curve.finite_mask()
            if not finite.any():
                continue
            times = curve.time[finite]
            values = curve.value[finite]
            order = np.argsort(times, kind="stable")
            times, values = times[order], values[order]
            if len(times) > max_points:
                indices = np.linspace(0, len(times) - 1, max_points, dtype=int)
                times, values = times[indices], values[indices]
            curves.append({
                "survey": survey,
                "release": curve.release,
                "object_id": object_id,
                "band": curve.band,
                "path": str(getattr(curve, "path", "") or ""),
                "value_kind": curve.value_kind,
                "time_system": curve.time_system,
                "points": int(finite.sum()),
                "time_start": float(np.min(times)),
                "time_end": float(np.max(times)),
                "separation_arcsec": round(float(separation), 4),
                "resolved": survey not in blended,
                "times": [round(float(value), 8) for value in times],
                "values": [round(float(value), 8) for value in values],
            })
            if len(curves) >= max_curves:
                break
        if len(curves) >= max_curves:
            break

    curves.sort(key=lambda item: (item["time_start"], item["survey"], item["band"]))
    events = [
        {key: item[key] for key in (
            "survey", "release", "object_id", "band", "time_system",
            "time_start", "time_end", "points", "separation_arcsec", "resolved",
        )}
        for item in curves
    ]
    return {
        "candidate_id": candidate_id,
        "radius_arcsec": float(radius_arcsec),
        "events": events,
        "curves": curves,
        "truncated": len(curves) >= max_curves,
        "warning": ("TESS entries may represent blended neighbourhood flux."
                    if "TESS" in {item["survey"] for item in curves} else None),
    }


# --- Human-in-the-loop labelling (plan section 22) ---------------------------

def labels_path(root: Path | None = None) -> Path:
    root = root or config.PATHS.projects
    return root / "labels.json"


def record_label(candidate_id: str, label: str, note: str = "",
                 root: Path | None = None) -> dict:
    """Store a researcher's judgement.

    Labels live in their own file, so regenerating candidates never discards
    human review — the expensive input in the whole loop.
    """
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}; expected one of {LABELS}")

    root = root or config.PATHS.projects
    _import_legacy_labels(root)
    return metadata.put_label(root, candidate_id, label, note)


def load_labels(root: Path | None = None) -> dict:
    root = root or config.PATHS.projects
    _import_legacy_labels(root)
    return metadata.labels(root)


def _import_legacy_labels(root: Path) -> None:
    """One-way, idempotent migration from the Phase 7 JSON label file."""
    path = labels_path(root)
    if not path.exists():
        return
    try:
        legacy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    current = metadata.labels(root)
    for candidate_id, entry in legacy.items():
        if candidate_id not in current and entry.get("label") in LABELS:
            metadata.put_label(root, candidate_id, entry["label"],
                               entry.get("note", ""))


def label_summary(root: Path | None = None) -> dict:
    labels = load_labels(root)
    counts = {name: 0 for name in LABELS}
    for entry in labels.values():
        name = entry.get("label")
        if name in counts:
            counts[name] += 1
    return {"total": len(labels), "by_label": counts}
