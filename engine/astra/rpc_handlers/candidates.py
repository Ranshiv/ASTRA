"""Candidate loading, spatial view, per-candidate detail/timeline/explain,
export, broadcast, and the review label/vote surface.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

from typing import Any

from .common import Handler, _workspace_root

from .. import (attribution, broadcast, crossmatch, exports, featurematrix)
from .. import candidates as candidates_mod

def _handle_candidates_load(params: dict[str, Any]) -> dict[str, Any]:
    root = _workspace_root(params.get("project_id"))
    built = candidates_mod.load(params.get("name", "default"), root)
    top = int(params.get("top", 50))
    labels = candidates_mod.load_labels(root)
    return {
        "count": len(built),
        "candidates": [
            {**c.to_dict(), "label": labels.get(c.candidate_id, {}).get("label")}
            for c in built[:top]
        ],
    }


# scoring.py:178 already refuses a parallax with SNR < 5 ("too noisy to
# use") when screening period-luminosity consistency. The spatial view reuses
# that exact threshold rather than inventing a second one: a distance judged
# unreliable enough to skip a luminosity check is unreliable enough to skip
# plotting in 3D space.
GAIA_PARALLAX_SNR_THRESHOLD = 5.0


# scoring.py:178 already refuses a parallax with SNR < 5 ("too noisy to
# use") when screening period-luminosity consistency. The spatial view reuses
# that exact threshold rather than inventing a second one: a distance judged
# unreliable enough to skip a luminosity check is unreliable enough to skip
# plotting in 3D space.

def _handle_candidates_spatial(params: dict[str, Any]) -> dict[str, Any]:
    """RA/Dec/Gaia-distance per candidate, for the 3D spatial view.

    The live candidate pipeline never joins Gaia distance columns -- only the
    offline ablation study does, via featurematrix.join_gaia_columns. Rather
    than changing candidate-build provenance (a FEATURE_VERSION-adjacent
    concern this view has no business touching), this reuses that same join
    against a deliberately empty FeatureMatrix built only from the already-
    loaded candidates' identities. Nothing is re-extracted or recomputed.
    """
    import numpy as np

    root = _workspace_root(params.get("project_id"))
    built = candidates_mod.load(params.get("name", "default"), root)
    top = int(params.get("top", 200))
    subset = built[:top]

    identities = [{"path": c.path, "object_id": c.object_id,
                  "survey": c.survey, "ra_deg": c.ra_deg, "dec_deg": c.dec_deg}
                 for c in subset]
    empty = featurematrix.FeatureMatrix(
        values=np.empty((len(subset), 0)), identities=identities,
        feature_names=())
    joined, diagnostics = featurematrix.join_gaia_columns(
        empty, radius_arcsec=crossmatch.DEFAULT_RADIUS_ARCSEC, projects_root=root)

    distance_col = joined.feature_names.index("gaia_distance_pc")
    abs_g_col = joined.feature_names.index("gaia_abs_g_mag")
    snr_col = joined.feature_names.index("gaia_parallax_snr")
    matched_col = joined.feature_names.index("gaia_matched")
    ra_now_col = joined.feature_names.index("gaia_ra_now_deg")
    dec_now_col = joined.feature_names.index("gaia_dec_now_deg")

    points = []
    reliable = 0
    for candidate, row in zip(subset, joined.values):
        snr = row[snr_col]
        distance_reliable = bool(
            row[matched_col] == 1.0 and np.isfinite(row[distance_col])
            and np.isfinite(snr) and snr >= GAIA_PARALLAX_SNR_THRESHOLD)
        if distance_reliable:
            reliable += 1
        points.append({
            "candidate_id": candidate.candidate_id,
            "ra_deg": candidate.ra_deg,
            "dec_deg": candidate.dec_deg,
            "gaia_distance_pc": (float(row[distance_col])
                                 if np.isfinite(row[distance_col]) else None),
            "gaia_abs_g_mag": (float(row[abs_g_col])
                               if np.isfinite(row[abs_g_col]) else None),
            "gaia_parallax_snr": float(snr) if np.isfinite(snr) else None,
            "distance_reliable": distance_reliable,
            "gaia_ra_now_deg": (float(row[ra_now_col])
                                if np.isfinite(row[ra_now_col]) else None),
            "gaia_dec_now_deg": (float(row[dec_now_col])
                                 if np.isfinite(row[dec_now_col]) else None),
            "score_total": candidate.score.get("total"),
        })

    return {
        "points": points,
        "total": len(subset),
        "reliable": reliable,
        "snr_threshold": GAIA_PARALLAX_SNR_THRESHOLD,
        "gaia_matched": diagnostics["matched"],
        "gaia_match_rate": diagnostics["match_rate"],
    }


def _handle_candidate_get(params: dict[str, Any]) -> dict[str, Any]:
    root = _workspace_root(params.get("project_id"))
    built = candidates_mod.load(params.get("name", "default"), root)
    candidate_id = params["candidate_id"]
    labels = candidates_mod.load_labels(root)
    for candidate in built:
        if candidate.candidate_id == candidate_id:
            return {**candidate.to_dict(), "review": labels.get(candidate_id)}
    raise KeyError(f"candidate not found: {candidate_id}")


def _handle_candidate_explain(params: dict[str, Any]) -> dict[str, Any]:
    root = _workspace_root(params.get("project_id"))
    name = params.get("name", "default")
    candidate_id = params["candidate_id"]

    built = candidates_mod.load(name, root)
    candidate = next((c for c in built if c.candidate_id == candidate_id), None)
    if candidate is None:
        return {"candidate_id": candidate_id, "explainable": False,
               "reason": "candidate not found"}

    matrix = featurematrix.load(name, root)
    row_index = next((i for i, identity in enumerate(matrix.identities)
                      if identity.get("path") == candidate.path), None)
    if row_index is None:
        return {"candidate_id": candidate_id, "explainable": False,
               "reason": "candidate not found in current feature matrix"}

    top = int(params.get("top", 10))
    if params.get("stable"):
        result = attribution.explain_candidate_stable(matrix, row_index, top=top)
    else:
        result = attribution.explain_candidate(matrix, row_index, top=top)
    return {"candidate_id": candidate_id, **result}


def _handle_candidate_timeline(params: dict[str, Any]) -> dict[str, Any]:
    return candidates_mod.timeline(
        str(params["candidate_id"]),
        params.get("name", "default"),
        _workspace_root(params.get("project_id")),
        radius_arcsec=float(params.get("radius_arcsec", 30.0)),
        max_curves=int(params.get("max_curves", 24)),
        max_points=int(params.get("max_points", 180)),
    )


def _handle_candidates_export(params: dict[str, Any]) -> dict[str, Any]:
    return exports.export_candidates(params.get("name", "default"),
                                     params.get("format", "csv"),
                                     _workspace_root(params.get("project_id")))


def _handle_candidate_broadcast(params: dict[str, Any]) -> dict[str, Any]:
    return broadcast.generate_feed(
        params.get("name", "default"),
        float(params.get("threshold", broadcast.DEFAULT_THRESHOLD)),
        _workspace_root(params.get("project_id")))


def _handle_label(params: dict[str, Any]) -> dict[str, Any]:
    entry = candidates_mod.record_label(
        params["candidate_id"], params["label"], params.get("note", ""),
        _workspace_root(params.get("project_id")))
    return {"candidate_id": params["candidate_id"], **entry}


def _handle_labels(params: dict[str, Any]) -> dict[str, Any]:
    return candidates_mod.label_summary(_workspace_root(params.get("project_id")))


def _handle_label_vote(params: dict[str, Any]) -> dict[str, Any]:
    return candidates_mod.cast_label_vote(
        params["candidate_id"], params["reviewer_id"], params["label"],
        params.get("note", ""), _workspace_root(params.get("project_id")))


def _handle_label_votes(params: dict[str, Any]) -> dict[str, Any]:
    root = _workspace_root(params.get("project_id"))
    candidate_id = params["candidate_id"]
    return {"votes": candidates_mod.label_votes(candidate_id, root),
            "tally": candidates_mod.label_vote_tally(candidate_id, root)}


def _handle_label_vote_promote(params: dict[str, Any]) -> dict[str, Any]:
    root = _workspace_root(params.get("project_id"))
    kwargs: dict[str, Any] = {}
    if "min_votes" in params:
        kwargs["min_votes"] = int(params["min_votes"])
    if "min_agreement" in params:
        kwargs["min_agreement"] = float(params["min_agreement"])
    return candidates_mod.promote_vote_consensus(params["candidate_id"], root, **kwargs)


HANDLERS: dict[str, Handler] = {
    "candidates.load": _handle_candidates_load,
    "candidates.spatial": _handle_candidates_spatial,
    "candidates.get": _handle_candidate_get,
    "candidates.timeline": _handle_candidate_timeline,
    "candidates.export": _handle_candidates_export,
    "candidates.broadcast": _handle_candidate_broadcast,
    "candidates.explain": _handle_candidate_explain,
    "candidates.label": _handle_label,
    "candidates.labels": _handle_labels,
    "candidates.vote": _handle_label_vote,
    "candidates.votes": _handle_label_votes,
    "candidates.vote_promote": _handle_label_vote_promote,
}
