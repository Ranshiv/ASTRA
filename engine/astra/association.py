"""Conservative event-to-candidate association.

Event packets and point-source candidates have different identities.  This
module links them only when a sky-localization test and a temporal test both
support the link.  Associations are additive review evidence: candidate
scores and ranks are never changed by this job.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import candidates, config, events, healpix_common, significance

SCHEMA_VERSION = 1
DEFAULT_RADIUS_ARCSEC = 30.0
DEFAULT_WINDOW_DAYS = 30.0


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def angular_separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation with correct RA wrap at 0/360 degrees."""
    first_ra, first_dec, second_ra, second_dec = map(
        math.radians, (float(ra1), float(dec1), float(ra2), float(dec2)))
    cosine = (math.sin(first_dec) * math.sin(second_dec) +
              math.cos(first_dec) * math.cos(second_dec) *
              math.cos(first_ra - second_ra))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _parse_time(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        # Values in the astronomical JD range are unambiguously JD; MJD is
        # useful for candidate metadata and is converted to JD first.
        if number > 2_000_000:
            unix = (number - 2440587.5) * 86400.0
        elif number > 30_000:
            unix = (number + 2400000.5 - 2440587.5) * 86400.0
        else:
            return None
        return datetime.fromtimestamp(unix, tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _candidate_times(candidate: Any) -> tuple[datetime | None, datetime | None]:
    features = candidate.features if hasattr(candidate, "features") else candidate.get("features", {})
    if not isinstance(features, dict):
        features = {}
    values = candidate if isinstance(candidate, dict) else candidate.__dict__
    starts = [features.get(key) for key in ("time_start", "start_time", "observation_start")]
    ends = [features.get(key) for key in ("time_end", "end_time", "observation_end")]
    midpoint = [features.get(key) for key in ("event_time", "mid_time", "observation_time")]
    start = next((item for item in (_parse_time(value) for value in starts) if item), None)
    end = next((item for item in (_parse_time(value) for value in ends) if item), None)
    if start is None and end is None:
        point = next((item for item in (_parse_time(value) for value in midpoint) if item), None)
        if point is None:
            point = _parse_time(values.get("event_time"))
        return point, point
    return start or end, end or start


def _event_time(event: dict[str, Any]) -> datetime | None:
    return _parse_time(event.get("event_time"))


def _point_localization(localization: dict[str, Any]) -> tuple[float, float, float] | None:
    ra = _number(localization.get("ra_deg"))
    dec = _number(localization.get("dec_deg"))
    if ra is None or dec is None:
        return None
    radius = _number(localization.get("error_radius_arcsec"))
    return ra, dec, max(float(radius or DEFAULT_RADIUS_ARCSEC), 0.1)


def _healpix_probability(ra: float, dec: float, localization: dict[str, Any]) -> dict[str, Any] | None:
    """Thin wrapper over the shared `healpix_common.pixel_probability`.

    Kept as a module-level function (rather than inlining the call at each
    use site) because it also normalizes this module's own `pixels`/
    `pixel` key aliasing and adds `target_pixel` to the result, which
    `healpix_common`'s generic contract deliberately does not carry (it is
    specific to how this module's callers report the match).
    """
    pixels = localization.get("pixels")
    nside = _number(localization.get("healpix_nside"))
    if not isinstance(pixels, list) or nside is None or nside < 1:
        return None
    clean = []
    for item in pixels:
        if not isinstance(item, dict):
            continue
        index = _number(item.get("index", item.get("pixel")))
        probability = _number(item.get("probability"))
        if index is None or probability is None or probability < 0:
            continue
        clean.append({"index": int(index), "probability": float(probability)})
    if not clean:
        return None
    try:
        result = healpix_common.pixel_probability(
            float(ra), float(dec), nside=int(nside), sparse_pixels=clean,
            precomputed_credible_levels=False)
        target = healpix_common._target_pixel(float(ra), float(dec), int(nside), "nested")
    except Exception:
        return None
    if result is None:
        return None
    return {"pixel_probability": result["pixel_probability"],
            "credible_level": result["credible_level"], "target_pixel": target}


def _spatial_match(ra: float, dec: float, localization: dict[str, Any],
                   radius_arcsec: float) -> dict[str, Any]:
    healpix = _healpix_probability(ra, dec, localization)
    if healpix is not None:
        return {"matched": bool(healpix["pixel_probability"] > 0),
                "state": "healpix", **healpix}
    point = _point_localization(localization)
    if point is None:
        return {"matched": False, "state": "unlocalized"}
    event_ra, event_dec, event_radius = point
    separation = angular_separation_deg(ra, dec, event_ra, event_dec) * 3600.0
    return {"matched": bool(separation <= event_radius + radius_arcsec),
            "state": "point", "separation_arcsec": round(separation, 5),
            "allowed_arcsec": round(event_radius + radius_arcsec, 5)}


def _temporal_match(candidate: Any, event_time: datetime | None,
                    window_days: float) -> dict[str, Any]:
    start, end = _candidate_times(candidate)
    if event_time is None:
        return {"matched": False, "state": "event_time_unknown"}
    if start is None or end is None:
        return {"matched": False, "state": "candidate_time_unknown"}
    margin = timedelta(days=float(window_days))
    matched = start - margin <= event_time <= end + margin
    distance = min(abs((event_time - start).total_seconds()),
                   abs((event_time - end).total_seconds())) / 86400.0
    return {"matched": matched, "state": "known", "distance_days": round(distance, 6)}


def associate_one(candidate: Any, event: dict[str, Any], *, radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
                  window_days: float = DEFAULT_WINDOW_DAYS,
                  allow_unknown_time: bool = False) -> dict[str, Any]:
    ra = _number(candidate.get("ra_deg") if isinstance(candidate, dict) else getattr(candidate, "ra_deg", None))
    dec = _number(candidate.get("dec_deg") if isinstance(candidate, dict) else getattr(candidate, "dec_deg", None))
    localization = event.get("localization") if isinstance(event.get("localization"), dict) else {}
    spatial = _spatial_match(ra, dec, localization, radius_arcsec) if ra is not None and dec is not None \
        else {"matched": False, "state": "candidate_position_unknown"}
    temporal = _temporal_match(candidate, _event_time(event), window_days)
    temporal_ok = temporal["matched"] or (allow_unknown_time and temporal["state"] in {
        "event_time_unknown", "candidate_time_unknown"})
    matched = bool(spatial["matched"] and temporal_ok)
    reasons = []
    if spatial["matched"]:
        reasons.append("sky localization overlaps candidate")
    else:
        reasons.append(f"spatial test: {spatial['state']}")
    if temporal["matched"]:
        reasons.append("observation time overlaps event window")
    elif temporal_ok:
        reasons.append("temporal metadata unavailable; allowed by explicit option")
    else:
        reasons.append(f"temporal test: {temporal['state']}")
    return {
        "schema_version": SCHEMA_VERSION, "event_id": event.get("event_id"),
        "provider": event.get("provider"), "matched": matched,
        "spatial": spatial, "temporal": temporal, "reasons": reasons,
        "window_days": float(window_days), "radius_arcsec": float(radius_arcsec),
    }


def _latest_packets(packet_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse packet revisions to the latest one received per event_id."""
    latest: dict[str, dict[str, Any]] = {}
    for packet in packet_rows:
        key = str(packet.get("event_id") or packet.get("packet_key"))
        previous = latest.get(key)
        if previous is None or str(packet.get("received_utc", "")) > str(previous.get("received_utc", "")):
            latest[key] = packet
    return list(latest.values())


def fetch_latest_events(*, root: Path | None = None, provider: str | None = None,
                        event_id: str | None = None, limit: int = 2000
                        ) -> list[dict[str, Any]]:
    """The latest packet revision per event_id, across all ingested providers.

    Shared entry point for anything that needs "one row per known event" --
    both `associate_candidates` (candidate-to-event linking) and
    `event_to_event_correlation`/`calibrate_event_graph`'s RPC callers
    (event-to-event linking) read the exact same deduplicated view, so a
    revised packet cannot silently produce inconsistent results between the
    two kinds of association in the same project.
    """
    root = root or config.PATHS.projects
    packet_rows = events.list_events(root=root, provider=provider, event_id=event_id,
                                     limit=limit, packets=True)
    return _latest_packets(packet_rows)


def associate_candidates(name: str = "default", *, root: Path | None = None,
                         provider: str | None = None, event_id: str | None = None,
                         radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
                         window_days: float = DEFAULT_WINDOW_DAYS,
                         allow_unknown_time: bool = False) -> dict[str, Any]:
    """Attach conservative event links to a candidate file."""
    if radius_arcsec <= 0 or window_days < 0:
        raise ValueError("association radius must be positive and window non-negative")
    root = root or config.PATHS.projects
    built = candidates.load(name, root)
    event_rows = fetch_latest_events(root=root, provider=provider, event_id=event_id)
    total_matches = 0
    candidate_matches = 0
    for candidate in built:
        associations = []
        for event in event_rows:
            result = associate_one(candidate, event, radius_arcsec=radius_arcsec,
                                   window_days=window_days,
                                   allow_unknown_time=allow_unknown_time)
            if result["matched"]:
                associations.append(result)
        candidate.event_ids = sorted({*candidate.event_ids,
                                      *[str(item["event_id"]) for item in associations]})
        candidate.explanation["event_associations"] = {
            "schema_version": SCHEMA_VERSION, "checked_events": len(event_rows),
            "matches": associations, "allow_unknown_time": allow_unknown_time,
        }
        candidate.provenance_refs.extend({
            "kind": "event_association", "event_id": item["event_id"],
            "provider": item.get("provider"), "method": "spatial_and_temporal",
            "schema_version": SCHEMA_VERSION,
        } for item in associations)
        total_matches += len(associations)
        candidate_matches += bool(associations)
    path = candidates.save(built, name, root)
    return {"name": name, "events_checked": len(event_rows),
            "candidates": len(built), "candidate_matches": candidate_matches,
            "associations": total_matches, "output_path": str(path),
            "radius_arcsec": float(radius_arcsec), "window_days": float(window_days),
            "allow_unknown_time": bool(allow_unknown_time)}


# ---------------------------------------------------------------------------
# Cross-messenger event-to-event correlation.
#
# Everything above this point links an EVENT to a point-source CANDIDATE.
# What follows links one EVENT to a DIFFERENT EVENT reported by a different
# provider -- a GW trigger and an FRB burst, say -- which is a genuinely new
# question this codebase had no answer for before: `events.py`'s own
# clustering (`metadata.list_event_clusters`) groups strictly by one event's
# own `event_id`, never across providers. Unlike `associate_one`'s
# conservative boolean pass/fail test, this is a continuous likelihood-ratio
# statistic that requires calibration (`calibrate_event_graph`, below)
# before its scale means anything -- it is explicitly new, unprecedented
# statistical machinery in this codebase, not a reuse-and-extend job.
#
# The exact statistical model chosen here (Rayleigh-vs-uniform-sky spatial
# term, uniform-window-vs-uniform-baseline temporal term) is a documented,
# defensible DEFAULT, not a settled scientific claim -- the per-provider
# positional/timing uncertainty model, and how to combine three or more
# simultaneously coincident messengers, are real statistics decisions a
# domain expert should confirm before this is trusted for a real search.
# See docs/LIMITATIONS.md for the explicit open-questions note.
#
# Deliberately NOT wired into scoring.WEIGHTS/combine() -- the same
# restraint applied everywhere else in this module, doubly warranted here
# since Λ itself is brand new and has no calibration track record yet.
# ---------------------------------------------------------------------------

def _event_point_and_sigma(event: dict[str, Any]) -> tuple[float, float, float] | None:
    """Collapse one event's localization to (ra_deg, dec_deg, sigma_arcsec).

    A point localization is used as-is. A HEALPix localization is collapsed
    via `healpix_common.effective_point_and_radius` -- an approximation
    (see that function's docstring), appropriate for a coincidence screen,
    not a rigorous map-to-map comparison. `None` when the event has no
    usable position at all.
    """
    localization = event.get("localization") if isinstance(event.get("localization"), dict) else {}
    kind = localization.get("type")
    if kind == "point":
        point = _point_localization(localization)
        return point
    if kind == "healpix":
        pixels = localization.get("pixels")
        nside = _number(localization.get("healpix_nside"))
        if not isinstance(pixels, list) or nside is None or nside < 1:
            return None
        try:
            summary = healpix_common.effective_point_and_radius(
                nside=int(nside), sparse_pixels=pixels, precomputed_credible_levels=False)
        except Exception:
            return None
        if summary is None:
            return None
        return summary["ra_deg"], summary["dec_deg"], max(summary["radius_arcsec"], 0.1)
    return None


def _spatial_likelihood_ratio(delta_theta_arcsec: float, sigma_combined_arcsec: float) -> float:
    """Rayleigh-vs-uniform-sky likelihood ratio for a 2D positional offset.

    Numerator: a Rayleigh distribution over the offset magnitude, the
    standard small-angle model for the separation between two
    independently-measured positions of the SAME true source. Denominator:
    an isotropic (uniform-on-sphere) background, whose own offset-magnitude
    density from an arbitrary reference point is proportional to
    sin(theta). A documented, tunable default -- see this module's
    docstring above -- not a claimed definitive treatment.
    """
    theta = max(math.radians(delta_theta_arcsec / 3600.0), 1e-12)
    sigma = math.radians(max(sigma_combined_arcsec, 1e-6) / 3600.0)
    same_source = (theta / sigma ** 2) * math.exp(-theta ** 2 / (2.0 * sigma ** 2))
    background = max(math.sin(theta), 1e-300) / 2.0
    return same_source / background


def _temporal_likelihood_ratio(delta_t_days: float, window_days: float,
                               background_window_days: float) -> float:
    """Uniform-in-window-vs-uniform-in-baseline likelihood ratio.

    Numerator: uniform density within the physically plausible coincidence
    window. Denominator: uniform density over `background_window_days`, the
    caller-supplied search baseline (e.g. the total time span of ingested
    packets) -- a real, tunable parameter, not a universal constant.
    """
    if window_days <= 0 or abs(delta_t_days) > window_days:
        return 0.0
    return background_window_days / window_days


def event_to_event_correlation(events_list: list[dict[str, Any]], *,
                               window_days: float = DEFAULT_WINDOW_DAYS,
                               background_window_days: float = 365.0
                               ) -> list[dict[str, Any]]:
    """Pairwise cross-messenger correlation statistic for distinct events.

    For every pair of events reported by DIFFERENT providers, computes the
    temporal and spatial offset and a likelihood-ratio association
    statistic Λ = P(Δt, Δθ | same source) / P(Δt, Δθ | independent
    sources) (see `_spatial_likelihood_ratio`/`_temporal_likelihood_ratio`
    for the exact model). Pairs sharing a provider are skipped -- that is
    what `associate_one`/`associate_candidates` already cover, and mixing
    the two would double up on the codebase's one actually-validated
    matching path for no benefit. Pairs missing a usable time or position on
    either side are silently skipped (matching this module's existing
    "unknown is not a match" discipline), not treated as non-associations.
    """
    results: list[dict[str, Any]] = []
    for index, first in enumerate(events_list):
        for second in events_list[index + 1:]:
            if str(first.get("provider")) == str(second.get("provider")):
                continue
            first_time = _event_time(first)
            second_time = _event_time(second)
            if first_time is None or second_time is None:
                continue
            first_point = _event_point_and_sigma(first)
            second_point = _event_point_and_sigma(second)
            if first_point is None or second_point is None:
                continue
            delta_t_days = (second_time - first_time).total_seconds() / 86400.0
            ra1, dec1, sigma1 = first_point
            ra2, dec2, sigma2 = second_point
            delta_theta_arcsec = angular_separation_deg(ra1, dec1, ra2, dec2) * 3600.0
            sigma_combined = math.sqrt(sigma1 ** 2 + sigma2 ** 2)
            spatial_ratio = _spatial_likelihood_ratio(delta_theta_arcsec, sigma_combined)
            temporal_ratio = _temporal_likelihood_ratio(
                delta_t_days, window_days, background_window_days)
            log_spatial = math.log(spatial_ratio) if spatial_ratio > 0 else float("-inf")
            log_temporal = math.log(temporal_ratio) if temporal_ratio > 0 else float("-inf")
            log_bayes_factor = log_spatial + log_temporal
            results.append({
                "event_a": first.get("event_id"), "provider_a": first.get("provider"),
                "event_b": second.get("event_id"), "provider_b": second.get("provider"),
                "delta_t_days": round(delta_t_days, 6),
                "delta_theta_arcsec": round(delta_theta_arcsec, 3),
                "sigma_combined_arcsec": round(sigma_combined, 3),
                "spatial_likelihood_ratio": spatial_ratio,
                "temporal_likelihood_ratio": temporal_ratio,
                "log_bayes_factor": log_bayes_factor,
                "bayes_factor": (math.exp(log_bayes_factor) if math.isfinite(log_bayes_factor)
                                 else (0.0 if log_bayes_factor == float("-inf") else float("inf"))),
                "window_days": float(window_days),
                "background_window_days": float(background_window_days),
            })
    return results


def calibrate_event_graph(events_list: list[dict[str, Any]], *,
                          window_days: float = DEFAULT_WINDOW_DAYS,
                          background_window_days: float = 365.0,
                          n_trials: int = 200, scramble_min_days: float | None = None,
                          scramble_max_days: float | None = None, seed: int = 42
                          ) -> dict[str, Any]:
    """Scrambled-time-shift null calibration for the cross-event Bayes factor.

    Observed statistic: log Bayes factors from `event_to_event_correlation`
    on the real, unshifted event set. Null/background statistic: the same
    computation repeated `n_trials` times with every event's time shifted by
    an independent random offset, while spatial structure is left untouched
    -- the standard multi-messenger "time-slide" background construction,
    not a bespoke invention.

    `scramble_min_days`/`scramble_max_days` default to `10 * window_days`
    and `background_window_days` respectively when not supplied. This
    matters, not just a convenience default: the shift must be (a) large
    enough that no genuine association can survive it (hence the
    `window_days` multiple as the floor) while (b) staying on the SAME
    order of magnitude as the real observing baseline
    (`background_window_days`) so the scrambled population's own incidental
    coincidence RATE stays comparable to the real data's -- shifting by
    centuries when the real baseline is a year would spread every trial's
    events across a baseline far wider than the one actually being
    searched, silently starving the reference population of any incidental
    matches at all and making `significance.calibrate` report "no finite
    scores" even for a real, well-populated background. The exact bounds
    remain a real search-design choice a domain expert should confirm; see
    this module's docstring above and docs/LIMITATIONS.md.

    Feeds both populations into `significance.calibrate` -- the same
    empirical-tail/FDR machinery already used for candidate-score
    calibration, reused here unchanged for a different statistic. Its
    `estimated_fdr` IS the false-coincidence rate this function exists to
    produce; no separate implementation is needed.
    """
    scramble_min_days = float(scramble_min_days) if scramble_min_days is not None \
        else 10.0 * float(window_days)
    scramble_max_days = float(scramble_max_days) if scramble_max_days is not None \
        else float(background_window_days)
    if scramble_max_days <= scramble_min_days:
        scramble_max_days = scramble_min_days * 2.0

    observed_pairs = event_to_event_correlation(
        events_list, window_days=window_days, background_window_days=background_window_days)
    observed_scores = [pair["log_bayes_factor"] for pair in observed_pairs
                       if math.isfinite(pair["log_bayes_factor"])]

    rng = np.random.default_rng(seed)
    reference_scores: list[float] = []
    for _ in range(max(1, int(n_trials))):
        shifted: list[dict[str, Any]] = []
        for event in events_list:
            event_time = _event_time(event)
            if event_time is None:
                shifted.append(event)
                continue
            offset_days = float(rng.uniform(scramble_min_days, scramble_max_days))
            if rng.random() < 0.5:
                offset_days = -offset_days
            new_time = event_time + timedelta(days=offset_days)
            shifted.append({**event, "event_time": new_time.isoformat()})
        trial_pairs = event_to_event_correlation(
            shifted, window_days=window_days, background_window_days=background_window_days)
        reference_scores.extend(pair["log_bayes_factor"] for pair in trial_pairs
                                if math.isfinite(pair["log_bayes_factor"]))

    calibration = significance.calibrate(observed_scores, reference_scores=reference_scores)
    return {
        "schema_version": SCHEMA_VERSION, "observed_pairs": len(observed_pairs),
        "observed_finite_scores": len(observed_scores), "n_trials": int(n_trials),
        "reference_pairs": len(reference_scores), "window_days": float(window_days),
        "background_window_days": float(background_window_days),
        "scramble_min_days": scramble_min_days, "scramble_max_days": scramble_max_days,
        "calibration": calibration,
    }
