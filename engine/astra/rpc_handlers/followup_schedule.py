"""Follow-up planning/tracking, night scheduling, the discard-pile scan,
literature search, physical (SED) characterization, and digital-twin
transfer scoring.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

from typing import Any

from . import common
from .common import Handler, _workspace_root

from .. import (candidates as candidates_mod, evidence, followup, literature,
                sed, significance, surveys, tensors)

def _handle_followup_plan(params: dict[str, Any]) -> dict:
    return followup.plan(
        ra_deg=float(params["ra_deg"]), dec_deg=float(params["dec_deg"]),
        start_utc=params.get("start_utc"),
        duration_hours=float(params.get("duration_hours", 12.0)),
        latitude_deg=float(params.get("latitude_deg", 43.65)),
        longitude_deg=float(params.get("longitude_deg", -79.38)),
        min_altitude_deg=float(params.get("min_altitude_deg", 30.0)),
        cadence_minutes=int(params.get("cadence_minutes", 10)),
        target_id=params.get("target_id"),
        twilight_sun_altitude_deg=float(params.get("twilight_sun_altitude_deg", -18.0)),
        min_moon_separation_deg=float(params.get("min_moon_separation_deg", 0.0)),
        max_moon_illumination=float(params.get("max_moon_illumination", 1.0)),
        max_airmass=params.get("max_airmass"),
        weather=params.get("weather"),
        facility_name=params.get("facility_name"),
        facility_constraints=params.get("facility_constraints"),
    )


def _handle_followup_request(params: dict[str, Any]) -> dict[str, Any]:
    return candidates_mod.request_followup(
        params["candidate_id"], params.get("facility_name", ""),
        params.get("note", ""), _workspace_root(params.get("project_id")))


def _handle_followup_result(params: dict[str, Any]) -> dict[str, Any]:
    return candidates_mod.record_followup_result(
        params["request_id"], params["status"], params.get("note", ""),
        _workspace_root(params.get("project_id")))


def _handle_followup_history(params: dict[str, Any]) -> list[dict]:
    return candidates_mod.followup_history(
        params["candidate_id"], _workspace_root(params.get("project_id")))


def _handle_schedule_build_night(params: dict[str, Any]) -> dict[str, Any]:
    """Build one night's observing sequence (Direction 1, "closed-loop
    decision-theoretic scheduling"): greedy insertion by entropy-per-hour
    priority plus a slew-reducing local search, over `followup.plan`'s real
    visibility windows for each candidate. Never submits anything to a
    facility -- see `schedule.py`'s own module docstring for this
    module's stated scope, and `docs/DEFERRED.txt` for why live
    submission is not implemented.
    """
    from .. import schedule as sch

    kwargs: dict[str, Any] = {}
    if "duration_hours" in params:
        kwargs["duration_hours"] = float(params["duration_hours"])
    if "exposure_hours" in params:
        kwargs["exposure_hours"] = float(params["exposure_hours"])
    if "local_search_passes" in params:
        kwargs["local_search_passes"] = int(params["local_search_passes"])
    if "followup_kwargs" in params:
        kwargs["followup_kwargs"] = dict(params["followup_kwargs"])
    result = sch.build_night_schedule(
        list(params["candidates"]), start_utc=str(params["start_utc"]), **kwargs)
    return result.to_dict()


def _handle_schedule_replan(params: dict[str, Any]) -> dict[str, Any]:
    """Re-solve the rest of a night from `from_utc` onward, preserving
    every observation already executed -- a mid-night weather change or a
    new alert never rewrites history, only what has not happened yet."""
    from .. import schedule as sch

    original = params["schedule"]
    observations = [sch.ScheduledObservation(**obs) for obs in original["observations"]]
    schedule_obj = sch.NightSchedule(
        start_utc=original["start_utc"], duration_hours=float(original["duration_hours"]),
        exposure_hours=float(original["exposure_hours"]), observations=observations,
        unscheduled_candidate_ids=list(original.get("unscheduled_candidate_ids", [])))

    kwargs: dict[str, Any] = {}
    if "exposure_hours" in params:
        kwargs["exposure_hours"] = float(params["exposure_hours"])
    if "local_search_passes" in params:
        kwargs["local_search_passes"] = int(params["local_search_passes"])
    if "followup_kwargs" in params:
        kwargs["followup_kwargs"] = dict(params["followup_kwargs"])
    result = sch.replan(
        schedule_obj, executed_candidate_ids=list(params["executed_candidate_ids"]),
        remaining_candidates=list(params["remaining_candidates"]),
        from_utc=str(params["from_utc"]), **kwargs)
    return result.to_dict()


def _handle_discard_scan(params: dict[str, Any]) -> dict[str, Any]:
    """Scan one real ZTF source for coherent discarded-epoch runs (Direction
    2, "anomalies in the discard pile"): epochs the survey's own
    `catflags` strip before any candidate is ever assembled, via
    `discard_pile.scan_source` and `surveys.ztf.ZTFConnector`'s already-real
    `fetch_light_curves_with_quality`. Cross-survey corroboration
    (`discard_corroboration.py`) and pixel-level adjudication
    (`discard_adjudication.py`) are pure functions over already-fetched
    data and are not yet wired to a real acquisition path over RPC -- see
    docs/DEFERRED.txt.
    """
    from .. import discard_pile
    from ..surveys.base import SourceRef
    from ..surveys.ztf import ZTFConnector

    source = SourceRef(survey="ZTF", object_id=str(params["object_id"]),
                       ra_deg=float(params["ra_deg"]), dec_deg=float(params["dec_deg"]))
    records = discard_pile.scan_source(
        ZTFConnector(), source,
        min_run_length=int(params.get("min_run_length", discard_pile.DEFAULT_MIN_RUN_LENGTH)))
    return {"object_id": source.object_id, "records": [record.to_dict() for record in records]}


def _handle_literature_status(_params: dict[str, Any]) -> dict:
    return literature.status()


def _handle_literature_search(params: dict[str, Any]) -> dict:
    providers = params.get("providers", ("ads", "arxiv"))
    return literature.search(
        object_id=str(params.get("object_id", "")),
        terms=params.get("terms", []), event_ids=params.get("event_ids", []),
        providers=providers, limit=int(params.get("limit", 20)),
        root=_workspace_root(params.get("project_id")),
        refresh=bool(params.get("refresh", False)), offline=bool(params.get("offline", False)),
    )


def _handle_literature_enrich(params: dict[str, Any]) -> dict:
    return literature.enrich_candidates(
        name=str(params.get("name", "default")),
        root=_workspace_root(params.get("project_id")),
        refresh=bool(params.get("refresh", False)), offline=bool(params.get("offline", False)),
        include_arxiv=bool(params.get("include_arxiv", True)),
        limit=int(params.get("limit", 20)),
    )


def _handle_physical_characterize(params: dict[str, Any]) -> dict:
    return sed.characterize(params.get("photometry", {}),
                            extinction=params.get("extinction"),
                            source=str(params.get("source", "caller")))


def _handle_physical_enrich(params: dict[str, Any]) -> dict:
    return sed.characterize_candidate(
        name=str(params.get("name", "default")),
        root=_workspace_root(params.get("project_id")),
        extinction=params.get("extinction"),
    )


def _handle_digital_twin_fit_profile(params: dict[str, Any]) -> dict[str, Any]:
    """Fit a per-survey cadence/noise profile from real stored curves.

    Read-only diagnostic (backlog item 42): never writes into
    `scoring.WEIGHTS`/`evidence.py`, the same convention every other
    interpretation-only method here (`physical.characterize`,
    `significance.calibrate`) already follows.
    """
    from .. import survey_digital_twin as sdt

    profile = sdt.fit_survey_profile(
        str(params["survey"]), limit=int(params.get("limit", 500)),
        length=int(params.get("length", sdt.DEFAULT_LENGTH)),
    )
    return profile.to_dict()


def _handle_digital_twin_sample(params: dict[str, Any]) -> dict[str, Any]:
    """Fit a profile, then sample a synthetic batch from it.

    Returns aggregate stats via `SequenceBatch.to_dict()` (rows, length,
    mean coverage), not the raw `(n, 2, length)` array -- unsuitable as a
    JSON payload and unnecessary for a diagnostic summary. Read-only, same
    scoring/evidence non-goal as every handler in this group.
    """
    from .. import survey_digital_twin as sdt

    profile = sdt.fit_survey_profile(str(params["survey"]),
                                     limit=int(params.get("limit", 500)))
    batch = sdt.sample_synthetic_batch(
        profile, n=int(params.get("n", 50)), seed=int(params.get("seed", 42)))
    return {"profile": profile.to_dict(), "batch": batch.to_dict()}


def _handle_digital_twin_evaluate_distance(params: dict[str, Any]) -> dict[str, Any]:
    """Success criterion 1 (item 42): distance between simulated and real
    summary statistics. Read-only diagnostic; never touches ranking."""
    from .. import survey_digital_twin as sdt
    from .. import survey_digital_twin_eval as sdte

    survey = str(params["survey"])
    limit = int(params.get("limit", 500))
    profile = sdt.fit_survey_profile(survey, limit=limit)
    real = tensors.build(survey=survey, limit=limit)
    if len(real) < 2:
        return {"error": f"only {len(real)} usable real sequences; need at least 2",
                "rows": len(real)}
    synthetic = sdt.sample_synthetic_batch(
        profile, n=len(real), seed=int(params.get("seed", 42)))
    distance = sdte.summary_statistic_distance(real.values, synthetic.values)
    return {"profile": profile.to_dict(), **distance}


def _handle_digital_twin_evaluate_transfer(params: dict[str, Any]) -> dict[str, Any]:
    """Success criterion 2 (item 42): transfer performance. Minutes, not
    seconds -- trains one autoencoder per seed per arm, same as
    `deep.compare`. Read-only diagnostic; never touches ranking."""
    common._require_torch()
    from .. import survey_digital_twin as sdt
    from .. import survey_digital_twin_eval as sdte

    survey = str(params["survey"])
    limit = int(params.get("limit", 500))
    profile = sdt.fit_survey_profile(survey, limit=limit)
    real = tensors.build(survey=survey, limit=limit)
    if len(real) < 10:
        return {"error": f"only {len(real)} usable real sequences; need at least 10",
                "rows": len(real)}
    synthetic = sdt.sample_synthetic_batch(
        profile, n=len(real), seed=int(params.get("seed", 42)))
    seeds = tuple(int(seed) for seed in params.get("seeds", (17, 29, 43)))
    result = sdte.evaluate_transfer_performance(
        real, synthetic, fraction=float(params.get("fraction", 0.1)),
        seeds=seeds, epochs=int(params.get("epochs", 15)))
    return {"profile": profile.to_dict(), **result}


HANDLERS: dict[str, Handler] = {
    "followup.plan": _handle_followup_plan,
    "followup.request": _handle_followup_request,
    "followup.result": _handle_followup_result,
    "followup.history": _handle_followup_history,
    "schedule.build_night": _handle_schedule_build_night,
    "schedule.replan": _handle_schedule_replan,
    "discard.scan": _handle_discard_scan,
    "literature.status": _handle_literature_status,
    "literature.search": _handle_literature_search,
    "literature.enrich": _handle_literature_enrich,
    "physical.characterize": _handle_physical_characterize,
    "physical.enrich": _handle_physical_enrich,
    "digital_twin.fit_profile": _handle_digital_twin_fit_profile,
    "digital_twin.sample": _handle_digital_twin_sample,
    "digital_twin.evaluate_distance": _handle_digital_twin_evaluate_distance,
    "digital_twin.evaluate_transfer": _handle_digital_twin_evaluate_transfer,
}
