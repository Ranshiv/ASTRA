"""TAP queries, calibrated significance/selection, the review queue, review
experiments, and cross-domain corroboration checks.

Split out of rpc.py (see that module's docstring for why); nothing here
changed behavior, only location.
"""

from __future__ import annotations

from typing import Any

from .common import Handler, _workspace_root

from .. import candidates as candidates_mod, config, crossmatch, evaluate, review, significance, tap

def _handle_tap_status(params: dict[str, Any]) -> dict:
    return tap.status(_workspace_root(params.get("project_id")) or config.PATHS.projects)


def _handle_tap_query(params: dict[str, Any]) -> dict:
    return tap.query(
        str(params["service"]), str(params["adql"]),
        release=str(params.get("release", "unknown")),
        root=_workspace_root(params.get("project_id")) or config.PATHS.projects,
        max_rows=int(params.get("max_rows", 200)), fmt=str(params.get("format", "csv")),
        refresh=bool(params.get("refresh", False)), offline=bool(params.get("offline", False)),
        timeout=float(params.get("timeout", 60.0)),
    )


def _handle_significance_calibrate(params: dict[str, Any]) -> dict:
    payload = significance.calibrate(
        params.get("scores", []), reference_scores=params.get("reference_scores"),
        threshold=params.get("threshold"), strata=params.get("strata"),
        method=str(params.get("method", "empirical_tail")),
    )
    if params.get("save", True):
        path = significance.save(payload, root=(_workspace_root(params.get("project_id"))
                                                or config.PATHS.root),
                                 kind="calibration", name=str(params.get("name", "default")))
        payload["path"] = str(path)
    return payload


def _handle_selection_evaluate(params: dict[str, Any]) -> dict:
    dimensions = tuple(str(item) for item in params.get(
        "dimensions", ("amplitude", "duration_days", "magnitude")))
    payload = significance.evaluate_selection(
        params.get("records", []), dimensions=dimensions, edges=params.get("edges"),
        fit_model=bool(params.get("fit_model", False)),
        model_features=tuple(str(item) for item in params.get("model_features", dimensions)),
        bootstrap_samples=int(params.get("bootstrap_samples", 0)),
        seed=int(params.get("seed", 42)),
    )
    if params.get("save", True):
        path = significance.save(payload, root=(_workspace_root(params.get("project_id"))
                                                or config.PATHS.root),
                                 kind="selection", name=str(params.get("name", "default")))
        payload["path"] = str(path)
    return payload


def _handle_review_next(params: dict[str, Any]) -> list[dict]:
    name = str(params.get("name", "default"))
    root = _workspace_root(params.get("project_id"))
    rows = candidates_mod.load(name, root)
    limit = int(params.get("limit", 20))
    if params.get("active", False):
        from .. import active_review
        weights = active_review.learn_reason_weights(rows, root)
        return active_review.reweighted_select_next(rows, weights, root, limit=limit)
    return review.select_next(rows, limit=limit)


def _handle_review_evaluate(params: dict[str, Any]) -> dict[str, Any]:
    return review.evaluate(params.get("name", "default"),
                           _workspace_root(params.get("project_id")))


def _handle_review_experiment_vote(params: dict[str, Any]) -> dict[str, Any]:
    """Cast one vote under the reviewer human-factors experiment (Direction
    6, "the review UI as a controlled experiment"): resolves the caller's
    (reviewer, candidate) pair to an experimental arm and records what was
    actually displayed, via `review_experiment.cast_experimental_vote`.
    `score_lookup` is the caller-supplied pool of {candidate_id: score} a
    `score_shuffled` decoy may be drawn from -- this handler does not
    resolve the pool itself, matching `discard_corroboration.py`'s "the
    caller resolves context, this module only analyses" layering.
    """
    from .. import review_experiment

    kwargs: dict[str, Any] = {}
    if "decision_latency_ms" in params:
        kwargs["decision_latency_ms"] = int(params["decision_latency_ms"])
    if "self_reported_confidence" in params:
        kwargs["self_reported_confidence"] = float(params["self_reported_confidence"])
    if "presentation_index" in params:
        kwargs["presentation_index"] = int(params["presentation_index"])
    return review_experiment.cast_experimental_vote(
        params["candidate_id"], params["reviewer_id"], params["label"],
        score_lookup={str(k): float(v) for k, v in params.get("score_lookup", {}).items()},
        note=params.get("note", ""), root=_workspace_root(params.get("project_id")),
        **kwargs)


def _handle_review_experiment_preregister(_params: dict[str, Any]) -> dict[str, Any]:
    from .. import review_experiment
    return review_experiment.save_preregistration()


def _handle_corroborate_equivalence(params: dict[str, Any]) -> dict[str, Any]:
    """Read-only diagnostic (Direction 3, "corroboration as a general
    multi-instrument anomaly library"): does the domain-general
    `corroborate.core` reproduce `crossmatch.group_sources`'s own
    behaviour across random synthetic fields. Never touches ranking or any
    existing candidate score -- `corroborate/` is an independent path, not
    a replacement for `crossmatch.py`."""
    from ..corroborate import eval as corroborate_eval
    return corroborate_eval.evaluate_astronomy_equivalence(
        n_trials=int(params.get("n_trials", 50)), seed=int(params.get("seed", 0)))


def _handle_corroborate_domain_transfer(params: dict[str, Any]) -> dict[str, Any]:
    """Read-only diagnostic: false-positive reduction from corroboration in
    astronomy and a synthetic GW-style domain, via the identical core
    algorithm. See `corroborate/eval.py`'s own docstring."""
    from ..corroborate import eval as corroborate_eval
    return corroborate_eval.evaluate_domain_transfer(
        astronomy_seed=int(params.get("astronomy_seed", 0)),
        gw_seed=int(params.get("gw_seed", 0)))


def _handle_corroborate_scaling(params: dict[str, Any]) -> dict[str, Any]:
    """Read-only diagnostic: corroboration's false-positive rate as a
    function of how correlated two instruments' systematics are (the
    research plan's "scaling claim")."""
    from ..corroborate import eval as corroborate_eval
    correlations = params.get("correlations")
    kwargs: dict[str, Any] = {"seed": int(params.get("seed", 0))}
    if correlations is not None:
        kwargs["correlations"] = tuple(float(v) for v in correlations)
    return corroborate_eval.evaluate_scaling_with_systematics_correlation(**kwargs)


HANDLERS: dict[str, Handler] = {
    "tap.status": _handle_tap_status,
    "tap.query": _handle_tap_query,
    "significance.calibrate": _handle_significance_calibrate,
    "selection.evaluate": _handle_selection_evaluate,
    "review.next": _handle_review_next,
    "candidates.evaluate": _handle_review_evaluate,
    "review.experiment.vote": _handle_review_experiment_vote,
    "review.experiment.preregister": _handle_review_experiment_preregister,
    "corroborate.equivalence": _handle_corroborate_equivalence,
    "corroborate.domain_transfer": _handle_corroborate_domain_transfer,
    "corroborate.scaling": _handle_corroborate_scaling,
}
