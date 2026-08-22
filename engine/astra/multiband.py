"""Multiband joint-period sidecar builder.

`features.multiband_periodic_features` has a real, measured cost profile (see
its docstring): affordable at production scale via astropy's "fast" method,
prohibitive via "flexible". This module is the explicit, opt-in job that
walks a population and builds the sidecar -- never called by
`featurematrix.build()`, the same "research action a scientist explicitly
runs" pattern `stageb.compare`/`ablation.repeated` already establish, not a
default-pipeline step.

A multiband result belongs to a GROUP of bands for one object, not one curve,
so it does not fit `modalitymatrix.py`'s per-curve `KEY_COLUMNS` naturally.
Rather than changing that shared constant (image/spectral sidecars already in
production depend on it), each multiband row is written under the sentinel
band key `"__multiband__"` -- callers that want the multiband result for an
object look it up by that key, not by any real band name.

This intentionally never touches `evidence.py`'s per-survey independent
period fits (`SurveyView.best_period_days`): those are what
`evidence.score_profile`'s `period_agreement` component (weight 0.27)
compares against each other, and a single joint period across bands would
delete that comparison, not improve it. The multiband sidecar is additional,
joinable evidence -- it is not, and must not become, a replacement for the
per-survey periods.
"""

from __future__ import annotations

from pathlib import Path

from . import config, features, modalitymatrix, store

SCHEMA_VERSION = 1
MULTIBAND_BAND_KEY = "__multiband__"


def _group_curves_by_object(root: Path, survey: str | None, limit: int
                            ) -> dict[tuple[str, str, str], list]:
    """(survey, release, object_id) -> curves, reading every stored curve once.

    Mirrors featurematrix.build's own path-walking (root/SURVEY, *.parquet
    glob) so a multiband run scopes identically to a normal feature build.
    """
    search_root = root / survey.upper() if survey else root
    if not search_root.exists():
        return {}

    paths = sorted(search_root.rglob("*.parquet"))[:limit]
    groups: dict[tuple[str, str, str], list] = {}
    for path in paths:
        try:
            curve = store.read_curve(path)
        except Exception:  # noqa: BLE001 - a corrupt file is skipped, not fatal
            continue
        key = (curve.source.survey, curve.release, curve.source.object_id)
        groups.setdefault(key, []).append(curve)
    return groups


def build_multiband_sidecar(survey: str | None = None, limit: int = 10_000,
                            name: str = "default", root: Path | None = None,
                            dataset_root: Path | None = None) -> dict:
    """Run the joint-period fit over every multi-band object and save the sidecar.

    Objects with fewer than two contributing bands are skipped entirely (the
    same guard `multiband_periodic_features` itself applies) -- they never
    get a sidecar row, rather than a row full of NaNs, so `join_rows`'s
    `matched_rows` count means "objects a joint fit was actually computed
    for."
    """
    dataset_root = dataset_root or config.PATHS.datasets
    groups = _group_curves_by_object(dataset_root, survey, limit)

    payloads: list[dict] = []
    identities: list[dict] = []
    skipped_single_band = 0

    for (group_survey, release, object_id), curves in groups.items():
        band_names = {str(curve.band) for curve in curves}
        if len(band_names) < 2:
            skipped_single_band += 1
            continue

        result = features.multiband_periodic_features(curves)
        payloads.append({
            "schema_version": SCHEMA_VERSION,
            "quality": {"bands_contributing": result["bands"],
                       "points": result["points"]},
            "features": {
                "best_period_days": result["best_period_days"],
                "best_power": result["best_power"],
                "period_snr": result["period_snr"],
            },
        })
        identities.append({"survey": group_survey, "release": release,
                           "object_id": object_id, "band": MULTIBAND_BAND_KEY})

    table = modalitymatrix.save_payloads(
        payloads, "multiband", name=name, root=root, identities=identities)

    return {
        "objects_scanned": len(groups),
        "objects_fit": len(payloads),
        "objects_skipped_single_band": skipped_single_band,
        "sidecar": table.to_dict(),
    }
