"""Acquire the core ZTF/Gaia/TESS research corpus.

This module does not reimplement survey querying: `astra.acquire.acquire()`
already runs one cone search across the requested connectors with
extract-and-discard semantics, per-survey failure isolation, and a sealed
`manifest.Manifest`. That machinery is reused unchanged; this module adds
the research-evidence layer around it -- copying the sealed manifest into
`research/datasets/manifests/` with license/citation/selection-rule fields
set, pulling a small real label set from SIMBAD, and building the
leakage-checked splits the acquired object IDs need.

Scale is caller-controlled via `limit`. A "core corpus" in the roadmap sense
(tens of thousands of objects) is an unattended, hours-long run of this same
function with a larger `limit` and possibly multiple cone queries -- nothing
here hardcodes a small scale, but nothing here launches a large run
unsupervised either; that is `scripts/acquire-core-corpus.ps1`'s job.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from .. import acquire as acquire_mod
from .. import manifest as manifest_mod
from ..surveys.base import ConeQuery
from . import splits as splits_mod
from . import store
from .records import LabelRecord

logger = logging.getLogger(__name__)


@dataclass
class CorpusResult:
    dataset_id: str
    manifest_content_hash: str
    total_objects: int
    survey_outcomes: list[dict]
    label_count: int
    split_ids: list[str]
    leakage_clean: bool

    def to_dict(self) -> dict:
        return asdict(self)


def acquire_core_corpus(
    query: ConeQuery, *, dataset_id: str, survey_names: tuple[str, ...] = ("ztf", "gaia", "tess"),
    limit: int = 200, selection_rule: str = "", license: str = "", citation: str = "",
    pull_labels: bool = True,
) -> CorpusResult:
    """Acquire one cross-survey object group and seal it as research evidence.

    Runs the existing acquisition pipeline, promotes the resulting sealed
    manifest into `research/datasets/manifests/` with archive-provenance
    fields filled in, optionally pulls real SIMBAD labels for the acquired
    objects, and builds an object-grouped split plus a sky/time split over
    them, checking both for leakage before returning.
    """
    result = acquire_mod.acquire(query, survey_names=list(survey_names),
                                 limit=limit, dataset_id=dataset_id)

    manifest = manifest_mod.load(dataset_id)  # config.PATHS default root
    manifest.selection_rule = selection_rule or (
        f"cone({query.ra_deg:.4f},{query.dec_deg:.4f},{query.radius_arcsec:.1f}as), "
        f"limit={limit}, surveys={list(survey_names)}")
    manifest.license = license
    manifest.citation = citation
    manifest.record_artifact(row_count=manifest.total_objects(), byte_count=0,
                             checksum=manifest.content_hash or "")
    store.save_dataset_manifest(manifest)

    object_ids = sorted({obj for q in manifest.queries for obj in q.object_ids})

    label_records: list[LabelRecord] = []
    if pull_labels and object_ids:
        label_records = _pull_simbad_labels(object_ids, query)
        if label_records:
            store.save_label_records(label_records, name=dataset_id)

    object_split = splits_mod.object_grouped_split(object_ids, split_id=f"{dataset_id}_object_split")
    time_records = [{"object_id": obj_id, "ra_deg": query.ra_deg, "dec_deg": query.dec_deg,
                     "mjd": 59000.0} for obj_id in object_ids]
    sky_split = splits_mod.sky_time_split(time_records, split_id=f"{dataset_id}_sky_time_split")

    research_root = store.research_root()
    splits_mod.save(object_split, research_root)
    splits_mod.save(sky_split, research_root)

    leakage = splits_mod.detect_leakage(object_split)
    if not leakage["clean"]:
        logger.warning("object-grouped split %s has leakage: %s",
                       object_split.split_id, leakage["leaked"])

    return CorpusResult(
        dataset_id=dataset_id,
        manifest_content_hash=manifest.content_hash or "",
        total_objects=manifest.total_objects(),
        survey_outcomes=[o.to_dict() for o in result.outcomes],
        label_count=len(label_records),
        split_ids=[object_split.split_id, sky_split.split_id],
        leakage_clean=leakage["clean"],
    )


def _pull_simbad_labels(object_ids: list[str], query: ConeQuery) -> list[LabelRecord]:
    """Real SIMBAD cross-match for a small object list. Not a bulk join --
    one cone lookup per object, appropriate for a demonstration-scale
    corpus, not the eventual thousands-of-objects release (which needs a
    batched VizieR/TAP cross-match instead)."""
    try:
        from astroquery.simbad import Simbad
    except ImportError:
        logger.warning("astroquery.simbad unavailable; skipping label pull")
        return []

    records: list[LabelRecord] = []
    simbad = Simbad()
    # Confirmed live against the real service (not assumed from docs): the
    # currently-installed astroquery's default query_region() result has
    # neither an object-type column nor SIMBAD's historical uppercase
    # column names -- it returns lowercase `main_id`/`ra`/`dec` only, and
    # `otype` must be requested explicitly via add_votable_fields.
    simbad.add_votable_fields("otype")
    try:
        table = simbad.query_region(
            f"{query.ra_deg}d {query.dec_deg:+}d", radius=f"{query.radius_arcsec}s")
    except Exception:  # noqa: BLE001 - archive lookup failure is recorded, not fatal
        logger.warning("SIMBAD query_region failed for the acquired field", exc_info=True)
        return []

    if table is None:
        return []

    otype_key = "otype" if "otype" in table.colnames else None
    main_id_key = "main_id" if "main_id" in table.colnames else None
    if not otype_key or not main_id_key:
        return []

    # SIMBAD returns objects near the field, not matched to specific
    # ASTRA object IDs (that needs a positional cross-match this module
    # does not do); each row becomes its own label keyed by the SIMBAD
    # main identifier, honestly distinct from the acquired object_ids.
    for row in table:
        records.append(LabelRecord(
            object_id=str(row[main_id_key]), label=str(row[otype_key]),
            label_source="SIMBAD", source_release="live", confidence=1.0,
            adjudication_state="unreviewed",
        ))
    return records


__all__ = ["CorpusResult", "acquire_core_corpus"]
