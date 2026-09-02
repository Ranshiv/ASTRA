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

import hashlib
import logging
from dataclasses import asdict, dataclass, field

import numpy as np

from .. import acquire as acquire_mod
from .. import config
from .. import crossmatch as crossmatch_mod
from .. import manifest as manifest_mod
from .. import metadata as metadata_mod
from .. import store as store_mod
from .. import timeframe as timeframe_mod
from ..surveys.base import ConeQuery, SourceRef
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
    dropped_no_photometry: int = 0

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
        acquired_sources = _acquired_sources(manifest, object_ids)
        label_records = _pull_simbad_labels(acquired_sources, query)
        if label_records:
            store.save_label_records(label_records, name=dataset_id)

    object_split = splits_mod.object_grouped_split(object_ids, split_id=f"{dataset_id}_object_split")
    time_records, dropped_no_photometry = _object_time_records(manifest)
    _assert_not_degenerate(time_records, dataset_id=dataset_id)
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
        dropped_no_photometry=dropped_no_photometry,
    )


def _object_time_records(manifest: manifest_mod.Manifest) -> tuple[list[dict], int]:
    """Real per-object sky position and epoch for the sky/time split.

    Pulls `ra_deg`/`dec_deg`/`time` straight from each object's own stored
    light curve (embedded in the curve's parquet metadata by
    `store.write_curve`), rather than reusing the cone's centre and a
    constant placeholder epoch for every object. The previous constant-input
    version put every object in one sky/time cell by construction, so
    `detect_leakage` reported `clean: true` because there was nothing to
    leak between, not because the split actually separated anything.

    Objects that matched a survey by position but have no stored, readable
    curve are dropped from the *time* split -- the same "absent, not zero"
    convention `FeatureMatrix.finite_mask()` already applies to scoring, see
    docs/DATA_CARD.md's "Missingness" section. They remain in the
    object-grouped split, which needs only an object ID. The drop count is
    returned so a caller can report it rather than silently losing objects.
    """
    root = config.PATHS.datasets
    records: list[dict] = []
    seen: set[str] = set()
    all_object_ids: set[str] = set()

    for q in manifest.queries:
        all_object_ids.update(q.object_ids)
        for object_id in q.object_ids:
            if object_id in seen:
                continue
            key = hashlib.sha256(
                f"{q.survey}/{q.release}/{object_id}".encode("utf-8")
            ).hexdigest()[:32]
            shard = root / q.survey.upper() / q.release / key[:2]
            matches = sorted(shard.glob(f"{key}_*.parquet")) if shard.exists() else []
            if not matches:
                continue
            try:
                curve = store_mod.read_curve(matches[0])
            except Exception:  # noqa: BLE001 - a corrupt/partial shard is a drop, not a crash
                logger.warning("could not read stored curve %s for time-split epoch",
                               matches[0], exc_info=True)
                continue
            if len(curve.time) == 0:
                continue
            mjd = float(np.median(curve.time))
            if curve.time_system in ("JD_UTC", "HJD_UTC"):
                mjd -= timeframe_mod.MJD_TO_JD
            records.append({
                "object_id": object_id,
                "ra_deg": curve.source.ra_deg,
                "dec_deg": curve.source.dec_deg,
                "mjd": mjd,
            })
            seen.add(object_id)

    dropped = len(all_object_ids - seen)
    if dropped:
        logger.info("%d/%d objects have no stored, readable curve; excluded from "
                    "the sky/time split", dropped, len(all_object_ids))
    return records, dropped


def _assert_not_degenerate(time_records: list[dict], *, dataset_id: str) -> None:
    """Refuse to build a sky/time split that would be leakage-clean by
    vacuity: if every object with real photometry falls in the same coarse
    sky cell and observing season, there is nothing for `detect_leakage` to
    actually check."""
    if len(time_records) <= 1:
        return
    cells = {
        (round(float(r["ra_deg"]) / 45.0), round(float(r["dec_deg"]) / 45.0),
         int(float(r["mjd"]) // 90))
        for r in time_records
    }
    if len(cells) <= 1:
        raise ValueError(
            f"sky/time split for {dataset_id!r} is degenerate: all "
            f"{len(time_records)} objects with stored photometry fall in one "
            "coarse sky/time cell -- refusing to build a split that would be "
            "leakage-clean by construction vacuity rather than by actually "
            "separating objects. Check that the acquired curves carry real, "
            "distinct positions/epochs.")


def _acquired_sources(manifest: manifest_mod.Manifest, object_ids: list[str]
                      ) -> list[SourceRef]:
    """Real positions for this manifest's own acquired objects, for the
    SIMBAD cross-match below.

    `acquire._acquire_one` calls `metadata.upsert_sources` at discovery time
    (before any light-curve fetch), so this is available even for objects
    whose photometry later failed to download -- a wider, more reliable
    position source than re-deriving positions from stored curves the way
    `_object_time_records` must (curves are extract-and-discard evidence,
    not guaranteed present; discovery rows are not). Filtered to this
    manifest's own object IDs so a label pull never draws in another
    acquisition's objects that happen to share the local metadata store.
    """
    wanted = set(object_ids)
    rows = metadata_mod.list_sources(config.PATHS.projects)
    sources: list[SourceRef] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        object_id = str(row["object_id"])
        if object_id not in wanted:
            continue
        key = (row["survey"], object_id)
        if key in seen or row["ra_deg"] is None or row["dec_deg"] is None:
            continue
        seen.add(key)
        sources.append(SourceRef(survey=row["survey"], object_id=object_id,
                                 ra_deg=float(row["ra_deg"]), dec_deg=float(row["dec_deg"]),
                                 extra=row.get("extra") or {}))
    return sources


def _pull_simbad_labels(sources: list[SourceRef], query: ConeQuery,
                        radius_arcsec: float = crossmatch_mod.DEFAULT_RADIUS_ARCSEC
                        ) -> list[LabelRecord]:
    """Real SIMBAD cross-match, positionally matched to specific ASTRA
    objects via `crossmatch.match_catalogs` -- not a field lookup keyed by
    SIMBAD's own identifier the way this used to work (docs/DATA_CARD.md's
    "Known biases": a `LabelRecord` from the old version was "known objects
    present in this field", never "this ASTRA object has this label").

    Not a bulk join -- one cone lookup for the whole field, appropriate at
    demonstration scale; the eventual thousands-of-objects release needs a
    batched VizieR/TAP cross-match instead (this function's per-source
    matching stays the same either way; only the SIMBAD query call needs
    batching).
    """
    if not sources:
        return []
    try:
        from astroquery.simbad import Simbad
    except ImportError:
        logger.warning("astroquery.simbad unavailable; skipping label pull")
        return []

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
    ra_key = "ra" if "ra" in table.colnames else None
    dec_key = "dec" if "dec" in table.colnames else None
    if not (otype_key and main_id_key and ra_key and dec_key):
        logger.warning("SIMBAD query_region result missing expected columns "
                       "(main_id/otype/ra/dec); got %s", table.colnames)
        return []

    counterparts: list[SourceRef] = []
    for row in table:
        try:
            ra = float(row[ra_key])
            dec = float(row[dec_key])
        except (TypeError, ValueError):
            continue  # a row with no usable coordinate cannot be matched
        counterparts.append(SourceRef(
            survey="SIMBAD", object_id=str(row[main_id_key]),
            ra_deg=ra, dec_deg=dec, extra={"otype": str(row[otype_key])}))

    matches = crossmatch_mod.match_catalogs(sources, counterparts, radius_arcsec=radius_arcsec)

    records: list[LabelRecord] = []
    for match in matches:
        # Confidence reflects match quality, not SIMBAD's own certainty:
        # tighter separation and fewer competing counterparts within the
        # radius both raise it; a crowded field or a marginal separation
        # both lower it, rather than reporting every match as equally sure.
        tightness = max(0.0, 1.0 - match.separation_arcsec / radius_arcsec)
        crowding_penalty = 1.0 / (1 + match.competitors)
        confidence = round(max(0.05, tightness * crowding_penalty), 4)
        records.append(LabelRecord(
            object_id=match.source.object_id,
            label=str(match.counterpart.extra.get("otype", "")),
            label_source="SIMBAD", source_release="live", confidence=confidence,
            adjudication_state="unreviewed",
        ))
    return records


__all__ = ["CorpusResult", "acquire_core_corpus"]
