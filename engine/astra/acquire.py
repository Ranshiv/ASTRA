"""Stage 1 acquisition (plan section 12), with extract-and-discard semantics.

The pipeline for one cone search is:

    cone search  ->  fetch light curves  ->  normalise  ->  Parquet  ->  cache trimmed

Raw downloads never become permanent. They land in the capped cache, get
extracted into the canonical store, and are evicted by `astra.cache` once the
cache exceeds its cap. The manifest records the query rather than the data, so
reproducing an experiment costs a re-fetch instead of a second copy on disk.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import cache, config, manifest as manifest_mod, metadata, project, store, surveys
from .jobs import JobCancelled
from .surveys.base import ConeQuery, LightCurve

# Every connector's cone_search() calls straight into astroquery or
# lightkurve -- ZTF and Gaia into astroquery's IRSA/Gaia clients, TESS into
# lightkurve's MAST search. None of that goes through netclient.py's
# throttle/retry/timeout wrapper, none of it checks job cancellation, and the
# outer per-survey progress message ("Searching ztf") is set once before the
# call and does not change again until it returns -- so a genuinely slow (or
# unresponsive) catalog query is indistinguishable, from the UI, from the
# whole application having hung. It has not: astroquery's own default timeout
# for this class of call is 60s, but that is astroquery's choice, not an
# ASTRA-enforced bound, and it offers no way to cancel or to report elapsed
# time while waiting.
CONE_SEARCH_TIMEOUT_S = 90.0
CONE_SEARCH_POLL_S = 1.0


def _cone_search_with_timeout(connector, query: ConeQuery, limit: int,
                              name: str, progress=None) -> list:
    """Run one connector's cone_search under an ASTRA-controlled bound.

    astroquery/lightkurve calls cannot be interrupted mid-flight, so this
    runs the call on a worker thread and polls it instead of calling it
    directly. That gives three things none of the connectors have on their
    own: a hard ceiling regardless of the library's internal timeout,
    responsive cancellation (checked every poll rather than only before and
    after the whole call), and a progress message that changes while
    genuinely still waiting, so a slow catalog reads as "still working" and
    not as "frozen".

    On timeout or cancellation the worker thread is abandoned rather than
    killed -- Python has no safe way to kill a thread blocked in a C
    extension's network call -- so it finishes on its own and its result is
    simply discarded. That is a wasted request, not a resource leak.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(connector.cone_search, query, limit)
    started = time.monotonic()
    try:
        while True:
            if progress is not None:
                progress.raise_if_cancelled()
            try:
                return future.result(timeout=CONE_SEARCH_POLL_S)
            except FutureTimeoutError:
                elapsed = time.monotonic() - started
                if elapsed > CONE_SEARCH_TIMEOUT_S:
                    raise TimeoutError(
                        f"{name} catalog query did not respond within "
                        f"{CONE_SEARCH_TIMEOUT_S:.0f}s"
                    ) from None
                if progress is not None:
                    progress.update(
                        phase="survey",
                        message=f"Querying {name} catalog… ({elapsed:.0f}s)")
    finally:
        executor.shutdown(wait=False)

LOGGER = logging.getLogger(__name__)

# How often to trim the raw download cache mid-run. Small enough that a
# TESS-heavy campaign cannot overshoot the cap by much, large enough that the
# directory walk is not a per-object cost.
CACHE_ENFORCE_EVERY = 25


@dataclass
class SurveyOutcome:
    survey: str
    release: str
    sources_found: int = 0
    curves_stored: int = 0
    points_stored: int = 0
    bytes_stored: int = 0
    skipped_existing: int = 0
    sources_stored: int = 0
    refused_capacity: int = 0
    # Objects that failed to fetch, and objects a resumed run skipped because
    # a previous run already completed them. `error` keeps only the most
    # recent message; `failed_objects` is the count that matters, and the full
    # per-object detail lives in the metadata index.
    failed_objects: int = 0
    already_fetched: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        payload = {
            "survey": self.survey,
            "release": self.release,
            "sources_found": self.sources_found,
            "curves_stored": self.curves_stored,
            "points_stored": self.points_stored,
            "mb_stored": round(self.bytes_stored / 1024 ** 2, 3),
            "skipped_existing": self.skipped_existing,
            "sources_stored": self.sources_stored,
            "refused_capacity": self.refused_capacity,
            "failed_objects": self.failed_objects,
            "already_fetched": self.already_fetched,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class AcquisitionResult:
    dataset_id: str
    query: ConeQuery
    project_id: str | None = None
    outcomes: list[SurveyOutcome] = field(default_factory=list)
    manifest_path: str | None = None
    content_hash: str | None = None

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "project_id": self.project_id,
            "query": {
                "ra_deg": self.query.ra_deg,
                "dec_deg": self.query.dec_deg,
                "radius_arcsec": self.query.radius_arcsec,
            },
            "surveys": [o.to_dict() for o in self.outcomes],
            "totals": {
                "curves": sum(o.curves_stored for o in self.outcomes),
                "points": sum(o.points_stored for o in self.outcomes),
                "mb": round(sum(o.bytes_stored for o in self.outcomes) / 1024 ** 2, 3),
            },
            "manifest_path": self.manifest_path,
            "content_hash": self.content_hash,
        }


def default_dataset_id(query: ConeQuery) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"cone_{query.key()}_{stamp}"


def acquire(
    query: ConeQuery,
    survey_names: list[str] | None = None,
    limit: int = 25,
    dataset_id: str | None = None,
    project_id: str | None = None,
    skip_existing: bool = True,
    progress=None,
    survey_options: dict[str, dict] | None = None,
) -> AcquisitionResult:
    """Run one acquisition across the requested surveys.

    A failing survey is recorded and the rest continue: an archive being down
    should degrade the dataset, not abort the run.

    `survey_options` carries per-survey connector constructor kwargs, e.g.
    `{"tess": {"author": "QLP"}}` to acquire via QLP instead of the default
    SPOC. The registry already forwards arbitrary kwargs to the connector
    class (`surveys.get(name, **kwargs)`); this is what actually supplies
    them. Omitting it reproduces today's behaviour exactly -- every connector
    is still built with its own defaults.
    """
    survey_names = survey_names or surveys.available()
    dataset_id = dataset_id or default_dataset_id(query)
    survey_options = survey_options or {}

    manifest_root = None
    if project_id:
        project.require_active(project_id)
        manifest_root = project.manifest_dir(project_id)

    record = manifest_mod.Manifest.create(dataset_id)
    result = AcquisitionResult(dataset_id=dataset_id, query=query, project_id=project_id)

    for survey_index, name in enumerate(survey_names, start=1):
        if progress is not None:
            progress.raise_if_cancelled()
            progress.update(
                phase="survey", message=f"Searching {name}",
                fraction=(survey_index - 1) / max(len(survey_names), 1),
                items_done=survey_index - 1, items_total=len(survey_names),
            )
        outcome = _acquire_one(name, query, limit, skip_existing, record, progress,
                               survey_options.get(name.lower(), {}))
        result.outcomes.append(outcome)

    if progress is not None:
        progress.update(phase="manifest", message="Sealing dataset manifest",
                        fraction=0.98, items_done=len(survey_names),
                        items_total=len(survey_names))

    record.seal()
    result.manifest_path = str(manifest_mod.save(record, manifest_root))
    result.content_hash = record.content_hash

    # Extraction is done, so the raw downloads are now disposable.
    cache.enforce_cap()
    return result


def _acquire_one(name: str, query: ConeQuery, limit: int,
                 skip_existing: bool,
                 record: manifest_mod.Manifest, progress=None,
                 connector_kwargs: dict | None = None) -> SurveyOutcome:
    try:
        connector = surveys.get(name, **(connector_kwargs or {}))
    except KeyError as exc:
        return SurveyOutcome(survey=name, release="unknown", error=str(exc))

    outcome = SurveyOutcome(survey=connector.name, release=connector.release)

    try:
        sources = _cone_search_with_timeout(connector, query, limit, name, progress)
    except JobCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - archive failures are data, not crashes
        outcome.error = f"cone search failed: {exc}"
        return outcome

    outcome.sources_found = len(sources)
    record.add(manifest_mod.SurveyQuery.from_cone(
        connector.name, connector.release, query, limit, sources))

    # Discovery is recorded before any fetch, so the object list survives a
    # crash and a resumed run can pick up from the cursor below.
    metadata.upsert_sources(config.PATHS.projects,
                            configured_source_records(connector, sources))
    outcome.sources_stored = len(sources)

    pending = _pending_for(connector, sources, skip_existing)
    outcome.already_fetched = len(sources) - len(pending)

    for index, source in enumerate(pending, start=1):
        if progress is not None:
            progress.raise_if_cancelled()
        _fetch_one(connector, source, outcome, skip_existing)
        if progress is not None:
            progress.update(
                phase="fetch", message=f"{connector.name}: {source.object_id}",
                fraction=index / max(len(pending), 1),
                items_done=index, items_total=len(pending),
            )
            progress.checkpoint({"survey": connector.name, "object_id": source.object_id})

        # The raw cache is enforced during the run, not only at the end. TESS
        # costs several MB of downloads per target, so a long campaign would
        # otherwise blow far past the cap before a single eviction happened.
        if index % CACHE_ENFORCE_EVERY == 0:
            cache.enforce_cap()

    return outcome


def _pending_for(connector, sources: list, skip_existing: bool) -> list:
    """Objects still needing a fetch, honouring the resumable cursor."""
    if not skip_existing:
        return list(sources)

    root = config.PATHS.projects
    outstanding = {row["source_key"] for row in
                   metadata.pending_sources(root, survey=connector.name)}
    return [source for source in sources
            if _source_key(connector, source) in outstanding]


def _fetch_one(connector, source, outcome: SurveyOutcome,
               skip_existing: bool) -> None:
    """Fetch and store one object, recording its outcome durably.

    Failures are counted and persisted per object. Previously a single
    overwritten `error` string was the only record, so a run that lost 500
    objects reported one message and no count — silent large-scale data loss.
    """
    key = _source_key(connector, source)
    root = config.PATHS.projects

    try:
        curves = connector.fetch_light_curves(source)
    except Exception as exc:  # noqa: BLE001 - one object must not stop the survey
        message = f"{type(exc).__name__}: {exc}"
        outcome.failed_objects += 1
        outcome.error = f"fetch failed for {source.object_id}: {message}"
        metadata.mark_source_fetched(root, key, metadata.FETCH_FAILED, message)
        LOGGER.warning("fetch failed survey=%s object=%s error=%s",
                       connector.name, source.object_id, message)
        return

    for curve in curves:
        _store_curve(curve, outcome, skip_existing)

    # A catalogue connector legitimately returns no curves; that is a
    # completed object, not a failure, and must not be retried forever.
    status = metadata.FETCH_DONE if curves else metadata.FETCH_EMPTY
    metadata.mark_source_fetched(root, key, status)


def _store_curve(curve: LightCurve, outcome: SurveyOutcome,
                 skip_existing: bool) -> None:
    if skip_existing and store.has_curve(curve):
        outcome.skipped_existing += 1
        return
    try:
        stored = store.write_curve(curve)
    except store.DatasetCapacityError as exc:
        outcome.refused_capacity += 1
        outcome.error = str(exc)
        return
    if stored.points == 0:
        return
    outcome.curves_stored += 1
    outcome.points_stored += stored.points
    outcome.bytes_stored += stored.bytes_on_disk


def _source_key(connector, source) -> str:
    """The durable identity of one object, shared by the store and the cursor."""
    return f"{connector.name}/{connector.release}/{source.object_id}"


def configured_source_records(connector, sources):
    """Convert connector results to durable metadata rows, including Gaia."""
    return [{
        "source_key": _source_key(connector, source),
        "survey": connector.name, "release": connector.release,
        "object_id": source.object_id, "ra_deg": source.ra_deg,
        "dec_deg": source.dec_deg, "extra": source.extra,
    } for source in sources]
