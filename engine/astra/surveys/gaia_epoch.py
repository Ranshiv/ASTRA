"""Chunked, checkpointed ingestion for Gaia DR4 epoch photometry.

`GaiaEpochAdapter` (gaia.py) validates one chunk of rows at a time but has no
notion of a multi-chunk run: no checkpoint, no resumability, no persistence.
DR4's expected scale (~400 TB, against DR3's ~10 TB) makes an implicit,
unresumable download unworkable, so this module adds the chunked-ingestion
layer on top -- copying the checkpoint-and-resume shape
`featurematrix.build_resumable()` already established (atomic JSON
checkpoint, immutable versioned Parquet parts, schema-hash-gated resume)
rather than inventing a new one.

Deliberately NOT a registered `SurveyConnector`: DR4's real delivery
mechanism -- a bulk file drop, a paginated REST endpoint, or an async TAP
job -- is unknown before the archive actually releases, and committing to a
`cone_search()`/`fetch_light_curves()` shape now would either misrepresent
capabilities that do not exist yet or lock in a delivery assumption that
DR4's real access terms may not match. `ChunkSource` is the seam that
absorbs that uncertainty: whatever the real delivery mechanism turns out to
be, it only has to become an `Iterable[list[dict]]` to plug in here
unchanged.

Deliberately NOT wired into `scoring.WEIGHTS`/`combine()`, for the same
reason `gw.py`/`frb.py` give: this is unvalidated (indeed, entirely
pre-release) evidence with no track record in this project. Nothing in this
module changes a candidate's score.

G-band scope only, by explicit decision: `GaiaEpochAdapter.required_columns`
covers `source_id`/`time`/`g_flux`/`g_flux_error` only. Gaia has not
published a DR4 epoch table schema, so BP/RP/RVS column names are not
guessed at here -- adding them is future work once a real (or even draft)
data model exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time as _time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import gaia
from .. import config

# A chunk is one caller-defined unit of incoming rows -- a delivery file, a
# page of a paginated response, whatever the eventual DR4 access layer hands
# over one call at a time. This module does not care how a chunk was formed.
ChunkSource = Iterable[list[dict]]

GAIA_EPOCH_SCHEMA_VERSION = 2

DEFAULT_BANDS = gaia.GaiaEpochAdapter.DEFAULT_BANDS
# Preserved for callers that only ever ingested G-band data.
PARQUET_COLUMNS = ("source_id", "time", "g_flux", "g_flux_error")


def schema_hash(bands: tuple[str, ...] = DEFAULT_BANDS) -> str:
    """Content hash of the accepted-row schema, mirroring features.schema_hash().

    Ties a checkpoint (and every Parquet part it references) to the exact
    required-columns contract that produced it -- now including the selected
    band set, so a checkpoint started with `bands=("g",)` can never be
    silently resumed or combined with rows ingested under `bands=("g", "bp")`
    even though both satisfy `GAIA_EPOCH_SCHEMA_VERSION`.
    """
    columns = gaia.GaiaEpochAdapter.columns_for_bands(bands)
    payload = f"v{GAIA_EPOCH_SCHEMA_VERSION}|" + "|".join(columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class IngestReport:
    """Durable accounting for one chunked, resumable ingestion run."""

    chunks_total: int
    chunks_completed: int
    chunks_failed: int
    rows_accepted: int
    rows_rejected: int
    rejection_histogram: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    rows_per_second: float = 0.0
    resumed: bool = False

    def to_dict(self) -> dict:
        return {
            "chunks_total": self.chunks_total,
            "chunks_completed": self.chunks_completed,
            "chunks_failed": self.chunks_failed,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "rejection_histogram": dict(self.rejection_histogram),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "rows_per_second": round(self.rows_per_second, 2),
            "resumed": self.resumed,
        }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.",
                                     suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _state_root(checkpoint: Path) -> Path:
    return checkpoint.parent / checkpoint.stem


def _load_state(checkpoint: Path, bands: tuple[str, ...]) -> tuple[dict, bool]:
    if not checkpoint.exists():
        return {}, False
    try:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, False
    resumed = (state.get("schema_version") == GAIA_EPOCH_SCHEMA_VERSION
              and state.get("schema_hash") == schema_hash(bands))
    return (state, True) if resumed else ({}, False)


def _fresh_state(bands: tuple[str, ...]) -> dict:
    return {
        "schema_version": GAIA_EPOCH_SCHEMA_VERSION,
        "schema_hash": schema_hash(bands),
        "bands": list(bands),
        "completed_chunk_ids": [],
        "failed_chunk_ids": [],
        "parts": [],
        "rows_accepted": 0,
        "rows_rejected": 0,
        "rejection_histogram": {},
    }


def _write_epoch_part(path: Path, rows: list[dict], bands: tuple[str, ...]) -> None:
    """Write one immutable Parquet part of accepted epoch rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    value_columns = gaia.GaiaEpochAdapter.columns_for_bands(bands)[1:]  # excludes source_id
    columns = {"source_id": pa.array([str(row["source_id"]) for row in rows], type=pa.string())}
    for column in value_columns:
        columns[column] = pa.array([float(row[column]) for row in rows], type=pa.float64())
    table = pa.table(columns, metadata={
        b"gaia_epoch_schema_version": str(GAIA_EPOCH_SCHEMA_VERSION).encode(),
        b"gaia_epoch_schema_hash": schema_hash(bands).encode(),
        b"gaia_epoch_bands": json.dumps(list(bands)).encode(),
    })
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd", compression_level=6)
    os.replace(temporary, path)


def ingest_resumable(chunk_source: ChunkSource, *, checkpoint: Path,
                     batch_size: int = 256, bands: tuple[str, ...] = DEFAULT_BANDS,
                     progress: Callable[[dict], None] | None = None) -> IngestReport:
    """Validate and persist Gaia DR4 epoch chunks with checkpoint-and-resume.

    Each item `chunk_source` yields is one caller-defined chunk (a list of
    row dicts); chunks are validated through
    `gaia.GaiaEpochAdapter.validate_chunk` and accepted rows are accumulated
    until `batch_size` rows are ready, at which point they are flushed as one
    immutable Parquet part under
    `config.PATHS.cache / "gaia_epoch" / <checkpoint stem> / part-NNNNNN.parquet`.

    Resumability: a chunk's position in the (deterministic) sequence
    `chunk_source` produces is its identity. On a resumed run the caller must
    supply a `chunk_source` that reproduces the same sequence -- exactly the
    contract `featurematrix.build_resumable()` places on its own re-derived
    `paths` list. Chunks already marked complete in the checkpoint are
    skipped without re-validating; the iterable is still advanced past them
    (a real delivery layer with its own skip/seek cursor would avoid that
    re-iteration cost -- out of scope until DR4's real access layer is
    known, see the module docstring).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    bands = tuple(bands)
    gaia.GaiaEpochAdapter.columns_for_bands(bands)  # raises on an unknown band up front
    checkpoint = Path(checkpoint)
    state_root = _state_root(checkpoint)
    state_root.mkdir(parents=True, exist_ok=True)

    state, resumed = _load_state(checkpoint, bands)
    if not resumed:
        state = _fresh_state(bands)

    completed = set(state.get("completed_chunk_ids", []))
    failed = set(state.get("failed_chunk_ids", []))
    parts: list[str] = list(state.get("parts", []))
    rows_accepted = int(state.get("rows_accepted", 0))
    rows_rejected = int(state.get("rows_rejected", 0))
    rejection_histogram: dict[str, int] = dict(state.get("rejection_histogram", {}))
    part_index = len(parts)
    pending_rows: list[dict] = []
    chunks_total = 0

    def _flush() -> None:
        nonlocal part_index, pending_rows
        if not pending_rows:
            return
        part_path = state_root / f"part-{part_index:06d}.parquet"
        _write_epoch_part(part_path, pending_rows, bands)
        parts.append(str(part_path))
        part_index += 1
        pending_rows = []

    def _save_and_report(started: float) -> None:
        state.update({
            "completed_chunk_ids": sorted(completed),
            "failed_chunk_ids": sorted(failed),
            "parts": parts,
            "rows_accepted": rows_accepted,
            "rows_rejected": rows_rejected,
            "rejection_histogram": rejection_histogram,
        })
        _atomic_json(checkpoint, state)
        if progress is not None:
            elapsed = max(_time.monotonic() - started, 1e-9)
            progress({
                "fraction": len(completed) / max(chunks_total, 1),
                "chunks_completed": len(completed),
                "chunks_total": chunks_total,
                "rows_per_second": rows_accepted / elapsed,
            })

    started = _time.monotonic()
    for chunk_index, chunk in enumerate(chunk_source):
        chunks_total = chunk_index + 1
        if chunk_index in completed:
            continue
        try:
            result = gaia.GaiaEpochAdapter.validate_chunk(chunk, bands=bands)
        except Exception:  # noqa: BLE001 - a chunk that cannot even be validated is failed, not fatal
            failed.add(chunk_index)
            _save_and_report(started)
            continue

        rows_accepted += result["accepted"]
        rows_rejected += result["rejected"]
        for rejection in result["rejections"]:
            reason = str(rejection.get("reason", "unknown"))
            rejection_histogram[reason] = rejection_histogram.get(reason, 0) + 1
        pending_rows.extend(result["rows"])
        completed.add(chunk_index)
        failed.discard(chunk_index)

        if len(pending_rows) >= batch_size:
            _flush()
        _save_and_report(started)

    _flush()
    state["parts"] = parts
    _atomic_json(checkpoint, state)

    elapsed = max(_time.monotonic() - started, 1e-9)
    return IngestReport(
        chunks_total=chunks_total, chunks_completed=len(completed),
        chunks_failed=len(failed), rows_accepted=rows_accepted,
        rows_rejected=rows_rejected, rejection_histogram=rejection_histogram,
        elapsed_seconds=elapsed, rows_per_second=rows_accepted / elapsed,
        resumed=resumed,
    )


def read_ingested_rows(checkpoint: Path) -> list[dict]:
    """Read every accepted row a checkpoint's Parquet parts still reference.

    A part written under a different `GAIA_EPOCH_SCHEMA_VERSION`/schema hash
    is skipped, not combined -- the same stale-schema-skip discipline
    `featurematrix._read_batch_parts` uses.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return []
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    checkpoint_bands = tuple(state.get("bands") or DEFAULT_BANDS)
    rows: list[dict] = []
    for part in state.get("parts", []):
        path = Path(part)
        if not path.exists():
            continue
        try:
            table = pq.read_table(path)
            metadata = table.schema.metadata or {}
            version = int(metadata.get(b"gaia_epoch_schema_version", b"0"))
            part_bands = tuple(json.loads(metadata.get(b"gaia_epoch_bands", b"[]").decode("utf-8"))
                               or checkpoint_bands)
            recorded_hash = metadata.get(b"gaia_epoch_schema_hash", b"").decode("utf-8")
            if (version != GAIA_EPOCH_SCHEMA_VERSION or recorded_hash != schema_hash(part_bands)
                    or part_bands != checkpoint_bands):
                continue
        except Exception:  # noqa: BLE001 - an unreadable/corrupt part is skipped
            continue
        value_columns = gaia.GaiaEpochAdapter.columns_for_bands(part_bands)[1:]
        columns = {"source_id": table.column("source_id").to_pylist()}
        for name in value_columns:
            columns[name] = table.column(name).to_pylist()
        for i in range(table.num_rows):
            rows.append({"source_id": columns["source_id"][i],
                        **{name: columns[name][i] for name in value_columns}})
    return rows


def cross_match_recall(ingested_rows: list[dict], *,
                       projects_root: Path | None = None) -> dict:
    """How many ingested DR4 epoch rows correspond to an already-catalogued DR3 source.

    The current G-band-only DR4 schema (source_id/time/g_flux/g_flux_error)
    carries no independent position of its own, so this is a `source_id`
    join against the existing DR3 catalogue (`surveys/gaia.py`'s
    `GaiaConnector`, persisted via `metadata.upsert_sources`), not a
    positional `crossmatch.match_catalogs()` call -- Gaia `source_id` is
    stable across data releases, so it is a legitimate join key on its own.
    "Recall" here means "fraction of distinct ingested sources we can
    already place on the sky", which is exactly what a researcher needs
    before trusting any epoch-derived feature for that source. A nonempty
    `unmatched_source_ids` is a genuine completeness signal (a source DR4
    reports that this installation's DR3 catalogue has never seen), not a
    bug in this function.
    """
    from .. import metadata

    projects_root = projects_root or config.PATHS.projects
    dr3_ids = {
        str(row["object_id"])
        for row in metadata.list_sources(projects_root)
        if row["survey"].upper() == "GAIA"
    }
    distinct_ids = sorted({str(row["source_id"]) for row in ingested_rows})
    matched_ids = [source_id for source_id in distinct_ids if source_id in dr3_ids]
    unmatched_source_ids = [source_id for source_id in distinct_ids if source_id not in dr3_ids]

    return {
        "checked": len(distinct_ids),
        "matched": len(matched_ids),
        "recall": (len(matched_ids) / len(distinct_ids)) if distinct_ids else float("nan"),
        "unmatched_source_ids": unmatched_source_ids,
    }


def positional_residual_self_consistency(ingested_rows: list[dict], *,
                                         projects_root: Path | None = None) -> dict:
    """Round-trip check of the proper-motion propagation math, not a DR4 residual.

    This is explicitly NOT a DR4-vs-DR3 astrometric residual: the current
    G-band-only schema reports no independent DR4 position at all (see
    `cross_match_recall`'s docstring), so there is nothing yet to compare
    DR4 astrometry against. What this DOES check, using only inputs this
    pipeline actually has today: for a source with two or more ingested
    epoch rows, propagating its DR3 (J2016.0) position out to each epoch
    time and then back to a common reference epoch must land at
    (near-)identical positions -- `crossmatch.propagate_position` evaluates
    cos(dec) at the *starting* position of each call, so a perfect
    cancellation is not guaranteed by construction; a real but small
    residual is expected and a large one would indicate a genuine bug in
    the propagation math, not just floating-point noise.

    `time` values are JD, matching the fixture convention already
    established by `tests/test_connectors.py`'s
    `GaiaEpochAdapter.validate_chunk` fixture (values like 2459000.1) and the
    JD convention `frb.py`/`gw.py` already use throughout this codebase --
    NOT the fractional-Julian-year convention `crossmatch.propagate_position`
    itself expects, so each time is converted via the standard Julian-epoch
    formula (`2000.0 + (JD - 2451545.0) / 365.25`, i.e. J2000.0 at
    JD 2451545.0, one Julian year = 365.25 days) before propagation. The DR4
    epoch table's real time column/unit is not yet published (see the module
    docstring), so this conversion is an explicit, documented assumption
    that will need revisiting once a real schema exists.
    """
    from .. import crossmatch, metadata

    projects_root = projects_root or config.PATHS.projects
    dr3_by_id = {
        str(row["object_id"]): row
        for row in metadata.list_sources(projects_root)
        if row["survey"].upper() == "GAIA"
    }

    times_by_source: dict[str, list[float]] = {}
    for row in ingested_rows:
        jd = float(row["time"])
        julian_year = 2000.0 + (jd - 2_451_545.0) / 365.25
        times_by_source.setdefault(str(row["source_id"]), []).append(julian_year)

    residuals_arcsec: list[float] = []
    checked_sources = 0
    for source_id, times in times_by_source.items():
        record = dr3_by_id.get(source_id)
        if record is None or len(times) < 2 or record.get("ra_deg") is None:
            continue
        checked_sources += 1
        extra: dict[str, Any] = record.get("extra") or {}
        pm_ra, pm_dec = extra.get("pmra"), extra.get("pmdec")
        reference_epoch = min(times)

        positions_at_reference = []
        for epoch_time in times:
            ra_t, dec_t = crossmatch.propagate_position(
                record["ra_deg"], record["dec_deg"], pm_ra, pm_dec,
                crossmatch.GAIA_EPOCH, epoch_time)
            ra_back, dec_back = crossmatch.propagate_position(
                ra_t, dec_t, pm_ra, pm_dec, epoch_time, reference_epoch)
            positions_at_reference.append((ra_back, dec_back))

        base_ra, base_dec = positions_at_reference[0]
        for ra, dec in positions_at_reference[1:]:
            residuals_arcsec.append(
                crossmatch.angular_separation_arcsec(base_ra, base_dec, ra, dec))

    if not residuals_arcsec:
        return {"checked_sources": checked_sources,
                "median_residual_arcsec": None, "p95_residual_arcsec": None}
    return {
        "checked_sources": checked_sources,
        "median_residual_arcsec": float(np.median(residuals_arcsec)),
        "p95_residual_arcsec": float(np.percentile(residuals_arcsec, 95)),
    }


def epoch_completeness(ingested_rows: list[dict], *,
                       expected_epochs_per_source: dict[str, int] | int) -> dict:
    """Fraction of expected epochs actually present per source, after ingestion.

    `expected_epochs_per_source` is either one shared expectation (an `int`,
    for a batch where every source was targeted the same number of times) or
    a per-`source_id` mapping (for a real cadence plan, where sky position
    and scanning-law visibility make the expected epoch count genuinely
    different source to source). A source present in `ingested_rows` but
    absent from a supplied mapping is not scored -- there is no expectation
    to compare against, which is a different situation from "0 of N epochs
    observed" and must not be silently conflated with it. Completeness above
    1.0 (more epochs observed than expected) is reported as-is, not clipped:
    it is a real, useful signal that the expectation itself needs revisiting.
    """
    observed: dict[str, int] = {}
    for row in ingested_rows:
        source_id = str(row["source_id"])
        observed[source_id] = observed.get(source_id, 0) + 1

    if isinstance(expected_epochs_per_source, int):
        if expected_epochs_per_source <= 0:
            raise ValueError("expected_epochs_per_source must be positive")
        expectations = {source_id: expected_epochs_per_source for source_id in observed}
    else:
        expectations = {str(key): int(value) for key, value in expected_epochs_per_source.items()
                        if str(key) in observed}

    per_source: dict[str, float] = {}
    for source_id, expected in expectations.items():
        if expected <= 0:
            continue
        per_source[source_id] = observed.get(source_id, 0) / expected

    completeness_values = list(per_source.values())
    return {
        "sources_observed": len(observed),
        "sources_scored": len(per_source),
        "per_source_completeness": per_source,
        "mean_completeness": float(np.mean(completeness_values)) if completeness_values else float("nan"),
        "median_completeness": float(np.median(completeness_values)) if completeness_values else float("nan"),
        "fully_complete_sources": sum(1 for value in completeness_values if value >= 1.0),
    }


def sustained_ingest_throughput(chunk_source: ChunkSource, *, checkpoint: Path,
                                batch_size: int = 256, bands: tuple[str, ...] = DEFAULT_BANDS,
                                window_seconds: float = 1.0) -> dict:
    """Sustained (windowed) ingest throughput, distinct from one run's average.

    `IngestReport.rows_per_second` is a single number over the whole call --
    it hides slow patches behind fast ones. This wraps `ingest_resumable`
    with a `progress` callback (already fired after every chunk) and buckets
    those callbacks into `window_seconds`-wide wall-clock windows, so a
    caller can see the *distribution* of throughput across a long run, not
    just its mean. `polars`/`duckdb` (already pinned project dependencies,
    otherwise unused in this module's pure-pyarrow write path) do the
    windowed aggregation here, exercising the columnar stack the project
    already committed to without changing how Parquet parts themselves are
    written.
    """
    import duckdb
    import polars as pl

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    samples: list[dict[str, float]] = []
    started = _time.monotonic()

    def _record(update: dict) -> None:
        samples.append({
            "elapsed_seconds": _time.monotonic() - started,
            "rows_per_second": float(update["rows_per_second"]),
        })

    report = ingest_resumable(chunk_source, checkpoint=checkpoint, batch_size=batch_size,
                              bands=bands, progress=_record)

    if not samples:
        return {"report": report.to_dict(), "windows": [], "peak_rows_per_second": 0.0,
                "sustained_rows_per_second_p50": 0.0, "sustained_rows_per_second_p95": 0.0}

    frame = pl.DataFrame(samples)
    windowed = duckdb.sql(f"""
        SELECT CAST(elapsed_seconds / {float(window_seconds)} AS BIGINT) AS window_index,
               max(rows_per_second) AS rows_per_second
        FROM frame
        GROUP BY window_index
        ORDER BY window_index
    """).pl()

    window_rates = windowed["rows_per_second"].to_list()
    return {
        "report": report.to_dict(),
        "window_seconds": float(window_seconds),
        "windows": window_rates,
        "peak_rows_per_second": float(max(window_rates)),
        "sustained_rows_per_second_p50": float(np.percentile(window_rates, 50)),
        "sustained_rows_per_second_p95": float(np.percentile(window_rates, 95)),
    }
