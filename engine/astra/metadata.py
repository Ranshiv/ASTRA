"""Small SQLite metadata index for reproducible local research state.

The Parquet store remains the source of truth for measurements.  This index
holds mutable metadata (sources, labels, notes, jobs and audit events) so a
pipeline rerun cannot attach a review to a different run-order identifier.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 3


def database_path(root: Path) -> Path:
    return root / "metadata.sqlite3"


def connect(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(database_path(root), timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY, applied_utc TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS sources (
        source_key TEXT PRIMARY KEY, survey TEXT NOT NULL, release TEXT NOT NULL,
        object_id TEXT NOT NULL, ra_deg REAL, dec_deg REAL,
        extra_json TEXT NOT NULL, discovered_utc TEXT NOT NULL,
        -- Acquisition state. NULL fetch_status means "discovered, not yet
        -- fetched", which is what makes a long campaign resumable: the fetch
        -- loop is driven from this cursor rather than from the cone search.
        fetch_status TEXT, attempts INTEGER NOT NULL DEFAULT 0,
        fetched_utc TEXT, last_error TEXT
      );
      CREATE TABLE IF NOT EXISTS labels (
        candidate_key TEXT PRIMARY KEY, label TEXT NOT NULL, note TEXT NOT NULL,
        recorded_utc TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY, method TEXT NOT NULL, status TEXT NOT NULL,
        submitted_utc TEXT NOT NULL, updated_utc TEXT NOT NULL,
        result_json TEXT, error TEXT,
        project_id TEXT,
        params_json TEXT NOT NULL DEFAULT '{}',
        progress_json TEXT,
        checkpoint_json TEXT,
        retry_count INTEGER NOT NULL DEFAULT 0,
        byte_count INTEGER NOT NULL DEFAULT 0,
        idempotency_key TEXT,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        error_kind TEXT
      );
      CREATE TABLE IF NOT EXISTS audit_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, event_utc TEXT NOT NULL,
        action TEXT NOT NULL, subject TEXT NOT NULL, details_json TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS catalog_cache (
        cache_key TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        release TEXT NOT NULL,
        query_hash TEXT NOT NULL,
        object_id TEXT,
        ra_deg REAL,
        dec_deg REAL,
        radius_arcsec REAL NOT NULL,
        query_json TEXT NOT NULL,
        status TEXT NOT NULL,
        response_json TEXT,
        error TEXT,
        fetched_utc TEXT NOT NULL,
        expires_utc TEXT NOT NULL,
        accessed_utc TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_catalog_cache_lookup
        ON catalog_cache(provider, release, query_hash);
    """)
    _add_missing_columns(db)
    db.execute("INSERT OR IGNORE INTO schema_migrations VALUES (?, ?)",
               (SCHEMA_VERSION, _now()))
    db.commit()
    return db


# Columns added after the table first shipped. CREATE TABLE IF NOT EXISTS does
# nothing to a database that already exists, so an existing store would silently
# keep the old shape and every cursor query would fail on a missing column.
_ADDED_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "sources": (
        ("fetch_status", "TEXT"),
        ("attempts", "INTEGER NOT NULL DEFAULT 0"),
        ("fetched_utc", "TEXT"),
        ("last_error", "TEXT"),
    ),
    "jobs": (
        ("project_id", "TEXT"),
        ("params_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("progress_json", "TEXT"),
        ("checkpoint_json", "TEXT"),
        ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
        ("byte_count", "INTEGER NOT NULL DEFAULT 0"),
        ("idempotency_key", "TEXT"),
        ("cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
        ("error_kind", "TEXT"),
    ),
}


def _add_missing_columns(db: sqlite3.Connection) -> None:
    """Idempotently bring an existing database up to the current shape."""
    for table, columns in _ADDED_COLUMNS.items():
        present = {row["name"] for row in
                   db.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, definition in columns:
            if name not in present:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_sources(root: Path, sources: list[dict]) -> int:
    if not sources:
        return 0
    with connect(root) as db:
        for source in sources:
            db.execute("""INSERT INTO sources
              (source_key,survey,release,object_id,ra_deg,dec_deg,extra_json,discovered_utc)
              VALUES (?,?,?,?,?,?,?,?)
              ON CONFLICT(source_key) DO UPDATE SET
                ra_deg=excluded.ra_deg, dec_deg=excluded.dec_deg,
                extra_json=excluded.extra_json""", (
                source["source_key"], source["survey"], source["release"],
                source["object_id"], source.get("ra_deg"), source.get("dec_deg"),
                json.dumps(source.get("extra", {}), sort_keys=True), _now()))
        db.commit()
    return len(sources)


def list_sources(root: Path) -> list[dict]:
    with connect(root) as db:
        rows = db.execute("SELECT * FROM sources ORDER BY survey,object_id").fetchall()
    return [{"source_key": row["source_key"], "survey": row["survey"],
             "release": row["release"], "object_id": row["object_id"],
             "ra_deg": row["ra_deg"], "dec_deg": row["dec_deg"],
             "extra": json.loads(row["extra_json"])} for row in rows]


# A source that has failed this many times is not retried again by a resumed
# run; it stays visible in the progress report as a permanent failure rather
# than consuming the campaign in retries.
MAX_FETCH_ATTEMPTS = 3

FETCH_DONE = "done"
FETCH_FAILED = "failed"
FETCH_EMPTY = "empty"


def pending_sources(root: Path, survey: str | None = None,
                    max_attempts: int = MAX_FETCH_ATTEMPTS,
                    limit: int | None = None) -> list[dict]:
    """Sources still needing a fetch — the resumable campaign cursor.

    Returns rows never attempted, plus rows that failed but still have retry
    budget. Because `upsert_sources` records discovery *before* the fetch loop
    runs, this survives a crash: a resumed run picks up exactly where it
    stopped instead of re-fetching everything.
    """
    clauses = ["(fetch_status IS NULL OR (fetch_status = ? AND attempts < ?))"]
    params: list[object] = [FETCH_FAILED, max_attempts]
    if survey:
        clauses.append("survey = ?")
        params.append(survey)

    query = ("SELECT * FROM sources WHERE " + " AND ".join(clauses)
             + " ORDER BY survey, object_id")
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with connect(root) as db:
        rows = db.execute(query, params).fetchall()
    return [_source_row(row) for row in rows]


def mark_source_fetched(root: Path, source_key: str, status: str,
                        error: str | None = None) -> None:
    """Record the outcome of one object's fetch, incrementing its attempts."""
    with connect(root) as db:
        db.execute("""UPDATE sources
          SET fetch_status = ?, attempts = attempts + 1,
              fetched_utc = ?, last_error = ?
          WHERE source_key = ?""", (status, _now(), error, source_key))
        db.commit()


def acquisition_progress(root: Path, survey: str | None = None) -> dict:
    """Counts by fetch state — the denominator a long run needs."""
    clause, params = ("WHERE survey = ?", [survey]) if survey else ("", [])
    with connect(root) as db:
        rows = db.execute(
            f"""SELECT COALESCE(fetch_status,'pending') AS state,
                       COUNT(*) AS n FROM sources {clause}
                GROUP BY state""", params).fetchall()
        failures = db.execute(
            f"""SELECT object_id, last_error, attempts FROM sources
                {clause + (' AND' if clause else 'WHERE')} fetch_status = ?
                ORDER BY object_id LIMIT 20""",
            [*params, FETCH_FAILED]).fetchall()

    counts = {row["state"]: row["n"] for row in rows}
    total = sum(counts.values())
    done = counts.get(FETCH_DONE, 0) + counts.get(FETCH_EMPTY, 0)
    return {
        "total": total,
        "pending": counts.get("pending", 0),
        "done": counts.get(FETCH_DONE, 0),
        "empty": counts.get(FETCH_EMPTY, 0),
        "failed": counts.get(FETCH_FAILED, 0),
        "complete_fraction": round(done / total, 4) if total else 0.0,
        "recent_failures": [
            {"object_id": row["object_id"], "error": row["last_error"],
             "attempts": row["attempts"]} for row in failures
        ],
    }


def _source_row(row: sqlite3.Row) -> dict:
    keys = set(row.keys())
    return {
        "source_key": row["source_key"], "survey": row["survey"],
        "release": row["release"], "object_id": row["object_id"],
        "ra_deg": row["ra_deg"], "dec_deg": row["dec_deg"],
        "extra": json.loads(row["extra_json"]),
        "fetch_status": row["fetch_status"] if "fetch_status" in keys else None,
        "attempts": row["attempts"] if "attempts" in keys else 0,
    }


def put_label(root: Path, candidate_key: str, label: str, note: str) -> dict:
    entry = {"label": label, "note": note, "recorded_utc": _now()}
    with connect(root) as db:
        db.execute("""INSERT INTO labels VALUES (?,?,?,?)
          ON CONFLICT(candidate_key) DO UPDATE SET label=excluded.label,
          note=excluded.note, recorded_utc=excluded.recorded_utc""",
                   (candidate_key, label, note, entry["recorded_utc"]))
        db.execute("""INSERT INTO audit_events
          (event_utc,action,subject,details_json) VALUES (?,?,?,?)""",
                   (_now(), "label", candidate_key, json.dumps(entry)))
        db.commit()
    return entry


def labels(root: Path) -> dict[str, dict]:
    with connect(root) as db:
        rows = db.execute("SELECT candidate_key,label,note,recorded_utc FROM labels").fetchall()
    return {row["candidate_key"]: {"label": row["label"], "note": row["note"],
                                    "recorded_utc": row["recorded_utc"]} for row in rows}


def move_label(root: Path, old_key: str, new_key: str) -> None:
    with connect(root) as db:
        db.execute("""UPDATE labels SET candidate_key=? WHERE candidate_key=?
          AND NOT EXISTS (SELECT 1 FROM labels WHERE candidate_key=?)""",
                   (new_key, old_key, new_key))
        db.commit()


def put_job(root: Path, job_id: str, method: str, status: str,
            result: object | None = None, error: str | None = None,
            *, project_id: str | None = None,
            params: object | None = None,
            progress: object | None = None,
            checkpoint: object | None = None,
            retry_count: int = 0,
            byte_count: int = 0,
            idempotency_key: str | None = None,
            cancel_requested: bool = False,
            error_kind: str | None = None) -> None:
    """Create or update a durable job record.

    The optional fields make this function backwards-compatible with the
    original seven-column job table while allowing an interrupted job to be
    resumed with its exact request, cursor, and progress state.
    """
    now = _now()
    with connect(root) as db:
        existing = db.execute("SELECT submitted_utc FROM jobs WHERE job_id=?",
                              (job_id,)).fetchone()
        encoded_params = json.dumps(params, sort_keys=True) if params is not None else "{}"
        encoded_progress = json.dumps(progress, sort_keys=True) if progress is not None else None
        encoded_checkpoint = json.dumps(checkpoint, sort_keys=True) if checkpoint is not None else None
        encoded_result = json.dumps(result, sort_keys=True) if result is not None else None
        if existing is None:
            db.execute("""INSERT INTO jobs
              (job_id,method,status,submitted_utc,updated_utc,result_json,error,
               project_id,params_json,progress_json,checkpoint_json,retry_count,
               byte_count,idempotency_key,cancel_requested,error_kind)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (job_id, method, status, now, now, encoded_result, error,
                        project_id, encoded_params, encoded_progress, encoded_checkpoint,
                        int(retry_count), int(byte_count), idempotency_key,
                        int(cancel_requested), error_kind))
        else:
            db.execute("""UPDATE jobs SET method=?, status=?, updated_utc=?,
              result_json=?, error=?, project_id=COALESCE(?,project_id),
              params_json=COALESCE(?,params_json), progress_json=?,
              checkpoint_json=COALESCE(?,checkpoint_json), retry_count=?,
              byte_count=?, idempotency_key=COALESCE(?,idempotency_key),
              cancel_requested=?, error_kind=? WHERE job_id=?""",
                       (method, status, now, encoded_result, error, project_id,
                        encoded_params if params is not None else None,
                        encoded_progress, encoded_checkpoint, int(retry_count),
                        int(byte_count), idempotency_key, int(cancel_requested),
                        error_kind, job_id))
        db.commit()


def get_job(root: Path, job_id: str) -> dict | None:
    with connect(root) as db:
        row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        return None
    def decode(name: str, default: object = None):
        value = row[name] if name in row.keys() else None
        return json.loads(value) if value else default

    return {"job_id": row["job_id"], "method": row["method"],
            "status": row["status"], "submitted_utc": row["submitted_utc"],
            "updated_utc": row["updated_utc"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error"],
            "project_id": row["project_id"] if "project_id" in row.keys() else None,
            "params": decode("params_json", {}),
            "progress": decode("progress_json", None),
            "checkpoint": decode("checkpoint_json", None),
            "retry_count": int(row["retry_count"] or 0) if "retry_count" in row.keys() else 0,
            "byte_count": int(row["byte_count"] or 0) if "byte_count" in row.keys() else 0,
            "idempotency_key": row["idempotency_key"] if "idempotency_key" in row.keys() else None,
            "cancel_requested": bool(row["cancel_requested"]) if "cancel_requested" in row.keys() else False,
            "error_kind": row["error_kind"] if "error_kind" in row.keys() else None}


def list_jobs(root: Path, *, statuses: tuple[str, ...] | None = None) -> list[dict]:
    with connect(root) as db:
        if statuses:
            marks = ",".join("?" for _ in statuses)
            rows = db.execute(f"SELECT job_id FROM jobs WHERE status IN ({marks}) ORDER BY submitted_utc",
                              statuses).fetchall()
        else:
            rows = db.execute("SELECT job_id FROM jobs ORDER BY submitted_utc").fetchall()
    return [get_job(root, row["job_id"]) for row in rows]


def find_job_by_idempotency(root: Path, key: str) -> dict | None:
    with connect(root) as db:
        row = db.execute("SELECT job_id FROM jobs WHERE idempotency_key=? ORDER BY submitted_utc DESC LIMIT 1",
                         (key,)).fetchone()
    return get_job(root, row["job_id"]) if row else None


def request_job_cancel(root: Path, job_id: str) -> None:
    with connect(root) as db:
        db.execute("UPDATE jobs SET cancel_requested=1, updated_utc=? WHERE job_id=?",
                   (_now(), job_id))
        db.commit()


def clear_job_cancel(root: Path, job_id: str) -> None:
    with connect(root) as db:
        db.execute("UPDATE jobs SET cancel_requested=0, updated_utc=? WHERE job_id=?",
                   (_now(), job_id))
        db.commit()


def get_catalog_cache(root: Path, cache_key: str) -> dict | None:
    """Read one catalog response without exposing any provider credential."""
    with connect(root) as db:
        row = db.execute("SELECT * FROM catalog_cache WHERE cache_key=?",
                         (cache_key,)).fetchone()
        if row is None:
            return None
        db.execute("UPDATE catalog_cache SET accessed_utc=? WHERE cache_key=?",
                   (_now(), cache_key))
        db.commit()
    return {
        "cache_key": row["cache_key"], "provider": row["provider"],
        "release": row["release"], "query_hash": row["query_hash"],
        "object_id": row["object_id"], "ra_deg": row["ra_deg"],
        "dec_deg": row["dec_deg"], "radius_arcsec": row["radius_arcsec"],
        "query": json.loads(row["query_json"]), "status": row["status"],
        "response": json.loads(row["response_json"]) if row["response_json"] else None,
        "error": row["error"], "fetched_utc": row["fetched_utc"],
        "expires_utc": row["expires_utc"], "accessed_utc": row["accessed_utc"],
    }


def put_catalog_cache(root: Path, *, cache_key: str, provider: str,
                      release: str, query_hash: str, query: dict,
                      object_id: str | None, ra_deg: float, dec_deg: float,
                      radius_arcsec: float, status: str,
                      response: object | None, error: str | None,
                      fetched_utc: str, expires_utc: str) -> None:
    """Persist a catalog result atomically with its provenance and TTL."""
    with connect(root) as db:
        db.execute("""INSERT INTO catalog_cache
          (cache_key,provider,release,query_hash,object_id,ra_deg,dec_deg,
           radius_arcsec,query_json,status,response_json,error,fetched_utc,
           expires_utc,accessed_utc)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
          ON CONFLICT(cache_key) DO UPDATE SET
            provider=excluded.provider, release=excluded.release,
            query_hash=excluded.query_hash, object_id=excluded.object_id,
            ra_deg=excluded.ra_deg, dec_deg=excluded.dec_deg,
            radius_arcsec=excluded.radius_arcsec, query_json=excluded.query_json,
            status=excluded.status, response_json=excluded.response_json,
            error=excluded.error, fetched_utc=excluded.fetched_utc,
            expires_utc=excluded.expires_utc, accessed_utc=excluded.accessed_utc""",
                   (cache_key, provider, release, query_hash, object_id,
                    float(ra_deg), float(dec_deg), float(radius_arcsec),
                    json.dumps(query, sort_keys=True, separators=(",", ":")),
                    status,
                    json.dumps(response, sort_keys=True) if response is not None else None,
                    error, fetched_utc, expires_utc, _now()))
        db.commit()


def catalog_cache_summary(root: Path) -> dict:
    """Return counts/expiry information for an offline status view."""
    with connect(root) as db:
        rows = db.execute("""SELECT provider,status,COUNT(*) AS count,
                                   MIN(expires_utc) AS earliest_expiry
                              FROM catalog_cache
                          GROUP BY provider,status
                          ORDER BY provider,status""").fetchall()
    return {"entries": [dict(row) for row in rows],
            "total": sum(int(row["count"]) for row in rows)}
