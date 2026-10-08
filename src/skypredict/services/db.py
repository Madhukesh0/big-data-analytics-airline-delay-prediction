"""SQLite repository layer for pipeline runs, model runs, and prediction logs.

All statements use parameterized SQL. Only run metadata, metrics, and
prediction summaries are stored here — no raw flight records and no
personal data.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..config import METRICS_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    source_type TEXT,
    engine TEXT,
    rows_read INTEGER,
    rows_accepted INTEGER,
    rows_duplicate INTEGER,
    rows_quarantined INTEGER,
    rows_cancelled INTEGER,
    rows_diverted INTEGER,
    error_summary TEXT
);
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    model_name TEXT,
    model_version TEXT,
    dataset_source TEXT,
    train_start TEXT,
    train_end TEXT,
    val_start TEXT,
    val_end TEXT,
    test_start TEXT,
    test_end TEXT,
    train_rows INTEGER,
    val_rows INTEGER,
    test_rows INTEGER,
    threshold REAL,
    metrics_json TEXT,
    artifact_path TEXT,
    error_summary TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_at TEXT NOT NULL,
    source TEXT,
    flight_date TEXT,
    carrier TEXT,
    origin TEXT,
    destination TEXT,
    scheduled_departure INTEGER,
    model_name TEXT,
    model_version TEXT,
    delay_probability REAL,
    predicted_delayed INTEGER,
    expected_delay_minutes REAL
);
"""


def utcnow() -> str:
    """ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path=METRICS_DB_PATH):
    """Open a SQLite connection with dict-style row access."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path=METRICS_DB_PATH) -> None:
    with connect(path) as conn:
        conn.executescript(_SCHEMA)


def insert_pipeline_run(
    path=METRICS_DB_PATH,
    *,
    started_at: str,
    finished_at: str | None = None,
    status: str = "running",
    source_type: str | None = None,
    engine: str | None = None,
    counts: dict | None = None,
    error_summary: str | None = None,
) -> int:
    counts = counts or {}
    with connect(path) as conn:
        cur = conn.execute(
            """INSERT INTO pipeline_runs
               (started_at, finished_at, status, source_type, engine,
                rows_read, rows_accepted, rows_duplicate, rows_quarantined,
                rows_cancelled, rows_diverted, error_summary)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                started_at,
                finished_at,
                status,
                source_type,
                engine,
                counts.get("rows_read"),
                counts.get("rows_accepted"),
                counts.get("rows_duplicate"),
                counts.get("rows_quarantined"),
                counts.get("rows_cancelled"),
                counts.get("rows_diverted"),
                (error_summary or None) and str(error_summary)[:500],
            ),
        )
        return int(cur.lastrowid)


def insert_model_run(
    path=METRICS_DB_PATH,
    *,
    started_at: str,
    finished_at: str | None = None,
    status: str = "running",
    model_name: str | None = None,
    model_version: str | None = None,
    dataset_source: str | None = None,
    date_ranges: dict | None = None,
    sample_counts: dict | None = None,
    threshold: float | None = None,
    metrics: dict | None = None,
    artifact_path: str | None = None,
    error_summary: str | None = None,
) -> int:
    ranges = date_ranges or {}
    samples = sample_counts or {}
    with connect(path) as conn:
        cur = conn.execute(
            """INSERT INTO model_runs
               (started_at, finished_at, status, model_name, model_version,
                dataset_source, train_start, train_end, val_start, val_end,
                test_start, test_end, train_rows, val_rows, test_rows,
                threshold, metrics_json, artifact_path, error_summary)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                started_at,
                finished_at,
                status,
                model_name,
                model_version,
                dataset_source,
                ranges.get("train_start"),
                ranges.get("train_end"),
                ranges.get("val_start"),
                ranges.get("val_end"),
                ranges.get("test_start"),
                ranges.get("test_end"),
                samples.get("train_rows"),
                samples.get("val_rows"),
                samples.get("test_rows"),
                threshold,
                json.dumps(metrics) if metrics is not None else None,
                artifact_path,
                (error_summary or None) and str(error_summary)[:500],
            ),
        )
        return int(cur.lastrowid)


def insert_prediction(path=METRICS_DB_PATH, **fields) -> int:
    with connect(path) as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (predicted_at, source, flight_date, carrier, origin, destination,
                scheduled_departure, model_name, model_version,
                delay_probability, predicted_delayed, expected_delay_minutes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fields.get("predicted_at", utcnow()),
                fields.get("source"),
                fields.get("flight_date"),
                fields.get("carrier"),
                fields.get("origin"),
                fields.get("destination"),
                fields.get("scheduled_departure"),
                fields.get("model_name"),
                fields.get("model_version"),
                fields.get("delay_probability"),
                fields.get("predicted_delayed"),
                fields.get("expected_delay_minutes"),
            ),
        )
        return int(cur.lastrowid)


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(r) for r in rows]


def list_pipeline_runs(path=METRICS_DB_PATH, limit: int = 100) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return _rows_to_dicts(rows)


def list_model_runs(path=METRICS_DB_PATH, limit: int = 100) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM model_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return _rows_to_dicts(rows)


def list_predictions(path=METRICS_DB_PATH, limit: int = 200) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return _rows_to_dicts(rows)
