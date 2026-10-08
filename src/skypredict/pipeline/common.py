"""Shared result type for ETL runs (both Spark and pandas engines)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EtlResult:
    """Outcome of one ingestion/feature run."""

    run_id: int
    source_type: str
    source_path: str
    engine: str  # "spark" or "pandas_fallback"
    counts: dict = field(default_factory=dict)
    quarantine_path: str | None = None
    flights_path: str | None = None
    features_path: str | None = None
    weather_used: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
