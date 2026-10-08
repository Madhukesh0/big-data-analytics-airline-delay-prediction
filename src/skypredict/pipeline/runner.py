"""ETL runner: engine selection (Spark or pandas fallback) and SQLite logging.

The Spark path is the primary big-data processing engine. The pandas path is
a clearly-labelled fallback for machines without Java; results carry the
engine name so the dashboard and logs never blur the distinction.
"""

from __future__ import annotations

from pathlib import Path

from ..config import (
    CURATED_FEATURES_DIR,
    CURATED_FLIGHTS_DIR,
    CURATED_WEATHER_DIR,
    METRICS_DB_PATH,
    QUARANTINE_DIR,
)
from ..services import db
from .common import EtlResult
from .etl_pd import run_etl_pandas
from .spark_session import check_spark_available, get_spark


def run_etl(
    source_csv: Path,
    source_type: str,
    *,
    require_target: bool = False,
    weather_csv: Path | None = None,
    engine: str = "auto",  # "auto" | "spark" | "pandas"
    db_path: Path = METRICS_DB_PATH,
    quarantine_dir: Path = QUARANTINE_DIR,
    flights_dir: Path = CURATED_FLIGHTS_DIR,
    features_dir: Path = CURATED_FEATURES_DIR,
    weather_dir: Path = CURATED_WEATHER_DIR,
) -> EtlResult:
    """Run ingestion + feature engineering, logging a pipeline run to SQLite.

    With ``engine="auto"`` Spark is used when available; otherwise the run
    falls back to the pandas engine (labelled ``pandas_fallback``).
    """
    db.init_db(db_path)
    started_at = db.utcnow()

    use_spark = False
    if engine in ("auto", "spark"):
        available, _message = check_spark_available()
        use_spark = available
        if engine == "spark" and not available:
            from .spark_session import SparkUnavailableError

            raise SparkUnavailableError(_message)

    try:
        if use_spark:
            spark = get_spark()
            try:
                result, _features = run_etl_spark_local(
                    spark, source_csv, source_type,
                    require_target=require_target, weather_csv=weather_csv,
                    quarantine_dir=quarantine_dir, flights_dir=flights_dir,
                    features_dir=features_dir, weather_dir=weather_dir,
                )
            finally:
                spark.stop()
        else:
            result = run_etl_pandas(
                source_csv, source_type,
                require_target=require_target, weather_csv=weather_csv,
                quarantine_dir=quarantine_dir, flights_dir=flights_dir,
                features_dir=features_dir, weather_dir=weather_dir,
            )
    except Exception as exc:
        db.insert_pipeline_run(
            db_path, started_at=started_at, finished_at=db.utcnow(), status="failed",
            source_type=source_type, engine=engine if not use_spark else "spark",
            error_summary=repr(exc),
        )
        raise

    finished_at = db.utcnow()
    result.started_at = started_at
    result.finished_at = finished_at
    result.run_id = db.insert_pipeline_run(
        db_path,
        started_at=started_at,
        finished_at=finished_at,
        status="completed",
        source_type=source_type,
        engine=result.engine,
        counts=result.counts,
    )
    return result


def run_etl_spark_local(spark, *args, **kwargs):
    """Import-time indirection so pandas-only environments can import this module."""
    from .etl import run_etl_spark

    return run_etl_spark(spark, *args, **kwargs)
