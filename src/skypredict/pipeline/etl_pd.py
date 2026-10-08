"""Pandas fallback ETL.

Used when Java/PySpark is unavailable so the rest of the project (training,
dashboard, predictions) still works. It applies the same validation rules,
writes the same Parquet layout, and is ALWAYS labelled as
``pandas_fallback`` — it is not distributed Spark processing.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import (
    CURATED_FEATURES_DIR,
    CURATED_FLIGHTS_DIR,
    CURATED_WEATHER_DIR,
    QUARANTINE_DIR,
)
from ..features import (
    add_historical_features_pandas,
    add_weather_features_pandas,
)
from ..data.validate import validate_and_clean
from .common import EtlResult


def _write_partitioned(df: pd.DataFrame, out_dir: Path) -> None:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir, index=False, partition_cols=["year", "month"])


def run_etl_pandas(
    source_csv: Path,
    source_type: str,
    *,
    require_target: bool = False,
    weather_csv: Path | None = None,
    quarantine_dir: Path = QUARANTINE_DIR,
    flights_dir: Path = CURATED_FLIGHTS_DIR,
    features_dir: Path = CURATED_FEATURES_DIR,
    weather_dir: Path = CURATED_WEATHER_DIR,
) -> EtlResult:
    raw = pd.read_csv(source_csv, dtype=str, keep_default_na=True)
    accepted, quarantine, ingest_counts = validate_and_clean(raw, require_target=require_target)

    quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path = None
    if not quarantine.empty:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        quarantine_path = str(quarantine_dir / f"{source_type}-{stamp}.csv")
        quarantine.to_csv(quarantine_path, index=False)

    if not accepted.empty:
        accepted["year"] = pd.to_datetime(accepted["flight_date"]).dt.year
        accepted["month"] = pd.to_datetime(accepted["flight_date"]).dt.month
    _write_partitioned(accepted, flights_dir)

    flights = pd.read_parquet(flights_dir)
    features = add_historical_features_pandas(flights)

    weather_used = False
    if weather_csv is not None:
        weather_path = Path(weather_csv)
        if not weather_path.exists():
            raise FileNotFoundError(f"Weather CSV not found: {weather_path}")
        weather = pd.read_csv(weather_path, dtype=str)
        required = {"airport", "observation_date", "observation_timestamp", "temperature_c",
                    "wind_speed_kmh", "precipitation_mm", "visibility_km"}
        missing = required - set(weather.columns)
        if missing:
            raise ValueError(
                f"Weather CSV missing required column(s): {sorted(missing)}. "
                f"Found: {sorted(weather.columns)}."
            )
        weather["observation_date"] = pd.to_datetime(weather["observation_date"], errors="coerce")
        weather["year"] = weather["observation_date"].dt.year
        weather["month"] = weather["observation_date"].dt.month
        weather = weather.dropna(subset=["observation_date"])
        weather["airport"] = weather["airport"].str.strip().str.upper()
        _write_partitioned(weather, weather_dir)
        weather_flat = pd.read_parquet(weather_dir)
        features = add_weather_features_pandas(features, weather_flat)
        weather_used = True

    features["year"] = pd.to_datetime(features["flight_date"]).dt.year
    features["month"] = pd.to_datetime(features["flight_date"]).dt.month
    _write_partitioned(features, features_dir)

    return EtlResult(
        run_id=0,
        source_type=source_type,
        source_path=str(source_csv),
        engine="pandas_fallback",
        counts=ingest_counts.as_dict(),
        quarantine_path=quarantine_path,
        flights_path=str(flights_dir),
        features_path=str(features_dir),
        weather_used=weather_used,
    )
