"""Training-data assembly and chronological splitting.

Boundary note: Spark builds the curated Parquet lake. Training itself runs
locally in scikit-learn/XGBoost — the feature table is COLLECTED to the
driver as a pandas DataFrame. This is fine for local/educational volumes
but does NOT scale to cluster-sized data; ``MAX_TRAIN_ROWS`` caps the
collected rows (deterministic down-sampling with a fixed seed) and the
limit is recorded in the run metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MAX_TRAIN_ROWS, TEST_FRACTION, VALIDATION_FRACTION
from ..features import assemble_feature_frame

TARGET = "is_delayed"
_TRAIN_COLUMNS = [
    "flight_date",
    "carrier",
    "origin",
    "destination",
    "sched_dep_minute",
    "sched_duration_minutes",
    "distance",
    "arrival_delay_minutes",
    "cancelled",
    "diverted",
    "is_delayed",
    "hist_carrier_delay_rate",
    "hist_carrier_avg_delay",
    "hist_route_delay_rate",
    "hist_route_avg_delay",
    "hist_origin_delay_rate",
    "hist_dest_delay_rate",
    "hist_route_flights",
    "weather_temperature_c",
    "weather_wind_speed_kmh",
    "weather_precipitation_mm",
    "weather_visibility_km",
]


def load_curated_features(features_dir: Path) -> pd.DataFrame:
    """Read the curated feature table from the Parquet lake."""
    path = Path(features_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Curated feature table not found at {path}. "
            "Run the ETL step first (Data & ETL view, or scripts/run_etl.py)."
        )
    df = pd.read_parquet(path)
    if "is_delayed" not in df.columns:
        raise ValueError(
            "Curated feature table has no 'is_delayed' column — the data was "
            "ingested without arrival_delay_minutes, so it cannot be used for training."
        )
    df = df[[c for c in _TRAIN_COLUMNS if c in df.columns]]
    df["is_delayed"] = pd.to_numeric(df["is_delayed"], errors="coerce").astype(float)
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    return df


def eligible_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude cancelled, diverted, and target-less flights from training."""
    mask = (
        (pd.to_numeric(df["cancelled"], errors="coerce").fillna(1) == 0)
        & (pd.to_numeric(df["diverted"], errors="coerce").fillna(1) == 0)
        & df["is_delayed"].notna()
    )
    return df[mask].copy()


def cap_training_rows(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Deterministically down-sample when the feature table exceeds the cap."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    sampled = df.sample(n=max_rows, random_state=seed)
    return sampled.sort_values("flight_date").reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split by DATE into ordered train -> validation -> test periods.

    Splits are on unique dates (never mid-date), ordered in time; preprocessing
    and threshold tuning must be fitted on train/validation only.
    """
    dates = np.array(sorted(pd.to_datetime(df["flight_date"]).dt.normalize().unique()))
    n_dates = len(dates)
    if n_dates < 3:
        raise ValueError(
            f"Chronological split needs at least 3 distinct flight dates, got {n_dates}. "
            "Provide data covering more days."
        )
    n_val = max(1, int(round(n_dates * validation_fraction)))
    n_test = max(1, int(round(n_dates * test_fraction)))
    n_train = n_dates - n_val - n_test
    if n_train < 1:
        raise ValueError(
            "Validation/test fractions leave no training dates. "
            f"Got {n_dates} dates with fractions {validation_fraction}/{test_fraction}."
        )

    train_dates = dates[:n_train]
    val_dates = dates[n_train : n_train + n_val]
    test_dates = dates[n_train + n_val :]

    date_col = pd.to_datetime(df["flight_date"]).dt.normalize()
    train = df[date_col.isin(train_dates)].copy()
    val = df[date_col.isin(val_dates)].copy()
    test = df[date_col.isin(test_dates)].copy()

    ranges = {
        "train_start": str(pd.Timestamp(train_dates[0]).date()),
        "train_end": str(pd.Timestamp(train_dates[-1]).date()),
        "val_start": str(pd.Timestamp(val_dates[0]).date()),
        "val_end": str(pd.Timestamp(val_dates[-1]).date()),
        "test_start": str(pd.Timestamp(test_dates[0]).date()),
        "test_end": str(pd.Timestamp(test_dates[-1]).date()),
        "train_dates": int(n_train),
        "val_dates": int(n_val),
        "test_dates": int(n_test),
    }
    return train, val, test, ranges


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Return (X feature frame, y target, untouched frame) for a split."""
    y = pd.to_numeric(df["is_delayed"], errors="coerce").to_numpy()
    X = assemble_feature_frame(df)
    return X, y, df
