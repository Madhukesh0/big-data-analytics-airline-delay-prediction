"""Shared test fixtures. Tests run against the project's src/ package."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from skypredict.data.demo_data import generate_demo_data
from skypredict.features import add_historical_features_pandas, add_weather_features_pandas


@pytest.fixture(scope="session")
def demo_data():
    """Small deterministic synthetic dataset (flights + weather)."""
    return generate_demo_data(days=16, flights_per_day=30, seed=7)


@pytest.fixture(scope="session")
def demo_features(demo_data):
    """Synthetic flights with point-in-time-safe historical features + weather."""
    flights, weather = demo_data
    feats = add_historical_features_pandas(flights)
    feats = add_weather_features_pandas(feats, weather)
    return feats


def make_bts_csv(path: Path) -> Path:
    """Small BTS-style CSV exercising column mapping, HHMM parsing, quarantine."""
    rows = [
        # valid rows (note BTS drops leading zeros in HHMM)
        ["2026-03-01", "AA", "ATL", "ORD", "830", "1130", -5.0, 0, 0, 720.0],
        ["2026-03-01", "DL", "ATL", "ORD", "1015", "1315", 25.0, 0, 0, 720.0],
        ["2026-03-02", "AA", "ATL", "ORD", "830", "1130", 90.0, 0, 0, 720.0],
        ["2026-03-02", "WN", "DFW", "DEN", "1530", "1700", 5.0, 0, 0, 660.0],
        ["2026-03-03", "UA", "SFO", "JFK", "605", "1450", 15.0, 0, 0, 2586.0],
        # duplicate of the first flight (same identity) -> dropped
        ["2026-03-01", "AA", "ATL", "ORD", "0830", "1130", -5.0, 0, 0, 720.0],
        # malformed rows -> quarantined with reasons
        ["not-a-date", "AA", "ATL", "ORD", "830", "1130", 0.0, 0, 0, 720.0],
        ["2026-03-04", "AA", "ATL", "ORD", "9999", "1130", 0.0, 0, 0, 720.0],
        ["2026-03-04", "AA", "ATL", "ORD", "830", "1130", "late", 0, 0, 720.0],
        # cancelled / diverted rows (excluded from training, counted separately)
        ["2026-03-05", "AS", "SEA", "LAX", "900", "1200", "", 1, 0, 954.0],
        ["2026-03-05", "AS", "SEA", "MIA", "700", "1600", "", 0, 1, 2716.0],
    ]
    columns = ["FlightDate", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
               "CRS_DEP_TIME", "CRS_ARR_TIME", "ARR_DELAY", "CANCELLED", "DIVERTED", "DISTANCE"]
    df = pd.DataFrame(rows, columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
