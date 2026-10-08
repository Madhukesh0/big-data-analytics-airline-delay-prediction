"""Leakage-prevention and point-in-time-safety tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skypredict.data.schema import FORBIDDEN_FEATURE_COLUMNS
from skypredict.features import (
    FEATURES,
    add_historical_features_pandas,
    add_weather_features_pandas,
    assemble_feature_frame,
)


def test_no_post_flight_column_in_feature_list():
    assert FORBIDDEN_FEATURE_COLUMNS.isdisjoint(set(FEATURES))


def test_assemble_never_selects_post_flight_columns():
    df = pd.DataFrame(
        {
            "flight_date": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "carrier": ["AA", "DL"],
            "origin": ["ATL", "DFW"],
            "destination": ["ORD", "DEN"],
            "sched_dep_minute": [510.0, 915.0],
            "arrival_delay_minutes": [90.0, 5.0],    # post-flight
            "departure_delay_minutes": [60.0, 2.0],  # post-flight
            "cancelled": [0, 0],
            "is_delayed": [1.0, 0.0],
            "distance": [720.0, 660.0],
        }
    )
    X = assemble_feature_frame(df)
    assert list(X.columns) == FEATURES
    assert FORBIDDEN_FEATURE_COLUMNS.isdisjoint(set(X.columns))
    assert X["dep_hour"].tolist() == [8.0, 15.0]
    assert X["route"].tolist() == ["ATL-ORD", "DFW-DEN"]


def test_missing_feature_inputs_become_nan():
    df = pd.DataFrame(
        {
            "carrier": ["AA"],
            "origin": ["ATL"],
            "destination": ["ORD"],
            "flight_date": pd.to_datetime(["2026-03-01"]),
            "sched_dep_minute": [510],
        }
    )
    X = assemble_feature_frame(df)
    assert list(X.columns) == FEATURES
    assert X["distance"].isna().all()
    assert X["hist_route_delay_rate"].isna().all()
    assert X["weather_temperature_c"].isna().all()


def test_historical_aggregates_exclude_same_and_future_dates():
    """Route delay rate on 2026-03-03 must only use 03-01/03-02 flights."""
    df = pd.DataFrame(
        {
            "flight_date": pd.to_datetime(
                ["2026-03-01", "2026-03-01", "2026-03-02", "2026-03-03", "2026-03-03"]
            ),
            "carrier": ["AA"] * 5,
            "origin": ["ATL"] * 5,
            "destination": ["ORD"] * 5,
            "sched_dep_minute": [510, 600, 510, 510, 600],
            "cancelled": [0, 0, 0, 0, 0],
            "diverted": [0, 0, 0, 0, 0],
            "arrival_delay_minutes": [90.0, -5.0, 25.0, 60.0, 20.0],
            "is_delayed": [1.0, 0.0, 1.0, 1.0, 1.0],
        }
    )
    feats = add_historical_features_pandas(df)
    mar2 = feats[feats["flight_date"] == "2026-03-02"].iloc[0]
    mar3 = feats[feats["flight_date"] == "2026-03-03"].iloc[0]
    assert mar2["hist_route_delay_rate"] == 0.5          # 1 of 2 flights on 03-01
    assert mar2["hist_route_avg_delay"] == 90.0          # only the delayed one
    assert mar3["hist_route_delay_rate"] == 2 / 3        # 03-01 + 03-02
    assert mar3["hist_route_flights"] == 3


def test_future_records_do_not_change_earlier_features():
    """The core point-in-time guarantee: appending future flights must not
    change any feature of earlier flights."""
    rng = np.random.default_rng(0)
    base = pd.DataFrame(
        {
            "flight_date": pd.date_range("2026-03-01", periods=5, freq="D").repeat(4),
            "carrier": rng.choice(["AA", "DL"], 20),
            "origin": rng.choice(["ATL", "DFW"], 20),
            "destination": rng.choice(["ORD", "DEN"], 20),
            "sched_dep_minute": rng.integers(300, 1200, 20).astype(float),
            "cancelled": np.zeros(20),
            "diverted": np.zeros(20),
            "arrival_delay_minutes": rng.uniform(-10, 120, 20).round(1),
        }
    )
    base["is_delayed"] = (base["arrival_delay_minutes"] >= 15).astype(float)

    future = base.iloc[:4].copy()
    future["flight_date"] = pd.Timestamp("2026-03-10")
    future["arrival_delay_minutes"] = 300.0
    future["is_delayed"] = 1.0
    extended = pd.concat([base, future], ignore_index=True)

    before = add_historical_features_pandas(base)
    after = add_historical_features_pandas(extended)
    early_cols = [c for c in after.columns if c.startswith("hist_")]
    pd.testing.assert_frame_equal(
        before[early_cols].reset_index(drop=True),
        after.iloc[: len(before)][early_cols].reset_index(drop=True),
    )


def test_weather_join_is_point_in_time_safe():
    df = pd.DataFrame(
        {
            "flight_date": pd.to_datetime(["2026-03-01", "2026-03-01"]),
            "carrier": ["AA", "AA"],
            "origin": ["ATL", "ATL"],
            "destination": ["ORD", "ORD"],
            "sched_dep_minute": [300.0, 900.0],  # cutoff 01:00 vs 13:00
        }
    )
    weather = pd.DataFrame(
        {
            "airport": ["ATL"],
            "observation_date": ["2026-03-01"],
            "observation_timestamp": ["2026-03-01 04:00:00"],  # after the 01:00 cutoff
            "temperature_c": [21.0],
            "wind_speed_kmh": [10.0],
            "precipitation_mm": [3.0],
            "visibility_km": [12.0],
        }
    )
    out = add_weather_features_pandas(df, weather)
    assert out["weather_temperature_c"].isna().iloc[0]      # obs after cutoff -> missing
    assert out["weather_temperature_c"].iloc[1] == 21.0     # obs before cutoff -> used

    # An observation timestamped in the FUTURE must never be used either.
    weather_future = weather.copy()
    weather_future["observation_timestamp"] = ["2026-03-02 04:00:00"]
    out2 = add_weather_features_pandas(df, weather_future)
    assert out2["weather_temperature_c"].isna().all()
