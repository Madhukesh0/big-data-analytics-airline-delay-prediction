"""Spark ETL tests. Skipped with a clear message when Java/PySpark is unavailable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.spark


def _spark_or_skip():
    pytest.importorskip("pyspark")
    from skypredict.pipeline.spark_session import check_spark_available

    ok, message = check_spark_available()
    if not ok:
        pytest.skip(f"Spark unavailable in this environment: {message}")
    from skypredict.pipeline.spark_session import get_spark

    return get_spark()


@pytest.fixture(scope="module")
def spark():
    spark = _spark_or_skip()
    yield spark
    spark.stop()


def _write_flights_csv(path: Path, n_days=8, per_day=12, seed=3) -> Path:
    from skypredict.data.demo_data import generate_demo_data

    flights, _w = generate_demo_data(days=n_days, flights_per_day=per_day, seed=seed)
    out = flights.copy()
    out["flight_date"] = pd.to_datetime(out["flight_date"]).dt.strftime("%Y-%m-%d")
    out["scheduled_departure"] = out["sched_dep_minute"].map(
        lambda m: int(m) // 60 * 100 + int(m) % 60)
    out["scheduled_arrival"] = out["sched_arr_minute"].map(
        lambda m: int(m) // 60 * 100 + int(m) % 60)
    out = out.drop(columns=["sched_dep_minute", "sched_arr_minute", "is_delayed"])
    out.to_csv(path, index=False)
    return path


def test_spark_etl_writes_curated_parquet(spark, tmp_path):
    from skypredict.pipeline.etl import run_etl_spark

    src = _write_flights_csv(tmp_path / "flights.csv")
    result, _features = run_etl_spark(
        spark, src, "synthetic_demo",
        quarantine_dir=tmp_path / "q", flights_dir=tmp_path / "f",
        features_dir=tmp_path / "feat", weather_dir=tmp_path / "w",
    )
    counts = result.counts
    assert counts["rows_read"] == 8 * 12
    assert counts["rows_accepted"] > 0
    assert (tmp_path / "feat").exists()
    features = pd.read_parquet(tmp_path / "feat")
    assert "is_delayed" in features.columns
    assert "hist_route_delay_rate" in features.columns
    assert {"year", "month"} <= set(pd.read_parquet(tmp_path / "f").columns)


def test_spark_matches_pandas_validation_counts(spark, tmp_path):
    from conftest import make_bts_csv
    from skypredict.data.validate import validate_and_clean
    from skypredict.pipeline.etl import run_etl_spark, standardize_spark

    src = make_bts_csv(tmp_path / "bts.csv")
    raw_pd = pd.read_csv(src, dtype=str)
    _a, _q, pd_counts = validate_and_clean(raw_pd)

    raw = spark.read.option("header", True).csv(str(src))
    _acc, _q2, spark_counts = standardize_spark(raw)
    assert spark_counts["rows_read"] == pd_counts.rows_read
    assert spark_counts["rows_accepted"] == pd_counts.accepted
    assert spark_counts["rows_duplicate"] == pd_counts.duplicates
    assert spark_counts["rows_quarantined"] == pd_counts.quarantined
    assert spark_counts["rows_cancelled"] == pd_counts.cancelled
    assert spark_counts["rows_diverted"] == pd_counts.diverted


def test_spark_features_point_in_time_safe_and_match_pandas(spark, tmp_path):
    from skypredict.features import add_historical_features_pandas
    from skypredict.pipeline.etl import add_historical_features_spark

    base = pd.DataFrame(
        {
            "flight_date": pd.date_range("2026-03-01", periods=4, freq="D").repeat(3),
            "carrier": ["AA", "DL", "UA"] * 4,
            "origin": ["ATL"] * 12,
            "destination": ["ORD"] * 12,
            "sched_dep_minute": [510.0, 600.0, 700.0] * 4,
            "cancelled": np.zeros(12),
            "diverted": np.zeros(12),
            "arrival_delay_minutes": [30.0, -5.0, 10.0, 90.0, 0.0, 25.0, 60.0, -8.0, 15.0, 5.0, 20.0, 120.0],
        }
    )
    base["is_delayed"] = (base["arrival_delay_minutes"] >= 15).astype("int")

    future = pd.DataFrame(
        {
            "flight_date": pd.to_datetime(["2026-03-20"] * 3),
            "carrier": ["AA", "DL", "UA"],
            "origin": ["ATL"] * 3,
            "destination": ["ORD"] * 3,
            "sched_dep_minute": [510.0, 600.0, 700.0],
            "cancelled": np.zeros(3),
            "diverted": np.zeros(3),
            "arrival_delay_minutes": [300.0, 300.0, 300.0],
            "is_delayed": np.ones(3, dtype="int"),
        }
    )

    def compute(frame):
        sdf = spark.createDataFrame(frame)
        out = add_historical_features_spark(sdf).toPandas()
        return out.sort_values(["flight_date", "sched_dep_minute"]).reset_index(drop=True)

    before = compute(base)
    after = compute(pd.concat([base, future], ignore_index=True))

    hist_cols = [c for c in before.columns if c.startswith("hist_")]
    pd.testing.assert_frame_equal(
        before[hist_cols], after.iloc[: len(before)][hist_cols], check_dtype=False
    )

    ref = add_historical_features_pandas(base).sort_values(
        ["flight_date", "sched_dep_minute"]).reset_index(drop=True)
    for col in ("hist_carrier_delay_rate", "hist_route_delay_rate"):
        np.testing.assert_allclose(
            before[col].to_numpy(), ref[col].to_numpy(), rtol=1e-6, atol=1e-9
        )


def test_spark_weather_join_matches_pandas(spark, tmp_path):
    from skypredict.pipeline.etl import add_weather_features_spark
    from skypredict.features import add_weather_features_pandas

    flights = pd.DataFrame(
        {
            "flight_date": pd.to_datetime(["2026-03-01", "2026-03-01"]),
            "carrier": ["AA", "AA"],
            "origin": ["ATL", "ATL"],
            "destination": ["ORD", "ORD"],
            "sched_dep_minute": [300.0, 900.0],
        }
    )
    weather = pd.DataFrame(
        {
            "airport": ["ATL"],
            "observation_date": pd.to_datetime(["2026-03-01"]),
            "observation_timestamp": pd.to_datetime(["2026-03-01 04:00:00"]),
            "temperature_c": [21.0],
            "wind_speed_kmh": [10.0],
            "precipitation_mm": [3.0],
            "visibility_km": [12.0],
        }
    )

    sdf = spark.createDataFrame(flights)
    wdf = spark.createDataFrame(weather)
    out_spark = add_weather_features_spark(sdf, wdf).toPandas().sort_values(
        "sched_dep_minute").reset_index(drop=True)
    out_pd = add_weather_features_pandas(flights, weather).sort_values(
        "sched_dep_minute").reset_index(drop=True)

    # obs at 04:00 is after the 01:00 cutoff of the first flight and before
    # the 13:00 cutoff of the second one — in BOTH engines.
    assert np.isnan(out_spark["weather_temperature_c"].iloc[0])
    assert out_spark["weather_temperature_c"].iloc[1] == 21.0
    assert out_pd["weather_temperature_c"].isna().iloc[0]
    assert out_pd["weather_temperature_c"].iloc[1] == 21.0
