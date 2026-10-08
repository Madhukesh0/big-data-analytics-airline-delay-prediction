"""Feature definitions shared by training and inference.

The prediction-time assumption is: a prediction is made TWO HOURS BEFORE
SCHEDULED DEPARTURE. Only fields knowable at that moment may be used as
features. Historical aggregates are point-in-time safe: they use records
from dates STRICTLY EARLIER than the flight being scored.

Forbidden post-flight columns (arrival delay, actual times, cancellation
outcomes, ...) are never selected by :func:`assemble_feature_frame`, so they
cannot leak into the model input even if present in the source frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_CATEGORICAL = ["carrier", "origin", "destination", "route"]

FEATURE_NUMERIC = [
    "dep_hour",
    "month",
    "day_of_week",
    "is_weekend",
    "distance",
    "sched_duration_minutes",
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

FEATURES = FEATURE_CATEGORICAL + FEATURE_NUMERIC

# Grouping keys -> output prefix for historical aggregates.
HIST_GROUPS: dict[str, list[str]] = {
    "carrier": ["carrier"],
    "route": ["origin", "destination"],
    "origin": ["origin"],
    "destination": ["destination"],
}
HIST_PREFIX: dict[str, str] = {
    "carrier": "hist_carrier_",
    "route": "hist_route_",
    "origin": "hist_origin_",
    "destination": "hist_dest_",
}

# Weather columns in source files -> canonical feature names.
WEATHER_SOURCE_MAP: dict[str, str] = {
    "temperature_c": "weather_temperature_c",
    "wind_speed_kmh": "weather_wind_speed_kmh",
    "precipitation_mm": "weather_precipitation_mm",
    "visibility_km": "weather_visibility_km",
}
WEATHER_COLUMNS = list(WEATHER_SOURCE_MAP.values())

# Inputs assemble_feature_frame reads from a standardized frame.
_FEATURE_INPUT_COLUMNS = [
    "carrier",
    "origin",
    "destination",
    "flight_date",
    "sched_dep_minute",
    "sched_duration_minutes",
    "distance",
    "hist_carrier_delay_rate",
    "hist_carrier_avg_delay",
    "hist_route_delay_rate",
    "hist_route_avg_delay",
    "hist_origin_delay_rate",
    "hist_dest_delay_rate",
    "hist_route_flights",
] + WEATHER_COLUMNS

PREDICTION_LEAD_MINUTES = 120  # 2 hours before scheduled departure


def _upper_category(series: pd.Series) -> np.ndarray:
    """Uppercased object array; missing values stay NaN."""
    values = series.astype("string").str.strip().str.upper()
    return np.where(values.isna(), np.nan, values.astype(object).to_numpy())


def assemble_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model input matrix (FEATURES columns, fixed order).

    Accepts any frame that contains some of the feature inputs (e.g. a
    curated feature table, or a standardized inference frame). Columns the
    caller did not provide become NaN and are imputed downstream by the
    fitted preprocessing pipeline. Post-flight columns are never selected.
    """
    out = pd.DataFrame(index=df.index)

    if "carrier" in df.columns:
        out["carrier"] = _upper_category(df["carrier"])
    if "origin" in df.columns:
        out["origin"] = _upper_category(df["origin"])
    if "destination" in df.columns:
        out["destination"] = _upper_category(df["destination"])
    if "origin" in out.columns and "destination" in out.columns:
        out["route"] = [
            f"{o}-{d}" if isinstance(o, str) and isinstance(d, str) else np.nan
            for o, d in zip(out["origin"], out["destination"])
        ]

    if "sched_dep_minute" in df.columns:
        minutes = pd.to_numeric(df["sched_dep_minute"], errors="coerce")
        out["dep_hour"] = (minutes // 60).astype(float)

    if "flight_date" in df.columns:
        fdate = pd.to_datetime(df["flight_date"], errors="coerce")
        out["month"] = fdate.dt.month.astype(float)
        out["day_of_week"] = (fdate.dt.dayofweek + 1).astype(float)  # Mon=1..Sun=7
        out["is_weekend"] = (fdate.dt.dayofweek >= 5).astype(float)

    for col in ("distance", "sched_duration_minutes"):
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    for col in _FEATURE_INPUT_COLUMNS:
        if col in _FEATURE_INPUT_COLUMNS and col not in out.columns and col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    for col in FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return out[FEATURES]


def cutoff_timestamp(flight_date: pd.Timestamp, sched_dep_minute: float) -> pd.Timestamp:
    """The prediction cutoff: scheduled departure minus 2 hours."""
    return flight_date + pd.Timedelta(minutes=float(sched_dep_minute) - PREDICTION_LEAD_MINUTES)


# ---------------------------------------------------------------------------
# Point-in-time-safe historical aggregates (pandas implementation)
# ---------------------------------------------------------------------------

def add_historical_features_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Add hist_* aggregate features using only STRICTLY EARLIER dates.

    For every group (carrier / route / origin / destination) the aggregates
    at date D summarize all eligible flights with date < D. Flights on the
    same date D are excluded, so adding future records can never change the
    features of an earlier flight.
    """
    out = df.copy().reset_index(drop=True)
    if out.empty:
        for prefix in HIST_PREFIX.values():
            for stat in ("flights", "delay_rate", "avg_delay"):
                out[prefix + stat] = np.nan
        return out

    if "is_delayed" not in out.columns:
        arr = pd.to_numeric(out["arrival_delay_minutes"], errors="coerce")
        has_outcome = arr.notna() & (out["cancelled"] == 0) & (out["diverted"] == 0)
        out["is_delayed"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out.loc[has_outcome, "is_delayed"] = (arr[has_outcome] >= 15).astype("int64")

    out["_date_ord"] = (pd.to_datetime(out["flight_date"]) - pd.Timestamp("1970-01-01")).dt.days
    eligible = (
        (out["cancelled"] == 0)
        & (out["diverted"] == 0)
        & out["is_delayed"].notna()
    )
    flag_numeric = pd.to_numeric(out["is_delayed"], errors="coerce").astype(float)
    out["_flag"] = np.where(eligible, flag_numeric, np.nan)
    out["_delay_min"] = np.where(
        (flag_numeric == 1).to_numpy(),
        pd.to_numeric(out["arrival_delay_minutes"], errors="coerce").astype(float).to_numpy(),
        np.nan,
    )

    for group_name, keys in HIST_GROUPS.items():
        prefix = HIST_PREFIX[group_name]
        per_day = (
            out[keys + ["_date_ord", "_flag", "_delay_min"]]
            .groupby(keys + ["_date_ord"], dropna=False, as_index=False)
            .agg(
                flag_sum=("_flag", "sum"),
                flag_cnt=("_flag", "count"),
                delay_sum=("_delay_min", "sum"),
                delay_cnt=("_delay_min", "count"),
            )
            .sort_values(keys + ["_date_ord"])
            .reset_index(drop=True)
        )
        # Cumulative sums, then shift by one date: the value stored at date D
        # therefore covers dates strictly before D.
        grouped = per_day.groupby(keys, dropna=False, sort=False)
        cum = grouped[["flag_sum", "flag_cnt", "delay_sum", "delay_cnt"]].cumsum()
        for col in cum.columns:
            per_day[col] = cum[col]
        shifted = per_day.groupby(keys, dropna=False, sort=False)[
            ["flag_sum", "flag_cnt", "delay_sum", "delay_cnt"]
        ].shift(1)
        for col in shifted.columns:
            per_day[col] = shifted[col]

        per_day = per_day.rename(columns={"_date_ord": "_asof_ord"})
        per_day[prefix + "flights"] = per_day["flag_cnt"]
        per_day[prefix + "delay_rate"] = per_day["flag_sum"] / per_day["flag_cnt"].replace(0, np.nan)
        per_day[prefix + "avg_delay"] = per_day["delay_sum"] / per_day["delay_cnt"].replace(0, np.nan)

        out = out.merge(
            per_day[keys + ["_asof_ord", prefix + "flights", prefix + "delay_rate", prefix + "avg_delay"]],
            left_on=keys + ["_date_ord"],
            right_on=keys + ["_asof_ord"],
            how="left",
        ).drop(columns=["_asof_ord"])

    out = out.drop(columns=["_date_ord", "_flag", "_delay_min"])
    for prefix in HIST_PREFIX.values():
        for stat in ("flights", "delay_rate", "avg_delay"):
            col = prefix + stat
            if col not in out.columns:
                out[col] = np.nan
    return out


def add_weather_features_pandas(df: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join weather observations available at the prediction cutoff.

    Weather rows whose observation timestamp is after the flight's cutoff
    (departure minus 2 hours) are treated as missing — future observations
    are never used.
    """
    out = df.copy().reset_index(drop=True)
    for col in WEATHER_COLUMNS:
        out[col] = np.nan
    if weather is None or weather.empty or "origin" not in out.columns:
        return out

    w = weather.copy()
    w["observation_timestamp"] = pd.to_datetime(w["observation_timestamp"], errors="coerce")
    w["observation_date"] = pd.to_datetime(w["observation_date"], errors="coerce").dt.normalize()
    w = w.rename(columns=WEATHER_SOURCE_MAP)

    out["_flight_dt"] = pd.to_datetime(out["flight_date"], errors="coerce").dt.normalize()
    minutes = pd.to_numeric(out["sched_dep_minute"], errors="coerce")
    cutoff = out["_flight_dt"] + pd.to_timedelta(minutes - PREDICTION_LEAD_MINUTES, unit="m")
    cutoff = cutoff.reset_index(drop=True)

    merged = out.drop(columns=WEATHER_COLUMNS).merge(
        w,
        left_on=["origin", "_flight_dt"],
        right_on=["airport", "observation_date"],
        how="left",
    )
    usable = (
        merged["observation_timestamp"].notna()
        & (merged["observation_timestamp"] <= cutoff)
    )
    for col in WEATHER_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = np.where(usable, merged[col], np.nan)

    merged = merged.drop(columns=["_flight_dt", "airport", "observation_date", "observation_timestamp"])
    return merged
