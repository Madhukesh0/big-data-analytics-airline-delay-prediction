"""Spark ETL: ingestion, validation, dedup, curated Parquet output, and
point-in-time-safe feature engineering.

Each ingestion run REPLACES the curated dataset (full refresh), so reruns
never accumulate duplicate output. Malformed rows are quarantined to CSV
with an explicit reason; nothing is discarded silently.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from ..config import (
    CURATED_FEATURES_DIR,
    CURATED_FLIGHTS_DIR,
    CURATED_WEATHER_DIR,
    QUARANTINE_DIR,
    DELAY_THRESHOLD_MINUTES,
)
from ..data.schema import COLUMN_SYNONYMS, REQUIRED_FOR_INGEST
from ..features import (
    HIST_GROUPS,
    HIST_PREFIX,
    PREDICTION_LEAD_MINUTES,
    WEATHER_SOURCE_MAP,
)
from .common import EtlResult


def _canonical_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, synonyms in COLUMN_SYNONYMS.items():
        for syn in synonyms:
            lookup[syn.strip().lower()] = canonical
    return lookup


def map_columns_spark(df):
    """Rename recognized source columns to canonical names (first wins)."""
    from pyspark.sql import functions as F

    lookup = _canonical_lookup()
    new_names: list[str] = []
    assigned: set[str] = set()
    for col in df.columns:
        target = lookup.get(str(col).strip().lower())
        if target and target not in assigned:
            new_names.append(target)
            assigned.add(target)
        else:
            new_names.append(col)
    renamed = df.toDF(*new_names)
    for canonical in assigned:
        renamed = renamed.withColumnRenamed(
            canonical, canonical
        )  # no-op, keeps API symmetric
    return renamed


def _blank_to_null(col):
    from pyspark.sql import functions as F

    return F.when(F.trim(col) == "", None).otherwise(F.trim(col))


def _hhmm_minutes(col):
    """Parse a BTS HHMM string column into minutes since midnight (int).

    Handles dropped leading zeros ("830" -> 08:30), "HH:MM" strings, the
    2400-midnight convention, and returns null for anything invalid.
    """
    from pyspark.sql import functions as F

    s = F.regexp_replace(F.trim(col.cast("string")), r"\.0+$", "")
    s = F.regexp_replace(s, ":", "")
    base = F.when(s.rlike("^[0-9]{1,4}$"), F.lpad(s, 4, "0"))
    hh = F.substring(base, 1, 2).cast("int")
    mm = F.substring(base, 3, 2).cast("int")
    return (
        F.when(s == "2400", F.lit(0))
        .when(base.isNotNull() & (hh <= 23) & (mm <= 59), hh * 60 + mm)
        .otherwise(F.lit(None).cast("int"))
    )


def _flight_date_expr(df):
    from pyspark.sql import functions as F

    s = F.trim(F.col("flight_date").cast("string"))
    return F.coalesce(
        F.to_date(s, "yyyy-MM-dd"),
        F.to_date(s, "yyyy/MM/dd"),
        F.to_date(s, "M/d/yyyy"),
        F.to_date(s, "MM/dd/yyyy"),
    )


def _add_optional_defaults(df):
    """Ensure optional columns exist so downstream expressions are uniform."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    optional = [
        "scheduled_arrival",
        "scheduled_duration_minutes",
        "arrival_delay_minutes",
        "cancelled",
        "diverted",
        "distance",
        "flight_number",
    ]
    for col in optional:
        if col not in df.columns:
            df = df.withColumn(col, F.lit(None).cast(StringType()))
    return df


def standardize_spark(df, require_target: bool = False):
    """Parse, validate, quarantine, and deduplicate a raw Spark DataFrame.

    Returns ``(accepted, quarantine, counts_dict)`` where counts_dict uses
    the same keys as the pandas IngestCounts.
    """
    from pyspark.sql import functions as F

    df = map_columns_spark(df)
    missing = [c for c in REQUIRED_FOR_INGEST if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) for ingestion: {missing}. "
            f"Columns found: {sorted(df.columns)}. See README for supported names."
        )
    df = _add_optional_defaults(df)

    rows_read = df.count()

    # Parse typed values (raw string columns are kept for reason detection).
    df = (
        df.withColumn("_flight_date", _flight_date_expr(df))
        .withColumn("_dep", _hhmm_minutes(F.col("scheduled_departure")))
        .withColumn("_arr", _hhmm_minutes(F.col("scheduled_arrival")))
        .withColumn("_raw_arr_delay", F.trim(F.col("arrival_delay_minutes")))
        .withColumn("_arr_delay", F.col("arrival_delay_minutes").cast("double"))
        .withColumn("_dur_raw", F.col("scheduled_duration_minutes").cast("double"))
        .withColumn("_dist", F.when(F.col("distance").cast("double") >= 0, F.col("distance").cast("double")))
        .withColumn("_cancelled", F.when(F.col("cancelled").cast("double") != 0, F.lit(1)).otherwise(F.lit(0)))
        .withColumn("_diverted", F.when(F.col("diverted").cast("double") != 0, F.lit(1)).otherwise(F.lit(0)))
    )

    df = df.withColumn(
        "_duration",
        F.when(F.col("_dur_raw") > 0, F.col("_dur_raw")).otherwise(
            F.when(F.col("_dep").isNotNull() & F.col("_arr").isNotNull(), F.col("_arr") - F.col("_dep") + F.when(F.col("_arr") - F.col("_dep") <= 0, F.lit(1440)).otherwise(F.lit(0)))
        ),
    )

    reason = (
        F.when(F.col("_flight_date").isNull(), F.lit("invalid or missing flight_date"))
        .when(F.col("carrier").isNull() | (F.trim(F.col("carrier")) == ""), F.lit("missing carrier"))
        .when(F.col("origin").isNull() | (F.trim(F.col("origin")) == ""), F.lit("missing origin"))
        .when(F.col("destination").isNull() | (F.trim(F.col("destination")) == ""), F.lit("missing destination"))
        .when(F.col("_dep").isNull(), F.lit("invalid or missing scheduled_departure"))
        .when(
            F.col("_raw_arr_delay").isNotNull()
            & (F.col("_raw_arr_delay") != "")
            & F.col("_arr_delay").isNull(),
            F.lit("invalid arrival_delay_minutes (not numeric)"),
        )
    )
    if require_target:
        # The target requirement applies to COMPLETED flights only: cancelled
        # and diverted flights legitimately have no arrival delay, and are
        # kept (and counted separately) so training can exclude them explicitly.
        reason = reason.when(
            (F.col("_cancelled") == 0)
            & (F.col("_diverted") == 0)
            & F.col("_arr_delay").isNull(),
            F.lit("missing arrival_delay_minutes (required for training)"),
        )
    reason = reason.otherwise(F.lit(None).cast("string"))
    df = df.withColumn("quarantine_reason", reason)

    quarantine = df.filter(F.col("quarantine_reason").isNotNull())
    accepted = df.filter(F.col("quarantine_reason").isNull())

    # De-duplicate on the documented flight identity.
    identity = ["_flight_date", "carrier", "origin", "destination", "_dep", "flight_number"]
    accepted_total = accepted.count()
    accepted = accepted.dropDuplicates(identity)
    accepted_count = accepted.count()

    accepted = (
        accepted.select(
            F.col("_flight_date").alias("flight_date"),
            F.upper(_blank_to_null(F.col("carrier"))).alias("carrier"),
            F.upper(_blank_to_null(F.col("origin"))).alias("origin"),
            F.upper(_blank_to_null(F.col("destination"))).alias("destination"),
            F.col("_dep").alias("sched_dep_minute"),
            F.col("_arr").alias("sched_arr_minute"),
            F.col("_duration").cast("double").alias("sched_duration_minutes"),
            F.col("_arr_delay").alias("arrival_delay_minutes"),
            F.col("_cancelled").alias("cancelled"),
            F.col("_diverted").alias("diverted"),
            F.col("_dist").alias("distance"),
            F.col("flight_number").alias("flight_number"),
        )
        .withColumn(
            "is_delayed",
            F.when(
                (F.col("cancelled") == 0)
                & (F.col("diverted") == 0)
                & F.col("arrival_delay_minutes").isNotNull(),
                (F.col("arrival_delay_minutes") >= DELAY_THRESHOLD_MINUTES).cast("int"),
            ),
        )
    )

    quarantined_count = rows_read - accepted_total
    counts = {
        "rows_read": rows_read,
        "rows_accepted": accepted_count,
        "rows_duplicate": accepted_total - accepted_count,
        "rows_quarantined": quarantined_count,
        "rows_cancelled": accepted.filter(F.col("cancelled") == 1).count(),
        "rows_diverted": accepted.filter(F.col("diverted") == 1).count(),
        "rows_missing_target": accepted.filter(F.col("arrival_delay_minutes").isNull()).count(),
    }
    return accepted, quarantine, counts


def write_quarantine_spark(quarantine_df, quarantine_dir: Path, source_type: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = quarantine_dir / f"{source_type}-{stamp}.csv"
    quarantine_df.toPandas().to_csv(out_path, index=False)
    return out_path


def write_curated(spark_df, out_dir: Path, partition_cols: list[str]) -> None:
    """Full-refresh write: the directory is replaced on every run."""
    shutil.rmtree(out_dir, ignore_errors=True)
    spark_df.write.mode("overwrite").partitionBy(*partition_cols).parquet(str(out_dir))


def add_historical_features_spark(flights_df):
    """Point-in-time-safe aggregates via RANGE windows on the date ordinal.

    ``rangeBetween(unboundedPreceding, -1)`` restricts each aggregate to rows
    whose date ordinal is <= current - 1, i.e. STRICTLY EARLIER dates.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    df = flights_df.withColumn(
        "_date_ord", F.datediff(F.col("flight_date"), F.lit("1970-01-01"))
    )
    flag = F.when(
        (F.col("cancelled") == 0)
        & (F.col("diverted") == 0)
        & F.col("is_delayed").isNotNull(),
        F.col("is_delayed").cast("double"),
    )
    delay_min = F.when(F.col("is_delayed") == 1, F.col("arrival_delay_minutes").cast("double"))

    for group_name, keys in HIST_GROUPS.items():
        prefix = HIST_PREFIX[group_name]
        w = Window.partitionBy(*keys).orderBy("_date_ord").rangeBetween(
            Window.unboundedPreceding, -1
        )
        df = (
            df.withColumn(prefix + "flights", F.count(flag).over(w))
            .withColumn(prefix + "delay_rate", F.avg(flag).over(w))
            .withColumn(prefix + "avg_delay", F.avg(delay_min).over(w))
        )
    return df.drop("_date_ord")


def add_weather_features_spark(features_df, weather_df):
    """Left-join weather observations available at the prediction cutoff."""
    from pyspark.sql import functions as F

    cutoff_ts = F.to_timestamp(
        F.from_unixtime(
            F.unix_timestamp(F.col("flight_date").cast("timestamp"))
            + (F.col("sched_dep_minute").cast("long") - PREDICTION_LEAD_MINUTES) * 60
        )
    )
    w = weather_df.select(
        F.col("airport").alias("_w_airport"),
        F.col("observation_date").alias("_w_date"),
        F.col("observation_timestamp").alias("_w_ts"),
        *[F.col(src).alias(dst) for src, dst in WEATHER_SOURCE_MAP.items()],
    )
    joined = features_df.withColumn("_cutoff_ts", cutoff_ts).join(
        w,
        (F.col("origin") == F.col("_w_airport"))
        & (F.col("flight_date") == F.col("_w_date")),
        "left",
    )
    for dst in WEATHER_SOURCE_MAP.values():
        joined = joined.withColumn(
            dst, F.when(F.col("_w_ts") <= F.col("_cutoff_ts"), F.col(dst))
        )
    return joined.drop("_w_airport", "_w_date", "_w_ts", "_cutoff_ts")


def ingest_weather_spark(spark, weather_csv) -> "pyspark.sql.DataFrame":
    """Standardize a weather CSV (airport, observation timestamp, measures)."""
    from pyspark.sql import functions as F

    required = ["airport", "observation_date", "observation_timestamp",
                "temperature_c", "wind_speed_kmh", "precipitation_mm", "visibility_km"]
    df = spark.read.option("header", True).csv(str(weather_csv))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Weather CSV missing required column(s): {missing}. "
            f"Found: {sorted(df.columns)}."
        )
    return (
        df.withColumn("airport", F.upper(_blank_to_null(F.col("airport"))))
        .withColumn("observation_date", F.to_date(F.trim(F.col("observation_date"))))
        .withColumn("observation_timestamp", F.to_timestamp(F.trim(F.col("observation_timestamp"))))
        .withColumn("temperature_c", F.col("temperature_c").cast("double"))
        .withColumn("wind_speed_kmh", F.col("wind_speed_kmh").cast("double"))
        .withColumn("precipitation_mm", F.col("precipitation_mm").cast("double"))
        .withColumn("visibility_km", F.col("visibility_km").cast("double"))
        .filter(F.col("airport").isNotNull() & F.col("observation_date").isNotNull())
    )


def run_etl_spark(
    spark,
    source_csv: Path,
    source_type: str,
    *,
    require_target: bool = False,
    weather_csv: Path | None = None,
    quarantine_dir: Path = QUARANTINE_DIR,
    flights_dir: Path = CURATED_FLIGHTS_DIR,
    features_dir: Path = CURATED_FEATURES_DIR,
    weather_dir: Path = CURATED_WEATHER_DIR,
) -> tuple[EtlResult, "pyspark.sql.DataFrame"]:
    """Run the full Spark ETL. Returns the result plus the features DataFrame."""
    from pyspark.sql import functions as F

    raw = spark.read.option("header", True).csv(str(source_csv))
    accepted, quarantine, counts = standardize_spark(raw, require_target=require_target)

    quarantine_path = None
    if quarantine.take(1):
        quarantine_path = str(write_quarantine_spark(quarantine, quarantine_dir, source_type))

    accepted = accepted.withColumn("year", F.year("flight_date")).withColumn(
        "month", F.month("flight_date")
    )
    write_curated(accepted, flights_dir, ["year", "month"])

    flights = spark.read.parquet(str(flights_dir))
    features = add_historical_features_spark(flights)

    weather_used = False
    if weather_csv is not None:
        weather = ingest_weather_spark(spark, weather_csv)
        weather = weather.withColumn("year", F.year("observation_date")).withColumn(
            "month", F.month("observation_date")
        )
        write_curated(weather.select("*"), weather_dir, ["year", "month"])
        weather_flat = spark.read.parquet(str(weather_dir))
        features = add_weather_features_spark(features, weather_flat)
        weather_used = True

    features = features.withColumn("year", F.year("flight_date")).withColumn(
        "month", F.month("flight_date")
    )
    write_curated(features, features_dir, ["year", "month"])

    result = EtlResult(
        run_id=0,
        source_type=source_type,
        source_path=str(source_csv),
        engine="spark",
        counts=counts,
        quarantine_path=quarantine_path,
        flights_path=str(flights_dir),
        features_path=str(features_dir),
        weather_used=weather_used,
    )
    return result, features
