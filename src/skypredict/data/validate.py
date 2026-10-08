"""Row-level validation and standardization (pandas).

This is the single pandas implementation of the ingestion rules. It is used
by the pandas fallback ETL, by the dashboard's quick "validate upload" step,
and by tests. The Spark ETL implements the same rules natively in
:mod:`skypredict.pipeline.etl` for large files.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import (
    REQUIRED_FOR_INGEST,
    REQUIRED_FOR_TRAINING,
    compute_duration_minutes,
    map_columns,
    parse_flight_date,
    parse_hhmm,
    require_columns,
)


@dataclass
class IngestCounts:
    """Row counts reported for every ingestion run."""

    rows_read: int = 0
    accepted: int = 0
    duplicates: int = 0
    quarantined: int = 0
    cancelled: int = 0
    diverted: int = 0
    missing_target: int = 0
    quarantined_by_reason: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "rows_read": self.rows_read,
            "rows_accepted": self.accepted,
            "rows_duplicate": self.duplicates,
            "rows_quarantined": self.quarantined,
            "rows_cancelled": self.cancelled,
            "rows_diverted": self.diverted,
            "rows_missing_target": self.missing_target,
            "quarantined_by_reason": dict(self.quarantined_by_reason),
        }


FLIGHT_IDENTITY = ["flight_date", "carrier", "origin", "destination", "sched_dep_minute", "flight_number"]


def _as_binary(series: pd.Series) -> pd.Series:
    """Coerce an optional 0/1 column; absent/malformed values become 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0).ne(0).astype("int64")


def standardize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, IngestCounts]:
    """Validate and standardize a flight DataFrame with canonical names.

    Returns ``(accepted, quarantine, counts)``.

    * ``accepted`` is standardized to internal column names with parsed dates
      and times (``sched_dep_minute`` etc.).
    * ``quarantine`` holds malformed rows plus a ``quarantine_reason`` column;
      nothing is discarded silently.
    """
    counts = IngestCounts(rows_read=len(df))
    df, _info = map_columns(df)
    require_columns(df, REQUIRED_FOR_INGEST, "ingestion")

    reasons: list[str | None] = []
    parsed_rows: list[dict] = []

    dep_raw = df["scheduled_departure"]
    arr_raw = df["scheduled_arrival"] if "scheduled_arrival" in df.columns else None
    dur_raw = (
        pd.to_numeric(df["scheduled_duration_minutes"], errors="coerce")
        if "scheduled_duration_minutes" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )
    delay_raw = (
        pd.to_numeric(df["arrival_delay_minutes"], errors="coerce")
        if "arrival_delay_minutes" in df.columns
        else pd.Series([np.nan] * len(df), index=df.index)
    )
    delay_raw_present = df["arrival_delay_minutes"] if "arrival_delay_minutes" in df.columns else None
    distance = (
        pd.to_numeric(df["distance"], errors="coerce")
        if "distance" in df.columns
        else pd.Series([np.nan] * len(df), index=df.index)
    )
    cancelled = _as_binary(df["cancelled"]) if "cancelled" in df.columns else pd.Series(0, index=df.index)
    diverted = _as_binary(df["diverted"]) if "diverted" in df.columns else pd.Series(0, index=df.index)
    flight_number = df["flight_number"] if "flight_number" in df.columns else pd.Series([None] * len(df), index=df.index)

    for i in range(len(df)):
        row = df.iloc[i]
        reason: str | None = None

        fdate = parse_flight_date(row.get("flight_date"))
        carrier = row.get("carrier")
        origin = row.get("origin")
        dest = row.get("destination")
        dep_min = parse_hhmm(dep_raw.iloc[i])
        arr_min = parse_hhmm(arr_raw.iloc[i]) if arr_raw is not None else None

        if fdate is None:
            reason = "invalid or missing flight_date"
        elif not str(carrier or "").strip():
            reason = "missing carrier"
        elif not str(origin or "").strip():
            reason = "missing origin"
        elif not str(dest or "").strip():
            reason = "missing destination"
        elif dep_min is None:
            reason = "invalid or missing scheduled_departure"
        elif delay_raw_present is not None:
            raw_delay = delay_raw_present.iloc[i]
            raw_str = str(raw_delay).strip()
            if raw_str and raw_str.lower() not in {"nan", "none", "null"} and pd.isna(delay_raw.iloc[i]):
                reason = "invalid arrival_delay_minutes (not numeric)"

        if reason is not None:
            reasons.append(reason)
            continue

        duration = compute_duration_minutes(dep_min, arr_min, dur_raw.iloc[i])
        parsed_rows.append(
            {
                "flight_date": fdate,
                "carrier": str(carrier).strip().upper(),
                "origin": str(origin).strip().upper(),
                "destination": str(dest).strip().upper(),
                "sched_dep_minute": dep_min,
                "sched_arr_minute": arr_min,
                "sched_duration_minutes": duration,
                "arrival_delay_minutes": None if pd.isna(delay_raw.iloc[i]) else float(delay_raw.iloc[i]),
                "cancelled": int(cancelled.iloc[i]),
                "diverted": int(diverted.iloc[i]),
                "distance": None if pd.isna(distance.iloc[i]) else float(distance.iloc[i]),
                "flight_number": None if pd.isna(flight_number.iloc[i]) else str(flight_number.iloc[i]),
            }
        )
        reasons.append(None)

    quarantine = df.copy()
    quarantine["quarantine_reason"] = reasons
    quarantine = quarantine[quarantine["quarantine_reason"].notna()]
    for reason in quarantine["quarantine_reason"]:
        counts.quarantined_by_reason[reason] = counts.quarantined_by_reason.get(reason, 0) + 1

    accepted = pd.DataFrame(parsed_rows)
    if accepted.empty:
        counts.quarantined = int(len(quarantine))
        return accepted, quarantine, counts

    # De-duplicate on a documented flight identity: date, carrier, origin,
    # destination, scheduled departure time and flight number (when present).
    before = len(accepted)
    identity = [c for c in FLIGHT_IDENTITY if not accepted[c].isna().all()]
    accepted = accepted.drop_duplicates(subset=identity, keep="first").reset_index(drop=True)
    counts.duplicates = before - len(accepted)

    counts.accepted = len(accepted)
    counts.quarantined = int(len(quarantine))
    counts.cancelled = int(accepted["cancelled"].sum())
    counts.diverted = int(accepted["diverted"].sum())
    counts.missing_target = int(accepted["arrival_delay_minutes"].isna().sum())

    accepted["is_delayed"] = _delay_flag(accepted)
    return accepted, quarantine, counts


def _delay_flag(df: pd.DataFrame) -> pd.Series:
    """is_delayed = 1 when arrival delay >= 15 minutes.

    Null for flights without a target (inference rows) and for
    cancelled/diverted flights, which are excluded from training.
    """
    threshold_ok = df["arrival_delay_minutes"] >= 15
    outcome_ok = (df["cancelled"] == 0) & (df["diverted"] == 0)
    flag = threshold_ok & outcome_ok & df["arrival_delay_minutes"].notna()
    result = pd.Series(pd.NA, index=df.index, dtype="Int64")
    result[df["arrival_delay_minutes"].notna() & outcome_ok] = flag.astype("int64")[
        df["arrival_delay_minutes"].notna() & outcome_ok
    ]
    return result


def eligible_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Training rows: not cancelled/diverted and with a valid target."""
    mask = (
        (df["cancelled"] == 0)
        & (df["diverted"] == 0)
        & df["arrival_delay_minutes"].notna()
        & df["is_delayed"].notna()
    )
    return df[mask].copy()


def validate_and_clean(df: pd.DataFrame, require_target: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, IngestCounts]:
    """Standardize a DataFrame and additionally enforce the target rule.

    With ``require_target=True`` (training data), rows without a valid
    ``arrival_delay_minutes`` are quarantined with an explicit reason.
    """
    accepted, quarantine, counts = standardize_frame(df)
    if require_target and not accepted.empty:
        # The target requirement applies to COMPLETED flights only: cancelled
        # and diverted flights legitimately have no arrival delay. They stay in
        # the accepted set so their counts are reported, and training excludes
        # them via :func:`eligible_for_training`.
        completed = (accepted["cancelled"] == 0) & (accepted["diverted"] == 0)
        has_target = accepted["arrival_delay_minutes"].notna()
        to_quarantine = completed & ~has_target
        missing = accepted[to_quarantine].drop(columns=["is_delayed"])
        missing["quarantine_reason"] = "missing arrival_delay_minutes (required for training)"
        quarantine = pd.concat([quarantine, missing], ignore_index=True)
        counts.quarantined_by_reason["missing arrival_delay_minutes (required for training)"] = int(
            to_quarantine.sum()
        )
        counts.quarantined += int(to_quarantine.sum())
        accepted = accepted[~to_quarantine].reset_index(drop=True)
        counts.accepted = len(accepted)
        accepted["is_delayed"] = _delay_flag(accepted)
    return accepted, quarantine, counts
