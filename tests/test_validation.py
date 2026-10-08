"""Validation tests: required columns, quarantine, duplicates, counts."""

from __future__ import annotations

import pandas as pd
import pytest

from skypredict.data.schema import MissingColumnsError
from skypredict.data.validate import validate_and_clean


def _bts_frame():
    return pd.DataFrame(
        [
            ["2026-03-01", "AA", "ATL", "ORD", "830", "1130", -5.0, 0, 0, 720.0],
            ["2026-03-01", "AA", "ATL", "ORD", "0830", "1130", -5.0, 0, 0, 720.0],  # duplicate
            ["2026-03-01", "DL", "ATL", "ORD", "1015", "1315", 25.0, 0, 0, 720.0],
            ["not-a-date", "UA", "SFO", "JFK", "605", "1450", 15.0, 0, 0, 2586.0],
            ["2026-03-02", "AA", "ATL", "ORD", "9999", "1130", 0.0, 0, 0, 720.0],
            ["2026-03-02", "AS", "SEA", "LAX", "900", "1200", None, 1, 0, 954.0],   # cancelled
        ],
        columns=["FlightDate", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
                 "CRS_DEP_TIME", "CRS_ARR_TIME", "ARR_DELAY", "CANCELLED", "DIVERTED", "DISTANCE"],
    )


def test_missing_required_columns_raises():
    df = pd.DataFrame({"FlightDate": ["2026-03-01"], "carrier": ["AA"]})
    with pytest.raises(MissingColumnsError) as excinfo:
        validate_and_clean(df)
    assert "scheduled_departure" in str(excinfo.value)


def test_malformed_rows_quarantined_with_reason():
    accepted, quarantine, counts = validate_and_clean(_bts_frame())
    assert counts.rows_read == 6
    assert len(quarantine) == 2
    reasons = set(quarantine["quarantine_reason"])
    assert "invalid or missing flight_date" in reasons
    assert "invalid or missing scheduled_departure" in reasons
    assert counts.quarantined == 2


def test_duplicates_removed_by_flight_identity():
    accepted, _q, counts = validate_and_clean(_bts_frame())
    assert counts.duplicates == 1
    assert counts.accepted == 3  # 6 read - 2 quarantined - 1 duplicate


def test_cancelled_counted_and_targetless_flag():
    accepted, _q, counts = validate_and_clean(_bts_frame())
    assert counts.cancelled == 1
    assert counts.diverted == 0
    cancelled_row = accepted[accepted["cancelled"] == 1].iloc[0]
    assert pd.isna(cancelled_row["is_delayed"])


def test_delay_threshold_15_minutes():
    df = pd.DataFrame(
        [
            ["2026-03-01", "AA", "ATL", "ORD", "830", "1130", 14.9, 0, 0, 720.0],
            ["2026-03-01", "DL", "ATL", "ORD", "831", "1130", 15.0, 0, 0, 720.0],
            ["2026-03-01", "UA", "ATL", "ORD", "832", "1130", -10.0, 0, 0, 720.0],
        ],
        columns=["FlightDate", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
                 "CRS_DEP_TIME", "CRS_ARR_TIME", "ARR_DELAY", "CANCELLED", "DIVERTED", "DISTANCE"],
    )
    accepted, _q, _c = validate_and_clean(df)
    assert list(accepted.sort_values("sched_dep_minute")["is_delayed"]) == [0, 1, 0]


def test_require_target_quarantines_missing_delay():
    df = _bts_frame()
    df.loc[len(df)] = ["2026-03-06", "AA", "ATL", "ORD", "900", "1200", None, 0, 0, 720.0]
    accepted, quarantine, counts = validate_and_clean(df, require_target=True)
    # Only COMPLETED flights without a target are quarantined for the target
    # rule; cancelled/diverted flights stay accepted and are counted separately.
    completed = (accepted["cancelled"] == 0) & (accepted["diverted"] == 0)
    assert accepted.loc[completed, "arrival_delay_minutes"].notna().all()
    assert "missing arrival_delay_minutes (required for training)" in quarantine["quarantine_reason"].values
    assert counts.cancelled == 1  # cancelled flight preserved and reported


def test_inference_file_without_target_is_accepted():
    df = _bts_frame().drop(columns=["ARR_DELAY"])
    accepted, quarantine, counts = validate_and_clean(df)
    assert not accepted.empty
    assert "invalid arrival_delay_minutes (not numeric)" not in quarantine["quarantine_reason"].values
    assert accepted["is_delayed"].isna().all()
