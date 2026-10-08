"""Column mapping, HHMM parsing, and date parsing tests."""

from __future__ import annotations

import pandas as pd

from skypredict.data.schema import (
    compute_duration_minutes,
    map_columns,
    parse_flight_date,
    parse_hhmm,
)


def test_bts_column_mapping():
    df = pd.DataFrame(columns=[
        "FlightDate", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
        "CRS_DEP_TIME", "CRS_ARR_TIME", "ARR_DELAY", "CANCELLED", "DIVERTED", "DISTANCE",
    ])
    mapped, info = map_columns(df)
    for canonical in ["flight_date", "carrier", "origin", "destination",
                      "scheduled_departure", "scheduled_arrival",
                      "arrival_delay_minutes", "cancelled", "diverted", "distance"]:
        assert canonical in mapped.columns
    assert info.renamed["OP_UNIQUE_CARRIER"] == "carrier"


def test_canonical_names_recognized_as_is():
    df = pd.DataFrame(columns=["flight_date", "carrier", "origin", "destination",
                               "scheduled_departure", "arrival_delay_minutes"])
    mapped, info = map_columns(df)
    assert info.unrecognized == []
    assert set(info.recognized) == set(df.columns)


def test_parse_hhmm():
    assert parse_hhmm(830) == 510
    assert parse_hhmm("830") == 510
    assert parse_hhmm("0830") == 510
    assert parse_hhmm("8:30") == 510
    assert parse_hhmm(0) == 0
    assert parse_hhmm("0") == 0
    assert parse_hhmm("30") == 30          # BTS drops leading zeros
    assert parse_hhmm("2400") == 0         # midnight convention
    assert parse_hhmm(2359) == 1439
    assert parse_hhmm("9999") is None      # invalid hour
    assert parse_hhmm("8300") is None
    assert parse_hhmm("") is None
    assert parse_hhmm(None) is None
    assert parse_hhmm(float("nan")) is None
    assert parse_hhmm("abc") is None


def test_parse_flight_date():
    assert parse_flight_date("2026-03-01") == pd.Timestamp("2026-03-01")
    assert parse_flight_date("2026/03/01") == pd.Timestamp("2026-03-01")
    assert parse_flight_date("3/1/2026") == pd.Timestamp("2026-03-01")
    assert parse_flight_date("") is None
    assert parse_flight_date("garbage") is None
    assert parse_flight_date(None) is None


def test_duration_computation():
    assert compute_duration_minutes(510, 690, None) == 180.0
    assert compute_duration_minutes(1400, 100, None) == 140.0   # overnight wrap (20:00 -> 01:40)
    assert compute_duration_minutes(510, None, 200.0) == 200.0  # explicit wins
    assert compute_duration_minutes(None, 690, None) is None
