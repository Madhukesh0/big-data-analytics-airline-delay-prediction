"""Canonical flight schema, BTS column mapping, and time parsing helpers.

The canonical schema is the internal representation every other module works
with. Ingestion maps common BTS column names (and plain canonical names) onto
this schema, then parses dates and HHMM times robustly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS: list[str] = [
    "flight_date",               # date (datetime64)
    "carrier",                   # IATA code, e.g. AA
    "origin",                    # IATA airport code
    "destination",               # IATA airport code
    "scheduled_departure",       # raw HHMM value as provided by the source
    "scheduled_arrival",         # raw HHMM value (optional)
    "scheduled_duration_minutes",  # optional if departure+arrival given
    "arrival_delay_minutes",     # target; required for training, not inference
    "cancelled",                 # 0/1, optional (absent -> 0)
    "diverted",                  # 0/1, optional (absent -> 0)
    "distance",                  # miles, optional
    "flight_number",             # optional
]

# canonical name -> accepted source column names (BTS and plain synonyms).
# Matching is case-insensitive and ignores surrounding whitespace.
COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "flight_date": ("flight_date", "flightdate", "date"),
    "carrier": ("carrier", "op_unique_carrier", "op_carrier", "reporting_airline"),
    "origin": ("origin", "originairportid", "originairportseqid"),
    "destination": ("destination", "dest", "destairportid", "destairportseqid"),
    "scheduled_departure": ("scheduled_departure", "crs_dep_time", "crsdeptime"),
    "scheduled_arrival": ("scheduled_arrival", "crs_arr_time", "crsarrtime"),
    "scheduled_duration_minutes": ("scheduled_duration_minutes", "crselapsedtime"),
    "arrival_delay_minutes": ("arrival_delay_minutes", "arr_delay", "arr_delay_new", "arrdelay"),
    "cancelled": ("cancelled",),
    "diverted": ("diverted",),
    "distance": ("distance", "distancemiles"),
    "flight_number": ("flight_number", "op_carrier_fl_num", "fl_num"),
}

REQUIRED_FOR_INGEST = ["flight_date", "carrier", "origin", "destination", "scheduled_departure"]
REQUIRED_FOR_TRAINING = REQUIRED_FOR_INGEST + ["arrival_delay_minutes"]

# Post-flight / outcome columns that must never reach the feature set.
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "arrival_delay_minutes",
        "departure_delay_minutes",
        "arr_delay",
        "dep_delay",
        "arrdelay",
        "depdelay",
        "actual_departure",
        "actual_arrival",
        "dep_time",
        "arr_time",
        "wheels_off",
        "wheels_on",
        "cancelled",
        "diverted",
        "cancellation_code",
        "is_delayed",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
    }
)


class MissingColumnsError(ValueError):
    """Raised when a source file is missing columns required for ingestion."""


@dataclass
class ColumnMappingInfo:
    """Result of mapping raw source columns to canonical names."""

    renamed: dict[str, str] = field(default_factory=dict)   # raw -> canonical
    recognized: list[str] = field(default_factory=list)     # canonical names present
    unrecognized: list[str] = field(default_factory=list)   # source columns left as-is


def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, ColumnMappingInfo]:
    """Rename recognized source columns to canonical names (case-insensitive).

    Already-canonical names are recognized as themselves. Unrecognized columns
    are carried along untouched so callers can decide what to do with them.
    """
    info = ColumnMappingInfo()
    lookup: dict[str, str] = {}
    for canonical, synonyms in COLUMN_SYNONYMS.items():
        for syn in synonyms:
            lookup[syn.strip().lower()] = canonical

    renames: dict[str, str] = {}
    assigned: set[str] = set()
    for raw_col in df.columns:
        target = lookup.get(str(raw_col).strip().lower())
        if target and target not in assigned:
            renames[raw_col] = target
            assigned.add(target)
            info.renamed[str(raw_col)] = target

    out = df.rename(columns=renames)
    info.recognized = [c for c in CANONICAL_COLUMNS if c in out.columns]
    info.unrecognized = [str(c) for c in out.columns if c not in info.recognized]
    return out, info


def require_columns(df: pd.DataFrame, required: list[str], purpose: str) -> None:
    """Raise MissingColumnsError if any required canonical column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingColumnsError(
            f"Missing required column(s) for {purpose}: {missing}. "
            f"Recognized columns: {sorted(set(df.columns))}. See README for the "
            f"supported canonical/BTS column names."
        )


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

_TRAILING_ZEROS_RE = re.compile(r"\.0+$")


def parse_hhmm(value) -> int | None:
    """Parse a BTS HHMM time value into minutes since midnight.

    Accepts ints/floats/strings such as 830, "830", "0830", "8:30", "30".
    BTS drops leading zeros, so "30" means 00:30. The special value 2400 is
    treated as midnight (00:00). Returns None for missing or invalid values.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    s = _TRAILING_ZEROS_RE.sub("", s).replace(":", "")
    if not s.isdigit():
        return None
    if s == "2400":
        return 0
    if len(s) > 4:
        return None
    s4 = s.zfill(4)
    hh, mm = int(s4[:2]), int(s4[2:])
    if hh <= 23 and mm <= 59:
        return hh * 60 + mm
    return None


def parse_flight_date(value) -> pd.Timestamp | None:
    """Parse a flight date accepting common formats (ISO first, then US)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    ts = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(s, format="%Y/%m/%d", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.normalize()


def compute_duration_minutes(
    dep_minute: int | None, arr_minute: int | None, explicit: float | None
) -> float | None:
    """Scheduled duration in minutes.

    Uses the explicit duration when present; otherwise derives it from
    scheduled arrival - departure, wrapping past midnight.
    """
    if explicit is not None and not (isinstance(explicit, float) and math.isnan(explicit)):
        if explicit > 0:
            return float(explicit)
    if dep_minute is None or arr_minute is None:
        return None
    duration = arr_minute - dep_minute
    if duration <= 0:  # overnight flight (or identical times)
        duration += 1440
    return float(duration)
