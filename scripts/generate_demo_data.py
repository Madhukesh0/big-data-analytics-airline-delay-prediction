"""Generate the deterministic synthetic demo dataset.

Example:
    python scripts/generate_demo_data.py --days 120 --flights-per-day 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skypredict.config import (  # noqa: E402
    DEMO_FLIGHTS_CSV,
    DEMO_WEATHER_CSV,
    DEMO_DAYS,
    DEMO_FLIGHTS_PER_DAY,
    RANDOM_SEED,
    ensure_runtime_dirs,
)
from skypredict.data.demo_data import SYNTHETIC_SOURCE_LABEL, generate_demo_data  # noqa: E402


def _to_hhmm(minute: float) -> int:
    minute = int(minute)
    return minute // 60 * 100 + minute % 60


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=DEMO_DAYS)
    parser.add_argument("--flights-per-day", type=int, default=DEMO_FLIGHTS_PER_DAY)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=DEMO_FLIGHTS_CSV)
    parser.add_argument("--weather-output", type=Path, default=DEMO_WEATHER_CSV)
    args = parser.parse_args()
    ensure_runtime_dirs()

    flights, weather = generate_demo_data(
        days=args.days, flights_per_day=args.flights_per_day, seed=args.seed
    )

    out = flights.copy()
    out["flight_date"] = pd.to_datetime(out["flight_date"]).dt.strftime("%Y-%m-%d")
    # CSV uses raw HHMM departure/arrival columns (canonical input format);
    # the pipeline derives sched_dep_minute/sched_arr_minute during ingestion.
    out["scheduled_departure"] = out["sched_dep_minute"].map(_to_hhmm)
    out["scheduled_arrival"] = out["sched_arr_minute"].map(_to_hhmm)
    out = out.rename(columns={"sched_duration_minutes": "scheduled_duration_minutes"})
    out = out.drop(columns=["sched_dep_minute", "sched_arr_minute", "is_delayed"])
    out.to_csv(args.output, index=False)

    weather_out = weather.copy()
    weather_out["observation_date"] = pd.to_datetime(weather_out["observation_date"]).dt.strftime("%Y-%m-%d")
    weather_out.to_csv(args.weather_output, index=False)

    delayed_rate = flights["arrival_delay_minutes"].ge(15).mean()
    print("=" * 64)
    print("SYNTHETIC DEMO DATA generated (not real airline data).")
    print(f"  source_type : {SYNTHETIC_SOURCE_LABEL}")
    print(f"  rows        : {len(flights)}  (days={args.days}, flights/day={args.flights_per_day}, seed={args.seed})")
    print(f"  date range  : {out['flight_date'].min()} .. {out['flight_date'].max()}")
    print(f"  delayed rate: {delayed_rate:.1%} (>= 15 min)")
    print(f"  flights CSV : {args.output}")
    print(f"  weather CSV : {args.weather_output}")
    return 0


if __name__ == "__main__":
    import pandas as pd  # placed here so --help stays fast

    raise SystemExit(main())
