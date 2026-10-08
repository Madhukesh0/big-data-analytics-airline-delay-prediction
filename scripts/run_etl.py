"""Run the ingestion/feature ETL from the command line.

Examples:
    python scripts/run_etl.py --source demo
    python scripts/run_etl.py --source data/uploads/my_bts_file.csv --require-target
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        help="'demo' or a path to a flight CSV file")
    parser.add_argument("--weather", default=None, help="optional weather CSV path")
    parser.add_argument("--require-target", action="store_true",
                        help="quarantine rows without arrival_delay_minutes (training data)")
    parser.add_argument("--engine", choices=["auto", "spark", "pandas"], default="auto")
    args = parser.parse_args()

    from skypredict.config import DEMO_FLIGHTS_CSV, DEMO_WEATHER_CSV, METRICS_DB_PATH, ensure_runtime_dirs
    from skypredict.data.demo_data import SYNTHETIC_SOURCE_LABEL
    from skypredict.pipeline.runner import run_etl

    ensure_runtime_dirs()

    if args.source == "demo":
        source_csv = DEMO_FLIGHTS_CSV
        source_type = SYNTHETIC_SOURCE_LABEL
        if not source_csv.exists():
            print("Demo data not found. Generate it first with:")
            print("  python scripts/generate_demo_data.py")
            return 1
    else:
        source_csv = Path(args.source)
        source_type = "uploaded_csv"
        if not source_csv.exists():
            print(f"Flight CSV not found: {source_csv}")
            return 1

    weather_csv = DEMO_WEATHER_CSV if (args.weather == "demo" and DEMO_WEATHER_CSV.exists()) else (
        Path(args.weather) if args.weather else None
    )

    print(f"Running ETL (engine={args.engine}) on {source_csv} ...")
    result = run_etl(
        source_csv,
        source_type,
        require_target=args.require_target,
        weather_csv=weather_csv,
        engine=args.engine,
        db_path=METRICS_DB_PATH,
    )
    print(json.dumps({
        "run_id": result.run_id,
        "engine": result.engine,
        "counts": result.counts,
        "flights_path": result.flights_path,
        "features_path": result.features_path,
        "quarantine_path": result.quarantine_path,
        "weather_used": result.weather_used,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
