"""Simulated live-flight feed (NOT a real Kafka stream or flight API).

This module implements a lightweight producer/consumer simulation: eligible
flight records are replayed in scheduled-time order through the prediction
function, one event at a time. Every prediction is yielded to the caller and
logged to SQLite. The dashboard labels this view explicitly as a simulation.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import METRICS_DB_PATH
from ..models.predict import prepare_inference_frame, score_dataframe
from . import db


def simulate_stream(
    flights_df: pd.DataFrame,
    bundle: dict,
    *,
    n_events: int = 50,
    sleep_seconds: float = 0.0,
    duration_bundle: dict | None = None,
    db_path: Path = METRICS_DB_PATH,
):
    """Replay eligible flights through the predictor, yielding event dicts.

    Events are emitted in scheduled-time order (flight_date, then scheduled
    departure minute). Each yielded event is also logged to the SQLite
    ``predictions`` table.
    """
    db.init_db(db_path)
    prepared = prepare_inference_frame(flights_df)

    order_cols = ["flight_date", "sched_dep_minute"]
    prepared = prepared.dropna(subset=["flight_date", "sched_dep_minute", "carrier",
                                       "origin", "destination"])
    if prepared.empty:
        return
    prepared = prepared.sort_values(order_cols).head(max(0, int(n_events)))

    scored = score_dataframe(bundle, prepared, duration_bundle)
    for i in range(len(scored)):
        row = scored.iloc[i]
        dep_minute = int(row.get("sched_dep_minute", 0) or 0)
        event = {
            "seq": i + 1,
            "flight_date": pd.Timestamp(row["flight_date"]).strftime("%Y-%m-%d"),
            "carrier": row.get("carrier"),
            "origin": row.get("origin"),
            "destination": row.get("destination"),
            # minutes since midnight rendered back as a HHMM value
            "scheduled_departure": dep_minute // 60 * 100 + dep_minute % 60,
            "delay_probability": float(row["delay_probability"]),
            "predicted_delayed": int(row["predicted_delayed"]),
        }
        if "expected_delay_minutes_if_delayed" in scored.columns:
            event["expected_delay_minutes_if_delayed"] = float(
                row["expected_delay_minutes_if_delayed"]
            )
        yield event

        db.insert_prediction(
            db_path,
            source="live_simulation",
            flight_date=event["flight_date"],
            carrier=event["carrier"],
            origin=event["origin"],
            destination=event["destination"],
            scheduled_departure=event["scheduled_departure"],
            model_name=bundle.get("model_name"),
            model_version=bundle.get("model_version"),
            delay_probability=event["delay_probability"],
            predicted_delayed=event["predicted_delayed"],
            expected_delay_minutes=event.get("expected_delay_minutes_if_delayed"),
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
