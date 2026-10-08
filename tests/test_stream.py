"""Simulated stream tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skypredict.services import db
from skypredict.services.stream_sim import simulate_stream


def _stub_bundle():
    class StubModel:
        def predict_proba(self, X):
            p = np.linspace(0.05, 0.95, len(X)) if len(X) else np.array([])
            return np.column_stack([1 - p, p])

    class Preprocessor:
        def transform(self, X):
            return X

    return {
        "model": StubModel(),
        "preprocessor": Preprocessor(),
        "threshold": 0.5,
        "model_name": "stub",
        "model_version": "stub-1",
    }


def test_stream_replays_in_scheduled_order_and_logs(tmp_path):
    bundle = _stub_bundle()
    flights = pd.DataFrame(
        {
            "flight_date": pd.to_datetime(
                ["2026-03-02", "2026-03-01", "2026-03-01", "2026-03-03"]
            ),
            "carrier": ["AA", "DL", "UA", "AA"],
            "origin": ["ATL", "ATL", "DFW", "ATL"],
            "destination": ["ORD", "ORD", "DEN", "LAX"],
            "sched_dep_minute": [510.0, 600.0, 420.0, 480.0],
        }
    )

    events = list(
        simulate_stream(flights, bundle, n_events=4, db_path=tmp_path / "m.db")
    )
    assert [e["seq"] for e in events] == [1, 2, 3, 4]
    # scheduled-time order: 03-01 420, 03-01 600, 03-02 510, 03-03 480
    assert [e["scheduled_departure"] for e in events] == [700, 1000, 830, 800]
    assert events[0]["carrier"] == "UA"
    assert events[2]["carrier"] == "AA"

    preds = db.list_predictions(tmp_path / "m.db")
    assert len(preds) == 4
    assert all(p["source"] == "live_simulation" for p in preds)
    assert preds[0]["model_name"] == "stub"


def test_stream_respects_n_events(tmp_path):
    bundle = _stub_bundle()
    flights = pd.DataFrame(
        {
            "flight_date": pd.date_range("2026-03-01", periods=10, freq="D"),
            "carrier": ["AA"] * 10,
            "origin": ["ATL"] * 10,
            "destination": ["ORD"] * 10,
            "sched_dep_minute": [600.0] * 10,
        }
    )
    events = list(simulate_stream(flights, bundle, n_events=3, db_path=tmp_path / "m.db"))
    assert len(events) == 3
    assert 0.0 <= events[0]["delay_probability"] <= 1.0
