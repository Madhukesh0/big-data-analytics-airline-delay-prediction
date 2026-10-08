"""Training + prediction integration test on a small deterministic dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from skypredict.data.demo_data import SYNTHETIC_SOURCE_LABEL
from skypredict.data.schema import FORBIDDEN_FEATURE_COLUMNS
from skypredict.models.dataset import load_curated_features
from skypredict.models.predict import prepare_inference_frame, score_dataframe, score_record
from skypredict.models.train import train_all


def test_train_and_predict_end_to_end(tmp_path, demo_features):
    models_dir = tmp_path / "models"
    db_path = tmp_path / "metrics.db"

    output = train_all(
        demo_features,
        dataset_source=SYNTHETIC_SOURCE_LABEL,
        models=("baseline", "random_forest", "xgboost"),
        models_dir=models_dir,
        db_path=db_path,
    )
    # Real artifacts written
    assert (models_dir / output.run_id / "metadata.json").exists()
    assert (models_dir / "latest.json").exists()

    # Metrics must be real numbers, not placeholders
    for name in ("logistic_regression_baseline", "random_forest"):
        res = output.model_results[name]
        assert res["available"], res.get("error")
        m = res["metrics"]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["precision_delayed"] <= 1.0
        assert 0.0 <= m["recall_delayed"] <= 1.0
        assert 0.0 < m["threshold"] < 1.0
        assert m["n"] == output.metadata["sample_counts"]["test_rows"]
    # ROC-AUC present only when both classes exist in the test split
    rf = output.model_results["random_forest"]["metrics"]
    if rf["delayed_rate"] in (0.0, 1.0):
        assert rf["roc_auc"] is None

    # XGBoost: either trained with real metrics or unavailable with a reason
    xgb = output.model_results["xgboost"]
    if xgb["available"]:
        assert 0.0 <= xgb["metrics"]["accuracy"] <= 1.0
    else:
        assert "xgboost" in xgb["error"].lower()

    # Leakage: no post-flight column among the saved model features
    from joblib import load

    rf_bundle = load(models_dir / output.run_id / "random_forest.joblib")
    assert FORBIDDEN_FEATURE_COLUMNS.isdisjoint(set(rf_bundle["feature_list"]))

    # Model runs were logged to SQLite
    from skypredict.services import db

    runs = db.list_model_runs(db_path)
    assert len(runs) >= 2
    assert any(r["model_name"] == "random_forest" and r["status"] == "completed" for r in runs)


def test_predict_without_target_columns(tmp_path, demo_features):
    """Inference must work on rows that carry no post-flight/target fields."""
    models_dir = tmp_path / "models"
    output = train_all(
        demo_features,
        dataset_source=SYNTHETIC_SOURCE_LABEL,
        models=("random_forest",),
        models_dir=models_dir,
        db_path=tmp_path / "metrics.db",
    )
    bundle_path = models_dir / output.run_id / "random_forest.joblib"
    assert bundle_path.exists()

    from joblib import load

    bundle = load(bundle_path)
    assert FORBIDDEN_FEATURE_COLUMNS.isdisjoint(set(bundle["feature_list"]))

    record = {
        "flight_date": "2026-06-15",
        "carrier": "AA",
        "origin": "ATL",
        "destination": "ORD",
        "scheduled_departure": 1830,  # HHMM
        "distance": 720.0,
    }
    row = score_record(bundle, record)
    assert 0.0 <= row["delay_probability"] <= 1.0
    assert row["predicted_delayed"] in (0, 1)

    # Batch scoring of BTS-style rows without any target column
    batch = prepare_inference_frame(
        pd.DataFrame(
            {
                "FlightDate": ["2026-06-15", "2026-06-16"],
                "OP_UNIQUE_CARRIER": ["AA", "DL"],
                "ORIGIN": ["ATL", "DFW"],
                "DEST": ["ORD", "DEN"],
                "CRS_DEP_TIME": ["830", "1545"],
                "DISTANCE": [720.0, 660.0],
            }
        )
    )
    assert batch["sched_dep_minute"].tolist() == [510.0, 945.0]
    scored = score_dataframe(bundle, batch)
    assert len(scored) == 2
    assert set(scored["predicted_delayed"]).issubset({0, 1})


def test_duration_model_exists_and_is_conditional(tmp_path, demo_features):
    models_dir = tmp_path / "models"
    output = train_all(
        demo_features,
        dataset_source=SYNTHETIC_SOURCE_LABEL,
        models=("random_forest",),
        models_dir=models_dir,
        db_path=tmp_path / "metrics.db",
    )
    dm = output.metadata["duration_model"]
    assert dm["available"], dm.get("note")
    assert dm["conditional_on_delayed"] is True
    m = dm["metrics"]
    assert m["mae"] is not None and m["mae"] > 0
    assert m["rmse"] >= m["mae"]
    assert (Path(dm["artifact"])).exists()


def test_curated_feature_loader_roundtrip(tmp_path, demo_features):
    from skypredict.pipeline.etl_pd import run_etl_pandas
    from skypredict.data.demo_data import generate_demo_data

    flights, weather = generate_demo_data(days=14, flights_per_day=25, seed=11)
    src = tmp_path / "demo_flights.csv"
    flights_out = flights.copy()
    flights_out["flight_date"] = flights_out["flight_date"].dt.strftime("%Y-%m-%d")
    flights_out["scheduled_departure"] = flights_out["sched_dep_minute"].map(
        lambda m: int(m) // 60 * 100 + int(m) % 60)
    flights_out["scheduled_arrival"] = flights_out["sched_arr_minute"].map(
        lambda m: int(m) // 60 * 100 + int(m) % 60)
    flights_out = flights_out.rename(columns={"sched_duration_minutes": "scheduled_duration_minutes"})
    flights_out = flights_out.drop(columns=["sched_dep_minute", "sched_arr_minute", "is_delayed"])
    flights_out.to_csv(src, index=False)
    wsrc = tmp_path / "demo_weather.csv"
    weather_out = weather.copy()
    weather_out["observation_date"] = weather_out["observation_date"].dt.strftime("%Y-%m-%d")
    weather_out.to_csv(wsrc, index=False)

    result = run_etl_pandas(
        src, SYNTHETIC_SOURCE_LABEL, weather_csv=wsrc,
        quarantine_dir=tmp_path / "q", flights_dir=tmp_path / "f",
        features_dir=tmp_path / "feat", weather_dir=tmp_path / "w",
    )
    assert result.engine == "pandas_fallback"
    assert result.counts["rows_read"] == len(flights)
    assert result.weather_used is True
    features = load_curated_features(tmp_path / "feat")
    assert "is_delayed" in features.columns
    assert "hist_route_delay_rate" in features.columns
    assert "weather_temperature_c" in features.columns
    assert features["is_delayed"].notna().sum() > 0
