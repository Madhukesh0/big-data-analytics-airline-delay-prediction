"""SQLite repository tests."""

from __future__ import annotations

from skypredict.services import db


def test_pipeline_model_and_prediction_logging(tmp_path):
    db_path = tmp_path / "metrics.db"
    db.init_db(db_path)

    run_id = db.insert_pipeline_run(
        db_path,
        started_at="2026-10-07T10:00:00+00:00",
        finished_at="2026-10-07T10:02:00+00:00",
        status="completed",
        source_type="synthetic_demo",
        engine="spark",
        counts={"rows_read": 7200, "rows_accepted": 7000, "rows_duplicate": 50,
                "rows_quarantined": 150, "rows_cancelled": 80, "rows_diverted": 20},
    )
    assert run_id == 1

    model_id = db.insert_model_run(
        db_path,
        started_at="2026-10-07T10:03:00+00:00",
        status="completed",
        model_name="random_forest",
        model_version="20261007-1-42",
        dataset_source="synthetic_demo",
        date_ranges={"train_start": "2026-01-01", "train_end": "2026-02-20",
                     "val_start": "2026-02-21", "val_end": "2026-02-28",
                     "test_start": "2026-03-01", "test_end": "2026-03-10"},
        sample_counts={"train_rows": 5000, "val_rows": 1000, "test_rows": 1000},
        threshold=0.42,
        metrics={"accuracy": 0.83, "f1_delayed": 0.61},
        artifact_path="models/x/random_forest.joblib",
    )
    assert model_id == 1

    pred_id = db.insert_prediction(
        db_path,
        source="single_form",
        flight_date="2026-04-01",
        carrier="AA",
        origin="ATL",
        destination="ORD",
        scheduled_departure=830,
        model_name="random_forest",
        model_version="20261007-1-42",
        delay_probability=0.37,
        predicted_delayed=0,
        expected_delay_minutes=6.2,
    )
    assert pred_id == 1

    pipelines = db.list_pipeline_runs(db_path)
    assert pipelines[0]["rows_read"] == 7200
    assert pipelines[0]["source_type"] == "synthetic_demo"
    assert pipelines[0]["error_summary"] is None

    models = db.list_model_runs(db_path)
    assert models[0]["threshold"] == 0.42
    assert '"accuracy": 0.83' in models[0]["metrics_json"].replace(" ", "").replace(",", ", ") or \
           models[0]["metrics_json"].startswith("{")
    assert models[0]["train_rows"] == 5000

    preds = db.list_predictions(db_path)
    assert preds[0]["carrier"] == "AA"
    assert preds[0]["delay_probability"] == 0.37


def test_failed_run_error_summary_recorded(tmp_path):
    db_path = tmp_path / "metrics.db"
    db.init_db(db_path)
    db.insert_pipeline_run(
        db_path, started_at="2026-10-07T11:00:00+00:00", status="failed",
        source_type="uploaded_csv", error_summary="ValueError('bad csv')",
    )
    runs = db.list_pipeline_runs(db_path)
    assert runs[0]["status"] == "failed"
    assert "bad csv" in runs[0]["error_summary"]
