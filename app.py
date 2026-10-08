"""SkyPredict — Airline Delay Prediction (Streamlit dashboard).

Course BCS714 D · Academic year 2026–2027.

An educational, local-first prototype. When synthetic demo data is used,
every view that shows data or metrics labels it as synthetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

from skypredict.config import (
    CURATED_FEATURES_DIR,
    DEMO_FLIGHTS_CSV,
    DEMO_WEATHER_CSV,
    METRICS_DB_PATH,
    MODELS_DIR,
    UPLOAD_DIR,
)
from skypredict.data.demo_data import SYNTHETIC_SOURCE_LABEL, generate_demo_data
from skypredict.data.validate import validate_and_clean
from skypredict.models import predict as predict_mod
from skypredict.models.dataset import load_curated_features
from skypredict.pipeline.runner import run_etl
from skypredict.pipeline.spark_session import check_spark_available
from skypredict.services import db
from skypredict.services.stream_sim import simulate_stream

st.set_page_config(
    page_title="SkyPredict — Airline Delay Prediction",
    page_icon="✈️",
    layout="wide",
)

FOOTER = (
    "SkyPredict — Airline Delay Prediction · Course **BCS714 D** · "
    "Academic year **2026–2027** · Educational, local-first prototype "
    "(not a production airline system)."
)


def _hhmm(minute) -> str:
    if minute is None or pd.isna(minute):
        return "?"
    minute = int(minute)
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _latest_completed_pipeline_run():
    runs = db.list_pipeline_runs(METRICS_DB_PATH, limit=25)
    for run in runs:
        if run["status"] == "completed":
            return run
    return None


def _curated_stats():
    try:
        df = load_curated_features(CURATED_FEATURES_DIR)
    except (FileNotFoundError, ValueError):
        return None
    delayed = df["is_delayed"]
    return {
        "rows": int(len(df)),
        "date_min": df["flight_date"].min(),
        "date_max": df["flight_date"].max(),
        "delayed_rate": float(delayed.mean()) if len(df) else 0.0,
        "cancelled": int(pd.to_numeric(df["cancelled"], errors="coerce").fillna(0).sum()),
        "diverted": int(pd.to_numeric(df["diverted"], errors="coerce").fillna(0).sum()),
        "eligible_rows": int(delayed.notna().sum()),
    }


def _model_status():
    latest = predict_mod.load_latest_run_info(MODELS_DIR)
    if not latest:
        return None
    runs = {m.get("run_id"): m for m in predict_mod.list_runs(MODELS_DIR)}
    meta = runs.get(latest.get("run_id"))
    return {"latest": latest, "metadata": meta}


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def view_overview() -> None:
    st.header("Overview")
    status = _model_status()
    stats = _curated_stats()
    run = _latest_completed_pipeline_run()

    col_data, col_model = st.columns(2)
    with col_data:
        st.subheader("Data")
        if run is None and stats is None:
            st.info("No data ingested yet. Go to **Data & ETL** to generate the "
                    "synthetic demo dataset or upload a CSV.")
        else:
            source_type = (run or {}).get("source_type", "unknown")
            if source_type == SYNTHETIC_SOURCE_LABEL:
                st.warning("Data source: **SYNTHETIC DEMO DATA** (generated, not real airline data).")
            else:
                st.write(f"Data source: `{source_type}` (file upload / local file)")
            if run is not None:
                st.write(f"Last ETL engine: **{run.get('engine')}** · run #{run.get('id')} · {run.get('finished_at')}")
            if stats is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Curated rows", stats["rows"])
                m2.metric("Eligible training rows", stats["eligible_rows"])
                m3.metric("Delayed rate", f"{stats['delayed_rate']:.1%}")
                st.write(
                    f"Date range: **{stats['date_min']:%Y-%m-%d} → {stats['date_max']:%Y-%m-%d}** · "
                    f"cancelled: {stats['cancelled']} · diverted: {stats['diverted']}"
                )

    with col_model:
        st.subheader("Model")
        if status is None:
            st.info("No model trained yet. Go to **Train & Evaluate** after ingesting data.")
        else:
            latest = status["latest"]
            meta = status["metadata"] or {}
            st.write(f"Latest run: `{latest.get('run_id')}` · trained {latest.get('trained_at')}")
            st.write(f"Dataset source: `{latest.get('dataset_source', 'unknown')}`")
            models = latest.get("models", [])
            st.write("Available models: " + (", ".join(f"`{m}`" for m in models) or "none"))
            if meta:
                mr = meta.get("models", {})
                rows = []
                for name, res in mr.items():
                    if res.get("available"):
                        m = res["metrics"]
                        rows.append({
                            "model": name,
                            "accuracy": round(m["accuracy"], 3),
                            "precision (delayed)": round(m["precision_delayed"], 3),
                            "recall (delayed)": round(m["recall_delayed"], 3),
                            "F1 (delayed)": round(m["f1_delayed"], 3),
                            "ROC-AUC": None if m["roc_auc"] is None else round(m["roc_auc"], 3),
                        })
                    else:
                        rows.append({"model": name, "accuracy": "unavailable",
                                     "precision (delayed)": res.get("error", "")[:60]})
                st.dataframe(pd.DataFrame(rows), width="stretch")
                if latest.get("dataset_source") == SYNTHETIC_SOURCE_LABEL:
                    st.caption("Metrics above are computed on SYNTHETIC DEMO DATA.")
    st.caption(FOOTER)


def view_data_etl() -> None:
    st.header("Data & ETL")

    spark_ok, spark_msg = check_spark_available()
    if spark_ok:
        st.success("Spark is available (PySpark, local mode).")
    else:
        st.warning(f"Spark unavailable — the ETL will use the clearly-labelled "
                   f"**pandas fallback**. Reason: {spark_msg}")

    left, right = st.columns(2)
    with left:
        st.subheader("Synthetic demo data")
        st.caption("Deterministic, seeded generator. Works fully offline. "
                   "All outputs are labelled SYNTHETIC DEMO DATA.")
        c1, c2 = st.columns(2)
        days = c1.number_input("Days", 7, 365, 120, key="demo_days")
        per_day = c2.number_input("Flights/day", 10, 500, 60, key="demo_per_day")
        if st.button("Generate demo data", type="primary"):
            with st.spinner("Generating synthetic demo data ..."):
                flights, weather = generate_demo_data(
                    days=int(days), flights_per_day=int(per_day)
                )
                demo = flights.copy()
                demo["flight_date"] = pd.to_datetime(demo["flight_date"]).dt.strftime("%Y-%m-%d")
                demo["scheduled_departure"] = demo["sched_dep_minute"].map(
                    lambda m: int(m) // 60 * 100 + int(m) % 60)
                demo["scheduled_arrival"] = demo["sched_arr_minute"].map(
                    lambda m: int(m) // 60 * 100 + int(m) % 60)
                demo = demo.rename(columns={"sched_duration_minutes": "scheduled_duration_minutes"})
                demo = demo.drop(columns=["sched_dep_minute", "sched_arr_minute", "is_delayed"])
                demo.to_csv(DEMO_FLIGHTS_CSV, index=False)
                w = weather.copy()
                w["observation_date"] = pd.to_datetime(w["observation_date"]).dt.strftime("%Y-%m-%d")
                w.to_csv(DEMO_WEATHER_CSV, index=False)
            st.success(f"Generated {len(flights)} synthetic flights → {DEMO_FLIGHTS_CSV}")

    with right:
        st.subheader("Upload flight CSV (+ optional weather CSV)")
        st.caption("Canonical or BTS column names are recognized "
                   "(FlightDate, OP_UNIQUE_CARRIER, ORIGIN, DEST, CRS_DEP_TIME, ARR_DELAY, ...). "
                   "See README for the full list.")
        flight_file = st.file_uploader("Flight CSV", type=["csv"], key="flights")
        weather_file = st.file_uploader("Optional weather CSV", type=["csv"], key="weather")
        if flight_file is not None:
            path = UPLOAD_DIR / flight_file.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(flight_file.getvalue())
            st.session_state["uploaded_flights"] = str(path)
            st.write(f"Saved to `{path}` ({flight_file.size:,} bytes)")
        if weather_file is not None:
            wpath = UPLOAD_DIR / weather_file.name
            wpath.write_bytes(weather_file.getvalue())
            st.session_state["uploaded_weather"] = str(wpath)
            st.write(f"Weather saved to `{wpath}`")

    # Quick validation of the selected source
    st.subheader("Validate source (quick check)")
    source_choice = st.radio(
        "Source to validate / ingest",
        ["Synthetic demo", "Uploaded file"],
        horizontal=True,
        key="source_choice",
    )
    source_csv = DEMO_FLIGHTS_CSV if source_choice == "Synthetic demo" else st.session_state.get("uploaded_flights")
    if source_choice == "Uploaded file" and not st.session_state.get("uploaded_flights"):
        st.info("Upload a flight CSV first, or switch to the synthetic demo source.")
        source_csv = None

    if source_csv is not None and Path(source_csv).exists():
        require_target = st.checkbox(
            "This file is TRAINING data (quarantine rows without arrival_delay_minutes)",
            value=(source_choice == "Synthetic demo"),
        )
        if st.button("Validate file"):
            raw = pd.read_csv(source_csv, dtype=str)
            accepted, quarantine, counts = validate_and_clean(raw, require_target=require_target)
            st.write(f"Recognized/standardized columns: `{sorted(accepted.columns)}`")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows read", counts.rows_read)
            c2.metric("Accepted", counts.accepted)
            c3.metric("Duplicates", counts.duplicates)
            c4.metric("Quarantined", counts.quarantined)
            if counts.quarantined_by_reason:
                st.write("Quarantine reasons:", counts.quarantined_by_reason)
            st.write(f"Cancelled: {counts.cancelled} · diverted: {counts.diverted} · "
                     f"missing target: {counts.missing_target}")
            if not quarantine.empty:
                st.dataframe(quarantine.head(10), width="stretch")

        st.subheader("Run Spark ingestion / feature processing")
        use_demo_weather = source_choice == "Synthetic demo" and DEMO_WEATHER_CSV.exists()
        run_weather = None
        if source_choice == "Synthetic demo" and use_demo_weather:
            run_weather = DEMO_WEATHER_CSV
            st.caption(f"Weather: synthetic demo weather will be used ({DEMO_WEATHER_CSV.name}).")
        elif st.session_state.get("uploaded_weather"):
            run_weather = Path(st.session_state["uploaded_weather"])
            st.caption(f"Weather: {run_weather}")

        if st.button("Run ETL", type="primary"):
            source_type = SYNTHETIC_SOURCE_LABEL if source_choice == "Synthetic demo" else "uploaded_csv"
            try:
                with st.spinner("Running ETL (this can take a while on first Spark start) ..."):
                    result = run_etl(
                        Path(source_csv),
                        source_type,
                        require_target=require_target,
                        weather_csv=run_weather,
                        engine="auto",
                        db_path=METRICS_DB_PATH,
                    )
                st.success(f"ETL completed — engine **{result.engine}**, run #{result.run_id}")
                st.json({
                    "counts": result.counts,
                    "flights_path": result.flights_path,
                    "features_path": result.features_path,
                    "quarantine_path": result.quarantine_path,
                    "weather_used": result.weather_used,
                })
            except Exception as exc:
                st.error(f"ETL failed: {exc}")
    st.caption(FOOTER)


def view_train_eval() -> None:
    st.header("Train & Evaluate")
    stats = _curated_stats()
    if stats is None or stats["eligible_rows"] == 0:
        st.info("No eligible training data yet. Ingest data in **Data & ETL** first "
                "(rows must include arrival_delay_minutes and not be cancelled/diverted).")
        st.caption(FOOTER)
        return

    status = _model_status()
    if status is None:
        st.info("No model trained yet — train one below.")
    else:
        st.caption(f"Latest model run: `{status['latest'].get('run_id')}` "
                   f"({status['latest'].get('trained_at')})")

    choices = st.multiselect(
        "Models to train & compare",
        ["baseline", "random_forest", "xgboost"],
        default=["baseline", "random_forest", "xgboost"],
    )
    if st.button("Start training and evaluation", type="primary") and choices:
        try:
            from skypredict.models.train import train_all

            features = load_curated_features(CURATED_FEATURES_DIR)
            source_type = (_latest_completed_pipeline_run() or {}).get("source_type", "unknown")
            with st.spinner("Training on the chronological train split; evaluating on the held-out test period ..."):
                output = train_all(features, dataset_source=source_type, models=tuple(choices))
            st.session_state["last_train_run_id"] = output.run_id
            st.success(f"Training complete — run `{output.run_id}`")

            ranges = output.metadata["date_ranges"]
            counts = output.metadata["sample_counts"]
            st.write(
                f"**Chronological splits** — Train: {ranges['train_start']} → {ranges['train_end']} "
                f"({counts['train_rows']} rows) · Validation: {ranges['val_start']} → {ranges['val_end']} "
                f"({counts['val_rows']} rows) · Test: {ranges['test_start']} → {ranges['test_end']} "
                f"({counts['test_rows']} rows)"
            )
            if source_type == SYNTHETIC_SOURCE_LABEL:
                st.warning("All metrics below are computed from a real model run on **SYNTHETIC DEMO DATA**.")
            if output.metadata.get("rows_capped_from"):
                st.caption(f"Training rows capped/sampled at {output.metadata['max_train_rows_cap']} "
                           f"(local-training memory limit).")

            for name, res in output.model_results.items():
                st.subheader(name)
                if not res.get("available"):
                    st.error(res.get("error", "unavailable"))
                    continue
                m = res["metrics"]
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Accuracy", f"{m['accuracy']:.3f}")
                k2.metric("Precision (delayed)", f"{m['precision_delayed']:.3f}")
                k3.metric("Recall (delayed)", f"{m['recall_delayed']:.3f}")
                k4.metric("F1 (delayed)", f"{m['f1_delayed']:.3f}")
                k5.metric("ROC-AUC", "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}")
                cm = m["confusion_matrix"]
                st.dataframe(
                    pd.DataFrame(
                        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                        index=["actual on-time", "actual delayed"],
                        columns=["pred on-time", "pred delayed"],
                    ),
                    width="stretch",
                )
                if name in ("random_forest", "xgboost"):
                    try:
                        bundle = predict_mod.load_bundle(MODELS_DIR, output.run_id, name)
                        importances = bundle["model"].feature_importances_
                        names = bundle["preprocessor"].get_feature_names_out()
                        imp = (
                            pd.DataFrame({"feature": names, "importance": importances})
                            .sort_values("importance", ascending=False)
                            .head(15)
                        )
                        st.bar_chart(imp.set_index("feature"))
                    except Exception as exc:
                        st.caption(f"Feature importance unavailable: {exc}")

            dm = output.metadata.get("duration_model", {})
            if dm.get("available"):
                st.subheader("duration_random_forest (delayed flights only)")
                dmm = dm["metrics"]
                k1, k2, k3 = st.columns(3)
                k1.metric("MAE (minutes)", f"{dmm['mae']:.1f}")
                k2.metric("RMSE (minutes)", f"{dmm['rmse']:.1f}")
                k3.metric("Test delayed rows", dmm["n"])
                st.caption("The duration model is CONDITIONAL on a flight being delayed; "
                           "it does not predict whether a flight is delayed.")
            elif dm.get("note"):
                st.info(dm["note"])
        except Exception as exc:
            st.error(f"Training failed: {exc}")
    st.caption(FOOTER)


def view_predictions() -> None:
    st.header("Predictions")
    status = _model_status()
    if status is None:
        st.info("No trained model. Load data (**Data & ETL**) and train (**Train & Evaluate**) first — "
                "SkyPredict never shows invented predictions.")
        st.caption(FOOTER)
        return

    latest = status["latest"]
    models = latest.get("models", [])
    if not models:
        st.error("Latest run has no usable models. Train again.")
        return
    model_name = st.selectbox("Model", models)
    dataset_note = ("**SYNTHETIC DEMO DATA**" if latest.get("dataset_source") == SYNTHETIC_SOURCE_LABEL
                    else f"`{latest.get('dataset_source')}`")
    st.caption(f"Model `{model_name}` (run `{latest['run_id']}`) trained on {dataset_note}.")

    tab_single, tab_record, tab_batch = st.tabs(["Single flight (form)", "Eligible record", "Batch CSV"])

    with tab_single:
        with st.form("single_flight"):
            c1, c2, c3, c4 = st.columns(4)
            fdate = c1.date_input("Flight date")
            carrier = c2.text_input("Carrier", "AA", max_chars=3).upper()
            origin = c3.text_input("Origin", "ATL", max_chars=3).upper()
            dest = c4.text_input("Destination", "ORD", max_chars=3).upper()
            c1b, c2b, c3b, c4b = st.columns(4)
            dep_time = c1b.time_input("Scheduled departure", pd.Timestamp("08:30:00").time())
            distance = c2b.number_input("Distance (miles, optional)", 0, 10000, 700)
            duration = c3b.number_input("Duration (minutes, optional)", 0, 800, 120)
            submitted = st.form_submit_button("Predict")
        if submitted:
            record = {
                "flight_date": fdate.strftime("%Y-%m-%d"),
                "carrier": carrier,
                "origin": origin,
                "destination": dest,
                "scheduled_departure": dep_time.hour * 100 + dep_time.minute,
                "distance": float(distance) if distance else None,
                "scheduled_duration_minutes": float(duration) if duration else None,
            }
            try:
                bundle = predict_mod.load_bundle(MODELS_DIR, latest["run_id"], model_name)
                duration_bundle = predict_mod.load_duration_bundle(MODELS_DIR, latest["run_id"])
                row = predict_mod.score_record(bundle, record, duration_bundle)
                k1, k2, k3 = st.columns(3)
                k1.metric("Delay probability", f"{row['delay_probability']:.1%}")
                k2.metric("Prediction", "DELAYED (≥15 min)" if row["predicted_delayed"] else "On time")
                if "expected_delay_minutes_if_delayed" in row and pd.notna(row.get("expected_delay_minutes_if_delayed")):
                    k3.metric("Expected delay if delayed", f"{row['expected_delay_minutes_if_delayed']:.0f} min")
                    st.caption(f"Probability-weighted expectation: "
                               f"{row.get('expected_delay_minutes', 0):.1f} minutes.")
                st.caption(f"Threshold applied: {row['applied_threshold']:.2f}. Historical "
                           f"aggregate features are unavailable for new flights and fall back "
                           f"to training medians (see README).")
                db.insert_prediction(
                    METRICS_DB_PATH,
                    source="single_form",
                    flight_date=record["flight_date"],
                    carrier=carrier,
                    origin=origin,
                    destination=dest,
                    scheduled_departure=record["scheduled_departure"],
                    model_name=bundle.get("model_name"),
                    model_version=bundle.get("model_version"),
                    delay_probability=float(row["delay_probability"]),
                    predicted_delayed=int(row["predicted_delayed"]),
                    expected_delay_minutes=(float(row["expected_delay_minutes"])
                                            if "expected_delay_minutes" in row and pd.notna(row.get("expected_delay_minutes")) else None),
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with tab_record:
        try:
            feats = load_curated_features(CURATED_FEATURES_DIR)
            sample = feats.dropna(subset=["flight_date"]).head(800).reset_index(drop=True)
            labels = [
                f"{r.flight_date:%Y-%m-%d} · {r.carrier} {r.origin}-{r.destination} · dep {_hhmm(r.sched_dep_minute)}"
                for r in sample.itertuples()
            ]
            idx = st.selectbox("Eligible record (from the curated feature table)", range(len(labels)),
                               format_func=lambda i: labels[i])
            if st.button("Score selected record"):
                record = sample.iloc[idx].to_dict()
                bundle = predict_mod.load_bundle(MODELS_DIR, latest["run_id"], model_name)
                duration_bundle = predict_mod.load_duration_bundle(MODELS_DIR, latest["run_id"])
                row = predict_mod.score_record(bundle, record, duration_bundle)
                k1, k2, k3 = st.columns(3)
                k1.metric("Delay probability", f"{row['delay_probability']:.1%}")
                k2.metric("Prediction", "DELAYED (≥15 min)" if row["predicted_delayed"] else "On time")
                if "expected_delay_minutes_if_delayed" in row and pd.notna(row.get("expected_delay_minutes_if_delayed")):
                    k3.metric("Expected delay if delayed", f"{row['expected_delay_minutes_if_delayed']:.0f} min")
                st.caption("Scoring uses this record's historical aggregate features as they were "
                           "computed by the ETL (point-in-time safe). The real outcome is not used "
                           "for prediction.")
        except Exception as exc:
            st.error(f"Could not load records: {exc}")

    with tab_batch:
        file = st.file_uploader("Batch CSV (no target column required)", type=["csv"], key="batch")
        if file is not None and st.button("Score batch"):
            try:
                raw = pd.read_csv(file, dtype=str)
                bundle = predict_mod.load_bundle(MODELS_DIR, latest["run_id"], model_name)
                duration_bundle = predict_mod.load_duration_bundle(MODELS_DIR, latest["run_id"])
                scored = predict_mod.score_dataframe(bundle, raw, duration_bundle)
                st.dataframe(scored, width="stretch")
                st.download_button(
                    "Download predictions CSV",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="skypredict_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Batch scoring failed: {exc}")
    st.caption(FOOTER)


def view_live_simulation() -> None:
    st.header("Live Simulation")
    st.warning("This is a **SIMULATION** implemented with a Python queue-style replay — "
               "it is NOT a real Kafka stream, live flight API, or production feed.")

    status = _model_status()
    stats = _curated_stats()
    if status is None or stats is None:
        st.info("Requires ingested data and a trained model (see **Data & ETL** and **Train & Evaluate**).")
        st.caption(FOOTER)
        return

    latest = status["latest"]
    models = latest.get("models", [])
    if not models:
        st.error("No usable model in the latest run. Train a model first.")
        return
    model_name = st.selectbox("Model", models, key="sim_model")
    n_events = st.slider("Number of flights to replay", 5, 500, 50)
    pace = st.slider("Delay between events (seconds)", 0.0, 1.0, 0.0, 0.05)

    if st.button("Start simulated feed", type="primary"):
        try:
            feats = load_curated_features(CURATED_FEATURES_DIR)
            bundle = predict_mod.load_bundle(MODELS_DIR, latest["run_id"], model_name)
            duration_bundle = predict_mod.load_duration_bundle(MODELS_DIR, latest["run_id"])
            placeholder = st.empty()
            progress = st.progress(0.0)
            collected: list[dict] = []
            gen = simulate_stream(
                feats, bundle,
                n_events=n_events, sleep_seconds=pace,
                duration_bundle=duration_bundle, db_path=METRICS_DB_PATH,
            )
            for event in gen:
                collected.append(event)
                progress.progress(min(len(collected) / max(n_events, 1), 1.0))
                if len(collected) % 10 == 0 or len(collected) == n_events:
                    placeholder.dataframe(pd.DataFrame(collected).tail(10), width="stretch")
            progress.progress(1.0)
            st.success(f"Simulation finished: {len(collected)} events predicted and logged to SQLite.")
            if collected:
                df = pd.DataFrame(collected)
                k1, k2 = st.columns(2)
                k1.metric("Events", len(df))
                k2.metric("Predicted delayed", f"{df['predicted_delayed'].mean():.0%}")
                placeholder.dataframe(df.tail(15), width="stretch")
                st.download_button("Download simulation events CSV",
                                   data=df.to_csv(index=False).encode("utf-8"),
                                   file_name="skypredict_simulation.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
    st.caption(FOOTER)


def view_history() -> None:
    st.header("Pipeline and model history (SQLite)")
    tab_p, tab_m, tab_pred = st.tabs(["Pipeline runs", "Model runs", "Prediction logs"])

    with tab_p:
        runs = db.list_pipeline_runs(METRICS_DB_PATH)
        if runs:
            st.dataframe(pd.DataFrame(runs), width="stretch")
        else:
            st.info("No pipeline runs logged yet.")

    with tab_m:
        runs = db.list_model_runs(METRICS_DB_PATH)
        if runs:
            df = pd.DataFrame(runs)
            if "metrics_json" in df.columns:
                df["metrics_json"] = df["metrics_json"].str.slice(0, 160)
            st.dataframe(df, width="stretch")
        else:
            st.info("No model runs logged yet.")

    with tab_pred:
        preds = db.list_predictions(METRICS_DB_PATH, limit=300)
        if preds:
            st.dataframe(pd.DataFrame(preds), width="stretch")
        else:
            st.info("No predictions logged yet. Use the Predictions view or the Live Simulation.")
    st.caption(FOOTER)


VIEWS = {
    "Overview": view_overview,
    "Data & ETL": view_data_etl,
    "Train & Evaluate": view_train_eval,
    "Predictions": view_predictions,
    "Live Simulation": view_live_simulation,
    "History": view_history,
}

view = st.sidebar.radio("Navigation", list(VIEWS), key="nav")
st.sidebar.divider()
st.sidebar.caption(FOOTER)
VIEWS[view]()
