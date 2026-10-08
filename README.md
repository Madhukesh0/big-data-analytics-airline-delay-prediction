# SkyPredict — Big Data Analytics for Airline Delay Prediction

**Course:** BCS714 D · **Academic year:** 2026–2027

SkyPredict is an educational, local-first prototype that uses historical flight data and
big-data processing (PySpark) to predict whether a flight will arrive **at least 15 minutes
late**, and — conditional on a delay — to estimate how long that delay will be. It ships with
a Streamlit dashboard, a partitioned Parquet data lake, local model training with real
calculated metrics, SQLite run logging, and an explicitly **simulated** live-flight feed.

> **This is not a production airline system.** It runs in local Spark mode, trains models
> locally (not on a cluster), and uses a simulation instead of a real flight API.

---

## 1. What it does (end-to-end workflow)

1. **Load** synthetic demo data, an uploaded CSV, or a local historical flight file.
2. **Validate & standardize** input (column mapping, date/HHMM parsing, de-duplication, quarantine).
3. **Process with PySpark** in local mode (`local[*]`), or a clearly-labelled pandas fallback.
4. **Write** raw/canonical and curated data to a partitioned **Parquet** data lake.
5. **Engineer features** that are point-in-time safe (no future or post-flight information).
6. **Train & compare** a logistic-regression baseline, a Random Forest, and XGBoost (when installed).
7. **Evaluate** on a **chronological** holdout period with real calculated metrics.
8. **Save** model artifacts and run metadata (`models/<run_id>/`).
9. **Predict** single flights and batches through the Streamlit dashboard.
10. **Simulate** a live-flight feed and log predictions.
11. **Log** pipeline runs, model runs, and predictions in **SQLite** (`data/metrics.db`).

---

## 2. Prediction definition and prediction-time assumption

- **Classification target:** `is_delayed = 1` when arrival delay is **≥ 15 minutes**, else `0`.
- **Regression target:** arrival delay in minutes for delayed flights.
  The predicted duration is **conditional on the flight being delayed**; it does not predict
  whether a delay occurs. The dashboard also shows a probability-weighted expectation
  (`P(delayed) × expected minutes if delayed`).
- **Prediction-time assumption:** every prediction is made **two hours before scheduled departure.**
- **Excluded from training:** cancelled and diverted flights (counted separately in ingestion stats).
- **Leakage prevention:** actual departure/arrival times, departure/arrival delay, cancellation
  or diversion outcome, and any target-derived value are **never** model inputs. The feature
  list is defined in `src/skypredict/features.py` and post-flight columns are blocked there.
- **Point-in-time-safe history:** carrier/route/origin/destination delay statistics use records
  from dates **strictly earlier** than the flight being scored. Tests assert that appending
  future flights does not change an earlier flight's features.
- **Boundary:** the Spark pipeline writes the lake, but training collects the feature table to
  the driver and fits scikit-learn/XGBoost **locally**. `SKYPREDICT_MAX_TRAIN_ROWS` caps the
  collected rows (deterministic sampling); the local mode is not distributed training.

---

## 3. Project layout

```
skypredict/
  app.py                      Streamlit dashboard (Overview, ETL, Train, Predict, Simulation, History)
  pyproject.toml              packaging + pytest config
  requirements.txt            dependencies
  src/skypredict/
    config.py                 paths, thresholds, Spark master, row caps
    features.py               feature definitions + point-in-time-safe engineering
    data/
      schema.py               canonical schema, BTS column mapping, HHMM/date parsing
      validate.py             standardize, quarantine, de-duplicate, counts
      demo_data.py            deterministic SYNTHETIC demo generator
    pipeline/
      spark_session.py        SparkSession + local Java discovery
      etl.py                  Spark ETL: parse, validate, Parquet, features, weather join
      etl_pd.py               pandas fallback ETL (same rules, same layout)
      runner.py               engine selection (auto/spark/pandas) + SQLite logging
    models/
      dataset.py              curated load, eligibility, chronological split, row cap
      preprocess.py           reusable ColumnTransformer (impute + one-hot)
      metrics.py              classification/regression metrics, threshold tuning
      train.py                baseline + Random Forest + XGBoost + duration regressor
      predict.py              load artifacts, single/batch prediction
    services/
      db.py                   SQLite repository (parameterized SQL)
      stream_sim.py           simulated producer/consumer feed
  scripts/
    generate_demo_data.py     generate the synthetic dataset
    run_etl.py                run ingestion/features from the CLI
    train_model.py            train + evaluate from the CLI
  tests/                      automated tests (see section 8)
  data/                       runtime: raw, uploads, lake (Parquet), quarantine, metrics.db
  models/                     runtime: model artifacts + metadata.json per run
  runtime/                    optional project-local Java 17 runtime (see section 5)
```

Runtime data and model artifacts are git-ignored.

---

## 4. Data sources and input schema

The real-data reference is the **U.S. Bureau of Transportation Statistics (BTS) Airline
On-Time Performance** dataset: <https://www.transtats.bts.gov/ONTIME/>

SkyPredict does **not** download BTS data automatically. To use real data, obtain a BTS CSV
(e.g. "Reporting Carrier On-Time Performance") and either upload it in the dashboard's
**Data & ETL** view or place it under `data/raw/` and run the ETL with `--source <path>`.

### Canonical schema

`flight_date`, `carrier`, `origin`, `destination`, `scheduled_departure`,
`scheduled_arrival` (or `scheduled_duration_minutes`), `arrival_delay_minutes`
(required for training, not for inference), `cancelled` (optional), `diverted` (optional),
`distance` (optional), plus optional weather fields with an observation timestamp.

### Recognized BTS column names

`FlightDate`, `YEAR`, `MONTH`, `DAY_OF_MONTH`, `OP_UNIQUE_CARRIER`, `OP_CARRIER`, `ORIGIN`,
`DEST`, `CRS_DEP_TIME`, `CRS_ARR_TIME`, `CRSELAPSEDTIME`, `ARR_DELAY`, `CANCELLED`,
`DIVERTED`, `DISTANCE`, `OP_CARRIER_FL_NUM`. BTS `HHMM` values are parsed tolerantly
(leading zeros dropped by BTS are handled: `830`, `0830`, `"8:30"`, and `2400` → midnight).

### Ingestion behaviour

- Required columns are validated; a missing one raises a clear error naming the column.
- Malformed rows (bad date, missing carrier/airport, invalid departure time, non-numeric delay)
  are written to `data/quarantine/` **with a reason per row** — never silently dropped.
- De-duplication uses a documented flight identity:
  `flight_date, carrier, origin, destination, scheduled departure, flight number`.
- Inference files may omit `arrival_delay_minutes`; training requires a valid target.
- The dashboard shows rows read / accepted / duplicate / quarantined / cancelled / diverted.

### Optional weather

An optional weather CSV with `airport`, `observation_date`, `observation_timestamp`,
`temperature_c`, `wind_speed_kmh`, `precipitation_mm`, `visibility_km`. Only observations
**at or before** the prediction cutoff are joined; if no weather data is supplied the pipeline
and dashboard work normally without it.

---

## 5. Installation

Requires **Python 3.9+** and, for the Spark path, **Java 17**.

```bash
cd skypredict
python -m venv .venv
# Windows (Git Bash / PowerShell):
.venv/Scripts/python.exe -m pip install -r requirements.txt
# macOS / Linux:
# .venv/bin/python -m pip install -r requirements.txt
```

### Java / PySpark requirements

PySpark needs a Java 17 runtime. SkyPredict looks for Java in this order:

1. `JAVA_HOME` if already set;
2. otherwise a project-local runtime under `runtime/jdk*` (it sets `JAVA_HOME` automatically).

If you have no system Java, place a JRE 17 in `runtime/` — e.g. download **Temurin JRE 17**
(build of OpenJDK) from <https://adoptium.net/temurin/releases/> and extract it so that
`runtime/jdk-17.x.y+z-jre/bin/java` exists. No system-wide change is made.

On **Windows**, writing Parquet also needs Hadoop's `winutils.exe`/`hadoop.dll`. If you place
them under `runtime/hadoop/bin/`, SkyPredict points `HADOOP_HOME` at them automatically
(no system-wide install).

**Without Java the project still works:** the ETL automatically uses the pandas fallback engine,
which applies the same validation/feature rules and writes the same Parquet layout. Every output
labels the engine, and the dashboard shows a warning so the distinction is never blurred.

---

## 6. Commands

All commands are run from the `skypredict/` directory with the virtualenv active.

```bash
# 1) Generate the deterministic SYNTHETIC demo dataset (works offline)
python scripts/generate_demo_data.py --days 120 --flights-per-day 60

# 2) Run ingestion + feature engineering (Spark when Java is available)
python scripts/run_etl.py --source demo
python scripts/run_etl.py --source demo --weather demo          # include demo weather
python scripts/run_etl.py --source data/raw/bts_2026_01.csv --require-target
python scripts/run_etl.py --source demo --engine pandas         # force the fallback

# 3) Train & evaluate (chronological holdout, real metrics)
python scripts/train_model.py --models all
python scripts/train_model.py --models random_forest

# 4) Run the test suite
python -m pytest tests -q
python -m pytest tests -q -m "not spark"      # skip Spark tests

# 5) Launch the dashboard
python -m streamlit run app.py
```

### Data-lake and artifact locations

| Path | Contents |
| --- | --- |
| `data/raw/` | generated demo data, local source files |
| `data/uploads/` | dashboard file uploads |
| `data/lake/curated/flights/` | standardized flights (Parquet, partitioned by year/month) |
| `data/lake/curated/features/` | engineered features (Parquet, partitioned by year/month) |
| `data/lake/curated/weather/` | weather observations (Parquet, when supplied) |
| `data/quarantine/` | malformed rows with a `quarantine_reason` column |
| `data/metrics.db` | SQLite: pipeline runs, model runs, prediction logs |
| `models/<run_id>/` | model artifacts (`*.joblib`) + `metadata.json` |
| `models/latest.json` | pointer to the most recent run |

Reruns **replace** the curated lake (full refresh), so repeated runs do not accumulate
duplicate output.

---

## 7. Dashboard views

- **Overview** — data source/type indicator (synthetic vs uploaded), row counts, date range,
  delayed rate, data-quality counts, model status and last run date. "Not trained" is clearly
  distinct from real metrics.
- **Data & ETL** — generate demo data; upload flight/weather CSV; validate and preview
  recognized columns and quarantine rows; run the ETL and see processed/quarantined/duplicate counts.
- **Train & Evaluate** — choose models, train, view real metrics, confusion matrix, feature
  importance, and chronological split ranges. Synthetic-demo results are labelled as such.
- **Predictions** — single-flight form, an eligible record from the curated table, and batch CSV
  scoring with downloadable results. Targets are never required for inference.
- **Live Simulation** — a clearly-labelled simulated producer/consumer that replays eligible
  flights in scheduled-time order, shows events + predictions, and logs them to SQLite.
- **History** — pipeline runs, model runs, and prediction logs from SQLite, including failures.

The app starts without a trained model: it shows instructions instead of invented metrics.

---

## 8. Tests

`python -m pytest tests -q` covers: BTS/canonical column mapping and HHMM/date parsing; missing
required columns; malformed-row quarantine with reasons; duplicate handling; cancelled/diverted
exclusion; the 15-minute target; prevention of post-flight columns entering features;
point-in-time safety of historical aggregates (including "future records don't change earlier
features"); weather join cutoff safety; chronological splitting; model training and prediction on
small deterministic data; correctly calculated metrics (hand-computed expectations); SQLite
pipeline/model/prediction logging; and simulated stream replay.

Spark tests (`-m spark`) are skipped automatically with a clear message when Java/PySpark is
unavailable, and also verify that the Spark and pandas engines produce matching counts and
identical point-in-time features.

---

## 9. Limitations

- **Local Spark only.** `local[*]` on one machine; the pipeline is not a cluster deployment.
- **Local model training.** Feature tables are collected to the driver for scikit-learn/XGBoost;
  `SKYPREDICT_MAX_TRAIN_ROWS` (default 100 000) caps this. Local training is not distributed.
- **No real-time airline API, no Kafka, no production data.** The live feed is a simulation.
- **Synthetic demo data is not real airline data** and is labelled as synthetic wherever shown.
- **XGBoost** is used when it imports successfully; otherwise the remaining models still run and
  the dashboard shows why XGBoost was unavailable.
- **Historical aggregates** are unavailable for brand-new routes/carriers and fall back to the
  training medians via the preprocessing pipeline.