"""Model loading and prediction (single flight and batch).

Inference never requires target columns. For flights outside the historical
range, historical aggregate features are simply absent (NaN) and fall back
to the training medians inside the fitted preprocessing pipeline — this is
stated in the dashboard and README.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import MODELS_DIR
from ..data.schema import map_columns, parse_flight_date, parse_hhmm
from ..features import assemble_feature_frame


def load_latest_run_info(models_dir: Path = MODELS_DIR) -> dict | None:
    """Return the latest.json pointer, or None when nothing is trained."""
    latest = Path(models_dir) / "latest.json"
    if not latest.exists():
        return None
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_runs(models_dir: Path = MODELS_DIR) -> list[dict]:
    """All saved model runs, newest first."""
    root = Path(models_dir)
    runs: list[dict] = []
    for meta_path in sorted(root.glob("*/metadata.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            runs.append(meta)
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def load_bundle(models_dir: Path, run_id: str, model_name: str) -> dict:
    path = Path(models_dir) / run_id / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def load_duration_bundle(models_dir: Path, run_id: str) -> dict | None:
    path = Path(models_dir) / run_id / "duration_random_forest.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def prepare_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map an uploaded/entered flight frame to canonical inference columns.

    Accepts canonical or BTS-style column names. Missing/invalid times and
    dates become NaN (the preprocessing imputers handle them); no target is
    required or used.
    """
    df, _info = map_columns(df)
    out = pd.DataFrame(index=df.index)

    if "flight_date" in df.columns:
        out["flight_date"] = df["flight_date"].apply(parse_flight_date)
    if "carrier" in df.columns:
        out["carrier"] = df["carrier"].astype("string").str.strip().str.upper()
    if "origin" in df.columns:
        out["origin"] = df["origin"].astype("string").str.strip().str.upper()
    if "destination" in df.columns:
        out["destination"] = df["destination"].astype("string").str.strip().str.upper()

    if "sched_dep_minute" in df.columns:
        out["sched_dep_minute"] = pd.to_numeric(df["sched_dep_minute"], errors="coerce")
    elif "scheduled_departure" in df.columns:
        out["sched_dep_minute"] = df["scheduled_departure"].apply(parse_hhmm)

    if "sched_arr_minute" in df.columns:
        out["sched_arr_minute"] = pd.to_numeric(df["sched_arr_minute"], errors="coerce")
    elif "scheduled_arrival" in df.columns:
        out["sched_arr_minute"] = df["scheduled_arrival"].apply(parse_hhmm)

    if "sched_duration_minutes" in df.columns:
        out["sched_duration_minutes"] = pd.to_numeric(df["sched_duration_minutes"], errors="coerce")

    if "distance" in df.columns:
        out["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    # Derive duration when not supplied directly.
    if "sched_duration_minutes" not in out.columns and {
        "sched_dep_minute", "sched_arr_minute"
    }.issubset(out.columns):
        dep = out["sched_dep_minute"]
        arr = out["sched_arr_minute"]
        dur = arr - dep
        dur = dur.where(dur > 0, dur + 1440)
        out["sched_duration_minutes"] = dur.where(dep.notna() & arr.notna())

    for col in ("hist_carrier_delay_rate", "hist_carrier_avg_delay",
                "hist_route_delay_rate", "hist_route_avg_delay",
                "hist_origin_delay_rate", "hist_dest_delay_rate",
                "hist_route_flights"):
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    from ..features import WEATHER_COLUMNS

    for col in WEATHER_COLUMNS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    return out


def score_dataframe(
    bundle: dict,
    df: pd.DataFrame,
    duration_bundle: dict | None = None,
) -> pd.DataFrame:
    """Score a canonical/standardized flight frame and return predictions.

    Output columns: ``delay_probability``, ``predicted_delayed``,
    ``expected_delay_minutes_if_delayed`` (conditional), and
    ``expected_delay_minutes`` (probability-weighted, when the duration
    model exists).
    """
    prepared = prepare_inference_frame(df)
    X = assemble_feature_frame(prepared)
    X_t = bundle["preprocessor"].transform(X)
    proba = bundle["model"].predict_proba(X_t)[:, 1]
    threshold = float(bundle.get("threshold", 0.5))

    out = df.reset_index(drop=True).copy()
    out["delay_probability"] = np.round(proba, 4)
    out["predicted_delayed"] = (proba >= threshold).astype(int)
    out["applied_threshold"] = threshold

    if duration_bundle is not None:
        X_dt = duration_bundle["preprocessor"].transform(X)
        minutes = duration_bundle["model"].predict(X_dt)
        out["expected_delay_minutes_if_delayed"] = np.round(minutes, 1)
        out["expected_delay_minutes"] = np.round(proba * minutes, 1)
        out["duration_note"] = "conditional on flight being delayed"
    return out


def score_record(bundle: dict, record: dict, duration_bundle: dict | None = None) -> dict:
    """Score one flight given a dict of canonical fields."""
    frame = pd.DataFrame([record])
    scored = score_dataframe(bundle, frame, duration_bundle)
    return scored.iloc[0].to_dict()
