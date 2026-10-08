"""Model training, comparison, and artifact management.

Trains and compares a logistic-regression baseline, a Random Forest
classifier, and (when xgboost is importable) an XGBoost classifier, plus a
Random Forest duration regressor fitted on delayed training flights only.

Threshold tuning uses VALIDATION data only. All metrics are computed from
real predictions on the held-out chronological test period and are stored
in the model metadata and SQLite — nothing is hard-coded.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..config import (
    MAX_TRAIN_ROWS,
    METRICS_DB_PATH,
    MIN_ROWS_FOR_DURATION_MODEL,
    MODELS_DIR,
    RANDOM_SEED,
)
from ..features import assemble_feature_frame
from ..services import db
from .dataset import (
    build_xy,
    cap_training_rows,
    chronological_split,
    eligible_training_rows,
)
from .metrics import choose_threshold, classification_metrics, regression_metrics
from .preprocess import build_preprocessor


@dataclass
class TrainOutput:
    run_id: str
    models_dir: Path
    metadata: dict
    model_results: dict = field(default_factory=dict)


def _xgboost_available() -> tuple[bool, str]:
    try:
        import xgboost  # noqa: F401

        return True, "available"
    except Exception as exc:  # ImportError or binary incompatibility
        return False, (
            "XGBoost is unavailable in this environment "
            f"({exc.__class__.__name__}: {exc}). The Random Forest workflow "
            "remains fully functional."
        )


def _estimator(name: str, seed: int):
    if name == "baseline":
        return Pipeline(
            steps=[
                ("scaler", None),  # replaced below; see build_model
                ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model name: {name}")


def _build_model(name: str, seed: int):
    """Wrap the estimator with scaling where needed (baseline = linear)."""
    if name == "baseline":
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )
    return _estimator(name, seed)


MODEL_DISPLAY_NAMES = {
    "baseline": "logistic_regression_baseline",
    "random_forest": "random_forest",
    "xgboost": "xgboost",
}


def train_all(
    features_df,
    *,
    dataset_source: str,
    models: tuple[str, ...] = ("baseline", "random_forest", "xgboost"),
    models_dir: Path = MODELS_DIR,
    db_path: Path = METRICS_DB_PATH,
    seed: int = RANDOM_SEED,
    max_rows: int = MAX_TRAIN_ROWS,
) -> TrainOutput:
    """Train the requested models on curated features and save artifacts."""
    db.init_db(db_path)
    started_at = db.utcnow()
    t0 = time.time()

    eligible = eligible_training_rows(features_df)
    if eligible.empty:
        raise ValueError(
            "No eligible training rows found (rows must not be cancelled/diverted "
            "and must have arrival_delay_minutes)."
        )
    eligible = cap_training_rows(eligible, max_rows, seed)

    train, val, test, ranges = chronological_split(eligible)
    X_train, y_train, _ = build_xy(train)
    X_val, y_val, _ = build_xy(val)
    X_test, y_test, _ = build_xy(test)

    preprocessor = build_preprocessor(scale=False)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{seed}"
    run_dir = Path(models_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_results: dict[str, dict] = {}
    saved_models: list[str] = []

    for name in models:
        display = MODEL_DISPLAY_NAMES[name]
        if name == "xgboost":
            available, reason = _xgboost_available()
            if not available:
                model_results[display] = {"available": False, "error": reason}
                continue
        try:
            model = _build_model(name, seed)
            model.fit(X_train_t, y_train)
            proba_val = model.predict_proba(X_val_t)[:, 1]
            threshold = choose_threshold(y_val, proba_val)
            proba_test = model.predict_proba(X_test_t)[:, 1]
            metrics = classification_metrics(y_test, proba_test, threshold)

            bundle = {
                "model": model,
                "preprocessor": preprocessor,
                "feature_list": list(X_train.columns),
                "threshold": threshold,
                "model_name": display,
                "model_version": run_id,
                "dataset_source": dataset_source,
                "date_ranges": ranges,
                "trained_at": started_at,
                "seed": seed,
                "versions": {
                    "python": platform.python_version(),
                    "scikit-learn": sklearn.__version__,
                },
            }
            artifact = run_dir / f"{display}.joblib"
            joblib.dump(bundle, artifact)
            saved_models.append(display)
            model_results[display] = {"available": True, "metrics": metrics, "artifact": str(artifact)}
            db.insert_model_run(
                db_path,
                started_at=started_at,
                finished_at=db.utcnow(),
                status="completed",
                model_name=display,
                model_version=run_id,
                dataset_source=dataset_source,
                date_ranges=ranges,
                sample_counts={
                    "train_rows": int(len(y_train)),
                    "val_rows": int(len(y_val)),
                    "test_rows": int(len(y_test)),
                },
                threshold=threshold,
                metrics=metrics,
                artifact_path=str(artifact),
            )
        except Exception as exc:
            model_results[display] = {"available": False, "error": repr(exc)}

    duration_info = _train_duration_model(
        train, val, test, preprocessor, run_dir, run_id, started_at, dataset_source, ranges, seed
    )

    metadata = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": db.utcnow(),
        "duration_seconds": round(time.time() - t0, 2),
        "dataset_source": dataset_source,
        "training_mode": "local scikit-learn/XGBoost (not distributed)",
        "max_train_rows_cap": max_rows,
        "rows_capped_from": int(len(eligible)),
        "date_ranges": ranges,
        "sample_counts": {
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
        },
        "models": model_results,
        "duration_model": duration_info,
        "seed": seed,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
        },
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    (Path(models_dir) / "latest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "trained_at": started_at,
                "models": saved_models,
                "dataset_source": dataset_source,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return TrainOutput(run_id=run_id, models_dir=run_dir, metadata=metadata, model_results=model_results)


def _train_duration_model(
    train, val, test, preprocessor, run_dir: Path, run_id: str, started_at: str,
    dataset_source: str, ranges: dict, seed: int,
) -> dict:
    """Duration regression on DELAYED training flights only.

    The predicted duration is CONDITIONAL on the flight being delayed; it is
    not an unconditional delay expectation.
    """
    train_delayed = train[pd.to_numeric(train["is_delayed"], errors="coerce") == 1]
    test_delayed = test[pd.to_numeric(test["is_delayed"], errors="coerce") == 1]
    info: dict = {"available": False}
    if len(train_delayed) < MIN_ROWS_FOR_DURATION_MODEL or len(test_delayed) < 1:
        info["note"] = (
            f"Too few delayed flights to train a duration model "
            f"(train delayed: {len(train_delayed)}, test delayed: {len(test_delayed)}, "
            f"minimum: {MIN_ROWS_FOR_DURATION_MODEL})."
        )
        return info

    X_train_d = assemble_feature_frame(train_delayed)
    y_train_d = pd.to_numeric(train_delayed["arrival_delay_minutes"], errors="coerce").to_numpy()
    X_test_d = assemble_feature_frame(test_delayed)
    y_test_d = pd.to_numeric(test_delayed["arrival_delay_minutes"], errors="coerce").to_numpy()

    X_train_dt = preprocessor.transform(X_train_d)
    X_test_dt = preprocessor.transform(X_test_d)

    model = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed
    )
    model.fit(X_train_dt, y_train_d)
    pred = model.predict(X_test_dt)
    metrics = regression_metrics(y_test_d, pred)

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_list": list(X_train_d.columns),
        "model_name": "duration_random_forest",
        "model_version": run_id,
        "conditional_on_delayed": True,
        "dataset_source": dataset_source,
        "trained_at": started_at,
        "seed": seed,
    }
    artifact = run_dir / "duration_random_forest.joblib"
    joblib.dump(bundle, artifact)
    info = {"available": True, "metrics": metrics, "artifact": str(artifact),
            "conditional_on_delayed": True}
    return info
