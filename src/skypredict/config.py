"""Central configuration: paths, defaults, and environment overrides.

All paths are resolved relative to the project root unless overridden via
environment variables, so the package works both from the repository and
after an editable install.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("SKYPREDICT_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("SKYPREDICT_MODELS_DIR", PROJECT_ROOT / "models"))

RAW_DIR = DATA_DIR / "raw"
UPLOAD_DIR = DATA_DIR / "uploads"
QUARANTINE_DIR = DATA_DIR / "quarantine"
CURATED_FLIGHTS_DIR = DATA_DIR / "lake" / "curated" / "flights"
CURATED_FEATURES_DIR = DATA_DIR / "lake" / "curated" / "features"
CURATED_WEATHER_DIR = DATA_DIR / "lake" / "curated" / "weather"
METRICS_DB_PATH = DATA_DIR / "metrics.db"

DEMO_FLIGHTS_CSV = RAW_DIR / "demo_flights.csv"
DEMO_WEATHER_CSV = RAW_DIR / "demo_weather.csv"

# Spark
SPARK_MASTER = os.environ.get("SKYPREDICT_SPARK_MASTER", "local[*]")
# Optional project-local Java runtime (downloaded to <project>/runtime by setup)
LOCAL_JRE_DIR = PROJECT_ROOT / "runtime"

# Machine learning
RANDOM_SEED = 42
DELAY_THRESHOLD_MINUTES = 15
PREDICTION_LEAD_HOURS = 2  # prediction-time assumption: 2 hours before scheduled departure
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
MAX_TRAIN_ROWS = int(os.environ.get("SKYPREDICT_MAX_TRAIN_ROWS", "100000"))
MIN_ROWS_FOR_DURATION_MODEL = 30

# Demo data
DEMO_DAYS = 120
DEMO_FLIGHTS_PER_DAY = 60


def ensure_runtime_dirs() -> None:
    """Create the local runtime directories if they do not exist."""
    for d in (DATA_DIR, RAW_DIR, UPLOAD_DIR, QUARANTINE_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
