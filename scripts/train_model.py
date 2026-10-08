"""Train and evaluate models from the command line.

Examples:
    python scripts/train_model.py --models all
    python scripts/train_model.py --models random_forest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skypredict.config import CURATED_FEATURES_DIR, METRICS_DB_PATH, MODELS_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="all",
        help="comma-separated subset of baseline,random_forest,xgboost (default: all)",
    )
    args = parser.parse_args()

    from skypredict.models.dataset import load_curated_features
    from skypredict.models.train import train_all

    choice = args.models.strip().lower()
    if choice == "all":
        models = ("baseline", "random_forest", "xgboost")
    else:
        models = tuple(m.strip() for m in choice.split(",") if m.strip())

    print("Loading curated features ...")
    features = load_curated_features(CURATED_FEATURES_DIR)
    source_type = "unknown"
    try:
        from skypredict.services import db

        runs = db.list_pipeline_runs(METRICS_DB_PATH, limit=5)
        completed = [r for r in runs if r["status"] == "completed"]
        if completed:
            source_type = completed[0]["source_type"] or "unknown"
    except Exception:
        pass
    if source_type == "synthetic_demo":
        print("NOTE: metrics below are computed on SYNTHETIC DEMO DATA.")

    print(f"Training models: {', '.join(models)} ...")
    output = train_all(features, dataset_source=source_type, models=models,
                       models_dir=MODELS_DIR, db_path=METRICS_DB_PATH)

    meta = output.metadata
    print(f"\nRun id        : {output.run_id}")
    print(f"Artifacts dir : {output.models_dir}")
    print(f"Train         : {meta['date_ranges']['train_start']} .. {meta['date_ranges']['train_end']} ({meta['sample_counts']['train_rows']} rows)")
    print(f"Validation    : {meta['date_ranges']['val_start']} .. {meta['date_ranges']['val_end']} ({meta['sample_counts']['val_rows']} rows)")
    print(f"Test          : {meta['date_ranges']['test_start']} .. {meta['date_ranges']['test_end']} ({meta['sample_counts']['test_rows']} rows)")

    for name, res in output.model_results.items():
        print(f"\n--- {name} ---")
        if not res.get("available"):
            print(f"  unavailable: {res.get('error')}")
            continue
        m = res["metrics"]
        print(f"  accuracy {m['accuracy']:.3f} | precision {m['precision_delayed']:.3f} "
              f"| recall {m['recall_delayed']:.3f} | F1 {m['f1_delayed']:.3f} "
              f"| ROC-AUC {m['roc_auc'] if m['roc_auc'] is not None else 'n/a'} "
              f"| threshold {m['threshold']:.3f}")
        cm = m["confusion_matrix"]
        print(f"  confusion (tn/fp/fn/tp): {cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']}")
    if meta.get("duration_model", {}).get("available"):
        dm = meta["duration_model"]["metrics"]
        print(f"\n--- duration_random_forest (delayed flights only) ---")
        print(f"  MAE {dm['mae']:.1f} min | RMSE {dm['rmse']:.1f} min (n={dm['n']})")
    elif meta.get("duration_model", {}).get("note"):
        print(f"\nduration model not trained: {meta['duration_model']['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
