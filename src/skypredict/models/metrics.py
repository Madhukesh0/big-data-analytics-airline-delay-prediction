"""Evaluation metrics, computed from real model predictions.

Accuracy alone is not relied upon because delayed flights are a minority
class; precision/recall/F1 for the delayed class are always reported, and
ROC-AUC only when both classes are present in the evaluation split.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score


def classification_metrics(y_true, proba: np.ndarray, threshold: float) -> dict:
    """Metrics for the delayed (positive) class at a fixed threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)

    roc_auc = None
    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, proba))

    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision_delayed": float(precision),
        "recall_delayed": float(recall),
        "f1_delayed": float(f1),
        "roc_auc": roc_auc,
        "delayed_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def regression_metrics(y_true, y_pred) -> dict:
    """MAE and RMSE for delay-duration regression (delayed flights only)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors))) if len(y_true) else None
    rmse = float(np.sqrt(np.mean(errors**2))) if len(y_true) else None
    return {"n": int(len(y_true)), "mae": mae, "rmse": rmse}


def choose_threshold(y_val, proba_val: np.ndarray) -> float:
    """Pick the classification threshold that maximizes F1 on VALIDATION data.

    The final test split is never used for threshold selection. Falls back
    to 0.5 when validation data is degenerate (single class, or empty).
    """
    y_val = np.asarray(y_val).astype(int)
    if len(y_val) == 0 or len(np.unique(y_val)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    if len(thresholds) == 0:
        return 0.5
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)
    best = int(np.argmax(f1))
    return float(np.clip(thresholds[best], 0.01, 0.99))
