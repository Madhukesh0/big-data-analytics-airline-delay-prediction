"""Metric computation tests with hand-calculated expectations."""

from __future__ import annotations

import numpy as np

from skypredict.models.metrics import (
    choose_threshold,
    classification_metrics,
    regression_metrics,
)


def test_classification_metrics_hand_computed():
    y_true = [0, 0, 0, 1, 1, 1, 1, 0, 1, 0]
    proba = [0.1, 0.2, 0.4, 0.9, 0.8, 0.3, 0.7, 0.6, 0.55, 0.05]
    m = classification_metrics(y_true, np.array(proba), threshold=0.5)
    # preds: 0 0 0 1 1 0 1 1 1 0 -> tn=4(fp=1 from 0.6), fn=1 (0.3), tp=4
    cm = m["confusion_matrix"]
    assert cm == {"tn": 4, "fp": 1, "fn": 1, "tp": 4}
    assert m["accuracy"] == 0.8
    assert m["precision_delayed"] == 4 / 5
    assert m["recall_delayed"] == 4 / 5
    assert abs(m["f1_delayed"] - 0.8) < 1e-9
    assert 0.5 < m["roc_auc"] <= 1.0


def test_roc_auc_absent_for_single_class():
    m = classification_metrics([1, 1, 1], np.array([0.9, 0.4, 0.6]), threshold=0.5)
    assert m["roc_auc"] is None


def test_regression_metrics_hand_computed():
    m = regression_metrics([10, 40], [20, 20])
    assert m["mae"] == 15.0                      # (|10| + |-20|) / 2
    assert abs(m["rmse"] - (250.0 ** 0.5)) < 1e-9  # sqrt((10^2 + 20^2)/2)


def test_threshold_uses_validation_not_test():
    # Validation clearly separates; a high threshold maximizes F1 there.
    y_val = [0, 0, 0, 0, 1, 1, 1, 1]
    proba_val = [0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95]
    t = choose_threshold(y_val, proba_val)
    assert 0.2 < t <= 0.8   # any threshold in (0.2, 0.8] separates perfectly
    # Degenerate validation -> default 0.5
    assert choose_threshold([1, 1, 1], np.array([0.9, 0.2, 0.4])) == 0.5
    assert choose_threshold([], np.array([])) == 0.5
