"""Chronological split tests."""

from __future__ import annotations

import pandas as pd
import pytest

from skypredict.models.dataset import (
    cap_training_rows,
    chronological_split,
    eligible_training_rows,
)


def _frame(n_days=12, per_day=6):
    rng = pd.date_range("2026-03-01", periods=n_days, freq="D")
    rows = []
    for i, day in enumerate(rng):
        for j in range(per_day):
            rows.append(
                {
                    "flight_date": day,
                    "carrier": ["AA", "DL", "UA"][j % 3],
                    "origin": ["ATL", "DFW", "ORD"][j % 3],
                    "destination": ["DEN", "LAX", "SEA"][j % 3],
                    "sched_dep_minute": 360 + j * 60,
                    "sched_duration_minutes": 120.0,
                    "distance": 700.0,
                    "cancelled": 0,
                    "diverted": 0,
                    "arrival_delay_minutes": 20.0 if (i + j) % 3 == 0 else 5.0,
                    "is_delayed": 1.0 if (i + j) % 3 == 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_split_is_chronological_and_disjoint():
    df = _frame()
    train, val, test, ranges = chronological_split(df)
    assert train["flight_date"].max() < val["flight_date"].min()
    assert val["flight_date"].max() < test["flight_date"].min()
    # No date may span two splits
    assert set(train["flight_date"]).isdisjoint(set(val["flight_date"]))
    assert set(val["flight_date"]).isdisjoint(set(test["flight_date"]))
    assert ranges["train_end"] < ranges["val_start"]
    assert ranges["val_end"] < ranges["test_start"]


def test_split_fractions_respected():
    df = _frame(n_days=20)
    _t, _v, _te, ranges = chronological_split(df, validation_fraction=0.15, test_fraction=0.15)
    # 20 dates -> 70/3/3 date split (fractions round to 3 each)
    assert ranges["train_dates"] == 14
    assert ranges["val_dates"] == 3
    assert ranges["test_dates"] == 3


def test_split_needs_three_dates():
    df = _frame(n_days=2, per_day=2)
    with pytest.raises(ValueError):
        chronological_split(df)


def test_eligible_rows_excludes_cancelled_diverted_and_targetless():
    df = _frame(n_days=3)
    extra = df.iloc[:2].copy()
    extra["cancelled"] = 1
    diverted = df.iloc[2:4].copy()
    diverted["diverted"] = 1
    no_target = df.iloc[4:6].copy()
    no_target["is_delayed"] = None
    mixed = pd.concat([df, extra, diverted, no_target], ignore_index=True)
    eligible = eligible_training_rows(mixed)
    assert (eligible["cancelled"] == 0).all()
    assert (eligible["diverted"] == 0).all()
    assert eligible["is_delayed"].notna().all()
    assert len(eligible) == len(df)


def test_cap_training_rows_deterministic():
    df = _frame(n_days=10, per_day=50)
    capped = cap_training_rows(df, max_rows=100, seed=42)
    assert len(capped) == 100
    assert list(capped["flight_date"]) == sorted(capped["flight_date"])
    assert capped.reset_index(drop=True).equals(cap_training_rows(df, max_rows=100, seed=42).reset_index(drop=True))
