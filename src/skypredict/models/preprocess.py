"""Reusable preprocessing pipeline.

One shared ColumnTransformer is fitted on the TRAINING split only, then
applied unchanged to validation, test, and inference data. Unseen
airport/carrier categories are handled by OneHotEncoder(handle_unknown=...);
missing numeric values are imputed with training medians.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..features import FEATURE_CATEGORICAL, FEATURE_NUMERIC


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    """Build the feature preprocessor.

    ``scale=True`` adds a (sparse-safe) StandardScaler for linear models;
    tree ensembles do not need it.
    """
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    numeric = Pipeline(steps=numeric_steps)
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, FEATURE_CATEGORICAL),
            ("numeric", numeric, FEATURE_NUMERIC),
        ]
    )
