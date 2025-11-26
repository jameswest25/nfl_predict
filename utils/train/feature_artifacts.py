from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureArtifacts:
    feature_columns: List[str]
    datetime_features: List[str]
    categorical_features: List[str]
    category_levels: Dict[str, List[str]]


def fit_feature_artifacts(df_train: pd.DataFrame, problem_config: dict) -> FeatureArtifacts:
    """
    Shared logic for deriving the frozen feature list and categorical encoders.
    """
    include_prefixes = tuple(problem_config.get("feature_prefixes_to_include") or [])
    other_features = problem_config.get("other_features_to_include") or []
    discard_cols = set(problem_config.get("columns_to_discard") or [])

    initial_features = [col for col in df_train.columns if col.startswith(include_prefixes)]
    initial_features.extend([col for col in other_features if col in df_train.columns])
    feature_columns = sorted(list({col for col in initial_features if col not in discard_cols}))

    temp_X = df_train[feature_columns]
    const_cols = [c for c in temp_X.columns if temp_X[c].nunique(dropna=False) <= 1]
    if const_cols:
        logger.warning(
            "Found %d columns that are constant in the training window: %s "
            "(retaining them so they remain available if they vary later).",
            len(const_cols),
            const_cols,
        )

    final_X = df_train[feature_columns].copy()
    datetime_cols: List[str] = []
    for col in final_X.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist():
        col_as_dt = pd.to_datetime(final_X[col], utc=True, errors="coerce")
        numeric = col_as_dt.astype("int64", copy=False).astype("float64")
        mask = col_as_dt.isna().to_numpy()
        numeric[mask] = np.nan
        final_X[col] = numeric / 1_000_000.0
        datetime_cols.append(col)

    num_cols = final_X.select_dtypes(include=np.number).columns
    cat_cols = list(final_X.columns.difference(num_cols))
    category_levels: Dict[str, List[str]] = {}
    for col in cat_cols:
        series = final_X[col]
        categories = series.astype("category").cat.categories.tolist()
        if not categories:
            categories = ["__missing__"]
        category_levels[col] = categories

    return FeatureArtifacts(
        feature_columns=feature_columns,
        datetime_features=datetime_cols,
        categorical_features=cat_cols,
        category_levels=category_levels,
    )


def apply_feature_artifacts(
    df: pd.DataFrame,
    artifacts: FeatureArtifacts,
    target_col: str | None,
) -> Tuple[pd.DataFrame, pd.Series | None]:
    """
    Apply frozen feature transformations to a new frame.
    """
    X = df.reindex(columns=artifacts.feature_columns)

    for col in artifacts.datetime_features or []:
        if col in X.columns:
            col_as_dt = pd.to_datetime(X[col], utc=True, errors="coerce")
            numeric = col_as_dt.astype("int64", copy=False).astype("float64")
            mask = col_as_dt.isna().to_numpy()
            numeric[mask] = np.nan
            X[col] = numeric / 1_000_000.0

    for col in artifacts.categorical_features or []:
        if col not in X.columns:
            continue
        levels = artifacts.category_levels.get(col) or []
        try:
            if "__missing__" in levels:
                X[col] = X[col].fillna("__missing__")
            X[col] = pd.Categorical(X[col], categories=levels)
        except Exception:
            pass

    y = df[target_col] if target_col and target_col in df.columns else None
    return X, y

