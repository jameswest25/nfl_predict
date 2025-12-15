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
    # Never silently include diagnostic MoE columns as training features.
    # These are intended for inspection and for explicit MoE-only model configs.
    initial_features = [c for c in initial_features if not str(c).endswith("_moe")]
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
    
    # Detect datetime columns more robustly
    # Check for: datetime64, datetimetz, and object columns that might contain dates
    for col in final_X.columns:
        dtype_str = str(final_X[col].dtype).lower()
        is_datetime = (
            'datetime' in dtype_str or 
            pd.api.types.is_datetime64_any_dtype(final_X[col])
        )
        # Also check for known datetime column patterns
        is_datetime_pattern = col in ('injury_snapshot_ts', 'opp_ctx_data_as_of', 
                                       'team_ctx_data_as_of', 'off_ctx_data_as_of')
        
        if is_datetime or is_datetime_pattern:
            try:
                col_as_dt = pd.to_datetime(final_X[col], utc=True, errors="coerce")
                numeric = col_as_dt.astype("int64", copy=False).astype("float64")
                mask = col_as_dt.isna().to_numpy()
                numeric[mask] = np.nan
                final_X[col] = numeric / 1_000_000.0
                datetime_cols.append(col)
            except Exception as e:
                logger.warning(f"Failed to convert {col} to numeric datetime: {e}")

    # IMPORTANT: XGBoost categorical support expects string categories.
    # Also: avoid treating bool/boolean columns as categoricals (xgboost can crash
    # when categories are booleans). Force them to numeric 0/1 instead.
    for col in list(final_X.columns):
        try:
            if pd.api.types.is_bool_dtype(final_X[col]) or str(final_X[col].dtype).lower() == "boolean":
                # Nullable boolean -> Int8 preserves missing; plain bool becomes 0/1.
                final_X[col] = final_X[col].astype("Int8")
        except Exception:
            continue

    num_cols = final_X.select_dtypes(include=[np.number]).columns
    cat_cols = list(final_X.columns.difference(num_cols))
    category_levels: Dict[str, List[str]] = {}
    for col in cat_cols:
        series = final_X[col]
        categories = series.astype("category").cat.categories.tolist()
        if not categories:
            categories = ["__missing__"]
        # Force categories to strings for xgboost's categorical backend.
        category_levels[col] = [str(c) for c in categories]

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

    # Handle datetime features stored in artifacts
    for col in artifacts.datetime_features or []:
        if col in X.columns:
            col_as_dt = pd.to_datetime(X[col], utc=True, errors="coerce")
            numeric = col_as_dt.astype("int64", copy=False).astype("float64")
            mask = col_as_dt.isna().to_numpy()
            numeric[mask] = np.nan
            X[col] = numeric / 1_000_000.0
    
    # Also detect and convert any datetime columns not in the stored list
    # This handles legacy models that didn't capture datetime features properly
    known_datetime_cols = {'injury_snapshot_ts', 'opp_ctx_data_as_of', 
                           'team_ctx_data_as_of', 'off_ctx_data_as_of'}
    for col in X.columns:
        if col in (artifacts.datetime_features or []):
            continue  # Already handled above
        dtype_str = str(X[col].dtype).lower()
        is_datetime = (
            'datetime' in dtype_str or 
            pd.api.types.is_datetime64_any_dtype(X[col]) or
            col in known_datetime_cols
        )
        if is_datetime:
            try:
                col_as_dt = pd.to_datetime(X[col], utc=True, errors="coerce")
                numeric = col_as_dt.astype("int64", copy=False).astype("float64")
                mask = col_as_dt.isna().to_numpy()
                numeric[mask] = np.nan
                X[col] = numeric / 1_000_000.0
            except Exception:
                pass  # Leave as-is if conversion fails

    for col in artifacts.categorical_features or []:
        if col not in X.columns:
            continue
        levels = artifacts.category_levels.get(col) or []
        try:
            if "__missing__" in levels:
                X[col] = X[col].fillna("__missing__")
            # Keep categorical domain as strings (xgboost categorical backend expects this).
            X[col] = X[col].astype(str)
            X[col] = pd.Categorical(X[col], categories=levels)
        except Exception:
            pass

    y = df[target_col] if target_col and target_col in df.columns else None
    return X, y

