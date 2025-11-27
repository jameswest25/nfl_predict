"""Target column validation for NFL player prediction models.

Target columns are created in utils/feature/player_game_level.py during aggregation.
This module validates that required targets exist and have correct dtypes.
"""

import logging

import polars as pl

try:  # pandas is optional when this module is used during feature generation
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None

from utils.feature.labels import DEFAULT_LABEL_VERSION, get_label_spec

logger = logging.getLogger(__name__)


# Expected target columns and their types
NFL_TARGET_SCHEMA = {
    # Binary classification targets
    "anytime_td": pl.Int8,  # Primary MVP target: did player score a TD?
    "anytime_td_offense": pl.Int8,
    "anytime_td_all": pl.Int8,
    "anytime_td_rush": pl.Int8,
    "anytime_td_rec": pl.Int8,
    "anytime_td_pass_thrown": pl.Int8,
    
    # Count/regression targets
    "td_count": pl.Int64,  # Number of TDs scored (receiving + rushing)
    "td_count_offense": pl.Int64,
    "td_count_all": pl.Int64,
    "passing_td": pl.Int64,  # Number of passing TDs (QB only)
    
    # Yardage regression targets
    "receiving_yards": pl.Float64,
    "rushing_yards": pl.Float64,
    "passing_yards": pl.Float64,
    
    # Component TD counts (for analysis/features, not direct targets)
    "receiving_td_count": pl.Int64,
    "rushing_td_count": pl.Int64,
    
    # Receiving-specific stats
    "target": pl.Int64,
    "reception": pl.Int64,
}


def validate_target_columns(df: pl.DataFrame, strict: bool = False) -> pl.DataFrame:
    """Validate that required NFL target columns exist with correct types.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame (typically player-game level data)
    strict : bool
        If True, raise error if any target is missing. 
        If False, log warning and create missing targets with nulls.

    Returns
    -------
    pl.DataFrame
        DataFrame with validated target columns
    """
    
    missing_targets = []
    wrong_dtype_targets = []
    
    for target_col, expected_dtype in NFL_TARGET_SCHEMA.items():
        if target_col not in df.columns:
            missing_targets.append(target_col)
        elif df[target_col].dtype != expected_dtype:
            wrong_dtype_targets.append(
                f"{target_col} (expected {expected_dtype}, got {df[target_col].dtype})"
            )
    
    if missing_targets:
        msg = f"Missing target columns: {missing_targets}"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg + " - creating with null values")
            # Add missing columns as nulls
            for col in missing_targets:
                df = df.with_columns(pl.lit(None).cast(NFL_TARGET_SCHEMA[col]).alias(col))
    
    if wrong_dtype_targets:
        logger.warning(f"Target columns with wrong dtype: {wrong_dtype_targets}")
        # Auto-cast to expected types
        for col, expected_dtype in NFL_TARGET_SCHEMA.items():
            if col in df.columns and df[col].dtype != expected_dtype:
                try:
                    df = df.with_columns(pl.col(col).cast(expected_dtype))
                    logger.info(f"Cast {col} to {expected_dtype}")
                except Exception as e:
                    logger.error(f"Failed to cast {col} to {expected_dtype}: {e}")
    
    logger.info(f"Validated {len(NFL_TARGET_SCHEMA)} NFL target columns")
    return df


def add_target_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure target columns exist (for backward compatibility with old MLB code).
    
    For NFL: targets are created during player-game aggregation.
    This function just validates they exist.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with player-game level data

    Returns
    -------
    pl.DataFrame
        DataFrame with validated target columns
    """
    return validate_target_columns(df, strict=False)


def require_target_column(
    df: pl.DataFrame,
    target_col: str,
    *,
    label_version: str | None = None,
    min_non_null: int = 1,
) -> pl.DataFrame:
    """Ensure the configured target column exists and is populated.

    Raises a ValueError if the column is missing or entirely null.
    """

    if target_col not in df.columns:
        alias_hint = ""
        if label_version:
            try:
                spec = get_label_spec(label_version or DEFAULT_LABEL_VERSION)
                if target_col in spec.aliases:
                    alias_hint = f" Target maps to '{spec.aliases[target_col]}' under {spec.name}."
            except Exception:
                alias_hint = ""
        raise ValueError(f"Target column '{target_col}' not found in feature frame.{alias_hint}")

    series = df[target_col]
    non_null: int
    total_rows = len(series)

    if isinstance(series, pl.Series):
        non_null = series.drop_nulls().height
    elif pd is not None and isinstance(series, pd.Series):
        non_null = int(series.dropna().shape[0])
    else:
        # Fallback for plain array-like objects
        non_null = int(sum(val is not None for val in series))

    if non_null < min_non_null:
        raise ValueError(
            f"Target column '{target_col}' has insufficient non-null rows "
            f"({non_null}/{total_rows})."
        )
    nulls = total_rows - non_null
    if nulls > 0:
        logger.warning("Target %s contains %d null rows; they will be dropped.", target_col, nulls)
    return df
