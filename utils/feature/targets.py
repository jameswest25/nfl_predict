"""Target column validation for NFL player prediction models.

Target columns are created in utils/feature/player_game_level.py during aggregation.
This module validates that required targets exist and have correct dtypes.
"""

import polars as pl
import logging

logger = logging.getLogger(__name__)


# Expected target columns and their types
NFL_TARGET_SCHEMA = {
    # Binary classification targets
    "anytime_td": pl.Int8,  # Primary MVP target: did player score a TD?
    
    # Count/regression targets
    "td_count": pl.Int64,  # Number of TDs scored (receiving + rushing)
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