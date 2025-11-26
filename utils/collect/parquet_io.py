#!/usr/bin/env python3
"""
Legacy Monolithic Parquet/CSV I/O Operations (DEPRECATED)

This module handles reading from and writing to monolithic Parquet and CSV files.
It exists only for backwards compatibility with the `backfill_weather` CLI command.

NOTE: New code should NOT use these functions. The modern NFL pipeline uses
partitioned Parquet files stored in:
    data/raw/pbp_by_day/season=YYYY/week=WW/date=YYYY-MM-DD/part.parquet

For new data access patterns, use:
    - pl.scan_parquet("data/raw/pbp_by_day/season=*/week=*/date=*/part.parquet")
    - Or the appropriate feature-level scan functions

Exports:
    - write_data(df: pd.DataFrame) -> None  [DEPRECATED]
    - load_data(start_date: str = None, end_date: str = None) -> pd.DataFrame  [DEPRECATED]
    - get_latest_date() -> dt.date | None  [DEPRECATED]
    - clean_overlaps(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame
"""

import datetime as dt
import logging
import warnings
from pathlib import Path

import pandas as pd

from utils.general.paths import PBP_RAW_PARQUET

# Legacy paths - only used by this deprecated module
_LEGACY_CSV = PBP_RAW_PARQUET.with_suffix(".csv")

logger = logging.getLogger(__name__)


def clean_overlaps(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """Remove overlapping dates, combine, and de-duplicate rows based on primary keys.
    
    This utility function is still valid for general DataFrame operations.
    """
    if new_df.empty:
        return existing_df
    if existing_df.empty:
        return new_df
    
    # Align columns to prevent FutureWarning and ensure schema consistency.
    all_cols = existing_df.columns.union(new_df.columns)
    existing_df = existing_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    combined = pd.concat([existing_df, new_df], ignore_index=True)

    # Remove duplicates on primary keys if available
    try:
        from utils.collect.timestamps import MERGE_KEYS as _MERGE_KEYS
        merge_keys = [k for k in _MERGE_KEYS if k in combined.columns]
        if merge_keys:
            before = len(combined)
            combined = combined.drop_duplicates(subset=merge_keys, keep="last")
            removed = before - len(combined)
            if removed:
                logger.info(
                    "clean_overlaps: removed %d duplicate rows based on %s",
                    removed,
                    merge_keys,
                )
    except Exception as e:
        logger.warning("clean_overlaps: could not de-duplicate on merge keys: %s", e)
    return combined


def write_data(df: pd.DataFrame):
    """Write DataFrame to monolithic files.
    
    DEPRECATED: This function writes to legacy monolithic files and should not
    be used in new code. The modern pipeline uses partitioned Parquet files.
    """
    warnings.warn(
        "write_data() is deprecated. Use partitioned Parquet writes instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    if df.empty:
        logger.warning("Empty DataFrame provided to write_data")
        return
    
    # Update Parquet file
    if PBP_RAW_PARQUET.exists():
        existing = pd.read_parquet(PBP_RAW_PARQUET)
        combined = clean_overlaps(df, existing)
    else:
        combined = df
    
    combined.to_parquet(
        PBP_RAW_PARQUET,
        compression='zstd',
        engine='pyarrow',
        index=False
    )
    
    # Update legacy CSV file (for extreme backwards compat only)
    if _LEGACY_CSV.exists():
        existing_csv = pd.read_csv(_LEGACY_CSV, low_memory=False)
        combined_csv = clean_overlaps(df, existing_csv)
    else:
        combined_csv = df
    
    combined_csv.to_csv(_LEGACY_CSV, index=False)
    
    logger.info(
        "Updated monolithic files: %d rows in Parquet, %d rows in CSV",
        len(combined),
        len(combined_csv),
    )


def load_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load data from monolithic files with optional date filtering.
    
    DEPRECATED: This function reads from legacy monolithic files and should not
    be used in new code. The modern pipeline uses partitioned Parquet files.
    
    For new code, use:
        pl.scan_parquet("data/raw/pbp_by_day/season=*/week=*/date=*/part.parquet")
    """
    warnings.warn(
        "load_data() is deprecated. Use partitioned Parquet scans instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    # Try Parquet first (faster), then CSV
    if PBP_RAW_PARQUET.exists():
        df = pd.read_parquet(PBP_RAW_PARQUET)
    elif _LEGACY_CSV.exists():
        df = pd.read_csv(_LEGACY_CSV, low_memory=False)
    else:
        logger.warning("No monolithic data files found")
        return pd.DataFrame()
    
    # Apply date filtering if requested
    if start_date or end_date:
        if start_date:
            df = df[df['game_date'] >= start_date]
        if end_date:
            df = df[df['game_date'] <= end_date]
    
    return df


def get_latest_date() -> dt.date | None:
    """Get the latest date from existing monolithic data.
    
    DEPRECATED: This function reads from legacy monolithic files.
    For new code, scan the partitioned Parquet directory structure.
    """
    warnings.warn(
        "get_latest_date() from monolithic files is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    data_file = PBP_RAW_PARQUET if PBP_RAW_PARQUET.exists() else _LEGACY_CSV
    
    if not data_file.exists():
        return None
    
    try:
        if data_file.suffix == '.parquet':
            df_dates = pd.read_parquet(data_file, columns=['game_date'])
        else:
            df_dates = pd.read_csv(data_file, usecols=['game_date'], parse_dates=['game_date'])
        
        latest_date = df_dates['game_date'].max()
        if pd.isna(latest_date):
            return None
        
        if hasattr(latest_date, 'date'):
            latest_date = latest_date.date()
        else:
            latest_date = pd.to_datetime(latest_date).date()
            
        logger.info(f"Found existing data through {latest_date}")
        return latest_date
    except Exception as e:
        logger.warning(f"Could not read existing data file: {e}")
        return None
