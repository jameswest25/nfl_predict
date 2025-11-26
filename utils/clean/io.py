"""
Clean stage I/O utilities for NFL data pipeline.

Handles saving cleaned data as partitioned Parquet files.
"""
import logging
import pandas as pd
from pathlib import Path

from .constants import CLEAN_DIR

__all__ = ["save_clean", "scan_cleaned"]


def save_clean(df: pd.DataFrame) -> Path:
    """Save cleaned data as one parquet per game_date inside CLEAN_DIR.
    
    Args:
        df: DataFrame with a 'game_date' column to partition by.
        
    Returns:
        Path to the clean directory where files were written.
        
    Raises:
        ValueError: If 'game_date' column is missing.
    """
    if "game_date" not in df.columns:
        raise ValueError("game_date column missing – cannot partition per day")

    # Ensure datetime → date for filename
    dates = pd.to_datetime(df["game_date"]).dt.date
    df = df.assign(_file_date=dates)

    total_rows = len(df)
    written = 0

    for day, day_df in df.groupby("_file_date"):
        out_dir = CLEAN_DIR / f"date={day}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part.parquet"
        day_df.drop(columns=["_file_date"]).to_parquet(out_path, index=False)
        written += len(day_df)

    logging.info(
        "Saved cleaned data partitioned by day → %d rows across %d files",
        written,
        df["_file_date"].nunique(),
    )

    return CLEAN_DIR


def scan_cleaned(base_dir: Path | str = None, columns: list[str] | None = None):
    """Return a Polars *lazy* scan over all daily parquet partitions.

    Parameters
    ----------
    base_dir : directory containing daily parquet files (default: CLEAN_DIR)
    columns  : optional list of columns to project (predicate push-down)
    
    Returns
    -------
    pl.LazyFrame
    """
    import polars as pl
    
    base = Path(base_dir) if base_dir else CLEAN_DIR
    if not base.exists():
        raise FileNotFoundError(base)

    pattern = str(base / "*.parquet")
    return pl.scan_parquet(pattern, low_memory=True)
