import polars as pl
from pathlib import Path
from typing import Union

from utils.clean.constants import CLEAN_DIR

__all__ = ["scan_cleaned"]


def scan_cleaned(base_dir: Union[str, Path] = CLEAN_DIR, columns: list[str] | None = None) -> pl.LazyFrame:
    """Return a Polars *lazy* scan over all daily parquet partitions.

    Parameters
    ----------
    base_dir : directory containing daily parquet files.
    columns  : optional list of columns to project (predicate push-down).
    """
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(base)

    pattern = str(base / "*.parquet")
    return pl.scan_parquet(pattern, low_memory=True, with_columns=columns) 