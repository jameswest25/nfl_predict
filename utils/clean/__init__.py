"""
Clean utilities module for NFL data cleaning and post-cleaning audit.

Exports:
    - save_clean: Save cleaned data as partitioned Parquet files by date
    - scan_cleaned: Get a Polars lazy scan over all cleaned daily partitions
    - run_clean_audit: Run audit on cleaned data
"""

from .io import save_clean, scan_cleaned
from .audit import run_clean_audit

__all__ = [
    "save_clean",
    "scan_cleaned",
    "run_clean_audit",
]
