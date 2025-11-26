"""
Missing Value Cleaning (Stub - Not Used in NFL)

This module is kept for compatibility but is not used in the NFL pipeline.
NFL data from nflfastR doesn't have missing value tokens like MLB Statcast does.
"""
import logging
import pandas as pd
from .constants import MISSING_VALUE_TOKENS


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stub function for MLB compatibility - not used in NFL pipeline.
    
    NFL data from nflfastR uses proper NA/null values, not string tokens
    like 'none' or 'error' that MLB Statcast used.
    
    Returns:
        DataFrame unchanged
    """
    logging.debug("clean_missing_values() called but not used for NFL data")
    return df
