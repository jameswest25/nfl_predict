"""
NFL Event Flagging (Stub - Not Used)

This module is kept for compatibility but is not used in the NFL pipeline.
NFL event information is handled directly in pipeline/clean.py via the 
play_outcome column derived from nflfastR fields.
"""
import logging
import pandas as pd

__all__ = ["add_event_contact_features"]


def add_event_contact_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stub function for MLB compatibility - not used in NFL pipeline.
    
    NFL play outcomes are derived from nflfastR fields:
    - pass_touchdown, rush_touchdown
    - complete_pass, incomplete_pass
    - interception, fumble, sack
    - penalty, qb_kneel, qb_spike
    
    Returns:
        DataFrame unchanged
    """
    logging.debug("add_event_contact_features() called but not used for NFL data")
    return df
