"""
Constants for NFL Data Cleaning

NFL-specific constants and paths for data cleaning and validation.
All paths should be imported from utils.general.paths.
"""

# Import from centralized modules (single sources of truth)
from utils.general.paths import DATA_CLEANED as CLEAN_DIR, PROJ_ROOT
from utils.general.constants import MISSING_VALUE_TOKENS, ROOF_CLOSED_TEAMS

__all__ = [
    "PROJ_ROOT",
    "CLEAN_DIR",
    "MISSING_VALUE_TOKENS",
    "ROOF_CLOSED_TEAMS",
]
