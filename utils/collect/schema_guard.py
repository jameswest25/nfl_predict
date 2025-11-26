#!/usr/bin/env python3
"""
Schema Drift Detection

Monitors Statcast data schema changes by tracking column names and types.
Provides alerts when API schema changes that could affect downstream processing.

Exports:
    - ExpectedSchema: Class containing expected column sets
    - check_schema_drift(df: pd.DataFrame, data_type: str) -> bool
    - assert_no_drift(df: pd.DataFrame, data_type: str) -> None
    - get_column_hash(df: pd.DataFrame) -> str
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Final, Set

import pandas as pd

from utils.general.paths import SCHEMA_HASH_FILE

logger = logging.getLogger(__name__)

class ExpectedSchema:
    """Expected schema definitions for testing and validation"""
    
    # NFL PBP core columns (nflfastR via nfl_data_py) - should always be present
    NFL_PBP_CORE: Set[str] = {
        'game_id', 'season', 'week', 'game_date', 'home_team', 'away_team',
        'qtr', 'down', 'ydstogo', 'yardline_100', 'posteam', 'defteam',
        'yards_gained', 'epa', 'wp', 'touchdown', 'interception', 'fumble', 'penalty',
        'rush', 'pass', 'play_type', 'play_id'
    }
    
    # NFL PBP optional columns - may vary by season/game
    NFL_PBP_OPTIONAL: Set[str] = {
        'series', 'drive', 'game_seconds_remaining', 'score_differential', 'air_yards',
        'complete_pass', 'incomplete_pass', 'posteam_timeouts', 'defteam_timeouts', 
        'roof', 'surface', 'temp', 'wind', 'passer_id', 'passer_player_name',
        'rusher_id', 'rusher_player_name', 'receiver_id', 'receiver_player_name',
        'sack', 'qb_dropback', 'shotgun', 'no_huddle', 'success', 'wpa',
        'home_score', 'away_score', 'posteam_score', 'defteam_score'
    }
    
    # Weather columns added by pipeline
    WEATHER_COLUMNS: Set[str] = {
        'temp', 'humidity', 'windspeed', 'winddir', 'windgust',
        'pressure', 'conditions', 'dewpoint', 'cloudcover', 'wind'
    }
    
    # Timestamp columns added by pipeline
    TIMESTAMP_COLUMNS: Set[str] = {
        'utc_ts'
    }
    
    # Legacy MLB columns (kept for backwards compatibility)
    STATCAST_CORE: Set[str] = {
        'game_pk', 'game_date', 'at_bat_number', 'pitch_number',
        'batter', 'pitcher', 'events', 'description', 'pitch_type'
    }
    STATCAST_OPTIONAL: Set[str] = {
        'release_speed', 'effective_speed', 'release_spin_rate',
        'launch_speed', 'launch_angle', 'hit_distance_sc',
        'inning', 'inning_topbot', 'balls', 'strikes', 'outs_when_up'
    }
    PLAYER_COLUMNS: Set[str] = {
        'batter_name', 'pitcher_name', 'batter_side', 'pitcher_handedness'
    }

def get_column_hash(df: pd.DataFrame) -> str:
    """Generate MD5 hash of column names *and dtypes* for schema comparison"""
    # Include dtype so that e.g. float→int changes trigger drift detection
    column_string = "|".join(
        f"{col}:{str(df[col].dtype)}" for col in sorted(df.columns)
    )
    return hashlib.md5(column_string.encode()).hexdigest()

def _load_schema_hash() -> dict:
    """Load schema hash from file"""
    if SCHEMA_HASH_FILE.exists():
        try:
            with open(SCHEMA_HASH_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load schema hash: {e}")
    return {}

def _save_schema_hash(schema_hash: dict):
    """Save schema hash to file"""
    try:
        # Ensure directory exists
        SCHEMA_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SCHEMA_HASH_FILE, 'w') as f:
            json.dump(schema_hash, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save schema hash: {e}")

def check_schema_drift(df: pd.DataFrame, data_type: str) -> bool:
    """Check for schema drift and handle accordingly"""
    if df.empty:
        return False
    
    current_hash = get_column_hash(df)
    stored_hashes = _load_schema_hash()
    
    if data_type in stored_hashes:
        if stored_hashes[data_type] != current_hash:
            logger.warning(f"Schema drift detected for {data_type}!")
            logger.warning(f"Previous hash: {stored_hashes[data_type]}")
            logger.warning(f"Current hash: {current_hash}")
            logger.warning(f"Current columns: {sorted(df.columns)}")
            
            # Update hash and trigger re-collection flag
            stored_hashes[data_type] = current_hash
            _save_schema_hash(stored_hashes)
            return True
    else:
        # First time seeing this data type
        stored_hashes[data_type] = current_hash
        _save_schema_hash(stored_hashes)
        logger.info(f"Recorded initial schema hash for {data_type}")
    
    return False

def assert_no_drift(df: pd.DataFrame, data_type: str):
    """Assert that no schema drift has occurred, raise exception if it has"""
    if check_schema_drift(df, data_type):
        raise ValueError(f"Schema drift detected for {data_type}. "
                        f"Current columns: {sorted(df.columns)}")

def validate_expected_columns(df: pd.DataFrame, data_type: str = "statcast") -> dict:
    """Validate DataFrame against expected schema and return analysis"""
    df_columns = set(df.columns)
    
    if data_type == "statcast":
        missing_core = ExpectedSchema.STATCAST_CORE - df_columns
        extra_columns = df_columns - (
            ExpectedSchema.STATCAST_CORE | 
            ExpectedSchema.STATCAST_OPTIONAL |
            ExpectedSchema.WEATHER_COLUMNS |
            ExpectedSchema.PLAYER_COLUMNS |
            ExpectedSchema.TIMESTAMP_COLUMNS
        )
        
        analysis = {
            "has_core_columns": len(missing_core) == 0,
            "missing_core": list(missing_core),
            "extra_columns": list(extra_columns),
            "total_columns": len(df_columns),
            "core_coverage": len(ExpectedSchema.STATCAST_CORE & df_columns) / len(ExpectedSchema.STATCAST_CORE)
        }
        
        if missing_core:
            logger.warning(f"Missing core {data_type} columns: {missing_core}")
        if extra_columns:
            logger.info(f"Extra columns in {data_type}: {extra_columns}")
        
        return analysis
    if data_type == "nfl_pbp":
        missing_core = ExpectedSchema.NFL_PBP_CORE - df_columns
        extra_columns = df_columns - (
            ExpectedSchema.NFL_PBP_CORE |
            ExpectedSchema.NFL_PBP_OPTIONAL |
            ExpectedSchema.WEATHER_COLUMNS |
            ExpectedSchema.TIMESTAMP_COLUMNS
        )
        analysis = {
            "has_core_columns": len(missing_core) == 0,
            "missing_core": list(missing_core),
            "extra_columns": list(extra_columns),
            "total_columns": len(df_columns),
            "core_coverage": len(ExpectedSchema.NFL_PBP_CORE & df_columns) / max(1, len(ExpectedSchema.NFL_PBP_CORE))
        }
        if missing_core:
            logger.warning(f"Missing core {data_type} columns: {missing_core}")
        if extra_columns:
            logger.info(f"Extra columns in {data_type}: {extra_columns}")
        return analysis
    
    return {"error": f"Unknown data type: {data_type}"}

def get_schema_summary(df: pd.DataFrame) -> dict:
    """Get a summary of DataFrame schema for debugging"""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "null_counts": df.isnull().sum().to_dict()
    } 