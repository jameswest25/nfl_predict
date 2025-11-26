#!/usr/bin/env python3
"""
SQLite Weather Cache Management

Manages SQLite-based weather cache operations with automatic migration from JSON format.
Integrates with Visual Crossing weather API for efficient caching and retrieval.

Exports:
    - init_db() -> None
    - load_cache(teams: List[str] = None, start_ts: int = None, end_ts: int = None) -> dict
    - save_cache(cache_deltas: dict) -> None
    - migrate_json_cache() -> None
    - attach_weather(df: pd.DataFrame) -> pd.DataFrame
"""

import json
import logging
import sqlite3
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.general.paths import WEATHER_CACHE_FILE, DATA_CACHE

logger = logging.getLogger(__name__)

# Constants
WEATHER_CACHE_JSON = DATA_CACHE / "weather_cache.json"  # For migration only

def init_db():
    """Initialize SQLite weather cache database and create necessary table and index"""
    with sqlite3.connect(WEATHER_CACHE_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weather_cache (
                team TEXT NOT NULL,
                ts_utc INTEGER NOT NULL,
                blob BLOB NOT NULL,
                PRIMARY KEY (team, ts_utc)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_team_ts ON weather_cache(team, ts_utc)")
    logger.debug("Weather cache database initialized")

def migrate_json_cache():
    """Migrate existing JSON weather cache to SQLite format, logging the number of entries moved and backing up the JSON file"""
    if not WEATHER_CACHE_JSON.exists():
        return  # No JSON cache to migrate
    
    if WEATHER_CACHE_FILE.exists():
        return  # Already migrated
    
    logger.info("Migrating JSON weather cache to SQLite...")
    
    try:
        # Load JSON cache
        with open(WEATHER_CACHE_JSON, 'r') as f:
            json_cache = json.load(f)
        
        # Initialize SQLite DB
        init_db()
        
        # Convert and save to SQLite
        save_cache(json_cache)
        
        # Backup and remove JSON file
        backup_file = WEATHER_CACHE_JSON.with_suffix('.json.backup')
        WEATHER_CACHE_JSON.rename(backup_file)
        
        logger.info(f"Migration completed: {len(json_cache)} entries moved to SQLite, JSON backed up to {backup_file.name}")
        
    except Exception as e:
        logger.error(f"Error migrating weather cache: {e}")

def load_cache(teams: List[str] = None, start_ts: int = None, end_ts: int = None) -> dict:
    """Load weather cache entries from SQLite with optional filtering by teams and timestamps, handling decompression and logging the number of entries loaded"""
    if not WEATHER_CACHE_FILE.exists():
        return {}
    
    cache = {}
    try:
        with sqlite3.connect(WEATHER_CACHE_FILE) as conn:
            query = "SELECT team, ts_utc, blob FROM weather_cache"
            params = []
            
            conditions = []
            if teams:
                conditions.append(f"team IN ({','.join(['?'] * len(teams))})")
                params.extend(teams)
            if start_ts is not None:
                conditions.append("ts_utc >= ?")
                params.append(start_ts)
            if end_ts is not None:
                conditions.append("ts_utc <= ?")
                params.append(end_ts)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            for team, ts_utc, blob in conn.execute(query, params):
                try:
                    data = json.loads(zlib.decompress(blob).decode())
                    timestamp_str = pd.to_datetime(ts_utc, unit='s', utc=True).strftime('%Y-%m-%dT%H:%M:%S')
                    cache_key = f"{team}_{timestamp_str}"
                    cache[cache_key] = data
                except Exception as e:
                    logger.warning(f"Could not decompress weather cache entry for {team}: {e}")
        
        logger.info(f"Loaded {len(cache)} weather cache entries from SQLite")
        return cache
    except Exception as e:
        logger.error(f"Error loading weather cache from SQLite: {e}")
        return {}

def save_cache(cache_deltas: dict):
    """Save weather cache deltas to SQLite, compressing the data and logging the number of entries saved"""
    if not cache_deltas:
        return
    
    init_db()
    
    try:
        with sqlite3.connect(WEATHER_CACHE_FILE) as conn:
            for cache_key, weather_data in cache_deltas.items():
                team, timestamp_str = cache_key.split('_', 1)
                ts_utc = int(pd.to_datetime(timestamp_str, utc=True).timestamp())
                blob = zlib.compress(json.dumps(weather_data).encode())
                
                conn.execute(
                    "INSERT OR REPLACE INTO weather_cache (team, ts_utc, blob) VALUES (?, ?, ?)",
                    (team, ts_utc, blob)
                )
        
        logger.info(f"Saved {len(cache_deltas)} weather cache entries to SQLite")
    except Exception as e:
        logger.error(f"Error saving weather cache to SQLite: {e}")

def attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Visual Crossing weather data to DataFrame, efficiently loading cache based on unique teams and timestamps"""
    try:
        # Ensure project root is on sys.path so 'utils.collect.visual_crossing_weather' resolves to our local module
        import sys as _sys
        from utils.general.paths import PROJ_ROOT as _PR
        pr_str = str(_PR)
        if pr_str not in _sys.path:
            _sys.path.insert(0, pr_str)

        import importlib as _imp
        # Import function and get module object (relative import avoids external name clash)
        from .visual_crossing_weather import attach_visual_crossing_weather
        weather_module = _imp.import_module('.visual_crossing_weather', package=__package__)
        
        # Determine location key column (neutral-site aware)
        location_col = 'stadium_key' if 'stadium_key' in df.columns else 'home_team'
        locations = df[location_col].dropna().astype(str).unique().tolist()

        weather_df = df.copy()
        if location_col != 'home_team':
            weather_df['weather_team_key'] = weather_df[location_col]
 
        if 'utc_ts' in df.columns:
            ts_series = df['utc_ts'].dropna()
            if not ts_series.empty:
                start_ts = int(ts_series.min().timestamp())
                end_ts = int(ts_series.max().timestamp())
            else:
                start_ts = end_ts = None
        else:
            start_ts = end_ts = None
        
        # Load weather cache efficiently from SQLite
        logger.info(f"Loading weather cache for {len(locations)} venues, time range: {start_ts}-{end_ts}")
 
        # Monkey-patch the weather module (already imported above) to use SQLite cache
        
        # Save original functions
        original_load_cache = getattr(weather_module, 'load_weather_cache', None)
        original_save_cache = getattr(weather_module, 'save_weather_cache', None)
        
        # Replace with SQLite versions
        weather_module.load_weather_cache = lambda: load_cache(locations, start_ts, end_ts)
        weather_module.save_weather_cache = save_cache
        
        try:
            result = attach_visual_crossing_weather(weather_df)
        finally:
            # Restore original functions
            if original_load_cache:
                weather_module.load_weather_cache = original_load_cache
            if original_save_cache:
                weather_module.save_weather_cache = original_save_cache
        
        return result
        
    except ImportError:
        # Module intentionally optional – skip without noisy warning
        logger.info("Skipping weather attachment: visual_crossing_weather module not available.")
        return df
    except Exception as e:
        logger.error(f"Error attaching weather data: {e}")
        return df 