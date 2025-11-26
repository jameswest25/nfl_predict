#!/usr/bin/env python3
"""
NFL Schedule Management

Handles fetching and caching NFL game schedules using nfl_data_py.
Provides functions to get upcoming games, game start times, and schedule metadata.

Exports:
    - get_schedule(seasons: list[int]) -> pd.DataFrame
    - get_upcoming_games(date: dt.date) -> pd.DataFrame
    - get_game_start_time(game_id: str) -> pd.Timestamp
    - cache_schedules(seasons: list[int]) -> None
"""

import logging
from pathlib import Path
from typing import List, Optional
import datetime as dt

import pandas as pd

try:
    from nfl_data_py import import_schedules
except ImportError:
    import_schedules = None

from utils.general.paths import COLLECT_CACHE_DIR

logger = logging.getLogger(__name__)

# Cache file for schedules
SCHEDULE_CACHE_FILE = COLLECT_CACHE_DIR / "nfl_schedules.parquet"

def get_schedule(seasons: List[int], force_refresh: bool = False) -> pd.DataFrame:
    """
    Get NFL schedules for specified seasons with caching.
    
    Parameters
    ----------
    seasons : list[int]
        List of NFL seasons (e.g., [2024, 2023])
    force_refresh : bool, default False
        If True, bypass cache and fetch fresh data
        
    Returns
    -------
    pd.DataFrame
        Schedule data with columns: game_id, season, week, gameday, home_team, 
        away_team, game_time_eastern, start_time_utc, roof, surface, etc.
    """
    if import_schedules is None:
        logger.error("nfl_data_py not installed. Install with: pip install nfl-data-py")
        return pd.DataFrame()
    
    # Try loading from cache first
    if not force_refresh and SCHEDULE_CACHE_FILE.exists():
        try:
            cached = pd.read_parquet(SCHEDULE_CACHE_FILE)
            cached_seasons = set(cached['season'].unique())
            requested_seasons = set(seasons)
            
            # Check if cache has all requested seasons
            if requested_seasons.issubset(cached_seasons):
                logger.info(f"Loaded schedules from cache for seasons: {sorted(seasons)}")
                return cached[cached['season'].isin(seasons)]
            else:
                missing = requested_seasons - cached_seasons
                logger.info(f"Cache missing seasons: {sorted(missing)}, refreshing...")
        except Exception as e:
            logger.warning(f"Could not load schedule cache: {e}")
    
    # Fetch from nfl_data_py
    logger.info(f"Fetching NFL schedules for seasons: {sorted(seasons)}")
    try:
        schedule = import_schedules(years=seasons)
        
        if schedule is None or len(schedule) == 0:
            logger.warning("No schedule data returned")
            return pd.DataFrame()
        
        # Standardize column names
        schedule = _standardize_schedule_columns(schedule)
        
        # Cache the results
        _update_schedule_cache(schedule)
        
        return schedule
        
    except Exception as e:
        logger.error(f"Failed to fetch NFL schedules: {e}")
        return pd.DataFrame()

def _standardize_schedule_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize schedule column names and types"""
    # Ensure core columns exist
    core_cols = ['game_id', 'season', 'week', 'gameday', 'home_team', 'away_team']
    for col in core_cols:
        if col not in df.columns:
            logger.warning(f"Missing expected column: {col}")
            df[col] = pd.NA
    
    # Parse dates
    if 'gameday' in df.columns:
        df['gameday'] = pd.to_datetime(df['gameday'], errors='coerce').dt.date
    
    # Create UTC timestamp from game_time_eastern if available
    if 'gametime' in df.columns or 'game_time_eastern' in df.columns:
        time_col = 'gametime' if 'gametime' in df.columns else 'game_time_eastern'
        
        # Combine gameday + time to create start_time_utc
        try:
            dt_str = df['gameday'].astype(str) + ' ' + df[time_col].astype(str)
            df['start_time_utc'] = (
                pd.to_datetime(dt_str, errors='coerce')
                .dt.tz_localize('US/Eastern', nonexistent='shift_forward', ambiguous='NaT')
                .dt.tz_convert('UTC')
            )
        except Exception as e:
            logger.warning(f"Could not create start_time_utc: {e}")
            df['start_time_utc'] = pd.NaT
    
    return df

def _update_schedule_cache(new_data: pd.DataFrame) -> None:
    """Update the schedule cache with new data"""
    if SCHEDULE_CACHE_FILE.exists():
        try:
            existing = pd.read_parquet(SCHEDULE_CACHE_FILE)
            # Combine and deduplicate by game_id
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=['game_id'], keep='last')
            combined.to_parquet(SCHEDULE_CACHE_FILE, index=False)
            logger.info(f"Updated schedule cache: {len(combined)} total games")
        except Exception as e:
            logger.error(f"Error updating schedule cache: {e}")
            # Fallback: save new data only
            new_data.to_parquet(SCHEDULE_CACHE_FILE, index=False)
    else:
        # Ensure directory exists
        SCHEDULE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        new_data.to_parquet(SCHEDULE_CACHE_FILE, index=False)
        logger.info(f"Created schedule cache with {len(new_data)} games")

def get_upcoming_games(date: dt.date, seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Get games scheduled for a specific date.
    
    Parameters
    ----------
    date : datetime.date
        Date to get games for
    seasons : list[int], optional
        Seasons to search (defaults to current season)
        
    Returns
    -------
    pd.DataFrame
        Schedule rows for games on the specified date
    """
    if seasons is None:
        # Default to current NFL season (starts in September)
        year = date.year
        if date.month < 3:  # Jan-Feb are previous season's playoffs
            year -= 1
        seasons = [year]
    
    schedule = get_schedule(seasons)
    
    if schedule.empty:
        return pd.DataFrame()
    
    # Filter by date
    if 'gameday' in schedule.columns:
        schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce').dt.date
        games = schedule[schedule['gameday'] == date]
        logger.info(f"Found {len(games)} games scheduled for {date}")
        return games
    else:
        logger.warning("Schedule missing 'gameday' column")
        return pd.DataFrame()

def get_game_start_time(game_id: str, seasons: Optional[List[int]] = None) -> Optional[pd.Timestamp]:
    """
    Get UTC start time for a specific game.
    
    Parameters
    ----------
    game_id : str
        NFL game ID
    seasons : list[int], optional
        Seasons to search (defaults to current season)
        
    Returns
    -------
    pd.Timestamp or None
        UTC timestamp of game start, or None if not found
    """
    if seasons is None:
        # Extract season from game_id if possible (format: YYYY_WW_AWAY_HOME)
        try:
            season = int(game_id.split('_')[0])
            seasons = [season]
        except:
            year = dt.date.today().year
            seasons = [year, year - 1]
    
    schedule = get_schedule(seasons)
    
    if schedule.empty:
        return None
    
    game = schedule[schedule['game_id'] == game_id]
    
    if game.empty:
        logger.warning(f"Game {game_id} not found in schedule")
        return None
    
    if 'start_time_utc' in game.columns:
        return game.iloc[0]['start_time_utc']
    else:
        logger.warning("Schedule missing 'start_time_utc' column")
        return None

def cache_schedules(seasons: List[int]) -> None:
    """
    Pre-cache schedules for specified seasons.
    
    Useful for ensuring schedules are available for offline analysis.
    
    Parameters
    ----------
    seasons : list[int]
        Seasons to cache
    """
    logger.info(f"Pre-caching schedules for seasons: {sorted(seasons)}")
    get_schedule(seasons, force_refresh=True)
    logger.info("Schedule caching complete")

def get_game_metadata(game_id: str, seasons: Optional[List[int]] = None) -> dict:
    """
    Get metadata for a specific game (teams, venue, conditions, etc.).
    
    Parameters
    ----------
    game_id : str
        NFL game ID
    seasons : list[int], optional
        Seasons to search
        
    Returns
    -------
    dict
        Game metadata including home_team, away_team, stadium, roof, surface, etc.
    """
    if seasons is None:
        try:
            season = int(game_id.split('_')[0])
            seasons = [season]
        except:
            year = dt.date.today().year
            seasons = [year, year - 1]
    
    schedule = get_schedule(seasons)
    
    if schedule.empty:
        return {}
    
    game = schedule[schedule['game_id'] == game_id]
    
    if game.empty:
        return {}
    
    # Return relevant metadata as dict
    meta = game.iloc[0].to_dict()
    
    # Filter to useful fields
    useful_fields = [
        'game_id', 'season', 'week', 'gameday', 
        'home_team', 'away_team', 'home_score', 'away_score',
        'location', 'roof', 'surface', 'temp', 'wind',
        'home_coach', 'away_coach', 'stadium'
    ]
    
    return {k: v for k, v in meta.items() if k in useful_fields}

