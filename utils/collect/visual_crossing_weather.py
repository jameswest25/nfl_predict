#!/usr/bin/env python3
"""
Visual Crossing Weather API Integration (Production Ready)

This module provides functions to fetch weather data from Visual Crossing API
with intelligent caching and batching to minimize API calls and improve performance.
Auto-detects plan capabilities (minute or hour level data) and adjusts accordingly.

Usage:
    from utils.collect.visual_crossing_weather import attach_visual_crossing_weather
    df_with_weather = attach_visual_crossing_weather(df)
"""

import json
import logging
import requests
import pandas as pd
import numpy as np
import io
import sys
from datetime import timedelta, datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import concurrent.futures
from functools import lru_cache
from collections import defaultdict
import random
import threading
import os
import pytz
from utils.general.paths import PROJ_ROOT, STADIUM_COORDS_FILE, COLLECT_CACHE_DIR
from pathlib import Path as _Path
import yaml as _yaml

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
# Read API key from env; fail fast on missing/401
import os as _os
def _load_api_key() -> str:
    key = _os.getenv("VISUAL_CROSSING_API_KEY", "")
    if key:
        return key
    try:
        cfg_fp = PROJ_ROOT / "config" / "config.yaml"
        cfg = _yaml.safe_load(cfg_fp.read_text())
        w = (cfg or {}).get("weather", {})
        return str(w.get("api_key", ""))
    except Exception:
        return ""
VISUAL_CROSSING_API_KEY = _load_api_key()
VISUAL_CROSSING_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 60  # seconds
MAX_FETCH_WORKERS = 2  # For concurrent API workers
MAX_MERGE_WORKERS = 4  # For concurrent merge operations
RATE_LIMIT_DELAY = 0.5  # seconds between API calls to avoid 429
MAX_BACKOFF_TIME = 300  # Maximum backoff time in seconds (5 minutes)
UNIT_GROUP = "us"  # "us" for imperial (°F, mph), "metric" for metric (°C, km/h)
USER_AGENT = "nfl_predict/1.0 (james@example.com)"  # Custom user agent for support

# Note: The ML model expects imperial units. If using metric, convert values accordingly.

# Quota management
QUOTA_ERROR_THRESHOLD = 5  # 403 errors before graceful exit
QUOTA_ERROR_WINDOW = 60  # Time window for error counting (seconds)

# Thread-local storage for HTTP sessions
thread_local = threading.local()

def get_session():
    """Get thread-local HTTP session with proper headers"""
    if not hasattr(thread_local, 'sess'):
        thread_local.sess = requests.Session()
        # Temporarily remove User-Agent to test if it's causing 401 errors
        # thread_local.sess.headers['User-Agent'] = USER_AGENT
    return thread_local.sess

# Quota error tracking for graceful exit
quota_error_times = []

# Teams with retractable roofs or domes (wind data not applicable)
# Updated to reflect NFL domes/retractable roofs by home team abbr (best-effort)
ROOF_CLOSED_TEAMS = {
    'ARI': True,  # State Farm Stadium (retractable roof)
    'ATL': True,  # Mercedes-Benz Stadium (fixed roof)
    'DAL': True,  # AT&T Stadium (retractable roof)
    'DET': True,  # Ford Field (fixed roof)
    'HOU': True,  # NRG Stadium (retractable roof)
    'IND': True,  # Lucas Oil Stadium (retractable roof)
    'LAR': True,  # SoFi (indoor)
    'LVR': True,  # Allegiant Stadium (domed)
    'MIN': True,  # U.S. Bank Stadium (fixed roof)
    'NO': True,   # Caesars Superdome (fixed roof)
}

# Core weather fields for ML (exclude rarely-used fields to reduce cache size)
# Note: windspeed/windgust are in mph, temp/feelslike are in °F
# Handle field name variations in Visual Crossing API
CORE_WEATHER_FIELDS = [
    'temp', 'feelslike', 'humidity', 'dewpoint',  # API switched to dewpoint
    'pressure',
    'windspeed', 'windgust', 'winddir',
    'cloudcover',
    'visibility', 'precip', 'precipprob', 'snow', 'preciptype',
    'conditions'
]

def check_quota_exhaustion():
    """Check if we've hit the quota error threshold for graceful exit"""
    global quota_error_times
    
    # Clean old errors outside the window
    current_time = time.time()
    quota_error_times[:] = [t for t in quota_error_times if current_time - t < QUOTA_ERROR_WINDOW]
    
    # Check if we've hit the threshold
    if len(quota_error_times) >= QUOTA_ERROR_THRESHOLD:
        logging.error(f"Quota exhausted: {len(quota_error_times)} 403 errors in {QUOTA_ERROR_WINDOW}s. Resuming tomorrow.")
        return True
    return False

def record_quota_error():
    """Record a quota error for tracking"""
    global quota_error_times
    quota_error_times.append(time.time())

def handle_quota_exhaustion():
    """Handle daily quota exhaustion with graceful exit"""
    next_start = (datetime.now(timezone.utc).date() + timedelta(days=1))
    logging.error(f"Daily quota exhausted; resume after 00:00 UTC on {next_start}")
    sys.exit(0)

@lru_cache(maxsize=1)
def load_stadium_coords() -> Dict:
    """Load stadium coordinates from JSON file (cached)"""
    try:
        with open(STADIUM_COORDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading stadium coordinates: {e}")
        return {}

def _weather_cache_path() -> Path:
    try:
        cfg_fp = PROJ_ROOT / "config" / "config.yaml"
        cfg = _yaml.safe_load(cfg_fp.read_text())
        w = (cfg or {}).get("weather", {})
        fname = str(w.get("cache_file", "weather_cache.json"))
        return COLLECT_CACHE_DIR / fname
    except Exception:
        return COLLECT_CACHE_DIR / "weather_cache.json"

def load_weather_cache() -> Dict:
    """Load existing weather cache from file with explicit dtype handling"""
    cache_fp = _weather_cache_path()
    if cache_fp.exists():
        try:
            with open(cache_fp, 'r') as f:
                cache = json.load(f)
            
            # Ensure proper dtypes for numeric fields
            for key, value in cache.items():
                for field in CORE_WEATHER_FIELDS:
                    if field in value and value[field] is not None:
                        if field in ['temp', 'humidity', 'windspeed', 'winddir', 'windgust', 'feelslike', 'pressure', 'cloudcover', 'precip', 'precipprob', 'dewpoint']:
                            value[field] = float(value[field]) if value[field] else None
                
                # Handle field fallbacks (dew/dewpoint)
                if 'dewpoint' not in value and 'dew' in value:
                    value['dewpoint'] = value.get('dew')
            
            return cache
        except Exception as e:
            logging.warning(f"Error loading weather cache: {e}")
            return {}
    return {}

def save_weather_cache(cache: Dict):
    """Save weather cache to file with atomic write and proper float conversion"""
    try:
        # Ensure cache directory exists
        cache_fp = _weather_cache_path()
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        cache_serializable = {}
        for key, value in cache.items():
            cache_serializable[key] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in value.items()
            }
        
        # Atomic write: write to temp file first, then replace
        temp_file = cache_fp.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(cache_serializable, f, indent=2)
        
        # Atomic replace
        temp_file.replace(cache_fp)
        logging.info(f"Weather cache saved with {len(cache)} entries")
    except Exception as e:
        logging.error(f"Error saving weather cache: {e}")

def get_cache_key(team: str, bucket_timestamp: pd.Timestamp) -> str:
    """Generate cache key for team and bucket timestamp"""
    return f"{team}_{bucket_timestamp.strftime('%Y-%m-%dT%H:%M:%S')}"

def fetch_weather_batch(lat: float, lon: float, start_time, end_time, team: str, tz: str = 'US/Eastern') -> Optional[Dict]:
    """
    Fetch weather data for a time range with retry logic and rate limiting
    
    Args:
        lat: Latitude
        lon: Longitude  
        start_time: Start time for weather data (UTC)
        end_time: End time for weather data (UTC)
        team: Team abbreviation for logging
        tz: Timezone for the stadium location
        
    Returns:
        Dictionary with weather data by bucket or None if error
        
    Note:
        Auto-detects plan capabilities (minute or hour level data) from API response
    """
    logging.info(f"=== FETCH WEATHER BATCH {team} START ===")
    logging.info(f"Team {team}: lat={lat:.4f}, lon={lon:.4f}, tz={tz}")
    logging.info(f"Team {team}: time range {start_time} to {end_time}")
    
    if os.environ.get('VC_FAKE') == '1':
        logging.info(f"Using fake weather data for {team} (VC_FAKE=1)")
        return {
            start_time.floor('60min'): {
                'temp': 72.0,
                'humidity': 65.0,
                'windspeed': 8.0,
                'winddir': 180.0,
                'conditions': 'Partly cloudy'
            }
        }
    
    # Convert UTC times to local times for API call
    # The API expects dates in the local timezone of the location
    start_local = start_time.tz_convert(tz)
    end_local = end_time.tz_convert(tz)
    
    logging.info(f"Team {team}: API call start_local={start_local}, end_local={end_local}, start_utc={start_time}, end_utc={end_time}")
    
    for attempt in range(MAX_RETRIES):
        try:
            if check_quota_exhaustion():
                logging.error(f"Team {team}: Quota exhausted, stopping")
                return None
            if attempt > 0:
                time.sleep(RATE_LIMIT_DELAY * (2 ** (attempt - 1)))
            # Try minute-level data first
            start_str = start_local.strftime('%Y-%m-%dT%H:%M:%S')
            end_str = end_local.strftime('%Y-%m-%dT%H:%M:%S')
            url = f"{VISUAL_CROSSING_BASE_URL}/{lat},{lon}/{start_str}/{end_str}"
            params = {
                'key': VISUAL_CROSSING_API_KEY,
                'unitGroup': UNIT_GROUP,
                'include': 'minutes,hours',
                'contentType': 'json'
            }
            if attempt == 0:
                logging.debug(f"Visual Crossing request URL: {url}")
                logging.debug(f"Visual Crossing request params: {params}")
                print(f"DEBUG: URL={url}")
                print(f"DEBUG: PARAMS={params}")
            logging.debug(f"Fetching weather data for {team}: {lat}, {lon} from {start_str} to {end_str}")
            if not VISUAL_CROSSING_API_KEY:
                logging.warning("Visual Crossing API key missing (VISUAL_CROSSING_API_KEY). Skipping.")
                return None
            response = get_session().get(url, params=params, timeout=30)
            logging.info(f"Team {team}: API response status: {response.status_code}")
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 30))
                base_wait = min(retry_after * (2 ** attempt), MAX_BACKOFF_TIME)
                jitter = random.uniform(0, 0.1)
                wait_time = base_wait * (1 + jitter)
                logging.warning(f"Rate limited for {team}. Retry-After: {retry_after}s, waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(wait_time)
                continue
            elif response.status_code == 403:
                logging.error(f"Visual Crossing daily quota exhausted for {team}: {response.status_code}")
                record_quota_error()
                if len(quota_error_times) >= 3:
                    handle_quota_exhaustion()
                return None
            elif response.status_code in (400, 401):
                # Fallback: try hourly data only (date in path, hours only)
                logging.info(f"Plan for {team} doesn't support minute data, falling back to hourly data")
                start_str = start_local.strftime('%Y-%m-%d')
                end_str = end_local.strftime('%Y-%m-%d')
                url = f"{VISUAL_CROSSING_BASE_URL}/{lat},{lon}/{start_str}/{end_str}"
                params['include'] = 'hours'
                # Remove time from params if present
                response = get_session().get(url, params=params, timeout=30)
                logging.info(f"Team {team}: Fallback API response status: {response.status_code}")
                if response.status_code != 200:
                    logging.error(f"Fallback to hourly data also failed for {team}: {response.status_code}")
                    return None
                # else, continue to parse as normal
            else:
                response.raise_for_status()
            
            data = response.json()
            logging.info(f"Team {team}: API response keys: {list(data.keys())}")
            
            plan_has_minutes = any(
                'minutes' in h and h['minutes'] for d in data.get('days', []) for h in d.get('hours', [])
            )
            bucket_minutes = 5 if plan_has_minutes else 60
            logging.info(f"Auto-detected plan for {team}: {'minute' if plan_has_minutes else 'hour'}-level data (bucket: {bucket_minutes} minutes)")
            
            weather_by_bucket = {}
            if 'days' in data and len(data['days']) > 0:
                logging.info(f"Team {team}: Processing {len(data['days'])} days")
                # DEBUG: Log the actual dates from the API response
                api_dates = [day_data.get('datetime', 'unknown') for day_data in data['days']]
                logging.info(f"Team {team}: API returned dates: {api_dates}")
                
                for day_data in data['days']:
                    if 'hours' in day_data:
                        logging.info(f"Team {team}: Processing {len(day_data['hours'])} hours for day {day_data.get('datetime', 'unknown')}")
                        for hour_data in day_data['hours']:
                            records = hour_data.get('minutes', []) if plan_has_minutes else [hour_data]
                            logging.info(f"Team {team}: Processing {len(records)} records for hour {hour_data.get('datetime', 'unknown')}")
                            for record in records:
                                # CRITICAL FIX: Visual Crossing API returns hour records with just time
                                # (e.g., "14:00:00") not full datetime. We need to combine with the day date.
                                day_date = day_data.get('datetime')  # e.g., "2025-06-17"
                                hour_time = record['datetime']       # e.g., "14:00:00" 
                                
                                # Combine day date + hour time to get proper local datetime
                                full_datetime_str = f"{day_date} {hour_time}"
                                ts = pd.to_datetime(full_datetime_str, errors='coerce')
                                if pd.isna(ts):
                                    logging.warning(f"Team {team}: Failed to parse datetime '{full_datetime_str}'")
                                    continue
                                
                                # DEBUG: Log the raw timestamp construction
                                logging.debug(f"Team {team}: Day={day_date}, Hour={hour_time}, Combined={full_datetime_str} -> {ts}")
                                
                                # Localize to team's timezone (Visual Crossing returns naive local time)
                                ts = ts.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
                                if pd.isna(ts):
                                    logging.warning(f"Team {team}: Localizing datetime produced NaT for '{full_datetime_str}' (tz={tz})")
                                    continue
                                
                                # Convert to UTC for bucketing and storage
                                ts = ts.tz_convert('UTC')
                                if pd.isna(ts):
                                    logging.warning(f"Team {team}: UTC conversion produced NaT for '{full_datetime_str}'")
                                    continue
                                
                                # Floor to bucket size (already in UTC)
                                bucket_time = ts.floor(f'{bucket_minutes}min')
                                
                                # DEBUG: Log the final bucket time
                                logging.debug(f"Team {team}: Final bucket time: {bucket_time}")
                                
                                # Always include all hourly records (let merge_asof handle matching)
                                if record.get('winddir') == '':
                                    record['winddir'] = None
                                if bucket_time not in weather_by_bucket:
                                    record = convert_to_imperial(record)
                                    
                                    # AGENT FIX: Filter out None values and use windgust fallback for windspeed
                                    weather_data = {}
                                    for field in CORE_WEATHER_FIELDS:
                                        if field in record and record[field] is not None:
                                            weather_data[field] = record[field]
                                    
                                    # Special handling for windspeed: use windgust as fallback if windspeed is None
                                    if 'windspeed' not in weather_data and record.get('windgust') is not None:
                                        # Use a conservative estimate: windgust is typically 1.3-1.5x windspeed
                                        # So windspeed ≈ windgust / 1.4
                                        estimated_windspeed = record['windgust'] / 1.4
                                        weather_data['windspeed'] = round(estimated_windspeed, 1)
                                        logging.debug(f"Team {team}: Using windgust fallback: windgust={record['windgust']} -> windspeed={estimated_windspeed:.1f}")
                                    
                                    weather_by_bucket[bucket_time] = weather_data
                                    
                                    if 'dewpoint' not in weather_by_bucket[bucket_time] and 'dew' in record and record['dew'] is not None:
                                        weather_by_bucket[bucket_time]['dewpoint'] = record['dew']
                                    unexpected_fields = [k for k in record.keys() if k not in CORE_WEATHER_FIELDS and not k.startswith('_')]
                                    if unexpected_fields and random.random() < 0.01:
                                        logging.debug(f"Unexpected fields for {team}: {unexpected_fields}")
                                logging.info(f"Team {team}: API record local={ts}, bucket_time_utc={bucket_time}")
            else:
                logging.warning(f"Team {team}: No 'days' data in API response")
            
            if weather_by_bucket:
                logging.info(f"Team {team}: Fetched {len(weather_by_bucket)} weather buckets ({bucket_minutes}-minute buckets)")
                logging.info(f"Team {team}: Sample bucket data: {list(weather_by_bucket.values())[0] if weather_by_bucket else 'None'}")
                return weather_by_bucket
            else:
                logging.warning(f"Team {team}: No weather data found")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed for {team} (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
        except Exception as e:
            logging.error(f"Error fetching weather data for {team}: {e}")
            return None
    
    logging.error(f"Failed to fetch weather data for {team} after {MAX_RETRIES} attempts")
    logging.info(f"=== FETCH WEATHER BATCH {team} END ===")
    return None

def fetch_weather_for_team_batch(team: str, buckets, cache: Dict) -> Dict:
    """
    Fetch weather data for a team's bucket range, using batching and caching
    
    Args:
        team: Team abbreviation
        buckets: List of bucket timestamps for this team
        cache: Weather cache dictionary
        
    Returns:
        Dictionary with only new cache entries (deltas)
    """
    logging.info(f"=== FETCH WEATHER FOR TEAM {team} START ===")
    logging.info(f"Team {team}: Processing {len(buckets)} buckets")
    logging.info(f"Team {team}: Bucket range: {min(buckets)} to {max(buckets)}")

    # Allow offline runs to skip API calls and rely solely on the existing cache.
    if os.environ.get("VC_SKIP_API") == "1":
        logging.info("VC_SKIP_API=1 set; skipping Visual Crossing API fetch for this team and using cache only")
        return {}
    
    if not buckets:
        logging.warning(f"Team {team}: No buckets to process")
        return {}
    
    # Get team location
    stadium_coords = load_stadium_coords()
    logging.info(f"Team {team}: Stadium coordinates available: {team in stadium_coords}")
    
    if team not in stadium_coords:
        logging.warning(f"No location data for team: {team} - skipping weather data")
        return {}
    
    location = stadium_coords[team]
    tz = location.get('tz', 'US/Eastern')  # Default to Eastern if not specified
    logging.info(f"Team {team}: Location = {location['lat']:.4f}, {location['lon']:.4f}, tz = {tz}")
    
    # Log fallback timezone usage
    if 'tz' not in location:
        logging.warning(f"No tz for {team}. Defaulting to Eastern")
    
    # Check cache for existing buckets
    missing_buckets = []
    for bucket in buckets:
        if pd.isna(bucket):
            continue
        cache_key = get_cache_key(team, bucket)
        if cache_key not in cache:
            missing_buckets.append(bucket)
    
    logging.info(f"Team {team}: {len(missing_buckets)} missing buckets out of {len(buckets)} total")
    
    if not missing_buckets:
        logging.debug(f"All weather data for {team} found in cache")
        return {}
    
    # Fetch missing buckets with proper padding
    if missing_buckets:
        min_time = min(missing_buckets)
        max_time = max(missing_buckets)
        
        # Compute bucket size from first bucket (crude but works)
        first_bucket = min(missing_buckets)
        
        # Use interval analysis to detect bucket size more robustly
        if len(missing_buckets) > 1:
            intervals = pd.Series(missing_buckets).sort_values().diff().dt.total_seconds() // 60
            if len(intervals.dropna()) > 0:
                bucket_minutes = int(intervals.mode()[0])
            else:
                bucket_minutes = 60  # fallback to hourly
        else:
            bucket_minutes = 60  # single bucket, assume hourly
            
        # FIX: Convert UTC bucket times to local timezone to determine the local calendar day
        # This ensures we fetch weather for the entire local day that contains the game times
        local_tz = pytz.timezone(tz)
        
        # Convert UTC bucket times to local time
        min_time_local = min_time.tz_convert(local_tz)
        max_time_local = max_time.tz_convert(local_tz)
        
        # Get the local calendar day range that contains all game times
        # CRITICAL FIX: Start from the PREVIOUS local day to capture late-night games
        # that may have pitches occurring before midnight local time but in the
        # previous UTC day. This handles cases like AZ games starting at 11:15 PM
        # local time where pitches occur at 2025-06-14 23:15 UTC but the local
        # day starts at 2025-06-15 00:00 UTC.
        start_local_date = (min_time_local - pd.Timedelta(days=1)).floor('D')
        # End at the end of the latest local day  
        end_local_date = (max_time_local + pd.Timedelta(days=1)).floor('D')
        
        # Convert back to UTC for the API call
        start_time_utc = start_local_date.tz_convert('UTC')
        end_time_utc = end_local_date.tz_convert('UTC')
        if pd.isna(start_time_utc) or pd.isna(end_time_utc):
            logging.warning(f"{team} – computed time range contains NaT (start={start_time_utc}, end={end_time_utc}); skipping weather fetch.")
            return {}
        
        logging.info(f"Team {team}: UTC bucket range: {min_time} to {max_time}")
        logging.info(f"Team {team}: Local bucket range: {min_time_local} to {max_time_local}")
        logging.info(f"Team {team}: Fetching weather for entire local day(s): {start_local_date.date()} to {end_local_date.date()}")
        logging.info(f"Team {team}: UTC API range: {start_time_utc} to {end_time_utc} (bucket size: {bucket_minutes}min)")
        
        weather_data = fetch_weather_batch(
            location['lat'], 
            location['lon'], 
            start_time_utc, 
            end_time_utc, 
            team,
            tz
        )
        
        if not weather_data:
            logging.warning(f"{team} – API returned no minutes/hours records (check lat/lon or date range)")
            return {}
        
        if weather_data:
            # Return only the new cache entries (deltas)
            new_cache_entries = {}
            for bucket_time, weather in weather_data.items():
                cache_key = get_cache_key(team, bucket_time)
                # Duplicate bucket guard
                if cache_key not in new_cache_entries:
                    new_cache_entries[cache_key] = weather
            
            logging.info(f"Team {team}: Fetched {len(new_cache_entries)} new weather buckets")
            logging.info(f"Team {team}: Sample weather data: {list(new_cache_entries.values())[0] if new_cache_entries else 'None'}")
            return new_cache_entries
    
    logging.info(f"=== FETCH WEATHER FOR TEAM {team} END ===")
    return {}

def merge_weather_for_team(args: Tuple) -> Tuple[pd.Index, pd.DataFrame]:
    """Helper function for parallel merge_asof operations using indices"""
    team, team_indices, team_weather, weather_columns, df = args
    
    if team_weather.empty or len(team_indices) == 0:
        return team_indices, pd.DataFrame()
    
    # Get team data using indices (no copy)
    team_data = df.loc[team_indices].copy()  # Only copy the subset we need
    
    # Filter weather columns to only those that exist in team_weather
    available_weather_cols = [col for col in weather_columns if col in team_weather.columns]
    
    # DEBUG: Show timestamp ranges
    logging.info(f"Team {team} merge debug:")
    logging.info(f"  Game data timestamps: {team_data['utc_ts'].min()} to {team_data['utc_ts'].max()}")
    logging.info(f"  Weather data timestamps: {team_weather['utc_ts'].min()} to {team_weather['utc_ts'].max()}")
    logging.info(f"  Game data sample: {team_data['utc_ts'].head(3).tolist()}")
    logging.info(f"  Weather data sample: {team_weather['utc_ts'].head(3).tolist()}")
    
    # Calculate dynamic tolerance based on bucket size
    # Use interval analysis to detect bucket size more robustly
    interval = (team_weather['utc_ts']
                .sort_values()
                .diff()  # series of timedeltas
                .dt.total_seconds() // 60)
    
    if len(interval.dropna()) > 0:
        bucket_minutes = int(interval.mode()[0])
    else:
        bucket_minutes = 60  # fallback to hourly
    
    # Use the FULL bucket size to ensure every pitch within the bucket matches.
    # Subtracting a minute could exclude pitches occurring in the final seconds
    # of the bucket (e.g., 12:04:59 for a 5-minute bucket).  A zero-second buffer
    # is safe because merge_asof requires the left key (pitch time) to be >= the
    # right key (bucket start); any pitch exactly at the next bucket will match
    # that later bucket instead.
    tolerance = pd.Timedelta(minutes=bucket_minutes)
    logging.info(f"  Bucket minutes: {bucket_minutes}, tolerance: {tolerance}")
    
    # Ensure both DataFrames have the same timezone for merge
    team_data_merge = team_data.copy()
    team_weather_merge = team_weather.copy()
    
    # Convert to UTC if needed and ensure both have the same timezone
    if team_data_merge['utc_ts'].dt.tz is None:
        team_data_merge['utc_ts'] = team_data_merge['utc_ts'].dt.tz_localize('UTC')
    else:
        team_data_merge['utc_ts'] = team_data_merge['utc_ts'].dt.tz_convert('UTC')
        
    if team_weather_merge['utc_ts'].dt.tz is None:
        team_weather_merge['utc_ts'] = team_weather_merge['utc_ts'].dt.tz_localize('UTC')
    else:
        team_weather_merge['utc_ts'] = team_weather_merge['utc_ts'].dt.tz_convert('UTC')
    
    # DEBUG: Show timezone info
    logging.info(f"  Game data tz: {team_data_merge['utc_ts'].dt.tz}")
    logging.info(f"  Weather data tz: {team_weather_merge['utc_ts'].dt.tz}")
    
    # Preserve original indices so we can align the merged result back to the
    # master dataframe after merge_asof. merge_asof constructs a brand-new
    # RangeIndex, so we stash the original labels before sorting.
    team_data_merge['__orig_idx__'] = team_data_merge.index

    # Drop empty weather columns from the COPY *after* adding the helper index
    # (dropping first would lose nothing but keeps the helper column logic
    # clearer).
    team_data_merge = team_data_merge.drop(columns=available_weather_cols, errors='ignore')
    
    # Use merge_asof for approximate timestamp matching
    merged = pd.merge_asof(
        team_data_merge.sort_values('utc_ts'),
        team_weather_merge[['utc_ts'] + available_weather_cols].sort_values('utc_ts'),
        on='utc_ts',
        direction='backward',
        tolerance=tolerance
    )
    
    # ------------------------------------------------------------------
    # Second-pass merge (forward) for any rows that failed to find a match
    # in the backward search.  This catches pitches that occur *before* the
    # first bucket of the day (rare) or that exceeded the tolerance window
    # for some reason (e.g., weird bucket gaps).
    # ------------------------------------------------------------------

    sample_col = available_weather_cols[0] if available_weather_cols else None
    if sample_col is not None:
        missing_mask = merged[sample_col].isna()
        missing_count = missing_mask.sum()
        if missing_count:
            logging.info(
                f"  {missing_count} rows still without weather after backward merge; running forward merge pass"
            )
            forward_merged = pd.merge_asof(
                team_data_merge.sort_values('utc_ts'),
                team_weather_merge[['utc_ts'] + available_weather_cols].sort_values('utc_ts'),
                on='utc_ts',
                direction='forward',
                tolerance=tolerance
            )

            # Combine: fill only the rows that were missing
            for col in available_weather_cols:
                merged.loc[missing_mask, col] = merged.loc[missing_mask, col].fillna(
                    forward_merged.loc[missing_mask, col]
                )
            filled_after = merged[sample_col].notna().sum()
            added = filled_after - (len(merged) - missing_count)
            logging.info(
                f"  Forward merge filled {added} additional rows"
            )
    
    # Restore original indices for reliable assignment back to the main df.
    if '__orig_idx__' in merged.columns:
        merged.set_index('__orig_idx__', inplace=True)
        merged.index.name = None  # match df default
        merged.drop(columns=['__orig_idx__'], inplace=True, errors='ignore')
    
    # DEBUG: Show merge results
    logging.info(f"  Merge result: {len(merged)} rows (original: {len(team_data_merge)})")
    if not merged.empty:
        weather_cols_found = [col for col in available_weather_cols if col in merged.columns and merged[col].notna().any()]
        logging.info(f"  Weather columns with data: {weather_cols_found}")
        if weather_cols_found:
            sample_col = weather_cols_found[0]
            non_null_count = merged[sample_col].notna().sum()
            logging.info(f"  Sample column '{sample_col}': {non_null_count}/{len(merged)} non-null values")
    else:
        logging.warning(f"  Merge returned empty DataFrame!")
    
    return team_indices, merged

def attach_visual_crossing_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach Visual Crossing weather data to Statcast dataframe (production ready)
    
    Args:
        df: DataFrame with Statcast data
        
    Returns:
        DataFrame with weather data attached
        
    Note:
        Auto-detects plan capabilities (minute or hour level data) and adjusts
        bucket sizes and tolerances accordingly
    """
    logging.info("=== WEATHER INTEGRATION DEBUG START ===")
    logging.info(f"Input DataFrame shape: {df.shape}")
    logging.info(f"Input DataFrame columns: {list(df.columns)}")

    df = df.copy()
    restore_home_team = False
    if 'weather_team_key' in df.columns:
        df['_weather_home_backup'] = df.get('home_team')
        df['home_team'] = df['weather_team_key']
        restore_home_team = True

    if 'utc_ts' not in df.columns or 'home_team' not in df.columns:
        logging.warning("Missing required columns (utc_ts or home_team) for weather attachment")
        return df
    
    # Load cache
    cache = load_weather_cache()
    cache_updated = False
    logging.info(f"Loaded weather cache with {len(cache)} entries")
    
    # Per-run cache snapshot: ensure cache is saved even if function crashes
    try:
        # Initialize weather columns
        weather_columns = CORE_WEATHER_FIELDS.copy()
        logging.info(f"Weather columns to be added: {weather_columns}")
        
        # Only initialize columns that we actually expect to get from the API
        for col in weather_columns:
            if col not in df.columns:
                df[col] = None
                logging.debug(f"Initialized missing column: {col}")
        
        # Pre-bucket timestamps (vectorized operation)
        logging.info("Pre-bucketing timestamps...")
        # Use 60-minute buckets initially (will be adjusted by auto-detection)
        df['bucket'] = df['utc_ts'].dt.floor('60min')
        
        # Skip NaT buckets
        before_bucket_filter = len(df)
        df = df.loc[df['bucket'].notna()].copy()
        after_bucket_filter = len(df)
        logging.info(f"Bucket filtering: {before_bucket_filter} -> {after_bucket_filter} rows")
        
        # Vectorized team_timestamps building (eliminates iterrows)
        team_col = 'home_team' if 'home_team' in df.columns else ('posteam' if 'posteam' in df.columns else None)
        if team_col is None:
            logging.warning("No team column found for weather attachment (home_team/posteam)")
            return df
        team_timestamps = (df[[team_col, 'bucket']]
                           .dropna()
                           .drop_duplicates()
                           .groupby(team_col)['bucket']
                           .apply(list)
                           .to_dict())

        # Filter out teams with empty bucket lists and log diagnostics
        empty_before = [t for t, b in team_timestamps.items() if not b]
        if empty_before:
            logging.warning(f"Teams with empty bucket lists (will skip): {empty_before[:10]}{'...' if len(empty_before)>10 else ''}")
        team_timestamps = {t: b for t, b in team_timestamps.items() if b}

        logging.info("Team timestamps breakdown (non-empty only):")
        shown = 0
        for team, buckets in team_timestamps.items():
            try:
                logging.info(f"  {team}: {len(buckets)} buckets, range: {min(buckets)} to {max(buckets)}")
            except Exception:
                logging.info(f"  {team}: {len(buckets)} buckets, range: (unavailable)")
            shown += 1
            if shown >= 20:
                break

        # If no buckets for any team, fallback to per-team min/max utc_ts
        if not team_timestamps:
            logging.warning("No teams with bucket lists; falling back to per-team min/max utc_ts to form ranges")
            if 'utc_ts' not in df.columns:
                logging.warning("utc_ts missing; cannot attach weather")
                return df
            gb = (df[[team_col, 'utc_ts']]
                  .dropna()
                  .groupby(team_col)['utc_ts']
                  .agg(['min','max']))
            team_timestamps = {}
            for team, row in gb.iterrows():
                start = pd.to_datetime(row['min']).floor('60min')
                end = pd.to_datetime(row['max']).floor('60min')
                if pd.isna(start) or pd.isna(end):
                    continue
                # Use just the endpoints; fetcher will expand the window
                team_timestamps[str(team)] = [start, end]

        # Fetch weather data for each team (with concurrent execution)
        if not team_timestamps:
            logging.info("No teams require weather fetch; skipping API calls.")
        else:
            logging.info(f"Fetching weather data for {len(team_timestamps)} teams...")
            
            # Dynamic thread-pool sizing to avoid overwhelming the API
            actual_workers = max(1, min(MAX_FETCH_WORKERS, len(team_timestamps)))
            logging.info(f"Using {actual_workers} concurrent workers")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
                future_to_team = {
                    executor.submit(fetch_weather_for_team_batch, team, buckets, cache): team
                    for team, buckets in team_timestamps.items()
                }
                
                # Thread-safe cache merging: collect deltas and merge at the end
                cache_deltas = {}
                for future in concurrent.futures.as_completed(future_to_team):
                    team = future_to_team[future]
                    try:
                        delta = future.result()
                        if delta:
                            cache_deltas.update(delta)
                            cache_updated = True
                            logging.info(f"Team {team}: Got {len(delta)} new cache entries")
                        else:
                            logging.warning(f"Team {team}: No new cache entries returned")
                    except Exception as e:
                        logging.error(f"Error fetching weather for {team}: {e}")
                
                # Merge all deltas into main cache
                if cache_deltas:
                    cache.update(cache_deltas)
                    logging.info(f"Updated cache with {len(cache_deltas)} new entries")
                else:
                    logging.warning("No cache deltas to update!")
        
        # Apply weather data using parallel merge_asof with indices
        logging.info("Applying weather data to dataframe...")
        
        # Create weather dataframe for merge_asof
        weather_rows = []
        logging.info(f"Processing {len(cache)} cache entries for weather rows...")
        
        for cache_key, weather_data in cache.items():
            # Skip non-weather cache keys (like '_plan')
            if cache_key.endswith('_plan'):
                continue
                
            team, timestamp_str = cache_key.split('_', 1)
            bucket_time = pd.to_datetime(timestamp_str, utc=True, errors='coerce')
            
            # Skip invalid timestamps
            if bucket_time is pd.NaT:
                logging.warning(f"Invalid timestamp for cache key: {cache_key}")
                continue
            
            row = {'home_team': team, 'utc_ts': bucket_time}
            row.update(weather_data)
            weather_rows.append(row)
        
        logging.info(f"Created {len(weather_rows)} weather rows from cache")
        
        if weather_rows:
            weather_df = pd.DataFrame(weather_rows)
            weather_df = weather_df.sort_values(['home_team', 'utc_ts'])
            # Ensure utc_ts is timezone-aware UTC
            if weather_df['utc_ts'].dt.tz is None or str(weather_df['utc_ts'].dt.tz) != 'UTC':
                weather_df['utc_ts'] = weather_df['utc_ts'].dt.tz_convert('UTC')
            logging.info(f"Weather DataFrame shape: {weather_df.shape}")
            logging.info(f"Weather DataFrame columns: {list(weather_df.columns)}")
            
            # Log weather data by team
            for team in weather_df['home_team'].unique():
                team_weather = weather_df[weather_df['home_team'] == team]
                logging.info(f"Team {team} weather buckets: {len(team_weather)} rows, first={team_weather['utc_ts'].min()}, last={team_weather['utc_ts'].max()}")
            
            # Parallel merge_asof for each team using indices
            merge_args = []
            for team in df[team_col].unique():
                if pd.isna(team):
                    continue
                team_indices = df.index[df[team_col] == team]
                team_weather = weather_df[weather_df['home_team'] == team].copy()
                merge_args.append((team, team_indices, team_weather, weather_columns, df))
                logging.info(f"Team {team}: {len(team_indices)} data rows, {len(team_weather)} weather rows")
            
            # Execute parallel merges with separate worker pool
            if merge_args:
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_MERGE_WORKERS) as executor:
                    merge_results = list(executor.map(merge_weather_for_team, merge_args))
            else:
                merge_results = []
            
            # Apply results back to original dataframe using indices
            for team_indices, merged_data in merge_results:
                if not merged_data.empty:
                    # Align on index to avoid positional mismatch after sorting
                    common_cols = [c for c in weather_columns if c in merged_data.columns]
                    df.loc[merged_data.index, common_cols] = merged_data[common_cols]
                    logging.info(f"Merged weather data for {len(merged_data)} rows")
                else:
                    logging.warning(f"No merged data for team indices: {len(team_indices)} rows")
        else:
            logging.error("No weather rows created from cache!")
        
        # Handle unit string conversions before numeric casting
        if 'windspeed' in df.columns:
            df['windspeed'] = df['windspeed'].replace({'Calm': 0}).infer_objects(copy=False)
        
        # Convert weather columns to proper numeric types
        for col in weather_columns:
            if col in df.columns and col not in ['conditions']:  # Keep text columns as object
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Only cast to float32 if we have any non-null values
                if df[col].notna().any():
                    df[col] = df[col].astype('float32', copy=False)
        
        # Handle roof/dome teams - set wind to 0 instead of null for defined values
        logging.info("Handling roof/dome teams...")
        roof_closed_mask = df['home_team'].map(ROOF_CLOSED_TEAMS).fillna(False).infer_objects(copy=False)
        if roof_closed_mask.any():
            # Set wind speed and gust to 0, wind direction to NaN
            if 'windspeed' in df.columns:
                df.loc[roof_closed_mask, 'windspeed'] = 0.0
            if 'windgust' in df.columns:
                df.loc[roof_closed_mask, 'windgust'] = 0.0
            if 'winddir' in df.columns:
                df.loc[roof_closed_mask, 'winddir'] = np.nan
            logging.info(f"Set wind data for {roof_closed_mask.sum()} roof/dome game rows")
        
        # Detect final bucket size from actual data (before dropping bucket column)
        if 'bucket' in df.columns:
            unique_minutes = sorted(df['bucket'].dt.minute.unique())
            final_bucket_minutes = 60 if unique_minutes == [0] else 5
            logging.info(f"Final bucket alignment: {unique_minutes} minutes ({final_bucket_minutes}-minute buckets)")
        
        # Clean up temporary column
        df = df.drop(columns='bucket')
        
        # Log coverage statistics with rounding
        weather_coverage = {}
        for col in weather_columns:
            if col in df.columns:
                coverage = round(df[col].notna().mean() * 100, 2)
                weather_coverage[col] = coverage
        
        logging.info(f"Weather data attached to {len(df)} rows")
        logging.info(f"Weather coverage: {weather_coverage}")
        
        # Log game data by team
        for team in df['home_team'].unique():
            team_data = df[df['home_team'] == team]
            logging.info(f"Team {team} game data: {len(team_data)} rows, first={team_data['utc_ts'].min()}, last={team_data['utc_ts'].max()}")
        
    finally:
        # Per-run cache snapshot: always save cache even if function crashes
        if cache_updated:
            save_weather_cache(cache)
            logging.info("Weather cache saved")
        else:
            logging.warning("No cache updates to save")
        if restore_home_team and '_weather_home_backup' in df.columns:
            df['home_team'] = df['_weather_home_backup']
            df = df.drop(columns=['_weather_home_backup'])
    
    logging.info("=== WEATHER INTEGRATION DEBUG END ===")
    return df

def test_weather_integration():
    """Test the optimized weather integration with auto-detection"""
    import pandas as pd
    
    print("Testing weather integration with auto-detection...")
    
    # Create a small test dataframe for a different team
    test_data = {
        'game_pk': [718220, 718220, 718220, 718220, 718220],
        'home_team': ['SEA', 'SEA', 'SEA', 'SEA', 'SEA'],  # Seattle instead of NYY
        'utc_ts': [
            '2025-06-15T18:00:00Z',
            '2025-06-15T19:00:00Z',
            '2025-06-15T20:00:00Z',
            '2025-06-15T21:00:00Z',
            '2025-06-15T22:00:00Z'
        ],
        'batter_side': ['R', 'L', 'R', 'L', 'S']
    }
    
    df = pd.DataFrame(test_data)
    df['utc_ts'] = pd.to_datetime(df['utc_ts'])
    
    print("Testing production-ready weather integration...")
    print(f"Input data:\n{df}")
    
    # Attach weather data
    df_with_weather = attach_visual_crossing_weather(df)
    
    print(f"\nOutput data:\n{df_with_weather}")
    
    # Check weather columns
    weather_cols = [col for col in df_with_weather.columns if col in [
        'temp', 'humidity', 'windspeed', 'winddir', 'conditions'
    ]]
    
    print(f"\nWeather columns populated:")
    for col in weather_cols:
        non_null = df_with_weather[col].notna().sum()
        print(f"  {col}: {non_null}/{len(df_with_weather)} rows")
    
    # Show bucket alignment for verification
    if 'bucket' in df_with_weather.columns:
        unique_minutes = sorted(df_with_weather['bucket'].dt.minute.unique())
        bucket_minutes = 60 if unique_minutes == [0] else 5
        print(f"\nBucket minute alignment: {unique_minutes}")
        print(f"Auto-detected bucket size: {bucket_minutes} minutes")
        print(f"Plan type: {'hour' if bucket_minutes == 60 else 'minute'}-level data")

def convert_to_imperial(record: Dict) -> Dict:
    """Convert metric weather values to imperial for ML model compatibility"""
    if UNIT_GROUP == "metric":
        # Convert temperature from °C to °F
        if 'temp' in record and record['temp'] is not None:
            record['temp'] = (record['temp'] * 9/5) + 32
        if 'feelslike' in record and record['feelslike'] is not None:
            record['feelslike'] = (record['feelslike'] * 9/5) + 32
            
        # Convert wind speed from km/h to mph
        if 'windspeed' in record and record['windspeed'] is not None:
            record['windspeed'] = record['windspeed'] * 0.621371
        if 'windgust' in record and record['windgust'] is not None:
            record['windgust'] = record['windgust'] * 0.621371
    
    return record

if __name__ == "__main__":
    test_weather_integration() 
