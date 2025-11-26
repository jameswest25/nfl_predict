#!/usr/bin/env python3
"""Central project path definitions (moved from utils.collect.paths)."""

from __future__ import annotations

from pathlib import Path
from typing import Final
import os

# ------------------------------------------------------------------------------------
# Core directories (project root, raw data, config, etc.)
# ------------------------------------------------------------------------------------

PROJ_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# Raw data tree
DATA_RAW: Final[Path] = PROJ_ROOT / "data" / "raw"
DATA_CONFIG: Final[Path] = DATA_RAW / "config"
DATA_PBP: Final[Path] = DATA_RAW / "pbp"  # NFL play-by-play data

for d in (DATA_RAW, DATA_CONFIG, DATA_PBP):
    d.mkdir(parents=True, exist_ok=True)

# Odds snapshots collected from external providers
ODDS_SNAPSHOT_DIR: Final[Path] = DATA_RAW / "odds_snapshots"
ODDS_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Player-level market snapshots (raw bookmaker props)
PLAYER_ODDS_RAW_DIR: Final[Path] = DATA_RAW / "player_odds"
PLAYER_ODDS_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Structured metadata artifacts
ASOF_METADATA_PATH: Final[Path] = PROJ_ROOT / "data" / "processed" / "asof_metadata.parquet"

# Processed player market features
PLAYER_MARKET_PROCESSED_DIR: Final[Path] = PROJ_ROOT / "data" / "processed" / "player_market"
PLAYER_MARKET_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Quarterback style profile artifacts
QB_PROFILE_DIR: Final[Path] = PROJ_ROOT / "data" / "processed" / "qb_profiles"
QB_PROFILE_TEAM_DIR: Final[Path] = PROJ_ROOT / "data" / "processed" / "qb_profiles_team"
for d in (QB_PROFILE_DIR, QB_PROFILE_TEAM_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Travel and rest context features
TRAVEL_CALENDAR_DIR: Final[Path] = PROJ_ROOT / "data" / "processed" / "travel_calendar"
TRAVEL_CALENDAR_DIR.mkdir(parents=True, exist_ok=True)

# Time-aware roster snapshots (collected from ESPN)
ROSTER_SNAPSHOT_DIR: Final[Path] = PROJ_ROOT / "data" / "processed" / "roster_snapshots"
ROSTER_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Defensive opponent splits
OPPONENT_SPLIT_DIR: Final[Path] = PROJ_ROOT / "data" / "processed" / "opponent_splits"
OPPONENT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Weather forecast storage (snapshots at fixed lead times)
WEATHER_FORECAST_DIR: Final[Path] = DATA_RAW / "weather_snapshots"
WEATHER_FORECAST_DIR.mkdir(parents=True, exist_ok=True)

# Optional observed weather archive (for evaluation/backfill)
WEATHER_OBSERVED_DIR: Final[Path] = DATA_RAW / "weather_observed"
WEATHER_OBSERVED_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------------
# Consolidated cache & audit structure (root-level)
# ------------------------------------------------------------------------------------

CACHE_ROOT: Final[Path] = PROJ_ROOT / "cache"
AUDIT_ROOT: Final[Path] = PROJ_ROOT / "audit"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

# Module-scoped sub-folders
COLLECT_CACHE_DIR = CACHE_ROOT / "collect"
CLEAN_CACHE_DIR   = CACHE_ROOT / "clean"
FEATURE_CACHE_DIR = CACHE_ROOT / "feature"
TRAIN_CACHE_DIR   = CACHE_ROOT / "train"
PREDICT_CACHE_DIR = CACHE_ROOT / "predict"
for d in (COLLECT_CACHE_DIR, CLEAN_CACHE_DIR, FEATURE_CACHE_DIR, TRAIN_CACHE_DIR, PREDICT_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

COLLECT_AUDIT_DIR = AUDIT_ROOT / "collect"
CLEAN_AUDIT_DIR   = AUDIT_ROOT / "clean"
FEATURE_AUDIT_DIR = AUDIT_ROOT / "feature"
TRAIN_AUDIT_DIR   = AUDIT_ROOT / "train"
PREDICT_AUDIT_DIR = AUDIT_ROOT / "predict"
for d in (COLLECT_AUDIT_DIR, CLEAN_AUDIT_DIR, FEATURE_AUDIT_DIR, TRAIN_AUDIT_DIR, PREDICT_AUDIT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------------
# Collect-step cache files (legacy names kept for backward compatibility)
# ------------------------------------------------------------------------------------
ID_CACHE_FILE         = COLLECT_CACHE_DIR / "id_cache.parquet"
TIMESTAMPS_CACHE_FILE = COLLECT_CACHE_DIR / "timestamps.parquet"
WEATHER_CACHE_FILE    = COLLECT_CACHE_DIR / "weather.db"
SCHEMA_HASH_FILE      = COLLECT_CACHE_DIR / "schema_hash.json"
FEED_CACHE_DIR        = COLLECT_CACHE_DIR / "feeds"
FEED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Clean-step cache
HAND_CACHE_JSON = CLEAN_CACHE_DIR / "handedness.json" 

# ------------------------------------------------------------------------------------
# Configuration files (ballpark metadata) – keep legacy aliases
# ------------------------------------------------------------------------------------

STADIUM_COORDS_JSON: Final[Path] = DATA_CONFIG / "stadium_coords.json"
PARK_ORIENTATION_JSON: Final[Path] = DATA_CONFIG / "park_orientation.json"

# Legacy aliases used in pipeline code
STADIUM_COORDS_FILE: Final[Path] = STADIUM_COORDS_JSON
PARK_ORIENTATION_FILE: Final[Path] = PARK_ORIENTATION_JSON

# ------------------------------------------------------------------------------------
# Back-compat: expose DATA_CACHE pointing at collect cache folder
# ------------------------------------------------------------------------------------
DATA_CACHE: Final[Path] = COLLECT_CACHE_DIR 

# ------------------------------------------------------------------------------------
# Raw data file paths (NFL play-by-play stored in daily partitions)
# ------------------------------------------------------------------------------------

# NFL PBP data is stored in partitioned structure:
# data/raw/pbp_by_day/season=YYYY/week=WW/date=YYYY-MM-DD/part.parquet
NFL_PBP_RAW_DIR: Final[Path] = DATA_RAW / "pbp_by_day"
NFL_PBP_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Monolithic raw file (used only by legacy backfill_weather function)
PBP_RAW_PARQUET: Final[Path] = DATA_RAW / "pbp_raw.parquet"

# ------------------------------------------------------------------------------------
# Processed data directories (feature pipeline outputs)
# ------------------------------------------------------------------------------------

DATA_PROCESSED: Final[Path] = PROJ_ROOT / "data" / "processed"
DATA_CLEANED: Final[Path] = PROJ_ROOT / "data" / "cleaned"

# Intermediate aggregation levels
PLAY_BY_WEEK_DIR: Final[Path] = DATA_PROCESSED / "play_by_week"
DRIVE_BY_WEEK_DIR: Final[Path] = DATA_PROCESSED / "drive_by_week"
GAME_BY_WEEK_DIR: Final[Path] = DATA_PROCESSED / "game_by_week"
PLAYER_DRIVE_BY_WEEK_DIR: Final[Path] = DATA_PROCESSED / "player_drive_by_week"
PLAYER_GAME_BY_WEEK_DIR: Final[Path] = DATA_PROCESSED / "player_game_by_week"

# Final feature matrix output
FINAL_FEATURES_DIR: Final[Path] = DATA_PROCESSED / "final"
FINAL_FEATURES_PARQUET: Final[Path] = FINAL_FEATURES_DIR / "processed.parquet"

# Context history files
TEAM_CONTEXT_HISTORY_PATH: Final[Path] = DATA_PROCESSED / "team_context_history.parquet"
OFFENSE_CONTEXT_HISTORY_PATH: Final[Path] = DATA_PROCESSED / "offense_context_history.parquet"
PLAYER_DRIVE_HISTORY_PATH: Final[Path] = DATA_PROCESSED / "player_drive_context_history.parquet"

# Create directories (excluding individual files)
for d in (
    DATA_PROCESSED, DATA_CLEANED, PLAY_BY_WEEK_DIR, DRIVE_BY_WEEK_DIR,
    GAME_BY_WEEK_DIR, PLAYER_DRIVE_BY_WEEK_DIR, PLAYER_GAME_BY_WEEK_DIR,
    FINAL_FEATURES_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------------
# Feature-step cache directories
# ------------------------------------------------------------------------------------

ROSTER_CACHE_DIR: Final[Path] = FEATURE_CACHE_DIR / "rosters"
INJURY_CACHE_DIR: Final[Path] = FEATURE_CACHE_DIR / "injuries"
INJURY_TRANSACTION_CACHE_DIR: Final[Path] = FEATURE_CACHE_DIR / "injury_transactions"
SNAP_CACHE_DIR: Final[Path] = FEATURE_CACHE_DIR / "snap_counts"
DAILY_TOTALS_CACHE_DIR: Final[Path] = FEATURE_CACHE_DIR / "daily_totals"

for d in (ROSTER_CACHE_DIR, INJURY_CACHE_DIR, INJURY_TRANSACTION_CACHE_DIR, SNAP_CACHE_DIR, DAILY_TOTALS_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Schedule cache
SCHEDULE_CACHE_PATH: Final[Path] = COLLECT_CACHE_DIR / "nfl_schedules.parquet"
