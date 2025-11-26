#!/usr/bin/env python3
"""
NFL Data Collection Pipeline

A comprehensive, modular pipeline for collecting, processing, and caching NFL play-by-play data.
Provides clean interfaces for data ingestion with built-in caching, error handling, and
performance optimizations.

Key Modules:
    - nfl_schedules: NFL schedule management and caching
    - weather_cache: Efficient SQLite-based weather data caching
    - parquet_io: File I/O operations for partitioned data
    - dtype_opt: Memory usage optimization for large datasets
    - schema_guard: Data consistency and drift detection
    - visual_crossing_weather: Weather data enrichment
    
Architecture:
    - Modular design with clear separation of concerns
    - Centralized configuration through paths.py
    - Comprehensive error handling and logging
    - Built-in caching for API responses and processed data
    - Memory-efficient data processing with optional optimizations
"""

# Export all module functions for easy access
from utils.general.paths import PROJ_ROOT

# NFL-specific modules
from .nfl_schedules import (
    get_schedule, get_upcoming_games, get_game_start_time,
    cache_schedules, get_game_metadata
)

# Core data handling modules
from .parquet_io import write_data, load_data, get_latest_date, clean_overlaps
from .weather_cache import (
    init_db as init_weather_db, 
    load_cache as load_weather_cache, 
    save_cache as save_weather_cache, 
    migrate_json_cache, 
    attach_weather
)
from .dtype_opt import OptimizationConfig, optimize_dtypes, get_memory_usage, maybe_optimize
from .schema_guard import (
    ExpectedSchema, check_schema_drift, assert_no_drift, 
    get_column_hash, validate_expected_columns, get_schema_summary
)

# Visual Crossing Weather (sport-agnostic)
from .visual_crossing_weather import attach_visual_crossing_weather
from .weather_forecasts import collect_weather_forecasts

# Audit modules
from .data_audit import audit_dataset, audit_field, generate_audit_report
from .audit import run_collect_audit

__all__ = [
    # Paths
    'PROJ_ROOT',
    
    # NFL Schedules
    'get_schedule', 'get_upcoming_games', 'get_game_start_time',
    'cache_schedules', 'get_game_metadata',
    
    # Parquet I/O
    'write_data', 'load_data', 'get_latest_date', 'clean_overlaps',
    
    # Weather Cache
    'init_weather_db', 'load_weather_cache', 'save_weather_cache', 
    'migrate_json_cache', 'attach_weather',
    
    # Dtype Optimization
    'OptimizationConfig', 'optimize_dtypes', 'get_memory_usage', 'maybe_optimize',
    
    # Schema Guard
    'ExpectedSchema', 'check_schema_drift', 'assert_no_drift', 'get_column_hash',
    'validate_expected_columns', 'get_schema_summary',
    
    # Data Audit
    'audit_dataset', 'audit_field', 'generate_audit_report',
    'run_collect_audit',

    # Visual Crossing Weather
    'attach_visual_crossing_weather',
    'collect_weather_forecasts',
] 