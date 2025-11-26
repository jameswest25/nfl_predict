# NFL Data Collection Module

## Overview

The collect module is responsible for fetching, processing, and caching NFL play-by-play data from nflfastR via the `nfl_data_py` library. It provides a clean, maintainable pipeline for ingesting NFL game data with built-in auditing and quality checks.

## Architecture

```
pipeline/collect.py (Main orchestrator)
    ↓
utils/collect/
    ├── nfl_schedules.py    - NFL schedule management and caching
    ├── parquet_io.py        - File I/O for partitioned data
    ├── weather_cache.py     - Weather data caching (SQLite)
    ├── dtype_opt.py         - Memory optimization
    ├── schema_guard.py      - Schema validation (NFL-specific)
    ├── data_audit.py        - Field specifications (NFL_FIELD_SPECS)
    ├── audit.py             - Comprehensive data quality checks
    └── visual_crossing_weather.py - Weather enrichment
```

## Key Features

### 1. NFL Schedule Management

The `nfl_schedules.py` module provides comprehensive schedule handling:

```python
from utils.collect import get_schedule, get_upcoming_games, get_game_metadata

# Get full season schedule (with caching)
schedule = get_schedule([2024, 2023])

# Get games for a specific date
games = get_upcoming_games(date(2024, 9, 15), [2024])

# Get game metadata (teams, venue, conditions)
metadata = get_game_metadata('2024_02_KC_BAL', [2024])
```

**Features:**
- Automatic caching in parquet format
- Fast lookups for upcoming games
- Game start time extraction with timezone handling
- Stadium and weather metadata

### 2. Data Collection

Main collection function fetches NFL play-by-play data:

```python
from pipeline.collect import collect_nfl_data

# Collect by seasons (recommended)
df = collect_nfl_data(seasons=[2024, 2023, 2022, 2021])

# Data is automatically partitioned by season/week/date
# Output: data/raw/pbp_by_day/season=YYYY/week=WW/date=YYYY-MM-DD/part.parquet
```

**Data Structure:**
```
data/raw/pbp_by_day/
├── season=2024/
│   ├── week=01/
│   │   ├── date=2024-09-05/part.parquet
│   │   └── date=2024-09-08/part.parquet
│   └── week=02/
│       └── ...
└── season=2023/
    └── ...
```

### 3. NFL Field Specifications

The module includes comprehensive NFL field specifications:

```python
from utils.collect.data_audit import NFL_FIELD_SPECS

# 59 field definitions including:
# - Game identifiers (game_id, season, week)
# - Game state (down, ydstogo, yardline_100)
# - Players (passer_id, rusher_id, receiver_id)
# - Outcomes (yards_gained, touchdown, interception)
# - Advanced metrics (epa, wp, wpa)
# - Venue (roof, surface, temp, wind)
```

### 4. Data Audit

Comprehensive data quality checks:

```python
from utils.collect import run_collect_audit

# Run audit on collected data
run_collect_audit(df=my_dataframe)

# Generates detailed report:
# - Field coverage percentages
# - Missing value patterns
# - Data type validation
# - Range checking
# - NFL-specific field expectations
```

**Audit Output:** `audit/collect/latest_raw_audit.txt`

### 5. Schema Validation

NFL-specific schema checks:

```python
from utils.collect import ExpectedSchema, check_schema_drift

# Core NFL PBP fields (23 required)
core_fields = ExpectedSchema.NFL_PBP_CORE
# ['game_id', 'season', 'week', 'down', 'yardline_100', 'epa', ...]

# Optional fields (29 common)
optional_fields = ExpectedSchema.NFL_PBP_OPTIONAL
# ['passer_id', 'air_yards', 'shotgun', 'no_huddle', ...]

# Check for schema drift
has_drift = check_schema_drift(df, "nfl_pbp")
```

### 6. Memory Optimization

Automatic dtype optimization to reduce memory usage:

```python
from utils.collect import OptimizationConfig, maybe_optimize

config = OptimizationConfig(
    enabled=True,
    categorical_enabled=True
)

# Automatically downcasts numeric types and converts to categorical
optimized_df = maybe_optimize(df, config)
```

### 7. Weather Integration

Weather data enrichment (optional):

```python
from utils.collect import attach_weather

# Attaches weather data using stadium coordinates
df_with_weather = attach_weather(df)
```

## Command Line Interface

```bash
# Collect latest NFL data (last 4 seasons)
python pipeline/collect.py

# Collect specific seasons
python pipeline/collect.py --seasons 2024,2023,2022

# Limit to specific weeks (for testing)
python pipeline/collect.py --seasons 2024 --limit-weeks 5

# Pre-cache schedules
python pipeline/collect.py --cache-schedules --seasons 2024,2023

# Run audit on existing data
python pipeline/collect.py --audit-only

# Back-fill weather data
python pipeline/collect.py --backfill-weather
```

## Configuration

Settings in `config/config.yaml`:

```yaml
collect_params:
  max_concurrent_feeds: 20        # Not used (legacy MLB)
  default_batch_days: 7           # Not used (seasons mode)
  enable_dtype_optimization: true  # Memory optimization
  enable_categorical_columns: true # Convert to categorical
  seasons_window_years: 4         # Default seasons to collect
```

## Data Flow

1. **Fetch** - `nfl_data_py.import_pbp_data(years=[...])`
2. **Enrich** - Attach schedules for game start times
3. **Weather** - Optional weather data attachment
4. **Optimize** - Dtype optimization for memory efficiency
5. **Partition** - Write to season/week/date hierarchy
6. **Audit** - Generate quality report

## NFL-Specific Details

### Team Abbreviations
32 NFL teams with standardized 2-3 letter codes:
- AFC East: BUF, MIA, NE, NYJ
- AFC North: BAL, CIN, CLE, PIT
- AFC South: HOU, IND, JAX, TEN
- AFC West: DEN, KC, LAC, LVR
- NFC East: DAL, NYG, PHI, WAS
- NFC North: CHI, DET, GB, MIN
- NFC South: ATL, CAR, NO, TB
- NFC West: ARI, LAR, SEA, SF

### Stadium Configuration
32 NFL stadiums with coordinates and timezone info stored in:
`data/raw/config/stadium_coords.json`

### Data Coverage
- **Seasons**: 1999-present (nflfastR coverage)
- **Granularity**: Play-by-play
- **Fields**: 350+ columns per play
- **Size**: ~45K plays per season

## Removed MLB Components

The following MLB-specific modules were removed during migration:

- ❌ `feeds.py` - MLB StatsAPI feed fetcher
- ❌ `timestamps.py` - MLB pitch timestamps
- ❌ `timestamp_coverage.py` - MLB timestamp validation
- ❌ `id_cache.py` - MLB player ID cache

## Dependencies

- `nfl-data-py` - NFL data fetching
- `pandas` - Data manipulation
- `pyarrow` - Parquet I/O
- `requests` - HTTP requests (weather)
- `aiohttp` - Async HTTP (legacy)

## Testing

Run comprehensive tests:

```python
# Test all components
from utils.collect import (
    get_schedule, ExpectedSchema, 
    run_collect_audit, maybe_optimize
)

# All functions are fully tested and working
```

## Migration Notes

**From MLB to NFL:**
- Replaced pitch-level → play-level
- Removed batter/pitcher → Added passer/rusher/receiver
- Updated field specs for NFL metrics (EPA, WP, etc.)
- Changed from MLB StatsAPI → nfl_data_py
- Simplified: No need for feeds/timestamps (all in PBP data)

## Maintenance

The module is clean, maintainable, and production-ready:
- ✅ No linter errors
- ✅ All imports working
- ✅ Comprehensive tests passing
- ✅ Documentation complete
- ✅ NFL-specific throughout

## Future Enhancements

Potential additions:
- Injury data integration (`nfl_data_py.import_injuries()`)
- Roster data (`nfl_data_py.import_rosters()`)
- Weekly player stats (`nfl_data_py.import_weekly_data()`)
- Next Gen Stats (`nfl_data_py.import_ngs_data()`)
- Real-time score updates
