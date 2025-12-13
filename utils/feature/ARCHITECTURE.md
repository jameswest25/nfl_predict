# Feature Engineering Module Architecture

## Overview

The `utils/feature` module is responsible for all feature engineering in the NFL prediction pipeline. It transforms raw play-by-play data into a rich feature matrix used for training and inference.

**Entry Points:**
- `pipeline/feature.py` - Orchestrates the feature pipeline (calls into utils/feature)
- `python main.py feature` - CLI command to run the full feature pipeline

**Output:**
- `data/processed/final/processed.parquet` - Final feature matrix (~30k rows, ~1300 columns)

---

## Directory Structure

```
utils/feature/
├── __init__.py              # Main exports
├── ARCHITECTURE.md          # This document
│
├── builders/                # Build intermediate data levels
│   ├── __init__.py
│   ├── play_level.py        # Play-level slices from cleaned data
│   ├── drive_level.py       # Drive-level team aggregates
│   ├── game_level.py        # Game-level team aggregates
│   ├── player_drive_level.py # Player-drive aggregates
│   ├── player_game_level.py  # Player-game aggregates (MAIN)
│   ├── opponent_splits.py    # Opponent defensive splits
│   ├── data_loaders.py       # Data loading functions (rosters, injuries, etc.)
│   ├── injury_enrichment.py  # Injury feature computation
│   └── player_aggregation.py # Player stat aggregation functions
│
├── enrichment/              # Add external/context data
│   ├── __init__.py
│   ├── odds.py              # NFL betting odds features
│   ├── weather_features.py  # Weather forecast enrichment
│   ├── team_context.py      # Team-level context features
│   ├── offense_context.py   # Offensive coordinator/QB context
│   ├── qb_profiles.py       # QB profile features
│   ├── travel_calendar.py   # Travel and rest features
│   ├── asof.py              # As-of time utilities
│   ├── asof_metadata.py     # As-of metadata management
│   └── asof_enrichment.py   # As-of metadata enrichment
│
├── rolling/                 # Rolling window features
│   ├── __init__.py
│   ├── rolling_window.py    # Main rolling window computation
│   ├── daily_totals.py      # Daily totals cache
│   ├── stats.py             # Rolling stat definitions
│   └── pace.py              # Pace-related metrics
│
├── derived/                 # Derived feature engineering
│   ├── __init__.py
│   ├── usage_features.py    # Usage model helper features
│   ├── position_features.py # Position group and role flags
│   ├── catch_rate_features.py # Historical catch rate features
│   ├── target_depth_features.py # Air yards and depth features
│   ├── snap_features.py     # Snap-related features
│   ├── historical_share_features.py # Historical share features
│   └── market_features.py   # Market/odds interaction features
│
└── core/                    # Core utilities
    ├── __init__.py
    ├── labels.py            # Label definitions (anytime_td, etc.)
    ├── targets.py           # Target column validation
    ├── leak_guard.py        # Leakage prevention utilities
    └── shared.py            # Shared helpers (TD rate history)
```

---

## Module Responsibilities

### `builders/` - Data Level Construction

**Purpose:** Transform raw data into hierarchical aggregation levels.

| File | Responsibility |
|------|----------------|
| `play_level.py` | Slice raw plays by date range |
| `drive_level.py` | Aggregate plays to drive-team level |
| `game_level.py` | Aggregate plays to game-team level |
| `player_drive_level.py` | Aggregate plays to player-drive level |
| `player_game_level.py` | **Main aggregation** - creates one row per (player, game) |
| `data_loaders.py` | Load external data (rosters, injuries, snap counts, depth charts) |
| `injury_enrichment.py` | Compute injury history rates and apply ML model |
| `player_aggregation.py` | Core aggregation functions (passers, rushers, receivers) |

**Key Pattern:** Each builder creates a specific grain of data. The `player_game_level.py` file is the central hub that calls other modules.

### `enrichment/` - External Data Enrichment

**Purpose:** Add context from external sources (odds, weather, schedules).

| File | Responsibility |
|------|----------------|
| `odds.py` | Betting odds features (moneyline, spread, totals, player props) |
| `weather_features.py` | Weather forecast at game time |
| `team_context.py` | Team-level offensive tendencies |
| `offense_context.py` | Offensive coordinator and QB-specific context |
| `qb_profiles.py` | QB historical performance profiles |
| `travel_calendar.py` | Rest days, travel distance, timezone changes |
| `asof*.py` | Point-in-time correctness (prevent data leakage) |

**Key Pattern:** All enrichment functions take a DataFrame and return an enriched DataFrame. They should be idempotent.

### `rolling/` - Rolling Window Features

**Purpose:** Compute historical rolling statistics (3-game, 5-game, season-to-date).

| File | Responsibility |
|------|----------------|
| `rolling_window.py` | Main rolling feature computation |
| `daily_totals.py` | Cache of daily player totals for efficiency |
| `stats.py` | Definitions of which stats to roll and how |
| `pace.py` | Team pace and tempo metrics |

**Key Pattern:** Rolling features use a cached daily totals layer to avoid recomputing from play-level data.

### `derived/` - Derived Features

**Purpose:** Compute engineered features from base aggregates.

| File | Responsibility |
|------|----------------|
| `usage_features.py` | Usage model helpers (touch counts, share estimates) |
| `position_features.py` | Position group routing, specialist flags |
| `catch_rate_features.py` | Historical catch rate efficiency |
| `target_depth_features.py` | Air yards, depth of target archetypes |
| `snap_features.py` | Snap expectations, role stability |
| `historical_share_features.py` | Historical target/carry share |
| `market_features.py` | Market-derived features (implied totals) |

**Key Pattern:** Derived features depend on rolling features being computed first. Order matters in `pipeline/feature.py`.

### `core/` - Core Utilities

**Purpose:** Shared constants, labels, and validation.

| File | Responsibility |
|------|----------------|
| `labels.py` | Label column definitions (anytime_td variants) |
| `targets.py` | Validate required target columns exist |
| `leak_guard.py` | Detect and prevent data leakage |
| `shared.py` | Shared helpers (TD rate history, drive aggregates) |

---

## Naming Conventions

### Column Naming

| Prefix | Meaning | Example |
|--------|---------|---------|
| `3g_`, `5g_` | 3-game or 5-game rolling average | `3g_targets`, `5g_rushing_yards` |
| `std_` | Rolling standard deviation | `std_targets_l5` |
| `_prev` | Previous game value | `offense_pct_prev` |
| `_l3`, `_l5` | Last 3 or 5 games | `hist_catch_rate_l3` |
| `hist_` | Historical (rolling) version | `hist_snaps_l3` |
| `team_ctx_`, `opp_ctx_` | Team/opponent context | `team_ctx_pass_rate_l3` |
| `qb_profile_` | QB profile features | `qb_profile_completion_rate_l5` |
| `injury_` | Injury-related features | `injury_inactive_probability` |
| `market_` | Market/betting features | `market_anytime_td_prob` |
| `weather_` | Weather features | `weather_temperature_f` |
| `travel_` | Travel/rest features | `travel_rest_days` |
| `is_` | Binary flag | `is_home`, `is_fullback` |
| `role_` | Role classification flag | `role_primary_red_zone_target` |
| `ps_` | Pre-snap features | `ps_route_participation_pct` |

### Function Naming

| Pattern | Usage | Example |
|---------|-------|---------|
| `add_*_features()` | Add features to DataFrame | `add_rolling_snap_features()` |
| `compute_*()` | Compute and return new data | `compute_td_labels()` |
| `build_*()` | Build/create data structure | `build_player_game_level()` |
| `load_*()` | Load data from disk/API | `load_rosters_for_years()` |
| `attach_*()` | Join external data | `attach_asof_metadata()` |
| `_private_func()` | Internal helper (not exported) | `_aggregate_passers()` |

---

## Adding New Features

### Step 1: Determine the Right Location

| Feature Type | Location |
|--------------|----------|
| Rolling/historical stat | `rolling/` or update `rolling/stats.py` |
| External data enrichment | `enrichment/` |
| Derived from existing features | `derived/` |
| New data source loading | `builders/data_loaders.py` |
| New label/target | `core/labels.py` |

### Step 2: Implement the Feature

**Template for a derived feature function:**

```python
def add_my_new_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add my new feature group.
    
    Features added:
    - my_feature_1: Description
    - my_feature_2: Description
    
    Parameters
    ----------
    df : pl.DataFrame
        Feature matrix with required columns: col_a, col_b
        
    Returns
    -------
    pl.DataFrame
        DataFrame with new features added
    """
    logger = logging.getLogger(__name__)
    
    # Check required columns exist
    required = {"col_a", "col_b"}
    if not required.issubset(set(df.columns)):
        logger.warning("Missing required columns for my_new_features")
        return df
    
    # Compute features
    df = df.with_columns([
        (pl.col("col_a") / pl.col("col_b").clip(lower=1))
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias("my_feature_1"),
    ])
    
    # Log what was added
    logger.info(f"    Added my_feature_1 (mean={df['my_feature_1'].mean():.3f})")
    
    return df
```

### Step 3: Export the Function

Add to the appropriate `__init__.py`:

```python
# In utils/feature/derived/__init__.py
from utils.feature.derived.my_module import add_my_new_features

__all__ = [
    # ... existing exports
    "add_my_new_features",
]
```

### Step 4: Integrate into Pipeline

Add to `pipeline/feature.py` in the correct order:

```python
# After rolling features, before final validation
df = add_my_new_features(df)
```

### Step 5: Update Config (if needed)

If the feature should be used in training, add to `config/training.yaml`:

```yaml
problems:
  anytime_td:
    features:
      include:
        - my_feature_1
```

---

## Common Patterns

### Pattern 1: Safe Column Access

Always check if columns exist before using them:

```python
if "target" in df.columns:
    df = df.with_columns(pl.col("target").fill_null(0))
else:
    df = df.with_columns(pl.lit(0).alias("target"))
```

### Pattern 2: Type Safety

Always cast to explicit types:

```python
df = df.with_columns([
    pl.col("my_float").cast(pl.Float32),
    pl.col("my_flag").cast(pl.Int8),
    pl.col("my_string").cast(pl.Utf8),
])
```

### Pattern 3: Null Handling

Fill nulls appropriately based on semantics:

```python
# For counts/sums: fill with 0
pl.col("targets").fill_null(0)

# For rates/ratios: fill with 0 or a sensible default
pl.col("catch_rate").fill_null(0.5)  # League average

# For flags: fill with 0 (false)
pl.col("is_starter").fill_null(0)

# For historical features with no history: use explicit null or 0
pl.col("hist_yards_l3").fill_null(0)
```

### Pattern 4: Avoid Data Leakage

Never use future information:

```python
# WRONG: Uses current game data
df.with_columns(pl.col("targets").alias("feature"))

# CORRECT: Uses shifted (previous) data
df.with_columns(
    pl.col("targets").shift(1).over("player_id").alias("targets_prev")
)
```

### Pattern 5: Logging

Always log what was added:

```python
logger.info(f"    Added {len(new_cols)} my_feature columns")
logger.info(f"    my_feature coverage: {non_null_count}/{total_count} rows")
```

---

## Feature Ordering in Pipeline

The order in `pipeline/feature.py` matters. Follow this sequence:

1. **Load base data** (`build_daily_cache_range`)
2. **Attach as-of metadata** (`attach_asof_metadata`)
3. **Position/role features** (`add_position_group`, `add_specialist_role_flags`)
4. **Context enrichment** (`add_team_context_features`, `add_offense_context_features`)
5. **Weather enrichment** (`add_weather_forecast_features`)
6. **Odds enrichment** (`add_nfl_odds_features_to_df`)
7. **Rolling features** (`add_rolling_features`) - **Most features depend on this**
8. **Snap features** (`add_rolling_snap_features`, `add_role_stability_features`)
9. **Historical shares** (`add_historical_share_features`)
10. **Expected snaps** (`add_expected_snap_features`) - Depends on historical shares
11. **Market features** (`add_market_features`)
12. **Usage features** (`add_usage_helper_features`)
13. **MoE features** (`add_moe_position_features`)
14. **Catch rate features** (`add_catch_rate_features`)
15. **Target depth features** (`add_target_depth_features`)
16. **Role flags** (`add_role_share_flags`)
17. **Drop leakage columns** (`drop_leakage_columns`)
18. **Schema validation** (`enforce_leak_guard`)

---

## Testing Requirements

### Before Committing

1. **Syntax check:**
   ```bash
   python -m py_compile utils/feature/my_module.py
   ```

2. **Import test:**
   ```bash
   python -c "from utils.feature.derived.my_module import add_my_features"
   ```

3. **Full pipeline test:**
   ```bash
   python main.py feature
   ```

4. **Feature quality check:**
   ```python
   import polars as pl
   df = pl.read_parquet("data/processed/final/processed.parquet")
   
   # Check your feature
   print(df["my_feature"].null_count() / df.height)  # Should be <50%
   print(df["my_feature"].n_unique())  # Should be >1 (not constant)
   ```

### Unit Tests

Add tests in `tests/` for complex logic:

```python
def test_my_feature_computation():
    # Create minimal test DataFrame
    df = pl.DataFrame({
        "player_id": ["A", "A", "B"],
        "col_a": [10, 20, 30],
        "col_b": [5, 10, 15],
    })
    
    result = add_my_new_features(df)
    
    assert "my_feature_1" in result.columns
    assert result["my_feature_1"].null_count() == 0
```

---

## Pitfalls to Avoid

### ❌ Circular Imports

```python
# WRONG: Creates circular import
# In builders/player_game_level.py
from utils.feature.derived.usage_features import get_usage_config

# CORRECT: Import at function level if needed
def my_function():
    from utils.feature.derived.usage_features import get_usage_config
```

### ❌ Modifying Global State

```python
# WRONG: Modifies global cache
_CACHE = {}
def load_data():
    _CACHE["data"] = expensive_computation()

# CORRECT: Use function-scoped or explicit cache paths
@lru_cache(maxsize=128)
def load_data():
    return expensive_computation()
```

### ❌ Hardcoded Paths

```python
# WRONG
path = Path("/Users/james/data/rosters.parquet")

# CORRECT: Use utils.general.paths
from utils.general.paths import ROSTER_CACHE_DIR
path = ROSTER_CACHE_DIR / "rosters.parquet"
```

### ❌ Missing Error Handling

```python
# WRONG: Crashes on missing data
df = pl.read_parquet(path)

# CORRECT: Handle gracefully
try:
    df = pl.read_parquet(path)
except Exception as e:
    logger.warning(f"Failed to load {path}: {e}")
    df = pl.DataFrame()
```

### ❌ Leaking Future Information

```python
# WRONG: Uses same-game touchdown data as feature
df.with_columns(pl.col("anytime_td").alias("td_feature"))

# CORRECT: Use historical data only
df.with_columns(
    pl.col("anytime_td").shift(1).over("player_id").alias("td_prev")
)
```

---

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `pipeline/feature.py` | ~1000 | Main orchestration |
| `builders/player_game_level.py` | ~3000 | Core player-game aggregation |
| `builders/data_loaders.py` | ~720 | External data loading |
| `enrichment/odds.py` | ~1500 | Odds feature engineering |
| `rolling/rolling_window.py` | ~560 | Rolling feature computation |
| `core/shared.py` | ~780 | TD rate history features |

---

## Contact / Maintenance

This module is critical infrastructure. When making changes:

1. **Test locally** with `python main.py feature`
2. **Verify feature quality** (null rates, variance)
3. **Check training** with `python main.py train`
4. **Document** any new feature groups in this file

Last updated: December 2024



