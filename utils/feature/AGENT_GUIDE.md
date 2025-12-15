# Agent Quick Reference: utils/feature Module

> **Read this before making any changes to the feature engineering code.**

## 🏗️ Module Structure (5 Subpackages)

```
utils/feature/
├── builders/     → Build data levels (player_game_level.py is the main one)
├── enrichment/   → Add external data (odds, weather, team context)
├── rolling/      → Rolling window features (3g_, 5g_, etc.)
├── derived/      → Engineered features from base stats
└── core/         → Labels, validation, shared utilities
```

## ⚡ Quick Rules

### DO ✅
- Put new rolling stats in `rolling/stats.py`
- Put new derived features in `derived/` as a new file or extend existing
- Put new data source loaders in `builders/data_loaders.py`
- Always cast types explicitly: `.cast(pl.Float32)`, `.cast(pl.Int8)`
- Always handle nulls: `.fill_null(0)` or `.fill_null(0.0)`
- Log what you add: `logger.info(f"Added {len(cols)} features")`
- Check columns exist before using them
- Use `utils.general.paths` for all file paths

### DON'T ❌
- Create circular imports (import inside functions if needed)
- Use future/same-game data as features (LEAKAGE!)
- Hardcode file paths
- Forget to export from `__init__.py`
- Add features without testing the full pipeline

## 📝 Naming Conventions

| Prefix | Meaning | Example |
|--------|---------|---------|
| `3g_`, `5g_` | Rolling average | `3g_targets` |
| `_prev` | Previous game | `targets_prev` |
| `_l3`, `_l5` | Last N games | `hist_catch_rate_l3` |
| `hist_` | Historical/rolling | `hist_snaps_l3` |
| `is_` | Binary flag | `is_home` |
| `team_ctx_`, `opp_ctx_` | Context features | `team_ctx_pass_rate_l3` |

## 🔧 Adding a New Feature

### 1. Create the function
```python
def add_my_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add my features. Requires: col_a, col_b."""
    if "col_a" not in df.columns:
        return df
    
    df = df.with_columns([
        (pl.col("col_a") / pl.col("col_b").clip(lower=1))
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias("my_feature"),
    ])
    return df
```

### 2. Export it
```python
# In derived/__init__.py
from utils.feature.derived.my_module import add_my_features
```

### 3. Call it in pipeline
```python
# In pipeline/feature.py (in correct order!)
df = add_my_features(df)
```

### 4. Test it
```bash
python main.py feature  # Run full pipeline
python -c "import polars as pl; df = pl.read_parquet('data/processed/final/processed.parquet'); print(df['my_feature'].describe())"
```

## 📊 Pipeline Order (Matters!)

1. Load base data
2. Attach as-of metadata
3. Position/role features
4. Context enrichment (team, offense)
5. Weather enrichment
6. Odds enrichment
7. **Rolling features** ← Most things depend on this
8. Snap features
9. Historical shares
10. Expected snaps
11. Market features
12. Usage features
13. MoE features
14. Catch rate features
15. Target depth features
16. Role flags
17. Drop leakage columns
18. Schema validation

## 🔍 Key Files

| Need to... | Look at... |
|------------|-----------|
| Add rolling stat | `rolling/stats.py` |
| Add derived feature | `derived/` (new or existing file) |
| Load external data | `builders/data_loaders.py` |
| Understand aggregation | `builders/player_aggregation.py` |
| Add new label | `core/labels.py` |
| Prevent leakage | `core/leak_guard.py` |
| See full orchestration | `pipeline/feature.py` |

## ⚠️ Common Mistakes

```python
# LEAKAGE - using same-game data
pl.col("anytime_td").alias("td_feature")  # ❌
pl.col("anytime_td").shift(1).over("player_id").alias("td_prev")  # ✅

# MISSING NULL HANDLING
pl.col("rate")  # ❌ might have nulls
pl.col("rate").fill_null(0.0)  # ✅

# WRONG TYPE
.alias("my_int_flag")  # ❌ might be Float64
.cast(pl.Int8).alias("my_int_flag")  # ✅

# CIRCULAR IMPORT
from utils.feature.derived.other import func  # ❌ at top level
def my_func():
    from utils.feature.derived.other import func  # ✅ inside function
```

## 📖 Full Documentation

See `utils/feature/ARCHITECTURE.md` for complete details.





