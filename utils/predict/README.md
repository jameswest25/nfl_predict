# NFL Prediction Utilities

This package contains the prediction pipeline utilities for NFL player predictions.
The architecture ensures **strict feature parity** between training and inference.

## Module Structure

```
utils/predict/
├── __init__.py          # Package exports
├── features.py          # UNIFIED feature builder (core of parity)
├── scaffold.py          # Player-game scaffold builder
├── loaders.py           # Data loaders for context features
├── injuries.py          # Injury feature enrichment
├── inference.py         # Model loading and prediction
├── output.py            # Output formatting and composed features
└── README.md            # This file
```

## Key Design Principles

### 1. Unified Feature Builder

The `build_features()` function in `features.py` is the **single entry point** for
feature generation. Both training and inference use this function to ensure strict
parity. The only difference is controlled by the `is_inference` flag:

```python
from utils.predict.features import build_features

# For training
features = build_features(df, is_inference=False)

# For inference
features = build_features(df, is_inference=True)
```

### 2. Minimal Mode-Specific Branching

The `is_inference` flag controls only a few necessary differences:
- **Odds handling**: Training drops rows without odds; inference keeps all rows
- **Context loading**: Inference reads from pre-computed history caches
- **Drive history**: Inference excludes current games from history computation

### 3. Feature Order Consistency

All derived features are applied in a strict order (see `_add_all_derived_features()`):
1. Position and role features
2. Usage helper features
3. MoE position-specific features
4. Catch rate and target depth features
5. Snap features
6. Historical share features
7. Role share flags
8. Market features

## Usage

### For Predictions (predict.py)

```python
from utils.predict import (
    load_schedule,
    load_rosters,
    build_scaffold,
    build_features,
    attach_injury_features,
)
import polars as pl

# 1. Load schedule and roster
games = load_schedule(start_date, end_date)
roster = load_rosters(games["season"].unique())

# 2. Build scaffold
scaffold = build_scaffold(games, roster)

# 3. Build features using unified builder
pl_df = pl.from_pandas(scaffold)
pl_df = build_features(pl_df, is_inference=True)
pl_df = attach_injury_features(pl_df)
```

### For Training (feature.py)

The training pipeline can also use `build_features()` for feature enrichment,
though it currently has its own implementation for historical reasons.

## Parity Verification

Run the parity test to verify training and inference produce identical features:

```bash
python tests/verify_parity_strict.py
```

## Module Details

### features.py

Core feature building logic:
- `build_features()` - Main entry point
- Rolling window features
- Team/offense context
- Weather, odds, opponent splits
- Drive history
- All derived features

### scaffold.py

Builds the player-game scaffold:
- `load_schedule()` - Load NFL game schedule
- `load_rosters()` - Load weekly roster data
- `build_scaffold()` - Create player-game rows

### loaders.py

Data loaders for context features:
- `load_qb_profile_features()` - QB profile data
- `load_travel_calendar_features()` - Travel/rest features
- `load_injury_history_features()` - Injury priors
- `load_ps_baselines()` - Pre-snap participation baselines

### injuries.py

Injury feature enrichment:
- `ensure_injury_cache()` - Ensure fresh injury data
- `attach_injury_features()` - Add injury features to dataframe

### inference.py

Model loading and prediction:
- `load_artifacts()` - Load model artifacts
- `predict_moe()` - MoE (per-position) predictions
- `predict_global()` - Global model predictions
- `apply_*_cap()` - Position-specific caps and guards

### output.py

Output formatting:
- `inject_composed_features()` - Compute expected values
- `format_output()` - Format final prediction output
- `save_predictions()` - Save to CSV
