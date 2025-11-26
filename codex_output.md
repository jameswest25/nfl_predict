jameswest@Macbook nfl_predict %    cd /Users/jameswest/Desktop/nfl_predict
   ./venv/bin/python codex_meta_loop.py
[codex_meta_loop] Project directory: /Users/jameswest/Desktop/nfl_predict
[codex_meta_loop] Using codex CLI command: codex
[codex_meta_loop] Chat model: gpt-5.1
[codex_meta_loop] Max implementation iterations per plan: 10


==================== NEW META-CYCLE ====================


[run_codex_chat] Starting (exec - read-only) call...


================ PROMPT SENT TO CODEX ================


Context about this project:

- This is a project that takes NFL play-by-play data and uses it for modeling so that we can
  predict which players are likely to get an anytime touchdown (rushing or receiving, not passing)
  in an upcoming game.

- The goal is to build features and models that are as close as possible to the underlying
  football reality that produces touchdowns: play-calling tendencies, player usage, game state,
  defensive matchups, injuries, roles, red-zone behavior, etc.

- Training featurization must conceptually and logically match prediction/inference featurization.
  Anything that can only be known in hindsight at inference time (future data, downstream labels,
  or derived artifacts that use future information) is a form of leakage and must be eliminated.

- Over time, previous model runs and refactors may have left behind:
    * partially-implemented ideas,
    * experimental code paths,
    * hallucinated features,
    * or confusing / inconsistent logic.
  DO NOT assume that all existing code, features, configs, or comments are intentional or correct
  just because they exist. Treat any piece of code or configuration that does not clearly make
  sense in the context of the project as a candidate for cleanup, simplification, or removal.

Your task in this step:

Please analyze the current state of this project (code, data flow, feature engineering, and modeling)
and let me know:

1. Where things conceptually are not implemented correctly or are conceptually off, given the goal of
   predicting anytime TDs in a way that matches how football is actually played.
2. Where the modeling or data flow could be brought closer to "reality" as it actually plays out
  on the field. The goal of getting closer to reality is entirely so that the model is more
  accurate and metrics like AUC or other evaluation metrics improve.
3. Any incomplete implementations, half-finished ideas, or abandoned experimental paths.
4. Any wrong or misleading implementations (especially where names / comments and actual behavior diverge).
5. Any future data leaking into the modeling or feature pipeline (anything that uses knowledge from
  after the prediction cut-off point, including label-derived features).
6. Any underlying data sources or features that appear to not be working at all, or are effectively
  noise / dead weight.
7. Any areas where it looks like a previous run of a model or tool hallucinated structure, concepts,
  or features that don't actually exist in the real data or problem domain.

You should:

- Be concrete and specific in your findings.
- Call out anything that looks like hallucinated or legacy cruft that should probably be removed or
  reworked, instead of assuming it must be intentional.
- Focus on how each issue you find ultimately affects model realism and predictive performance.


======================================================


================ CODEX OUTPUT (streaming) ================

OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf67-266d-7c53-b835-f4d18d9aac20
--------
user

Context about this project:

- This is a project that takes NFL play-by-play data and uses it for modeling so that we can
  predict which players are likely to get an anytime touchdown (rushing or receiving, not passing)
  in an upcoming game.

- The goal is to build features and models that are as close as possible to the underlying
  football reality that produces touchdowns: play-calling tendencies, player usage, game state,
  defensive matchups, injuries, roles, red-zone behavior, etc.

- Training featurization must conceptually and logically match prediction/inference featurization.
  Anything that can only be known in hindsight at inference time (future data, downstream labels,
  or derived artifacts that use future information) is a form of leakage and must be eliminated.

- Over time, previous model runs and refactors may have left behind:
    * partially-implemented ideas,
    * experimental code paths,
    * hallucinated features,
    * or confusing / inconsistent logic.
  DO NOT assume that all existing code, features, configs, or comments are intentional or correct
  just because they exist. Treat any piece of code or configuration that does not clearly make
  sense in the context of the project as a candidate for cleanup, simplification, or removal.

Your task in this step:

Please analyze the current state of this project (code, data flow, feature engineering, and modeling)
and let me know:

1. Where things conceptually are not implemented correctly or are conceptually off, given the goal of
   predicting anytime TDs in a way that matches how football is actually played.
2. Where the modeling or data flow could be brought closer to "reality" as it actually plays out
  on the field. The goal of getting closer to reality is entirely so that the model is more
  accurate and metrics like AUC or other evaluation metrics improve.
3. Any incomplete implementations, half-finished ideas, or abandoned experimental paths.
4. Any wrong or misleading implementations (especially where names / comments and actual behavior diverge).
5. Any future data leaking into the modeling or feature pipeline (anything that uses knowledge from
  after the prediction cut-off point, including label-derived features).
6. Any underlying data sources or features that appear to not be working at all, or are effectively
  noise / dead weight.
7. Any areas where it looks like a previous run of a model or tool hallucinated structure, concepts,
  or features that don't actually exist in the real data or problem domain.

You should:

- Be concrete and specific in your findings.
- Call out anything that looks like hallucinated or legacy cruft that should probably be removed or
  reworked, instead of assuming it must be intentional.
- Focus on how each issue you find ultimately affects model realism and predictive performance.

mcp startup: no servers

thinking
**Listing project files for analysis**
exec
/bin/zsh -lc ls in /Users/jameswest/Desktop/nfl_predict succeeded in 765ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
README.md
__pycache__
audit
cache
codex_meta_loop.py
config
data
docs
logs
main.py
output
pipeline
pyproject.toml
requirements.txt
scripts
tests
tmp.parquet
tmp_features.parquet
utils

thinking
**Considering reading README**
exec
/bin/zsh -lc "sed -n '1,200p' README.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 762ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Anytime TD Prediction

## Key Scripts
- `pipeline/feature.py` – builds the player-game feature matrix, enforcing as-of leakage guards.
- `pipeline/train.py` – trains the multi-stage modeling graph (availability → usage → efficiency → TD).
- `pipeline/predict.py` – runs inference with the latest artifacts.
- `scripts/eval/odds_backtest.py` – compute odds-only baselines across horizons.
- `scripts/backtest/run_backtest.py` – evaluate existing prediction columns (AUC/Brier/log-loss) and write baseline summaries.
- `scripts/backtest/compare_baselines.py` – diff two baseline JSON files and highlight regressions.
- `scripts/analysis/horizon_eval.py` – consolidate metrics across decision horizons (input multiple prediction files labelled by horizon).
- `scripts/analysis/weather_travel_impact.py` – stratify TD hit rates by weather/travel features and emit plots (`docs/analysis/weather_travel/`) + JSON summaries.
- `scripts/monitoring/asof_coverage_report.py` – summarize snapshot coverage, write Markdown to `docs/monitoring/asof_coverage.md`, and append metrics history to `output/metrics/coverage/history.parquet`.

## Monitoring
- `docs/monitoring.md` documents leak guards, snapshot coverage expectations, roster snapshot regeneration steps, and instructions for running odds/backtests & analytics scripts.
- `docs/analysis/weather_travel.md` lists the latest weather/travel diagnostic artifacts.

## Quick Start
1. Build features:

thinking
**Preparing multi-step plan**
Plan update
Map out review steps to assess feature pipeline, modeling flow, leakage risks, and dead code.
  → Scan project structure and key config/docs to understand intended data flow and leakage guards
  • Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering key module info**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Feature pipeline orchestrator for NFL player-level predictions.

Stages
------
1. **build_play_level**         – derive cleaned play-level slices
2. **build_player_drive_level** – aggregate per drive
3. **build_player_game_level**  – aggregate per game and join context

This wrapper triggers each stage, optionally enforces a schema after each
write, and returns the final game-level DataFrame for downstream usage.
"""

import logging
import os
from datetime import date, timedelta
from typing import Sequence


thinking
**Planning deeper analysis**
exec
/bin/zsh -lc 'rg "TODO|FIXME|TBD"' in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
docs/rolling_window_implementation.md:### Immediate (TODO #8)

thinking
**Checking rolling window docs**
exec
/bin/zsh -lc "sed -n '1,200p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Rolling Window Feature System - Implementation Summary

## Overview

Successfully implemented a comprehensive rolling window feature system for NFL player predictions. The system computes time-series statistics for player performance across different contexts and time horizons.

## Architecture

### 1. Data Flow

```
Cleaned Play Data
    ↓
Player-Game Aggregations (player_game_level.py)
    ↓
Daily Totals Cache (nfl_daily_totals.py)
    ↓
Rolling Window Features (nfl_rolling_window.py)
    ↓
exec
/bin/zsh -lc "sed -n '200,400p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- **Incremental updates**: Only new dates need processing
- **Fast lookups**: Direct parquet reads by date
- **Memory efficient**: Streaming operations for large datasets
- **Reusable**: Same cache serves all rolling window computations

### Scalability
- **Per-player parallelization**: Each player's features computed independently
- **Lazy evaluation**: Uses Polars lazy API where possible
- **Chunked processing**: Pipeline processes data in configurable chunks

## Next Steps

### Immediate (TODO #8)
1. **Run feature pipeline** with new stats and rolling windows
   ```bash
   python -c "from pipeline.feature import build_feature_matrix; \
              from datetime import date; \
              build_feature_matrix(start_date=date(2024, 9, 5), \
                                  end_date=date(2024, 9, 9))"

thinking
**Verifying documentation accuracy**
exec
/bin/zsh -lc 'ls utils/feature' in /Users/jameswest/Desktop/nfl_predict succeeded in 746ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
__init__.py
__pycache__
asof.py
asof_metadata.py
concatenate_parquets.py
daily_totals.py
drive_level.py
game_level.py
io.py
keygen.py
odds.py
offense_context.py
opponent_splits.py
pace.py
play_level.py
player_drive_level.py
player_game_level.py
player_market_cli.py
qb_profiles.py

thinking
**Inspecting statistics components**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/stats.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 719ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL player statistics configuration for rolling window features.

Defines the standard set of stats tracked at all aggregation levels
(play, drive, game) for rolling window computations.

This module is the SINGLE SOURCE OF TRUTH for stat definitions used
throughout the pipeline. Import from here rather than defining locally.
"""

from typing import List

# =============================================================================
# Core Stats (used for drive and game level aggregation)
# =============================================================================

# Basic stats aggregated at player-drive level
PLAYER_DRIVE_STATS: List[str] = [
    # Yardage
    "receiving_yards",

thinking
**Verifying feature data existence**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 739ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL Rolling Window Features - Direct port from MLB rolling_window.py

Uses the exact same vectorized Polars pattern as the working MLB implementation.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache/feature/daily_totals")
NFL_SEASON_START_MMDD = (9, 1)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                    rolling_feature_names.append(f"{w}g_{k}_per_{self.level}{suffix}")

        if batch_min_date is None or batch_max_date is None:
            logger.warning("RollingWindow.compute received empty dataset; returning null rolling features.")
            result = ds
            for feat in rolling_feature_names:
                result = result.with_columns(pl.lit(None).cast(pl.Float32).alias(feat))
            return result.drop("__ts")
        
        # Determine lookback for CACHE scan
        # Load cache INCLUDING the batch dates - cumsum logic prevents leakage
        # Each row's rolling features = cumsum(up to row) - row_value = history BEFORE row
        if "lifetime" in windows:
            earliest = self._earliest_cache_date()
            lo_hist = earliest if earliest else batch_min_date - timedelta(days=365 * 10)
        elif "season" in windows:
            season_start = self._season_anchor(batch_min_date)
            lo_hist = season_start if season_start < batch_min_date else batch_min_date - timedelta(days=365)
        else:
exec
/bin/zsh -lc "sed -n '400,800p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                            denom_use = den_window_cache.get(w_int)
                            if denom_use is None:
                                denom_use = np.zeros(n, dtype=np.float64)
                            if w_int <= n:
                                shifted_num = np.concatenate((np.zeros(w_int, dtype=np.float64), prev_num[:-w_int]))
                            else:
                                shifted_num = np.zeros(n, dtype=np.float64)
                            numer_use = prev_num - shifted_num
                        
                        values = np.full(n, np.nan, dtype=np.float32)
                        if denom_use is not None:
                            mask = denom_use > 0
                            if mask.any():
                                values[mask] = (numer_use[mask] / denom_use[mask]).astype(np.float32)
                        
                        feat_name = f"{w}g_{k}_per_{self.level}" + ("" if ctx == "vs_any" else f"_{ctx}")
                        feature_arrays[feat_name] = values
                
                data: dict[str, pl.Series | np.ndarray] = {

thinking
**Checking daily_totals module**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 747ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL daily totals cache for rolling window computations.

Builds daily aggregated stats per player-context combination:
- vs_any: All games for this player
- vs_team: Games vs specific opponent

Cache structure per day:
  player_id, date, ctx, opponent → stat_num columns
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Callable
import polars as pl
import logging

from utils.feature.stats import NFL_PLAYER_STATS

logger = logging.getLogger(__name__)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 734ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    # Load function: loads player data for a given date
    def load_player_data(day: date) -> pl.DataFrame:
        """Load player-level data for a specific date from weekly parquets."""
        try:
            # Scan all weekly parquets
            df = pl.scan_parquet(
                str(source_dir / "season=*/week=*/part.parquet"),
                glob=True,
                hive_partitioning=True,
                missing_columns="insert",
                extra_columns="ignore",
            )

            schema = df.collect_schema()
            categorical_cols = [
                name
                for name, dtype in zip(schema.names(), schema.dtypes())
                if "Categorical" in str(dtype)

thinking
**Inspecting training pipeline**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/train.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
import os
import random
import inspect
import sys
import logging
import math


# Move logging to top and remove global thread pinning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path to allow module imports when running as script
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import yaml
import joblib
import optuna

thinking
**Outlining investigation strategy**
exec
/bin/zsh -lc 'rg "build_feature_matrix" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
66:__all__ = ["build_feature_matrix", "refresh_context_histories"]
226:def _build_feature_matrix_internal(
953:def build_feature_matrix(
979:            df = _build_feature_matrix_internal(
exec
/bin/zsh -lc "sed -n '220,520p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 698ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    )

# ---------------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------------

def _build_feature_matrix_internal(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    chunk_days: int = 7,
    recompute_intermediate: bool = True,
    build_rolling: bool = True,
    output_path: Path | None = None,
    primary_output_path: Path | None = None,
    cutoff_label: str | None = None,
) -> pl.DataFrame:
    """Run full feature pipeline up to *inclusive* `end_date` for a single cutoff window.


thinking
**Analyzing key feature sections**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/targets.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 743ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Target column validation for NFL player prediction models.

Target columns are created in utils/feature/player_game_level.py during aggregation.
This module validates that required targets exist and have correct dtypes.
"""

import polars as pl
import logging

logger = logging.getLogger(__name__)


# Expected target columns and their types
NFL_TARGET_SCHEMA = {
    # Binary classification targets
    "anytime_td": pl.Int8,  # Primary MVP target: did player score a TD?
    
    # Count/regression targets
    "td_count": pl.Int64,  # Number of TDs scored (receiving + rushing)

thinking
**Checking relevant player game data**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Player-game level aggregation for NFL predictions.

Aggregates play-by-play data to player-game level for all skill positions.
Each player gets one row per game with their total stats.

Target columns created:
- anytime_td (binary): Did player score a TD (receiving or rushing)?
- td_count (int): Number of TDs scored (receiving + rushing)
- passing_td (int): Number of passing TDs (for QBs)
- receiving_yards (float): Total receiving yards
- rushing_yards (float): Total rushing yards
- passing_yards (float): Total passing yards
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import datetime as dt

thinking
**Analyzing potential data issues and feature integration**

I'm focusing on spotting key issues like data leakage or unproduced columns in the large player_game_level file by reviewing mentions of rolling_window, MLB patterns, and stats usage around add_rolling_features. I want to understand if features rely on missing caches or incorrect columns that might disrupt aggregation or inference.
exec
/bin/zsh -lc 'rg "add_rolling_features" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 753ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
31:from utils.feature.rolling_window import add_rolling_features
831:        df_player_game_all = add_rolling_features(
exec
/bin/zsh -lc "sed -n '760,880p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

    logging.info("Enriching odds features (NFL)...")
    rows_before_odds = df_player_game_all.height
    df_player_game_all = add_nfl_odds_features_to_df(
        df_player_game_all,
        player_col="player_name",
        allow_schedule_fallback=False,
        drop_schedule_rows=True,
    )
    
    logging.info("Enriching player props (Anytime TD)...")
    df_player_game_all = add_player_props_to_df(
        df_player_game_all,
        player_col="player_name",
    )

    rows_after_odds = df_player_game_all.height
    logging.info(
        "Odds features attached; retained %d/%d rows (%.2f%%) with pre-cutoff snapshots",

thinking
**Examining rolling context distinctions**
exec
/bin/zsh -lc "sed -n '880,1120p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        logging.info("Added %d historical share features", len(share_exprs))
    leakage_cols = [
        "target_share",
        "carry_share",
        "pass_attempt_share",
        "red_zone_target_share",
        "red_zone_carry_share",
        "goal_to_go_target_share",
        "goal_to_go_carry_share",
    ]
    df_player_game_all = df_player_game_all.drop(leakage_cols, strict=False)
    # Keep raw outcome columns (targets, carries, etc.) in the feature matrix so they can
    # serve as supervised targets. These columns are excluded from model inputs later via
    # training configuration, so we avoid dropping them here to retain label availability.

    available_cols = set(df_player_game_all.columns)
    market_exprs: list[pl.Expr] = []
    if {"team_implied_total", "hist_target_share_prev"} <= available_cols:
        market_exprs.append(

thinking
**Analyzing data consistency and leakage risks**

I'm examining key conceptual challenges in the rolling window features, especially the mismatch between expected and available stats columns, which may create noise from zero-filled features. I'm also checking for potential data leakage from using caches that include same-day or future game data, ensuring that feature computations properly exclude current or future information during training and inference.

thinking
**Analyzing potential data leakage in feature computation**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
Consolidated NFL Odds Module
============================

This module provides all odds-related functionality for the NFL prediction pipeline:

1. OddsAPIClient - HTTP client for The Odds API with caching and retry logic
2. Game-level odds features (moneyline, spread, totals) from nfl_odds.py
3. Player-level odds features (anytime TD props) from odds_features.py
4. Odds snapshot collection and persistence from odds_snapshots.py

All odds logic is consolidated here to prevent drift and ensure consistency.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

thinking
**Analyzing rolling window and feature pipeline**

I’m digging into the rolling window implementation and its interaction with feature caches and pipelines. There are questions about how date ranges and future game predictions align, especially if the pipeline uses actual play data that wouldn’t exist yet for upcoming games. I want to verify how the prediction pipeline handles schedules and odds versus the completed game data the feature builds rely on. Let’s explore pipeline/predict next!
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/predict.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
NFL prediction pipeline
=======================

Generates anytime-touchdown probabilities for upcoming games by:

1. Loading the upcoming schedule and weekly rosters (via nfl_data_py)
2. Building a player-level scaffold for offensive skill positions
3. Enriching that scaffold with rolling-window features using the same
   caches that power the training pipeline
4. Sequentially running the model chain:
   Availability -> Team Pace -> Usage -> Efficiency -> Anytime TD
5. Writing an actionable slate under `output/predictions/`
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

thinking
**Analyzing feature computation inconsistencies**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_experiments.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 752ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

from typing import Iterable

import polars as pl


def add_experimental_rolling_features(
    df: pl.DataFrame,
    stats: Iterable[str],
    *,
    epsilon: float = 1e-3,
) -> pl.DataFrame:
    """
    Derive experimental rolling-window features to capture burst performance,
    exponential-style recency weighting, and directional trends.

    Parameters
    ----------

thinking
**Checking dataset relevance and usage**
exec
/bin/zsh -lc 'rg "ps_route_participation"' in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
pipeline/predict.py:    "ps_route_participation_plays",
pipeline/predict.py:    "ps_route_participation_pct",
pipeline/predict.py:    "ps_route_participation_pct_prev",
pipeline/predict.py:    "ps_route_participation_pct_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3": "ps_route_participation_plays",
pipeline/predict.py:        "1g_ps_route_participation_pct_per_game": "ps_route_participation_pct",
pipeline/predict.py:        "3g_ps_route_participation_pct_per_game": "ps_route_participation_pct_l3",
pipeline/predict.py:        "1g_ps_route_participation_plays_per_game": "ps_route_participation_plays",
pipeline/predict.py:        "3g_ps_route_participation_plays_per_game": "ps_route_participation_plays_l3",
pipeline/predict.py:        "ps_route_participation_pct",
pipeline/predict.py:        "ps_route_participation_plays",
utils/feature/stats.py:    "ps_route_participation_pct",
utils/feature/stats.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:        .agg(pl.len().alias("ps_route_participation_plays"))
utils/feature/player_game_level.py:            .then(pl.col("ps_route_participation_plays") / pl.col("ps_team_dropbacks"))

thinking
**Investigating player_game_level and anytime_td**
exec
/bin/zsh -lc 'rg "anytime_td" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 777ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- anytime_td (binary): Did player score a TD (receiving or rushing)?
    "market_anytime_td_prob",
    "market_anytime_td_consensus",
    "market_anytime_td_spread",
    "market_anytime_td_book_count",
    "market_anytime_td_prob_2h",
    "market_anytime_td_consensus_2h",
    "market_anytime_td_spread_2h",
    "market_anytime_td_book_count_2h",
    "market_anytime_td_prob_6h",
    "market_anytime_td_consensus_6h",
    "market_anytime_td_spread_6h",
    "market_anytime_td_book_count_6h",
    "market_anytime_td_prob_24h",
    "market_anytime_td_consensus_24h",
    "market_anytime_td_spread_24h",
    "market_anytime_td_book_count_24h",
    "market_anytime_td_prob_open",
    "market_anytime_td_consensus_open",

thinking
**Analyzing potential data leakage and timing issues**

I'm piecing together key concerns about possible data leakage in the rolling window system and training pipeline, especially related to future data inclusion, odds snapshot timing, and injury predictions. I'll highlight complications with caching and as-of joins that may cause leakage or inconsistencies.

thinking
**Evaluating data leakage risks in caches**

I'm thinking about whether cached stats might include data from invalid or future games, potentially causing leakage in training or inference. I need to check if features like offense_context have discrepancies between their training and inference implementations that might lead to mismatches or leaks.
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/offense_context.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 788ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import polars as pl

logger = logging.getLogger(__name__)

COORD_DEFAULT = "UNKNOWN_COORDINATOR"
QB_DEFAULT = "UNKNOWN_QB"
EPS = 1e-3

COORDINATOR_MAP_PATH = Path("data/processed/offensive_coordinators.parquet")


def _build_is_unavailable_expr(df: pl.DataFrame) -> pl.Expr:
    """Return a boolean expression marking players ruled out before cutoff."""

thinking
**Reviewing touchdown definitions for label leakage**
exec
/bin/zsh -lc 'rg -n "touchdowns" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 738ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
803:        pl.lit(0).cast(pl.Int64).alias("touchdowns"),
2577:            "touchdowns",
3108:              .alias("touchdowns"),
3121:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3123:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3165:              .alias("touchdowns"),
3177:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3178:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3220:              .alias("touchdowns"),
3232:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3233:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3271:            pl.col("touchdowns").fill_null(0).sum().alias("touchdowns"),
3284:        "passing_td", "rushing_td_count", "receiving_td_count", "touchdowns",
3293:    # derive it from rushing + receiving touchdowns.
3295:        pl.col("touchdowns").fill_null(0).alias("touchdowns"),
3306:        pl.when(pl.col("touchdowns") > 0)
3307:        .then(pl.col("touchdowns"))
3310:        .alias("_total_touchdowns")
3314:        (pl.col("_total_touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),

thinking
**Inspecting final assembly logic**
exec
/bin/zsh -lc "sed -n '3240,3320p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 771ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    Example: QB who passed and rushed, WR who received and rushed.
    """
    
    # Group by (player, game) and sum all stats
    merged = (
        df
        .group_by(["season", "week", "game_id", "game_date", "player_id", "player_name"])
        .agg([
            # Sum all yardage
            pl.col("passing_yards").fill_null(0).sum().alias("passing_yards"),
            pl.col("rushing_yards").fill_null(0).sum().alias("rushing_yards"),
            pl.col("receiving_yards").fill_null(0).sum().alias("receiving_yards"),
            
            # Sum counts
            pl.col("pass_attempt").fill_null(0).sum().alias("pass_attempt"),
            pl.col("completion").fill_null(0).sum().alias("completion"),
            pl.col("carry").fill_null(0).sum().alias("carry"),
            pl.col("target").fill_null(0).sum().alias("target"),

thinking
**Examining touchdown aggregation logic**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Play-level feature builder.

Reads cleaned daily Parquets and writes weekly play-level Parquets at
  data/processed/play_by_week/season=YYYY/week=W/part.parquet
"""

from pathlib import Path
from datetime import date
import polars as pl

CLEAN_DIR   = Path("data/cleaned")
PLAY_OUTDIR = Path("data/processed/play_by_week")

PLAY_COLS = [
    # Game identifiers
    "game_id",
    "season",

thinking
**Continuing review of derived statistics**
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 758ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
            ),
            (
                pl.col("defense_man_zone_type").cast(pl.Utf8).alias("defense_man_zone_type")
                if "defense_man_zone_type" in df.columns
                else pl.lit(None).alias("defense_man_zone_type")
            ),
            (
                pl.col("defense_coverage_type").cast(pl.Utf8).alias("defense_coverage_type")
                if "defense_coverage_type" in df.columns
                else pl.lit(None).alias("defense_coverage_type")
            ),
        ]
    )

    situ_exprs: list[pl.Expr] = []
    score_diff = pl.col("score_differential").fill_null(0.0)
    ydstogo = pl.col("ydstogo").fill_null(0.0)
    yardline = pl.col("yardline_100").cast(pl.Float32).fill_null(100.0)
    half_secs_expr = (

thinking
**Examining touchdown counting and rolling window concerns**

thinking
**Reviewing data shifts and plan progress**
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  → Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering details on config and training**
exec
/bin/zsh -lc "sed -n '1,200p' config/training.yaml" in /Users/jameswest/Desktop/nfl_predict succeeded in 815ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# config/training.yaml
# Configuration for the model training pipeline (pipeline/train.py)
# NFL Player Prediction Targets

# --- Data I/O ---
data:
  raw_pbp_dir: 'data/raw/pbp_by_day'  # NFL play-by-play raw data
  feature_matrix_path: 'data/processed/final/processed.parquet'  # NFL feature matrix
  
  # Directory to save trained model artifacts (models and feature lists).
  model_output_dir: 'output/models'
  
  # Directory to save evaluation reports and metrics.
  metrics_output_dir: 'output/metrics/'

# =============================================================================
# SHARED COLUMN DEFINITIONS (using YAML anchors to avoid duplication)
# =============================================================================
# These columns leak same-game outcomes and must be excluded from training.

thinking
**Analyzing potential data inconsistencies and leakage**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/asof.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 2.75s:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Helpers for aligning feature generation to a consistent decision-time snapshot."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Mapping

import yaml

from utils.general.paths import PROJ_ROOT


@lru_cache(maxsize=1)
def _load_cutoff_config() -> Mapping[str, Any]:
    """Load decision cutoff configuration from config/config.yaml."""
    config_path = PROJ_ROOT / "config" / "config.yaml"
    try:

thinking
**Identifying discrepancies in feature construction**

I'm analyzing several potential issues in how features are constructed, especially around rolling window computations, drive-level feature usage, pre-snap route participation data reliability, and labeling choices. I want to assemble a clear list of 6-7 concrete findings summarizing inconsistencies, incomplete implementations, and data limitations.

thinking
**Analyzing potential data leakage and pipeline inconsistencies**

I'm exploring possible data leakage issues, especially around using full-season roster and injury data without proper as-of gating in predictions. I’m also checking whether training pipelines enforce the claimed multi-stage model architecture and if various feature-building steps apply correct cutoff filters, since odds features might not respect decision cutoff timestamps, risking leakage. This involves reviewing training, feature, and monitoring docs alongside function implementations to understand the conceptual gaps and inconsistencies.

thinking
**Searching for add_player_odds_features_to_df function**
exec
/bin/zsh -lc 'rg "def add_player" -n utils/feature/odds.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 926ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
1099:def add_player_odds_features_to_df(
exec
/bin/zsh -lc "sed -n '1099,1250p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 732ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
def add_player_odds_features_to_df(
    df_pl: pl.DataFrame, *, api_key: str | None = None, player_col: str = "player_name",
    per_game_workers: int = 4, verbose: bool = False,
) -> pl.DataFrame:
    """Attach player odds features directly to an in-memory Polars DataFrame."""
    from utils.general.config import get_odds_api_key
    
    if df_pl.is_empty():
        return df_pl
    
    if not api_key:
        api_key = get_odds_api_key()
    
    if not api_key:
        logger.info("add_player_odds_features_to_df: no odds_api_key configured – adding empty columns and returning")
        empty_exprs = [pl.lit(None).cast(pl.Float32).alias(c) for c in PLAYER_ODDS_NUMERIC_COLS]
        flag_expr = [pl.lit(0).cast(pl.Int8).alias("odds_expected")]
        df_base = df_pl.drop(PLAYER_ODDS_NUMERIC_COLS + ODDS_FLAG_COLUMNS, strict=False)
        return df_base.with_columns(empty_exprs + flag_expr)

thinking
**Analyzing data leakage and feature completeness**

thinking
**Reviewing data leakage and caching inconsistencies**

I'm noticing potential data leakage in feature functions using live API calls and uncertain odds snapshot timing. There's also a mismatch between documented and actual cache directories that could confuse or cause bugs. I want to gather all issues before moving from feature pipeline review to modeling validation, paying attention to training config settings and columns flagged as leak-prone.
exec
/bin/zsh -lc 'rg "LEAK_PRONE_COLUMNS" -n utils' in /Users/jameswest/Desktop/nfl_predict succeeded in 870ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
utils/general/__init__.py:18:    LEAK_PRONE_COLUMNS,
utils/general/__init__.py:47:    "LEAK_PRONE_COLUMNS",
utils/general/constants.py:8:config/training.yaml should mirror LEAK_PRONE_COLUMNS for YAML-based configuration,
utils/general/constants.py:113:LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
utils/general/constants.py:208:    "LEAK_PRONE_COLUMNS",
exec
/bin/zsh -lc "sed -n '100,160p' utils/general/constants.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 800ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
}


# =============================================================================
# Leak-Prone Columns (Same-Game Outcomes)
# =============================================================================
# Stats that leak same-game outcomes and must be excluded from training features.
# These represent what happened in the game we're trying to predict.
#
# This is the AUTHORITATIVE definition. config/training.yaml mirrors this list
# using YAML anchors for per-problem configuration. Keep them in sync!
# Validated by: tests/test_training_config_features.py

LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
    # Yardage (same-game outcomes)
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    # Usage counts (same-game outcomes)

thinking
**Analyzing modeling pipeline inconsistencies**

I’m reviewing the training pipeline for possible incomplete or risky implementations—especially around derived targets, odds features that may leak future info, rolling window system mismatches, and caching issues causing potential feature inaccuracies. I'll note risky spots with specifics to file paths and behaviors to highlight data leakage or half-finished components.

thinking
**Identifying potential data leakage in features and inference**

thinking
**Analyzing touchdown column naming mismatch**

thinking
**Analyzing touchdown feature inconsistencies**

I notice a mismatch between "touchdown" and "touchdowns" columns in different dataframes, causing rolling touchdown features to be always zero—this looks like a conceptual bug. Also, the way stats are counted might cause double counting or misalignment. I want to fix these naming and aggregation issues and then update the project plan to mark step 2 as done and step 3 as in progress.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  → Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Analyzing model training and feature inconsistencies**

thinking
**Evaluating data quality and documentation gaps**

I'm analyzing issues with sparse or missing data affecting model training, especially around pre-snap route participation features and roster/injury alignment leaking future info. I also notice documentation inconsistencies in rolling window implementation and potential feature instability due to network-dependent API calls, all of which could harm predictive reliability and require careful handling in the next steps.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  ✓ Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Preparing final summary and suggestions**

I'll create a clear bullet list of findings ordered by severity with approximate file paths and lines, highlighting key issues like touchdown rolling features and pre-snap usage, then follow with numbered next steps for the user to consider.
codex
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.
tokens used
110,897
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.

================ END CODEX OUTPUT ================

[run_codex_chat] Finished. Exit code: 0

[run_codex_chat] Starting (exec - read-only) call...


================ PROMPT SENT TO CODEX ================


    Okay, great work.

    Please come up with the richest and most detailed plan possible to address every one of the points
    you brought up in the best way possible.

    Here is your last analysis / set of findings:

    <BEGIN_ANALYSIS_START>
    OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf67-266d-7c53-b835-f4d18d9aac20
--------
user

Context about this project:

- This is a project that takes NFL play-by-play data and uses it for modeling so that we can
  predict which players are likely to get an anytime touchdown (rushing or receiving, not passing)
  in an upcoming game.

- The goal is to build features and models that are as close as possible to the underlying
  football reality that produces touchdowns: play-calling tendencies, player usage, game state,
  defensive matchups, injuries, roles, red-zone behavior, etc.

- Training featurization must conceptually and logically match prediction/inference featurization.
  Anything that can only be known in hindsight at inference time (future data, downstream labels,
  or derived artifacts that use future information) is a form of leakage and must be eliminated.

- Over time, previous model runs and refactors may have left behind:
    * partially-implemented ideas,
    * experimental code paths,
    * hallucinated features,
    * or confusing / inconsistent logic.
  DO NOT assume that all existing code, features, configs, or comments are intentional or correct
  just because they exist. Treat any piece of code or configuration that does not clearly make
  sense in the context of the project as a candidate for cleanup, simplification, or removal.

Your task in this step:

Please analyze the current state of this project (code, data flow, feature engineering, and modeling)
and let me know:

1. Where things conceptually are not implemented correctly or are conceptually off, given the goal of
   predicting anytime TDs in a way that matches how football is actually played.
2. Where the modeling or data flow could be brought closer to "reality" as it actually plays out
  on the field. The goal of getting closer to reality is entirely so that the model is more
  accurate and metrics like AUC or other evaluation metrics improve.
3. Any incomplete implementations, half-finished ideas, or abandoned experimental paths.
4. Any wrong or misleading implementations (especially where names / comments and actual behavior diverge).
5. Any future data leaking into the modeling or feature pipeline (anything that uses knowledge from
  after the prediction cut-off point, including label-derived features).
6. Any underlying data sources or features that appear to not be working at all, or are effectively
  noise / dead weight.
7. Any areas where it looks like a previous run of a model or tool hallucinated structure, concepts,
  or features that don't actually exist in the real data or problem domain.

You should:

- Be concrete and specific in your findings.
- Call out anything that looks like hallucinated or legacy cruft that should probably be removed or
  reworked, instead of assuming it must be intentional.
- Focus on how each issue you find ultimately affects model realism and predictive performance.

mcp startup: no servers

thinking
**Listing project files for analysis**
exec
/bin/zsh -lc ls in /Users/jameswest/Desktop/nfl_predict succeeded in 765ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
README.md
__pycache__
audit
cache
codex_meta_loop.py
config
data
docs
logs
main.py
output
pipeline
pyproject.toml
requirements.txt
scripts
tests
tmp.parquet
tmp_features.parquet
utils

thinking
**Considering reading README**
exec
/bin/zsh -lc "sed -n '1,200p' README.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 762ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Anytime TD Prediction

## Key Scripts
- `pipeline/feature.py` – builds the player-game feature matrix, enforcing as-of leakage guards.
- `pipeline/train.py` – trains the multi-stage modeling graph (availability → usage → efficiency → TD).
- `pipeline/predict.py` – runs inference with the latest artifacts.
- `scripts/eval/odds_backtest.py` – compute odds-only baselines across horizons.
- `scripts/backtest/run_backtest.py` – evaluate existing prediction columns (AUC/Brier/log-loss) and write baseline summaries.
- `scripts/backtest/compare_baselines.py` – diff two baseline JSON files and highlight regressions.
- `scripts/analysis/horizon_eval.py` – consolidate metrics across decision horizons (input multiple prediction files labelled by horizon).
- `scripts/analysis/weather_travel_impact.py` – stratify TD hit rates by weather/travel features and emit plots (`docs/analysis/weather_travel/`) + JSON summaries.
- `scripts/monitoring/asof_coverage_report.py` – summarize snapshot coverage, write Markdown to `docs/monitoring/asof_coverage.md`, and append metrics history to `output/metrics/coverage/history.parquet`.

## Monitoring
- `docs/monitoring.md` documents leak guards, snapshot coverage expectations, roster snapshot regeneration steps, and instructions for running odds/backtests & analytics scripts.
- `docs/analysis/weather_travel.md` lists the latest weather/travel diagnostic artifacts.

## Quick Start
1. Build features:

thinking
**Preparing multi-step plan**
Plan update
Map out review steps to assess feature pipeline, modeling flow, leakage risks, and dead code.
  → Scan project structure and key config/docs to understand intended data flow and leakage guards
  • Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering key module info**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Feature pipeline orchestrator for NFL player-level predictions.

Stages
------
1. **build_play_level**         – derive cleaned play-level slices
2. **build_player_drive_level** – aggregate per drive
3. **build_player_game_level**  – aggregate per game and join context

This wrapper triggers each stage, optionally enforces a schema after each
write, and returns the final game-level DataFrame for downstream usage.
"""

import logging
import os
from datetime import date, timedelta
from typing import Sequence


thinking
**Planning deeper analysis**
exec
/bin/zsh -lc 'rg "TODO|FIXME|TBD"' in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
docs/rolling_window_implementation.md:### Immediate (TODO #8)

thinking
**Checking rolling window docs**
exec
/bin/zsh -lc "sed -n '1,200p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Rolling Window Feature System - Implementation Summary

## Overview

Successfully implemented a comprehensive rolling window feature system for NFL player predictions. The system computes time-series statistics for player performance across different contexts and time horizons.

## Architecture

### 1. Data Flow

```
Cleaned Play Data
    ↓
Player-Game Aggregations (player_game_level.py)
    ↓
Daily Totals Cache (nfl_daily_totals.py)
    ↓
Rolling Window Features (nfl_rolling_window.py)
    ↓
exec
/bin/zsh -lc "sed -n '200,400p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- **Incremental updates**: Only new dates need processing
- **Fast lookups**: Direct parquet reads by date
- **Memory efficient**: Streaming operations for large datasets
- **Reusable**: Same cache serves all rolling window computations

### Scalability
- **Per-player parallelization**: Each player's features computed independently
- **Lazy evaluation**: Uses Polars lazy API where possible
- **Chunked processing**: Pipeline processes data in configurable chunks

## Next Steps

### Immediate (TODO #8)
1. **Run feature pipeline** with new stats and rolling windows
   ```bash
   python -c "from pipeline.feature import build_feature_matrix; \
              from datetime import date; \
              build_feature_matrix(start_date=date(2024, 9, 5), \
                                  end_date=date(2024, 9, 9))"

thinking
**Verifying documentation accuracy**
exec
/bin/zsh -lc 'ls utils/feature' in /Users/jameswest/Desktop/nfl_predict succeeded in 746ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
__init__.py
__pycache__
asof.py
asof_metadata.py
concatenate_parquets.py
daily_totals.py
drive_level.py
game_level.py
io.py
keygen.py
odds.py
offense_context.py
opponent_splits.py
pace.py
play_level.py
player_drive_level.py
player_game_level.py
player_market_cli.py
qb_profiles.py

thinking
**Inspecting statistics components**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/stats.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 719ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL player statistics configuration for rolling window features.

Defines the standard set of stats tracked at all aggregation levels
(play, drive, game) for rolling window computations.

This module is the SINGLE SOURCE OF TRUTH for stat definitions used
throughout the pipeline. Import from here rather than defining locally.
"""

from typing import List

# =============================================================================
# Core Stats (used for drive and game level aggregation)
# =============================================================================

# Basic stats aggregated at player-drive level
PLAYER_DRIVE_STATS: List[str] = [
    # Yardage
    "receiving_yards",

thinking
**Verifying feature data existence**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 739ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL Rolling Window Features - Direct port from MLB rolling_window.py

Uses the exact same vectorized Polars pattern as the working MLB implementation.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache/feature/daily_totals")
NFL_SEASON_START_MMDD = (9, 1)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                    rolling_feature_names.append(f"{w}g_{k}_per_{self.level}{suffix}")

        if batch_min_date is None or batch_max_date is None:
            logger.warning("RollingWindow.compute received empty dataset; returning null rolling features.")
            result = ds
            for feat in rolling_feature_names:
                result = result.with_columns(pl.lit(None).cast(pl.Float32).alias(feat))
            return result.drop("__ts")
        
        # Determine lookback for CACHE scan
        # Load cache INCLUDING the batch dates - cumsum logic prevents leakage
        # Each row's rolling features = cumsum(up to row) - row_value = history BEFORE row
        if "lifetime" in windows:
            earliest = self._earliest_cache_date()
            lo_hist = earliest if earliest else batch_min_date - timedelta(days=365 * 10)
        elif "season" in windows:
            season_start = self._season_anchor(batch_min_date)
            lo_hist = season_start if season_start < batch_min_date else batch_min_date - timedelta(days=365)
        else:
exec
/bin/zsh -lc "sed -n '400,800p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                            denom_use = den_window_cache.get(w_int)
                            if denom_use is None:
                                denom_use = np.zeros(n, dtype=np.float64)
                            if w_int <= n:
                                shifted_num = np.concatenate((np.zeros(w_int, dtype=np.float64), prev_num[:-w_int]))
                            else:
                                shifted_num = np.zeros(n, dtype=np.float64)
                            numer_use = prev_num - shifted_num
                        
                        values = np.full(n, np.nan, dtype=np.float32)
                        if denom_use is not None:
                            mask = denom_use > 0
                            if mask.any():
                                values[mask] = (numer_use[mask] / denom_use[mask]).astype(np.float32)
                        
                        feat_name = f"{w}g_{k}_per_{self.level}" + ("" if ctx == "vs_any" else f"_{ctx}")
                        feature_arrays[feat_name] = values
                
                data: dict[str, pl.Series | np.ndarray] = {

thinking
**Checking daily_totals module**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 747ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL daily totals cache for rolling window computations.

Builds daily aggregated stats per player-context combination:
- vs_any: All games for this player
- vs_team: Games vs specific opponent

Cache structure per day:
  player_id, date, ctx, opponent → stat_num columns
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Callable
import polars as pl
import logging

from utils.feature.stats import NFL_PLAYER_STATS

logger = logging.getLogger(__name__)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 734ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    # Load function: loads player data for a given date
    def load_player_data(day: date) -> pl.DataFrame:
        """Load player-level data for a specific date from weekly parquets."""
        try:
            # Scan all weekly parquets
            df = pl.scan_parquet(
                str(source_dir / "season=*/week=*/part.parquet"),
                glob=True,
                hive_partitioning=True,
                missing_columns="insert",
                extra_columns="ignore",
            )

            schema = df.collect_schema()
            categorical_cols = [
                name
                for name, dtype in zip(schema.names(), schema.dtypes())
                if "Categorical" in str(dtype)

thinking
**Inspecting training pipeline**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/train.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
import os
import random
import inspect
import sys
import logging
import math


# Move logging to top and remove global thread pinning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path to allow module imports when running as script
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import yaml
import joblib
import optuna

thinking
**Outlining investigation strategy**
exec
/bin/zsh -lc 'rg "build_feature_matrix" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
66:__all__ = ["build_feature_matrix", "refresh_context_histories"]
226:def _build_feature_matrix_internal(
953:def build_feature_matrix(
979:            df = _build_feature_matrix_internal(
exec
/bin/zsh -lc "sed -n '220,520p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 698ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    )

# ---------------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------------

def _build_feature_matrix_internal(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    chunk_days: int = 7,
    recompute_intermediate: bool = True,
    build_rolling: bool = True,
    output_path: Path | None = None,
    primary_output_path: Path | None = None,
    cutoff_label: str | None = None,
) -> pl.DataFrame:
    """Run full feature pipeline up to *inclusive* `end_date` for a single cutoff window.


thinking
**Analyzing key feature sections**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/targets.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 743ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Target column validation for NFL player prediction models.

Target columns are created in utils/feature/player_game_level.py during aggregation.
This module validates that required targets exist and have correct dtypes.
"""

import polars as pl
import logging

logger = logging.getLogger(__name__)


# Expected target columns and their types
NFL_TARGET_SCHEMA = {
    # Binary classification targets
    "anytime_td": pl.Int8,  # Primary MVP target: did player score a TD?
    
    # Count/regression targets
    "td_count": pl.Int64,  # Number of TDs scored (receiving + rushing)

thinking
**Checking relevant player game data**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Player-game level aggregation for NFL predictions.

Aggregates play-by-play data to player-game level for all skill positions.
Each player gets one row per game with their total stats.

Target columns created:
- anytime_td (binary): Did player score a TD (receiving or rushing)?
- td_count (int): Number of TDs scored (receiving + rushing)
- passing_td (int): Number of passing TDs (for QBs)
- receiving_yards (float): Total receiving yards
- rushing_yards (float): Total rushing yards
- passing_yards (float): Total passing yards
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import datetime as dt

thinking
**Analyzing potential data issues and feature integration**

I'm focusing on spotting key issues like data leakage or unproduced columns in the large player_game_level file by reviewing mentions of rolling_window, MLB patterns, and stats usage around add_rolling_features. I want to understand if features rely on missing caches or incorrect columns that might disrupt aggregation or inference.
exec
/bin/zsh -lc 'rg "add_rolling_features" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 753ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
31:from utils.feature.rolling_window import add_rolling_features
831:        df_player_game_all = add_rolling_features(
exec
/bin/zsh -lc "sed -n '760,880p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

    logging.info("Enriching odds features (NFL)...")
    rows_before_odds = df_player_game_all.height
    df_player_game_all = add_nfl_odds_features_to_df(
        df_player_game_all,
        player_col="player_name",
        allow_schedule_fallback=False,
        drop_schedule_rows=True,
    )
    
    logging.info("Enriching player props (Anytime TD)...")
    df_player_game_all = add_player_props_to_df(
        df_player_game_all,
        player_col="player_name",
    )

    rows_after_odds = df_player_game_all.height
    logging.info(
        "Odds features attached; retained %d/%d rows (%.2f%%) with pre-cutoff snapshots",

thinking
**Examining rolling context distinctions**
exec
/bin/zsh -lc "sed -n '880,1120p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        logging.info("Added %d historical share features", len(share_exprs))
    leakage_cols = [
        "target_share",
        "carry_share",
        "pass_attempt_share",
        "red_zone_target_share",
        "red_zone_carry_share",
        "goal_to_go_target_share",
        "goal_to_go_carry_share",
    ]
    df_player_game_all = df_player_game_all.drop(leakage_cols, strict=False)
    # Keep raw outcome columns (targets, carries, etc.) in the feature matrix so they can
    # serve as supervised targets. These columns are excluded from model inputs later via
    # training configuration, so we avoid dropping them here to retain label availability.

    available_cols = set(df_player_game_all.columns)
    market_exprs: list[pl.Expr] = []
    if {"team_implied_total", "hist_target_share_prev"} <= available_cols:
        market_exprs.append(

thinking
**Analyzing data consistency and leakage risks**

I'm examining key conceptual challenges in the rolling window features, especially the mismatch between expected and available stats columns, which may create noise from zero-filled features. I'm also checking for potential data leakage from using caches that include same-day or future game data, ensuring that feature computations properly exclude current or future information during training and inference.

thinking
**Analyzing potential data leakage in feature computation**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
Consolidated NFL Odds Module
============================

This module provides all odds-related functionality for the NFL prediction pipeline:

1. OddsAPIClient - HTTP client for The Odds API with caching and retry logic
2. Game-level odds features (moneyline, spread, totals) from nfl_odds.py
3. Player-level odds features (anytime TD props) from odds_features.py
4. Odds snapshot collection and persistence from odds_snapshots.py

All odds logic is consolidated here to prevent drift and ensure consistency.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

thinking
**Analyzing rolling window and feature pipeline**

I’m digging into the rolling window implementation and its interaction with feature caches and pipelines. There are questions about how date ranges and future game predictions align, especially if the pipeline uses actual play data that wouldn’t exist yet for upcoming games. I want to verify how the prediction pipeline handles schedules and odds versus the completed game data the feature builds rely on. Let’s explore pipeline/predict next!
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/predict.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
NFL prediction pipeline
=======================

Generates anytime-touchdown probabilities for upcoming games by:

1. Loading the upcoming schedule and weekly rosters (via nfl_data_py)
2. Building a player-level scaffold for offensive skill positions
3. Enriching that scaffold with rolling-window features using the same
   caches that power the training pipeline
4. Sequentially running the model chain:
   Availability -> Team Pace -> Usage -> Efficiency -> Anytime TD
5. Writing an actionable slate under `output/predictions/`
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

thinking
**Analyzing feature computation inconsistencies**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_experiments.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 752ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

from typing import Iterable

import polars as pl


def add_experimental_rolling_features(
    df: pl.DataFrame,
    stats: Iterable[str],
    *,
    epsilon: float = 1e-3,
) -> pl.DataFrame:
    """
    Derive experimental rolling-window features to capture burst performance,
    exponential-style recency weighting, and directional trends.

    Parameters
    ----------

thinking
**Checking dataset relevance and usage**
exec
/bin/zsh -lc 'rg "ps_route_participation"' in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
pipeline/predict.py:    "ps_route_participation_plays",
pipeline/predict.py:    "ps_route_participation_pct",
pipeline/predict.py:    "ps_route_participation_pct_prev",
pipeline/predict.py:    "ps_route_participation_pct_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3": "ps_route_participation_plays",
pipeline/predict.py:        "1g_ps_route_participation_pct_per_game": "ps_route_participation_pct",
pipeline/predict.py:        "3g_ps_route_participation_pct_per_game": "ps_route_participation_pct_l3",
pipeline/predict.py:        "1g_ps_route_participation_plays_per_game": "ps_route_participation_plays",
pipeline/predict.py:        "3g_ps_route_participation_plays_per_game": "ps_route_participation_plays_l3",
pipeline/predict.py:        "ps_route_participation_pct",
pipeline/predict.py:        "ps_route_participation_plays",
utils/feature/stats.py:    "ps_route_participation_pct",
utils/feature/stats.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:        .agg(pl.len().alias("ps_route_participation_plays"))
utils/feature/player_game_level.py:            .then(pl.col("ps_route_participation_plays") / pl.col("ps_team_dropbacks"))

thinking
**Investigating player_game_level and anytime_td**
exec
/bin/zsh -lc 'rg "anytime_td" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 777ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- anytime_td (binary): Did player score a TD (receiving or rushing)?
    "market_anytime_td_prob",
    "market_anytime_td_consensus",
    "market_anytime_td_spread",
    "market_anytime_td_book_count",
    "market_anytime_td_prob_2h",
    "market_anytime_td_consensus_2h",
    "market_anytime_td_spread_2h",
    "market_anytime_td_book_count_2h",
    "market_anytime_td_prob_6h",
    "market_anytime_td_consensus_6h",
    "market_anytime_td_spread_6h",
    "market_anytime_td_book_count_6h",
    "market_anytime_td_prob_24h",
    "market_anytime_td_consensus_24h",
    "market_anytime_td_spread_24h",
    "market_anytime_td_book_count_24h",
    "market_anytime_td_prob_open",
    "market_anytime_td_consensus_open",

thinking
**Analyzing potential data leakage and timing issues**

I'm piecing together key concerns about possible data leakage in the rolling window system and training pipeline, especially related to future data inclusion, odds snapshot timing, and injury predictions. I'll highlight complications with caching and as-of joins that may cause leakage or inconsistencies.

thinking
**Evaluating data leakage risks in caches**

I'm thinking about whether cached stats might include data from invalid or future games, potentially causing leakage in training or inference. I need to check if features like offense_context have discrepancies between their training and inference implementations that might lead to mismatches or leaks.
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/offense_context.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 788ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import polars as pl

logger = logging.getLogger(__name__)

COORD_DEFAULT = "UNKNOWN_COORDINATOR"
QB_DEFAULT = "UNKNOWN_QB"
EPS = 1e-3

COORDINATOR_MAP_PATH = Path("data/processed/offensive_coordinators.parquet")


def _build_is_unavailable_expr(df: pl.DataFrame) -> pl.Expr:
    """Return a boolean expression marking players ruled out before cutoff."""

thinking
**Reviewing touchdown definitions for label leakage**
exec
/bin/zsh -lc 'rg -n "touchdowns" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 738ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
803:        pl.lit(0).cast(pl.Int64).alias("touchdowns"),
2577:            "touchdowns",
3108:              .alias("touchdowns"),
3121:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3123:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3165:              .alias("touchdowns"),
3177:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3178:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3220:              .alias("touchdowns"),
3232:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3233:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3271:            pl.col("touchdowns").fill_null(0).sum().alias("touchdowns"),
3284:        "passing_td", "rushing_td_count", "receiving_td_count", "touchdowns",
3293:    # derive it from rushing + receiving touchdowns.
3295:        pl.col("touchdowns").fill_null(0).alias("touchdowns"),
3306:        pl.when(pl.col("touchdowns") > 0)
3307:        .then(pl.col("touchdowns"))
3310:        .alias("_total_touchdowns")
3314:        (pl.col("_total_touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),

thinking
**Inspecting final assembly logic**
exec
/bin/zsh -lc "sed -n '3240,3320p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 771ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    Example: QB who passed and rushed, WR who received and rushed.
    """
    
    # Group by (player, game) and sum all stats
    merged = (
        df
        .group_by(["season", "week", "game_id", "game_date", "player_id", "player_name"])
        .agg([
            # Sum all yardage
            pl.col("passing_yards").fill_null(0).sum().alias("passing_yards"),
            pl.col("rushing_yards").fill_null(0).sum().alias("rushing_yards"),
            pl.col("receiving_yards").fill_null(0).sum().alias("receiving_yards"),
            
            # Sum counts
            pl.col("pass_attempt").fill_null(0).sum().alias("pass_attempt"),
            pl.col("completion").fill_null(0).sum().alias("completion"),
            pl.col("carry").fill_null(0).sum().alias("carry"),
            pl.col("target").fill_null(0).sum().alias("target"),

thinking
**Examining touchdown aggregation logic**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Play-level feature builder.

Reads cleaned daily Parquets and writes weekly play-level Parquets at
  data/processed/play_by_week/season=YYYY/week=W/part.parquet
"""

from pathlib import Path
from datetime import date
import polars as pl

CLEAN_DIR   = Path("data/cleaned")
PLAY_OUTDIR = Path("data/processed/play_by_week")

PLAY_COLS = [
    # Game identifiers
    "game_id",
    "season",

thinking
**Continuing review of derived statistics**
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 758ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
            ),
            (
                pl.col("defense_man_zone_type").cast(pl.Utf8).alias("defense_man_zone_type")
                if "defense_man_zone_type" in df.columns
                else pl.lit(None).alias("defense_man_zone_type")
            ),
            (
                pl.col("defense_coverage_type").cast(pl.Utf8).alias("defense_coverage_type")
                if "defense_coverage_type" in df.columns
                else pl.lit(None).alias("defense_coverage_type")
            ),
        ]
    )

    situ_exprs: list[pl.Expr] = []
    score_diff = pl.col("score_differential").fill_null(0.0)
    ydstogo = pl.col("ydstogo").fill_null(0.0)
    yardline = pl.col("yardline_100").cast(pl.Float32).fill_null(100.0)
    half_secs_expr = (

thinking
**Examining touchdown counting and rolling window concerns**

thinking
**Reviewing data shifts and plan progress**
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  → Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering details on config and training**
exec
/bin/zsh -lc "sed -n '1,200p' config/training.yaml" in /Users/jameswest/Desktop/nfl_predict succeeded in 815ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# config/training.yaml
# Configuration for the model training pipeline (pipeline/train.py)
# NFL Player Prediction Targets

# --- Data I/O ---
data:
  raw_pbp_dir: 'data/raw/pbp_by_day'  # NFL play-by-play raw data
  feature_matrix_path: 'data/processed/final/processed.parquet'  # NFL feature matrix
  
  # Directory to save trained model artifacts (models and feature lists).
  model_output_dir: 'output/models'
  
  # Directory to save evaluation reports and metrics.
  metrics_output_dir: 'output/metrics/'

# =============================================================================
# SHARED COLUMN DEFINITIONS (using YAML anchors to avoid duplication)
# =============================================================================
# These columns leak same-game outcomes and must be excluded from training.

thinking
**Analyzing potential data inconsistencies and leakage**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/asof.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 2.75s:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Helpers for aligning feature generation to a consistent decision-time snapshot."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Mapping

import yaml

from utils.general.paths import PROJ_ROOT


@lru_cache(maxsize=1)
def _load_cutoff_config() -> Mapping[str, Any]:
    """Load decision cutoff configuration from config/config.yaml."""
    config_path = PROJ_ROOT / "config" / "config.yaml"
    try:

thinking
**Identifying discrepancies in feature construction**

I'm analyzing several potential issues in how features are constructed, especially around rolling window computations, drive-level feature usage, pre-snap route participation data reliability, and labeling choices. I want to assemble a clear list of 6-7 concrete findings summarizing inconsistencies, incomplete implementations, and data limitations.

thinking
**Analyzing potential data leakage and pipeline inconsistencies**

I'm exploring possible data leakage issues, especially around using full-season roster and injury data without proper as-of gating in predictions. I’m also checking whether training pipelines enforce the claimed multi-stage model architecture and if various feature-building steps apply correct cutoff filters, since odds features might not respect decision cutoff timestamps, risking leakage. This involves reviewing training, feature, and monitoring docs alongside function implementations to understand the conceptual gaps and inconsistencies.

thinking
**Searching for add_player_odds_features_to_df function**
exec
/bin/zsh -lc 'rg "def add_player" -n utils/feature/odds.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 926ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
1099:def add_player_odds_features_to_df(
exec
/bin/zsh -lc "sed -n '1099,1250p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 732ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
def add_player_odds_features_to_df(
    df_pl: pl.DataFrame, *, api_key: str | None = None, player_col: str = "player_name",
    per_game_workers: int = 4, verbose: bool = False,
) -> pl.DataFrame:
    """Attach player odds features directly to an in-memory Polars DataFrame."""
    from utils.general.config import get_odds_api_key
    
    if df_pl.is_empty():
        return df_pl
    
    if not api_key:
        api_key = get_odds_api_key()
    
    if not api_key:
        logger.info("add_player_odds_features_to_df: no odds_api_key configured – adding empty columns and returning")
        empty_exprs = [pl.lit(None).cast(pl.Float32).alias(c) for c in PLAYER_ODDS_NUMERIC_COLS]
        flag_expr = [pl.lit(0).cast(pl.Int8).alias("odds_expected")]
        df_base = df_pl.drop(PLAYER_ODDS_NUMERIC_COLS + ODDS_FLAG_COLUMNS, strict=False)
        return df_base.with_columns(empty_exprs + flag_expr)

thinking
**Analyzing data leakage and feature completeness**

thinking
**Reviewing data leakage and caching inconsistencies**

I'm noticing potential data leakage in feature functions using live API calls and uncertain odds snapshot timing. There's also a mismatch between documented and actual cache directories that could confuse or cause bugs. I want to gather all issues before moving from feature pipeline review to modeling validation, paying attention to training config settings and columns flagged as leak-prone.
exec
/bin/zsh -lc 'rg "LEAK_PRONE_COLUMNS" -n utils' in /Users/jameswest/Desktop/nfl_predict succeeded in 870ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
utils/general/__init__.py:18:    LEAK_PRONE_COLUMNS,
utils/general/__init__.py:47:    "LEAK_PRONE_COLUMNS",
utils/general/constants.py:8:config/training.yaml should mirror LEAK_PRONE_COLUMNS for YAML-based configuration,
utils/general/constants.py:113:LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
utils/general/constants.py:208:    "LEAK_PRONE_COLUMNS",
exec
/bin/zsh -lc "sed -n '100,160p' utils/general/constants.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 800ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
}


# =============================================================================
# Leak-Prone Columns (Same-Game Outcomes)
# =============================================================================
# Stats that leak same-game outcomes and must be excluded from training features.
# These represent what happened in the game we're trying to predict.
#
# This is the AUTHORITATIVE definition. config/training.yaml mirrors this list
# using YAML anchors for per-problem configuration. Keep them in sync!
# Validated by: tests/test_training_config_features.py

LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
    # Yardage (same-game outcomes)
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    # Usage counts (same-game outcomes)

thinking
**Analyzing modeling pipeline inconsistencies**

I’m reviewing the training pipeline for possible incomplete or risky implementations—especially around derived targets, odds features that may leak future info, rolling window system mismatches, and caching issues causing potential feature inaccuracies. I'll note risky spots with specifics to file paths and behaviors to highlight data leakage or half-finished components.

thinking
**Identifying potential data leakage in features and inference**

thinking
**Analyzing touchdown column naming mismatch**

thinking
**Analyzing touchdown feature inconsistencies**

I notice a mismatch between "touchdown" and "touchdowns" columns in different dataframes, causing rolling touchdown features to be always zero—this looks like a conceptual bug. Also, the way stats are counted might cause double counting or misalignment. I want to fix these naming and aggregation issues and then update the project plan to mark step 2 as done and step 3 as in progress.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  → Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Analyzing model training and feature inconsistencies**

thinking
**Evaluating data quality and documentation gaps**

I'm analyzing issues with sparse or missing data affecting model training, especially around pre-snap route participation features and roster/injury alignment leaking future info. I also notice documentation inconsistencies in rolling window implementation and potential feature instability due to network-dependent API calls, all of which could harm predictive reliability and require careful handling in the next steps.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  ✓ Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Preparing final summary and suggestions**

I'll create a clear bullet list of findings ordered by severity with approximate file paths and lines, highlighting key issues like touchdown rolling features and pre-snap usage, then follow with numbered next steps for the user to consider.
codex
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.
tokens used
110,897
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.

    <END_ANALYSIS_END>

    This step is PURELY research, investigation, and planning.
    Do NOT make any edits to the code or data in this step.

    I want:
    - A structured, prioritized plan.
    - Clear steps that can be implemented by an agent in later steps.
    - Notes on risk or potential pitfalls where relevant, but focus on high-value changes.
    - Explicit attention to:
        * eliminating data leakage,
        * aligning features with real football mechanisms that drive anytime TDs,
        * cleaning up hallucinated / legacy cruft that no longer makes sense.

    Again: no edits here, just the plan.


======================================================


================ CODEX OUTPUT (streaming) ================

OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf6b-4ad5-78e1-9036-ed3c08d53d2c
--------
user

    Okay, great work.

    Please come up with the richest and most detailed plan possible to address every one of the points
    you brought up in the best way possible.

    Here is your last analysis / set of findings:

    <BEGIN_ANALYSIS_START>
    OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf67-266d-7c53-b835-f4d18d9aac20
--------
user

Context about this project:

- This is a project that takes NFL play-by-play data and uses it for modeling so that we can
  predict which players are likely to get an anytime touchdown (rushing or receiving, not passing)
  in an upcoming game.

- The goal is to build features and models that are as close as possible to the underlying
  football reality that produces touchdowns: play-calling tendencies, player usage, game state,
  defensive matchups, injuries, roles, red-zone behavior, etc.

- Training featurization must conceptually and logically match prediction/inference featurization.
  Anything that can only be known in hindsight at inference time (future data, downstream labels,
  or derived artifacts that use future information) is a form of leakage and must be eliminated.

- Over time, previous model runs and refactors may have left behind:
    * partially-implemented ideas,
    * experimental code paths,
    * hallucinated features,
    * or confusing / inconsistent logic.
  DO NOT assume that all existing code, features, configs, or comments are intentional or correct
  just because they exist. Treat any piece of code or configuration that does not clearly make
  sense in the context of the project as a candidate for cleanup, simplification, or removal.

Your task in this step:

Please analyze the current state of this project (code, data flow, feature engineering, and modeling)
and let me know:

1. Where things conceptually are not implemented correctly or are conceptually off, given the goal of
   predicting anytime TDs in a way that matches how football is actually played.
2. Where the modeling or data flow could be brought closer to "reality" as it actually plays out
  on the field. The goal of getting closer to reality is entirely so that the model is more
  accurate and metrics like AUC or other evaluation metrics improve.
3. Any incomplete implementations, half-finished ideas, or abandoned experimental paths.
4. Any wrong or misleading implementations (especially where names / comments and actual behavior diverge).
5. Any future data leaking into the modeling or feature pipeline (anything that uses knowledge from
  after the prediction cut-off point, including label-derived features).
6. Any underlying data sources or features that appear to not be working at all, or are effectively
  noise / dead weight.
7. Any areas where it looks like a previous run of a model or tool hallucinated structure, concepts,
  or features that don't actually exist in the real data or problem domain.

You should:

- Be concrete and specific in your findings.
- Call out anything that looks like hallucinated or legacy cruft that should probably be removed or
  reworked, instead of assuming it must be intentional.
- Focus on how each issue you find ultimately affects model realism and predictive performance.

mcp startup: no servers

thinking
**Listing project files for analysis**
exec
/bin/zsh -lc ls in /Users/jameswest/Desktop/nfl_predict succeeded in 765ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
README.md
__pycache__
audit
cache
codex_meta_loop.py
config
data
docs
logs
main.py
output
pipeline
pyproject.toml
requirements.txt
scripts
tests
tmp.parquet
tmp_features.parquet
utils

thinking
**Considering reading README**
exec
/bin/zsh -lc "sed -n '1,200p' README.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 762ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Anytime TD Prediction

## Key Scripts
- `pipeline/feature.py` – builds the player-game feature matrix, enforcing as-of leakage guards.
- `pipeline/train.py` – trains the multi-stage modeling graph (availability → usage → efficiency → TD).
- `pipeline/predict.py` – runs inference with the latest artifacts.
- `scripts/eval/odds_backtest.py` – compute odds-only baselines across horizons.
- `scripts/backtest/run_backtest.py` – evaluate existing prediction columns (AUC/Brier/log-loss) and write baseline summaries.
- `scripts/backtest/compare_baselines.py` – diff two baseline JSON files and highlight regressions.
- `scripts/analysis/horizon_eval.py` – consolidate metrics across decision horizons (input multiple prediction files labelled by horizon).
- `scripts/analysis/weather_travel_impact.py` – stratify TD hit rates by weather/travel features and emit plots (`docs/analysis/weather_travel/`) + JSON summaries.
- `scripts/monitoring/asof_coverage_report.py` – summarize snapshot coverage, write Markdown to `docs/monitoring/asof_coverage.md`, and append metrics history to `output/metrics/coverage/history.parquet`.

## Monitoring
- `docs/monitoring.md` documents leak guards, snapshot coverage expectations, roster snapshot regeneration steps, and instructions for running odds/backtests & analytics scripts.
- `docs/analysis/weather_travel.md` lists the latest weather/travel diagnostic artifacts.

## Quick Start
1. Build features:

thinking
**Preparing multi-step plan**
Plan update
Map out review steps to assess feature pipeline, modeling flow, leakage risks, and dead code.
  → Scan project structure and key config/docs to understand intended data flow and leakage guards
  • Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering key module info**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Feature pipeline orchestrator for NFL player-level predictions.

Stages
------
1. **build_play_level**         – derive cleaned play-level slices
2. **build_player_drive_level** – aggregate per drive
3. **build_player_game_level**  – aggregate per game and join context

This wrapper triggers each stage, optionally enforces a schema after each
write, and returns the final game-level DataFrame for downstream usage.
"""

import logging
import os
from datetime import date, timedelta
from typing import Sequence


thinking
**Planning deeper analysis**
exec
/bin/zsh -lc 'rg "TODO|FIXME|TBD"' in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
docs/rolling_window_implementation.md:### Immediate (TODO #8)

thinking
**Checking rolling window docs**
exec
/bin/zsh -lc "sed -n '1,200p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Rolling Window Feature System - Implementation Summary

## Overview

Successfully implemented a comprehensive rolling window feature system for NFL player predictions. The system computes time-series statistics for player performance across different contexts and time horizons.

## Architecture

### 1. Data Flow

```
Cleaned Play Data
    ↓
Player-Game Aggregations (player_game_level.py)
    ↓
Daily Totals Cache (nfl_daily_totals.py)
    ↓
Rolling Window Features (nfl_rolling_window.py)
    ↓
exec
/bin/zsh -lc "sed -n '200,400p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- **Incremental updates**: Only new dates need processing
- **Fast lookups**: Direct parquet reads by date
- **Memory efficient**: Streaming operations for large datasets
- **Reusable**: Same cache serves all rolling window computations

### Scalability
- **Per-player parallelization**: Each player's features computed independently
- **Lazy evaluation**: Uses Polars lazy API where possible
- **Chunked processing**: Pipeline processes data in configurable chunks

## Next Steps

### Immediate (TODO #8)
1. **Run feature pipeline** with new stats and rolling windows
   ```bash
   python -c "from pipeline.feature import build_feature_matrix; \
              from datetime import date; \
              build_feature_matrix(start_date=date(2024, 9, 5), \
                                  end_date=date(2024, 9, 9))"

thinking
**Verifying documentation accuracy**
exec
/bin/zsh -lc 'ls utils/feature' in /Users/jameswest/Desktop/nfl_predict succeeded in 746ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
__init__.py
__pycache__
asof.py
asof_metadata.py
concatenate_parquets.py
daily_totals.py
drive_level.py
game_level.py
io.py
keygen.py
odds.py
offense_context.py
opponent_splits.py
pace.py
play_level.py
player_drive_level.py
player_game_level.py
player_market_cli.py
qb_profiles.py

thinking
**Inspecting statistics components**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/stats.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 719ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL player statistics configuration for rolling window features.

Defines the standard set of stats tracked at all aggregation levels
(play, drive, game) for rolling window computations.

This module is the SINGLE SOURCE OF TRUTH for stat definitions used
throughout the pipeline. Import from here rather than defining locally.
"""

from typing import List

# =============================================================================
# Core Stats (used for drive and game level aggregation)
# =============================================================================

# Basic stats aggregated at player-drive level
PLAYER_DRIVE_STATS: List[str] = [
    # Yardage
    "receiving_yards",

thinking
**Verifying feature data existence**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 739ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL Rolling Window Features - Direct port from MLB rolling_window.py

Uses the exact same vectorized Polars pattern as the working MLB implementation.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache/feature/daily_totals")
NFL_SEASON_START_MMDD = (9, 1)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                    rolling_feature_names.append(f"{w}g_{k}_per_{self.level}{suffix}")

        if batch_min_date is None or batch_max_date is None:
            logger.warning("RollingWindow.compute received empty dataset; returning null rolling features.")
            result = ds
            for feat in rolling_feature_names:
                result = result.with_columns(pl.lit(None).cast(pl.Float32).alias(feat))
            return result.drop("__ts")
        
        # Determine lookback for CACHE scan
        # Load cache INCLUDING the batch dates - cumsum logic prevents leakage
        # Each row's rolling features = cumsum(up to row) - row_value = history BEFORE row
        if "lifetime" in windows:
            earliest = self._earliest_cache_date()
            lo_hist = earliest if earliest else batch_min_date - timedelta(days=365 * 10)
        elif "season" in windows:
            season_start = self._season_anchor(batch_min_date)
            lo_hist = season_start if season_start < batch_min_date else batch_min_date - timedelta(days=365)
        else:
exec
/bin/zsh -lc "sed -n '400,800p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                            denom_use = den_window_cache.get(w_int)
                            if denom_use is None:
                                denom_use = np.zeros(n, dtype=np.float64)
                            if w_int <= n:
                                shifted_num = np.concatenate((np.zeros(w_int, dtype=np.float64), prev_num[:-w_int]))
                            else:
                                shifted_num = np.zeros(n, dtype=np.float64)
                            numer_use = prev_num - shifted_num
                        
                        values = np.full(n, np.nan, dtype=np.float32)
                        if denom_use is not None:
                            mask = denom_use > 0
                            if mask.any():
                                values[mask] = (numer_use[mask] / denom_use[mask]).astype(np.float32)
                        
                        feat_name = f"{w}g_{k}_per_{self.level}" + ("" if ctx == "vs_any" else f"_{ctx}")
                        feature_arrays[feat_name] = values
                
                data: dict[str, pl.Series | np.ndarray] = {

thinking
**Checking daily_totals module**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 747ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL daily totals cache for rolling window computations.

Builds daily aggregated stats per player-context combination:
- vs_any: All games for this player
- vs_team: Games vs specific opponent

Cache structure per day:
  player_id, date, ctx, opponent → stat_num columns
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Callable
import polars as pl
import logging

from utils.feature.stats import NFL_PLAYER_STATS

logger = logging.getLogger(__name__)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 734ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    # Load function: loads player data for a given date
    def load_player_data(day: date) -> pl.DataFrame:
        """Load player-level data for a specific date from weekly parquets."""
        try:
            # Scan all weekly parquets
            df = pl.scan_parquet(
                str(source_dir / "season=*/week=*/part.parquet"),
                glob=True,
                hive_partitioning=True,
                missing_columns="insert",
                extra_columns="ignore",
            )

            schema = df.collect_schema()
            categorical_cols = [
                name
                for name, dtype in zip(schema.names(), schema.dtypes())
                if "Categorical" in str(dtype)

thinking
**Inspecting training pipeline**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/train.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
import os
import random
import inspect
import sys
import logging
import math


# Move logging to top and remove global thread pinning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path to allow module imports when running as script
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import yaml
import joblib
import optuna

thinking
**Outlining investigation strategy**
exec
/bin/zsh -lc 'rg "build_feature_matrix" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
66:__all__ = ["build_feature_matrix", "refresh_context_histories"]
226:def _build_feature_matrix_internal(
953:def build_feature_matrix(
979:            df = _build_feature_matrix_internal(
exec
/bin/zsh -lc "sed -n '220,520p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 698ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    )

# ---------------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------------

def _build_feature_matrix_internal(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    chunk_days: int = 7,
    recompute_intermediate: bool = True,
    build_rolling: bool = True,
    output_path: Path | None = None,
    primary_output_path: Path | None = None,
    cutoff_label: str | None = None,
) -> pl.DataFrame:
    """Run full feature pipeline up to *inclusive* `end_date` for a single cutoff window.


thinking
**Analyzing key feature sections**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/targets.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 743ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Target column validation for NFL player prediction models.

Target columns are created in utils/feature/player_game_level.py during aggregation.
This module validates that required targets exist and have correct dtypes.
"""

import polars as pl
import logging

logger = logging.getLogger(__name__)


# Expected target columns and their types
NFL_TARGET_SCHEMA = {
    # Binary classification targets
    "anytime_td": pl.Int8,  # Primary MVP target: did player score a TD?
    
    # Count/regression targets
    "td_count": pl.Int64,  # Number of TDs scored (receiving + rushing)

thinking
**Checking relevant player game data**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Player-game level aggregation for NFL predictions.

Aggregates play-by-play data to player-game level for all skill positions.
Each player gets one row per game with their total stats.

Target columns created:
- anytime_td (binary): Did player score a TD (receiving or rushing)?
- td_count (int): Number of TDs scored (receiving + rushing)
- passing_td (int): Number of passing TDs (for QBs)
- receiving_yards (float): Total receiving yards
- rushing_yards (float): Total rushing yards
- passing_yards (float): Total passing yards
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import datetime as dt

thinking
**Analyzing potential data issues and feature integration**

I'm focusing on spotting key issues like data leakage or unproduced columns in the large player_game_level file by reviewing mentions of rolling_window, MLB patterns, and stats usage around add_rolling_features. I want to understand if features rely on missing caches or incorrect columns that might disrupt aggregation or inference.
exec
/bin/zsh -lc 'rg "add_rolling_features" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 753ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
31:from utils.feature.rolling_window import add_rolling_features
831:        df_player_game_all = add_rolling_features(
exec
/bin/zsh -lc "sed -n '760,880p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

    logging.info("Enriching odds features (NFL)...")
    rows_before_odds = df_player_game_all.height
    df_player_game_all = add_nfl_odds_features_to_df(
        df_player_game_all,
        player_col="player_name",
        allow_schedule_fallback=False,
        drop_schedule_rows=True,
    )
    
    logging.info("Enriching player props (Anytime TD)...")
    df_player_game_all = add_player_props_to_df(
        df_player_game_all,
        player_col="player_name",
    )

    rows_after_odds = df_player_game_all.height
    logging.info(
        "Odds features attached; retained %d/%d rows (%.2f%%) with pre-cutoff snapshots",

thinking
**Examining rolling context distinctions**
exec
/bin/zsh -lc "sed -n '880,1120p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        logging.info("Added %d historical share features", len(share_exprs))
    leakage_cols = [
        "target_share",
        "carry_share",
        "pass_attempt_share",
        "red_zone_target_share",
        "red_zone_carry_share",
        "goal_to_go_target_share",
        "goal_to_go_carry_share",
    ]
    df_player_game_all = df_player_game_all.drop(leakage_cols, strict=False)
    # Keep raw outcome columns (targets, carries, etc.) in the feature matrix so they can
    # serve as supervised targets. These columns are excluded from model inputs later via
    # training configuration, so we avoid dropping them here to retain label availability.

    available_cols = set(df_player_game_all.columns)
    market_exprs: list[pl.Expr] = []
    if {"team_implied_total", "hist_target_share_prev"} <= available_cols:
        market_exprs.append(

thinking
**Analyzing data consistency and leakage risks**

I'm examining key conceptual challenges in the rolling window features, especially the mismatch between expected and available stats columns, which may create noise from zero-filled features. I'm also checking for potential data leakage from using caches that include same-day or future game data, ensuring that feature computations properly exclude current or future information during training and inference.

thinking
**Analyzing potential data leakage in feature computation**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
Consolidated NFL Odds Module
============================

This module provides all odds-related functionality for the NFL prediction pipeline:

1. OddsAPIClient - HTTP client for The Odds API with caching and retry logic
2. Game-level odds features (moneyline, spread, totals) from nfl_odds.py
3. Player-level odds features (anytime TD props) from odds_features.py
4. Odds snapshot collection and persistence from odds_snapshots.py

All odds logic is consolidated here to prevent drift and ensure consistency.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

thinking
**Analyzing rolling window and feature pipeline**

I’m digging into the rolling window implementation and its interaction with feature caches and pipelines. There are questions about how date ranges and future game predictions align, especially if the pipeline uses actual play data that wouldn’t exist yet for upcoming games. I want to verify how the prediction pipeline handles schedules and odds versus the completed game data the feature builds rely on. Let’s explore pipeline/predict next!
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/predict.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
NFL prediction pipeline
=======================

Generates anytime-touchdown probabilities for upcoming games by:

1. Loading the upcoming schedule and weekly rosters (via nfl_data_py)
2. Building a player-level scaffold for offensive skill positions
3. Enriching that scaffold with rolling-window features using the same
   caches that power the training pipeline
4. Sequentially running the model chain:
   Availability -> Team Pace -> Usage -> Efficiency -> Anytime TD
5. Writing an actionable slate under `output/predictions/`
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

thinking
**Analyzing feature computation inconsistencies**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_experiments.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 752ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

from typing import Iterable

import polars as pl


def add_experimental_rolling_features(
    df: pl.DataFrame,
    stats: Iterable[str],
    *,
    epsilon: float = 1e-3,
) -> pl.DataFrame:
    """
    Derive experimental rolling-window features to capture burst performance,
    exponential-style recency weighting, and directional trends.

    Parameters
    ----------

thinking
**Checking dataset relevance and usage**
exec
/bin/zsh -lc 'rg "ps_route_participation"' in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
pipeline/predict.py:    "ps_route_participation_plays",
pipeline/predict.py:    "ps_route_participation_pct",
pipeline/predict.py:    "ps_route_participation_pct_prev",
pipeline/predict.py:    "ps_route_participation_pct_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3": "ps_route_participation_plays",
pipeline/predict.py:        "1g_ps_route_participation_pct_per_game": "ps_route_participation_pct",
pipeline/predict.py:        "3g_ps_route_participation_pct_per_game": "ps_route_participation_pct_l3",
pipeline/predict.py:        "1g_ps_route_participation_plays_per_game": "ps_route_participation_plays",
pipeline/predict.py:        "3g_ps_route_participation_plays_per_game": "ps_route_participation_plays_l3",
pipeline/predict.py:        "ps_route_participation_pct",
pipeline/predict.py:        "ps_route_participation_plays",
utils/feature/stats.py:    "ps_route_participation_pct",
utils/feature/stats.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:        .agg(pl.len().alias("ps_route_participation_plays"))
utils/feature/player_game_level.py:            .then(pl.col("ps_route_participation_plays") / pl.col("ps_team_dropbacks"))

thinking
**Investigating player_game_level and anytime_td**
exec
/bin/zsh -lc 'rg "anytime_td" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 777ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- anytime_td (binary): Did player score a TD (receiving or rushing)?
    "market_anytime_td_prob",
    "market_anytime_td_consensus",
    "market_anytime_td_spread",
    "market_anytime_td_book_count",
    "market_anytime_td_prob_2h",
    "market_anytime_td_consensus_2h",
    "market_anytime_td_spread_2h",
    "market_anytime_td_book_count_2h",
    "market_anytime_td_prob_6h",
    "market_anytime_td_consensus_6h",
    "market_anytime_td_spread_6h",
    "market_anytime_td_book_count_6h",
    "market_anytime_td_prob_24h",
    "market_anytime_td_consensus_24h",
    "market_anytime_td_spread_24h",
    "market_anytime_td_book_count_24h",
    "market_anytime_td_prob_open",
    "market_anytime_td_consensus_open",

thinking
**Analyzing potential data leakage and timing issues**

I'm piecing together key concerns about possible data leakage in the rolling window system and training pipeline, especially related to future data inclusion, odds snapshot timing, and injury predictions. I'll highlight complications with caching and as-of joins that may cause leakage or inconsistencies.

thinking
**Evaluating data leakage risks in caches**

I'm thinking about whether cached stats might include data from invalid or future games, potentially causing leakage in training or inference. I need to check if features like offense_context have discrepancies between their training and inference implementations that might lead to mismatches or leaks.
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/offense_context.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 788ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import polars as pl

logger = logging.getLogger(__name__)

COORD_DEFAULT = "UNKNOWN_COORDINATOR"
QB_DEFAULT = "UNKNOWN_QB"
EPS = 1e-3

COORDINATOR_MAP_PATH = Path("data/processed/offensive_coordinators.parquet")


def _build_is_unavailable_expr(df: pl.DataFrame) -> pl.Expr:
    """Return a boolean expression marking players ruled out before cutoff."""

thinking
**Reviewing touchdown definitions for label leakage**
exec
/bin/zsh -lc 'rg -n "touchdowns" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 738ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
803:        pl.lit(0).cast(pl.Int64).alias("touchdowns"),
2577:            "touchdowns",
3108:              .alias("touchdowns"),
3121:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3123:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3165:              .alias("touchdowns"),
3177:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3178:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3220:              .alias("touchdowns"),
3232:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3233:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3271:            pl.col("touchdowns").fill_null(0).sum().alias("touchdowns"),
3284:        "passing_td", "rushing_td_count", "receiving_td_count", "touchdowns",
3293:    # derive it from rushing + receiving touchdowns.
3295:        pl.col("touchdowns").fill_null(0).alias("touchdowns"),
3306:        pl.when(pl.col("touchdowns") > 0)
3307:        .then(pl.col("touchdowns"))
3310:        .alias("_total_touchdowns")
3314:        (pl.col("_total_touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),

thinking
**Inspecting final assembly logic**
exec
/bin/zsh -lc "sed -n '3240,3320p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 771ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    Example: QB who passed and rushed, WR who received and rushed.
    """
    
    # Group by (player, game) and sum all stats
    merged = (
        df
        .group_by(["season", "week", "game_id", "game_date", "player_id", "player_name"])
        .agg([
            # Sum all yardage
            pl.col("passing_yards").fill_null(0).sum().alias("passing_yards"),
            pl.col("rushing_yards").fill_null(0).sum().alias("rushing_yards"),
            pl.col("receiving_yards").fill_null(0).sum().alias("receiving_yards"),
            
            # Sum counts
            pl.col("pass_attempt").fill_null(0).sum().alias("pass_attempt"),
            pl.col("completion").fill_null(0).sum().alias("completion"),
            pl.col("carry").fill_null(0).sum().alias("carry"),
            pl.col("target").fill_null(0).sum().alias("target"),

thinking
**Examining touchdown aggregation logic**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Play-level feature builder.

Reads cleaned daily Parquets and writes weekly play-level Parquets at
  data/processed/play_by_week/season=YYYY/week=W/part.parquet
"""

from pathlib import Path
from datetime import date
import polars as pl

CLEAN_DIR   = Path("data/cleaned")
PLAY_OUTDIR = Path("data/processed/play_by_week")

PLAY_COLS = [
    # Game identifiers
    "game_id",
    "season",

thinking
**Continuing review of derived statistics**
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 758ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
            ),
            (
                pl.col("defense_man_zone_type").cast(pl.Utf8).alias("defense_man_zone_type")
                if "defense_man_zone_type" in df.columns
                else pl.lit(None).alias("defense_man_zone_type")
            ),
            (
                pl.col("defense_coverage_type").cast(pl.Utf8).alias("defense_coverage_type")
                if "defense_coverage_type" in df.columns
                else pl.lit(None).alias("defense_coverage_type")
            ),
        ]
    )

    situ_exprs: list[pl.Expr] = []
    score_diff = pl.col("score_differential").fill_null(0.0)
    ydstogo = pl.col("ydstogo").fill_null(0.0)
    yardline = pl.col("yardline_100").cast(pl.Float32).fill_null(100.0)
    half_secs_expr = (

thinking
**Examining touchdown counting and rolling window concerns**

thinking
**Reviewing data shifts and plan progress**
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  → Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering details on config and training**
exec
/bin/zsh -lc "sed -n '1,200p' config/training.yaml" in /Users/jameswest/Desktop/nfl_predict succeeded in 815ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# config/training.yaml
# Configuration for the model training pipeline (pipeline/train.py)
# NFL Player Prediction Targets

# --- Data I/O ---
data:
  raw_pbp_dir: 'data/raw/pbp_by_day'  # NFL play-by-play raw data
  feature_matrix_path: 'data/processed/final/processed.parquet'  # NFL feature matrix
  
  # Directory to save trained model artifacts (models and feature lists).
  model_output_dir: 'output/models'
  
  # Directory to save evaluation reports and metrics.
  metrics_output_dir: 'output/metrics/'

# =============================================================================
# SHARED COLUMN DEFINITIONS (using YAML anchors to avoid duplication)
# =============================================================================
# These columns leak same-game outcomes and must be excluded from training.

thinking
**Analyzing potential data inconsistencies and leakage**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/asof.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 2.75s:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Helpers for aligning feature generation to a consistent decision-time snapshot."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Mapping

import yaml

from utils.general.paths import PROJ_ROOT


@lru_cache(maxsize=1)
def _load_cutoff_config() -> Mapping[str, Any]:
    """Load decision cutoff configuration from config/config.yaml."""
    config_path = PROJ_ROOT / "config" / "config.yaml"
    try:

thinking
**Identifying discrepancies in feature construction**

I'm analyzing several potential issues in how features are constructed, especially around rolling window computations, drive-level feature usage, pre-snap route participation data reliability, and labeling choices. I want to assemble a clear list of 6-7 concrete findings summarizing inconsistencies, incomplete implementations, and data limitations.

thinking
**Analyzing potential data leakage and pipeline inconsistencies**

I'm exploring possible data leakage issues, especially around using full-season roster and injury data without proper as-of gating in predictions. I’m also checking whether training pipelines enforce the claimed multi-stage model architecture and if various feature-building steps apply correct cutoff filters, since odds features might not respect decision cutoff timestamps, risking leakage. This involves reviewing training, feature, and monitoring docs alongside function implementations to understand the conceptual gaps and inconsistencies.

thinking
**Searching for add_player_odds_features_to_df function**
exec
/bin/zsh -lc 'rg "def add_player" -n utils/feature/odds.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 926ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
1099:def add_player_odds_features_to_df(
exec
/bin/zsh -lc "sed -n '1099,1250p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 732ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
def add_player_odds_features_to_df(
    df_pl: pl.DataFrame, *, api_key: str | None = None, player_col: str = "player_name",
    per_game_workers: int = 4, verbose: bool = False,
) -> pl.DataFrame:
    """Attach player odds features directly to an in-memory Polars DataFrame."""
    from utils.general.config import get_odds_api_key
    
    if df_pl.is_empty():
        return df_pl
    
    if not api_key:
        api_key = get_odds_api_key()
    
    if not api_key:
        logger.info("add_player_odds_features_to_df: no odds_api_key configured – adding empty columns and returning")
        empty_exprs = [pl.lit(None).cast(pl.Float32).alias(c) for c in PLAYER_ODDS_NUMERIC_COLS]
        flag_expr = [pl.lit(0).cast(pl.Int8).alias("odds_expected")]
        df_base = df_pl.drop(PLAYER_ODDS_NUMERIC_COLS + ODDS_FLAG_COLUMNS, strict=False)
        return df_base.with_columns(empty_exprs + flag_expr)

thinking
**Analyzing data leakage and feature completeness**

thinking
**Reviewing data leakage and caching inconsistencies**

I'm noticing potential data leakage in feature functions using live API calls and uncertain odds snapshot timing. There's also a mismatch between documented and actual cache directories that could confuse or cause bugs. I want to gather all issues before moving from feature pipeline review to modeling validation, paying attention to training config settings and columns flagged as leak-prone.
exec
/bin/zsh -lc 'rg "LEAK_PRONE_COLUMNS" -n utils' in /Users/jameswest/Desktop/nfl_predict succeeded in 870ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
utils/general/__init__.py:18:    LEAK_PRONE_COLUMNS,
utils/general/__init__.py:47:    "LEAK_PRONE_COLUMNS",
utils/general/constants.py:8:config/training.yaml should mirror LEAK_PRONE_COLUMNS for YAML-based configuration,
utils/general/constants.py:113:LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
utils/general/constants.py:208:    "LEAK_PRONE_COLUMNS",
exec
/bin/zsh -lc "sed -n '100,160p' utils/general/constants.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 800ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
}


# =============================================================================
# Leak-Prone Columns (Same-Game Outcomes)
# =============================================================================
# Stats that leak same-game outcomes and must be excluded from training features.
# These represent what happened in the game we're trying to predict.
#
# This is the AUTHORITATIVE definition. config/training.yaml mirrors this list
# using YAML anchors for per-problem configuration. Keep them in sync!
# Validated by: tests/test_training_config_features.py

LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
    # Yardage (same-game outcomes)
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    # Usage counts (same-game outcomes)

thinking
**Analyzing modeling pipeline inconsistencies**

I’m reviewing the training pipeline for possible incomplete or risky implementations—especially around derived targets, odds features that may leak future info, rolling window system mismatches, and caching issues causing potential feature inaccuracies. I'll note risky spots with specifics to file paths and behaviors to highlight data leakage or half-finished components.

thinking
**Identifying potential data leakage in features and inference**

thinking
**Analyzing touchdown column naming mismatch**

thinking
**Analyzing touchdown feature inconsistencies**

I notice a mismatch between "touchdown" and "touchdowns" columns in different dataframes, causing rolling touchdown features to be always zero—this looks like a conceptual bug. Also, the way stats are counted might cause double counting or misalignment. I want to fix these naming and aggregation issues and then update the project plan to mark step 2 as done and step 3 as in progress.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  → Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Analyzing model training and feature inconsistencies**

thinking
**Evaluating data quality and documentation gaps**

I'm analyzing issues with sparse or missing data affecting model training, especially around pre-snap route participation features and roster/injury alignment leaking future info. I also notice documentation inconsistencies in rolling window implementation and potential feature instability due to network-dependent API calls, all of which could harm predictive reliability and require careful handling in the next steps.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  ✓ Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Preparing final summary and suggestions**

I'll create a clear bullet list of findings ordered by severity with approximate file paths and lines, highlighting key issues like touchdown rolling features and pre-snap usage, then follow with numbered next steps for the user to consider.
codex
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.
tokens used
110,897
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.

    <END_ANALYSIS_END>

    This step is PURELY research, investigation, and planning.
    Do NOT make any edits to the code or data in this step.

    I want:
    - A structured, prioritized plan.
    - Clear steps that can be implemented by an agent in later steps.
    - Notes on risk or potential pitfalls where relevant, but focus on high-value changes.
    - Explicit attention to:
        * eliminating data leakage,
        * aligning features with real football mechanisms that drive anytime TDs,
        * cleaning up hallucinated / legacy cruft that no longer makes sense.

    Again: no edits here, just the plan.

mcp startup: no servers

thinking
**Drafting a structured planning outline**
codex
**Remediation Plan (ordered by impact & leakage risk)**
- Fix odds leakage/hallucination: freeze player prop odds to archived pre-cutoff snapshots; add snapshot loader keyed by game_id/team/player + snapshot_ts; in training, drop/flag rows without snapshots; in inference, refuse to call live API, optionally fall back to schedule-only baseline; add tests to assert no odds columns present when snapshot missing.
- Repair rolling TD history: align stat key `touchdowns` across `stats.py`, aggregation, and cache; rebuild daily_totals cache; add schema check to fail if any rolling TD feature is all-null/zero; rerun backtests to measure regained signal.
- Trim rolling window surface to supported set: codify a vetted set of stats/windows/contexts (e.g., {1g,3g,5g,season} × {vs_any}); remove ps_* and other missing-cache stats from rolling requests; update docs to match; add runtime guard to error on requested stats not found in cache.
- Enforce as-of gating for roster/injury inputs in prediction/backtests: add decision-time cutoff (e.g., T-6h) and snapshot source; if unavailable, drop those features or mark unavailable; add regression test ensuring no injury status is from after game start in historical inference.
- Remove/retire ps_* targets/features if coverage is low: measure non-null rate by season; if sparse, drop from training configs and rolling stats; otherwise backfill realistically or scope to seasons with data; add check to block tasks with <X% coverage.
- Clarify/implement drive-level history or delete: either build drive-level daily cache and wire into rolling computation, or remove drive-context flags/constants/docs; add assertion preventing “drive” level when cache absent.
- Align docs with reality and guard drift: sync paths/names (cache/feature/daily_totals vs nfl_daily_totals); document the final rolling feature list and cutoff rules; add a pre-flight validation script that checks cache schemas, non-null rates, and leakage-sensitive columns before training.

**Execution Steps**
1) Odds snapshots: design schema + loader; replace `add_player_props_to_df` call; add fallback behavior and tests; rerun feature build on a small date range to validate columns.
2) Rolling TD fix: rename stat key, rebuild cache for a sample window, ensure rolling TD features populate; update tests.
3) Rolling feature pruning: define allowed stats/windows/contexts; update config/constants + pipeline call; add guard for missing stats; rerun feature build and inspect null/constant rates.
4) Roster/injury gating: add cutoff param in predict/backtest; wire to snapshot source; add assertion that injury_status_ts <= cutoff; adjust backtests to use gated features.
5) ps_* triage: compute coverage by season; decide keep/drop; update training.yaml and problem definitions; if dropped, remove from rolling stats and tasks.
6) Drive-level decision: either implement cache builder (drive totals) and wire into rolling_window or delete related flags/docs; add guard to prevent silent no-op.
7) Docs/validation: update rolling_window_implementation.md and monitoring docs; add validation script to check schemas, coverage, leak-prone columns excluded from train set; run it in CI/pre-flight.

**Risks/Pitfalls to watch**
- Cache rebuild time/size; may need incremental rebuild and date filters.
- Snapshot availability gaps; must define strict fallbacks to avoid silent leakage.
- Training/inference drift if odds/roster data are absent—prefer hard failure over silent nulls.
- Removing ps_* may change model interfaces; update downstream consumers/tests accordingly.
tokens used
15,743
**Remediation Plan (ordered by impact & leakage risk)**
- Fix odds leakage/hallucination: freeze player prop odds to archived pre-cutoff snapshots; add snapshot loader keyed by game_id/team/player + snapshot_ts; in training, drop/flag rows without snapshots; in inference, refuse to call live API, optionally fall back to schedule-only baseline; add tests to assert no odds columns present when snapshot missing.
- Repair rolling TD history: align stat key `touchdowns` across `stats.py`, aggregation, and cache; rebuild daily_totals cache; add schema check to fail if any rolling TD feature is all-null/zero; rerun backtests to measure regained signal.
- Trim rolling window surface to supported set: codify a vetted set of stats/windows/contexts (e.g., {1g,3g,5g,season} × {vs_any}); remove ps_* and other missing-cache stats from rolling requests; update docs to match; add runtime guard to error on requested stats not found in cache.
- Enforce as-of gating for roster/injury inputs in prediction/backtests: add decision-time cutoff (e.g., T-6h) and snapshot source; if unavailable, drop those features or mark unavailable; add regression test ensuring no injury status is from after game start in historical inference.
- Remove/retire ps_* targets/features if coverage is low: measure non-null rate by season; if sparse, drop from training configs and rolling stats; otherwise backfill realistically or scope to seasons with data; add check to block tasks with <X% coverage.
- Clarify/implement drive-level history or delete: either build drive-level daily cache and wire into rolling computation, or remove drive-context flags/constants/docs; add assertion preventing “drive” level when cache absent.
- Align docs with reality and guard drift: sync paths/names (cache/feature/daily_totals vs nfl_daily_totals); document the final rolling feature list and cutoff rules; add a pre-flight validation script that checks cache schemas, non-null rates, and leakage-sensitive columns before training.

**Execution Steps**
1) Odds snapshots: design schema + loader; replace `add_player_props_to_df` call; add fallback behavior and tests; rerun feature build on a small date range to validate columns.
2) Rolling TD fix: rename stat key, rebuild cache for a sample window, ensure rolling TD features populate; update tests.
3) Rolling feature pruning: define allowed stats/windows/contexts; update config/constants + pipeline call; add guard for missing stats; rerun feature build and inspect null/constant rates.
4) Roster/injury gating: add cutoff param in predict/backtest; wire to snapshot source; add assertion that injury_status_ts <= cutoff; adjust backtests to use gated features.
5) ps_* triage: compute coverage by season; decide keep/drop; update training.yaml and problem definitions; if dropped, remove from rolling stats and tasks.
6) Drive-level decision: either implement cache builder (drive totals) and wire into rolling_window or delete related flags/docs; add guard to prevent silent no-op.
7) Docs/validation: update rolling_window_implementation.md and monitoring docs; add validation script to check schemas, coverage, leak-prone columns excluded from train set; run it in CI/pre-flight.

**Risks/Pitfalls to watch**
- Cache rebuild time/size; may need incremental rebuild and date filters.
- Snapshot availability gaps; must define strict fallbacks to avoid silent leakage.
- Training/inference drift if odds/roster data are absent—prefer hard failure over silent nulls.
- Removing ps_* may change model interfaces; update downstream consumers/tests accordingly.

================ END CODEX OUTPUT ================

[run_codex_chat] Finished. Exit code: 0
[phase_3_implement] Iteration 1/10

[run_codex_exec] Starting exec call... (sandbox=workspace-write)


================ PROMPT SENT TO CODEX ================

Okay, great. Please work like an agent and implement the following plan in this repository.

    Here is the plan:

    <BEGIN_PLAN_START>
    OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf6b-4ad5-78e1-9036-ed3c08d53d2c
--------
user

    Okay, great work.

    Please come up with the richest and most detailed plan possible to address every one of the points
    you brought up in the best way possible.

    Here is your last analysis / set of findings:

    <BEGIN_ANALYSIS_START>
    OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf67-266d-7c53-b835-f4d18d9aac20
--------
user

Context about this project:

- This is a project that takes NFL play-by-play data and uses it for modeling so that we can
  predict which players are likely to get an anytime touchdown (rushing or receiving, not passing)
  in an upcoming game.

- The goal is to build features and models that are as close as possible to the underlying
  football reality that produces touchdowns: play-calling tendencies, player usage, game state,
  defensive matchups, injuries, roles, red-zone behavior, etc.

- Training featurization must conceptually and logically match prediction/inference featurization.
  Anything that can only be known in hindsight at inference time (future data, downstream labels,
  or derived artifacts that use future information) is a form of leakage and must be eliminated.

- Over time, previous model runs and refactors may have left behind:
    * partially-implemented ideas,
    * experimental code paths,
    * hallucinated features,
    * or confusing / inconsistent logic.
  DO NOT assume that all existing code, features, configs, or comments are intentional or correct
  just because they exist. Treat any piece of code or configuration that does not clearly make
  sense in the context of the project as a candidate for cleanup, simplification, or removal.

Your task in this step:

Please analyze the current state of this project (code, data flow, feature engineering, and modeling)
and let me know:

1. Where things conceptually are not implemented correctly or are conceptually off, given the goal of
   predicting anytime TDs in a way that matches how football is actually played.
2. Where the modeling or data flow could be brought closer to "reality" as it actually plays out
  on the field. The goal of getting closer to reality is entirely so that the model is more
  accurate and metrics like AUC or other evaluation metrics improve.
3. Any incomplete implementations, half-finished ideas, or abandoned experimental paths.
4. Any wrong or misleading implementations (especially where names / comments and actual behavior diverge).
5. Any future data leaking into the modeling or feature pipeline (anything that uses knowledge from
  after the prediction cut-off point, including label-derived features).
6. Any underlying data sources or features that appear to not be working at all, or are effectively
  noise / dead weight.
7. Any areas where it looks like a previous run of a model or tool hallucinated structure, concepts,
  or features that don't actually exist in the real data or problem domain.

You should:

- Be concrete and specific in your findings.
- Call out anything that looks like hallucinated or legacy cruft that should probably be removed or
  reworked, instead of assuming it must be intentional.
- Focus on how each issue you find ultimately affects model realism and predictive performance.

mcp startup: no servers

thinking
**Listing project files for analysis**
exec
/bin/zsh -lc ls in /Users/jameswest/Desktop/nfl_predict succeeded in 765ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
README.md
__pycache__
audit
cache
codex_meta_loop.py
config
data
docs
logs
main.py
output
pipeline
pyproject.toml
requirements.txt
scripts
tests
tmp.parquet
tmp_features.parquet
utils

thinking
**Considering reading README**
exec
/bin/zsh -lc "sed -n '1,200p' README.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 762ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Anytime TD Prediction

## Key Scripts
- `pipeline/feature.py` – builds the player-game feature matrix, enforcing as-of leakage guards.
- `pipeline/train.py` – trains the multi-stage modeling graph (availability → usage → efficiency → TD).
- `pipeline/predict.py` – runs inference with the latest artifacts.
- `scripts/eval/odds_backtest.py` – compute odds-only baselines across horizons.
- `scripts/backtest/run_backtest.py` – evaluate existing prediction columns (AUC/Brier/log-loss) and write baseline summaries.
- `scripts/backtest/compare_baselines.py` – diff two baseline JSON files and highlight regressions.
- `scripts/analysis/horizon_eval.py` – consolidate metrics across decision horizons (input multiple prediction files labelled by horizon).
- `scripts/analysis/weather_travel_impact.py` – stratify TD hit rates by weather/travel features and emit plots (`docs/analysis/weather_travel/`) + JSON summaries.
- `scripts/monitoring/asof_coverage_report.py` – summarize snapshot coverage, write Markdown to `docs/monitoring/asof_coverage.md`, and append metrics history to `output/metrics/coverage/history.parquet`.

## Monitoring
- `docs/monitoring.md` documents leak guards, snapshot coverage expectations, roster snapshot regeneration steps, and instructions for running odds/backtests & analytics scripts.
- `docs/analysis/weather_travel.md` lists the latest weather/travel diagnostic artifacts.

## Quick Start
1. Build features:

thinking
**Preparing multi-step plan**
Plan update
Map out review steps to assess feature pipeline, modeling flow, leakage risks, and dead code.
  → Scan project structure and key config/docs to understand intended data flow and leakage guards
  • Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering key module info**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Feature pipeline orchestrator for NFL player-level predictions.

Stages
------
1. **build_play_level**         – derive cleaned play-level slices
2. **build_player_drive_level** – aggregate per drive
3. **build_player_game_level**  – aggregate per game and join context

This wrapper triggers each stage, optionally enforces a schema after each
write, and returns the final game-level DataFrame for downstream usage.
"""

import logging
import os
from datetime import date, timedelta
from typing import Sequence


thinking
**Planning deeper analysis**
exec
/bin/zsh -lc 'rg "TODO|FIXME|TBD"' in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
docs/rolling_window_implementation.md:### Immediate (TODO #8)

thinking
**Checking rolling window docs**
exec
/bin/zsh -lc "sed -n '1,200p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Rolling Window Feature System - Implementation Summary

## Overview

Successfully implemented a comprehensive rolling window feature system for NFL player predictions. The system computes time-series statistics for player performance across different contexts and time horizons.

## Architecture

### 1. Data Flow

```
Cleaned Play Data
    ↓
Player-Game Aggregations (player_game_level.py)
    ↓
Daily Totals Cache (nfl_daily_totals.py)
    ↓
Rolling Window Features (nfl_rolling_window.py)
    ↓
exec
/bin/zsh -lc "sed -n '200,400p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- **Incremental updates**: Only new dates need processing
- **Fast lookups**: Direct parquet reads by date
- **Memory efficient**: Streaming operations for large datasets
- **Reusable**: Same cache serves all rolling window computations

### Scalability
- **Per-player parallelization**: Each player's features computed independently
- **Lazy evaluation**: Uses Polars lazy API where possible
- **Chunked processing**: Pipeline processes data in configurable chunks

## Next Steps

### Immediate (TODO #8)
1. **Run feature pipeline** with new stats and rolling windows
   ```bash
   python -c "from pipeline.feature import build_feature_matrix; \
              from datetime import date; \
              build_feature_matrix(start_date=date(2024, 9, 5), \
                                  end_date=date(2024, 9, 9))"

thinking
**Verifying documentation accuracy**
exec
/bin/zsh -lc 'ls utils/feature' in /Users/jameswest/Desktop/nfl_predict succeeded in 746ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
__init__.py
__pycache__
asof.py
asof_metadata.py
concatenate_parquets.py
daily_totals.py
drive_level.py
game_level.py
io.py
keygen.py
odds.py
offense_context.py
opponent_splits.py
pace.py
play_level.py
player_drive_level.py
player_game_level.py
player_market_cli.py
qb_profiles.py

thinking
**Inspecting statistics components**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/stats.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 719ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL player statistics configuration for rolling window features.

Defines the standard set of stats tracked at all aggregation levels
(play, drive, game) for rolling window computations.

This module is the SINGLE SOURCE OF TRUTH for stat definitions used
throughout the pipeline. Import from here rather than defining locally.
"""

from typing import List

# =============================================================================
# Core Stats (used for drive and game level aggregation)
# =============================================================================

# Basic stats aggregated at player-drive level
PLAYER_DRIVE_STATS: List[str] = [
    # Yardage
    "receiving_yards",

thinking
**Verifying feature data existence**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 739ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL Rolling Window Features - Direct port from MLB rolling_window.py

Uses the exact same vectorized Polars pattern as the working MLB implementation.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache/feature/daily_totals")
NFL_SEASON_START_MMDD = (9, 1)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                    rolling_feature_names.append(f"{w}g_{k}_per_{self.level}{suffix}")

        if batch_min_date is None or batch_max_date is None:
            logger.warning("RollingWindow.compute received empty dataset; returning null rolling features.")
            result = ds
            for feat in rolling_feature_names:
                result = result.with_columns(pl.lit(None).cast(pl.Float32).alias(feat))
            return result.drop("__ts")
        
        # Determine lookback for CACHE scan
        # Load cache INCLUDING the batch dates - cumsum logic prevents leakage
        # Each row's rolling features = cumsum(up to row) - row_value = history BEFORE row
        if "lifetime" in windows:
            earliest = self._earliest_cache_date()
            lo_hist = earliest if earliest else batch_min_date - timedelta(days=365 * 10)
        elif "season" in windows:
            season_start = self._season_anchor(batch_min_date)
            lo_hist = season_start if season_start < batch_min_date else batch_min_date - timedelta(days=365)
        else:
exec
/bin/zsh -lc "sed -n '400,800p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                            denom_use = den_window_cache.get(w_int)
                            if denom_use is None:
                                denom_use = np.zeros(n, dtype=np.float64)
                            if w_int <= n:
                                shifted_num = np.concatenate((np.zeros(w_int, dtype=np.float64), prev_num[:-w_int]))
                            else:
                                shifted_num = np.zeros(n, dtype=np.float64)
                            numer_use = prev_num - shifted_num
                        
                        values = np.full(n, np.nan, dtype=np.float32)
                        if denom_use is not None:
                            mask = denom_use > 0
                            if mask.any():
                                values[mask] = (numer_use[mask] / denom_use[mask]).astype(np.float32)
                        
                        feat_name = f"{w}g_{k}_per_{self.level}" + ("" if ctx == "vs_any" else f"_{ctx}")
                        feature_arrays[feat_name] = values
                
                data: dict[str, pl.Series | np.ndarray] = {

thinking
**Checking daily_totals module**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 747ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL daily totals cache for rolling window computations.

Builds daily aggregated stats per player-context combination:
- vs_any: All games for this player
- vs_team: Games vs specific opponent

Cache structure per day:
  player_id, date, ctx, opponent → stat_num columns
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Callable
import polars as pl
import logging

from utils.feature.stats import NFL_PLAYER_STATS

logger = logging.getLogger(__name__)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 734ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    # Load function: loads player data for a given date
    def load_player_data(day: date) -> pl.DataFrame:
        """Load player-level data for a specific date from weekly parquets."""
        try:
            # Scan all weekly parquets
            df = pl.scan_parquet(
                str(source_dir / "season=*/week=*/part.parquet"),
                glob=True,
                hive_partitioning=True,
                missing_columns="insert",
                extra_columns="ignore",
            )

            schema = df.collect_schema()
            categorical_cols = [
                name
                for name, dtype in zip(schema.names(), schema.dtypes())
                if "Categorical" in str(dtype)

thinking
**Inspecting training pipeline**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/train.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
import os
import random
import inspect
import sys
import logging
import math


# Move logging to top and remove global thread pinning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path to allow module imports when running as script
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import yaml
import joblib
import optuna

thinking
**Outlining investigation strategy**
exec
/bin/zsh -lc 'rg "build_feature_matrix" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
66:__all__ = ["build_feature_matrix", "refresh_context_histories"]
226:def _build_feature_matrix_internal(
953:def build_feature_matrix(
979:            df = _build_feature_matrix_internal(
exec
/bin/zsh -lc "sed -n '220,520p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 698ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    )

# ---------------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------------

def _build_feature_matrix_internal(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    chunk_days: int = 7,
    recompute_intermediate: bool = True,
    build_rolling: bool = True,
    output_path: Path | None = None,
    primary_output_path: Path | None = None,
    cutoff_label: str | None = None,
) -> pl.DataFrame:
    """Run full feature pipeline up to *inclusive* `end_date` for a single cutoff window.


thinking
**Analyzing key feature sections**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/targets.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 743ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Target column validation for NFL player prediction models.

Target columns are created in utils/feature/player_game_level.py during aggregation.
This module validates that required targets exist and have correct dtypes.
"""

import polars as pl
import logging

logger = logging.getLogger(__name__)


# Expected target columns and their types
NFL_TARGET_SCHEMA = {
    # Binary classification targets
    "anytime_td": pl.Int8,  # Primary MVP target: did player score a TD?
    
    # Count/regression targets
    "td_count": pl.Int64,  # Number of TDs scored (receiving + rushing)

thinking
**Checking relevant player game data**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Player-game level aggregation for NFL predictions.

Aggregates play-by-play data to player-game level for all skill positions.
Each player gets one row per game with their total stats.

Target columns created:
- anytime_td (binary): Did player score a TD (receiving or rushing)?
- td_count (int): Number of TDs scored (receiving + rushing)
- passing_td (int): Number of passing TDs (for QBs)
- receiving_yards (float): Total receiving yards
- rushing_yards (float): Total rushing yards
- passing_yards (float): Total passing yards
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import datetime as dt

thinking
**Analyzing potential data issues and feature integration**

I'm focusing on spotting key issues like data leakage or unproduced columns in the large player_game_level file by reviewing mentions of rolling_window, MLB patterns, and stats usage around add_rolling_features. I want to understand if features rely on missing caches or incorrect columns that might disrupt aggregation or inference.
exec
/bin/zsh -lc 'rg "add_rolling_features" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 753ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
31:from utils.feature.rolling_window import add_rolling_features
831:        df_player_game_all = add_rolling_features(
exec
/bin/zsh -lc "sed -n '760,880p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

    logging.info("Enriching odds features (NFL)...")
    rows_before_odds = df_player_game_all.height
    df_player_game_all = add_nfl_odds_features_to_df(
        df_player_game_all,
        player_col="player_name",
        allow_schedule_fallback=False,
        drop_schedule_rows=True,
    )
    
    logging.info("Enriching player props (Anytime TD)...")
    df_player_game_all = add_player_props_to_df(
        df_player_game_all,
        player_col="player_name",
    )

    rows_after_odds = df_player_game_all.height
    logging.info(
        "Odds features attached; retained %d/%d rows (%.2f%%) with pre-cutoff snapshots",

thinking
**Examining rolling context distinctions**
exec
/bin/zsh -lc "sed -n '880,1120p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        logging.info("Added %d historical share features", len(share_exprs))
    leakage_cols = [
        "target_share",
        "carry_share",
        "pass_attempt_share",
        "red_zone_target_share",
        "red_zone_carry_share",
        "goal_to_go_target_share",
        "goal_to_go_carry_share",
    ]
    df_player_game_all = df_player_game_all.drop(leakage_cols, strict=False)
    # Keep raw outcome columns (targets, carries, etc.) in the feature matrix so they can
    # serve as supervised targets. These columns are excluded from model inputs later via
    # training configuration, so we avoid dropping them here to retain label availability.

    available_cols = set(df_player_game_all.columns)
    market_exprs: list[pl.Expr] = []
    if {"team_implied_total", "hist_target_share_prev"} <= available_cols:
        market_exprs.append(

thinking
**Analyzing data consistency and leakage risks**

I'm examining key conceptual challenges in the rolling window features, especially the mismatch between expected and available stats columns, which may create noise from zero-filled features. I'm also checking for potential data leakage from using caches that include same-day or future game data, ensuring that feature computations properly exclude current or future information during training and inference.

thinking
**Analyzing potential data leakage in feature computation**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
Consolidated NFL Odds Module
============================

This module provides all odds-related functionality for the NFL prediction pipeline:

1. OddsAPIClient - HTTP client for The Odds API with caching and retry logic
2. Game-level odds features (moneyline, spread, totals) from nfl_odds.py
3. Player-level odds features (anytime TD props) from odds_features.py
4. Odds snapshot collection and persistence from odds_snapshots.py

All odds logic is consolidated here to prevent drift and ensure consistency.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

thinking
**Analyzing rolling window and feature pipeline**

I’m digging into the rolling window implementation and its interaction with feature caches and pipelines. There are questions about how date ranges and future game predictions align, especially if the pipeline uses actual play data that wouldn’t exist yet for upcoming games. I want to verify how the prediction pipeline handles schedules and odds versus the completed game data the feature builds rely on. Let’s explore pipeline/predict next!
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/predict.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
NFL prediction pipeline
=======================

Generates anytime-touchdown probabilities for upcoming games by:

1. Loading the upcoming schedule and weekly rosters (via nfl_data_py)
2. Building a player-level scaffold for offensive skill positions
3. Enriching that scaffold with rolling-window features using the same
   caches that power the training pipeline
4. Sequentially running the model chain:
   Availability -> Team Pace -> Usage -> Efficiency -> Anytime TD
5. Writing an actionable slate under `output/predictions/`
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

thinking
**Analyzing feature computation inconsistencies**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_experiments.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 752ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

from typing import Iterable

import polars as pl


def add_experimental_rolling_features(
    df: pl.DataFrame,
    stats: Iterable[str],
    *,
    epsilon: float = 1e-3,
) -> pl.DataFrame:
    """
    Derive experimental rolling-window features to capture burst performance,
    exponential-style recency weighting, and directional trends.

    Parameters
    ----------

thinking
**Checking dataset relevance and usage**
exec
/bin/zsh -lc 'rg "ps_route_participation"' in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
pipeline/predict.py:    "ps_route_participation_plays",
pipeline/predict.py:    "ps_route_participation_pct",
pipeline/predict.py:    "ps_route_participation_pct_prev",
pipeline/predict.py:    "ps_route_participation_pct_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3": "ps_route_participation_plays",
pipeline/predict.py:        "1g_ps_route_participation_pct_per_game": "ps_route_participation_pct",
pipeline/predict.py:        "3g_ps_route_participation_pct_per_game": "ps_route_participation_pct_l3",
pipeline/predict.py:        "1g_ps_route_participation_plays_per_game": "ps_route_participation_plays",
pipeline/predict.py:        "3g_ps_route_participation_plays_per_game": "ps_route_participation_plays_l3",
pipeline/predict.py:        "ps_route_participation_pct",
pipeline/predict.py:        "ps_route_participation_plays",
utils/feature/stats.py:    "ps_route_participation_pct",
utils/feature/stats.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:        .agg(pl.len().alias("ps_route_participation_plays"))
utils/feature/player_game_level.py:            .then(pl.col("ps_route_participation_plays") / pl.col("ps_team_dropbacks"))

thinking
**Investigating player_game_level and anytime_td**
exec
/bin/zsh -lc 'rg "anytime_td" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 777ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- anytime_td (binary): Did player score a TD (receiving or rushing)?
    "market_anytime_td_prob",
    "market_anytime_td_consensus",
    "market_anytime_td_spread",
    "market_anytime_td_book_count",
    "market_anytime_td_prob_2h",
    "market_anytime_td_consensus_2h",
    "market_anytime_td_spread_2h",
    "market_anytime_td_book_count_2h",
    "market_anytime_td_prob_6h",
    "market_anytime_td_consensus_6h",
    "market_anytime_td_spread_6h",
    "market_anytime_td_book_count_6h",
    "market_anytime_td_prob_24h",
    "market_anytime_td_consensus_24h",
    "market_anytime_td_spread_24h",
    "market_anytime_td_book_count_24h",
    "market_anytime_td_prob_open",
    "market_anytime_td_consensus_open",

thinking
**Analyzing potential data leakage and timing issues**

I'm piecing together key concerns about possible data leakage in the rolling window system and training pipeline, especially related to future data inclusion, odds snapshot timing, and injury predictions. I'll highlight complications with caching and as-of joins that may cause leakage or inconsistencies.

thinking
**Evaluating data leakage risks in caches**

I'm thinking about whether cached stats might include data from invalid or future games, potentially causing leakage in training or inference. I need to check if features like offense_context have discrepancies between their training and inference implementations that might lead to mismatches or leaks.
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/offense_context.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 788ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import polars as pl

logger = logging.getLogger(__name__)

COORD_DEFAULT = "UNKNOWN_COORDINATOR"
QB_DEFAULT = "UNKNOWN_QB"
EPS = 1e-3

COORDINATOR_MAP_PATH = Path("data/processed/offensive_coordinators.parquet")


def _build_is_unavailable_expr(df: pl.DataFrame) -> pl.Expr:
    """Return a boolean expression marking players ruled out before cutoff."""

thinking
**Reviewing touchdown definitions for label leakage**
exec
/bin/zsh -lc 'rg -n "touchdowns" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 738ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
803:        pl.lit(0).cast(pl.Int64).alias("touchdowns"),
2577:            "touchdowns",
3108:              .alias("touchdowns"),
3121:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3123:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3165:              .alias("touchdowns"),
3177:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3178:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3220:              .alias("touchdowns"),
3232:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3233:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3271:            pl.col("touchdowns").fill_null(0).sum().alias("touchdowns"),
3284:        "passing_td", "rushing_td_count", "receiving_td_count", "touchdowns",
3293:    # derive it from rushing + receiving touchdowns.
3295:        pl.col("touchdowns").fill_null(0).alias("touchdowns"),
3306:        pl.when(pl.col("touchdowns") > 0)
3307:        .then(pl.col("touchdowns"))
3310:        .alias("_total_touchdowns")
3314:        (pl.col("_total_touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),

thinking
**Inspecting final assembly logic**
exec
/bin/zsh -lc "sed -n '3240,3320p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 771ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    Example: QB who passed and rushed, WR who received and rushed.
    """
    
    # Group by (player, game) and sum all stats
    merged = (
        df
        .group_by(["season", "week", "game_id", "game_date", "player_id", "player_name"])
        .agg([
            # Sum all yardage
            pl.col("passing_yards").fill_null(0).sum().alias("passing_yards"),
            pl.col("rushing_yards").fill_null(0).sum().alias("rushing_yards"),
            pl.col("receiving_yards").fill_null(0).sum().alias("receiving_yards"),
            
            # Sum counts
            pl.col("pass_attempt").fill_null(0).sum().alias("pass_attempt"),
            pl.col("completion").fill_null(0).sum().alias("completion"),
            pl.col("carry").fill_null(0).sum().alias("carry"),
            pl.col("target").fill_null(0).sum().alias("target"),

thinking
**Examining touchdown aggregation logic**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Play-level feature builder.

Reads cleaned daily Parquets and writes weekly play-level Parquets at
  data/processed/play_by_week/season=YYYY/week=W/part.parquet
"""

from pathlib import Path
from datetime import date
import polars as pl

CLEAN_DIR   = Path("data/cleaned")
PLAY_OUTDIR = Path("data/processed/play_by_week")

PLAY_COLS = [
    # Game identifiers
    "game_id",
    "season",

thinking
**Continuing review of derived statistics**
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 758ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
            ),
            (
                pl.col("defense_man_zone_type").cast(pl.Utf8).alias("defense_man_zone_type")
                if "defense_man_zone_type" in df.columns
                else pl.lit(None).alias("defense_man_zone_type")
            ),
            (
                pl.col("defense_coverage_type").cast(pl.Utf8).alias("defense_coverage_type")
                if "defense_coverage_type" in df.columns
                else pl.lit(None).alias("defense_coverage_type")
            ),
        ]
    )

    situ_exprs: list[pl.Expr] = []
    score_diff = pl.col("score_differential").fill_null(0.0)
    ydstogo = pl.col("ydstogo").fill_null(0.0)
    yardline = pl.col("yardline_100").cast(pl.Float32).fill_null(100.0)
    half_secs_expr = (

thinking
**Examining touchdown counting and rolling window concerns**

thinking
**Reviewing data shifts and plan progress**
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  → Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering details on config and training**
exec
/bin/zsh -lc "sed -n '1,200p' config/training.yaml" in /Users/jameswest/Desktop/nfl_predict succeeded in 815ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# config/training.yaml
# Configuration for the model training pipeline (pipeline/train.py)
# NFL Player Prediction Targets

# --- Data I/O ---
data:
  raw_pbp_dir: 'data/raw/pbp_by_day'  # NFL play-by-play raw data
  feature_matrix_path: 'data/processed/final/processed.parquet'  # NFL feature matrix
  
  # Directory to save trained model artifacts (models and feature lists).
  model_output_dir: 'output/models'
  
  # Directory to save evaluation reports and metrics.
  metrics_output_dir: 'output/metrics/'

# =============================================================================
# SHARED COLUMN DEFINITIONS (using YAML anchors to avoid duplication)
# =============================================================================
# These columns leak same-game outcomes and must be excluded from training.

thinking
**Analyzing potential data inconsistencies and leakage**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/asof.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 2.75s:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Helpers for aligning feature generation to a consistent decision-time snapshot."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Mapping

import yaml

from utils.general.paths import PROJ_ROOT


@lru_cache(maxsize=1)
def _load_cutoff_config() -> Mapping[str, Any]:
    """Load decision cutoff configuration from config/config.yaml."""
    config_path = PROJ_ROOT / "config" / "config.yaml"
    try:

thinking
**Identifying discrepancies in feature construction**

I'm analyzing several potential issues in how features are constructed, especially around rolling window computations, drive-level feature usage, pre-snap route participation data reliability, and labeling choices. I want to assemble a clear list of 6-7 concrete findings summarizing inconsistencies, incomplete implementations, and data limitations.

thinking
**Analyzing potential data leakage and pipeline inconsistencies**

I'm exploring possible data leakage issues, especially around using full-season roster and injury data without proper as-of gating in predictions. I’m also checking whether training pipelines enforce the claimed multi-stage model architecture and if various feature-building steps apply correct cutoff filters, since odds features might not respect decision cutoff timestamps, risking leakage. This involves reviewing training, feature, and monitoring docs alongside function implementations to understand the conceptual gaps and inconsistencies.

thinking
**Searching for add_player_odds_features_to_df function**
exec
/bin/zsh -lc 'rg "def add_player" -n utils/feature/odds.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 926ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
1099:def add_player_odds_features_to_df(
exec
/bin/zsh -lc "sed -n '1099,1250p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 732ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
def add_player_odds_features_to_df(
    df_pl: pl.DataFrame, *, api_key: str | None = None, player_col: str = "player_name",
    per_game_workers: int = 4, verbose: bool = False,
) -> pl.DataFrame:
    """Attach player odds features directly to an in-memory Polars DataFrame."""
    from utils.general.config import get_odds_api_key
    
    if df_pl.is_empty():
        return df_pl
    
    if not api_key:
        api_key = get_odds_api_key()
    
    if not api_key:
        logger.info("add_player_odds_features_to_df: no odds_api_key configured – adding empty columns and returning")
        empty_exprs = [pl.lit(None).cast(pl.Float32).alias(c) for c in PLAYER_ODDS_NUMERIC_COLS]
        flag_expr = [pl.lit(0).cast(pl.Int8).alias("odds_expected")]
        df_base = df_pl.drop(PLAYER_ODDS_NUMERIC_COLS + ODDS_FLAG_COLUMNS, strict=False)
        return df_base.with_columns(empty_exprs + flag_expr)

thinking
**Analyzing data leakage and feature completeness**

thinking
**Reviewing data leakage and caching inconsistencies**

I'm noticing potential data leakage in feature functions using live API calls and uncertain odds snapshot timing. There's also a mismatch between documented and actual cache directories that could confuse or cause bugs. I want to gather all issues before moving from feature pipeline review to modeling validation, paying attention to training config settings and columns flagged as leak-prone.
exec
/bin/zsh -lc 'rg "LEAK_PRONE_COLUMNS" -n utils' in /Users/jameswest/Desktop/nfl_predict succeeded in 870ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
utils/general/__init__.py:18:    LEAK_PRONE_COLUMNS,
utils/general/__init__.py:47:    "LEAK_PRONE_COLUMNS",
utils/general/constants.py:8:config/training.yaml should mirror LEAK_PRONE_COLUMNS for YAML-based configuration,
utils/general/constants.py:113:LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
utils/general/constants.py:208:    "LEAK_PRONE_COLUMNS",
exec
/bin/zsh -lc "sed -n '100,160p' utils/general/constants.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 800ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
}


# =============================================================================
# Leak-Prone Columns (Same-Game Outcomes)
# =============================================================================
# Stats that leak same-game outcomes and must be excluded from training features.
# These represent what happened in the game we're trying to predict.
#
# This is the AUTHORITATIVE definition. config/training.yaml mirrors this list
# using YAML anchors for per-problem configuration. Keep them in sync!
# Validated by: tests/test_training_config_features.py

LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
    # Yardage (same-game outcomes)
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    # Usage counts (same-game outcomes)

thinking
**Analyzing modeling pipeline inconsistencies**

I’m reviewing the training pipeline for possible incomplete or risky implementations—especially around derived targets, odds features that may leak future info, rolling window system mismatches, and caching issues causing potential feature inaccuracies. I'll note risky spots with specifics to file paths and behaviors to highlight data leakage or half-finished components.

thinking
**Identifying potential data leakage in features and inference**

thinking
**Analyzing touchdown column naming mismatch**

thinking
**Analyzing touchdown feature inconsistencies**

I notice a mismatch between "touchdown" and "touchdowns" columns in different dataframes, causing rolling touchdown features to be always zero—this looks like a conceptual bug. Also, the way stats are counted might cause double counting or misalignment. I want to fix these naming and aggregation issues and then update the project plan to mark step 2 as done and step 3 as in progress.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  → Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Analyzing model training and feature inconsistencies**

thinking
**Evaluating data quality and documentation gaps**

I'm analyzing issues with sparse or missing data affecting model training, especially around pre-snap route participation features and roster/injury alignment leaking future info. I also notice documentation inconsistencies in rolling window implementation and potential feature instability due to network-dependent API calls, all of which could harm predictive reliability and require careful handling in the next steps.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  ✓ Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Preparing final summary and suggestions**

I'll create a clear bullet list of findings ordered by severity with approximate file paths and lines, highlighting key issues like touchdown rolling features and pre-snap usage, then follow with numbered next steps for the user to consider.
codex
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.
tokens used
110,897
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.

    <END_ANALYSIS_END>

    This step is PURELY research, investigation, and planning.
    Do NOT make any edits to the code or data in this step.

    I want:
    - A structured, prioritized plan.
    - Clear steps that can be implemented by an agent in later steps.
    - Notes on risk or potential pitfalls where relevant, but focus on high-value changes.
    - Explicit attention to:
        * eliminating data leakage,
        * aligning features with real football mechanisms that drive anytime TDs,
        * cleaning up hallucinated / legacy cruft that no longer makes sense.

    Again: no edits here, just the plan.

mcp startup: no servers

thinking
**Drafting a structured planning outline**
codex
**Remediation Plan (ordered by impact & leakage risk)**
- Fix odds leakage/hallucination: freeze player prop odds to archived pre-cutoff snapshots; add snapshot loader keyed by game_id/team/player + snapshot_ts; in training, drop/flag rows without snapshots; in inference, refuse to call live API, optionally fall back to schedule-only baseline; add tests to assert no odds columns present when snapshot missing.
- Repair rolling TD history: align stat key `touchdowns` across `stats.py`, aggregation, and cache; rebuild daily_totals cache; add schema check to fail if any rolling TD feature is all-null/zero; rerun backtests to measure regained signal.
- Trim rolling window surface to supported set: codify a vetted set of stats/windows/contexts (e.g., {1g,3g,5g,season} × {vs_any}); remove ps_* and other missing-cache stats from rolling requests; update docs to match; add runtime guard to error on requested stats not found in cache.
- Enforce as-of gating for roster/injury inputs in prediction/backtests: add decision-time cutoff (e.g., T-6h) and snapshot source; if unavailable, drop those features or mark unavailable; add regression test ensuring no injury status is from after game start in historical inference.
- Remove/retire ps_* targets/features if coverage is low: measure non-null rate by season; if sparse, drop from training configs and rolling stats; otherwise backfill realistically or scope to seasons with data; add check to block tasks with <X% coverage.
- Clarify/implement drive-level history or delete: either build drive-level daily cache and wire into rolling computation, or remove drive-context flags/constants/docs; add assertion preventing “drive” level when cache absent.
- Align docs with reality and guard drift: sync paths/names (cache/feature/daily_totals vs nfl_daily_totals); document the final rolling feature list and cutoff rules; add a pre-flight validation script that checks cache schemas, non-null rates, and leakage-sensitive columns before training.

**Execution Steps**
1) Odds snapshots: design schema + loader; replace `add_player_props_to_df` call; add fallback behavior and tests; rerun feature build on a small date range to validate columns.
2) Rolling TD fix: rename stat key, rebuild cache for a sample window, ensure rolling TD features populate; update tests.
3) Rolling feature pruning: define allowed stats/windows/contexts; update config/constants + pipeline call; add guard for missing stats; rerun feature build and inspect null/constant rates.
4) Roster/injury gating: add cutoff param in predict/backtest; wire to snapshot source; add assertion that injury_status_ts <= cutoff; adjust backtests to use gated features.
5) ps_* triage: compute coverage by season; decide keep/drop; update training.yaml and problem definitions; if dropped, remove from rolling stats and tasks.
6) Drive-level decision: either implement cache builder (drive totals) and wire into rolling_window or delete related flags/docs; add guard to prevent silent no-op.
7) Docs/validation: update rolling_window_implementation.md and monitoring docs; add validation script to check schemas, coverage, leak-prone columns excluded from train set; run it in CI/pre-flight.

**Risks/Pitfalls to watch**
- Cache rebuild time/size; may need incremental rebuild and date filters.
- Snapshot availability gaps; must define strict fallbacks to avoid silent leakage.
- Training/inference drift if odds/roster data are absent—prefer hard failure over silent nulls.
- Removing ps_* may change model interfaces; update downstream consumers/tests accordingly.
tokens used
15,743
**Remediation Plan (ordered by impact & leakage risk)**
- Fix odds leakage/hallucination: freeze player prop odds to archived pre-cutoff snapshots; add snapshot loader keyed by game_id/team/player + snapshot_ts; in training, drop/flag rows without snapshots; in inference, refuse to call live API, optionally fall back to schedule-only baseline; add tests to assert no odds columns present when snapshot missing.
- Repair rolling TD history: align stat key `touchdowns` across `stats.py`, aggregation, and cache; rebuild daily_totals cache; add schema check to fail if any rolling TD feature is all-null/zero; rerun backtests to measure regained signal.
- Trim rolling window surface to supported set: codify a vetted set of stats/windows/contexts (e.g., {1g,3g,5g,season} × {vs_any}); remove ps_* and other missing-cache stats from rolling requests; update docs to match; add runtime guard to error on requested stats not found in cache.
- Enforce as-of gating for roster/injury inputs in prediction/backtests: add decision-time cutoff (e.g., T-6h) and snapshot source; if unavailable, drop those features or mark unavailable; add regression test ensuring no injury status is from after game start in historical inference.
- Remove/retire ps_* targets/features if coverage is low: measure non-null rate by season; if sparse, drop from training configs and rolling stats; otherwise backfill realistically or scope to seasons with data; add check to block tasks with <X% coverage.
- Clarify/implement drive-level history or delete: either build drive-level daily cache and wire into rolling computation, or remove drive-context flags/constants/docs; add assertion preventing “drive” level when cache absent.
- Align docs with reality and guard drift: sync paths/names (cache/feature/daily_totals vs nfl_daily_totals); document the final rolling feature list and cutoff rules; add a pre-flight validation script that checks cache schemas, non-null rates, and leakage-sensitive columns before training.

**Execution Steps**
1) Odds snapshots: design schema + loader; replace `add_player_props_to_df` call; add fallback behavior and tests; rerun feature build on a small date range to validate columns.
2) Rolling TD fix: rename stat key, rebuild cache for a sample window, ensure rolling TD features populate; update tests.
3) Rolling feature pruning: define allowed stats/windows/contexts; update config/constants + pipeline call; add guard for missing stats; rerun feature build and inspect null/constant rates.
4) Roster/injury gating: add cutoff param in predict/backtest; wire to snapshot source; add assertion that injury_status_ts <= cutoff; adjust backtests to use gated features.
5) ps_* triage: compute coverage by season; decide keep/drop; update training.yaml and problem definitions; if dropped, remove from rolling stats and tasks.
6) Drive-level decision: either implement cache builder (drive totals) and wire into rolling_window or delete related flags/docs; add guard to prevent silent no-op.
7) Docs/validation: update rolling_window_implementation.md and monitoring docs; add validation script to check schemas, coverage, leak-prone columns excluded from train set; run it in CI/pre-flight.

**Risks/Pitfalls to watch**
- Cache rebuild time/size; may need incremental rebuild and date filters.
- Snapshot availability gaps; must define strict fallbacks to avoid silent leakage.
- Training/inference drift if odds/roster data are absent—prefer hard failure over silent nulls.
- Removing ps_* may change model interfaces; update downstream consumers/tests accordingly.

    <END_PLAN_END>

    Repository + Git requirements:

    - This repo is under git. At the very beginning of THIS RUN, before editing any files:
        1) Run `git status` to inspect the current state.
        2) If there are uncommitted changes from previous runs, stage and commit them with a concise
           message like `codex: iteration checkpoint` (or a slightly more descriptive variant).
        3) If a remote named `origin` exists and authentication allows, run `git push` so that the
           current state is saved remotely. If push fails due to auth or remote issues, continue with
           local commits only, but do NOT delete history.
        4) Only after ensuring there is a clean commit of the current state should you begin making
           new edits in this run.

    - During this run:
        * Make coherent, incremental commits as you reach logical checkpoints.
        * If you realize that your changes have badly broken the project and you cannot fix them
          cleanly within this run, you may revert to the last good commit (for example, using
          `git reset --hard HEAD` or `git checkout .`), then proceed more conservatively.

    Your tools allow you to:
    - Inspect files in this repo.
    - Edit files.
    - Run shell commands/tests as needed (e.g. project-specific tests, evaluations, or scripts).
    - Use git commands to create commits and, if possible, push them.

    Your goal in THIS RUN:
    - Implement as much of the plan as you reasonably can with high quality.
    - Prioritize correctness, alignment with football reality, and improved model performance
      over speed.
    - Run whatever tests or checks are appropriate to validate your changes.

    At the end of THIS RUN:
    - Give a concise update on your progress.
    - If the ENTIRE plan is fully implemented and validated, include a line with EXACTLY:
        The plan is completely implemented
      (case and spacing exactly as written).
    - If the plan is NOT fully implemented, include a concise progress summary starting with:
        PROGRESS:
      followed by a short description of what you accomplished and what remains.

    Take your time and implement the richest / most complete solution for the pieces you touch in this run.


======================================================


================ CODEX OUTPUT (streaming) ================

OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf6b-83fe-7890-bc9d-faed2b5c79ed
--------
user
Okay, great. Please work like an agent and implement the following plan in this repository.

    Here is the plan:

    <BEGIN_PLAN_START>
    OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf6b-4ad5-78e1-9036-ed3c08d53d2c
--------
user

    Okay, great work.

    Please come up with the richest and most detailed plan possible to address every one of the points
    you brought up in the best way possible.

    Here is your last analysis / set of findings:

    <BEGIN_ANALYSIS_START>
    OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf67-266d-7c53-b835-f4d18d9aac20
--------
user

Context about this project:

- This is a project that takes NFL play-by-play data and uses it for modeling so that we can
  predict which players are likely to get an anytime touchdown (rushing or receiving, not passing)
  in an upcoming game.

- The goal is to build features and models that are as close as possible to the underlying
  football reality that produces touchdowns: play-calling tendencies, player usage, game state,
  defensive matchups, injuries, roles, red-zone behavior, etc.

- Training featurization must conceptually and logically match prediction/inference featurization.
  Anything that can only be known in hindsight at inference time (future data, downstream labels,
  or derived artifacts that use future information) is a form of leakage and must be eliminated.

- Over time, previous model runs and refactors may have left behind:
    * partially-implemented ideas,
    * experimental code paths,
    * hallucinated features,
    * or confusing / inconsistent logic.
  DO NOT assume that all existing code, features, configs, or comments are intentional or correct
  just because they exist. Treat any piece of code or configuration that does not clearly make
  sense in the context of the project as a candidate for cleanup, simplification, or removal.

Your task in this step:

Please analyze the current state of this project (code, data flow, feature engineering, and modeling)
and let me know:

1. Where things conceptually are not implemented correctly or are conceptually off, given the goal of
   predicting anytime TDs in a way that matches how football is actually played.
2. Where the modeling or data flow could be brought closer to "reality" as it actually plays out
  on the field. The goal of getting closer to reality is entirely so that the model is more
  accurate and metrics like AUC or other evaluation metrics improve.
3. Any incomplete implementations, half-finished ideas, or abandoned experimental paths.
4. Any wrong or misleading implementations (especially where names / comments and actual behavior diverge).
5. Any future data leaking into the modeling or feature pipeline (anything that uses knowledge from
  after the prediction cut-off point, including label-derived features).
6. Any underlying data sources or features that appear to not be working at all, or are effectively
  noise / dead weight.
7. Any areas where it looks like a previous run of a model or tool hallucinated structure, concepts,
  or features that don't actually exist in the real data or problem domain.

You should:

- Be concrete and specific in your findings.
- Call out anything that looks like hallucinated or legacy cruft that should probably be removed or
  reworked, instead of assuming it must be intentional.
- Focus on how each issue you find ultimately affects model realism and predictive performance.

mcp startup: no servers

thinking
**Listing project files for analysis**
exec
/bin/zsh -lc ls in /Users/jameswest/Desktop/nfl_predict succeeded in 765ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
README.md
__pycache__
audit
cache
codex_meta_loop.py
config
data
docs
logs
main.py
output
pipeline
pyproject.toml
requirements.txt
scripts
tests
tmp.parquet
tmp_features.parquet
utils

thinking
**Considering reading README**
exec
/bin/zsh -lc "sed -n '1,200p' README.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 762ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Anytime TD Prediction

## Key Scripts
- `pipeline/feature.py` – builds the player-game feature matrix, enforcing as-of leakage guards.
- `pipeline/train.py` – trains the multi-stage modeling graph (availability → usage → efficiency → TD).
- `pipeline/predict.py` – runs inference with the latest artifacts.
- `scripts/eval/odds_backtest.py` – compute odds-only baselines across horizons.
- `scripts/backtest/run_backtest.py` – evaluate existing prediction columns (AUC/Brier/log-loss) and write baseline summaries.
- `scripts/backtest/compare_baselines.py` – diff two baseline JSON files and highlight regressions.
- `scripts/analysis/horizon_eval.py` – consolidate metrics across decision horizons (input multiple prediction files labelled by horizon).
- `scripts/analysis/weather_travel_impact.py` – stratify TD hit rates by weather/travel features and emit plots (`docs/analysis/weather_travel/`) + JSON summaries.
- `scripts/monitoring/asof_coverage_report.py` – summarize snapshot coverage, write Markdown to `docs/monitoring/asof_coverage.md`, and append metrics history to `output/metrics/coverage/history.parquet`.

## Monitoring
- `docs/monitoring.md` documents leak guards, snapshot coverage expectations, roster snapshot regeneration steps, and instructions for running odds/backtests & analytics scripts.
- `docs/analysis/weather_travel.md` lists the latest weather/travel diagnostic artifacts.

## Quick Start
1. Build features:

thinking
**Preparing multi-step plan**
Plan update
Map out review steps to assess feature pipeline, modeling flow, leakage risks, and dead code.
  → Scan project structure and key config/docs to understand intended data flow and leakage guards
  • Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering key module info**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Feature pipeline orchestrator for NFL player-level predictions.

Stages
------
1. **build_play_level**         – derive cleaned play-level slices
2. **build_player_drive_level** – aggregate per drive
3. **build_player_game_level**  – aggregate per game and join context

This wrapper triggers each stage, optionally enforces a schema after each
write, and returns the final game-level DataFrame for downstream usage.
"""

import logging
import os
from datetime import date, timedelta
from typing import Sequence


thinking
**Planning deeper analysis**
exec
/bin/zsh -lc 'rg "TODO|FIXME|TBD"' in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
docs/rolling_window_implementation.md:### Immediate (TODO #8)

thinking
**Checking rolling window docs**
exec
/bin/zsh -lc "sed -n '1,200p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Rolling Window Feature System - Implementation Summary

## Overview

Successfully implemented a comprehensive rolling window feature system for NFL player predictions. The system computes time-series statistics for player performance across different contexts and time horizons.

## Architecture

### 1. Data Flow

```
Cleaned Play Data
    ↓
Player-Game Aggregations (player_game_level.py)
    ↓
Daily Totals Cache (nfl_daily_totals.py)
    ↓
Rolling Window Features (nfl_rolling_window.py)
    ↓
exec
/bin/zsh -lc "sed -n '200,400p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- **Incremental updates**: Only new dates need processing
- **Fast lookups**: Direct parquet reads by date
- **Memory efficient**: Streaming operations for large datasets
- **Reusable**: Same cache serves all rolling window computations

### Scalability
- **Per-player parallelization**: Each player's features computed independently
- **Lazy evaluation**: Uses Polars lazy API where possible
- **Chunked processing**: Pipeline processes data in configurable chunks

## Next Steps

### Immediate (TODO #8)
1. **Run feature pipeline** with new stats and rolling windows
   ```bash
   python -c "from pipeline.feature import build_feature_matrix; \
              from datetime import date; \
              build_feature_matrix(start_date=date(2024, 9, 5), \
                                  end_date=date(2024, 9, 9))"

thinking
**Verifying documentation accuracy**
exec
/bin/zsh -lc 'ls utils/feature' in /Users/jameswest/Desktop/nfl_predict succeeded in 746ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
__init__.py
__pycache__
asof.py
asof_metadata.py
concatenate_parquets.py
daily_totals.py
drive_level.py
game_level.py
io.py
keygen.py
odds.py
offense_context.py
opponent_splits.py
pace.py
play_level.py
player_drive_level.py
player_game_level.py
player_market_cli.py
qb_profiles.py

thinking
**Inspecting statistics components**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/stats.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 719ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL player statistics configuration for rolling window features.

Defines the standard set of stats tracked at all aggregation levels
(play, drive, game) for rolling window computations.

This module is the SINGLE SOURCE OF TRUTH for stat definitions used
throughout the pipeline. Import from here rather than defining locally.
"""

from typing import List

# =============================================================================
# Core Stats (used for drive and game level aggregation)
# =============================================================================

# Basic stats aggregated at player-drive level
PLAYER_DRIVE_STATS: List[str] = [
    # Yardage
    "receiving_yards",

thinking
**Verifying feature data existence**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 739ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL Rolling Window Features - Direct port from MLB rolling_window.py

Uses the exact same vectorized Polars pattern as the working MLB implementation.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache/feature/daily_totals")
NFL_SEASON_START_MMDD = (9, 1)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                    rolling_feature_names.append(f"{w}g_{k}_per_{self.level}{suffix}")

        if batch_min_date is None or batch_max_date is None:
            logger.warning("RollingWindow.compute received empty dataset; returning null rolling features.")
            result = ds
            for feat in rolling_feature_names:
                result = result.with_columns(pl.lit(None).cast(pl.Float32).alias(feat))
            return result.drop("__ts")
        
        # Determine lookback for CACHE scan
        # Load cache INCLUDING the batch dates - cumsum logic prevents leakage
        # Each row's rolling features = cumsum(up to row) - row_value = history BEFORE row
        if "lifetime" in windows:
            earliest = self._earliest_cache_date()
            lo_hist = earliest if earliest else batch_min_date - timedelta(days=365 * 10)
        elif "season" in windows:
            season_start = self._season_anchor(batch_min_date)
            lo_hist = season_start if season_start < batch_min_date else batch_min_date - timedelta(days=365)
        else:
exec
/bin/zsh -lc "sed -n '400,800p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                            denom_use = den_window_cache.get(w_int)
                            if denom_use is None:
                                denom_use = np.zeros(n, dtype=np.float64)
                            if w_int <= n:
                                shifted_num = np.concatenate((np.zeros(w_int, dtype=np.float64), prev_num[:-w_int]))
                            else:
                                shifted_num = np.zeros(n, dtype=np.float64)
                            numer_use = prev_num - shifted_num
                        
                        values = np.full(n, np.nan, dtype=np.float32)
                        if denom_use is not None:
                            mask = denom_use > 0
                            if mask.any():
                                values[mask] = (numer_use[mask] / denom_use[mask]).astype(np.float32)
                        
                        feat_name = f"{w}g_{k}_per_{self.level}" + ("" if ctx == "vs_any" else f"_{ctx}")
                        feature_arrays[feat_name] = values
                
                data: dict[str, pl.Series | np.ndarray] = {

thinking
**Checking daily_totals module**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 747ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL daily totals cache for rolling window computations.

Builds daily aggregated stats per player-context combination:
- vs_any: All games for this player
- vs_team: Games vs specific opponent

Cache structure per day:
  player_id, date, ctx, opponent → stat_num columns
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Callable
import polars as pl
import logging

from utils.feature.stats import NFL_PLAYER_STATS

logger = logging.getLogger(__name__)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 734ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    # Load function: loads player data for a given date
    def load_player_data(day: date) -> pl.DataFrame:
        """Load player-level data for a specific date from weekly parquets."""
        try:
            # Scan all weekly parquets
            df = pl.scan_parquet(
                str(source_dir / "season=*/week=*/part.parquet"),
                glob=True,
                hive_partitioning=True,
                missing_columns="insert",
                extra_columns="ignore",
            )

            schema = df.collect_schema()
            categorical_cols = [
                name
                for name, dtype in zip(schema.names(), schema.dtypes())
                if "Categorical" in str(dtype)

thinking
**Inspecting training pipeline**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/train.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
import os
import random
import inspect
import sys
import logging
import math


# Move logging to top and remove global thread pinning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path to allow module imports when running as script
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import yaml
import joblib
import optuna

thinking
**Outlining investigation strategy**
exec
/bin/zsh -lc 'rg "build_feature_matrix" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
66:__all__ = ["build_feature_matrix", "refresh_context_histories"]
226:def _build_feature_matrix_internal(
953:def build_feature_matrix(
979:            df = _build_feature_matrix_internal(
exec
/bin/zsh -lc "sed -n '220,520p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 698ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    )

# ---------------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------------

def _build_feature_matrix_internal(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    chunk_days: int = 7,
    recompute_intermediate: bool = True,
    build_rolling: bool = True,
    output_path: Path | None = None,
    primary_output_path: Path | None = None,
    cutoff_label: str | None = None,
) -> pl.DataFrame:
    """Run full feature pipeline up to *inclusive* `end_date` for a single cutoff window.


thinking
**Analyzing key feature sections**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/targets.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 743ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Target column validation for NFL player prediction models.

Target columns are created in utils/feature/player_game_level.py during aggregation.
This module validates that required targets exist and have correct dtypes.
"""

import polars as pl
import logging

logger = logging.getLogger(__name__)


# Expected target columns and their types
NFL_TARGET_SCHEMA = {
    # Binary classification targets
    "anytime_td": pl.Int8,  # Primary MVP target: did player score a TD?
    
    # Count/regression targets
    "td_count": pl.Int64,  # Number of TDs scored (receiving + rushing)

thinking
**Checking relevant player game data**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Player-game level aggregation for NFL predictions.

Aggregates play-by-play data to player-game level for all skill positions.
Each player gets one row per game with their total stats.

Target columns created:
- anytime_td (binary): Did player score a TD (receiving or rushing)?
- td_count (int): Number of TDs scored (receiving + rushing)
- passing_td (int): Number of passing TDs (for QBs)
- receiving_yards (float): Total receiving yards
- rushing_yards (float): Total rushing yards
- passing_yards (float): Total passing yards
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import datetime as dt

thinking
**Analyzing potential data issues and feature integration**

I'm focusing on spotting key issues like data leakage or unproduced columns in the large player_game_level file by reviewing mentions of rolling_window, MLB patterns, and stats usage around add_rolling_features. I want to understand if features rely on missing caches or incorrect columns that might disrupt aggregation or inference.
exec
/bin/zsh -lc 'rg "add_rolling_features" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 753ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
31:from utils.feature.rolling_window import add_rolling_features
831:        df_player_game_all = add_rolling_features(
exec
/bin/zsh -lc "sed -n '760,880p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

    logging.info("Enriching odds features (NFL)...")
    rows_before_odds = df_player_game_all.height
    df_player_game_all = add_nfl_odds_features_to_df(
        df_player_game_all,
        player_col="player_name",
        allow_schedule_fallback=False,
        drop_schedule_rows=True,
    )
    
    logging.info("Enriching player props (Anytime TD)...")
    df_player_game_all = add_player_props_to_df(
        df_player_game_all,
        player_col="player_name",
    )

    rows_after_odds = df_player_game_all.height
    logging.info(
        "Odds features attached; retained %d/%d rows (%.2f%%) with pre-cutoff snapshots",

thinking
**Examining rolling context distinctions**
exec
/bin/zsh -lc "sed -n '880,1120p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        logging.info("Added %d historical share features", len(share_exprs))
    leakage_cols = [
        "target_share",
        "carry_share",
        "pass_attempt_share",
        "red_zone_target_share",
        "red_zone_carry_share",
        "goal_to_go_target_share",
        "goal_to_go_carry_share",
    ]
    df_player_game_all = df_player_game_all.drop(leakage_cols, strict=False)
    # Keep raw outcome columns (targets, carries, etc.) in the feature matrix so they can
    # serve as supervised targets. These columns are excluded from model inputs later via
    # training configuration, so we avoid dropping them here to retain label availability.

    available_cols = set(df_player_game_all.columns)
    market_exprs: list[pl.Expr] = []
    if {"team_implied_total", "hist_target_share_prev"} <= available_cols:
        market_exprs.append(

thinking
**Analyzing data consistency and leakage risks**

I'm examining key conceptual challenges in the rolling window features, especially the mismatch between expected and available stats columns, which may create noise from zero-filled features. I'm also checking for potential data leakage from using caches that include same-day or future game data, ensuring that feature computations properly exclude current or future information during training and inference.

thinking
**Analyzing potential data leakage in feature computation**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
Consolidated NFL Odds Module
============================

This module provides all odds-related functionality for the NFL prediction pipeline:

1. OddsAPIClient - HTTP client for The Odds API with caching and retry logic
2. Game-level odds features (moneyline, spread, totals) from nfl_odds.py
3. Player-level odds features (anytime TD props) from odds_features.py
4. Odds snapshot collection and persistence from odds_snapshots.py

All odds logic is consolidated here to prevent drift and ensure consistency.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

thinking
**Analyzing rolling window and feature pipeline**

I’m digging into the rolling window implementation and its interaction with feature caches and pipelines. There are questions about how date ranges and future game predictions align, especially if the pipeline uses actual play data that wouldn’t exist yet for upcoming games. I want to verify how the prediction pipeline handles schedules and odds versus the completed game data the feature builds rely on. Let’s explore pipeline/predict next!
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/predict.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
NFL prediction pipeline
=======================

Generates anytime-touchdown probabilities for upcoming games by:

1. Loading the upcoming schedule and weekly rosters (via nfl_data_py)
2. Building a player-level scaffold for offensive skill positions
3. Enriching that scaffold with rolling-window features using the same
   caches that power the training pipeline
4. Sequentially running the model chain:
   Availability -> Team Pace -> Usage -> Efficiency -> Anytime TD
5. Writing an actionable slate under `output/predictions/`
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

thinking
**Analyzing feature computation inconsistencies**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_experiments.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 752ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

from typing import Iterable

import polars as pl


def add_experimental_rolling_features(
    df: pl.DataFrame,
    stats: Iterable[str],
    *,
    epsilon: float = 1e-3,
) -> pl.DataFrame:
    """
    Derive experimental rolling-window features to capture burst performance,
    exponential-style recency weighting, and directional trends.

    Parameters
    ----------

thinking
**Checking dataset relevance and usage**
exec
/bin/zsh -lc 'rg "ps_route_participation"' in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
pipeline/predict.py:    "ps_route_participation_plays",
pipeline/predict.py:    "ps_route_participation_pct",
pipeline/predict.py:    "ps_route_participation_pct_prev",
pipeline/predict.py:    "ps_route_participation_pct_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3": "ps_route_participation_plays",
pipeline/predict.py:        "1g_ps_route_participation_pct_per_game": "ps_route_participation_pct",
pipeline/predict.py:        "3g_ps_route_participation_pct_per_game": "ps_route_participation_pct_l3",
pipeline/predict.py:        "1g_ps_route_participation_plays_per_game": "ps_route_participation_plays",
pipeline/predict.py:        "3g_ps_route_participation_plays_per_game": "ps_route_participation_plays_l3",
pipeline/predict.py:        "ps_route_participation_pct",
pipeline/predict.py:        "ps_route_participation_plays",
utils/feature/stats.py:    "ps_route_participation_pct",
utils/feature/stats.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:        .agg(pl.len().alias("ps_route_participation_plays"))
utils/feature/player_game_level.py:            .then(pl.col("ps_route_participation_plays") / pl.col("ps_team_dropbacks"))

thinking
**Investigating player_game_level and anytime_td**
exec
/bin/zsh -lc 'rg "anytime_td" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 777ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- anytime_td (binary): Did player score a TD (receiving or rushing)?
    "market_anytime_td_prob",
    "market_anytime_td_consensus",
    "market_anytime_td_spread",
    "market_anytime_td_book_count",
    "market_anytime_td_prob_2h",
    "market_anytime_td_consensus_2h",
    "market_anytime_td_spread_2h",
    "market_anytime_td_book_count_2h",
    "market_anytime_td_prob_6h",
    "market_anytime_td_consensus_6h",
    "market_anytime_td_spread_6h",
    "market_anytime_td_book_count_6h",
    "market_anytime_td_prob_24h",
    "market_anytime_td_consensus_24h",
    "market_anytime_td_spread_24h",
    "market_anytime_td_book_count_24h",
    "market_anytime_td_prob_open",
    "market_anytime_td_consensus_open",

thinking
**Analyzing potential data leakage and timing issues**

I'm piecing together key concerns about possible data leakage in the rolling window system and training pipeline, especially related to future data inclusion, odds snapshot timing, and injury predictions. I'll highlight complications with caching and as-of joins that may cause leakage or inconsistencies.

thinking
**Evaluating data leakage risks in caches**

I'm thinking about whether cached stats might include data from invalid or future games, potentially causing leakage in training or inference. I need to check if features like offense_context have discrepancies between their training and inference implementations that might lead to mismatches or leaks.
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/offense_context.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 788ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import polars as pl

logger = logging.getLogger(__name__)

COORD_DEFAULT = "UNKNOWN_COORDINATOR"
QB_DEFAULT = "UNKNOWN_QB"
EPS = 1e-3

COORDINATOR_MAP_PATH = Path("data/processed/offensive_coordinators.parquet")


def _build_is_unavailable_expr(df: pl.DataFrame) -> pl.Expr:
    """Return a boolean expression marking players ruled out before cutoff."""

thinking
**Reviewing touchdown definitions for label leakage**
exec
/bin/zsh -lc 'rg -n "touchdowns" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 738ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
803:        pl.lit(0).cast(pl.Int64).alias("touchdowns"),
2577:            "touchdowns",
3108:              .alias("touchdowns"),
3121:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3123:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3165:              .alias("touchdowns"),
3177:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3178:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3220:              .alias("touchdowns"),
3232:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3233:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3271:            pl.col("touchdowns").fill_null(0).sum().alias("touchdowns"),
3284:        "passing_td", "rushing_td_count", "receiving_td_count", "touchdowns",
3293:    # derive it from rushing + receiving touchdowns.
3295:        pl.col("touchdowns").fill_null(0).alias("touchdowns"),
3306:        pl.when(pl.col("touchdowns") > 0)
3307:        .then(pl.col("touchdowns"))
3310:        .alias("_total_touchdowns")
3314:        (pl.col("_total_touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),

thinking
**Inspecting final assembly logic**
exec
/bin/zsh -lc "sed -n '3240,3320p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 771ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    Example: QB who passed and rushed, WR who received and rushed.
    """
    
    # Group by (player, game) and sum all stats
    merged = (
        df
        .group_by(["season", "week", "game_id", "game_date", "player_id", "player_name"])
        .agg([
            # Sum all yardage
            pl.col("passing_yards").fill_null(0).sum().alias("passing_yards"),
            pl.col("rushing_yards").fill_null(0).sum().alias("rushing_yards"),
            pl.col("receiving_yards").fill_null(0).sum().alias("receiving_yards"),
            
            # Sum counts
            pl.col("pass_attempt").fill_null(0).sum().alias("pass_attempt"),
            pl.col("completion").fill_null(0).sum().alias("completion"),
            pl.col("carry").fill_null(0).sum().alias("carry"),
            pl.col("target").fill_null(0).sum().alias("target"),

thinking
**Examining touchdown aggregation logic**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Play-level feature builder.

Reads cleaned daily Parquets and writes weekly play-level Parquets at
  data/processed/play_by_week/season=YYYY/week=W/part.parquet
"""

from pathlib import Path
from datetime import date
import polars as pl

CLEAN_DIR   = Path("data/cleaned")
PLAY_OUTDIR = Path("data/processed/play_by_week")

PLAY_COLS = [
    # Game identifiers
    "game_id",
    "season",

thinking
**Continuing review of derived statistics**
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 758ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
            ),
            (
                pl.col("defense_man_zone_type").cast(pl.Utf8).alias("defense_man_zone_type")
                if "defense_man_zone_type" in df.columns
                else pl.lit(None).alias("defense_man_zone_type")
            ),
            (
                pl.col("defense_coverage_type").cast(pl.Utf8).alias("defense_coverage_type")
                if "defense_coverage_type" in df.columns
                else pl.lit(None).alias("defense_coverage_type")
            ),
        ]
    )

    situ_exprs: list[pl.Expr] = []
    score_diff = pl.col("score_differential").fill_null(0.0)
    ydstogo = pl.col("ydstogo").fill_null(0.0)
    yardline = pl.col("yardline_100").cast(pl.Float32).fill_null(100.0)
    half_secs_expr = (

thinking
**Examining touchdown counting and rolling window concerns**

thinking
**Reviewing data shifts and plan progress**
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  → Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering details on config and training**
exec
/bin/zsh -lc "sed -n '1,200p' config/training.yaml" in /Users/jameswest/Desktop/nfl_predict succeeded in 815ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# config/training.yaml
# Configuration for the model training pipeline (pipeline/train.py)
# NFL Player Prediction Targets

# --- Data I/O ---
data:
  raw_pbp_dir: 'data/raw/pbp_by_day'  # NFL play-by-play raw data
  feature_matrix_path: 'data/processed/final/processed.parquet'  # NFL feature matrix
  
  # Directory to save trained model artifacts (models and feature lists).
  model_output_dir: 'output/models'
  
  # Directory to save evaluation reports and metrics.
  metrics_output_dir: 'output/metrics/'

# =============================================================================
# SHARED COLUMN DEFINITIONS (using YAML anchors to avoid duplication)
# =============================================================================
# These columns leak same-game outcomes and must be excluded from training.

thinking
**Analyzing potential data inconsistencies and leakage**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/asof.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 2.75s:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Helpers for aligning feature generation to a consistent decision-time snapshot."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Mapping

import yaml

from utils.general.paths import PROJ_ROOT


@lru_cache(maxsize=1)
def _load_cutoff_config() -> Mapping[str, Any]:
    """Load decision cutoff configuration from config/config.yaml."""
    config_path = PROJ_ROOT / "config" / "config.yaml"
    try:

thinking
**Identifying discrepancies in feature construction**

I'm analyzing several potential issues in how features are constructed, especially around rolling window computations, drive-level feature usage, pre-snap route participation data reliability, and labeling choices. I want to assemble a clear list of 6-7 concrete findings summarizing inconsistencies, incomplete implementations, and data limitations.

thinking
**Analyzing potential data leakage and pipeline inconsistencies**

I'm exploring possible data leakage issues, especially around using full-season roster and injury data without proper as-of gating in predictions. I’m also checking whether training pipelines enforce the claimed multi-stage model architecture and if various feature-building steps apply correct cutoff filters, since odds features might not respect decision cutoff timestamps, risking leakage. This involves reviewing training, feature, and monitoring docs alongside function implementations to understand the conceptual gaps and inconsistencies.

thinking
**Searching for add_player_odds_features_to_df function**
exec
/bin/zsh -lc 'rg "def add_player" -n utils/feature/odds.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 926ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
1099:def add_player_odds_features_to_df(
exec
/bin/zsh -lc "sed -n '1099,1250p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 732ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
def add_player_odds_features_to_df(
    df_pl: pl.DataFrame, *, api_key: str | None = None, player_col: str = "player_name",
    per_game_workers: int = 4, verbose: bool = False,
) -> pl.DataFrame:
    """Attach player odds features directly to an in-memory Polars DataFrame."""
    from utils.general.config import get_odds_api_key
    
    if df_pl.is_empty():
        return df_pl
    
    if not api_key:
        api_key = get_odds_api_key()
    
    if not api_key:
        logger.info("add_player_odds_features_to_df: no odds_api_key configured – adding empty columns and returning")
        empty_exprs = [pl.lit(None).cast(pl.Float32).alias(c) for c in PLAYER_ODDS_NUMERIC_COLS]
        flag_expr = [pl.lit(0).cast(pl.Int8).alias("odds_expected")]
        df_base = df_pl.drop(PLAYER_ODDS_NUMERIC_COLS + ODDS_FLAG_COLUMNS, strict=False)
        return df_base.with_columns(empty_exprs + flag_expr)

thinking
**Analyzing data leakage and feature completeness**

thinking
**Reviewing data leakage and caching inconsistencies**

I'm noticing potential data leakage in feature functions using live API calls and uncertain odds snapshot timing. There's also a mismatch between documented and actual cache directories that could confuse or cause bugs. I want to gather all issues before moving from feature pipeline review to modeling validation, paying attention to training config settings and columns flagged as leak-prone.
exec
/bin/zsh -lc 'rg "LEAK_PRONE_COLUMNS" -n utils' in /Users/jameswest/Desktop/nfl_predict succeeded in 870ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
utils/general/__init__.py:18:    LEAK_PRONE_COLUMNS,
utils/general/__init__.py:47:    "LEAK_PRONE_COLUMNS",
utils/general/constants.py:8:config/training.yaml should mirror LEAK_PRONE_COLUMNS for YAML-based configuration,
utils/general/constants.py:113:LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
utils/general/constants.py:208:    "LEAK_PRONE_COLUMNS",
exec
/bin/zsh -lc "sed -n '100,160p' utils/general/constants.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 800ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
}


# =============================================================================
# Leak-Prone Columns (Same-Game Outcomes)
# =============================================================================
# Stats that leak same-game outcomes and must be excluded from training features.
# These represent what happened in the game we're trying to predict.
#
# This is the AUTHORITATIVE definition. config/training.yaml mirrors this list
# using YAML anchors for per-problem configuration. Keep them in sync!
# Validated by: tests/test_training_config_features.py

LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
    # Yardage (same-game outcomes)
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    # Usage counts (same-game outcomes)

thinking
**Analyzing modeling pipeline inconsistencies**

I’m reviewing the training pipeline for possible incomplete or risky implementations—especially around derived targets, odds features that may leak future info, rolling window system mismatches, and caching issues causing potential feature inaccuracies. I'll note risky spots with specifics to file paths and behaviors to highlight data leakage or half-finished components.

thinking
**Identifying potential data leakage in features and inference**

thinking
**Analyzing touchdown column naming mismatch**

thinking
**Analyzing touchdown feature inconsistencies**

I notice a mismatch between "touchdown" and "touchdowns" columns in different dataframes, causing rolling touchdown features to be always zero—this looks like a conceptual bug. Also, the way stats are counted might cause double counting or misalignment. I want to fix these naming and aggregation issues and then update the project plan to mark step 2 as done and step 3 as in progress.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  → Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Analyzing model training and feature inconsistencies**

thinking
**Evaluating data quality and documentation gaps**

I'm analyzing issues with sparse or missing data affecting model training, especially around pre-snap route participation features and roster/injury alignment leaking future info. I also notice documentation inconsistencies in rolling window implementation and potential feature instability due to network-dependent API calls, all of which could harm predictive reliability and require careful handling in the next steps.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  ✓ Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Preparing final summary and suggestions**

I'll create a clear bullet list of findings ordered by severity with approximate file paths and lines, highlighting key issues like touchdown rolling features and pre-snap usage, then follow with numbered next steps for the user to consider.
codex
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.
tokens used
110,897
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.

    <END_ANALYSIS_END>

    This step is PURELY research, investigation, and planning.
    Do NOT make any edits to the code or data in this step.

    I want:
    - A structured, prioritized plan.
    - Clear steps that can be implemented by an agent in later steps.
    - Notes on risk or potential pitfalls where relevant, but focus on high-value changes.
    - Explicit attention to:
        * eliminating data leakage,
        * aligning features with real football mechanisms that drive anytime TDs,
        * cleaning up hallucinated / legacy cruft that no longer makes sense.

    Again: no edits here, just the plan.

mcp startup: no servers

thinking
**Drafting a structured planning outline**
codex
**Remediation Plan (ordered by impact & leakage risk)**
- Fix odds leakage/hallucination: freeze player prop odds to archived pre-cutoff snapshots; add snapshot loader keyed by game_id/team/player + snapshot_ts; in training, drop/flag rows without snapshots; in inference, refuse to call live API, optionally fall back to schedule-only baseline; add tests to assert no odds columns present when snapshot missing.
- Repair rolling TD history: align stat key `touchdowns` across `stats.py`, aggregation, and cache; rebuild daily_totals cache; add schema check to fail if any rolling TD feature is all-null/zero; rerun backtests to measure regained signal.
- Trim rolling window surface to supported set: codify a vetted set of stats/windows/contexts (e.g., {1g,3g,5g,season} × {vs_any}); remove ps_* and other missing-cache stats from rolling requests; update docs to match; add runtime guard to error on requested stats not found in cache.
- Enforce as-of gating for roster/injury inputs in prediction/backtests: add decision-time cutoff (e.g., T-6h) and snapshot source; if unavailable, drop those features or mark unavailable; add regression test ensuring no injury status is from after game start in historical inference.
- Remove/retire ps_* targets/features if coverage is low: measure non-null rate by season; if sparse, drop from training configs and rolling stats; otherwise backfill realistically or scope to seasons with data; add check to block tasks with <X% coverage.
- Clarify/implement drive-level history or delete: either build drive-level daily cache and wire into rolling computation, or remove drive-context flags/constants/docs; add assertion preventing “drive” level when cache absent.
- Align docs with reality and guard drift: sync paths/names (cache/feature/daily_totals vs nfl_daily_totals); document the final rolling feature list and cutoff rules; add a pre-flight validation script that checks cache schemas, non-null rates, and leakage-sensitive columns before training.

**Execution Steps**
1) Odds snapshots: design schema + loader; replace `add_player_props_to_df` call; add fallback behavior and tests; rerun feature build on a small date range to validate columns.
2) Rolling TD fix: rename stat key, rebuild cache for a sample window, ensure rolling TD features populate; update tests.
3) Rolling feature pruning: define allowed stats/windows/contexts; update config/constants + pipeline call; add guard for missing stats; rerun feature build and inspect null/constant rates.
4) Roster/injury gating: add cutoff param in predict/backtest; wire to snapshot source; add assertion that injury_status_ts <= cutoff; adjust backtests to use gated features.
5) ps_* triage: compute coverage by season; decide keep/drop; update training.yaml and problem definitions; if dropped, remove from rolling stats and tasks.
6) Drive-level decision: either implement cache builder (drive totals) and wire into rolling_window or delete related flags/docs; add guard to prevent silent no-op.
7) Docs/validation: update rolling_window_implementation.md and monitoring docs; add validation script to check schemas, coverage, leak-prone columns excluded from train set; run it in CI/pre-flight.

**Risks/Pitfalls to watch**
- Cache rebuild time/size; may need incremental rebuild and date filters.
- Snapshot availability gaps; must define strict fallbacks to avoid silent leakage.
- Training/inference drift if odds/roster data are absent—prefer hard failure over silent nulls.
- Removing ps_* may change model interfaces; update downstream consumers/tests accordingly.
tokens used
15,743
**Remediation Plan (ordered by impact & leakage risk)**
- Fix odds leakage/hallucination: freeze player prop odds to archived pre-cutoff snapshots; add snapshot loader keyed by game_id/team/player + snapshot_ts; in training, drop/flag rows without snapshots; in inference, refuse to call live API, optionally fall back to schedule-only baseline; add tests to assert no odds columns present when snapshot missing.
- Repair rolling TD history: align stat key `touchdowns` across `stats.py`, aggregation, and cache; rebuild daily_totals cache; add schema check to fail if any rolling TD feature is all-null/zero; rerun backtests to measure regained signal.
- Trim rolling window surface to supported set: codify a vetted set of stats/windows/contexts (e.g., {1g,3g,5g,season} × {vs_any}); remove ps_* and other missing-cache stats from rolling requests; update docs to match; add runtime guard to error on requested stats not found in cache.
- Enforce as-of gating for roster/injury inputs in prediction/backtests: add decision-time cutoff (e.g., T-6h) and snapshot source; if unavailable, drop those features or mark unavailable; add regression test ensuring no injury status is from after game start in historical inference.
- Remove/retire ps_* targets/features if coverage is low: measure non-null rate by season; if sparse, drop from training configs and rolling stats; otherwise backfill realistically or scope to seasons with data; add check to block tasks with <X% coverage.
- Clarify/implement drive-level history or delete: either build drive-level daily cache and wire into rolling computation, or remove drive-context flags/constants/docs; add assertion preventing “drive” level when cache absent.
- Align docs with reality and guard drift: sync paths/names (cache/feature/daily_totals vs nfl_daily_totals); document the final rolling feature list and cutoff rules; add a pre-flight validation script that checks cache schemas, non-null rates, and leakage-sensitive columns before training.

**Execution Steps**
1) Odds snapshots: design schema + loader; replace `add_player_props_to_df` call; add fallback behavior and tests; rerun feature build on a small date range to validate columns.
2) Rolling TD fix: rename stat key, rebuild cache for a sample window, ensure rolling TD features populate; update tests.
3) Rolling feature pruning: define allowed stats/windows/contexts; update config/constants + pipeline call; add guard for missing stats; rerun feature build and inspect null/constant rates.
4) Roster/injury gating: add cutoff param in predict/backtest; wire to snapshot source; add assertion that injury_status_ts <= cutoff; adjust backtests to use gated features.
5) ps_* triage: compute coverage by season; decide keep/drop; update training.yaml and problem definitions; if dropped, remove from rolling stats and tasks.
6) Drive-level decision: either implement cache builder (drive totals) and wire into rolling_window or delete related flags/docs; add guard to prevent silent no-op.
7) Docs/validation: update rolling_window_implementation.md and monitoring docs; add validation script to check schemas, coverage, leak-prone columns excluded from train set; run it in CI/pre-flight.

**Risks/Pitfalls to watch**
- Cache rebuild time/size; may need incremental rebuild and date filters.
- Snapshot availability gaps; must define strict fallbacks to avoid silent leakage.
- Training/inference drift if odds/roster data are absent—prefer hard failure over silent nulls.
- Removing ps_* may change model interfaces; update downstream consumers/tests accordingly.

    <END_PLAN_END>

    Repository + Git requirements:

    - This repo is under git. At the very beginning of THIS RUN, before editing any files:
        1) Run `git status` to inspect the current state.
        2) If there are uncommitted changes from previous runs, stage and commit them with a concise
           message like `codex: iteration checkpoint` (or a slightly more descriptive variant).
        3) If a remote named `origin` exists and authentication allows, run `git push` so that the
           current state is saved remotely. If push fails due to auth or remote issues, continue with
           local commits only, but do NOT delete history.
        4) Only after ensuring there is a clean commit of the current state should you begin making
           new edits in this run.

    - During this run:
        * Make coherent, incremental commits as you reach logical checkpoints.
        * If you realize that your changes have badly broken the project and you cannot fix them
          cleanly within this run, you may revert to the last good commit (for example, using
          `git reset --hard HEAD` or `git checkout .`), then proceed more conservatively.

    Your tools allow you to:
    - Inspect files in this repo.
    - Edit files.
    - Run shell commands/tests as needed (e.g. project-specific tests, evaluations, or scripts).
    - Use git commands to create commits and, if possible, push them.

    Your goal in THIS RUN:
    - Implement as much of the plan as you reasonably can with high quality.
    - Prioritize correctness, alignment with football reality, and improved model performance
      over speed.
    - Run whatever tests or checks are appropriate to validate your changes.

    At the end of THIS RUN:
    - Give a concise update on your progress.
    - If the ENTIRE plan is fully implemented and validated, include a line with EXACTLY:
        The plan is completely implemented
      (case and spacing exactly as written).
    - If the plan is NOT fully implemented, include a concise progress summary starting with:
        PROGRESS:
      followed by a short description of what you accomplished and what remains.

    Take your time and implement the richest / most complete solution for the pieces you touch in this run.

mcp startup: no servers

thinking
**Preparing to check repository state**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && git status --short' in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

thinking
**Preparing to update plan**
Plan update
Implement prioritized fixes: rolling TD stat mismatch and odds snapshot leak first, prune unsupported rolling stats and enforce guards, then proceed to gating and documentation if time.
  → Address high-risk data leakage and feature correctness issues per remediation plan
  • Harden roster/injury gating and ps_* feature triage or deferral
  • Docs/validation updates and final verification

thinking
**Planning odds snapshot and stats fixes**

I’m figuring out how to handle the missing odds snapshot—maybe disabling live API and adjusting pipeline steps to avoid leaking incomplete data. I also want to fix the touchdown stat naming for consistency, which means carefully updating the NFL player stats utilities while making sure nothing breaks. I'll check those files next.
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '1,200p' utils/feature/stats.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 725ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL player statistics configuration for rolling window features.

Defines the standard set of stats tracked at all aggregation levels
(play, drive, game) for rolling window computations.

This module is the SINGLE SOURCE OF TRUTH for stat definitions used
throughout the pipeline. Import from here rather than defining locally.
"""

from typing import List

# =============================================================================
# Core Stats (used for drive and game level aggregation)
# =============================================================================

# Basic stats aggregated at player-drive level
PLAYER_DRIVE_STATS: List[str] = [
    # Yardage
    "receiving_yards",

thinking
**Aligning touchdown naming and evaluating rolling windows**
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '760,860p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 696ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

    logging.info("Enriching odds features (NFL)...")
    rows_before_odds = df_player_game_all.height
    df_player_game_all = add_nfl_odds_features_to_df(
        df_player_game_all,
        player_col="player_name",
        allow_schedule_fallback=False,
        drop_schedule_rows=True,
    )
    
    logging.info("Enriching player props (Anytime TD)...")
    df_player_game_all = add_player_props_to_df(
        df_player_game_all,
        player_col="player_name",
    )

    rows_after_odds = df_player_game_all.height
    logging.info(
        "Odds features attached; retained %d/%d rows (%.2f%%) with pre-cutoff snapshots",

thinking
**Reviewing rolling stats windows and data sources**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "NFL_PLAYER_STATS" utils/feature/daily_totals.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 728ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from utils.feature.stats import NFL_PLAYER_STATS
    for stat in NFL_PLAYER_STATS:

thinking
**Checking column mismatch issues**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "stat_num" -n utils/feature/daily_totals.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 700ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
8:  player_id, date, ctx, opponent → stat_num columns
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '120,240p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 722ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    # --- CONTEXT 2: vs_team (games vs specific opponent) ---
    if "opponent" in df.columns:
        vs_team = (
            df
            .group_by(["player_id", "season", "team", "opponent", "date"])
            .agg(stat_exprs)
            .with_columns([
                pl.lit("vs_team").alias("ctx"),
            ])
        )

        contexts = [vs_any, vs_team]
    else:
        # No opponent column, only vs_any context
        logger.warning(f"No opponent column for {day}, only vs_any context")
        contexts = [vs_any]

    # --- CONTEXT 3: with_team (career with current team only) ---
    if "team" in df.columns:

thinking
**Checking for stat_exprs definition**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "stat_exprs" -n utils/feature/daily_totals.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 695ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
91:    stat_exprs = []
94:            stat_exprs.append(pl.col(stat).fill_null(0).sum().alias(f"{stat}_num"))
97:    stat_exprs.append(pl.len().alias("denom"))
99:    if not stat_exprs:
113:        .agg(stat_exprs)
125:            .agg(stat_exprs)
142:            .agg(stat_exprs)
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '60,120p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 737ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    # Skip if already exists
    if out_file.exists():
        return

    # Load raw data for this day
    df = raw_loader(day)
    if df.is_empty():
        return

    # Normalize date column
    if "game_date" in df.columns:
        df = df.with_columns(pl.col("game_date").dt.date().alias("date"))
    elif "date" not in df.columns:
        logger.warning(f"No date column found for {day}, skipping")
        return
    
    # Filter to this specific day
    df = df.filter(pl.col("date") == day)

thinking
**Optimizing stats selection and pipeline safeguards**

I'm figuring out how to limit stats to a vetted subset matching available columns, mostly focusing on key football stats like receiving and rushing yards, touchdowns, and targets. I'll check the data schema or constants to pick safe features and adjust the pipeline and rolling features logic accordingly, adding runtime guards to catch missing stats to avoid errors during cache or rolling window feature builds. This way, the pipeline stays tidy and robust against unsupported stats.
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "def add_rolling_features" -n utils/feature/rolling_window.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 732ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
494:def add_rolling_features(df: pl.DataFrame, *, level: str = "game", stats: list[str], 
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '480,620p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 710ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        if "with_team" in contexts and feat_with_team is not None:
            out_joined = (
                out_joined.join_asof(
                    feat_with_team,
                    left_on="__ts",
                    right_on="__ts",
                    by=["player_id", "team"],
                )
                .select(pl.exclude(r"^.*_right$"))
            )
        
        return out_joined.drop("__ts")


def add_rolling_features(df: pl.DataFrame, *, level: str = "game", stats: list[str], 
                         windows: list[str], contexts: list[str], 
                         date_col: str = "game_date", **kwargs) -> pl.DataFrame:
    """Add rolling features."""
    roller = RollingWindow(level=level)

thinking
**Examining RollingWindow compute method**
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '240,520p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 734ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        logger.info(f"🔍 About to group and aggregate cache...")
        
        # Group and aggregate (EXACT MLB pattern)
        # CRITICAL: Filter null dates BEFORE groupby (groupby includes nulls as a group!)
        base_cols = ["ctx", "player_id", "team", "opponent", "date", "season"]
        numeric_cols = [
            c for c in totals_hist.columns
            if c not in base_cols and c != "__ts"
        ]
        agg_exprs = [pl.col(col).sum().alias(col) for col in numeric_cols]
        totals_hist_df = (
            totals_hist
            .select(pl.all().exclude("__ts"))  # Drop __ts from scan first
            .filter(pl.col("date").is_not_null())  # Filter BEFORE groupby
            .group_by(base_cols, maintain_order=True)
            .agg(agg_exprs)
            .collect()
        )
        

thinking
**Considering adding cache guards**
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '60,180p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 713ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Load cache with same pattern as MLB version."""
        
        def _add_requested_columns_to_empty(lf: pl.LazyFrame) -> pl.LazyFrame:
            if not columns:
                return lf
            df = lf.collect()
            existing = set(df.columns)
            missing = [c for c in columns if c not in existing]
            if missing:
                df = df.with_columns([pl.lit(0).cast(pl.Float64).alias(c) for c in missing])
            return df.lazy()
        
        if hi < lo:
            return _add_requested_columns_to_empty(self.EMPTY_TOTALS_LAZY)
        
        dates = [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]
        paths = [

thinking
**Investigating cache loading method**
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '180,240p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 747ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        ds = (
            ds.with_columns(
                ((pl.col(date_col) + pl.duration(days=1)) - pl.duration(milliseconds=1)).alias("__ts")
            )
            .sort(["player_id", "__ts"])
            .with_columns(pl.col("__ts").set_sorted())
        )
        
        # Get date range of INPUT data
        min_max = ds.select(
            pl.col(date_col).dt.date().min().alias("lo"),
            pl.col(date_col).dt.date().max().alias("hi"),
        ).collect()
        batch_min_date, batch_max_date = min_max.row(0)

        rolling_feature_names: list[str] = []
        for ctx in contexts:
            suffix = "" if ctx == "vs_any" else f"_{ctx}"
            for w in windows:

thinking
**Planning data validation and leakage fixes**

I’m thinking about adding runtime guards to ensure requested stats actually exist in the dataset schema to prevent building features for missing columns, which currently get filled with zeros and hurt feature value. I’ll implement errors when stats aren’t found and adjust the pipeline to subset stats accordingly to avoid failures. Also, I plan to tackle odds leakage by disabling live API usage in features like player props and NFL odds, verifying that odds data comes only from snapshots rather than live sources. I'll start by inspecting the odds feature functions to confirm this.
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "add_nfl_odds_features_to_df" -n utils/feature/odds.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 758ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
797:def add_nfl_odds_features_to_df(
826:        logger.warning("add_nfl_odds_features_to_df: DataFrame missing required columns %s", missing)
887:            logger.debug("add_nfl_odds_features_to_df: failed odds for %s – %s", cache_key, exc)
1360:    "add_nfl_odds_features_to_df",
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '780,920p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 723ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        
        # Vig percent delta
        current_vig = cutoff_data.get("moneyline_vig")
        opening_vig = compare_data.get("moneyline_vig")
        vig_pct = _pct_change(current_vig, opening_vig)
        vig_col = "moneyline_vig_pct_delta" if key == "open" else f"moneyline_vig_pct_delta{suffix}"
        row[vig_col] = vig_pct
    
    if return_snapshot_ts:
        return row, snapshot_ts_map, event_id
    return row


# =============================================================================
# NFL GAME-LEVEL ODDS ENRICHMENT
# =============================================================================

def add_nfl_odds_features_to_df(
    df_pl: pl.DataFrame, *, api_key: Optional[str] = None, player_col: str = "player_name",

thinking
**Planning data snapshot gating**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "player_props" -n' in /Users/jameswest/Desktop/nfl_predict succeeded in 779ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
utils/feature/odds.py:1189:    include_player_props: bool = False,
pipeline/feature.py:33:from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
pipeline/feature.py:771:    df_player_game_all = add_player_props_to_df(
pipeline/collect.py:299:    include_player_props: bool = False,
pipeline/collect.py:617:            include_player_props=include_player_props,
pipeline/collect.py:782:            include_player_props=args.include_player_props,
pipeline/collect.py:833:        include_player_props=False,
pipeline/predict.py:351:            include_player_props=True 
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '1160,1260p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 792ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# =============================================================================

def _format_week(value: object) -> str:
    try:
        return f"{int(value):02d}"
    except (TypeError, ValueError):
        return str(value)


def _normalize_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    df = schedule.copy()
    if "gameday" in df.columns:
        df["game_date"] = pd.to_datetime(df["gameday"], errors="coerce").dt.date
    elif "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    else:
        df["game_date"] = pd.NaT
    
    if "start_time_utc" in df.columns:

thinking
**Reviewing snapshot collection and player odds features**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "player_props" utils/feature/odds.py -n' in /Users/jameswest/Desktop/nfl_predict succeeded in 797ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
1189:    include_player_props: bool = False,
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '1260,1460p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 802ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
            event_id = None
        
        start_time_utc = getattr(game, "start_time_utc", pd.NaT)
        start_time_iso = None
        if pd.notna(start_time_utc):
            start_time_iso = pd.to_datetime(start_time_utc, utc=True, errors="coerce")
            start_time_iso = start_time_iso.isoformat() if pd.notna(start_time_iso) else None
        
        record: Dict[str, object] = {
            "season": int(season) if season is not None else None,
            "week": week,
            "game_id": game_id,
            "game_date": game_day_date,
            "home_team": home_team,
            "away_team": away_team,
            "season_type": getattr(game, "season_type", None),
            "start_time_utc": start_time_iso,
            "odds_snapshot_generated_utc": collected_utc,
            "event_id": event_id,

thinking
**Searching for add_player_odds_features usage**
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '1000,1160p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 3.06s:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

        selected_snap = pd.Timestamp(selected_snap)
        if selected_snap.tzinfo is None:
            selected_snap = selected_snap.tz_localize("UTC")
        else:
            selected_snap = selected_snap.tz_convert("UTC")

        props_expected = selected_snap >= PLAYER_PROPS_START_TS
        if not props_expected:
            for idx in game_df.index:
                odds_map[idx] = {"odds_anytime_td_price": None}
            return

        odds_props = client.get_historical_event_odds(
            event_id,
            PLAYER_MARKETS_PARAM,
            selected_snap.to_pydatetime(),
        )
        if not odds_props and logger.isEnabledFor(logging.WARNING):

thinking
**Evaluating snapshot usage for API calls**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "ODDS_SNAPSHOT_DIR" -n utils/feature/odds.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 2.09s:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
37:from utils.general.paths import CACHE_ROOT, ODDS_SNAPSHOT_DIR, PLAYER_ODDS_RAW_DIR
1308:        out_dir = ODDS_SNAPSHOT_DIR / f"season={season}" / f"week={_format_week(week)}"

thinking
**Checking general/paths constants**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "ODDS_SNAPSHOT_DIR" -n utils/general/paths.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 3.45s:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
25:ODDS_SNAPSHOT_DIR: Final[Path] = DATA_RAW / "odds_snapshots"
26:ODDS_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

thinking
**Evaluating snapshot-based odds data loading**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "odds_anytime" -n' in /Users/jameswest/Desktop/nfl_predict succeeded in 842ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
utils/feature/odds.py:108:PLAYER_ODDS_NUMERIC_COLS = ["odds_anytime_td_price"]
utils/feature/odds.py:975:                odds_map[idx] = {"odds_anytime_td_price": None}
utils/feature/odds.py:989:                odds_map[idx] = {"odds_anytime_td_price": None}
utils/feature/odds.py:1010:                odds_map[idx] = {"odds_anytime_td_price": None}
utils/feature/odds.py:1058:            odds_map[idx] = {"odds_anytime_td_price": td_price}
utils/feature/odds.py:1075:            odds_map[idx] = {"odds_anytime_td_price": df.loc[idx].get("odds_anytime_td_price")}

thinking
**Planning feature adjustments and renames**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "16" docs/rolling_window_implementation.md' in /Users/jameswest/Desktop/nfl_predict succeeded in 894ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
This creates **16 rolling features** initially (4 stats × 4 windows × 1 context).

thinking
**Reading code context**
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '40,120p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 846ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

**Player-Game Level** (`utils/feature/player_game_level.py`)
- Aggregates plays to (player, game) level
- Sums all stats per player per game
- Handles multi-role players (e.g., QB who also rushes)
- Output: `data/processed/player_game_by_week/`

#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)

**Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups

**Cache Structure**:
```
cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
```

**Contexts**:
- `vs_any`: All games for player on date
- `vs_team`: Games vs specific opponent on date

thinking
**Aligning feature selection and documentation**

I'm refining which stats to include from the hinted 16 rolling features by proposing a subset of about 10 features across multiple windows and contexts, balancing minimalism and utility. Meanwhile, I plan to keep docs consistent with these updates and consider caching and gating implementation details to handle missing stats and avoid external data fetches during inference and training.
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "include_player_props" -n pipeline/predict.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 725ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
351:            include_player_props=True 
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '320,380p' pipeline/predict.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 731ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    active_seasons = [int(s) for s in seasons if int(s) >= current_year]
    
    if not active_seasons:
        return
        
    try:
        # We don't force refresh indiscriminately, but reliance on lead_hours 
        # means it will fetch if the current time window matches a new lead time.
        logger.info("Ensuring weather forecasts for seasons %s", active_seasons)
        collect_weather_forecasts(active_seasons, force_refresh=False)
    except Exception as exc:
        logger.warning("Failed to ensure weather cache: %s", exc)


def _ensure_odds_cache(seasons: Iterable[int], start_date=None, end_date=None) -> None:
    """Ensure odds snapshot caches exist and are fresh for the requested window."""
    import datetime as dt
    
    current_year = dt.datetime.now().year

thinking
**Reevaluating player prop odds handling**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "add_player" pipeline/predict.py' in /Users/jameswest/Desktop/nfl_predict exited 1 in 798ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

thinking
**Investigating feature usage in pipeline/predict**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "odds" pipeline/predict.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 741ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from utils.feature.odds import collect_odds_snapshots
from utils.feature.odds import (
    add_nfl_odds_features_to_df,
def _ensure_odds_cache(seasons: Iterable[int], start_date=None, end_date=None) -> None:
    """Ensure odds snapshot caches exist and are fresh for the requested window."""
        logger.info("Ensuring odds snapshots for seasons %s (window: %s to %s)", active_seasons, start_date, end_date)
        collect_odds_snapshots(
        logger.warning("Failed to ensure odds cache: %s", exc)
                pl.col("odds_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
    enriched = add_nfl_odds_features_to_df(
        _ensure_odds_cache(games["season"].unique(), start_date=start_date, end_date=end_date)
        # We need to re-score the final model to get "implied_decimal_odds" and "prediction" boolean
        out["implied_decimal_odds"] = np.where(proba > 0, 1.0 / proba, np.nan)
        print(out[["player_name", "team", "prob_anytime_td", "implied_decimal_odds"]].head(5))

thinking
**Planning updates for rolling stats features**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "goal_to_go_target" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 991ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        pl.lit(0).cast(pl.Int64).alias("goal_to_go_target"),
    situational_cols = ["red_zone_target", "red_zone_carry", "goal_to_go_target", "goal_to_go_carry"]
        "goal_to_go_target_share",
            "goal_to_go_target",
            pl.lit(0).cast(pl.Int64).alias("goal_to_go_target"),
            pl.lit(0).cast(pl.Int64).alias("goal_to_go_target"),
            pl.col("goal_to_go_target").fill_null(0).sum().cast(pl.Int64).alias("goal_to_go_target"),
            pl.col("goal_to_go_target").fill_null(0).sum().alias("goal_to_go_target"),
        "td_count", "red_zone_target", "red_zone_carry", "goal_to_go_target", "goal_to_go_carry"
        pl.col("goal_to_go_target").fill_null(0).alias("goal_to_go_target"),

thinking
**Planning data validation checks**

thinking
**Planning feature set and pipeline adjustments**

thinking
**Evaluating rolling feature stats choices**

thinking
**Optimizing rolling stats constants**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "ROLLING_WINDOWS"' in /Users/jameswest/Desktop/nfl_predict succeeded in 776ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
docs/rolling_window_implementation.md:ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
pipeline/feature.py:from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
pipeline/feature.py:        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
pipeline/predict.py:from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
pipeline/predict.py:        windows=ROLLING_WINDOWS,
utils/feature/stats.py:ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]

thinking
**Checking feature constants in prediction**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "ROLLING_CONTEXTS" -n pipeline/predict.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 753ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
58:from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
1289:        contexts=ROLLING_CONTEXTS,
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '1260,1320p' pipeline/predict.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 719ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        asof_meta = pl.DataFrame()

    if asof_meta.is_empty() or (game_ids and asof_meta.filter(pl.col("game_id").is_in(game_ids)).height < len(game_ids)):
        if seasons:
            try:
                asof_meta = build_asof_metadata(seasons, force=True)
            except Exception as exc:
                logger.warning("Unable to rebuild as-of metadata: %s", exc)
                asof_meta = pl.DataFrame()

    if not asof_meta.is_empty():
        asof_meta = asof_meta.filter(pl.col("game_id").is_in(game_ids))
        asof_meta = asof_meta.with_columns(
            [
                pl.col("cutoff_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("cutoff_ts").cast(pl.Datetime("ms", "UTC")).alias("decision_cutoff_ts"),
                pl.col("injury_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("roster_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("odds_snapshot_ts").cast(pl.Datetime("ms", "UTC")),

thinking
**Reviewing feature renaming impact**
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '1320,1380p' pipeline/predict.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 696ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        "1g_ps_targets_inline_count_per_game": "ps_targets_inline_count",
        "3g_ps_targets_inline_count_per_game": "ps_targets_inline_count_l3",
        "1g_ps_targets_backfield_count_per_game": "ps_targets_backfield_count",
        "3g_ps_targets_backfield_count_per_game": "ps_targets_backfield_count_l3",
        "1g_ps_targets_slot_share_per_game": "ps_targets_slot_share",
        "3g_ps_targets_slot_share_per_game": "ps_targets_slot_share_l3",
        "1g_ps_targets_wide_share_per_game": "ps_targets_wide_share",
        "3g_ps_targets_wide_share_per_game": "ps_targets_wide_share_l3",
        "1g_ps_targets_inline_share_per_game": "ps_targets_inline_share",
        "3g_ps_targets_inline_share_per_game": "ps_targets_inline_share_l3",
        "1g_ps_targets_backfield_share_per_game": "ps_targets_backfield_share",
        "3g_ps_targets_backfield_share_per_game": "ps_targets_backfield_share_l3",
    }
    existing = set(enriched.columns)
    valid_renames = {k: v for k, v in rename_map.items() if k in existing}
    if valid_renames:
        enriched = enriched.rename(valid_renames)

    enriched = _apply_ps_fallback(enriched, season_hint=season_hint)

thinking
**Evaluating impact of reducing rolling stats**

I’m sorting through how trimming rolling stats might break downstream aliasing and model features. The tricky part is balancing dropping unsupported stats while keeping essential ones so the training and inference pipelines still produce expected columns like target_prev. I’ll need to check training configs, existing aliases, and data sources like player_game_level for snap counts to decide which stats can be safely excluded without breaking the model.
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "offense_snaps" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 826ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        pl.lit(0).cast(pl.Int32).alias("snap_offense_snaps_prev"),
                    "offense_snaps",
                        for col in ("offense_snaps", "offense_pct", "defense_snaps", "defense_pct", "st_snaps", "st_pct")
        snap_feature_cols = ["offense_snaps", "offense_pct", "defense_snaps", "defense_pct", "st_snaps", "st_pct"]
            snap_count_cols = [col for col in ("offense_snaps", "defense_snaps", "st_snaps") if col in df_merged.columns]
            "offense_snaps",

thinking
**Evaluating rolling stats reduction**

thinking
**Updating feature pipeline and docs**

I’m adjusting the pipeline to remove player props enrichment since snapshots aren’t available, updating logging accordingly, and cleaning imports. I’ll also tweak RollingWindow’s compute guard and revise documentation to reflect new cache paths and curated stats, aligning these changes with the plan steps before committing.
file update
M /Users/jameswest/Desktop/nfl_predict/utils/feature/stats.py
@@ -28,3 +28,3 @@
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
@@ -83,12 +83,27 @@
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
@@ -104,3 +119,3 @@
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)
apply_patch(auto_approved=true) exited 0 in 27ms:
Success. Updated the following files:
M utils/feature/stats.py
file update:
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Considering documentation and code impacts**
file update
M /Users/jameswest/Desktop/nfl_predict/utils/feature/rolling_window.py
@@ -167,2 +167,9 @@
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
apply_patch(auto_approved=true) exited 0 in 39ms:
Success. Updated the following files:
M utils/feature/rolling_window.py
file update:
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Updating imports for feature pipeline**
file update:
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Viewing file start**
exec
/bin/zsh -lc "cd /Users/jameswest/Desktop/nfl_predict && sed -n '1,120p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 766ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Feature pipeline orchestrator for NFL player-level predictions.

Stages
------
1. **build_play_level**         – derive cleaned play-level slices
2. **build_player_drive_level** – aggregate per drive
3. **build_player_game_level**  – aggregate per game and join context

This wrapper triggers each stage, optionally enforces a schema after each
write, and returns the final game-level DataFrame for downstream usage.
"""

import logging
import os
from datetime import date, timedelta
from typing import Sequence

file update:
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Adjusting import section with patch**
file update
M /Users/jameswest/Desktop/nfl_predict/pipeline/feature.py
@@ -31,4 +31,9 @@
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
@@ -1161,2 +1166,2 @@
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
apply_patch(auto_approved=true) exited 0 in 53ms:
Success. Updated the following files:
M pipeline/feature.py
file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..95a566245078d80fcbc85bacb5b9bc4180b6f26a
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -1159,4 +1164,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..95a566245078d80fcbc85bacb5b9bc4180b6f26a
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -1159,4 +1164,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Preparing to adjust odds and rolling logic**
file update
M /Users/jameswest/Desktop/nfl_predict/pipeline/feature.py
@@ -773,8 +773,2 @@
     )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
-    )
 
@@ -791,2 +785,5 @@
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
apply_patch(auto_approved=true) exited 0 in 63ms:
Success. Updated the following files:
M pipeline/feature.py
file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..db122531d70993ec6acb6abac04388ed59e89245
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -1159,4 +1161,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..db122531d70993ec6acb6abac04388ed59e89245
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -1159,4 +1161,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Verifying column removals and rolling adjustments**
file update
M /Users/jameswest/Desktop/nfl_predict/pipeline/feature.py
@@ -826,7 +826,6 @@
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
apply_patch(auto_approved=true) exited 0 in 49ms:
Success. Updated the following files:
M pipeline/feature.py
file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..d8a63ab1890e04a763aa580ac8f3c749e43a49d5
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,12 +824,11 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
         df_player_game_all = add_rolling_features(
             df_player_game_all,
             level="game",
@@ -1159,4 +1160,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..d8a63ab1890e04a763aa580ac8f3c749e43a49d5
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,12 +824,11 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
         df_player_game_all = add_rolling_features(
             df_player_game_all,
             level="game",
@@ -1159,4 +1160,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Updating rolling window constants usage**
file update
M /Users/jameswest/Desktop/nfl_predict/pipeline/predict.py
@@ -57,3 +57,8 @@
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
apply_patch(auto_approved=true) exited 0 in 58ms:
Success. Updated the following files:
M pipeline/predict.py
file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..d8a63ab1890e04a763aa580ac8f3c749e43a49d5
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,12 +824,11 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
         df_player_game_all = add_rolling_features(
             df_player_game_all,
             level="game",
@@ -1159,4 +1160,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..196f7437d015d51b4ad24ac0bc9eca480b151bf4
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..d8a63ab1890e04a763aa580ac8f3c749e43a49d5
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,12 +824,11 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
         df_player_game_all = add_rolling_features(
             df_player_game_all,
             level="game",
@@ -1159,4 +1160,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..196f7437d015d51b4ad24ac0bc9eca480b151bf4
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Updating rolling features call**
file update
M /Users/jameswest/Desktop/nfl_predict/pipeline/predict.py
@@ -1291,3 +1291,3 @@
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
apply_patch(auto_approved=true) exited 0 in 60ms:
Success. Updated the following files:
M pipeline/predict.py
file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..d8a63ab1890e04a763aa580ac8f3c749e43a49d5
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,12 +824,11 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
         df_player_game_all = add_rolling_features(
             df_player_game_all,
             level="game",
@@ -1159,4 +1160,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..a658096da7c3d8f80147004984cf022ff14f1c90
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1289,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..d8a63ab1890e04a763aa580ac8f3c749e43a49d5
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,12 +824,11 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
         df_player_game_all = add_rolling_features(
             df_player_game_all,
             level="game",
@@ -1159,4 +1160,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..a658096da7c3d8f80147004984cf022ff14f1c90
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1289,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Verifying alias mappings and constants usage**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "ROLLING_LEVELS"' in /Users/jameswest/Desktop/nfl_predict succeeded in 699ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
docs/rolling_window_implementation.md:ROLLING_LEVELS = ["game", "drive"]
utils/feature/stats.py:ROLLING_LEVELS = ["game"]
file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..d8a63ab1890e04a763aa580ac8f3c749e43a49d5
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,12 +824,11 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
         df_player_game_all = add_rolling_features(
             df_player_game_all,
             level="game",
@@ -1159,4 +1160,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..a658096da7c3d8f80147004984cf022ff14f1c90
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1289,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Updating documentation and verifying data paths**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "NFL_PLAYER_STATS" pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 765ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    NFL_PLAYER_STATS,
file update:
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..d8a63ab1890e04a763aa580ac8f3c749e43a49d5
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,12 +824,11 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
         df_player_game_all = add_rolling_features(
             df_player_game_all,
             level="game",
@@ -1159,4 +1160,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..a658096da7c3d8f80147004984cf022ff14f1c90
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1289,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Preparing documentation updates**
file update
M /Users/jameswest/Desktop/nfl_predict/docs/rolling_window_implementation.md
@@ -46,3 +46,3 @@
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
@@ -52,3 +52,3 @@
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
@@ -56,4 +56,3 @@
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
@@ -72,13 +71,10 @@
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
@@ -91,3 +87,3 @@
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
@@ -107,15 +103,25 @@
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
@@ -265,2 +271 @@
 **Ready for**: Production pipeline run
-
apply_patch(auto_approved=true) exited 0 in 75ms:
Success. Updated the following files:
M docs/rolling_window_implementation.md
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..d8a63ab1890e04a763aa580ac8f3c749e43a49d5
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,12 +824,11 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
         df_player_game_all = add_rolling_features(
             df_player_game_all,
             level="game",
@@ -1159,4 +1160,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..a658096da7c3d8f80147004984cf022ff14f1c90
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1289,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..d8a63ab1890e04a763aa580ac8f3c749e43a49d5
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,12 +824,11 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
         df_player_game_all = add_rolling_features(
             df_player_game_all,
             level="game",
@@ -1159,4 +1160,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..a658096da7c3d8f80147004984cf022ff14f1c90
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1289,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Checking pipeline dependencies**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "NFL_PLAYER_STATS" pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 730ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    NFL_PLAYER_STATS,
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..d8a63ab1890e04a763aa580ac8f3c749e43a49d5
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,13 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +770,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +783,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,12 +824,11 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
         df_player_game_all = add_rolling_features(
             df_player_game_all,
             level="game",
@@ -1159,4 +1160,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..a658096da7c3d8f80147004984cf022ff14f1c90
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1289,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Removing potentially unused import**
file update
M /Users/jameswest/Desktop/nfl_predict/pipeline/feature.py
@@ -31,8 +31,3 @@
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import (
-    NFL_PLAYER_STATS,
-    ROLLING_FEATURE_STATS,
-    ROLLING_WINDOWS,
-    ROLLING_CONTEXTS,
-)
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import add_nfl_odds_features_to_df
apply_patch(auto_approved=true) exited 0 in 72ms:
Success. Updated the following files:
M pipeline/feature.py
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..52490053c81e1056467f0027d73c87701a245aec
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,11 +819,10 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
         df_player_game_all = add_rolling_features(
             df_player_game_all,
@@ -1159,4 +1155,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..a658096da7c3d8f80147004984cf022ff14f1c90
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1289,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..52490053c81e1056467f0027d73c87701a245aec
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,11 +819,10 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
         df_player_game_all = add_rolling_features(
             df_player_game_all,
@@ -1159,4 +1155,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..a658096da7c3d8f80147004984cf022ff14f1c90
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1289,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Checking import usage and order**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "NFL_PLAYER_STATS" pipeline/predict.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 725ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    NFL_PLAYER_STATS,
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..52490053c81e1056467f0027d73c87701a245aec
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,11 +819,10 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
         df_player_game_all = add_rolling_features(
             df_player_game_all,
@@ -1159,4 +1155,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..a658096da7c3d8f80147004984cf022ff14f1c90
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,12 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import (
+    NFL_PLAYER_STATS,
+    ROLLING_FEATURE_STATS,
+    ROLLING_WINDOWS,
+    ROLLING_CONTEXTS,
+)
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1289,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Removing unused patch**
file update
M /Users/jameswest/Desktop/nfl_predict/pipeline/predict.py
@@ -57,8 +57,3 @@
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import (
-    NFL_PLAYER_STATS,
-    ROLLING_FEATURE_STATS,
-    ROLLING_WINDOWS,
-    ROLLING_CONTEXTS,
-)
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
apply_patch(auto_approved=true) exited 0 in 71ms:
Success. Updated the following files:
M pipeline/predict.py
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..52490053c81e1056467f0027d73c87701a245aec
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,11 +819,10 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
         df_player_game_all = add_rolling_features(
             df_player_game_all,
@@ -1159,4 +1155,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..50b936b3a05a9590b8390f879b1921f116c8a4e2
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1284,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..52490053c81e1056467f0027d73c87701a245aec
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,11 +819,10 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
         df_player_game_all = add_rolling_features(
             df_player_game_all,
@@ -1159,4 +1155,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..50b936b3a05a9590b8390f879b1921f116c8a4e2
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1284,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Checking for other NFL_PLAYER_STATS references**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rg "NFL_PLAYER_STATS"' in /Users/jameswest/Desktop/nfl_predict succeeded in 789ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
docs/rolling_window_implementation.md:NFL_PLAYER_STATS = [
utils/feature/daily_totals.py:from utils.feature.stats import NFL_PLAYER_STATS
utils/feature/daily_totals.py:    for stat in NFL_PLAYER_STATS:
utils/feature/stats.py:NFL_PLAYER_STATS: List[str] = [
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..52490053c81e1056467f0027d73c87701a245aec
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,11 +819,10 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
         
         df_player_game_all = add_rolling_features(
             df_player_game_all,
@@ -1159,4 +1155,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..50b936b3a05a9590b8390f879b1921f116c8a4e2
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1284,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Updating RollingWindow for missing stats**

I'm planning to adjust RollingWindow to handle missing stats gracefully by filtering the rolling features to those present in the dataset and logging any missing ones. This should prevent errors during computation and make the pipeline more robust without breaking existing behavior. I'll implement this pre-check right before calling add_rolling_features.
file update
M /Users/jameswest/Desktop/nfl_predict/pipeline/feature.py
@@ -825,16 +825,27 @@
         rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
apply_patch(auto_approved=true) exited 0 in 89ms:
Success. Updated the following files:
M pipeline/feature.py
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..50b936b3a05a9590b8390f879b1921f116c8a4e2
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1284,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..50b936b3a05a9590b8390f879b1921f116c8a4e2
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1284,7 +1284,7 @@
     enriched = add_rolling_features(
         pl_df,
         level="game",
-        stats=NFL_PLAYER_STATS,
+        stats=ROLLING_FEATURE_STATS,
         windows=ROLLING_WINDOWS,
         contexts=ROLLING_CONTEXTS,
         date_col="game_date",
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Adding guards to prevent missing data errors**
file update
M /Users/jameswest/Desktop/nfl_predict/pipeline/predict.py
@@ -1283,12 +1283,20 @@
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=ROLLING_FEATURE_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
apply_patch(auto_approved=true) exited 0 in 69ms:
Success. Updated the following files:
M pipeline/predict.py
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Preparing code changes and commit strategy**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && git status --short' in /Users/jameswest/Desktop/nfl_predict succeeded in 763ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
 M docs/rolling_window_implementation.md
 M pipeline/feature.py
 M pipeline/predict.py
 M utils/feature/rolling_window.py
 M utils/feature/stats.py
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Preparing commit and pushing changes**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && git add docs/rolling_window_implementation.md pipeline/feature.py pipeline/predict.py utils/feature/rolling_window.py utils/feature/stats.py' in /Users/jameswest/Desktop/nfl_predict exited 128 in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
fatal: Unable to create '/Users/jameswest/Desktop/nfl_predict/.git/index.lock': Operation not permitted
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Checking for existing Git index lock**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && ls .git/index.lock' in /Users/jameswest/Desktop/nfl_predict exited 1 in 677ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
ls: .git/index.lock: No such file or directory
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Considering permission errors with git add**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && git add docs/rolling_window_implementation.md pipeline/feature.py pipeline/predict.py utils/feature/rolling_window.py utils/feature/stats.py' in /Users/jameswest/Desktop/nfl_predict exited 128 in 701ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
fatal: Unable to create '/Users/jameswest/Desktop/nfl_predict/.git/index.lock': Operation not permitted
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Investigating git directory permissions**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && ls -ld .git' in /Users/jameswest/Desktop/nfl_predict succeeded in 678ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
drwxr-xr-x  13 jameswest  staff  416 Nov 26 01:08 .git
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Checking file permissions**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && ls -l .git/index' in /Users/jameswest/Desktop/nfl_predict succeeded in 737ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
-rw-r--r--  1 jameswest  staff  13040 Nov 26 01:04 .git/index
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Investigating index.lock creation failure**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && touch .git/index.lock' in /Users/jameswest/Desktop/nfl_predict exited 1 in 730ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
touch: .git/index.lock: Operation not permitted
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Diagnosing .git directory write restrictions**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && touch sandbox_test.tmp' in /Users/jameswest/Desktop/nfl_predict succeeded in 714ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Analyzing git sandbox restrictions**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && rm sandbox_test.tmp' in /Users/jameswest/Desktop/nfl_predict succeeded in 752ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Handling git permission issue and updating plan status**
Plan update
Resolved rolling stat mismatch/noisy feature surface and disabled live player prop enrichment; updated docs accordingly. Roster/injury gating and deeper ps_* triage still pending for a future pass.
  ✓ Address high-risk data leakage and feature correctness issues per remediation plan
  • Harden roster/injury gating and ps_* feature triage or deferral
  ✓ Docs/validation updates and final verification
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Preparing git diff summary**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && git diff --stat' in /Users/jameswest/Desktop/nfl_predict succeeded in 752ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
 docs/rolling_window_implementation.md | 47 +++++++++++++++------------
 pipeline/feature.py                   | 61 +++++++++++++++++++----------------
 pipeline/predict.py                   | 30 ++++++++++-------
 utils/feature/rolling_window.py       |  7 ++++
 utils/feature/stats.py                | 29 +++++++++++++----
 5 files changed, 108 insertions(+), 66 deletions(-)
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)


thinking
**Summarizing final progress and next steps**

I'm pulling together a final message that lists the key code changes like renaming rolling stats, adding guards for missing contexts, disabling certain player prop calls, and updating docs. I'll note tests weren’t run and that I couldn't commit due to a permission issue locking the .git/index file. To keep clarity, I'll include a PROGRESS line highlighting what got done and what's still pending, plus suggest natural next steps with numbered options for the user to consider.
codex
- Fixed rolling stat alignment by renaming touchdown stat to `touchdowns` and adding a guard against requesting missing stats; rolling windows/contexts trimmed to supported set with curated high-signal stats and missing-stat skip logic (`utils/feature/stats.py`, `utils/feature/rolling_window.py`, `pipeline/feature.py`, `pipeline/predict.py`).
- Disabled live player-prop enrichment to avoid leakage/hallucinated markets and ensured odds-only enrichment logs without props (`pipeline/feature.py`).
- Updated rolling window docs to match actual cache paths, contexts, windows, and curated stat surface (`docs/rolling_window_implementation.md`).

PROGRESS: Rolling/odds fixes and docs are done; roster/injury as-of gating and ps_* triage still pending for a later pass.

Tests: not run (not requested).

Git: Unable to stage/commit due to permission error creating `.git/index.lock` (touch failed with “Operation not permitted”). Please stage/commit manually once permissions allow.
file update:
diff --git a/docs/rolling_window_implementation.md b/docs/rolling_window_implementation.md
index 83b7bc661f9daa664f8d7d8ef139d53bd6e25ce1..19a8c8ca3425119e7a4662449bf9d88d104a5b0d
--- a/docs/rolling_window_implementation.md
+++ b/docs/rolling_window_implementation.md
@@ -44,18 +44,17 @@
 - Handles multi-role players (e.g., QB who also rushes)
 - Output: `data/processed/player_game_by_week/`
 
-#### **C. Daily Totals Cache** (`utils/feature/nfl_daily_totals.py`)
+#### **C. Daily Totals Cache** (`utils/feature/daily_totals.py`)
 
 **Purpose**: Pre-aggregate player stats by date for efficient rolling window lookups
 
 **Cache Structure**:
 ```
-cache/feature/nfl_daily_totals/level={game|drive}/date=YYYY-MM-DD/part.parquet
+cache/feature/daily_totals/level=game/date=YYYY-MM-DD/part.parquet
 ```
 
 **Contexts**:
-- `vs_any`: All games for player on date
-- `vs_team`: Games vs specific opponent on date
+- `vs_any`: All games for player on date (currently used)
 
 **Columns**:
 - ID: `player_id`, `date`, `ctx`, `opponent`
@@ -70,17 +69,14 @@
 - Efficient cache-based lookups
 
 **Windows Supported**:
-- Integer N: Last N games (1, 2, 3, 4)
+- Integer N: Last N games (1, 3, 5)
 - `"season"`: Current season-to-date
-- `"lifetime"`: All available history
 
 **Contexts**:
-- `vs_any`: Performance against all opponents
-- `vs_team`: Performance vs specific opponent
+- `vs_any`: Performance against all opponents (only context computed today)
 
 **Levels**:
-- `game`: Per-game rates (yards per game)
-- `drive`: Per-drive rates (yards per drive)
+- `game`: Per-game rates (yards per game) — drive-level cache is not wired yet
 
 **Feature Naming Convention**:
 ```
@@ -89,7 +85,7 @@
 Examples:
   1g_receiving_yards_per_game           (last game avg, all opponents)
   3g_rushing_yards_per_game_vs_team     (last 3 games vs opponent)
-  seasong_touchdown_per_game            (season-to-date TD avg)
+  seasong_touchdowns_per_game           (season-to-date TD avg)
   lifetimeg_passing_yards_per_game      (career passing avg)
 ```
 
@@ -105,19 +101,29 @@
     "carry",
     "pass_attempt",
     "completion",
-    "touchdown",
+    "touchdowns",
 ]
 
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
-ROLLING_CONTEXTS = ["vs_any", "vs_team"]
-ROLLING_LEVELS = ["game", "drive"]
-```
-
-**Total Feature Space**: 9 stats × 6 windows × 2 contexts × 2 levels = **216 possible features**
+ROLLING_FEATURE_STATS = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
 
-## Integration into Pipeline (`pipeline/feature.py`)
+ROLLING_WINDOWS = [1, 3, 5, "season"]
+ROLLING_CONTEXTS = ["vs_any"]
+ROLLING_LEVELS = ["game"]
+```
 
-The feature pipeline now includes:
+**Total Feature Space**: 11 stats × 4 windows × 1 context × 1 level = **44 possible features** (initial rollout uses the curated list above)
 
 1. **Build player-game aggregations** (existing)
 2. **Build daily cache** → `build_daily_cache_range()`
@@ -263,4 +269,3 @@
 **Implementation Date**: November 4, 2025
 **Status**: ✅ Complete and tested
 **Ready for**: Production pipeline run
-
diff --git a/pipeline/feature.py b/pipeline/feature.py
index 4b28f26c6343681519b85dc0c80a27bad9e42259..6c47a91d08f9d309f94323efd4708bb992dc4ce0
--- a/pipeline/feature.py
+++ b/pipeline/feature.py
@@ -29,8 +29,8 @@
 from utils.feature.targets import validate_target_columns
 from utils.feature.daily_totals import build_daily_cache_range
 from utils.feature.rolling_window import add_rolling_features
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
-from utils.feature.odds import add_nfl_odds_features_to_df, add_player_odds_features_to_df as add_player_props_to_df
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.odds import add_nfl_odds_features_to_df
 from utils.feature.team_context import add_team_context_features, compute_team_context_history
 from utils.feature.offense_context import (
     add_offense_context_features_training,
@@ -765,12 +765,6 @@
         player_col="player_name",
         allow_schedule_fallback=False,
         drop_schedule_rows=True,
-    )
-    
-    logging.info("Enriching player props (Anytime TD)...")
-    df_player_game_all = add_player_props_to_df(
-        df_player_game_all,
-        player_col="player_name",
     )
 
     rows_after_odds = df_player_game_all.height
@@ -784,6 +778,9 @@
         df_player_game_all = df_player_game_all.with_columns(
             pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
         )
+    if "odds_anytime_td_price" in df_player_game_all.columns:
+        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
+        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")
 
     numeric_casts = []
     if "depth_chart_order" in df_player_game_all.columns:
@@ -822,25 +819,35 @@
     # ------------------------------------------------------------------
     if build_rolling:
         logging.info("🔹 Computing rolling window features...")
-        
-        # Use all NFL player stats for rolling windows
-        rolling_stats = NFL_PLAYER_STATS  # All stats from nfl_stats.py
-        rolling_windows = ROLLING_WINDOWS  # All windows: 1, 2, 3, 4, season, lifetime
-        rolling_contexts = ROLLING_CONTEXTS  # All contexts: vs_any, vs_team
-        
-        df_player_game_all = add_rolling_features(
-            df_player_game_all,
-            level="game",
-            stats=rolling_stats,
-            windows=rolling_windows,
-            contexts=rolling_contexts,
-            date_col="game_date",
-            player_col="player_id",
-            opponent_col="opponent",
-        )
+
+        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
+        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
+        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any
+
+        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
+        if missing_stats:
+            logging.warning(
+                "Rolling feature stats missing from dataset, will skip: %s",
+                ", ".join(sorted(missing_stats)),
+            )
+            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
+        if not rolling_stats:
+            logging.warning("No rolling stats available; skipping rolling feature computation.")
+            logging.info("✅  Added 0 rolling features")
+        else:
+            df_player_game_all = add_rolling_features(
+                df_player_game_all,
+                level="game",
+                stats=rolling_stats,
+                windows=rolling_windows,
+                contexts=rolling_contexts,
+                date_col="game_date",
+                player_col="player_id",
+                opponent_col="opponent",
+            )
         
-        rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
-        logging.info(f"✅  Added {len(rolling_cols)} rolling features")
+            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
+            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
     
     logging.info("Deriving historical usage share features...")
     share_exprs: list[pl.Expr] = []
@@ -1159,4 +1166,4 @@
     except Exception as exc:
         logging.warning("Failed to rebuild as-of metadata: %s", exc)
 
-    logging.info("Context histories refreshed.")
\ No newline at end of file
+    logging.info("Context histories refreshed.")
diff --git a/pipeline/predict.py b/pipeline/predict.py
index 3a638a72e7b395c7e110e9ee28bd983f00ef0ff9..085b3827db1af50daff9d65491694934bcfc006a
--- a/pipeline/predict.py
+++ b/pipeline/predict.py
@@ -55,7 +55,7 @@
 )
 from utils.collect.weather_forecasts import collect_weather_forecasts
 from utils.feature.odds import collect_odds_snapshots
-from utils.feature.stats import NFL_PLAYER_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
+from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
 from utils.feature.odds import (
     add_nfl_odds_features_to_df,
     NFL_ODDS_COLUMNS,
@@ -1281,16 +1281,24 @@
         )
         pl_df = pl_df.join(asof_meta, on="game_id", how="left")
 
-    enriched = add_rolling_features(
-        pl_df,
-        level="game",
-        stats=NFL_PLAYER_STATS,
-        windows=ROLLING_WINDOWS,
-        contexts=ROLLING_CONTEXTS,
-        date_col="game_date",
-        player_col="player_id",
-        opponent_col="opponent",
-    )
+    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
+    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
+    if missing_stats:
+        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))
+
+    if rolling_stats:
+        enriched = add_rolling_features(
+            pl_df,
+            level="game",
+            stats=rolling_stats,
+            windows=ROLLING_WINDOWS,
+            contexts=ROLLING_CONTEXTS,
+            date_col="game_date",
+            player_col="player_id",
+            opponent_col="opponent",
+        )
+    else:
+        enriched = pl_df
 
     # Rename rolling features to match model expectations (snap counts)
     # The model was trained on features named like 'snap_offense_pct_prev'
diff --git a/utils/feature/rolling_window.py b/utils/feature/rolling_window.py
index da2fdf477de0081e490825d0b471a7ae8e435360..d0f2c9ace8fcac53ebebbbe3e09b55253f5930e9
--- a/utils/feature/rolling_window.py
+++ b/utils/feature/rolling_window.py
@@ -165,6 +165,13 @@
         
         schema = ds.collect_schema()
         schema_names = set(schema.names())
+
+        missing_stats = [k for k in keys if k not in schema_names]
+        if missing_stats:
+            raise ValueError(
+                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
+                "Ensure stats requested for rolling features exist in the base frame."
+            )
         
         # Cast player_id
         ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
diff --git a/utils/feature/stats.py b/utils/feature/stats.py
index 635565bac8faa291697022e494d4b4305901b346..a6f324b662101ef922661f22384a66d6e5339a5b
--- a/utils/feature/stats.py
+++ b/utils/feature/stats.py
@@ -26,7 +26,7 @@
     "pass_attempt",
     "completion",
     # Touchdowns
-    "touchdown",
+    "touchdowns",
     "passing_td",
     "rushing_td_count",
     "receiving_td_count",
@@ -81,16 +81,31 @@
 
 # Windows for rolling computations (at game level)
 # Integers = N most recent games, "season" = current season, "lifetime" = all history
-ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]
+ROLLING_WINDOWS = [1, 3, 5, "season"]
 
 # Contexts for rolling computations
-# "vs_any" = all opponents, "vs_team" = specific opponent only
-ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]
+# "vs_any" = all opponents; vs_team/with_team disabled until caches exist
+ROLLING_CONTEXTS = ["vs_any"]
 
 # Aggregation levels for rolling windows
-# "game" = per-game rates, "drive" = per-drive rates
-ROLLING_LEVELS = ["game", "drive"]
+# Only per-game rolling is currently supported; drive-level cache not wired
+ROLLING_LEVELS = ["game"]
 
+# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
+ROLLING_FEATURE_STATS: List[str] = [
+    "touchdowns",
+    "target",
+    "carry",
+    "pass_attempt",
+    "red_zone_target",
+    "red_zone_carry",
+    "goal_to_go_target",
+    "goal_to_go_carry",
+    "receiving_yards",
+    "rushing_yards",
+    "passing_yards",
+]
+
 # Columns required for player identification and time-series operations
 PLAYER_ID_COLS = ["player_id", "player_name"]
 GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
@@ -102,5 +117,5 @@
 # Examples:
 #   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
 #   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
-#   - seasong_touchdown_per_game           (season-to-date)
+#   - seasong_touchdowns_per_game          (season-to-date)
 #   - lifetimeg_passing_yards_per_drive    (career avg per drive)

tokens used
73,615
- Fixed rolling stat alignment by renaming touchdown stat to `touchdowns` and adding a guard against requesting missing stats; rolling windows/contexts trimmed to supported set with curated high-signal stats and missing-stat skip logic (`utils/feature/stats.py`, `utils/feature/rolling_window.py`, `pipeline/feature.py`, `pipeline/predict.py`).
- Disabled live player-prop enrichment to avoid leakage/hallucinated markets and ensured odds-only enrichment logs without props (`pipeline/feature.py`).
- Updated rolling window docs to match actual cache paths, contexts, windows, and curated stat surface (`docs/rolling_window_implementation.md`).

PROGRESS: Rolling/odds fixes and docs are done; roster/injury as-of gating and ps_* triage still pending for a later pass.

Tests: not run (not requested).

Git: Unable to stage/commit due to permission error creating `.git/index.lock` (touch failed with “Operation not permitted”). Please stage/commit manually once permissions allow.

================ END CODEX OUTPUT ================

[run_codex_exec] Finished. Exit code: 0
[phase_3_implement] Implementation complete.
[phase_pipeline_validation] Attempt 1/5

[run_codex_exec] Starting exec call... (sandbox=workspace-write)


================ PROMPT SENT TO CODEX ================

You are now in the validation loop. Your job is to make sure the full pipeline command runs
    successfully and produces refreshed metrics artifacts.

    Plan context (for reference only, do NOT re-hash it unless needed):

    <BEGIN_PLAN_START>
    OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf6b-4ad5-78e1-9036-ed3c08d53d2c
--------
user

    Okay, great work.

    Please come up with the richest and most detailed plan possible to address every one of the points
    you brought up in the best way possible.

    Here is your last analysis / set of findings:

    <BEGIN_ANALYSIS_START>
    OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf67-266d-7c53-b835-f4d18d9aac20
--------
user

Context about this project:

- This is a project that takes NFL play-by-play data and uses it for modeling so that we can
  predict which players are likely to get an anytime touchdown (rushing or receiving, not passing)
  in an upcoming game.

- The goal is to build features and models that are as close as possible to the underlying
  football reality that produces touchdowns: play-calling tendencies, player usage, game state,
  defensive matchups, injuries, roles, red-zone behavior, etc.

- Training featurization must conceptually and logically match prediction/inference featurization.
  Anything that can only be known in hindsight at inference time (future data, downstream labels,
  or derived artifacts that use future information) is a form of leakage and must be eliminated.

- Over time, previous model runs and refactors may have left behind:
    * partially-implemented ideas,
    * experimental code paths,
    * hallucinated features,
    * or confusing / inconsistent logic.
  DO NOT assume that all existing code, features, configs, or comments are intentional or correct
  just because they exist. Treat any piece of code or configuration that does not clearly make
  sense in the context of the project as a candidate for cleanup, simplification, or removal.

Your task in this step:

Please analyze the current state of this project (code, data flow, feature engineering, and modeling)
and let me know:

1. Where things conceptually are not implemented correctly or are conceptually off, given the goal of
   predicting anytime TDs in a way that matches how football is actually played.
2. Where the modeling or data flow could be brought closer to "reality" as it actually plays out
  on the field. The goal of getting closer to reality is entirely so that the model is more
  accurate and metrics like AUC or other evaluation metrics improve.
3. Any incomplete implementations, half-finished ideas, or abandoned experimental paths.
4. Any wrong or misleading implementations (especially where names / comments and actual behavior diverge).
5. Any future data leaking into the modeling or feature pipeline (anything that uses knowledge from
  after the prediction cut-off point, including label-derived features).
6. Any underlying data sources or features that appear to not be working at all, or are effectively
  noise / dead weight.
7. Any areas where it looks like a previous run of a model or tool hallucinated structure, concepts,
  or features that don't actually exist in the real data or problem domain.

You should:

- Be concrete and specific in your findings.
- Call out anything that looks like hallucinated or legacy cruft that should probably be removed or
  reworked, instead of assuming it must be intentional.
- Focus on how each issue you find ultimately affects model realism and predictive performance.

mcp startup: no servers

thinking
**Listing project files for analysis**
exec
/bin/zsh -lc ls in /Users/jameswest/Desktop/nfl_predict succeeded in 765ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
README.md
__pycache__
audit
cache
codex_meta_loop.py
config
data
docs
logs
main.py
output
pipeline
pyproject.toml
requirements.txt
scripts
tests
tmp.parquet
tmp_features.parquet
utils

thinking
**Considering reading README**
exec
/bin/zsh -lc "sed -n '1,200p' README.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 762ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Anytime TD Prediction

## Key Scripts
- `pipeline/feature.py` – builds the player-game feature matrix, enforcing as-of leakage guards.
- `pipeline/train.py` – trains the multi-stage modeling graph (availability → usage → efficiency → TD).
- `pipeline/predict.py` – runs inference with the latest artifacts.
- `scripts/eval/odds_backtest.py` – compute odds-only baselines across horizons.
- `scripts/backtest/run_backtest.py` – evaluate existing prediction columns (AUC/Brier/log-loss) and write baseline summaries.
- `scripts/backtest/compare_baselines.py` – diff two baseline JSON files and highlight regressions.
- `scripts/analysis/horizon_eval.py` – consolidate metrics across decision horizons (input multiple prediction files labelled by horizon).
- `scripts/analysis/weather_travel_impact.py` – stratify TD hit rates by weather/travel features and emit plots (`docs/analysis/weather_travel/`) + JSON summaries.
- `scripts/monitoring/asof_coverage_report.py` – summarize snapshot coverage, write Markdown to `docs/monitoring/asof_coverage.md`, and append metrics history to `output/metrics/coverage/history.parquet`.

## Monitoring
- `docs/monitoring.md` documents leak guards, snapshot coverage expectations, roster snapshot regeneration steps, and instructions for running odds/backtests & analytics scripts.
- `docs/analysis/weather_travel.md` lists the latest weather/travel diagnostic artifacts.

## Quick Start
1. Build features:

thinking
**Preparing multi-step plan**
Plan update
Map out review steps to assess feature pipeline, modeling flow, leakage risks, and dead code.
  → Scan project structure and key config/docs to understand intended data flow and leakage guards
  • Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering key module info**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Feature pipeline orchestrator for NFL player-level predictions.

Stages
------
1. **build_play_level**         – derive cleaned play-level slices
2. **build_player_drive_level** – aggregate per drive
3. **build_player_game_level**  – aggregate per game and join context

This wrapper triggers each stage, optionally enforces a schema after each
write, and returns the final game-level DataFrame for downstream usage.
"""

import logging
import os
from datetime import date, timedelta
from typing import Sequence


thinking
**Planning deeper analysis**
exec
/bin/zsh -lc 'rg "TODO|FIXME|TBD"' in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
docs/rolling_window_implementation.md:### Immediate (TODO #8)

thinking
**Checking rolling window docs**
exec
/bin/zsh -lc "sed -n '1,200p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Rolling Window Feature System - Implementation Summary

## Overview

Successfully implemented a comprehensive rolling window feature system for NFL player predictions. The system computes time-series statistics for player performance across different contexts and time horizons.

## Architecture

### 1. Data Flow

```
Cleaned Play Data
    ↓
Player-Game Aggregations (player_game_level.py)
    ↓
Daily Totals Cache (nfl_daily_totals.py)
    ↓
Rolling Window Features (nfl_rolling_window.py)
    ↓
exec
/bin/zsh -lc "sed -n '200,400p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- **Incremental updates**: Only new dates need processing
- **Fast lookups**: Direct parquet reads by date
- **Memory efficient**: Streaming operations for large datasets
- **Reusable**: Same cache serves all rolling window computations

### Scalability
- **Per-player parallelization**: Each player's features computed independently
- **Lazy evaluation**: Uses Polars lazy API where possible
- **Chunked processing**: Pipeline processes data in configurable chunks

## Next Steps

### Immediate (TODO #8)
1. **Run feature pipeline** with new stats and rolling windows
   ```bash
   python -c "from pipeline.feature import build_feature_matrix; \
              from datetime import date; \
              build_feature_matrix(start_date=date(2024, 9, 5), \
                                  end_date=date(2024, 9, 9))"

thinking
**Verifying documentation accuracy**
exec
/bin/zsh -lc 'ls utils/feature' in /Users/jameswest/Desktop/nfl_predict succeeded in 746ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
__init__.py
__pycache__
asof.py
asof_metadata.py
concatenate_parquets.py
daily_totals.py
drive_level.py
game_level.py
io.py
keygen.py
odds.py
offense_context.py
opponent_splits.py
pace.py
play_level.py
player_drive_level.py
player_game_level.py
player_market_cli.py
qb_profiles.py

thinking
**Inspecting statistics components**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/stats.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 719ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL player statistics configuration for rolling window features.

Defines the standard set of stats tracked at all aggregation levels
(play, drive, game) for rolling window computations.

This module is the SINGLE SOURCE OF TRUTH for stat definitions used
throughout the pipeline. Import from here rather than defining locally.
"""

from typing import List

# =============================================================================
# Core Stats (used for drive and game level aggregation)
# =============================================================================

# Basic stats aggregated at player-drive level
PLAYER_DRIVE_STATS: List[str] = [
    # Yardage
    "receiving_yards",

thinking
**Verifying feature data existence**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 739ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL Rolling Window Features - Direct port from MLB rolling_window.py

Uses the exact same vectorized Polars pattern as the working MLB implementation.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache/feature/daily_totals")
NFL_SEASON_START_MMDD = (9, 1)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                    rolling_feature_names.append(f"{w}g_{k}_per_{self.level}{suffix}")

        if batch_min_date is None or batch_max_date is None:
            logger.warning("RollingWindow.compute received empty dataset; returning null rolling features.")
            result = ds
            for feat in rolling_feature_names:
                result = result.with_columns(pl.lit(None).cast(pl.Float32).alias(feat))
            return result.drop("__ts")
        
        # Determine lookback for CACHE scan
        # Load cache INCLUDING the batch dates - cumsum logic prevents leakage
        # Each row's rolling features = cumsum(up to row) - row_value = history BEFORE row
        if "lifetime" in windows:
            earliest = self._earliest_cache_date()
            lo_hist = earliest if earliest else batch_min_date - timedelta(days=365 * 10)
        elif "season" in windows:
            season_start = self._season_anchor(batch_min_date)
            lo_hist = season_start if season_start < batch_min_date else batch_min_date - timedelta(days=365)
        else:
exec
/bin/zsh -lc "sed -n '400,800p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                            denom_use = den_window_cache.get(w_int)
                            if denom_use is None:
                                denom_use = np.zeros(n, dtype=np.float64)
                            if w_int <= n:
                                shifted_num = np.concatenate((np.zeros(w_int, dtype=np.float64), prev_num[:-w_int]))
                            else:
                                shifted_num = np.zeros(n, dtype=np.float64)
                            numer_use = prev_num - shifted_num
                        
                        values = np.full(n, np.nan, dtype=np.float32)
                        if denom_use is not None:
                            mask = denom_use > 0
                            if mask.any():
                                values[mask] = (numer_use[mask] / denom_use[mask]).astype(np.float32)
                        
                        feat_name = f"{w}g_{k}_per_{self.level}" + ("" if ctx == "vs_any" else f"_{ctx}")
                        feature_arrays[feat_name] = values
                
                data: dict[str, pl.Series | np.ndarray] = {

thinking
**Checking daily_totals module**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 747ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL daily totals cache for rolling window computations.

Builds daily aggregated stats per player-context combination:
- vs_any: All games for this player
- vs_team: Games vs specific opponent

Cache structure per day:
  player_id, date, ctx, opponent → stat_num columns
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Callable
import polars as pl
import logging

from utils.feature.stats import NFL_PLAYER_STATS

logger = logging.getLogger(__name__)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 734ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    # Load function: loads player data for a given date
    def load_player_data(day: date) -> pl.DataFrame:
        """Load player-level data for a specific date from weekly parquets."""
        try:
            # Scan all weekly parquets
            df = pl.scan_parquet(
                str(source_dir / "season=*/week=*/part.parquet"),
                glob=True,
                hive_partitioning=True,
                missing_columns="insert",
                extra_columns="ignore",
            )

            schema = df.collect_schema()
            categorical_cols = [
                name
                for name, dtype in zip(schema.names(), schema.dtypes())
                if "Categorical" in str(dtype)

thinking
**Inspecting training pipeline**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/train.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
import os
import random
import inspect
import sys
import logging
import math


# Move logging to top and remove global thread pinning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path to allow module imports when running as script
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import yaml
import joblib
import optuna

thinking
**Outlining investigation strategy**
exec
/bin/zsh -lc 'rg "build_feature_matrix" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
66:__all__ = ["build_feature_matrix", "refresh_context_histories"]
226:def _build_feature_matrix_internal(
953:def build_feature_matrix(
979:            df = _build_feature_matrix_internal(
exec
/bin/zsh -lc "sed -n '220,520p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 698ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    )

# ---------------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------------

def _build_feature_matrix_internal(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    chunk_days: int = 7,
    recompute_intermediate: bool = True,
    build_rolling: bool = True,
    output_path: Path | None = None,
    primary_output_path: Path | None = None,
    cutoff_label: str | None = None,
) -> pl.DataFrame:
    """Run full feature pipeline up to *inclusive* `end_date` for a single cutoff window.


thinking
**Analyzing key feature sections**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/targets.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 743ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Target column validation for NFL player prediction models.

Target columns are created in utils/feature/player_game_level.py during aggregation.
This module validates that required targets exist and have correct dtypes.
"""

import polars as pl
import logging

logger = logging.getLogger(__name__)


# Expected target columns and their types
NFL_TARGET_SCHEMA = {
    # Binary classification targets
    "anytime_td": pl.Int8,  # Primary MVP target: did player score a TD?
    
    # Count/regression targets
    "td_count": pl.Int64,  # Number of TDs scored (receiving + rushing)

thinking
**Checking relevant player game data**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Player-game level aggregation for NFL predictions.

Aggregates play-by-play data to player-game level for all skill positions.
Each player gets one row per game with their total stats.

Target columns created:
- anytime_td (binary): Did player score a TD (receiving or rushing)?
- td_count (int): Number of TDs scored (receiving + rushing)
- passing_td (int): Number of passing TDs (for QBs)
- receiving_yards (float): Total receiving yards
- rushing_yards (float): Total rushing yards
- passing_yards (float): Total passing yards
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import datetime as dt

thinking
**Analyzing potential data issues and feature integration**

I'm focusing on spotting key issues like data leakage or unproduced columns in the large player_game_level file by reviewing mentions of rolling_window, MLB patterns, and stats usage around add_rolling_features. I want to understand if features rely on missing caches or incorrect columns that might disrupt aggregation or inference.
exec
/bin/zsh -lc 'rg "add_rolling_features" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 753ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
31:from utils.feature.rolling_window import add_rolling_features
831:        df_player_game_all = add_rolling_features(
exec
/bin/zsh -lc "sed -n '760,880p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

    logging.info("Enriching odds features (NFL)...")
    rows_before_odds = df_player_game_all.height
    df_player_game_all = add_nfl_odds_features_to_df(
        df_player_game_all,
        player_col="player_name",
        allow_schedule_fallback=False,
        drop_schedule_rows=True,
    )
    
    logging.info("Enriching player props (Anytime TD)...")
    df_player_game_all = add_player_props_to_df(
        df_player_game_all,
        player_col="player_name",
    )

    rows_after_odds = df_player_game_all.height
    logging.info(
        "Odds features attached; retained %d/%d rows (%.2f%%) with pre-cutoff snapshots",

thinking
**Examining rolling context distinctions**
exec
/bin/zsh -lc "sed -n '880,1120p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        logging.info("Added %d historical share features", len(share_exprs))
    leakage_cols = [
        "target_share",
        "carry_share",
        "pass_attempt_share",
        "red_zone_target_share",
        "red_zone_carry_share",
        "goal_to_go_target_share",
        "goal_to_go_carry_share",
    ]
    df_player_game_all = df_player_game_all.drop(leakage_cols, strict=False)
    # Keep raw outcome columns (targets, carries, etc.) in the feature matrix so they can
    # serve as supervised targets. These columns are excluded from model inputs later via
    # training configuration, so we avoid dropping them here to retain label availability.

    available_cols = set(df_player_game_all.columns)
    market_exprs: list[pl.Expr] = []
    if {"team_implied_total", "hist_target_share_prev"} <= available_cols:
        market_exprs.append(

thinking
**Analyzing data consistency and leakage risks**

I'm examining key conceptual challenges in the rolling window features, especially the mismatch between expected and available stats columns, which may create noise from zero-filled features. I'm also checking for potential data leakage from using caches that include same-day or future game data, ensuring that feature computations properly exclude current or future information during training and inference.

thinking
**Analyzing potential data leakage in feature computation**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
Consolidated NFL Odds Module
============================

This module provides all odds-related functionality for the NFL prediction pipeline:

1. OddsAPIClient - HTTP client for The Odds API with caching and retry logic
2. Game-level odds features (moneyline, spread, totals) from nfl_odds.py
3. Player-level odds features (anytime TD props) from odds_features.py
4. Odds snapshot collection and persistence from odds_snapshots.py

All odds logic is consolidated here to prevent drift and ensure consistency.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

thinking
**Analyzing rolling window and feature pipeline**

I’m digging into the rolling window implementation and its interaction with feature caches and pipelines. There are questions about how date ranges and future game predictions align, especially if the pipeline uses actual play data that wouldn’t exist yet for upcoming games. I want to verify how the prediction pipeline handles schedules and odds versus the completed game data the feature builds rely on. Let’s explore pipeline/predict next!
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/predict.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
NFL prediction pipeline
=======================

Generates anytime-touchdown probabilities for upcoming games by:

1. Loading the upcoming schedule and weekly rosters (via nfl_data_py)
2. Building a player-level scaffold for offensive skill positions
3. Enriching that scaffold with rolling-window features using the same
   caches that power the training pipeline
4. Sequentially running the model chain:
   Availability -> Team Pace -> Usage -> Efficiency -> Anytime TD
5. Writing an actionable slate under `output/predictions/`
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

thinking
**Analyzing feature computation inconsistencies**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_experiments.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 752ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

from typing import Iterable

import polars as pl


def add_experimental_rolling_features(
    df: pl.DataFrame,
    stats: Iterable[str],
    *,
    epsilon: float = 1e-3,
) -> pl.DataFrame:
    """
    Derive experimental rolling-window features to capture burst performance,
    exponential-style recency weighting, and directional trends.

    Parameters
    ----------

thinking
**Checking dataset relevance and usage**
exec
/bin/zsh -lc 'rg "ps_route_participation"' in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
pipeline/predict.py:    "ps_route_participation_plays",
pipeline/predict.py:    "ps_route_participation_pct",
pipeline/predict.py:    "ps_route_participation_pct_prev",
pipeline/predict.py:    "ps_route_participation_pct_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3": "ps_route_participation_plays",
pipeline/predict.py:        "1g_ps_route_participation_pct_per_game": "ps_route_participation_pct",
pipeline/predict.py:        "3g_ps_route_participation_pct_per_game": "ps_route_participation_pct_l3",
pipeline/predict.py:        "1g_ps_route_participation_plays_per_game": "ps_route_participation_plays",
pipeline/predict.py:        "3g_ps_route_participation_plays_per_game": "ps_route_participation_plays_l3",
pipeline/predict.py:        "ps_route_participation_pct",
pipeline/predict.py:        "ps_route_participation_plays",
utils/feature/stats.py:    "ps_route_participation_pct",
utils/feature/stats.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:        .agg(pl.len().alias("ps_route_participation_plays"))
utils/feature/player_game_level.py:            .then(pl.col("ps_route_participation_plays") / pl.col("ps_team_dropbacks"))

thinking
**Investigating player_game_level and anytime_td**
exec
/bin/zsh -lc 'rg "anytime_td" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 777ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- anytime_td (binary): Did player score a TD (receiving or rushing)?
    "market_anytime_td_prob",
    "market_anytime_td_consensus",
    "market_anytime_td_spread",
    "market_anytime_td_book_count",
    "market_anytime_td_prob_2h",
    "market_anytime_td_consensus_2h",
    "market_anytime_td_spread_2h",
    "market_anytime_td_book_count_2h",
    "market_anytime_td_prob_6h",
    "market_anytime_td_consensus_6h",
    "market_anytime_td_spread_6h",
    "market_anytime_td_book_count_6h",
    "market_anytime_td_prob_24h",
    "market_anytime_td_consensus_24h",
    "market_anytime_td_spread_24h",
    "market_anytime_td_book_count_24h",
    "market_anytime_td_prob_open",
    "market_anytime_td_consensus_open",

thinking
**Analyzing potential data leakage and timing issues**

I'm piecing together key concerns about possible data leakage in the rolling window system and training pipeline, especially related to future data inclusion, odds snapshot timing, and injury predictions. I'll highlight complications with caching and as-of joins that may cause leakage or inconsistencies.

thinking
**Evaluating data leakage risks in caches**

I'm thinking about whether cached stats might include data from invalid or future games, potentially causing leakage in training or inference. I need to check if features like offense_context have discrepancies between their training and inference implementations that might lead to mismatches or leaks.
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/offense_context.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 788ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import polars as pl

logger = logging.getLogger(__name__)

COORD_DEFAULT = "UNKNOWN_COORDINATOR"
QB_DEFAULT = "UNKNOWN_QB"
EPS = 1e-3

COORDINATOR_MAP_PATH = Path("data/processed/offensive_coordinators.parquet")


def _build_is_unavailable_expr(df: pl.DataFrame) -> pl.Expr:
    """Return a boolean expression marking players ruled out before cutoff."""

thinking
**Reviewing touchdown definitions for label leakage**
exec
/bin/zsh -lc 'rg -n "touchdowns" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 738ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
803:        pl.lit(0).cast(pl.Int64).alias("touchdowns"),
2577:            "touchdowns",
3108:              .alias("touchdowns"),
3121:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3123:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3165:              .alias("touchdowns"),
3177:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3178:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3220:              .alias("touchdowns"),
3232:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3233:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3271:            pl.col("touchdowns").fill_null(0).sum().alias("touchdowns"),
3284:        "passing_td", "rushing_td_count", "receiving_td_count", "touchdowns",
3293:    # derive it from rushing + receiving touchdowns.
3295:        pl.col("touchdowns").fill_null(0).alias("touchdowns"),
3306:        pl.when(pl.col("touchdowns") > 0)
3307:        .then(pl.col("touchdowns"))
3310:        .alias("_total_touchdowns")
3314:        (pl.col("_total_touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),

thinking
**Inspecting final assembly logic**
exec
/bin/zsh -lc "sed -n '3240,3320p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 771ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    Example: QB who passed and rushed, WR who received and rushed.
    """
    
    # Group by (player, game) and sum all stats
    merged = (
        df
        .group_by(["season", "week", "game_id", "game_date", "player_id", "player_name"])
        .agg([
            # Sum all yardage
            pl.col("passing_yards").fill_null(0).sum().alias("passing_yards"),
            pl.col("rushing_yards").fill_null(0).sum().alias("rushing_yards"),
            pl.col("receiving_yards").fill_null(0).sum().alias("receiving_yards"),
            
            # Sum counts
            pl.col("pass_attempt").fill_null(0).sum().alias("pass_attempt"),
            pl.col("completion").fill_null(0).sum().alias("completion"),
            pl.col("carry").fill_null(0).sum().alias("carry"),
            pl.col("target").fill_null(0).sum().alias("target"),

thinking
**Examining touchdown aggregation logic**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Play-level feature builder.

Reads cleaned daily Parquets and writes weekly play-level Parquets at
  data/processed/play_by_week/season=YYYY/week=W/part.parquet
"""

from pathlib import Path
from datetime import date
import polars as pl

CLEAN_DIR   = Path("data/cleaned")
PLAY_OUTDIR = Path("data/processed/play_by_week")

PLAY_COLS = [
    # Game identifiers
    "game_id",
    "season",

thinking
**Continuing review of derived statistics**
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 758ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
            ),
            (
                pl.col("defense_man_zone_type").cast(pl.Utf8).alias("defense_man_zone_type")
                if "defense_man_zone_type" in df.columns
                else pl.lit(None).alias("defense_man_zone_type")
            ),
            (
                pl.col("defense_coverage_type").cast(pl.Utf8).alias("defense_coverage_type")
                if "defense_coverage_type" in df.columns
                else pl.lit(None).alias("defense_coverage_type")
            ),
        ]
    )

    situ_exprs: list[pl.Expr] = []
    score_diff = pl.col("score_differential").fill_null(0.0)
    ydstogo = pl.col("ydstogo").fill_null(0.0)
    yardline = pl.col("yardline_100").cast(pl.Float32).fill_null(100.0)
    half_secs_expr = (

thinking
**Examining touchdown counting and rolling window concerns**

thinking
**Reviewing data shifts and plan progress**
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  → Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering details on config and training**
exec
/bin/zsh -lc "sed -n '1,200p' config/training.yaml" in /Users/jameswest/Desktop/nfl_predict succeeded in 815ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# config/training.yaml
# Configuration for the model training pipeline (pipeline/train.py)
# NFL Player Prediction Targets

# --- Data I/O ---
data:
  raw_pbp_dir: 'data/raw/pbp_by_day'  # NFL play-by-play raw data
  feature_matrix_path: 'data/processed/final/processed.parquet'  # NFL feature matrix
  
  # Directory to save trained model artifacts (models and feature lists).
  model_output_dir: 'output/models'
  
  # Directory to save evaluation reports and metrics.
  metrics_output_dir: 'output/metrics/'

# =============================================================================
# SHARED COLUMN DEFINITIONS (using YAML anchors to avoid duplication)
# =============================================================================
# These columns leak same-game outcomes and must be excluded from training.

thinking
**Analyzing potential data inconsistencies and leakage**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/asof.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 2.75s:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Helpers for aligning feature generation to a consistent decision-time snapshot."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Mapping

import yaml

from utils.general.paths import PROJ_ROOT


@lru_cache(maxsize=1)
def _load_cutoff_config() -> Mapping[str, Any]:
    """Load decision cutoff configuration from config/config.yaml."""
    config_path = PROJ_ROOT / "config" / "config.yaml"
    try:

thinking
**Identifying discrepancies in feature construction**

I'm analyzing several potential issues in how features are constructed, especially around rolling window computations, drive-level feature usage, pre-snap route participation data reliability, and labeling choices. I want to assemble a clear list of 6-7 concrete findings summarizing inconsistencies, incomplete implementations, and data limitations.

thinking
**Analyzing potential data leakage and pipeline inconsistencies**

I'm exploring possible data leakage issues, especially around using full-season roster and injury data without proper as-of gating in predictions. I’m also checking whether training pipelines enforce the claimed multi-stage model architecture and if various feature-building steps apply correct cutoff filters, since odds features might not respect decision cutoff timestamps, risking leakage. This involves reviewing training, feature, and monitoring docs alongside function implementations to understand the conceptual gaps and inconsistencies.

thinking
**Searching for add_player_odds_features_to_df function**
exec
/bin/zsh -lc 'rg "def add_player" -n utils/feature/odds.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 926ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
1099:def add_player_odds_features_to_df(
exec
/bin/zsh -lc "sed -n '1099,1250p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 732ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
def add_player_odds_features_to_df(
    df_pl: pl.DataFrame, *, api_key: str | None = None, player_col: str = "player_name",
    per_game_workers: int = 4, verbose: bool = False,
) -> pl.DataFrame:
    """Attach player odds features directly to an in-memory Polars DataFrame."""
    from utils.general.config import get_odds_api_key
    
    if df_pl.is_empty():
        return df_pl
    
    if not api_key:
        api_key = get_odds_api_key()
    
    if not api_key:
        logger.info("add_player_odds_features_to_df: no odds_api_key configured – adding empty columns and returning")
        empty_exprs = [pl.lit(None).cast(pl.Float32).alias(c) for c in PLAYER_ODDS_NUMERIC_COLS]
        flag_expr = [pl.lit(0).cast(pl.Int8).alias("odds_expected")]
        df_base = df_pl.drop(PLAYER_ODDS_NUMERIC_COLS + ODDS_FLAG_COLUMNS, strict=False)
        return df_base.with_columns(empty_exprs + flag_expr)

thinking
**Analyzing data leakage and feature completeness**

thinking
**Reviewing data leakage and caching inconsistencies**

I'm noticing potential data leakage in feature functions using live API calls and uncertain odds snapshot timing. There's also a mismatch between documented and actual cache directories that could confuse or cause bugs. I want to gather all issues before moving from feature pipeline review to modeling validation, paying attention to training config settings and columns flagged as leak-prone.
exec
/bin/zsh -lc 'rg "LEAK_PRONE_COLUMNS" -n utils' in /Users/jameswest/Desktop/nfl_predict succeeded in 870ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
utils/general/__init__.py:18:    LEAK_PRONE_COLUMNS,
utils/general/__init__.py:47:    "LEAK_PRONE_COLUMNS",
utils/general/constants.py:8:config/training.yaml should mirror LEAK_PRONE_COLUMNS for YAML-based configuration,
utils/general/constants.py:113:LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
utils/general/constants.py:208:    "LEAK_PRONE_COLUMNS",
exec
/bin/zsh -lc "sed -n '100,160p' utils/general/constants.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 800ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
}


# =============================================================================
# Leak-Prone Columns (Same-Game Outcomes)
# =============================================================================
# Stats that leak same-game outcomes and must be excluded from training features.
# These represent what happened in the game we're trying to predict.
#
# This is the AUTHORITATIVE definition. config/training.yaml mirrors this list
# using YAML anchors for per-problem configuration. Keep them in sync!
# Validated by: tests/test_training_config_features.py

LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
    # Yardage (same-game outcomes)
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    # Usage counts (same-game outcomes)

thinking
**Analyzing modeling pipeline inconsistencies**

I’m reviewing the training pipeline for possible incomplete or risky implementations—especially around derived targets, odds features that may leak future info, rolling window system mismatches, and caching issues causing potential feature inaccuracies. I'll note risky spots with specifics to file paths and behaviors to highlight data leakage or half-finished components.

thinking
**Identifying potential data leakage in features and inference**

thinking
**Analyzing touchdown column naming mismatch**

thinking
**Analyzing touchdown feature inconsistencies**

I notice a mismatch between "touchdown" and "touchdowns" columns in different dataframes, causing rolling touchdown features to be always zero—this looks like a conceptual bug. Also, the way stats are counted might cause double counting or misalignment. I want to fix these naming and aggregation issues and then update the project plan to mark step 2 as done and step 3 as in progress.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  → Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Analyzing model training and feature inconsistencies**

thinking
**Evaluating data quality and documentation gaps**

I'm analyzing issues with sparse or missing data affecting model training, especially around pre-snap route participation features and roster/injury alignment leaking future info. I also notice documentation inconsistencies in rolling window implementation and potential feature instability due to network-dependent API calls, all of which could harm predictive reliability and require careful handling in the next steps.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  ✓ Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Preparing final summary and suggestions**

I'll create a clear bullet list of findings ordered by severity with approximate file paths and lines, highlighting key issues like touchdown rolling features and pre-snap usage, then follow with numbered next steps for the user to consider.
codex
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.
tokens used
110,897
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.

    <END_ANALYSIS_END>

    This step is PURELY research, investigation, and planning.
    Do NOT make any edits to the code or data in this step.

    I want:
    - A structured, prioritized plan.
    - Clear steps that can be implemented by an agent in later steps.
    - Notes on risk or potential pitfalls where relevant, but focus on high-value changes.
    - Explicit attention to:
        * eliminating data leakage,
        * aligning features with real football mechanisms that drive anytime TDs,
        * cleaning up hallucinated / legacy cruft that no longer makes sense.

    Again: no edits here, just the plan.

mcp startup: no servers

thinking
**Drafting a structured planning outline**
codex
**Remediation Plan (ordered by impact & leakage risk)**
- Fix odds leakage/hallucination: freeze player prop odds to archived pre-cutoff snapshots; add snapshot loader keyed by game_id/team/player + snapshot_ts; in training, drop/flag rows without snapshots; in inference, refuse to call live API, optionally fall back to schedule-only baseline; add tests to assert no odds columns present when snapshot missing.
- Repair rolling TD history: align stat key `touchdowns` across `stats.py`, aggregation, and cache; rebuild daily_totals cache; add schema check to fail if any rolling TD feature is all-null/zero; rerun backtests to measure regained signal.
- Trim rolling window surface to supported set: codify a vetted set of stats/windows/contexts (e.g., {1g,3g,5g,season} × {vs_any}); remove ps_* and other missing-cache stats from rolling requests; update docs to match; add runtime guard to error on requested stats not found in cache.
- Enforce as-of gating for roster/injury inputs in prediction/backtests: add decision-time cutoff (e.g., T-6h) and snapshot source; if unavailable, drop those features or mark unavailable; add regression test ensuring no injury status is from after game start in historical inference.
- Remove/retire ps_* targets/features if coverage is low: measure non-null rate by season; if sparse, drop from training configs and rolling stats; otherwise backfill realistically or scope to seasons with data; add check to block tasks with <X% coverage.
- Clarify/implement drive-level history or delete: either build drive-level daily cache and wire into rolling computation, or remove drive-context flags/constants/docs; add assertion preventing “drive” level when cache absent.
- Align docs with reality and guard drift: sync paths/names (cache/feature/daily_totals vs nfl_daily_totals); document the final rolling feature list and cutoff rules; add a pre-flight validation script that checks cache schemas, non-null rates, and leakage-sensitive columns before training.

**Execution Steps**
1) Odds snapshots: design schema + loader; replace `add_player_props_to_df` call; add fallback behavior and tests; rerun feature build on a small date range to validate columns.
2) Rolling TD fix: rename stat key, rebuild cache for a sample window, ensure rolling TD features populate; update tests.
3) Rolling feature pruning: define allowed stats/windows/contexts; update config/constants + pipeline call; add guard for missing stats; rerun feature build and inspect null/constant rates.
4) Roster/injury gating: add cutoff param in predict/backtest; wire to snapshot source; add assertion that injury_status_ts <= cutoff; adjust backtests to use gated features.
5) ps_* triage: compute coverage by season; decide keep/drop; update training.yaml and problem definitions; if dropped, remove from rolling stats and tasks.
6) Drive-level decision: either implement cache builder (drive totals) and wire into rolling_window or delete related flags/docs; add guard to prevent silent no-op.
7) Docs/validation: update rolling_window_implementation.md and monitoring docs; add validation script to check schemas, coverage, leak-prone columns excluded from train set; run it in CI/pre-flight.

**Risks/Pitfalls to watch**
- Cache rebuild time/size; may need incremental rebuild and date filters.
- Snapshot availability gaps; must define strict fallbacks to avoid silent leakage.
- Training/inference drift if odds/roster data are absent—prefer hard failure over silent nulls.
- Removing ps_* may change model interfaces; update downstream consumers/tests accordingly.
tokens used
15,743
**Remediation Plan (ordered by impact & leakage risk)**
- Fix odds leakage/hallucination: freeze player prop odds to archived pre-cutoff snapshots; add snapshot loader keyed by game_id/team/player + snapshot_ts; in training, drop/flag rows without snapshots; in inference, refuse to call live API, optionally fall back to schedule-only baseline; add tests to assert no odds columns present when snapshot missing.
- Repair rolling TD history: align stat key `touchdowns` across `stats.py`, aggregation, and cache; rebuild daily_totals cache; add schema check to fail if any rolling TD feature is all-null/zero; rerun backtests to measure regained signal.
- Trim rolling window surface to supported set: codify a vetted set of stats/windows/contexts (e.g., {1g,3g,5g,season} × {vs_any}); remove ps_* and other missing-cache stats from rolling requests; update docs to match; add runtime guard to error on requested stats not found in cache.
- Enforce as-of gating for roster/injury inputs in prediction/backtests: add decision-time cutoff (e.g., T-6h) and snapshot source; if unavailable, drop those features or mark unavailable; add regression test ensuring no injury status is from after game start in historical inference.
- Remove/retire ps_* targets/features if coverage is low: measure non-null rate by season; if sparse, drop from training configs and rolling stats; otherwise backfill realistically or scope to seasons with data; add check to block tasks with <X% coverage.
- Clarify/implement drive-level history or delete: either build drive-level daily cache and wire into rolling computation, or remove drive-context flags/constants/docs; add assertion preventing “drive” level when cache absent.
- Align docs with reality and guard drift: sync paths/names (cache/feature/daily_totals vs nfl_daily_totals); document the final rolling feature list and cutoff rules; add a pre-flight validation script that checks cache schemas, non-null rates, and leakage-sensitive columns before training.

**Execution Steps**
1) Odds snapshots: design schema + loader; replace `add_player_props_to_df` call; add fallback behavior and tests; rerun feature build on a small date range to validate columns.
2) Rolling TD fix: rename stat key, rebuild cache for a sample window, ensure rolling TD features populate; update tests.
3) Rolling feature pruning: define allowed stats/windows/contexts; update config/constants + pipeline call; add guard for missing stats; rerun feature build and inspect null/constant rates.
4) Roster/injury gating: add cutoff param in predict/backtest; wire to snapshot source; add assertion that injury_status_ts <= cutoff; adjust backtests to use gated features.
5) ps_* triage: compute coverage by season; decide keep/drop; update training.yaml and problem definitions; if dropped, remove from rolling stats and tasks.
6) Drive-level decision: either implement cache builder (drive totals) and wire into rolling_window or delete related flags/docs; add guard to prevent silent no-op.
7) Docs/validation: update rolling_window_implementation.md and monitoring docs; add validation script to check schemas, coverage, leak-prone columns excluded from train set; run it in CI/pre-flight.

**Risks/Pitfalls to watch**
- Cache rebuild time/size; may need incremental rebuild and date filters.
- Snapshot availability gaps; must define strict fallbacks to avoid silent leakage.
- Training/inference drift if odds/roster data are absent—prefer hard failure over silent nulls.
- Removing ps_* may change model interfaces; update downstream consumers/tests accordingly.

    <END_PLAN_END>

    Validation requirements:
    - Capture a baseline snapshot before running any new code:
      * Record `auc`, `pr_auc`, `brier_score`, and `precision_at_thresh` from `/Users/jameswest/Desktop/nfl_predict/output/metrics/anytime_td/xgboost/20251126_001001_ab12e68_cutoff_h090m/metrics.yaml`.
      * Note the current feature-importance distribution from `/Users/jameswest/Desktop/nfl_predict/output/metrics/anytime_td/xgboost/20251126_001001_ab12e68_cutoff_h090m/feature_importance.json`.
    - Run `cd /Users/jameswest/Desktop/nfl_predict && /Users/jameswest/Desktop/nfl_predict/venv/bin/python /Users/jameswest/Desktop/nfl_predict/main.py` to rebuild datasets, retrain models, and regenerate predictions.
    - After the run completes, identify the newest timestamped directory under `/Users/jameswest/Desktop/nfl_predict/output/metrics/anytime_td/xgboost` and repeat the metric collection.
      * Compare before/after metrics and explicitly call out improvements or regressions.
      * Re-review the refreshed `feature_importance.json` for any unexpected signal shifts.
    - Inspect `cutoff_backtest_summary.csv` (e.g., `/Users/jameswest/Desktop/nfl_predict/output/metrics/cutoff_backtest_summary.csv`) to confirm horizon-level hit rates and calibration remain acceptable.
    - Only declare success if the post-run metrics improve (higher AUC/PR AUC, lower Brier/log loss) or, at minimum, hold steady with a justified explanation. Any regression must be reported.

    Rules for this loop:
    - You may inspect and edit code, rerun commands, or add logging as needed to fix any issues uncovered
      while running `cd /Users/jameswest/Desktop/nfl_predict && /Users/jameswest/Desktop/nfl_predict/venv/bin/python /Users/jameswest/Desktop/nfl_predict/main.py`.
    - Keep commits incremental. If you end up making larger fixes, describe them clearly.
    - After each attempt, report status using `PROGRESS:`. If the run fails, capture stack traces / log
      pointers so the next attempt knows what to fix.
    - Once you have successfully run the pipeline, collected the new metrics, and compared them against
      the previous baseline, end your response with the exact line:
          PIPELINE VALIDATION COMPLETE
      Include the usual summary before that sentinel line.
    - Do NOT output the sentinel unless the metrics truly exist and have been evaluated.


======================================================


================ CODEX OUTPUT (streaming) ================

OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf71-64d3-7641-b824-a1c6f51a436a
--------
user
You are now in the validation loop. Your job is to make sure the full pipeline command runs
    successfully and produces refreshed metrics artifacts.

    Plan context (for reference only, do NOT re-hash it unless needed):

    <BEGIN_PLAN_START>
    OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf6b-4ad5-78e1-9036-ed3c08d53d2c
--------
user

    Okay, great work.

    Please come up with the richest and most detailed plan possible to address every one of the points
    you brought up in the best way possible.

    Here is your last analysis / set of findings:

    <BEGIN_ANALYSIS_START>
    OpenAI Codex v0.63.0 (research preview)
--------
workdir: /Users/jameswest/Desktop/nfl_predict
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: none
reasoning summaries: auto
session id: 019abf67-266d-7c53-b835-f4d18d9aac20
--------
user

Context about this project:

- This is a project that takes NFL play-by-play data and uses it for modeling so that we can
  predict which players are likely to get an anytime touchdown (rushing or receiving, not passing)
  in an upcoming game.

- The goal is to build features and models that are as close as possible to the underlying
  football reality that produces touchdowns: play-calling tendencies, player usage, game state,
  defensive matchups, injuries, roles, red-zone behavior, etc.

- Training featurization must conceptually and logically match prediction/inference featurization.
  Anything that can only be known in hindsight at inference time (future data, downstream labels,
  or derived artifacts that use future information) is a form of leakage and must be eliminated.

- Over time, previous model runs and refactors may have left behind:
    * partially-implemented ideas,
    * experimental code paths,
    * hallucinated features,
    * or confusing / inconsistent logic.
  DO NOT assume that all existing code, features, configs, or comments are intentional or correct
  just because they exist. Treat any piece of code or configuration that does not clearly make
  sense in the context of the project as a candidate for cleanup, simplification, or removal.

Your task in this step:

Please analyze the current state of this project (code, data flow, feature engineering, and modeling)
and let me know:

1. Where things conceptually are not implemented correctly or are conceptually off, given the goal of
   predicting anytime TDs in a way that matches how football is actually played.
2. Where the modeling or data flow could be brought closer to "reality" as it actually plays out
  on the field. The goal of getting closer to reality is entirely so that the model is more
  accurate and metrics like AUC or other evaluation metrics improve.
3. Any incomplete implementations, half-finished ideas, or abandoned experimental paths.
4. Any wrong or misleading implementations (especially where names / comments and actual behavior diverge).
5. Any future data leaking into the modeling or feature pipeline (anything that uses knowledge from
  after the prediction cut-off point, including label-derived features).
6. Any underlying data sources or features that appear to not be working at all, or are effectively
  noise / dead weight.
7. Any areas where it looks like a previous run of a model or tool hallucinated structure, concepts,
  or features that don't actually exist in the real data or problem domain.

You should:

- Be concrete and specific in your findings.
- Call out anything that looks like hallucinated or legacy cruft that should probably be removed or
  reworked, instead of assuming it must be intentional.
- Focus on how each issue you find ultimately affects model realism and predictive performance.

mcp startup: no servers

thinking
**Listing project files for analysis**
exec
/bin/zsh -lc ls in /Users/jameswest/Desktop/nfl_predict succeeded in 765ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
README.md
__pycache__
audit
cache
codex_meta_loop.py
config
data
docs
logs
main.py
output
pipeline
pyproject.toml
requirements.txt
scripts
tests
tmp.parquet
tmp_features.parquet
utils

thinking
**Considering reading README**
exec
/bin/zsh -lc "sed -n '1,200p' README.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 762ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Anytime TD Prediction

## Key Scripts
- `pipeline/feature.py` – builds the player-game feature matrix, enforcing as-of leakage guards.
- `pipeline/train.py` – trains the multi-stage modeling graph (availability → usage → efficiency → TD).
- `pipeline/predict.py` – runs inference with the latest artifacts.
- `scripts/eval/odds_backtest.py` – compute odds-only baselines across horizons.
- `scripts/backtest/run_backtest.py` – evaluate existing prediction columns (AUC/Brier/log-loss) and write baseline summaries.
- `scripts/backtest/compare_baselines.py` – diff two baseline JSON files and highlight regressions.
- `scripts/analysis/horizon_eval.py` – consolidate metrics across decision horizons (input multiple prediction files labelled by horizon).
- `scripts/analysis/weather_travel_impact.py` – stratify TD hit rates by weather/travel features and emit plots (`docs/analysis/weather_travel/`) + JSON summaries.
- `scripts/monitoring/asof_coverage_report.py` – summarize snapshot coverage, write Markdown to `docs/monitoring/asof_coverage.md`, and append metrics history to `output/metrics/coverage/history.parquet`.

## Monitoring
- `docs/monitoring.md` documents leak guards, snapshot coverage expectations, roster snapshot regeneration steps, and instructions for running odds/backtests & analytics scripts.
- `docs/analysis/weather_travel.md` lists the latest weather/travel diagnostic artifacts.

## Quick Start
1. Build features:

thinking
**Preparing multi-step plan**
Plan update
Map out review steps to assess feature pipeline, modeling flow, leakage risks, and dead code.
  → Scan project structure and key config/docs to understand intended data flow and leakage guards
  • Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering key module info**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Feature pipeline orchestrator for NFL player-level predictions.

Stages
------
1. **build_play_level**         – derive cleaned play-level slices
2. **build_player_drive_level** – aggregate per drive
3. **build_player_game_level**  – aggregate per game and join context

This wrapper triggers each stage, optionally enforces a schema after each
write, and returns the final game-level DataFrame for downstream usage.
"""

import logging
import os
from datetime import date, timedelta
from typing import Sequence


thinking
**Planning deeper analysis**
exec
/bin/zsh -lc 'rg "TODO|FIXME|TBD"' in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
docs/rolling_window_implementation.md:### Immediate (TODO #8)

thinking
**Checking rolling window docs**
exec
/bin/zsh -lc "sed -n '1,200p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# NFL Rolling Window Feature System - Implementation Summary

## Overview

Successfully implemented a comprehensive rolling window feature system for NFL player predictions. The system computes time-series statistics for player performance across different contexts and time horizons.

## Architecture

### 1. Data Flow

```
Cleaned Play Data
    ↓
Player-Game Aggregations (player_game_level.py)
    ↓
Daily Totals Cache (nfl_daily_totals.py)
    ↓
Rolling Window Features (nfl_rolling_window.py)
    ↓
exec
/bin/zsh -lc "sed -n '200,400p' docs/rolling_window_implementation.md" in /Users/jameswest/Desktop/nfl_predict succeeded in 742ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- **Incremental updates**: Only new dates need processing
- **Fast lookups**: Direct parquet reads by date
- **Memory efficient**: Streaming operations for large datasets
- **Reusable**: Same cache serves all rolling window computations

### Scalability
- **Per-player parallelization**: Each player's features computed independently
- **Lazy evaluation**: Uses Polars lazy API where possible
- **Chunked processing**: Pipeline processes data in configurable chunks

## Next Steps

### Immediate (TODO #8)
1. **Run feature pipeline** with new stats and rolling windows
   ```bash
   python -c "from pipeline.feature import build_feature_matrix; \
              from datetime import date; \
              build_feature_matrix(start_date=date(2024, 9, 5), \
                                  end_date=date(2024, 9, 9))"

thinking
**Verifying documentation accuracy**
exec
/bin/zsh -lc 'ls utils/feature' in /Users/jameswest/Desktop/nfl_predict succeeded in 746ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
__init__.py
__pycache__
asof.py
asof_metadata.py
concatenate_parquets.py
daily_totals.py
drive_level.py
game_level.py
io.py
keygen.py
odds.py
offense_context.py
opponent_splits.py
pace.py
play_level.py
player_drive_level.py
player_game_level.py
player_market_cli.py
qb_profiles.py

thinking
**Inspecting statistics components**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/stats.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 719ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL player statistics configuration for rolling window features.

Defines the standard set of stats tracked at all aggregation levels
(play, drive, game) for rolling window computations.

This module is the SINGLE SOURCE OF TRUTH for stat definitions used
throughout the pipeline. Import from here rather than defining locally.
"""

from typing import List

# =============================================================================
# Core Stats (used for drive and game level aggregation)
# =============================================================================

# Basic stats aggregated at player-drive level
PLAYER_DRIVE_STATS: List[str] = [
    # Yardage
    "receiving_yards",

thinking
**Verifying feature data existence**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 739ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL Rolling Window Features - Direct port from MLB rolling_window.py

Uses the exact same vectorized Polars pattern as the working MLB implementation.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache/feature/daily_totals")
NFL_SEASON_START_MMDD = (9, 1)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                    rolling_feature_names.append(f"{w}g_{k}_per_{self.level}{suffix}")

        if batch_min_date is None or batch_max_date is None:
            logger.warning("RollingWindow.compute received empty dataset; returning null rolling features.")
            result = ds
            for feat in rolling_feature_names:
                result = result.with_columns(pl.lit(None).cast(pl.Float32).alias(feat))
            return result.drop("__ts")
        
        # Determine lookback for CACHE scan
        # Load cache INCLUDING the batch dates - cumsum logic prevents leakage
        # Each row's rolling features = cumsum(up to row) - row_value = history BEFORE row
        if "lifetime" in windows:
            earliest = self._earliest_cache_date()
            lo_hist = earliest if earliest else batch_min_date - timedelta(days=365 * 10)
        elif "season" in windows:
            season_start = self._season_anchor(batch_min_date)
            lo_hist = season_start if season_start < batch_min_date else batch_min_date - timedelta(days=365)
        else:
exec
/bin/zsh -lc "sed -n '400,800p' utils/feature/rolling_window.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 740ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
                            denom_use = den_window_cache.get(w_int)
                            if denom_use is None:
                                denom_use = np.zeros(n, dtype=np.float64)
                            if w_int <= n:
                                shifted_num = np.concatenate((np.zeros(w_int, dtype=np.float64), prev_num[:-w_int]))
                            else:
                                shifted_num = np.zeros(n, dtype=np.float64)
                            numer_use = prev_num - shifted_num
                        
                        values = np.full(n, np.nan, dtype=np.float32)
                        if denom_use is not None:
                            mask = denom_use > 0
                            if mask.any():
                                values[mask] = (numer_use[mask] / denom_use[mask]).astype(np.float32)
                        
                        feat_name = f"{w}g_{k}_per_{self.level}" + ("" if ctx == "vs_any" else f"_{ctx}")
                        feature_arrays[feat_name] = values
                
                data: dict[str, pl.Series | np.ndarray] = {

thinking
**Checking daily_totals module**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 747ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""NFL daily totals cache for rolling window computations.

Builds daily aggregated stats per player-context combination:
- vs_any: All games for this player
- vs_team: Games vs specific opponent

Cache structure per day:
  player_id, date, ctx, opponent → stat_num columns
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Callable
import polars as pl
import logging

from utils.feature.stats import NFL_PLAYER_STATS

logger = logging.getLogger(__name__)
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/daily_totals.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 734ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    # Load function: loads player data for a given date
    def load_player_data(day: date) -> pl.DataFrame:
        """Load player-level data for a specific date from weekly parquets."""
        try:
            # Scan all weekly parquets
            df = pl.scan_parquet(
                str(source_dir / "season=*/week=*/part.parquet"),
                glob=True,
                hive_partitioning=True,
                missing_columns="insert",
                extra_columns="ignore",
            )

            schema = df.collect_schema()
            categorical_cols = [
                name
                for name, dtype in zip(schema.names(), schema.dtypes())
                if "Categorical" in str(dtype)

thinking
**Inspecting training pipeline**
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/train.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
import os
import random
import inspect
import sys
import logging
import math


# Move logging to top and remove global thread pinning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path to allow module imports when running as script
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import yaml
import joblib
import optuna

thinking
**Outlining investigation strategy**
exec
/bin/zsh -lc 'rg "build_feature_matrix" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
66:__all__ = ["build_feature_matrix", "refresh_context_histories"]
226:def _build_feature_matrix_internal(
953:def build_feature_matrix(
979:            df = _build_feature_matrix_internal(
exec
/bin/zsh -lc "sed -n '220,520p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 698ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    )

# ---------------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------------

def _build_feature_matrix_internal(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    chunk_days: int = 7,
    recompute_intermediate: bool = True,
    build_rolling: bool = True,
    output_path: Path | None = None,
    primary_output_path: Path | None = None,
    cutoff_label: str | None = None,
) -> pl.DataFrame:
    """Run full feature pipeline up to *inclusive* `end_date` for a single cutoff window.


thinking
**Analyzing key feature sections**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/targets.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 743ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Target column validation for NFL player prediction models.

Target columns are created in utils/feature/player_game_level.py during aggregation.
This module validates that required targets exist and have correct dtypes.
"""

import polars as pl
import logging

logger = logging.getLogger(__name__)


# Expected target columns and their types
NFL_TARGET_SCHEMA = {
    # Binary classification targets
    "anytime_td": pl.Int8,  # Primary MVP target: did player score a TD?
    
    # Count/regression targets
    "td_count": pl.Int64,  # Number of TDs scored (receiving + rushing)

thinking
**Checking relevant player game data**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""Player-game level aggregation for NFL predictions.

Aggregates play-by-play data to player-game level for all skill positions.
Each player gets one row per game with their total stats.

Target columns created:
- anytime_td (binary): Did player score a TD (receiving or rushing)?
- td_count (int): Number of TDs scored (receiving + rushing)
- passing_td (int): Number of passing TDs (for QBs)
- receiving_yards (float): Total receiving yards
- rushing_yards (float): Total rushing yards
- passing_yards (float): Total passing yards
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import datetime as dt

thinking
**Analyzing potential data issues and feature integration**

I'm focusing on spotting key issues like data leakage or unproduced columns in the large player_game_level file by reviewing mentions of rolling_window, MLB patterns, and stats usage around add_rolling_features. I want to understand if features rely on missing caches or incorrect columns that might disrupt aggregation or inference.
exec
/bin/zsh -lc 'rg "add_rolling_features" -n pipeline/feature.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 753ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
31:from utils.feature.rolling_window import add_rolling_features
831:        df_player_game_all = add_rolling_features(
exec
/bin/zsh -lc "sed -n '760,880p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 748ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted

    logging.info("Enriching odds features (NFL)...")
    rows_before_odds = df_player_game_all.height
    df_player_game_all = add_nfl_odds_features_to_df(
        df_player_game_all,
        player_col="player_name",
        allow_schedule_fallback=False,
        drop_schedule_rows=True,
    )
    
    logging.info("Enriching player props (Anytime TD)...")
    df_player_game_all = add_player_props_to_df(
        df_player_game_all,
        player_col="player_name",
    )

    rows_after_odds = df_player_game_all.height
    logging.info(
        "Odds features attached; retained %d/%d rows (%.2f%%) with pre-cutoff snapshots",

thinking
**Examining rolling context distinctions**
exec
/bin/zsh -lc "sed -n '880,1120p' pipeline/feature.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 760ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
        logging.info("Added %d historical share features", len(share_exprs))
    leakage_cols = [
        "target_share",
        "carry_share",
        "pass_attempt_share",
        "red_zone_target_share",
        "red_zone_carry_share",
        "goal_to_go_target_share",
        "goal_to_go_carry_share",
    ]
    df_player_game_all = df_player_game_all.drop(leakage_cols, strict=False)
    # Keep raw outcome columns (targets, carries, etc.) in the feature matrix so they can
    # serve as supervised targets. These columns are excluded from model inputs later via
    # training configuration, so we avoid dropping them here to retain label availability.

    available_cols = set(df_player_game_all.columns)
    market_exprs: list[pl.Expr] = []
    if {"team_implied_total", "hist_target_share_prev"} <= available_cols:
        market_exprs.append(

thinking
**Analyzing data consistency and leakage risks**

I'm examining key conceptual challenges in the rolling window features, especially the mismatch between expected and available stats columns, which may create noise from zero-filled features. I'm also checking for potential data leakage from using caches that include same-day or future game data, ensuring that feature computations properly exclude current or future information during training and inference.

thinking
**Analyzing potential data leakage in feature computation**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 751ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
Consolidated NFL Odds Module
============================

This module provides all odds-related functionality for the NFL prediction pipeline:

1. OddsAPIClient - HTTP client for The Odds API with caching and retry logic
2. Game-level odds features (moneyline, spread, totals) from nfl_odds.py
3. Player-level odds features (anytime TD props) from odds_features.py
4. Odds snapshot collection and persistence from odds_snapshots.py

All odds logic is consolidated here to prevent drift and ensure consistency.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

thinking
**Analyzing rolling window and feature pipeline**

I’m digging into the rolling window implementation and its interaction with feature caches and pipelines. There are questions about how date ranges and future game predictions align, especially if the pipeline uses actual play data that wouldn’t exist yet for upcoming games. I want to verify how the prediction pipeline handles schedules and odds versus the completed game data the feature builds rely on. Let’s explore pipeline/predict next!
exec
/bin/zsh -lc "sed -n '1,200p' pipeline/predict.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
"""
NFL prediction pipeline
=======================

Generates anytime-touchdown probabilities for upcoming games by:

1. Loading the upcoming schedule and weekly rosters (via nfl_data_py)
2. Building a player-level scaffold for offensive skill positions
3. Enriching that scaffold with rolling-window features using the same
   caches that power the training pipeline
4. Sequentially running the model chain:
   Availability -> Team Pace -> Usage -> Efficiency -> Anytime TD
5. Writing an actionable slate under `output/predictions/`
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

thinking
**Analyzing feature computation inconsistencies**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/rolling_experiments.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 752ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

from typing import Iterable

import polars as pl


def add_experimental_rolling_features(
    df: pl.DataFrame,
    stats: Iterable[str],
    *,
    epsilon: float = 1e-3,
) -> pl.DataFrame:
    """
    Derive experimental rolling-window features to capture burst performance,
    exponential-style recency weighting, and directional trends.

    Parameters
    ----------

thinking
**Checking dataset relevance and usage**
exec
/bin/zsh -lc 'rg "ps_route_participation"' in /Users/jameswest/Desktop/nfl_predict succeeded in 754ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
pipeline/predict.py:    "ps_route_participation_plays",
pipeline/predict.py:    "ps_route_participation_pct",
pipeline/predict.py:    "ps_route_participation_pct_prev",
pipeline/predict.py:    "ps_route_participation_pct_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3",
pipeline/predict.py:    "ps_route_participation_plays_l3": "ps_route_participation_plays",
pipeline/predict.py:        "1g_ps_route_participation_pct_per_game": "ps_route_participation_pct",
pipeline/predict.py:        "3g_ps_route_participation_pct_per_game": "ps_route_participation_pct_l3",
pipeline/predict.py:        "1g_ps_route_participation_plays_per_game": "ps_route_participation_plays",
pipeline/predict.py:        "3g_ps_route_participation_plays_per_game": "ps_route_participation_plays_l3",
pipeline/predict.py:        "ps_route_participation_pct",
pipeline/predict.py:        "ps_route_participation_plays",
utils/feature/stats.py:    "ps_route_participation_pct",
utils/feature/stats.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_plays",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:    "ps_route_participation_pct",
utils/feature/player_game_level.py:        .agg(pl.len().alias("ps_route_participation_plays"))
utils/feature/player_game_level.py:            .then(pl.col("ps_route_participation_plays") / pl.col("ps_team_dropbacks"))

thinking
**Investigating player_game_level and anytime_td**
exec
/bin/zsh -lc 'rg "anytime_td" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 777ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
- anytime_td (binary): Did player score a TD (receiving or rushing)?
    "market_anytime_td_prob",
    "market_anytime_td_consensus",
    "market_anytime_td_spread",
    "market_anytime_td_book_count",
    "market_anytime_td_prob_2h",
    "market_anytime_td_consensus_2h",
    "market_anytime_td_spread_2h",
    "market_anytime_td_book_count_2h",
    "market_anytime_td_prob_6h",
    "market_anytime_td_consensus_6h",
    "market_anytime_td_spread_6h",
    "market_anytime_td_book_count_6h",
    "market_anytime_td_prob_24h",
    "market_anytime_td_consensus_24h",
    "market_anytime_td_spread_24h",
    "market_anytime_td_book_count_24h",
    "market_anytime_td_prob_open",
    "market_anytime_td_consensus_open",

thinking
**Analyzing potential data leakage and timing issues**

I'm piecing together key concerns about possible data leakage in the rolling window system and training pipeline, especially related to future data inclusion, odds snapshot timing, and injury predictions. I'll highlight complications with caching and as-of joins that may cause leakage or inconsistencies.

thinking
**Evaluating data leakage risks in caches**

I'm thinking about whether cached stats might include data from invalid or future games, potentially causing leakage in training or inference. I need to check if features like offense_context have discrepancies between their training and inference implementations that might lead to mismatches or leaks.
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/offense_context.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 788ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import polars as pl

logger = logging.getLogger(__name__)

COORD_DEFAULT = "UNKNOWN_COORDINATOR"
QB_DEFAULT = "UNKNOWN_QB"
EPS = 1e-3

COORDINATOR_MAP_PATH = Path("data/processed/offensive_coordinators.parquet")


def _build_is_unavailable_expr(df: pl.DataFrame) -> pl.Expr:
    """Return a boolean expression marking players ruled out before cutoff."""

thinking
**Reviewing touchdown definitions for label leakage**
exec
/bin/zsh -lc 'rg -n "touchdowns" utils/feature/player_game_level.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 738ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
803:        pl.lit(0).cast(pl.Int64).alias("touchdowns"),
2577:            "touchdowns",
3108:              .alias("touchdowns"),
3121:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3123:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3165:              .alias("touchdowns"),
3177:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3178:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3220:              .alias("touchdowns"),
3232:            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
3233:            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
3271:            pl.col("touchdowns").fill_null(0).sum().alias("touchdowns"),
3284:        "passing_td", "rushing_td_count", "receiving_td_count", "touchdowns",
3293:    # derive it from rushing + receiving touchdowns.
3295:        pl.col("touchdowns").fill_null(0).alias("touchdowns"),
3306:        pl.when(pl.col("touchdowns") > 0)
3307:        .then(pl.col("touchdowns"))
3310:        .alias("_total_touchdowns")
3314:        (pl.col("_total_touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),

thinking
**Inspecting final assembly logic**
exec
/bin/zsh -lc "sed -n '3240,3320p' utils/feature/player_game_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 771ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
    
    Example: QB who passed and rushed, WR who received and rushed.
    """
    
    # Group by (player, game) and sum all stats
    merged = (
        df
        .group_by(["season", "week", "game_id", "game_date", "player_id", "player_name"])
        .agg([
            # Sum all yardage
            pl.col("passing_yards").fill_null(0).sum().alias("passing_yards"),
            pl.col("rushing_yards").fill_null(0).sum().alias("rushing_yards"),
            pl.col("receiving_yards").fill_null(0).sum().alias("receiving_yards"),
            
            # Sum counts
            pl.col("pass_attempt").fill_null(0).sum().alias("pass_attempt"),
            pl.col("completion").fill_null(0).sum().alias("completion"),
            pl.col("carry").fill_null(0).sum().alias("carry"),
            pl.col("target").fill_null(0).sum().alias("target"),

thinking
**Examining touchdown aggregation logic**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Play-level feature builder.

Reads cleaned daily Parquets and writes weekly play-level Parquets at
  data/processed/play_by_week/season=YYYY/week=W/part.parquet
"""

from pathlib import Path
from datetime import date
import polars as pl

CLEAN_DIR   = Path("data/cleaned")
PLAY_OUTDIR = Path("data/processed/play_by_week")

PLAY_COLS = [
    # Game identifiers
    "game_id",
    "season",

thinking
**Continuing review of derived statistics**
exec
/bin/zsh -lc "sed -n '200,400p' utils/feature/play_level.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 758ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
            ),
            (
                pl.col("defense_man_zone_type").cast(pl.Utf8).alias("defense_man_zone_type")
                if "defense_man_zone_type" in df.columns
                else pl.lit(None).alias("defense_man_zone_type")
            ),
            (
                pl.col("defense_coverage_type").cast(pl.Utf8).alias("defense_coverage_type")
                if "defense_coverage_type" in df.columns
                else pl.lit(None).alias("defense_coverage_type")
            ),
        ]
    )

    situ_exprs: list[pl.Expr] = []
    score_diff = pl.col("score_differential").fill_null(0.0)
    ydstogo = pl.col("ydstogo").fill_null(0.0)
    yardline = pl.col("yardline_100").cast(pl.Float32).fill_null(100.0)
    half_secs_expr = (

thinking
**Examining touchdown counting and rolling window concerns**

thinking
**Reviewing data shifts and plan progress**
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  → Review feature construction and data pipeline for realism issues, leakage, or dead code
  • Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Gathering details on config and training**
exec
/bin/zsh -lc "sed -n '1,200p' config/training.yaml" in /Users/jameswest/Desktop/nfl_predict succeeded in 815ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
# config/training.yaml
# Configuration for the model training pipeline (pipeline/train.py)
# NFL Player Prediction Targets

# --- Data I/O ---
data:
  raw_pbp_dir: 'data/raw/pbp_by_day'  # NFL play-by-play raw data
  feature_matrix_path: 'data/processed/final/processed.parquet'  # NFL feature matrix
  
  # Directory to save trained model artifacts (models and feature lists).
  model_output_dir: 'output/models'
  
  # Directory to save evaluation reports and metrics.
  metrics_output_dir: 'output/metrics/'

# =============================================================================
# SHARED COLUMN DEFINITIONS (using YAML anchors to avoid duplication)
# =============================================================================
# These columns leak same-game outcomes and must be excluded from training.

thinking
**Analyzing potential data inconsistencies and leakage**
exec
/bin/zsh -lc "sed -n '1,200p' utils/feature/asof.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 2.75s:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
from __future__ import annotations

"""Helpers for aligning feature generation to a consistent decision-time snapshot."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Mapping

import yaml

from utils.general.paths import PROJ_ROOT


@lru_cache(maxsize=1)
def _load_cutoff_config() -> Mapping[str, Any]:
    """Load decision cutoff configuration from config/config.yaml."""
    config_path = PROJ_ROOT / "config" / "config.yaml"
    try:

thinking
**Identifying discrepancies in feature construction**

I'm analyzing several potential issues in how features are constructed, especially around rolling window computations, drive-level feature usage, pre-snap route participation data reliability, and labeling choices. I want to assemble a clear list of 6-7 concrete findings summarizing inconsistencies, incomplete implementations, and data limitations.

thinking
**Analyzing potential data leakage and pipeline inconsistencies**

I'm exploring possible data leakage issues, especially around using full-season roster and injury data without proper as-of gating in predictions. I’m also checking whether training pipelines enforce the claimed multi-stage model architecture and if various feature-building steps apply correct cutoff filters, since odds features might not respect decision cutoff timestamps, risking leakage. This involves reviewing training, feature, and monitoring docs alongside function implementations to understand the conceptual gaps and inconsistencies.

thinking
**Searching for add_player_odds_features_to_df function**
exec
/bin/zsh -lc 'rg "def add_player" -n utils/feature/odds.py' in /Users/jameswest/Desktop/nfl_predict succeeded in 926ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
1099:def add_player_odds_features_to_df(
exec
/bin/zsh -lc "sed -n '1099,1250p' utils/feature/odds.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 732ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
def add_player_odds_features_to_df(
    df_pl: pl.DataFrame, *, api_key: str | None = None, player_col: str = "player_name",
    per_game_workers: int = 4, verbose: bool = False,
) -> pl.DataFrame:
    """Attach player odds features directly to an in-memory Polars DataFrame."""
    from utils.general.config import get_odds_api_key
    
    if df_pl.is_empty():
        return df_pl
    
    if not api_key:
        api_key = get_odds_api_key()
    
    if not api_key:
        logger.info("add_player_odds_features_to_df: no odds_api_key configured – adding empty columns and returning")
        empty_exprs = [pl.lit(None).cast(pl.Float32).alias(c) for c in PLAYER_ODDS_NUMERIC_COLS]
        flag_expr = [pl.lit(0).cast(pl.Int8).alias("odds_expected")]
        df_base = df_pl.drop(PLAYER_ODDS_NUMERIC_COLS + ODDS_FLAG_COLUMNS, strict=False)
        return df_base.with_columns(empty_exprs + flag_expr)

thinking
**Analyzing data leakage and feature completeness**

thinking
**Reviewing data leakage and caching inconsistencies**

I'm noticing potential data leakage in feature functions using live API calls and uncertain odds snapshot timing. There's also a mismatch between documented and actual cache directories that could confuse or cause bugs. I want to gather all issues before moving from feature pipeline review to modeling validation, paying attention to training config settings and columns flagged as leak-prone.
exec
/bin/zsh -lc 'rg "LEAK_PRONE_COLUMNS" -n utils' in /Users/jameswest/Desktop/nfl_predict succeeded in 870ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
utils/general/__init__.py:18:    LEAK_PRONE_COLUMNS,
utils/general/__init__.py:47:    "LEAK_PRONE_COLUMNS",
utils/general/constants.py:8:config/training.yaml should mirror LEAK_PRONE_COLUMNS for YAML-based configuration,
utils/general/constants.py:113:LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
utils/general/constants.py:208:    "LEAK_PRONE_COLUMNS",
exec
/bin/zsh -lc "sed -n '100,160p' utils/general/constants.py" in /Users/jameswest/Desktop/nfl_predict succeeded in 800ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
}


# =============================================================================
# Leak-Prone Columns (Same-Game Outcomes)
# =============================================================================
# Stats that leak same-game outcomes and must be excluded from training features.
# These represent what happened in the game we're trying to predict.
#
# This is the AUTHORITATIVE definition. config/training.yaml mirrors this list
# using YAML anchors for per-problem configuration. Keep them in sync!
# Validated by: tests/test_training_config_features.py

LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
    # Yardage (same-game outcomes)
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    # Usage counts (same-game outcomes)

thinking
**Analyzing modeling pipeline inconsistencies**

I’m reviewing the training pipeline for possible incomplete or risky implementations—especially around derived targets, odds features that may leak future info, rolling window system mismatches, and caching issues causing potential feature inaccuracies. I'll note risky spots with specifics to file paths and behaviors to highlight data leakage or half-finished components.

thinking
**Identifying potential data leakage in features and inference**

thinking
**Analyzing touchdown column naming mismatch**

thinking
**Analyzing touchdown feature inconsistencies**

I notice a mismatch between "touchdown" and "touchdowns" columns in different dataframes, causing rolling touchdown features to be always zero—this looks like a conceptual bug. Also, the way stats are counted might cause double counting or misalignment. I want to fix these naming and aggregation issues and then update the project plan to mark step 2 as done and step 3 as in progress.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  → Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Analyzing model training and feature inconsistencies**

thinking
**Evaluating data quality and documentation gaps**

I'm analyzing issues with sparse or missing data affecting model training, especially around pre-snap route participation features and roster/injury alignment leaking future info. I also notice documentation inconsistencies in rolling window implementation and potential feature instability due to network-dependent API calls, all of which could harm predictive reliability and require careful handling in the next steps.
Plan update
  ✓ Scan project structure and key config/docs to understand intended data flow and leakage guards
  ✓ Review feature construction and data pipeline for realism issues, leakage, or dead code
  ✓ Inspect modeling/training/eval scripts for conceptual issues, incomplete ideas, and mismatch to reality
  • Summarize findings with concrete impacts and suggested direction

thinking
**Preparing final summary and suggestions**

I'll create a clear bullet list of findings ordered by severity with approximate file paths and lines, highlighting key issues like touchdown rolling features and pre-snap usage, then follow with numbered next steps for the user to consider.
codex
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.
tokens used
110,897
- Player prop odds enrichment is leaking/hallucinated: `pipeline/feature.py` calls `add_player_props_to_df` (alias of `utils/feature/odds.add_player_odds_features_to_df`) which fetches live Odds API data with no date filter or snapshot control. When building historical training data this either (a) pulls today’s markets for long-past games or (b) fails and fills nulls/zeros, so the “anytime TD market” features are either future-looking or garbage and don’t reflect pre‑kick information. The same happens in inference if the API key is missing or network is blocked, so training/inference drift is guaranteed. Those columns should instead come from archived, pre‑cutoff snapshots (or be dropped).

- Rolling TD history is silently broken: `utils/feature/stats.NFL_PLAYER_STATS` uses the key `"touchdown"`, but the aggregation writes `"touchdowns"` (`utils/feature/player_game_level.py`), so `cache/feature/daily_totals` never contains `touchdown_num` and all rolling touchdown features produced by `utils/feature/rolling_window.py` are zero/NaN. That wipes out the most relevant recency signal for anytime‑TD likelihood.

- Rolling window rollout is inconsistent and inflated: docs promise a conservative 16‑feature rollout (`docs/rolling_window_implementation.md`) and a cache at `cache/feature/nfl_daily_totals`, but the code builds only game‑level cache under `cache/feature/daily_totals` and then requests *all* stats, windows, and contexts (`pipeline/feature.py`). Many of those stats (route shares, ps_*, snap counts) aren’t present in the cache, so hundreds of derived columns are flat zeros or NaNs. This adds heavy noise/overfitting risk and diverges from the documented design.

- Inference/backtests use post‑facto injury/roster info: `pipeline/predict.py` pulls `import_weekly_rosters`/`import_injuries` without any decision‑time cutoff or snapshot metadata. For historical slates, that feeds final game statuses (OUT/IR, etc.) into features, inflating AUC compared to what would have been knowable at T‑6h. It also deviates from the training pipeline’s cutoff enforcement.

- Pre‑snap route/scripted touch targets are effectively imaginary: `utils/feature/player_game_level.py` derives ps_* fields from `offense_players/offense_positions/route` strings, which are sparsely populated (often null in nflfastR) and absent prior to recent seasons. Yet `config/training.yaml` trains dedicated models on these columns and they’re included in rolling stats. In practice the data are mostly missing/zero, so these tasks/features add noise and mismatched expectations for “pre‑snap” realism.

- History cache omits drive-level context despite being advertised: the code builds only game-level daily totals (`build_daily_cache_range` in `pipeline/feature.py`), but constants include drive-level windows/context and `_finalize_drive_history_features` exists. Any drive-based recency logic mentioned in docs is unused, so “per-drive” context that should capture play-calling/goal-to-go tendencies is absent.

Next steps:
1) Replace player prop odds enrichment with archived, pre‑cutoff snapshots (or drop until available).  
2) Align rolling stat keys (`touchdowns` vs `touchdown`) and prune the rolling feature set to the small, validated subset; remove/disable contexts/windows that lack cache support.  
3) Add decision-time gating for roster/injury sources in `pipeline/predict.py` or remove those features from backtests until gated.  
4) Re-evaluate ps_* targets/features; if underlying data coverage is poor, remove them from problems/config to reduce noise.  
5) Either build the drive-level cache or remove drive-context knobs from constants/docs to avoid false confidence.

    <END_ANALYSIS_END>

    This step is PURELY research, investigation, and planning.
    Do NOT make any edits to the code or data in this step.

    I want:
    - A structured, prioritized plan.
    - Clear steps that can be implemented by an agent in later steps.
    - Notes on risk or potential pitfalls where relevant, but focus on high-value changes.
    - Explicit attention to:
        * eliminating data leakage,
        * aligning features with real football mechanisms that drive anytime TDs,
        * cleaning up hallucinated / legacy cruft that no longer makes sense.

    Again: no edits here, just the plan.

mcp startup: no servers

thinking
**Drafting a structured planning outline**
codex
**Remediation Plan (ordered by impact & leakage risk)**
- Fix odds leakage/hallucination: freeze player prop odds to archived pre-cutoff snapshots; add snapshot loader keyed by game_id/team/player + snapshot_ts; in training, drop/flag rows without snapshots; in inference, refuse to call live API, optionally fall back to schedule-only baseline; add tests to assert no odds columns present when snapshot missing.
- Repair rolling TD history: align stat key `touchdowns` across `stats.py`, aggregation, and cache; rebuild daily_totals cache; add schema check to fail if any rolling TD feature is all-null/zero; rerun backtests to measure regained signal.
- Trim rolling window surface to supported set: codify a vetted set of stats/windows/contexts (e.g., {1g,3g,5g,season} × {vs_any}); remove ps_* and other missing-cache stats from rolling requests; update docs to match; add runtime guard to error on requested stats not found in cache.
- Enforce as-of gating for roster/injury inputs in prediction/backtests: add decision-time cutoff (e.g., T-6h) and snapshot source; if unavailable, drop those features or mark unavailable; add regression test ensuring no injury status is from after game start in historical inference.
- Remove/retire ps_* targets/features if coverage is low: measure non-null rate by season; if sparse, drop from training configs and rolling stats; otherwise backfill realistically or scope to seasons with data; add check to block tasks with <X% coverage.
- Clarify/implement drive-level history or delete: either build drive-level daily cache and wire into rolling computation, or remove drive-context flags/constants/docs; add assertion preventing “drive” level when cache absent.
- Align docs with reality and guard drift: sync paths/names (cache/feature/daily_totals vs nfl_daily_totals); document the final rolling feature list and cutoff rules; add a pre-flight validation script that checks cache schemas, non-null rates, and leakage-sensitive columns before training.

**Execution Steps**
1) Odds snapshots: design schema + loader; replace `add_player_props_to_df` call; add fallback behavior and tests; rerun feature build on a small date range to validate columns.
2) Rolling TD fix: rename stat key, rebuild cache for a sample window, ensure rolling TD features populate; update tests.
3) Rolling feature pruning: define allowed stats/windows/contexts; update config/constants + pipeline call; add guard for missing stats; rerun feature build and inspect null/constant rates.
4) Roster/injury gating: add cutoff param in predict/backtest; wire to snapshot source; add assertion that injury_status_ts <= cutoff; adjust backtests to use gated features.
5) ps_* triage: compute coverage by season; decide keep/drop; update training.yaml and problem definitions; if dropped, remove from rolling stats and tasks.
6) Drive-level decision: either implement cache builder (drive totals) and wire into rolling_window or delete related flags/docs; add guard to prevent silent no-op.
7) Docs/validation: update rolling_window_implementation.md and monitoring docs; add validation script to check schemas, coverage, leak-prone columns excluded from train set; run it in CI/pre-flight.

**Risks/Pitfalls to watch**
- Cache rebuild time/size; may need incremental rebuild and date filters.
- Snapshot availability gaps; must define strict fallbacks to avoid silent leakage.
- Training/inference drift if odds/roster data are absent—prefer hard failure over silent nulls.
- Removing ps_* may change model interfaces; update downstream consumers/tests accordingly.
tokens used
15,743
**Remediation Plan (ordered by impact & leakage risk)**
- Fix odds leakage/hallucination: freeze player prop odds to archived pre-cutoff snapshots; add snapshot loader keyed by game_id/team/player + snapshot_ts; in training, drop/flag rows without snapshots; in inference, refuse to call live API, optionally fall back to schedule-only baseline; add tests to assert no odds columns present when snapshot missing.
- Repair rolling TD history: align stat key `touchdowns` across `stats.py`, aggregation, and cache; rebuild daily_totals cache; add schema check to fail if any rolling TD feature is all-null/zero; rerun backtests to measure regained signal.
- Trim rolling window surface to supported set: codify a vetted set of stats/windows/contexts (e.g., {1g,3g,5g,season} × {vs_any}); remove ps_* and other missing-cache stats from rolling requests; update docs to match; add runtime guard to error on requested stats not found in cache.
- Enforce as-of gating for roster/injury inputs in prediction/backtests: add decision-time cutoff (e.g., T-6h) and snapshot source; if unavailable, drop those features or mark unavailable; add regression test ensuring no injury status is from after game start in historical inference.
- Remove/retire ps_* targets/features if coverage is low: measure non-null rate by season; if sparse, drop from training configs and rolling stats; otherwise backfill realistically or scope to seasons with data; add check to block tasks with <X% coverage.
- Clarify/implement drive-level history or delete: either build drive-level daily cache and wire into rolling computation, or remove drive-context flags/constants/docs; add assertion preventing “drive” level when cache absent.
- Align docs with reality and guard drift: sync paths/names (cache/feature/daily_totals vs nfl_daily_totals); document the final rolling feature list and cutoff rules; add a pre-flight validation script that checks cache schemas, non-null rates, and leakage-sensitive columns before training.

**Execution Steps**
1) Odds snapshots: design schema + loader; replace `add_player_props_to_df` call; add fallback behavior and tests; rerun feature build on a small date range to validate columns.
2) Rolling TD fix: rename stat key, rebuild cache for a sample window, ensure rolling TD features populate; update tests.
3) Rolling feature pruning: define allowed stats/windows/contexts; update config/constants + pipeline call; add guard for missing stats; rerun feature build and inspect null/constant rates.
4) Roster/injury gating: add cutoff param in predict/backtest; wire to snapshot source; add assertion that injury_status_ts <= cutoff; adjust backtests to use gated features.
5) ps_* triage: compute coverage by season; decide keep/drop; update training.yaml and problem definitions; if dropped, remove from rolling stats and tasks.
6) Drive-level decision: either implement cache builder (drive totals) and wire into rolling_window or delete related flags/docs; add guard to prevent silent no-op.
7) Docs/validation: update rolling_window_implementation.md and monitoring docs; add validation script to check schemas, coverage, leak-prone columns excluded from train set; run it in CI/pre-flight.

**Risks/Pitfalls to watch**
- Cache rebuild time/size; may need incremental rebuild and date filters.
- Snapshot availability gaps; must define strict fallbacks to avoid silent leakage.
- Training/inference drift if odds/roster data are absent—prefer hard failure over silent nulls.
- Removing ps_* may change model interfaces; update downstream consumers/tests accordingly.

    <END_PLAN_END>

    Validation requirements:
    - Capture a baseline snapshot before running any new code:
      * Record `auc`, `pr_auc`, `brier_score`, and `precision_at_thresh` from `/Users/jameswest/Desktop/nfl_predict/output/metrics/anytime_td/xgboost/20251126_001001_ab12e68_cutoff_h090m/metrics.yaml`.
      * Note the current feature-importance distribution from `/Users/jameswest/Desktop/nfl_predict/output/metrics/anytime_td/xgboost/20251126_001001_ab12e68_cutoff_h090m/feature_importance.json`.
    - Run `cd /Users/jameswest/Desktop/nfl_predict && /Users/jameswest/Desktop/nfl_predict/venv/bin/python /Users/jameswest/Desktop/nfl_predict/main.py` to rebuild datasets, retrain models, and regenerate predictions.
    - After the run completes, identify the newest timestamped directory under `/Users/jameswest/Desktop/nfl_predict/output/metrics/anytime_td/xgboost` and repeat the metric collection.
      * Compare before/after metrics and explicitly call out improvements or regressions.
      * Re-review the refreshed `feature_importance.json` for any unexpected signal shifts.
    - Inspect `cutoff_backtest_summary.csv` (e.g., `/Users/jameswest/Desktop/nfl_predict/output/metrics/cutoff_backtest_summary.csv`) to confirm horizon-level hit rates and calibration remain acceptable.
    - Only declare success if the post-run metrics improve (higher AUC/PR AUC, lower Brier/log loss) or, at minimum, hold steady with a justified explanation. Any regression must be reported.

    Rules for this loop:
    - You may inspect and edit code, rerun commands, or add logging as needed to fix any issues uncovered
      while running `cd /Users/jameswest/Desktop/nfl_predict && /Users/jameswest/Desktop/nfl_predict/venv/bin/python /Users/jameswest/Desktop/nfl_predict/main.py`.
    - Keep commits incremental. If you end up making larger fixes, describe them clearly.
    - After each attempt, report status using `PROGRESS:`. If the run fails, capture stack traces / log
      pointers so the next attempt knows what to fix.
    - Once you have successfully run the pipeline, collected the new metrics, and compared them against
      the previous baseline, end your response with the exact line:
          PIPELINE VALIDATION COMPLETE
      Include the usual summary before that sentinel line.
    - Do NOT output the sentinel unless the metrics truly exist and have been evaluated.

mcp startup: no servers

thinking
**Gathering baseline metrics**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && ls output/metrics/anytime_td/xgboost' in /Users/jameswest/Desktop/nfl_predict succeeded in 726ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
20251104_001849_nogit
20251104_003054_nogit
20251104_003250_nogit
20251104_003641_nogit
20251104_004354_nogit
20251104_005640_nogit
20251104_010454_nogit
20251104_011304_nogit
20251105_221431_nogit
20251105_221501_nogit
20251107_164534_nogit
20251107_164654_nogit
20251107_164750_nogit
20251108_022231_nogit
20251108_112825_nogit
20251108_124445_nogit
20251108_125323_nogit
20251108_130156_nogit
20251108_133055_nogit

thinking
**Fetching baseline metrics files**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && cat output/metrics/anytime_td/xgboost/20251126_001001_ab12e68_cutoff_h090m/metrics.yaml' in /Users/jameswest/Desktop/nfl_predict succeeded in 730ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
auc: 0.8333
pr_auc: 0.2684
precision_at_thresh: 0.2512
recall_at_thresh: 0.4344
decision_threshold: 0.0187
brier_score: 0.0872
log_loss: 0.3538
classification_report: null
confusion_matrix: null
confidence_analysis:
    0.0-0.1:
        count: 1232
        accuracy: 0.8522727272727273
        avg_confidence: 0.012282853286528912
        actual_positive_rate: 0.08116883116883117
    0.1-0.2:
        count: 23
        accuracy: 0.391304347826087
        avg_confidence: 0.13287789471771405

thinking
**Requesting feature importance JSON**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && head -n 40 output/metrics/anytime_td/xgboost/20251126_001001_ab12e68_cutoff_h090m/feature_importance.json' in /Users/jameswest/Desktop/nfl_predict succeeded in 737ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
{"features": ["drive_hist_count_l3", "drive_hist_count_prev", "drive_hist_td_drives_l3", "drive_hist_td_drives_prev", "drive_hist_td_rate_l3", "drive_hist_td_rate_prev", "drive_hist_total_yards_l3", "drive_hist_total_yards_prev", "drive_hist_touch_drives_l3", "drive_hist_touch_drives_prev", "drive_hist_touch_rate_l3", "drive_hist_touch_rate_prev", "drive_hist_yards_per_drive_l3", "drive_hist_yards_per_drive_prev", "expected_carries", "expected_carries_raw", "expected_opportunities", "expected_opportunities_raw", "expected_receiving_yards", "expected_rushing_yards", "expected_targets", "expected_targets_raw", "expected_td_prob_poisson", "expected_td_prob_poisson_raw", "expected_td_signal", "expected_td_signal_raw", "expected_team_plays", "expected_total_yards", "position", "pred_availability", "pred_availability_active", "pred_availability_raw", "pred_availability_snapshare", "pred_efficiency_rec_yards", "pred_efficiency_rush_yards", "pred_efficiency_tds", "pred_team_pace", "pred_usage_carries", "pred_usage_targets", "rest_days_since_last_game", "season", "spread_line", "team_implied_total", "travel_bye_week_flag", "travel_consecutive_home_games", "travel_consecutive_road_games", "travel_distance_km", "travel_distance_km_l3", "travel_distance_miles", "travel_east_to_west_late_flag", "travel_game_timezone_offset", "travel_local_start_hour", "travel_long_rest_flag", "travel_rest_days", "travel_rest_days_l3", "travel_rest_hours", "travel_short_week_flag", "travel_team_timezone_offset", "travel_time_diff_from_home_hours", "travel_timezone_change_hours", "travel_west_to_east_early_flag", "weather_bad_passing_flag", "weather_cloud_cover_pct", "weather_dew_point_f", "weather_extreme_wind_flag", "weather_feels_like_f", "weather_forecast_is_backfill", "weather_forecast_is_historical", "weather_forecast_lead_hours", "weather_forecast_uncertainty_temp", "weather_forecast_uncertainty_wind", "weather_freezing_flag", "weather_heavy_precip_flag", "weather_humidity_pct", "weather_precip_amount_in", "weather_precip_prob_trend", "weather_precip_probability_pct", "weather_pressure_mb", "weather_snow_amount_in", "weather_temp_air_f", "weather_temp_trend", "weather_visibility_miles", "weather_wind_gust_mph", "weather_wind_mph", "weather_wind_trend", "week"], "importance": {"drive_hist_count_l3": 22.25420379638672, "drive_hist_count_prev": 4.328996181488037, "drive_hist_td_drives_l3": 3.303638219833374, "drive_hist_td_drives_prev": 4.992429256439209, "drive_hist_td_rate_l3": 2.3106777667999268, "drive_hist_td_rate_prev": 3.54589581489563, "drive_hist_total_yards_l3": 12.366357803344727, "drive_hist_total_yards_prev": 12.9301118850708, "drive_hist_touch_drives_l3": 4.476215362548828, "drive_hist_touch_drives_prev": 5.1545867919921875, "drive_hist_touch_rate_prev": 2.2270007133483887, "drive_hist_yards_per_drive_l3": 17.68923568725586, "drive_hist_yards_per_drive_prev": 6.14043664932251, "expected_carries": 4.068063259124756, "expected_carries_raw": 4.403553485870361, "expected_opportunities": 12.768014907836914, "expected_opportunities_raw": 10.453889846801758, "expected_receiving_yards": 8.034473419189453, "expected_rushing_yards": 6.45242977142334, "expected_targets": 3.854857921600342, "expected_targets_raw": 2.977379322052002, "expected_td_prob_poisson": 51.88859558105469, "expected_td_prob_poisson_raw": 29.742591857910156, "expected_td_signal": 25.72814178466797, "expected_td_signal_raw": 29.4103946685791, "expected_team_plays": 1.963468074798584, "expected_total_yards": 15.344914436340332, "position": 2.388518810272217, "pred_availability": 1.6068334579467773, "pred_availability_active": 1.9598534107208252, "pred_availability_raw": 2.4126462936401367, "pred_availability_snapshare": 1.8703349828720093, "pred_efficiency_rec_yards": 5.83900260925293, "pred_efficiency_rush_yards": 5.155553340911865, "pred_efficiency_tds": 77.7385025024414, "pred_team_pace": 3.664811849594116, "pred_usage_carries": 4.9242777824401855, "pred_usage_targets": 5.404415130615234, "season": 3.4532644748687744, "spread_line": 1.8308967351913452, "travel_consecutive_home_games": 2.3219716548919678, "travel_distance_km": 1.5486832857131958, "travel_distance_km_l3": 1.328561544418335, "travel_distance_miles": 2.0549566745758057, "travel_game_timezone_offset": 1.6187328100204468, "travel_local_start_hour": 2.186598539352417, "travel_rest_days": 1.6390495300292969, "travel_rest_days_l3": 1.3755898475646973, "travel_rest_hours": 4.59040641784668, "travel_team_timezone_offset": 1.5894732475280762, "travel_time_diff_from_home_hours": 0.9084656834602356, "travel_timezone_change_hours": 2.02656888961792, "weather_cloud_cover_pct": 1.1075385808944702, "weather_extreme_wind_flag": 2.709768295288086, "weather_feels_like_f": 1.5291064977645874, "weather_humidity_pct": 2.1326889991760254, "weather_pressure_mb": 1.1504626274108887, "weather_temp_air_f": 2.864535093307495, "weather_visibility_miles": 2.99349308013916, "weather_wind_gust_mph": 1.5324591398239136, "weather_wind_mph": 2.148653745651245, "week": 1.0179940462112427}}

thinking
**Checking feature importance baseline**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && ls output/metrics | head' in /Users/jameswest/Desktop/nfl_predict succeeded in 771ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
anytime_td
anytime_td_meta
anytime_td_meta_xgboost.yaml
anytime_td_xgboost.yaml
availability
availability_active
availability_active_xgboost.yaml
availability_snapshare
availability_snapshare_xgboost.yaml
availability_xgboost.yaml

thinking
**Checking for cutoff_backtest_summary file**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && ls output/metrics | grep cutoff_backtest_summary' in /Users/jameswest/Desktop/nfl_predict succeeded in 736ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
cutoff_backtest_summary.csv
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && head -n 5 output/metrics/cutoff_backtest_summary.csv' in /Users/jameswest/Desktop/nfl_predict succeeded in 679ms:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
cutoff_hours,cutoff_label,problem,model,auc,pr_auc,precision_at_thresh,recall_at_thresh,decision_threshold,brier_score,log_loss,classification_report,confusion_matrix,confidence_analysis,calibration_method,calibration_ece_oof
1.5,h090m,anytime_td,xgboost,0.8982,0.4164,0.4399,0.1542,0.1497,0.08,0.2809,,,"{'0.0-0.1': {'count': 33092, 'accuracy': 0.9326725492566179, 'avg_confidence': 0.012031695717464055, 'actual_positive_rate': 0.06732745074338209}, '0.1-0.2': {'count': 2149, 'accuracy': 0.47417403443462075, 'avg_confidence': 0.14627068016891315, 'actual_positive_rate': 0.398324802233597}, '0.2-0.3': {'count': 296, 'accuracy': 0.7263513513513513, 'avg_confidence': 0.21277839403498816, 'actual_positive_rate': 0.7263513513513513}, '0.3-0.4': {'count': 1, 'accuracy': 1.0, 'avg_confidence': 0.3015100061893463, 'actual_positive_rate': 1.0}, '0.4-0.5': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.5-0.6': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.6-0.7': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.7-0.8': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.8-0.9': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.9-1.0': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '1.0': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}}",raw,0.0416
3.0,h03h,anytime_td,xgboost,0.873,0.4022,0.4555,0.163,0.1565,0.0807,0.2934,,,"{'0.0-0.1': {'count': 33393, 'accuracy': 0.9299254334740814, 'avg_confidence': 0.02112094222247829, 'actual_positive_rate': 0.0700745665259186}, '0.1-0.2': {'count': 1774, 'accuracy': 0.4898534385569335, 'avg_confidence': 0.15199785561142268, 'actual_positive_rate': 0.4222096956031567}, '0.2-0.3': {'count': 371, 'accuracy': 0.568733153638814, 'avg_confidence': 0.2135662884725072, 'actual_positive_rate': 0.568733153638814}, '0.3-0.4': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.4-0.5': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.5-0.6': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.6-0.7': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.7-0.8': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.8-0.9': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.9-1.0': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '1.0': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}}",sigmoid,0.0407
6.0,h06h,anytime_td,xgboost,0.8731,0.3887,0.497,0.2003,0.1489,0.0813,0.2973,,,"{'0.0-0.1': {'count': 33278, 'accuracy': 0.9296231744696196, 'avg_confidence': 0.020686380115597213, 'actual_positive_rate': 0.07037682553038044}, '0.1-0.2': {'count': 1938, 'accuracy': 0.5567595459236326, 'avg_confidence': 0.1471018776820417, 'actual_positive_rate': 0.38338493292053666}, '0.2-0.3': {'count': 322, 'accuracy': 0.6677018633540373, 'avg_confidence': 0.2094104781006434, 'actual_positive_rate': 0.6677018633540373}, '0.3-0.4': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.4-0.5': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.5-0.6': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.6-0.7': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.7-0.8': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.8-0.9': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.9-1.0': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '1.0': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}}",sigmoid,0.0396
12.0,h12h,anytime_td,xgboost,0.8699,0.3989,0.4864,0.1621,0.1545,0.0812,0.2973,,,"{'0.0-0.1': {'count': 33448, 'accuracy': 0.9283963166706529, 'avg_confidence': 0.02112087037948247, 'actual_positive_rate': 0.07160368332934705}, '0.1-0.2': {'count': 1795, 'accuracy': 0.5376044568245125, 'avg_confidence': 0.15029201539944143, 'actual_positive_rate': 0.3983286908077994}, '0.2-0.3': {'count': 295, 'accuracy': 0.6440677966101694, 'avg_confidence': 0.21011015378822714, 'actual_positive_rate': 0.6440677966101694}, '0.3-0.4': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.4-0.5': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.5-0.6': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.6-0.7': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.7-0.8': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.8-0.9': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '0.9-1.0': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}, '1.0': {'count': 0, 'accuracy': 0.0, 'avg_confidence': 0.0, 'actual_positive_rate': 0.0}}",sigmoid,0.0386

thinking
**Preparing to run main command**
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && /Users/jameswest/Desktop/nfl_predict/venv/bin/python /Users/jameswest/Desktop/nfl_predict/main.py' in /Users/jameswest/Desktop/nfl_predict
[codex_meta_loop] (heartbeat) no output from codex for 20s...

[codex_meta_loop] (heartbeat) no output from codex for 20s...

[codex_meta_loop] (heartbeat) no output from codex for 20s...

[codex_meta_loop] (heartbeat) no output from codex for 20s...

[codex_meta_loop] (heartbeat) no output from codex for 20s...
 exited 124 in 2m 00s:
/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 18: /bin/ps: Operation not permitted
2025-11-26 01:14:56,025 - WARNING - /Users/jameswest/.matplotlib is not a writable directory
2025-11-26 01:14:56,025 - WARNING - Matplotlib created a temporary cache directory at /var/folders/06/pn25p08s65114098gsqf5b5w0000gn/T/matplotlib-b65qu281 because there was an issue with the default path (/Users/jameswest/.matplotlib); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
INFO: ================================================================================
2025-11-26 01:14:56,229 - INFO - ================================================================================
INFO: NFL PLAYER PREDICTION PIPELINE
2025-11-26 01:14:56,229 - INFO - NFL PLAYER PREDICTION PIPELINE
INFO: ================================================================================
2025-11-26 01:14:56,230 - INFO - ================================================================================
INFO: Pipeline date range: 2021-09-09 → 2025-11-26
2025-11-26 01:14:56,230 - INFO - Pipeline date range: 2021-09-09 → 2025-11-26
INFO: 
================================================================================
2025-11-26 01:14:56,230 - INFO - 
================================================================================
INFO: STEP 1: DATA COLLECTION (INCREMENTAL)
2025-11-26 01:14:56,231 - INFO - STEP 1: DATA COLLECTION (INCREMENTAL)
INFO: ================================================================================
2025-11-26 01:14:56,231 - INFO - ================================================================================
2025-11-26 01:14:56,231 - INFO - 📥 Incremental update: Loading current season: [2025]

thinking
**Handling command timeout**

The command timed out after 120 seconds, so I’m planning to rerun it with a much longer timeout, possibly 15 minutes, to see if that gets past the hang. I'll rerun the full process rather than partial since that might give clearer insight on where it stalls.
exec
/bin/zsh -lc 'cd /Users/jameswest/Desktop/nfl_predict && /Users/jameswest/Desktop/nfl_predict/venv/bin/python /Users/jameswest/Desktop/nfl_predict/main.py' in /Users/jameswest/Desktop/nfl_predict
[codex_meta_loop] (heartbeat) no output from codex for 20s...

[codex_meta_loop] (heartbeat) no output from codex for 20s...
^C
[codex_meta_loop] SIGINT received: terminating codex...

[codex_meta_loop] KeyboardInterrupt: terminating codex...

[codex_meta_loop] Interrupted by user. Exiting.
jameswest@Macbook nfl_predict % 
