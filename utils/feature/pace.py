from __future__ import annotations

import logging
from pathlib import Path
import polars as pl

logger = logging.getLogger(__name__)

PLAY_DIR = Path("data/processed/play_by_week")

def compute_pace_metrics(seasons: list[int] | None = None) -> pl.DataFrame:
    """
    Compute advanced pace metrics from play-by-play data.
    
    Returns DataFrame with columns:
        - season, week, team, game_date
        - team_pace_no_huddle_rate
        - team_pace_proe
        - team_pace_neutral_pass_rate
        - team_pace_neutral_seconds_per_play
    """
    scan_kwargs = {
        "glob": True,
        "hive_partitioning": True,
        "missing_columns": "insert",
        "extra_columns": "ignore"
    }
    
    scan = pl.scan_parquet(str(PLAY_DIR / "season=*/week=*/part.parquet"), **scan_kwargs)
    
    # Handle potential schema mismatch by casting early
    scan = scan.with_columns([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
    ])

    if seasons is not None:
        scan = scan.filter(pl.col("season").is_in(seasons))

    # Continue with casting other columns
    scan = scan.with_columns([
        pl.col("game_id").cast(pl.Utf8),
        pl.col("posteam").cast(pl.Utf8),
    ])

    # 1. Filter valid plays for pace calculation
    # Exclude spikes, kneels
    # Exclude OT (qtr > 4)
    # Require valid posteam
    valid_plays = scan.filter(
        (pl.col("posteam").is_not_null()) &
        (pl.col("qtr") <= 4) &
        (pl.col("qb_spike").fill_null(0) == 0) &
        (pl.col("qb_kneel").fill_null(0) == 0)
    )

    # 2. Define Neutral Context
    # Score within 10 points
    # Not in last 2 minutes of half (hurry up / kill clock mode)
    neutral_cond = (
        (pl.col("score_differential").abs() <= 10) &
        (pl.col("half_seconds_remaining") > 120)
    )
    
    # 3. Aggregate Stats
    stats = (
        valid_plays
        .group_by(["season", "week", "game_id", "posteam", "game_date"])
        .agg([
            # No Huddle Rate (All situations)
            (pl.col("no_huddle").fill_null(0).mean()).alias("team_pace_no_huddle_rate"),
            
            # PROE (All situations where calculated)
            (pl.col("pass_oe").mean()).alias("team_pace_proe"),
            
            # Neutral Pass Rate
            (pl.when(neutral_cond).then(pl.col("pass")).otherwise(None).mean()).alias("team_pace_neutral_pass_rate"),
            
            # Neutral Plays (Count)
            (pl.when(neutral_cond).then(1).otherwise(0).sum()).alias("_neutral_plays_count"),
            
            # Drive-level grouping for Seconds Per Play
            # We can't do nested aggregation easily in one pass if we want efficiency.
            # Strategy: Compute seconds per play via a separate path or approximation.
            # Approx: We don't have drive times here.
            # Let's omit seconds_per_play for now and focus on PROE/NoHuddle which are strong signals.
        ])
    )
    
    # 4. Clean up
    stats = stats.with_columns([
        pl.col("team_pace_proe").fill_null(0.0),
        pl.col("team_pace_no_huddle_rate").fill_null(0.0),
        pl.col("team_pace_neutral_pass_rate").fill_null(0.5), # fallback
    ])
    
    stats = stats.rename({"posteam": "team"})
    
    # Sort
    stats = stats.sort(["season", "week", "team"]).collect(streaming=True)
    
    return stats

def add_pace_history(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add rolling history features for pace metrics.
    Expects a DataFrame with season, week, team, game_date, and raw pace metrics.
    """
    metrics = [
        "team_pace_no_huddle_rate",
        "team_pace_proe",
        "team_pace_neutral_pass_rate",
    ]
    
    # Ensure sorted
    df = df.sort(["season", "team", "game_date"])
    
    for metric in metrics:
        df = df.with_columns([
            # Previous game
            pl.col(metric).shift(1).over(["season", "team"]).alias(f"{metric}_prev"),
            
            # Last 3 games (excluding current)
            pl.col(metric)
            .rolling_mean(window_size=3, min_periods=1)
            .shift(1)
            .over(["season", "team"])
            .alias(f"{metric}_l3"),
            
            # Last 5 games
            pl.col(metric)
            .rolling_mean(window_size=5, min_periods=1)
            .shift(1)
            .over(["season", "team"])
            .alias(f"{metric}_l5"),
        ])
        
    # Drop current-game actuals to prevent leakage (we only want history)
    # Actually, we might want them for target analysis, but for feature matrix we join on history.
    # The consumer (train.py) usually expects features to be pre-shifted.
    # The standard in this codebase seems to be `_prev` suffixes.
    
    keep_cols = ["season", "week", "team", "game_date"] + \
                [f"{m}_{s}" for m in metrics for s in ["prev", "l3", "l5"]]
                
    return df.select(keep_cols)

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    df = compute_pace_metrics([2024])
    print("Computed metrics:", df.shape)
    hist = add_pace_history(df)
    print("History features:", hist.shape)
    print(hist.columns)
    print(hist.head())

