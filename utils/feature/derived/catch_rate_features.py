"""Catch rate efficiency features for NFL player predictions.

Historical catch rate is crucial for efficiency_rec_success model.
Without it, model can only predict the mean (~0.70).

Key insight: Catch rate varies significantly by:
- Position: RB=0.77, TE=0.70, WR=0.62
- Player skill (hands, route running)
- Target depth (short passes have higher catch rate)
"""

from __future__ import annotations

import logging

import polars as pl

__all__ = [
    "add_catch_rate_features",
    "compute_historical_catch_rates",
    "add_position_catch_rate_priors",
]

logger = logging.getLogger(__name__)


def compute_historical_catch_rates(df: pl.DataFrame) -> pl.DataFrame:
    """Compute historical catch rate features from rolling statistics.
    
    Computes:
    - hist_catch_rate_career: Career catch rate
    - hist_catch_rate_l3: Rolling 3-game catch rate
    - hist_catch_rate_l5: Rolling 5-game catch rate
    - hist_catch_rate_prev: Previous game catch rate
    - hist_targets_count_l3/l5: Target volume for uncertainty estimation
    
    Note: If hist_catch_rate_* columns already exist (pre-computed from historical data),
    they are preserved to maintain parity between training and inference.
    """
    if "reception" not in df.columns or "target" not in df.columns:
        return df
    
    # Check if key historical columns already exist with valid values
    cols = set(df.columns)
    already_have_hist = (
        "hist_catch_rate_l3" in cols and 
        "hist_catch_rate_l5" in cols and
        df.filter(pl.col("hist_catch_rate_l3").is_not_null()).height > 0
    )
    
    if already_have_hist:
        logger.debug("Skipping hist catch rate computation - already present from historical data")
        return df
    
    logger.info("Computing historical catch rate features...")
    
    # Sort by player and game for proper rolling calculations
    df = df.sort(["player_id", "game_date"])
    
    # Compute cumulative receptions and targets per player
    df = df.with_columns([
        pl.col("reception").fill_null(0).cum_sum().over("player_id").alias("_cum_rec"),
        pl.col("target").fill_null(0).cum_sum().over("player_id").alias("_cum_target"),
    ])
    
    # Shift to get "prior" values (before this game)
    df = df.with_columns([
        (pl.col("_cum_rec").shift(1).over("player_id").fill_null(0)).alias("_prior_rec"),
        (pl.col("_cum_target").shift(1).over("player_id").fill_null(0)).alias("_prior_target"),
    ])
    
    # Compute career catch rate
    df = df.with_columns([
        (pl.col("_prior_rec") / (pl.col("_prior_target") + 1e-6))
        .clip(0, 1.0)
        .cast(pl.Float32)
        .alias("hist_catch_rate_career")
    ])
    
    # Compute L3 and L5 rolling catch rates
    df = df.with_columns([
        pl.col("reception").fill_null(0)
        .rolling_sum(window_size=3, min_samples=1)
        .shift(1)
        .over("player_id")
        .alias("_rec_l3"),
        pl.col("target").fill_null(0)
        .rolling_sum(window_size=3, min_samples=1)
        .shift(1)
        .over("player_id")
        .alias("_target_l3"),
        pl.col("reception").fill_null(0)
        .rolling_sum(window_size=5, min_samples=1)
        .shift(1)
        .over("player_id")
        .alias("_rec_l5"),
        pl.col("target").fill_null(0)
        .rolling_sum(window_size=5, min_samples=1)
        .shift(1)
        .over("player_id")
        .alias("_target_l5"),
        # Previous game values
        pl.col("reception").fill_null(0).shift(1).over("player_id").alias("_rec_prev"),
        pl.col("target").fill_null(0).shift(1).over("player_id").alias("_target_prev"),
    ])
    
    # Compute catch rates
    df = df.with_columns([
        (pl.col("_rec_l3").fill_null(0) / (pl.col("_target_l3").fill_null(0) + 1e-6))
        .clip(0, 1.0)
        .cast(pl.Float32)
        .alias("hist_catch_rate_l3"),
        (pl.col("_rec_l5").fill_null(0) / (pl.col("_target_l5").fill_null(0) + 1e-6))
        .clip(0, 1.0)
        .cast(pl.Float32)
        .alias("hist_catch_rate_l5"),
        (pl.col("_rec_prev").fill_null(0) / (pl.col("_target_prev").fill_null(0) + 1e-6))
        .clip(0, 1.0)
        .cast(pl.Float32)
        .alias("hist_catch_rate_prev"),
    ])
    
    # Target volume for uncertainty (low targets = high variance)
    df = df.with_columns([
        pl.col("_target_l3").fill_null(0).cast(pl.Int32).alias("hist_targets_count_l3"),
        pl.col("_target_l5").fill_null(0).cast(pl.Int32).alias("hist_targets_count_l5"),
    ])
    
    # Drop temporary columns
    df = df.drop([
        "_cum_rec", "_cum_target", "_prior_rec", "_prior_target",
        "_rec_l3", "_target_l3", "_rec_l5", "_target_l5", "_rec_prev", "_target_prev"
    ])
    
    logger.info("    Added historical catch rate features: hist_catch_rate_l3, hist_catch_rate_l5, hist_catch_rate_prev, hist_catch_rate_career")
    
    return df


def add_position_catch_rate_priors(df: pl.DataFrame) -> pl.DataFrame:
    """Add position-based catch rate prior.
    
    RBs catch almost everything (checkdowns), WRs have more drops.
    """
    if "position" not in df.columns:
        return df
    
    exprs = []
    
    exprs.append(
        pl.when(pl.col("position") == "RB").then(0.77)
        .when(pl.col("position") == "TE").then(0.70)
        .when(pl.col("position") == "WR").then(0.62)
        .when(pl.col("position") == "FB").then(0.75)
        .otherwise(0.70)  # Default
        .cast(pl.Float32)
        .alias("position_catch_rate_prior")
    )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_shrunk_catch_rate(df: pl.DataFrame) -> pl.DataFrame:
    """Add shrunk catch rate estimate (blend player history with position prior).
    
    Use more prior for low-volume players, more history for high-volume.
    At 10 targets, roughly 50% history, 50% position prior.
    """
    if "position" not in df.columns or "hist_catch_rate_l5" not in df.columns:
        return df
    
    if "hist_targets_count_l5" not in df.columns:
        return df
    
    # Position priors
    def _pos_prior() -> pl.Expr:
        return (
            pl.when(pl.col("position") == "RB").then(0.77)
            .when(pl.col("position") == "TE").then(0.70)
            .when(pl.col("position") == "WR").then(0.62)
            .otherwise(0.70)
        )
    
    df = df.with_columns([
        pl.when(pl.col("hist_targets_count_l5") >= 20)
        .then(pl.col("hist_catch_rate_l5"))  # Trust history for high-volume
        .when(pl.col("hist_targets_count_l5") >= 5)
        .then(pl.col("hist_catch_rate_l5") * 0.7 + _pos_prior() * 0.3)  # Blend for medium-volume
        .otherwise(_pos_prior())  # Use position prior for low-volume
        .cast(pl.Float32)
        .alias("shrunk_catch_rate_estimate")
    ])
    
    return df


def add_depth_adjusted_catch_rate(df: pl.DataFrame) -> pl.DataFrame:
    """Add target depth impact on catch rate.
    
    Deeper targets have lower catch rates.
    Each yard of depth reduces catch rate by ~1.5%.
    """
    if "hist_air_yards_per_target_l3" not in df.columns:
        return df
    
    df = df.with_columns([
        (0.70 - pl.col("hist_air_yards_per_target_l3").fill_null(5) * 0.015)
        .clip(0.40, 0.90)
        .cast(pl.Float32)
        .alias("depth_adjusted_catch_rate_prior")
    ])
    
    return df


def add_catch_rate_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all catch rate efficiency features.
    
    This is the main entry point that applies all catch rate feature functions.
    Returns the dataframe with all catch rate features added.
    """
    before_cols = len(df.columns)
    
    df = compute_historical_catch_rates(df)
    df = add_position_catch_rate_priors(df)
    df = add_shrunk_catch_rate(df)
    df = add_depth_adjusted_catch_rate(df)
    
    added = len(df.columns) - before_cols
    logger.info("    Added %d catch rate efficiency features", added)
    
    return df

