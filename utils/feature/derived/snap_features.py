"""Snap-related features for NFL player predictions.

This module provides features related to snap participation, expected snaps,
and role stability indicators.

Key insights:
- snap_volatility_cv: Low snap players have 2.4x higher CV (0.85 vs 0.35)
- was_backup_last_game: Backup->8 snaps vs Starter->37 snaps (4.6x diff)
- starter_rate_l5: Low snap=11%, High snap=74% (6.8x difference)
- max_snap_pct_l5: Low snap=31%, High snap=85% (2.7x difference)
"""

from __future__ import annotations

import logging

import polars as pl

__all__ = [
    "add_rolling_snap_features",
    "add_expected_snap_features",
    "add_role_stability_features",
    "add_market_odds_flag",
    "add_snap_features",
]

logger = logging.getLogger(__name__)


def add_rolling_snap_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute rolling snap features across all weeks.
    
    This fixes the issue where per-week parquet files can't compute rolling
    means because each player only appears once per week.
    """
    required_cols = {"player_id", "game_date", "offense_pct"}
    if not required_cols.issubset(set(df.columns)):
        return df
    
    logger.info("Computing rolling snap features across all weeks...")
    df = df.sort(["player_id", "game_date"])
    
    snap_pct_cols = [col for col in ("offense_pct", "defense_pct", "st_pct") if col in df.columns]
    rolling_exprs = []
    lag_exprs = []
    
    for col in snap_pct_cols:
        feature_base = col.replace("_pct", "")
        # Previous game value
        lag_exprs.append(
            pl.col(col)
            .shift(1)
            .over(["player_id"])
            .alias(f"snap_{feature_base}_pct_prev")
        )
        # Rolling 3-game mean (shifted to avoid leakage)
        rolling_exprs.append(
            pl.col(col)
            .rolling_mean(window_size=3, min_samples=1)
            .shift(1)
            .over(["player_id"])
            .alias(f"snap_{feature_base}_pct_l3")
        )
    
    if lag_exprs:
        df = df.with_columns(lag_exprs)
    if rolling_exprs:
        df = df.with_columns(rolling_exprs)
    
    # Update has_no_snap_history based on the computed rolling feature
    if "snap_offense_pct_l3" in df.columns:
        df = df.with_columns(
            pl.col("snap_offense_pct_l3")
            .is_null()
            .cast(pl.Int8)
            .alias("has_no_snap_history")
        )
    
    # Log coverage
    total = df.height
    with_history = df.filter(pl.col("snap_offense_pct_l3").is_not_null()).height if "snap_offense_pct_l3" in df.columns else 0
    logger.info("    Rolling snap features: %d/%d rows have history (%.1f%%)", with_history, total, 100 * with_history / total if total > 0 else 0)
    
    return df


def add_expected_snap_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add expected snaps features based on team context + player usage.
    
    This converts relative snap % to absolute expected snaps.
    Key insight: FBs have 22% snap rate but only ~14 actual snaps
    because their team only uses them for specific situations.
    
    PARITY: If expected_snaps features already exist with values, skip computation.
    """
    expected_snaps_cols = {"team_ctx_offensive_plays_l3", "snap_offense_pct_l3"}
    if not expected_snaps_cols.issubset(set(df.columns)):
        return df
    
    cols = set(df.columns)
    
    # PARITY: Skip if expected_snaps features already exist with values
    if ("expected_snaps_l3" in cols and 
        df.filter(pl.col("expected_snaps_l3").is_not_null()).height > 0):
        logger.debug("Skipping expected snap feature computation - already present")
        return df
    
    logger.info("Adding expected snaps features...")
    
    exprs = []
    
    # Expected total snaps = team plays * player snap %
    if "expected_snaps_l3" not in cols:
        exprs.append(
            (pl.col("team_ctx_offensive_plays_l3") * pl.col("snap_offense_pct_l3"))
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("expected_snaps_l3")
        )
    
    # Also add previous game version
    if {"team_ctx_offensive_plays_prev", "snap_offense_pct_prev"} <= cols:
        if "expected_snaps_prev" not in cols:
            exprs.append(
                (pl.col("team_ctx_offensive_plays_prev") * pl.col("snap_offense_pct_prev"))
                .fill_null(0.0)
                .cast(pl.Float32)
                .alias("expected_snaps_prev")
            )

    # Return-specialist flag (deterministic; no imputation).
    # High ST snap share + very low offensive snap share suggests return role.
    if {"snap_st_pct_l3", "snap_offense_pct_l3"} <= cols and "is_return_specialist" not in cols:
        exprs.append(
            (
                pl.col("snap_st_pct_l3").is_not_null()
                & pl.col("snap_offense_pct_l3").is_not_null()
                & (pl.col("snap_st_pct_l3") >= 0.25)
                & (pl.col("snap_offense_pct_l3") <= 0.15)
            )
            .cast(pl.Int8)
            .alias("is_return_specialist")
        )
    
    # Expected red zone snaps
    rz_cols = {"team_ctx_offensive_plays_l3", "team_ctx_red_zone_play_rate_l3", 
               "hist_red_zone_carry_share_l3", "hist_red_zone_target_share_l3"}
    if rz_cols <= set(df.columns):
        exprs.append(
            (
                pl.col("team_ctx_offensive_plays_l3") *
                pl.col("team_ctx_red_zone_play_rate_l3") *
                (pl.col("hist_red_zone_carry_share_l3").fill_null(0) + pl.col("hist_red_zone_target_share_l3").fill_null(0))
            )
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("expected_rz_touches_l3")
        )
    
    # Expected goal line snaps
    gl_cols = {"team_ctx_offensive_plays_l3", "team_ctx_goal_to_go_play_rate_l3",
               "hist_goal_to_go_carry_share_l3", "hist_goal_to_go_target_share_l3"}
    if gl_cols <= set(df.columns):
        exprs.append(
            (
                pl.col("team_ctx_offensive_plays_l3") *
                pl.col("team_ctx_goal_to_go_play_rate_l3") *
                (pl.col("hist_goal_to_go_carry_share_l3").fill_null(0) + pl.col("hist_goal_to_go_target_share_l3").fill_null(0))
            )
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("expected_gl_touches_l3")
        )
    
    if exprs:
        df = df.with_columns(exprs)
        if "expected_snaps_l3" in df.columns:
            exp_snaps_mean = df.get_column("expected_snaps_l3").mean()
            logger.info("    Added %d expected snap features (expected_snaps_l3 mean=%.1f)",
                        len(exprs), exp_snaps_mean)
    
    return df


def add_role_stability_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add role stability features for specialist detection.
    
    These help distinguish "sometimes starter" from "consistent starter".
    """
    if "offense_pct" not in df.columns or "player_id" not in df.columns:
        return df
    
    logger.info("Adding role stability features...")
    df = df.sort(["player_id", "game_date"])
    
    exprs = []
    cols = set(df.columns)
    
    # Snap volatility (CV = std / mean)
    if "snap_offense_pct_l3" in cols:
        exprs.append(
            pl.col("offense_pct")
            .rolling_std(window_size=5, min_samples=2)
            .shift(1)
            .over(["player_id"])
            .alias("snap_pct_std_l5")
        )
        exprs.append(
            pl.col("offense_pct")
            .rolling_mean(window_size=5, min_samples=2)
            .shift(1)
            .over(["player_id"])
            .alias("snap_pct_mean_l5")
        )
    
    # ABSOLUTE snap history (not %) - critical for specialists
    # FB hist_snaps_l3 mean=14.2 vs actual=14.1 (very accurate!)
    if "snaps_label" in cols:
        exprs.append(
            pl.col("snaps_label")
            .rolling_mean(window_size=3, min_samples=1)
            .shift(1)
            .over(["player_id"])
            .cast(pl.Float32)
            .alias("hist_snaps_l3")
        )
        exprs.append(
            pl.col("snaps_label")
            .shift(1)
            .over(["player_id"])
            .cast(pl.Float32)
            .alias("hist_snaps_prev")
        )
    
    # Was backup last game (snap_pct < 20%)
    if "snap_offense_pct_prev" in cols:
        exprs.append(
            (pl.col("snap_offense_pct_prev") < 0.20)
            .cast(pl.Int8)
            .alias("was_backup_last_game")
        )
    
    # Starter rate (% of last 5 games with >50% snaps)
    exprs.append(
        (pl.col("offense_pct") > 0.5)
        .cast(pl.Int8)
        .rolling_sum(window_size=5, min_samples=1)
        .shift(1)
        .over(["player_id"])
        .truediv(5.0)
        .alias("starter_rate_l5")
    )
    
    # Max snap % in window (role ceiling)
    exprs.append(
        pl.col("offense_pct")
        .rolling_max(window_size=5, min_samples=1)
        .shift(1)
        .over(["player_id"])
        .alias("max_snap_pct_l5")
    )
    
    # Min snap % in window (role floor)
    exprs.append(
        pl.col("offense_pct")
        .rolling_min(window_size=5, min_samples=1)
        .shift(1)
        .over(["player_id"])
        .alias("min_snap_pct_l5")
    )
    
    if exprs:
        df = df.with_columns(exprs)
        
        # Calculate CV after we have std and mean
        if {"snap_pct_std_l5", "snap_pct_mean_l5"} <= set(df.columns):
            df = df.with_columns(
                (pl.col("snap_pct_std_l5") / pl.col("snap_pct_mean_l5").clip(0.01, None))
                .cast(pl.Float32)
                .alias("snap_volatility_cv_l5")
            )
        
        # Snap ceiling = max_snap_pct_l5 * team_plays (absolute upper bound)
        if {"max_snap_pct_l5", "team_ctx_offensive_plays_l3"} <= set(df.columns):
            df = df.with_columns(
                (pl.col("max_snap_pct_l5") * pl.col("team_ctx_offensive_plays_l3"))
                .cast(pl.Float32)
                .alias("snap_ceiling_l5")
            )
            ceiling_mean = df.get_column("snap_ceiling_l5").drop_nulls().mean()
            logger.info("    snap_ceiling_l5 added (mean=%.1f)", ceiling_mean)
        
        # Capped expected snaps = min(expected_snaps, ceiling)
        if {"expected_snaps_l3", "snap_ceiling_l5"} <= set(df.columns):
            df = df.with_columns(
                pl.min_horizontal("expected_snaps_l3", "snap_ceiling_l5")
                .cast(pl.Float32)
                .alias("capped_expected_snaps_l3")
            )
            capped_mean = df.get_column("capped_expected_snaps_l3").drop_nulls().mean()
            logger.info("    capped_expected_snaps_l3 added (mean=%.1f)", capped_mean)
        
        # Log coverage
        starter_rate_coverage = df.filter(pl.col("starter_rate_l5").is_not_null()).height
        logger.info("    Role stability features: %d/%d rows have starter_rate_l5 (%.1f%%)",
                    starter_rate_coverage, df.height,
                    100 * starter_rate_coverage / df.height if df.height > 0 else 0)
    
    return df


def add_market_odds_flag(df: pl.DataFrame) -> pl.DataFrame:
    """Add binary flag for market odds presence.
    
    Null market odds = player likely not starting (92.5% of QBs with odds play).
    This makes the null signal explicit rather than treating it as missing data.
    """
    if "market_anytime_td_spread" not in df.columns:
        return df
    
    logger.info("Adding has_market_odds binary flag...")
    df = df.with_columns(
        pl.col("market_anytime_td_spread")
        .is_not_null()
        .cast(pl.Int8)
        .alias("has_market_odds")
    )
    
    with_odds = df.filter(pl.col("has_market_odds") == 1).height
    total = df.height
    logger.info("    has_market_odds: %d/%d rows have odds (%.1f%%)", with_odds, total, 100 * with_odds / total if total > 0 else 0)
    
    return df


def add_snap_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all snap-related features.
    
    This is the main entry point that applies all snap feature functions.
    Returns the dataframe with all snap features added.
    """
    before_cols = len(df.columns)
    
    df = add_rolling_snap_features(df)
    df = add_expected_snap_features(df)
    df = add_role_stability_features(df)
    df = add_market_odds_flag(df)
    
    added = len(df.columns) - before_cols
    logger.info("    Total snap-related features added: %d", added)
    
    return df

