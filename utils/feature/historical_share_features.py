"""Historical usage share features for NFL player predictions.

This module computes player-level historical share features (e.g., target share,
carry share) based on prior games. These are crucial for understanding player
roles within their teams.
"""

from __future__ import annotations

import logging

import polars as pl

__all__ = [
    "add_historical_share_features",
    "add_combined_usage_features",
    "add_role_share_flags",
]

logger = logging.getLogger(__name__)


def add_historical_share_features(df: pl.DataFrame) -> pl.DataFrame:
    """Derive historical usage share features.
    
    Computes share features like hist_target_share_prev, hist_carry_share_l3, etc.
    based on player's rolling stats relative to team totals.
    """
    logger.info("Deriving historical usage share features...")
    
    share_exprs: list[pl.Expr] = []
    share_specs = [
        ("target", "targets"),
        ("carry", "carries"),
        ("pass_attempt", "pass_attempts"),
        ("red_zone_target", "red_zone_targets"),
        ("red_zone_carry", "red_zone_carries"),
        ("goal_to_go_target", "goal_to_go_targets"),
        ("goal_to_go_carry", "goal_to_go_carries"),
    ]
    available_cols = set(df.columns)
    
    for stat, team_metric in share_specs:
        player_prev = f"1g_{stat}_per_game"
        player_l3 = f"3g_{stat}_per_game"
        team_prev = f"team_ctx_{team_metric}_prev"
        team_l3 = f"team_ctx_{team_metric}_l3"
        
        if {player_prev, team_prev} <= available_cols:
            share_exprs.append(
                pl.when(pl.col(team_prev) > 0)
                .then(pl.col(player_prev) / pl.col(team_prev))
                .otherwise(0.0)
                .cast(pl.Float32)
                .alias(f"hist_{stat}_share_prev")
            )
        
        if {player_l3, team_l3} <= available_cols:
            share_exprs.append(
                pl.when(pl.col(team_l3) > 0)
                .then(pl.col(player_l3) / pl.col(team_l3))
                .otherwise(0.0)
                .cast(pl.Float32)
                .alias(f"hist_{stat}_share_l3")
            )
    
    if share_exprs:
        df = df.with_columns(share_exprs)
        logger.info("Added %d historical share features", len(share_exprs))
    
    return df


def add_combined_usage_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add combined usage features for specialist role identification.
    
    FBs have ~2% combined usage vs 24% for regular RBs - very discriminating.
    Low usage players (<5%) average only 13 snaps.
    """
    usage_cols_present = {"hist_carry_share_l3", "hist_target_share_l3"} <= set(df.columns)
    if not usage_cols_present:
        return df
    
    logger.info("Adding combined usage role features...")
    
    df = df.with_columns([
        # Combined carry + target share (strong role indicator)
        (pl.col("hist_carry_share_l3").fill_null(0) + pl.col("hist_target_share_l3").fill_null(0))
        .alias("combined_usage_share_l3"),
        # Low usage flag: <5% combined share = specialist/backup role
        (
            (pl.col("hist_carry_share_l3").fill_null(0) + pl.col("hist_target_share_l3").fill_null(0)) < 0.05
        ).cast(pl.Int8).alias("is_low_usage_role"),
    ])
    
    low_usage_count = df.filter(pl.col("is_low_usage_role") == 1).height
    logger.info("    is_low_usage_role: %d rows flagged (%.1f%%)",
                low_usage_count, 100 * low_usage_count / df.height if df.height > 0 else 0)
    
    return df


def add_role_share_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Add heuristic role flags from historical shares.
    
    Uses 70th percentile thresholds within team-position groups to identify
    primary red-zone targets and goal-line backs.
    """
    role_share_specs = [
        ("hist_red_zone_target_share_l3", "role_primary_red_zone_target"),
        ("hist_goal_to_go_carry_share_l3", "role_goal_line_back"),
    ]
    
    group_cols = [col for col in ("team", "position") if col in df.columns]
    if not group_cols:
        return df
    
    for share_col, flag_name in role_share_specs:
        if share_col not in df.columns:
            continue
        
        thresholds = (
            df.select(group_cols + [share_col])
            .drop_nulls()
            .group_by(group_cols)
            .agg(
                pl.col(share_col)
                .quantile(0.7, interpolation="higher")
                .alias(f"__{flag_name}_threshold")
            )
        )
        
        # Global fallback if a group has too few samples
        fallback_expr = df.select(
            pl.col(share_col).fill_null(0.0).quantile(0.7, interpolation="higher")
        )
        fallback = fallback_expr.item() if fallback_expr.height else 0.0
        
        if thresholds.is_empty():
            df = df.with_columns(
                pl.when(pl.col(share_col).is_not_null())
                .then(pl.col(share_col).fill_null(0.0) >= pl.lit(fallback))
                .otherwise(0)
                .cast(pl.Int8)
                .alias(flag_name)
            )
            continue
        
        df = df.join(thresholds, on=group_cols, how="left")
        threshold_col = f"__{flag_name}_threshold"
        df = df.with_columns(
            pl.when(pl.col(share_col).is_not_null())
            .then(
                pl.col(share_col).fill_null(0.0)
                >= pl.col(threshold_col).fill_null(fallback)
            )
            .otherwise(0)
            .cast(pl.Int8)
            .alias(flag_name)
        )
        
        flagged = df.filter(pl.col(flag_name) == 1).height
        logger.info(
            "Role flag %s: %d rows flagged using 70th percentile threshold.",
            flag_name,
            flagged,
        )
        df = df.drop(threshold_col, strict=False)
    
    return df


def drop_leakage_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop same-game leakage columns (current-game share values).
    
    These columns would leak current-game info which is unavailable at prediction time.
    """
    leakage_cols = [
        "target_share",
        "carry_share",
        "pass_attempt_share",
        "red_zone_target_share",
        "red_zone_carry_share",
        "goal_to_go_target_share",
        "goal_to_go_carry_share",
    ]
    return df.drop(leakage_cols, strict=False)

