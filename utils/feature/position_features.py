"""Position-specific and MoE (Mixture of Experts) features for NFL player predictions.

This module provides features that help per-position models by providing explicit role signals.
These features are added AFTER rolling features are computed so we have hist_carry_share_l3 etc.

Features include:
- MoE routing (position_group for RB, WR, TE, QB)
- RB role features (lead back, committee, receiving, goal-line)
- TE role features (receiving TE, blocking TE, target concentration)
- Specialist role flags (fullback, special teams only)
"""

from __future__ import annotations

import logging

import polars as pl

__all__ = [
    "add_position_group",
    "add_specialist_role_flags",
    "add_moe_position_features",
]

logger = logging.getLogger(__name__)


def add_position_group(df: pl.DataFrame) -> pl.DataFrame:
    """Add position_group field for per-position Mixture of Experts modeling.
    
    Maps all positions to RB, WR, TE, QB for consistent model routing.
    """
    if "position" not in df.columns:
        return df
    
    logger.info("Adding position_group for MoE model routing...")
    
    df = df.with_columns([
        pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
        .then(pl.lit("RB"))
        .when(pl.col("position") == "WR")
        .then(pl.lit("WR"))
        .when(pl.col("position") == "TE")
        .then(pl.lit("TE"))
        .when(pl.col("position") == "QB")
        .then(pl.lit("QB"))
        .otherwise(pl.lit("WR"))  # Default unknown positions to WR
        .alias("position_group")
    ])
    
    # Log position_group distribution
    pos_counts = df.get_column("position_group").value_counts().sort("count", descending=True)
    logger.info("    position_group distribution:")
    for row in pos_counts.iter_rows():
        logger.info("      %s: %d rows", row[0], row[1])
    
    return df


def add_specialist_role_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Add specialist role flags for snap prediction.
    
    Fullbacks average 14 snaps vs 24 for regular RBs - strong signal.
    """
    if "depth_chart_position" not in df.columns:
        return df
    
    logger.info("Adding specialist role flags...")
    
    df = df.with_columns([
        # Binary FB flag - FBs have very different snap patterns
        (pl.col("depth_chart_position") == "FB")
        .cast(pl.Int8)
        .alias("is_fullback"),
        # Punters/Kickers rarely get offensive snaps
        pl.col("depth_chart_position")
        .is_in(["P", "K"])
        .cast(pl.Int8)
        .alias("is_special_teams_only"),
    ])
    
    fb_count = df.filter(pl.col("is_fullback") == 1).height
    logger.info("    is_fullback: %d rows flagged", fb_count)
    
    return df


def add_moe_rb_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add RB-specific MoE features.
    
    These help the RB expert model understand player roles.
    """
    cols = set(df.columns)
    exprs = []
    
    # rb_is_lead_back (carry share > 40%) - 0.536 correlation with snaps!
    if {"hist_carry_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((pl.col("hist_carry_share_l3").fill_null(0.0) > 0.40).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_lead_back")
        )
        
        # rb_is_committee_back (carry share 15-40%)
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then(
                ((pl.col("hist_carry_share_l3").fill_null(0.0) >= 0.15) &
                 (pl.col("hist_carry_share_l3").fill_null(0.0) <= 0.40)).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_committee_back")
        )
    
    # rb_is_receiving_back (target share > carry share)
    if {"hist_target_share_l3", "hist_carry_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then(
                (pl.col("hist_target_share_l3").fill_null(0.0) >
                 pl.col("hist_carry_share_l3").fill_null(0.0)).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_receiving_back")
        )
    
    # rb_is_goal_line_rb (goal line carry share > 20%)
    if {"hist_goal_to_go_carry_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((pl.col("hist_goal_to_go_carry_share_l3").fill_null(0.0) > 0.20).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_goal_line_rb")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_moe_te_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add TE-specific MoE features.
    
    These help the TE expert model distinguish receiving TEs from blocking TEs.
    """
    cols = set(df.columns)
    exprs = []
    
    # te_is_receiving_te (target share > 10%) - 0.524 correlation!
    if {"hist_target_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then((pl.col("hist_target_share_l3").fill_null(0.0) > 0.10).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_receiving_te")
        )
    
    # te_is_blocking_te (high snap, low target)
    if {"snap_offense_pct_l3", "hist_target_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                ((pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.40) &
                 (pl.col("hist_target_share_l3").fill_null(0.0) < 0.08)).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_blocking_te")
        )
    
    # te_target_concentration (target share / snap share)
    if {"hist_target_share_l3", "snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                pl.col("hist_target_share_l3").fill_null(0.0) /
                (pl.col("snap_offense_pct_l3").fill_null(0.01) + 0.01)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_target_concentration")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_moe_position_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all MoE position-specific features.
    
    This is the main entry point that applies all position feature functions.
    Returns the dataframe with all MoE features added.
    """
    logger.info("Adding MoE position-specific features...")
    
    before_cols = len(df.columns)
    
    df = add_moe_rb_features(df)
    df = add_moe_te_features(df)
    
    added = len(df.columns) - before_cols
    logger.info("    Added %d MoE position-specific features", added)
    
    return df

