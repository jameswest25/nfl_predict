"""Usage model helper features for NFL player predictions.

This module provides features that help predict player usage (targets, carries, snaps).
Key insights addressed:
- Inactive players get high usage predictions because hist_target_share dominates
- QBs predicted to receive targets (99.2% have 0 targets)
- hist_target_share_l3 (importance 1.60) dominates pred_snaps (0.81)

Features include:
- Position eligibility flags (QBs and OL don't receive targets)
- Inactive-adjusted historical usage
- Snap-weighted usage interactions
- Position-based baselines and caps
- Zero-snap indicators
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass

__all__ = [
    "add_usage_helper_features",
    "add_position_eligibility_flags",
    "add_inactive_adjusted_usage",
    "add_snap_weighted_usage",
    "add_position_baselines",
    "add_zero_usage_indicators",
    "add_usage_carry_enhancements",
]

logger = logging.getLogger(__name__)


def add_position_eligibility_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Add binary flags for position eligibility in receiving/rushing.
    
    QBs, OL, K, P, LS don't receive targets (non-receiving positions).
    WR, TE are primary receiving positions.
    RBs can receive but at lower rates.
    """
    if "position" not in df.columns:
        return df
    
    exprs = [
        pl.col("position").is_in(["QB", "OL", "K", "P", "LS"]).cast(pl.Int8).alias("is_non_receiving_position"),
        pl.col("position").is_in(["WR", "TE"]).cast(pl.Int8).alias("is_primary_receiving_position"),
        pl.col("position").is_in(["RB"]).cast(pl.Int8).alias("is_rb_position"),
    ]
    return df.with_columns(exprs)


def add_inactive_adjusted_usage(df: pl.DataFrame) -> pl.DataFrame:
    """Add usage signals adjusted for inactive status.
    
    When is_inactive=1, zero out the historical share signal.
    This prevents inactive players from getting high usage predictions.
    """
    if "is_inactive" not in df.columns:
        return df
    
    exprs = []
    
    if "hist_target_share_l3" in df.columns:
        exprs.append(
            (pl.col("hist_target_share_l3") * (1 - pl.col("is_inactive")))
            .cast(pl.Float32)
            .alias("active_adjusted_target_share_l3")
        )
    
    if "hist_carry_share_l3" in df.columns:
        exprs.append(
            (pl.col("hist_carry_share_l3") * (1 - pl.col("is_inactive")))
            .cast(pl.Float32)
            .alias("active_adjusted_carry_share_l3")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_snap_weighted_usage(df: pl.DataFrame) -> pl.DataFrame:
    """Add snap-weighted usage interactions.
    
    These dampen historical usage signal when expected snaps are low.
    Critical: if expected_snaps=5 and hist_target_share=0.20, the weighted
    value will be much lower than if expected_snaps=50.
    """
    # Find the best snap column to use
    snap_col = None
    for col in ["expected_snaps_l3", "snap_offense_pct_l3"]:
        if col in df.columns:
            snap_col = col
            break
    
    if snap_col is None:
        return df
    
    exprs = []
    
    if "hist_target_share_l3" in df.columns:
        exprs.append(
            (pl.col(snap_col) * pl.col("hist_target_share_l3"))
            .cast(pl.Float32)
            .alias("snap_weighted_target_share")
        )
    
    if "hist_carry_share_l3" in df.columns:
        exprs.append(
            (pl.col(snap_col) * pl.col("hist_carry_share_l3"))
            .cast(pl.Float32)
            .alias("snap_weighted_carry_share")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_position_baselines(df: pl.DataFrame) -> pl.DataFrame:
    """Add position-based baseline and cap features.
    
    These encode the expected usage by position, giving the model a strong prior.
    Based on data analysis:
    - QBs: 99.2% have 0 targets, mean = 0.0003
    - WRs: 64% have >0 targets, mean = 0.10
    - TEs: 57% have >0 targets, mean = 0.07
    - RBs: 52% have >0 targets, mean = 0.04
    """
    if "position" not in df.columns:
        return df
    
    exprs = []
    
    # Position-capped target share (for QBs)
    if "hist_target_share_l3" in df.columns:
        exprs.append(
            pl.when(pl.col("position") == "QB")
            .then(pl.col("hist_target_share_l3").clip(0, 0.05))  # Cap QB targets at 5%
            .otherwise(pl.col("hist_target_share_l3"))
            .cast(pl.Float32)
            .alias("position_capped_target_share_l3")
        )
    
    # Position target baseline (expected target share by position)
    exprs.append(
        pl.when(pl.col("position") == "QB").then(0.001)
        .when(pl.col("position") == "WR").then(0.10)
        .when(pl.col("position") == "TE").then(0.07)
        .when(pl.col("position") == "RB").then(0.04)
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("position_target_baseline")
    )
    
    # Position carry baseline
    exprs.append(
        pl.when(pl.col("position") == "QB").then(0.001)  # Gadget QBs only
        .when(pl.col("position") == "RB").then(0.12)
        .when(pl.col("position") == "FB").then(0.02)
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("position_carry_baseline")
    )
    
    # Position-specific maximum target share (caps)
    # Based on data: QB max=0.08, RB max=0.39, WR max=0.62, TE max=0.50
    exprs.append(
        pl.when(pl.col("position") == "QB").then(0.02)   # Near-zero for QBs
        .when(pl.col("position") == "RB").then(0.20)     # RBs rarely exceed 20%
        .when(pl.col("position") == "WR").then(0.40)     # WRs can hit 40%
        .when(pl.col("position") == "TE").then(0.30)     # TEs can hit 30%
        .otherwise(0.05)
        .cast(pl.Float32)
        .alias("position_max_target_share")
    )
    
    # Position-adjusted snap signal
    # Snap-to-target correlation varies by position:
    # QB: 0.054 (near zero), WR: 0.651, TE: 0.618, RB: 0.555
    if "snap_offense_pct_l3" in df.columns:
        exprs.append(
            pl.when(pl.col("position") == "QB")
            .then(pl.col("snap_offense_pct_l3") * 0.001)  # Near-zero for QBs
            .when(pl.col("position") == "RB")
            .then(pl.col("snap_offense_pct_l3") * 0.5)    # Half signal for RBs
            .otherwise(pl.col("snap_offense_pct_l3"))     # Full signal for WR/TE
            .cast(pl.Float32)
            .alias("position_adjusted_snap_signal")
        )
    
    # Target opportunity rate (targets per snap)
    if "hist_target_share_l3" in df.columns and "snap_offense_pct_l3" in df.columns:
        exprs.append(
            (pl.col("hist_target_share_l3") / pl.col("snap_offense_pct_l3").clip(0.01, None))
            .clip(0, 2.0)  # Cap at 2x (can't have more targets than snaps)
            .cast(pl.Float32)
            .alias("target_opportunity_rate")
        )
    
    # Expected target share zeroed for non-receivers
    exprs.append(
        pl.when(pl.col("position").is_in(["QB", "OL", "K", "P", "LS"]))
        .then(0.0)  # Hard zero for non-receivers
        .when(pl.col("position") == "RB")
        .then(0.04)  # RB baseline
        .when(pl.col("position") == "TE")
        .then(0.07)  # TE baseline
        .when(pl.col("position") == "WR")
        .then(0.10)  # WR baseline
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("receiver_expected_target_share")
    )
    
    # Expected targets scaled by snap prediction
    if "snap_offense_pct_l3" in df.columns:
        exprs.append(
            pl.when(pl.col("position").is_in(["QB", "OL", "K", "P", "LS"]))
            .then(0.0)  # Zero for non-receivers
            .otherwise(
                pl.col("snap_offense_pct_l3") *
                pl.when(pl.col("position") == "WR").then(0.20)
                .when(pl.col("position") == "TE").then(0.15)
                .when(pl.col("position") == "RB").then(0.08)
                .otherwise(0.0)
            )
            .cast(pl.Float32)
            .alias("receiver_snap_adjusted_expected")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_zero_usage_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Add indicators for likely-zero usage scenarios.
    
    These combine multiple signals that player won't have usage.
    """
    exprs = []
    
    # Zero-snap indicator - strong signal that usage should be 0
    if "snap_offense_pct_l3" in df.columns:
        exprs.append(
            (pl.col("snap_offense_pct_l3") < 0.05).cast(pl.Int8).alias("has_minimal_snap_history")
        )
    
    # Combined activity signal
    inactive_expr = pl.lit(0)
    if "is_inactive" in df.columns:
        inactive_expr = pl.col("is_inactive")
    
    if "snap_offense_pct_l3" in df.columns:
        no_snap_history = pl.col("snap_offense_pct_l3").fill_null(0) < 0.05
        exprs.append(
            (inactive_expr.cast(pl.Int8) | no_snap_history.cast(pl.Int8))
            .cast(pl.Int8)
            .alias("likely_zero_usage")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_rb_usage_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add RB-specific usage features.
    
    Addresses issues like:
    - Receiving back indicator (Ekeler, McCaffrey, Kamara have high target share)
    - Goal-line back indicator (Henry, Chubb, Jacobs have low target share)
    - Backup RB features (different carry share patterns)
    """
    cols = set(df.columns)
    exprs = []
    
    # Receiving back indicator
    if {"position", "hist_target_share_l3"} <= cols:
        exprs.append(
            (
                (pl.col("position") == "RB") &
                (pl.col("hist_target_share_l3") > 0.05)
            ).cast(pl.Int8).alias("is_receiving_back")
        )
        
        # RB target workload tier
        # 0=No targets, 1=Rare, 2=Occasional, 3=Regular, 4=Pass-catching back
        exprs.append(
            pl.when(pl.col("position") != "RB").then(pl.lit(-1))
            .when(pl.col("hist_target_share_l3").fill_null(0) > 0.08).then(pl.lit(4))
            .when(pl.col("hist_target_share_l3").fill_null(0) > 0.05).then(pl.lit(3))
            .when(pl.col("hist_target_share_l3").fill_null(0) > 0.02).then(pl.lit(2))
            .when(pl.col("hist_target_share_l3").fill_null(0) > 0.005).then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("rb_target_workload_tier")
        )
        
        # Pass-catching back (clearer signal)
        exprs.append(
            (
                (pl.col("position") == "RB") &
                (pl.col("hist_target_share_l3").fill_null(0) > 0.05)
            ).cast(pl.Int8).alias("is_pass_catching_back")
        )
    
    # Goal-line back indicator
    if {"position", "hist_carry_share_l3", "hist_target_share_l3"} <= cols:
        exprs.append(
            (
                (pl.col("position") == "RB") &
                (pl.col("hist_carry_share_l3").fill_null(0) > 0.20) &
                (pl.col("hist_target_share_l3").fill_null(0) < 0.04)
            ).cast(pl.Int8).alias("is_goal_line_back")
        )
        
        # RB receiving ratio: target share relative to total usage
        exprs.append(
            pl.when(pl.col("position") == "RB")
            .then(
                pl.col("hist_target_share_l3").fill_null(0) /
                (pl.col("hist_carry_share_l3").fill_null(0) + pl.col("hist_target_share_l3").fill_null(0) + 0.001)
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("rb_receiving_ratio")
        )
    
    # RB target interaction features
    if {"position", "hist_target_share_l3", "snap_offense_pct_l3"} <= cols:
        # RB expected targets
        exprs.append(
            pl.when(pl.col("position") == "RB")
            .then(
                pl.col("snap_offense_pct_l3").fill_null(0) *
                pl.col("hist_target_share_l3").fill_null(0)
            )
            .otherwise(
                pl.col("snap_offense_pct_l3").fill_null(0) *
                pl.col("hist_target_share_l3").fill_null(0)
            )
            .cast(pl.Float32)
            .alias("rb_expected_target_share")
        )
        
        # RB target ceiling
        exprs.append(
            pl.when(pl.col("position") == "RB")
            .then(
                (pl.col("hist_target_share_l3").fill_null(0) * 1.3).clip(0, 0.25)
            )
            .otherwise(pl.lit(1.0))
            .cast(pl.Float32)
            .alias("rb_target_ceiling")
        )
    
    # Backup RB features
    if {"position", "hist_carry_share_l3"} <= cols:
        # Binary backup indicator
        exprs.append(
            (
                (pl.col("position") == "RB") &
                (pl.col("hist_carry_share_l3").fill_null(0) < 0.15)
            ).cast(pl.Int8).alias("is_backup_rb")
        )
        
        # RB workload tier
        # 0=Inactive/Gadget, 1=Backup, 2=Committee, 3=Starter, 4=Elite
        exprs.append(
            pl.when(pl.col("position") != "RB").then(pl.lit(-1))
            .when(pl.col("hist_carry_share_l3").fill_null(0) > 0.40).then(pl.lit(4))
            .when(pl.col("hist_carry_share_l3").fill_null(0) > 0.25).then(pl.lit(3))
            .when(pl.col("hist_carry_share_l3").fill_null(0) > 0.10).then(pl.lit(2))
            .when(pl.col("hist_carry_share_l3").fill_null(0) > 0.03).then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("rb_workload_tier")
        )
    
    # RB snap-to-carry ratio
    if {"position", "hist_carry_share_l3", "snap_offense_pct_l3"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "RB")
            .then(
                (pl.col("hist_carry_share_l3").fill_null(0) /
                 pl.col("snap_offense_pct_l3").clip(0.01, None))
                .clip(0, 2.0)
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("rb_carry_per_snap_ratio")
        )
    
    # RB expected carries
    if {"position", "hist_carry_share_l3", "expected_snaps_l3", "snap_offense_pct_l3"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "RB")
            .then(
                pl.col("expected_snaps_l3").fill_null(0) *
                pl.col("hist_carry_share_l3").fill_null(0) /
                pl.col("snap_offense_pct_l3").fill_null(0.5).clip(0.1, None)
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("rb_expected_carries")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_usage_helper_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all usage helper features to the dataframe.
    
    This is the main entry point that applies all usage feature functions.
    Returns the dataframe with all usage helper features added.
    """
    logger.info("Adding usage model helper features...")
    
    before_cols = len(df.columns)
    
    df = add_position_eligibility_flags(df)
    df = add_inactive_adjusted_usage(df)
    df = add_snap_weighted_usage(df)
    df = add_position_baselines(df)
    df = add_zero_usage_indicators(df)
    df = add_rb_usage_features(df)
    df = add_usage_carry_enhancements(df)
    
    added = len(df.columns) - before_cols
    logger.info("    Added %d usage helper features", added)
    
    return df


def add_usage_carry_enhancements(df: pl.DataFrame) -> pl.DataFrame:
    """Add carry-specific nuance features to aid MoE for usage_carries.
    
    - QB siphon/designed runs: capture QB run share that caps RB carries
    - Carry volatility/trend: simple volatility proxy on carry share
    - Situational carry rates: short-yardage and third-down carry rates
    - Gadget propensity: carries per route for WR/TE gadget roles
    - Team/opp run context: neutral run rate and opponent run-rate allowed
    """
    cols = set(df.columns)
    exprs: list[pl.Expr] = []

    # QB siphon / designed run share (team-level to apply to all positions)
    if {"qb_profile_team_rush_attempts_l3", "qb_profile_team_dropbacks_l3"} <= cols:
        qb_siphon = (
            pl.col("qb_profile_team_rush_attempts_l3").fill_null(0.0)
            / (pl.col("qb_profile_team_dropbacks_l3").fill_null(0.01) + 0.01)
        ).clip(0.0, 1.0)
        exprs.append(qb_siphon.cast(pl.Float32).alias("qb_carry_siphon_l3"))

        if "qb_profile_team_scramble_rate_l3" in cols:
            exprs.append(
                (qb_siphon - pl.col("qb_profile_team_scramble_rate_l3").fill_null(0.0))
                .clip(0.0, 1.0)
                .cast(pl.Float32)
                .alias("qb_designed_run_share_l3")
            )
    # Expose player-level scramble rate as a lightweight feature
    if "qb_profile_scramble_rate_l3" in cols:
        exprs.append(
            pl.col("qb_profile_scramble_rate_l3")
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("qb_scramble_rate_l3")
        )

    # Carry-share trend & volatility proxy (carry-specific, not snaps)
    if {"hist_carry_share_prev", "hist_carry_share_l3"} <= cols:
        trend = (
            pl.col("hist_carry_share_prev").fill_null(0.0)
            - pl.col("hist_carry_share_l3").fill_null(0.0)
        )
        exprs.append(trend.cast(pl.Float32).alias("carry_share_trend"))
        exprs.append(
            (trend.abs() / (pl.col("hist_carry_share_l3").fill_null(0.001).abs() + 0.001))
            .clip(0.0, 5.0)
            .cast(pl.Float32)
            .alias("carry_share_cv_like")
        )

    # Situational carry rates from rolling counts
    if {"3g_short_yard_carry_per_game", "3g_carry_per_game"} <= cols:
        exprs.append(
            (
                pl.col("3g_short_yard_carry_per_game").fill_null(0.0)
                / (pl.col("3g_carry_per_game").fill_null(0.0) + 0.001)
            )
            .clip(0.0, 1.0)
            .cast(pl.Float32)
            .alias("hist_short_yard_carry_rate_l3")
        )
    if {"3g_third_down_carry_per_game", "3g_carry_per_game"} <= cols:
        exprs.append(
            (
                pl.col("3g_third_down_carry_per_game").fill_null(0.0)
                / (pl.col("3g_carry_per_game").fill_null(0.0) + 0.001)
            )
            .clip(0.0, 1.0)
            .cast(pl.Float32)
            .alias("hist_third_down_carry_rate_l3")
        )

    # Gadget/jet proxy: carries per route participation
    if {"hist_carry_share_l3", "ps_hist_route_participation_pct_l3"} <= cols:
        exprs.append(
            (
                pl.col("hist_carry_share_l3").fill_null(0.0)
                / (pl.col("ps_hist_route_participation_pct_l3").fill_null(0.01) + 0.01)
            )
            .clip(0.0, 3.0)
            .cast(pl.Float32)
            .alias("hist_carry_per_route_l3")
        )
    if {"hist_carry_share_l3", "hist_target_share_l3", "position"} <= cols:
        exprs.append(
            (
                (pl.col("hist_carry_share_l3").fill_null(0.0) > 0.0)
                & (pl.col("hist_target_share_l3").fill_null(0.0) < 0.12)
                & (pl.col("position").is_in(["WR", "TE"]))
            )
            .cast(pl.Int8)
            .alias("wr_gadget_flag")
        )

    # Team/opp run bias context (carry-specific)
    if "team_ctx_rush_rate_l3" in cols:
        exprs.append(
            pl.col("team_ctx_rush_rate_l3")
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("team_ctx_run_rate_neutral_l3")
        )
    if "team_ctx_rush_rate_prev" in cols:
        exprs.append(
            pl.col("team_ctx_rush_rate_prev")
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("team_ctx_run_rate_prev")
        )
    if "opp_ctx_rush_rate_l3" in cols:
        exprs.append(
            pl.col("opp_ctx_rush_rate_l3")
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("opp_run_rate_allowed_l3")
        )

    if exprs:
        df = df.with_columns(exprs)

    return df

