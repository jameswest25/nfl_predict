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


def add_context_gaps_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add small, low-collinearity context features to close conceptual gaps:
    - Practice trajectory (DNP -> LP -> FP vs regressions)
    - Depth chart churn (promotion/demotion week over week)
    - OC change flags (recent playcaller changes)
    - Special teams load (prev ST snap pct)
    - Late-week activation flag from roster status
    """
    cols = set(df.columns)
    exprs: list[pl.Expr] = []

    # Practice trajectory: map statuses to numeric (DNP=0, LP=1, FP=2)
    # Use when/then chains instead of replace for compatibility
    if {"injury_practice_status_day1", "injury_practice_status_day2", "injury_practice_status_day3"} <= cols:
        def _map_status(col_name: str) -> pl.Expr:
            return (
                pl.when(pl.col(col_name).cast(pl.Utf8).str.to_uppercase().str.contains("DNP")).then(0)
                .when(pl.col(col_name).cast(pl.Utf8).str.to_uppercase().str.contains("LP")).then(1)
                .when(pl.col(col_name).cast(pl.Utf8).str.to_uppercase().str.contains("LIMITED")).then(1)
                .when(pl.col(col_name).cast(pl.Utf8).str.to_uppercase().str.contains("FP")).then(2)
                .when(pl.col(col_name).cast(pl.Utf8).str.to_uppercase().str.contains("FULL")).then(2)
                .otherwise(None)
                .cast(pl.Int8)
            )
        
        d1 = _map_status("injury_practice_status_day1")
        d2 = _map_status("injury_practice_status_day2")
        d3 = _map_status("injury_practice_status_day3")

        exprs.append(
            (
                (d1.is_not_null()) & (d2.is_not_null()) & (d3.is_not_null()) &
                (d1 <= d2) & (d2 <= d3) & (d1 < d3)
            ).cast(pl.Int8).alias("practice_trend_up")
        )
        exprs.append(
            (
                (d1.is_not_null()) & (d2.is_not_null()) & (d3.is_not_null()) &
                (d1 >= d2) & (d2 >= d3) & (d1 > d3)
            ).cast(pl.Int8).alias("practice_trend_down")
        )

    # Depth chart churn: previous order and delta
    # PARITY: Skip if already exist with values (loaded from training output)
    if {"depth_chart_order", "player_id"} <= cols:
        if "depth_chart_order_prev" not in cols:
            prev_order = pl.col("depth_chart_order").shift(1).over("player_id")
            exprs.append(prev_order.alias("depth_chart_order_prev"))
            delta_expr = (pl.col("depth_chart_order") - prev_order)
            exprs.append(delta_expr.alias("depth_chart_order_change"))
            exprs.append(
                (delta_expr < 0)
                .cast(pl.Int8)
                .alias("depth_chart_promotion")
            )
            exprs.append(
                (delta_expr > 0)
                .cast(pl.Int8)
                .alias("depth_chart_demotion")
            )

    # OC / scheme shifts: playcaller change last game and last 2 games
    if {"offensive_coordinator", "team"} <= cols:
        oc_prev = pl.col("offensive_coordinator").shift(1).over("team")
        oc_change_flag = (
            (oc_prev.is_not_null()) &
            (pl.col("offensive_coordinator").is_not_null()) &
            (pl.col("offensive_coordinator") != oc_prev)
        ).cast(pl.Int8).alias("oc_changed_last_game")
        exprs.append(oc_change_flag)

    # Compute oc_change_last2 in a second pass to avoid nested window expressions
    oc_last2_exprs: list[pl.Expr] = []
    if "oc_changed_last_game" in df.columns:
        oc_last2_exprs.append(
            (
                (pl.col("oc_changed_last_game") == 1) |
                (pl.col("oc_changed_last_game").shift(1).over("team") == 1)
            ).cast(pl.Int8).alias("oc_change_last2")
        )

    # Special teams load: previous ST snap pct as a cap indicator for fringe players
    if "snap_st_pct_prev" in cols:
        exprs.append(
            pl.col("snap_st_pct_prev").alias("st_snap_pct_prev")
        )
        exprs.append(
            (pl.col("snap_st_pct_prev").fill_null(0.0) > 0.50)
            .cast(pl.Int8)
            .alias("st_heavy_prev")
        )

    # Late-week activation flag
    if "status" in cols:
        exprs.append(
            (pl.col("status") == "ACTIVATED").cast(pl.Int8).alias("roster_late_activation")
        )

    if exprs:
        df = df.lazy().with_columns(exprs).collect()

    # Second-pass window columns that depend on earlier columns (avoid nested windows)
    if "oc_changed_last_game" in df.columns and "team" in df.columns:
        df = df.lazy().with_columns(
            (
                (pl.col("oc_changed_last_game") == 1) |
                (pl.col("oc_changed_last_game").shift(1).over("team") == 1)
            ).cast(pl.Int8).alias("oc_change_last2")
        ).collect()

    return df


def add_moe_rb_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add RB-specific MoE features.
    
    These help the RB expert model understand player roles.
    Key signals identified through analysis:
    - Volatility: Low CV RBs = 26 snaps avg vs High CV = 10 snaps
    - Snap trend: Increasing = 25 snaps avg vs Stable = 14 snaps
    - Depth chart: RB1 = 29.5 snaps, RB2 = 18.0, RB3 = 7.1
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
        
        # rb_is_workhorse (carry share > 55%) - the bell cow RBs
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((pl.col("hist_carry_share_l3").fill_null(0.0) > 0.55).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_workhorse")
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

    # Red-zone / goal-to-go targeting for RBs (rare but important receiving role)
    if {"hist_red_zone_target_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((pl.col("hist_red_zone_target_share_l3").fill_null(0.0) > 0.08).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_high_rz_target")
        )
    if {"hist_goal_to_go_target_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((pl.col("hist_goal_to_go_target_share_l3").fill_null(0.0) > 0.05).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_high_g2g_target")
        )
    
    # rb_is_goal_line_rb (goal line carry share > 20%)
    if {"hist_goal_to_go_carry_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((pl.col("hist_goal_to_go_carry_share_l3").fill_null(0.0) > 0.20).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_goal_line_rb")
        )
        # High goal-to-go carry flag (strong signal for usage_carries)
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((pl.col("hist_goal_to_go_carry_share_l3").fill_null(0.0) > 0.30).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_high_g2g_carry")
        )

    # Red-zone carry flag
    if {"hist_red_zone_carry_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((pl.col("hist_red_zone_carry_share_l3").fill_null(0.0) > 0.25).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_high_rz_carry")
        )
    
    # rb_snap_tier (binned snap percentage for RBs)
    # Tier 0: <20%, Tier 1: 20-40%, Tier 2: 40-60%, Tier 3: >60%
    if {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then(
                pl.when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.60)
                .then(pl.lit(3))
                .when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.40)
                .then(pl.lit(2))
                .when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.20)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_snap_tier")
        )
    
    # rb_volatility_tier - CRITICAL SIGNAL: Low CV RBs = 26 snaps, High CV = 10 snaps
    # Tier 0: CV >= 0.7 (volatile), Tier 1: 0.3-0.7 (moderate), Tier 2: <0.3 (stable)
    if {"snap_volatility_cv_l5", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then(
                pl.when(pl.col("snap_volatility_cv_l5").fill_null(1.0) < 0.30)
                .then(pl.lit(2))  # Stable role - highest snaps
                .when(pl.col("snap_volatility_cv_l5").fill_null(1.0) < 0.70)
                .then(pl.lit(1))  # Moderate volatility
                .otherwise(pl.lit(0))  # High volatility - lowest snaps
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_volatility_tier")
        )
        
        # rb_is_stable_role (CV < 0.3) - binary flag for stable workload
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((pl.col("snap_volatility_cv_l5").fill_null(1.0) < 0.30).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_stable_role")
        )
    
    # rb_snap_trend - Increasing workload is a strong signal (25 snaps vs 14)
    # Compares prev game to L3 avg
    if {"snap_offense_pct_prev", "snap_offense_pct_l3", "position"} <= cols:
        snap_trend = (
            pl.col("snap_offense_pct_prev").fill_null(0.0) - 
            pl.col("snap_offense_pct_l3").fill_null(0.0)
        )
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then(snap_trend.cast(pl.Float32))
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("rb_snap_trend")
        )
        
        # rb_is_trending_up (increasing by 5%+)
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((snap_trend > 0.05).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_trending_up")
        )
    
    # rb_depth_chart_tier - Depth chart order is highly predictive for RBs
    # RB1 = 29.5 snaps avg, RB2 = 18.0, RB3 = 7.1
    if {"depth_chart_order", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then(
                pl.when(pl.col("depth_chart_order").fill_null(3) == 1)
                .then(pl.lit(2))  # RB1
                .when(pl.col("depth_chart_order").fill_null(3) == 2)
                .then(pl.lit(1))  # RB2
                .otherwise(pl.lit(0))  # RB3+
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_depth_chart_tier")
        )
        
        # rb_is_rb1 - binary flag for primary back
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((pl.col("depth_chart_order").fill_null(3) == 1).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_rb1")
        )
    
    # rb_ceiling_ratio - Max observed snap % as a ratio to avg (indicates upside)
    if {"max_snap_pct_l5", "snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then(
                (pl.col("max_snap_pct_l5").fill_null(0.0) /
                 (pl.col("snap_offense_pct_l3").fill_null(0.01) + 0.01))
                .clip(0.0, 3.0)
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("rb_ceiling_ratio")
        )
    
    # =================== NEW TEAM-LEVEL RB FEATURES ===================
    # rb_is_bellcow - Dominant back with >50% carry share (13.2% of RBs)
    if {"hist_carry_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then((pl.col("hist_carry_share_l3").fill_null(0.0) > 0.50).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_bellcow")
        )
    
    # rb_carry_dominance - Continuous carry share score (0.621 correlation with snaps!)
    # Scaled to emphasize high-share backs
    if {"hist_carry_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then(
                # Square to emphasize high-share backs
                (pl.col("hist_carry_share_l3").fill_null(0.0) ** 2 * 4.0)
                .clip(0.0, 2.0)
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("rb_carry_dominance")
        )
    
    # rb_pass_game_lift - Receiving back in pass-heavy offense
    # High target share × high team pass rate = more snaps in passing situations
    if {"hist_target_share_l3", "team_ctx_pass_rate_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then(
                (pl.col("hist_target_share_l3").fill_null(0.0) * 
                 pl.col("team_ctx_pass_rate_l3").fill_null(0.5) * 5.0)
                .clip(0.0, 2.0)
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("rb_pass_game_lift")
        )
    
    # rb_total_usage_score - Combined rushing + receiving usage
    # RBs who contribute in both phases get more snaps
    if {"hist_carry_share_l3", "hist_target_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then(
                (pl.col("hist_carry_share_l3").fill_null(0.0) + 
                 pl.col("hist_target_share_l3").fill_null(0.0) * 2.0)
                .clip(0.0, 1.5)
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("rb_total_usage_score")
        )
    
    # rb_team_rushing_volume - High volume rushing teams = more RB snaps
    if {"team_ctx_carries_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position").is_in(["RB", "HB", "FB"]))
            .then(
                (pl.col("team_ctx_carries_l3").fill_null(25.0) / 35.0)  # Normalize to ~1
                .clip(0.0, 2.0)
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("rb_team_rushing_volume")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_moe_te_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add TE-specific MoE features.
    
    These help the TE expert model distinguish receiving TEs from blocking TEs.
    Key signals identified through analysis:
    - Volatility: Low CV TEs = 34 snaps avg vs High CV = 13 snaps
    - Snap trend: Increasing = 30 snaps avg vs Decreasing = 23 snaps
    - Ceiling: High (>80%) = 44 snaps avg vs Low (<40%) = 10 snaps
    - Depth chart: TE1 = 39.7 snaps, TE2 = 22.4, TE3 = 12.6
    
    NOTE: We prioritize continuous features over sparse binary flags to avoid
    overfitting. The TE model was losing to Global because binary features like
    te_is_te1 (82% zeros) caused excessive partitioning. Continuous alternatives
    (te_snap_pct_scaled, te_target_share_scaled) provide smoother gradients.
    """
    cols = set(df.columns)
    exprs = []
    
    # --- CONTINUOUS TE FEATURES (preferred) ---
    # te_snap_pct_scaled - Normalized snap % for TEs (continuous, not binary)
    # This captures the same info as te_is_te1/te_snap_tier but smoothly
    if {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                pl.col("snap_offense_pct_l3").fill_null(0.0).cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_snap_pct_scaled")
        )
    
    # te_target_share_scaled - Normalized target share for TEs (continuous)
    if {"hist_target_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                (pl.col("hist_target_share_l3").fill_null(0.0) * 10.0)  # Scale to ~0-2 range
                .clip(0.0, 2.0)
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_target_share_scaled")
        )
    
    # te_stability_score - Inverse of volatility CV (higher = more stable)
    if {"snap_volatility_cv_l5", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                (1.0 - pl.col("snap_volatility_cv_l5").fill_null(1.0).clip(0.0, 1.0))
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_stability_score")
        )
    
    # te_ceiling_pct - Max snap pct as continuous measure
    if {"max_snap_pct_l5", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                pl.col("max_snap_pct_l5").fill_null(0.0).cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_ceiling_pct")
        )
    
    # te_depth_score - Continuous depth chart score (2=TE1, 1=TE2, 0=TE3+)
    if {"depth_chart_order", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                (3.0 - pl.col("depth_chart_order").fill_null(3).clip(1, 3).cast(pl.Float32))
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_depth_score")
        )
    
    # te_combined_role_score - CRITICAL: Combined continuous score
    # This is the most predictive feature - combines depth + stability + ceiling
    if {"depth_chart_order", "snap_volatility_cv_l5", "max_snap_pct_l5", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                # Depth chart contribution (0-2)
                (3.0 - pl.col("depth_chart_order").fill_null(3).clip(1, 3).cast(pl.Float32))
                +
                # Stability contribution (0-1)
                (1.0 - pl.col("snap_volatility_cv_l5").fill_null(1.0).clip(0.0, 1.0))
                +
                # Ceiling contribution (0-1)
                pl.col("max_snap_pct_l5").fill_null(0.0).clip(0.0, 1.0)
            )
            .otherwise(pl.lit(None))
            .cast(pl.Float32)
            .alias("te_combined_role_score")
        )
    
    # --- BINARY TE FEATURES (kept for backward compat, but less weight) ---
    # te_is_receiving_te (target share > 10%) - 0.524 correlation!
    if {"hist_target_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then((pl.col("hist_target_share_l3").fill_null(0.0) > 0.10).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_receiving_te")
        )
        
        # te_is_primary_target - elite receiving TEs (>15% target share)
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then((pl.col("hist_target_share_l3").fill_null(0.0) > 0.15).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_primary_target")
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

    # Red-zone / goal-to-go targeting flags
    if {"hist_red_zone_target_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then((pl.col("hist_red_zone_target_share_l3").fill_null(0.0) > 0.12).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_high_rz_target")
        )
    if {"hist_goal_to_go_target_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then((pl.col("hist_goal_to_go_target_share_l3").fill_null(0.0) > 0.08).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_high_g2g_target")
        )
    
    # te_is_te1 (snap_pct > 65%) - 0.554 correlation with snaps!
    # This identifies the primary TE on the team
    if {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then((pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.65).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_te1")
        )
    
    # te_snap_tier (binned snap percentage for TEs)
    # Tier 0: <30%, Tier 1: 30-50%, Tier 2: 50-70%, Tier 3: >70%
    if {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                pl.when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.70)
                .then(pl.lit(3))
                .when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.50)
                .then(pl.lit(2))
                .when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.30)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_snap_tier")
        )
    
    # te_expected_snaps_ratio - Expected snaps normalized by team context
    # This helps when status is missing by providing a continuous indicator
    if {"expected_snaps_l3", "snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                pl.col("expected_snaps_l3").fill_null(0.0) / 60.0  # Normalize to ~0-1 range
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_expected_snaps_ratio")
        )
    
    # te_is_high_snap_te - TEs with >40 expected snaps (strong starters)
    if {"expected_snaps_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then((pl.col("expected_snaps_l3").fill_null(0.0) > 40).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_high_snap_te")
        )
    
    # te_volatility_tier - CRITICAL SIGNAL: Low CV TEs = 34 snaps, High CV = 13 snaps
    # Tier 0: CV >= 0.7 (volatile), Tier 1: 0.3-0.7 (moderate), Tier 2: <0.3 (stable)
    if {"snap_volatility_cv_l5", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                pl.when(pl.col("snap_volatility_cv_l5").fill_null(1.0) < 0.30)
                .then(pl.lit(2))  # Stable role - highest snaps
                .when(pl.col("snap_volatility_cv_l5").fill_null(1.0) < 0.70)
                .then(pl.lit(1))  # Moderate volatility
                .otherwise(pl.lit(0))  # High volatility - lowest snaps
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_volatility_tier")
        )
        
        # te_is_stable_role (CV < 0.3) - binary flag for stable workload
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then((pl.col("snap_volatility_cv_l5").fill_null(1.0) < 0.30).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_stable_role")
        )
    
    # te_snap_trend - Increasing workload is a strong signal (30 snaps vs 23)
    # Compares prev game to L3 avg
    if {"snap_offense_pct_prev", "snap_offense_pct_l3", "position"} <= cols:
        snap_trend = (
            pl.col("snap_offense_pct_prev").fill_null(0.0) - 
            pl.col("snap_offense_pct_l3").fill_null(0.0)
        )
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(snap_trend.cast(pl.Float32))
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_snap_trend")
        )
        
        # te_is_trending_up (increasing by 5%+)
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then((snap_trend > 0.05).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_trending_up")
        )
    
    # te_depth_chart_tier - Depth chart order is highly predictive for TEs
    # TE1 = 39.7 snaps avg, TE2 = 22.4, TE3 = 12.6
    if {"depth_chart_order", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                pl.when(pl.col("depth_chart_order").fill_null(3) == 1)
                .then(pl.lit(2))  # TE1
                .when(pl.col("depth_chart_order").fill_null(3) == 2)
                .then(pl.lit(1))  # TE2
                .otherwise(pl.lit(0))  # TE3+
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_depth_chart_tier")
        )
    
    # te_ceiling_tier - Max snap % is highly predictive: High (>80%) = 44 snaps, Low (<40%) = 10 snaps
    if {"max_snap_pct_l5", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                pl.when(pl.col("max_snap_pct_l5").fill_null(0.0) > 0.80)
                .then(pl.lit(2))  # High ceiling
                .when(pl.col("max_snap_pct_l5").fill_null(0.0) > 0.40)
                .then(pl.lit(1))  # Medium ceiling
                .otherwise(pl.lit(0))  # Low ceiling
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_ceiling_tier")
        )
        
        # te_has_high_ceiling (>80% max snap pct)
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then((pl.col("max_snap_pct_l5").fill_null(0.0) > 0.80).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_has_high_ceiling")
        )
    
    # te_role_score - Combined score from multiple signals (higher = more snaps expected)
    # Combines depth chart, volatility, and ceiling signals
    if {"depth_chart_order", "snap_volatility_cv_l5", "max_snap_pct_l5", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                # Depth chart contribution (0-2)
                pl.when(pl.col("depth_chart_order").fill_null(3) == 1)
                .then(pl.lit(2.0))
                .when(pl.col("depth_chart_order").fill_null(3) == 2)
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
                +
                # Low volatility contribution (0-1)
                pl.when(pl.col("snap_volatility_cv_l5").fill_null(1.0) < 0.30)
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
                +
                # High ceiling contribution (0-1)
                pl.when(pl.col("max_snap_pct_l5").fill_null(0.0) > 0.80)
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
            )
            .otherwise(pl.lit(None))
            .cast(pl.Float32)
            .alias("te_role_score")
        )
    
    # te_route_participation - Route participation rate (0-1 scale)
    # Compute as route plays / team dropbacks to get actual participation rate
    # Note: ps_hist_route_participation_plays_l3 contains raw counts, not %
    if {"ps_hist_route_participation_plays_l3", "qb_profile_team_dropbacks_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                (pl.col("ps_hist_route_participation_plays_l3").fill_null(0.0) /
                 pl.col("qb_profile_team_dropbacks_l3").fill_null(1.0).clip(1.0, None))
                .clip(0.0, 1.0)
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_route_participation")
        )
    # Fallback: use snap share as proxy if dropbacks unavailable
    elif {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                pl.col("snap_offense_pct_l3").fill_null(0.0).cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_route_participation")
        )
    
    # =================== NEW TEAM-LEVEL TE FEATURES ===================
    # te_target_efficiency - Target share relative to snap share
    # High = receiving TE, Low = blocking TE
    if {"hist_target_share_l3", "snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                (pl.col("hist_target_share_l3").fill_null(0.0) / 
                 (pl.col("snap_offense_pct_l3").fill_null(0.01) + 0.01))
                .clip(0.0, 2.0)
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_target_efficiency")
        )
    
    # te_receiving_role_score - Combines target share and route participation
    # TEs with high receiving involvement get more predictable snap counts
    if {"hist_target_share_l3", "snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                # Target share contribution (scaled to 0-1)
                (pl.col("hist_target_share_l3").fill_null(0.0) * 5.0 +
                 # Snap share bonus (receiving TEs usually have high snap share too)
                 pl.col("snap_offense_pct_l3").fill_null(0.0) * 0.5)
                .clip(0.0, 2.0)
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_receiving_role_score")
        )
    
    # te_is_pure_blocker - High snaps but very low targets (blocking specialist)
    # These TEs have more predictable snap counts
    if {"hist_target_share_l3", "snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                ((pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.50) &
                 (pl.col("hist_target_share_l3").fill_null(0.0) < 0.05)).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_pure_blocker")
        )
    
    # te_role_clarity - How clearly defined is this TE's role?
    # Either pure blocker (high snaps, low targets) or receiving TE (high targets)
    # TEs in the middle have less predictable snaps
    if {"hist_target_share_l3", "snap_offense_pct_l3", "position"} <= cols:
        target_efficiency = (
            pl.col("hist_target_share_l3").fill_null(0.0) / 
            (pl.col("snap_offense_pct_l3").fill_null(0.01) + 0.01)
        )
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                # Clear blocker: high snaps, low efficiency
                pl.when(
                    (pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.50) &
                    (target_efficiency < 0.15)
                )
                .then(pl.lit(1.0))  # Clear blocking role
                # Clear receiver: high efficiency
                .when(target_efficiency > 0.25)
                .then(pl.lit(1.0))  # Clear receiving role
                # Unclear role (in the middle)
                .otherwise(pl.lit(0.5))
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_role_clarity")
        )
    
    # te_team_pass_volume - TEs on pass-heavy teams get more routes/snaps
    if {"team_ctx_pass_rate_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                (pl.col("team_ctx_pass_rate_l3").fill_null(0.5) * 2.0)
                .clip(0.0, 2.0)
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_team_pass_volume")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_moe_wr_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add WR-specific MoE features.
    
    These help the WR expert model distinguish between WR1/WR2/WR3 roles.
    Key signals identified:
    - Depth chart: WR1=45 snaps, WR2=19, WR3=11
    - Volatility: Low CV WRs get more consistent snaps
    - Snap trend: Increasing workload is predictive
    """
    cols = set(df.columns)
    exprs = []
    
    # --- CONTINUOUS WR FEATURES (preferred) ---
    # wr_snap_pct_scaled - Continuous snap % for WRs
    if {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(
                pl.col("snap_offense_pct_l3").fill_null(0.0).cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("wr_snap_pct_scaled")
        )
    
    # wr_stability_score - Inverse of volatility CV (higher = more stable)
    if {"snap_volatility_cv_l5", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(
                (1.0 - pl.col("snap_volatility_cv_l5").fill_null(1.0).clip(0.0, 1.0))
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("wr_stability_score")
        )
    
    # wr_ceiling_pct - Max snap pct as continuous measure
    if {"max_snap_pct_l5", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(
                pl.col("max_snap_pct_l5").fill_null(0.0).cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("wr_ceiling_pct")
        )
    
    # wr_depth_score - Continuous depth chart score (3=WR1, 2=WR2, 1=WR3, 0=WR4+)
    if {"depth_chart_order", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(
                (4.0 - pl.col("depth_chart_order").fill_null(4).clip(1, 4).cast(pl.Float32))
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("wr_depth_score")
        )
    
    # wr_combined_role_score - Combined continuous score
    if {"depth_chart_order", "snap_volatility_cv_l5", "max_snap_pct_l5", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(
                # Depth chart contribution (0-3)
                (4.0 - pl.col("depth_chart_order").fill_null(4).clip(1, 4).cast(pl.Float32))
                +
                # Stability contribution (0-1)
                (1.0 - pl.col("snap_volatility_cv_l5").fill_null(1.0).clip(0.0, 1.0))
                +
                # Ceiling contribution (0-1)
                pl.col("max_snap_pct_l5").fill_null(0.0).clip(0.0, 1.0)
            )
            .otherwise(pl.lit(None))
            .cast(pl.Float32)
            .alias("wr_combined_role_score")
        )
    
    # wr_snap_trend - Increasing workload is a signal
    if {"snap_offense_pct_prev", "snap_offense_pct_l3", "position"} <= cols:
        snap_trend = (
            pl.col("snap_offense_pct_prev").fill_null(0.0) - 
            pl.col("snap_offense_pct_l3").fill_null(0.0)
        )
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(snap_trend.cast(pl.Float32))
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("wr_snap_trend")
        )
        
        # wr_is_trending_up (increasing by 5%+)
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then((snap_trend > 0.05).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("wr_is_trending_up")
        )
    
    # wr_volatility_tier - Low CV WRs get more consistent snaps
    if {"snap_volatility_cv_l5", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(
                pl.when(pl.col("snap_volatility_cv_l5").fill_null(1.0) < 0.20)
                .then(pl.lit(2))  # Very stable
                .when(pl.col("snap_volatility_cv_l5").fill_null(1.0) < 0.50)
                .then(pl.lit(1))  # Moderate volatility
                .otherwise(pl.lit(0))  # High volatility
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("wr_volatility_tier")
        )
    
    # --- TIER/BINARY WR FEATURES ---
    # wr_depth_tier (binned snap percentage for WRs) - KEY MoE feature!
    # Tier 0: <40% (WR4+), Tier 1: 40-70% (WR3), Tier 2: 70-85% (WR2), Tier 3: >85% (WR1)
    if {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(
                pl.when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.85)
                .then(pl.lit(3))
                .when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.70)
                .then(pl.lit(2))
                .when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.40)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("wr_depth_tier")
        )
    
    # wr_snap_tier (binned snap percentage for WRs - finer granularity)
    if {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(
                pl.when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.70)
                .then(pl.lit(3))
                .when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.50)
                .then(pl.lit(2))
                .when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.30)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("wr_snap_tier")
        )
    
    # wr_is_wr1 (snap_pct > 75%) - high correlation with snaps
    if {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then((pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.75).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("wr_is_wr1")
        )
    
    # wr_is_low_usage (snap_pct < 20%) - identifies WR4/WR5
    if {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then((pl.col("snap_offense_pct_l3").fill_null(0.0) < 0.20).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("wr_is_low_usage")
        )
    
    # wr_target_concentration (target share / snap share)
    if {"hist_target_share_l3", "snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(
                pl.col("hist_target_share_l3").fill_null(0.0) /
                (pl.col("snap_offense_pct_l3").fill_null(0.01) + 0.01)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("wr_target_concentration")
        )

    # Red-zone / goal-to-go targeting flags (helps MoE split high-leverage roles)
    if {"hist_red_zone_target_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then((pl.col("hist_red_zone_target_share_l3").fill_null(0.0) > 0.20).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("wr_high_rz_target")
        )
    if {"hist_goal_to_go_target_share_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then((pl.col("hist_goal_to_go_target_share_l3").fill_null(0.0) > 0.10).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("wr_high_g2g_target")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_moe_qb_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add QB-specific MoE features.
    
    These help the QB expert model distinguish starters from backups.
    """
    cols = set(df.columns)
    exprs = []
    
    # qb_is_likely_starter (snap_pct > 80%) - strong predictor!
    # Uses snap_offense_pct_l3 which is computed across all weeks in pipeline/feature.py
    if {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "QB")
            .then((pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.80).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("qb_is_likely_starter")
        )
    
    # qb_snap_tier (binned snap percentage for QBs)
    # Tier 0: <20% (3rd string), Tier 1: 20-50% (backup), Tier 2: 50-80% (shared), Tier 3: >80% (starter)
    if {"snap_offense_pct_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "QB")
            .then(
                pl.when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.80)
                .then(pl.lit(3))
                .when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.50)
                .then(pl.lit(2))
                .when(pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.20)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("qb_snap_tier")
        )
    
    # qb_is_mobile (scramble rate > 5%) - affects rush TD potential
    if {"qb_profile_scramble_rate_l3", "position"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "QB")
            .then((pl.col("qb_profile_scramble_rate_l3").fill_null(0.0) > 0.05).cast(pl.Int8))
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("qb_is_mobile")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_moe_position_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all MoE position-specific features.
    
    This is the main entry point that applies all position feature functions.
    Returns the dataframe with all MoE features added.
    
    IMPORTANT: This function must be called AFTER snap rolling features are 
    computed (add_rolling_snap_features) so that snap_offense_pct_l3 exists.
    """
    logger.info("Adding MoE position-specific features...")
    
    before_cols = len(df.columns)
    
    df = add_context_gaps_features(df)
    df = add_moe_qb_features(df)
    df = add_moe_rb_features(df)
    df = add_moe_wr_features(df)
    df = add_moe_te_features(df)
    
    added = len(df.columns) - before_cols
    logger.info("    Added %d MoE position-specific features", added)
    
    return df

