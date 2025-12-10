"""Target depth features for NFL player predictions.

These features are for predicting average air yards per target.
Understanding target depth requires:
1. Position-based priors (RBs get ~0 yards, WRs get ~10 yards)
2. Route alignment (slot vs outside WRs)
3. QB tendencies (some QBs throw deeper than others)
4. Historical player target depth patterns
"""

from __future__ import annotations

import logging

import polars as pl

__all__ = [
    "add_target_depth_features",
    "add_position_depth_priors",
    "add_wr_alignment_features",
    "add_qb_depth_tendencies",
    "compute_historical_air_yards",
    "add_depth_archetypes",
]

logger = logging.getLogger(__name__)


def add_position_depth_priors(df: pl.DataFrame) -> pl.DataFrame:
    """Add position-based target depth priors.
    
    Based on data: WR=10.7, TE=6.2, RB=0.3, QB=0.97
    These are the median values by position.
    """
    if "position" not in df.columns:
        return df
    
    exprs = []
    
    # Position-based target depth prior
    exprs.append(
        pl.when(pl.col("position") == "WR").then(10.7)
        .when(pl.col("position") == "TE").then(6.2)
        .when(pl.col("position") == "RB").then(0.3)
        .when(pl.col("position") == "FB").then(0.0)
        .when(pl.col("position") == "QB").then(0.0)
        .otherwise(5.0)
        .cast(pl.Float32)
        .alias("position_target_depth_prior")
    )
    
    # Position-based variance prior (how much depth varies by position)
    # WRs have high variance (std=7.4), RBs have low variance (std=4.1)
    exprs.append(
        pl.when(pl.col("position") == "WR").then(7.4)
        .when(pl.col("position") == "TE").then(5.5)
        .when(pl.col("position") == "RB").then(4.1)
        .otherwise(5.0)
        .cast(pl.Float32)
        .alias("position_depth_variance")
    )
    
    # Position numeric encoding for stronger signal
    exprs.append(
        pl.when(pl.col("position") == "WR").then(3)  # Deepest
        .when(pl.col("position") == "TE").then(2)   # Medium
        .when(pl.col("position") == "RB").then(1)   # Short
        .when(pl.col("position") == "FB").then(0)   # Very short
        .when(pl.col("position") == "QB").then(0)   # Very short
        .otherwise(1)
        .cast(pl.Int8)
        .alias("position_depth_rank")
    )
    
    # Binary flag for short-target positions (RBs, FBs)
    exprs.append(
        (pl.col("position").is_in(["RB", "FB", "QB"]))
        .cast(pl.Int8)
        .alias("is_short_target_position")
    )
    
    # Binary flag for deep-target positions (WRs)
    exprs.append(
        (pl.col("position") == "WR")
        .cast(pl.Int8)
        .alias("is_deep_target_position")
    )
    
    # RB-specific: Expected near-zero depth (strong signal)
    exprs.append(
        pl.when(pl.col("position") == "RB").then(0.0)
        .otherwise(pl.lit(None))
        .cast(pl.Float32)
        .alias("rb_expected_target_depth")
    )
    
    # Position depth ceiling
    exprs.append(
        pl.when(pl.col("position") == "RB").then(3.0)
        .when(pl.col("position") == "FB").then(0.0)
        .when(pl.col("position") == "TE").then(12.0)
        .when(pl.col("position") == "WR").then(30.0)
        .otherwise(5.0)
        .cast(pl.Float32)
        .alias("position_depth_ceiling")
    )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_wr_alignment_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add WR alignment features (slot vs outside).
    
    Slot receivers get shorter targets (~10.3 yds), outside WRs get deeper (~13 yds).
    """
    cols = set(df.columns)
    exprs = []
    
    if "ps_hist_targets_slot_share_l3" in cols:
        # Slot share directly (high = shorter routes)
        exprs.append(
            pl.col("ps_hist_targets_slot_share_l3")
            .fill_null(0.33)  # Default to mixed if unknown
            .clip(0, 1)
            .cast(pl.Float32)
            .alias("wr_slot_tendency")
        )
        
        # Outside WR share (high = deeper routes)
        if "ps_hist_targets_wide_share_l3" in cols:
            exprs.append(
                pl.col("ps_hist_targets_wide_share_l3")
                .fill_null(0.33)
                .clip(0, 1)
                .cast(pl.Float32)
                .alias("wr_outside_tendency")
            )
            
            # Slot vs outside indicator (-1 = outside, 0 = mixed, +1 = slot)
            exprs.append(
                (pl.col("ps_hist_targets_slot_share_l3").fill_null(0.33) -
                 pl.col("ps_hist_targets_wide_share_l3").fill_null(0.33))
                .cast(pl.Float32)
                .alias("wr_slot_vs_outside")
            )
    
    # Position-adjusted target depth expectation
    if "position" in cols and "ps_hist_targets_slot_share_l3" in cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(
                # WR base (10.7) adjusted by slot vs outside tendency
                10.7 - 3.0 * (pl.col("ps_hist_targets_slot_share_l3").fill_null(0.33) - 0.33)
            )
            .when(pl.col("position") == "TE").then(6.2)
            .when(pl.col("position") == "RB").then(0.3)
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("adjusted_target_depth_prior")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_qb_depth_tendencies(df: pl.DataFrame) -> pl.DataFrame:
    """Add QB air yards tendencies.
    
    Some QBs throw deeper than others - use this as a feature.
    """
    cols = set(df.columns)
    exprs = []
    
    if "qb_profile_avg_air_yards_l3" in cols:
        exprs.append(
            pl.col("qb_profile_avg_air_yards_l3")
            .fill_null(8.0)  # League average
            .clip(3, 15)
            .cast(pl.Float32)
            .alias("qb_air_yards_tendency")
        )
        
        # Interaction: position depth prior * QB tendency
        if "position" in cols:
            exprs.append(
                (pl.when(pl.col("position") == "WR")
                .then(pl.col("qb_profile_avg_air_yards_l3").fill_null(8.0) / 8.0 * 10.7)
                .when(pl.col("position") == "TE")
                .then(pl.col("qb_profile_avg_air_yards_l3").fill_null(8.0) / 8.0 * 6.2)
                .when(pl.col("position") == "RB")
                .then(0.3)  # RBs don't scale with QB depth
                .otherwise(0.0))
                .cast(pl.Float32)
                .alias("qb_scaled_target_depth")
            )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_role_based_depth_tendencies(df: pl.DataFrame) -> pl.DataFrame:
    """Add role-based depth tendencies for RBs, WRs, and TEs."""
    cols = set(df.columns)
    exprs = []
    
    # RB target type features
    if {"position", "hist_target_share_l3"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "RB")
            .then(
                0.3 + pl.col("hist_target_share_l3").fill_null(0) * 5.0
            )
            .otherwise(pl.lit(None))
            .cast(pl.Float32)
            .alias("rb_receiving_depth_tendency")
        )
    
    # WR depth type based on route patterns
    if {"position", "ps_hist_targets_wide_share_l3"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "WR")
            .then(
                pl.col("ps_hist_targets_wide_share_l3").fill_null(0.33) * 3.0 + 9.0
            )
            .otherwise(pl.lit(None))
            .cast(pl.Float32)
            .alias("wr_route_depth_tendency")
        )
    
    # TE depth based on blocking vs receiving role
    if {"position", "hist_target_share_l3"} <= cols:
        exprs.append(
            pl.when(pl.col("position") == "TE")
            .then(
                4.0 + pl.col("hist_target_share_l3").fill_null(0) * 20.0
            )
            .otherwise(pl.lit(None))
            .cast(pl.Float32)
            .alias("te_receiving_depth_tendency")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def compute_historical_air_yards(df: pl.DataFrame) -> pl.DataFrame:
    """Compute historical player target depth (rolling average from previous games).
    
    This is the best feature - player's own historical air yards per target.
    """
    if "air_yards_all_targets" not in df.columns or "target" not in df.columns:
        return df
    
    # Sort by player and game_date
    df = df.sort(["player_id", "game_date"])
    
    exprs = []
    
    # Rolling sum of air yards (shifted to avoid leakage)
    exprs.append(
        pl.col("air_yards_all_targets")
        .rolling_sum(window_size=3, min_samples=1)
        .shift(1)
        .over(["player_id"])
        .fill_null(0)
        .cast(pl.Float32)
        .alias("hist_air_yards_sum_l3")
    )
    
    # Rolling sum of targets (shifted to avoid leakage)
    exprs.append(
        pl.col("target")
        .rolling_sum(window_size=3, min_samples=1)
        .shift(1)
        .over(["player_id"])
        .fill_null(0)
        .cast(pl.Float32)
        .alias("hist_targets_l3")
    )
    
    # PREVIOUS GAME air yards (strongest signal)
    exprs.append(
        pl.col("air_yards_all_targets")
        .shift(1)
        .over(["player_id"])
        .fill_null(0)
        .cast(pl.Float32)
        .alias("air_yards_prev_game")
    )
    
    # Previous game targets
    exprs.append(
        pl.col("target")
        .shift(1)
        .over(["player_id"])
        .fill_null(0)
        .cast(pl.Float32)
        .alias("targets_prev_game")
    )
    
    # 5-game rolling for more stable estimate
    exprs.append(
        pl.col("air_yards_all_targets")
        .rolling_sum(window_size=5, min_samples=1)
        .shift(1)
        .over(["player_id"])
        .fill_null(0)
        .cast(pl.Float32)
        .alias("hist_air_yards_sum_l5")
    )
    
    exprs.append(
        pl.col("target")
        .rolling_sum(window_size=5, min_samples=1)
        .shift(1)
        .over(["player_id"])
        .fill_null(0)
        .cast(pl.Float32)
        .alias("hist_targets_l5")
    )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_derived_depth_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add derived features from historical air yards."""
    cols = set(df.columns)
    
    if {"hist_air_yards_sum_l3", "hist_targets_l3"} <= cols:
        # Position priors
        def _pos_prior() -> pl.Expr:
            return (
                pl.when(pl.col("position") == "WR").then(10.7)
                .when(pl.col("position") == "TE").then(6.2)
                .when(pl.col("position") == "RB").then(0.3)
                .otherwise(5.0)
            )
        
        exprs = [
            # Historical air yards per target (player's own pattern)
            (pl.col("hist_air_yards_sum_l3") / pl.col("hist_targets_l3").clip(1, None))
            .fill_null(0)
            .cast(pl.Float32)
            .alias("hist_air_yards_per_target_l3"),
            
            # Previous game air yards per target (strongest signal)
            (pl.col("air_yards_prev_game") / pl.col("targets_prev_game").clip(1, None))
            .fill_null(0)
            .cast(pl.Float32)
            .alias("air_yards_per_target_prev"),
            
            # 5-game rolling for stability
            (pl.col("hist_air_yards_sum_l5") / pl.col("hist_targets_l5").clip(1, None))
            .fill_null(0)
            .cast(pl.Float32)
            .alias("hist_air_yards_per_target_l5"),
            
            # Combined: position prior weighted by player history
            pl.when(pl.col("hist_targets_l3") >= 3)
            .then(pl.col("hist_air_yards_sum_l3") / pl.col("hist_targets_l3").clip(1, None))
            .otherwise(_pos_prior())
            .cast(pl.Float32)
            .alias("expected_target_depth"),
        ]
        
        df = df.with_columns(exprs)
    
    return df


def add_depth_archetypes(df: pl.DataFrame) -> pl.DataFrame:
    """Add player archetypes based on historical depth."""
    cols = set(df.columns)
    
    if "hist_air_yards_per_target_l5" not in cols or "position" not in cols:
        return df
    
    # Position priors
    def _pos_prior() -> pl.Expr:
        return (
            pl.when(pl.col("position") == "WR").then(10.7)
            .when(pl.col("position") == "TE").then(6.2)
            .when(pl.col("position") == "RB").then(0.3)
            .otherwise(5.0)
        )
    
    exprs = [
        # WR archetype based on historical depth
        pl.when(pl.col("position") == "WR")
        .then(
            pl.when(pl.col("hist_air_yards_per_target_l5") > 15).then(pl.lit(4))  # Deep threat
            .when(pl.col("hist_air_yards_per_target_l5") > 10).then(pl.lit(3))   # Intermediate
            .when(pl.col("hist_air_yards_per_target_l5") > 5).then(pl.lit(2))    # Possession
            .otherwise(pl.lit(1))  # Short/Unknown
        )
        .otherwise(pl.lit(0))  # Non-WR
        .cast(pl.Int8)
        .alias("wr_depth_archetype"),
        
        # TE archetype (receiving TE vs blocking TE)
        pl.when(pl.col("position") == "TE")
        .then(
            pl.when(pl.col("hist_air_yards_per_target_l5") > 8).then(pl.lit(3))   # Receiving TE
            .when(pl.col("hist_air_yards_per_target_l5") > 4).then(pl.lit(2))    # Hybrid
            .otherwise(pl.lit(1))  # Blocking TE
        )
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("te_depth_archetype"),
        
        # Has enough history to make a good prediction
        (pl.col("hist_targets_l5") >= 5).cast(pl.Int8).alias("has_depth_history") if "hist_targets_l5" in cols else pl.lit(0).cast(pl.Int8).alias("has_depth_history"),
    ]
    
    if "hist_targets_l5" in cols:
        # Player's deviation from position mean
        exprs.append(
            pl.when(pl.col("hist_targets_l5") >= 3)
            .then(
                (pl.col("hist_air_yards_sum_l5") / pl.col("hist_targets_l5").clip(1, None)) - _pos_prior()
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("player_depth_vs_position")
        )
        
        # Target volume from history
        exprs.append(pl.col("hist_targets_l3").fill_null(0).cast(pl.Float32).alias("hist_target_volume_l3"))
        exprs.append(pl.col("hist_targets_l5").fill_null(0).cast(pl.Float32).alias("hist_target_volume_l5"))
        
        # Expected variance based on target volume
        exprs.append(
            pl.when(pl.col("hist_targets_l5") >= 15).then(0.2)
            .when(pl.col("hist_targets_l5") >= 10).then(0.4)
            .when(pl.col("hist_targets_l5") >= 5).then(0.6)
            .otherwise(0.8)
            .cast(pl.Float32)
            .alias("depth_uncertainty")
        )
        
        # Shrunk estimate: blend player history with position mean based on volume
        exprs.append(
            pl.when(pl.col("hist_targets_l5") >= 10)
            .then(
                0.9 * (pl.col("hist_air_yards_sum_l5") / pl.col("hist_targets_l5").clip(1, None))
                + 0.1 * _pos_prior()
            )
            .when(pl.col("hist_targets_l5") >= 6)
            .then(
                0.8 * (pl.col("hist_air_yards_sum_l5") / pl.col("hist_targets_l5").clip(1, None))
                + 0.2 * _pos_prior()
            )
            .when(pl.col("hist_targets_l5") >= 3)
            .then(
                0.7 * (pl.col("hist_air_yards_sum_l5") / pl.col("hist_targets_l5").clip(1, None))
                + 0.3 * _pos_prior()
            )
            .when(pl.col("hist_targets_l5") >= 1)
            .then(
                0.5 * (pl.col("hist_air_yards_sum_l5") / pl.col("hist_targets_l5").clip(1, None))
                + 0.5 * _pos_prior()
            )
            .otherwise(_pos_prior())
            .cast(pl.Float32)
            .alias("shrunk_depth_estimate")
        )
        
        # Best estimate
        exprs.append(
            pl.when(pl.col("hist_targets_l5") >= 5)
            .then(pl.col("hist_air_yards_sum_l5") / pl.col("hist_targets_l5").clip(1, None))
            .when(pl.col("hist_targets_l3") >= 2)
            .then(
                0.7 * (pl.col("hist_air_yards_sum_l3") / pl.col("hist_targets_l3").clip(1, None))
                + 0.3 * _pos_prior()
            )
            .otherwise(_pos_prior())
            .cast(pl.Float32)
            .alias("best_depth_estimate")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def add_weighted_depth_estimate(df: pl.DataFrame) -> pl.DataFrame:
    """Add weighted expected depth combining prev game with rolling."""
    cols = set(df.columns)
    
    if {"targets_prev_game", "air_yards_prev_game", "hist_air_yards_sum_l5", "hist_targets_l5"} <= cols:
        df = df.with_columns([
            pl.when(pl.col("targets_prev_game") >= 1)
            .then(
                0.6 * (pl.col("air_yards_prev_game") / pl.col("targets_prev_game").clip(1, None))
                + 0.4 * (pl.col("hist_air_yards_sum_l5") / pl.col("hist_targets_l5").clip(1, None))
            )
            .otherwise(
                pl.col("hist_air_yards_sum_l5") / pl.col("hist_targets_l5").clip(1, None)
            )
            .fill_null(0)
            .cast(pl.Float32)
            .alias("weighted_expected_depth")
        ])
    
    return df


def add_target_depth_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all target depth features.
    
    This is the main entry point that applies all target depth feature functions.
    Returns the dataframe with all target depth features added.
    """
    logger.info("Adding target depth features...")
    
    before_cols = len(df.columns)
    
    df = add_position_depth_priors(df)
    df = add_wr_alignment_features(df)
    df = add_qb_depth_tendencies(df)
    df = add_role_based_depth_tendencies(df)
    df = compute_historical_air_yards(df)
    df = add_derived_depth_features(df)
    df = add_depth_archetypes(df)
    df = add_weighted_depth_estimate(df)
    
    added = len(df.columns) - before_cols
    logger.info("    Added %d target depth features", added)
    
    return df

