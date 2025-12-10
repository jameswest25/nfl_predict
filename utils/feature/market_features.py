"""Market/odds-derived features for NFL player predictions.

This module computes features derived from betting market data including
team implied totals and market-weighted usage interactions.
"""

from __future__ import annotations

import logging

import polars as pl

__all__ = [
    "add_team_implied_totals",
    "add_market_interaction_features",
    "add_market_features",
]

logger = logging.getLogger(__name__)


def add_team_implied_totals(df: pl.DataFrame) -> pl.DataFrame:
    """Derive team and opponent implied totals from spread/total.
    
    Computes implied team scores based on Vegas lines.
    """
    odds_cols = set(df.columns)
    required_for_totals = {"total_line", "spread_line", "team", "home_team", "away_team"}
    
    if not required_for_totals <= odds_cols:
        return df
    
    total = pl.col("total_line").cast(pl.Float32)
    spread = pl.col("spread_line").cast(pl.Float32)
    
    # Home team total ≈ (total - home_spread) / 2 when spread is from the home perspective
    home_total = (total - spread) / 2.0
    away_total = total - home_total
    
    df = df.with_columns([
        pl.when(pl.col("team") == pl.col("home_team"))
        .then(home_total)
        .when(pl.col("team") == pl.col("away_team"))
        .then(away_total)
        .otherwise(None)
        .cast(pl.Float32)
        .alias("team_implied_total"),
        
        pl.when(pl.col("team") == pl.col("home_team"))
        .then(away_total)
        .when(pl.col("team") == pl.col("away_team"))
        .then(home_total)
        .otherwise(None)
        .cast(pl.Float32)
        .alias("opp_implied_total"),
    ])
    
    return df


def add_market_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add market-weighted player usage interaction features.
    
    Combines team implied totals with historical player shares to create
    expected value features.
    """
    available_cols = set(df.columns)
    market_exprs: list[pl.Expr] = []
    
    if {"team_implied_total", "hist_target_share_prev"} <= available_cols:
        market_exprs.append(
            (pl.col("team_implied_total") * pl.col("hist_target_share_prev"))
            .cast(pl.Float32)
            .alias("market_team_total_x_hist_target_prev")
        )
    
    if {"team_implied_total", "hist_carry_share_prev"} <= available_cols:
        market_exprs.append(
            (pl.col("team_implied_total") * pl.col("hist_carry_share_prev"))
            .cast(pl.Float32)
            .alias("market_team_total_x_hist_carry_prev")
        )
    
    if {"team_implied_total", "hist_target_share_l3"} <= available_cols:
        market_exprs.append(
            (pl.col("team_implied_total") * pl.col("hist_target_share_l3"))
            .cast(pl.Float32)
            .alias("market_team_total_x_hist_target_l3")
        )
    
    if {"team_implied_total", "hist_carry_share_l3"} <= available_cols:
        market_exprs.append(
            (pl.col("team_implied_total") * pl.col("hist_carry_share_l3"))
            .cast(pl.Float32)
            .alias("market_team_total_x_hist_carry_l3")
        )
    
    if market_exprs:
        df = df.with_columns(market_exprs)
        logger.info("Added %d market interaction features", len(market_exprs))
    
    return df


def add_pace_matchup_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add pace and defensive matchup features.
    
    Derives pace-related features from team context columns.
    """
    logger.info("Adding pace and defensive matchup features...")
    
    pace_exprs: list[pl.Expr] = []
    available_cols = set(df.columns)
    
    if "team_ctx_offensive_plays_prev" in available_cols:
        pace_exprs.append(
            pl.col("team_ctx_offensive_plays_prev").cast(pl.Float32).alias("team_pace_prev")
        )
    
    if "team_ctx_offensive_plays_l3" in available_cols:
        pace_exprs.append(
            pl.col("team_ctx_offensive_plays_l3").cast(pl.Float32).alias("team_pace_l3")
        )
    
    if "opp_ctx_offensive_plays_prev" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_offensive_plays_prev").cast(pl.Float32).alias("opp_pace_prev")
        )
    
    if "opp_ctx_offensive_plays_l3" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_offensive_plays_l3").cast(pl.Float32).alias("opp_pace_l3")
        )
    
    if {"team_ctx_offensive_plays_prev", "opp_ctx_offensive_plays_prev"} <= available_cols:
        pace_exprs.append(
            (pl.col("team_ctx_offensive_plays_prev") + pl.col("opp_ctx_offensive_plays_prev"))
            .cast(pl.Float32)
            .alias("matchup_pace_prev")
        )
        pace_exprs.append(
            (pl.col("team_ctx_offensive_plays_prev") - pl.col("opp_ctx_offensive_plays_prev"))
            .cast(pl.Float32)
            .alias("pace_diff_prev")
        )
    
    if {"team_ctx_offensive_plays_l3", "opp_ctx_offensive_plays_l3"} <= available_cols:
        pace_exprs.append(
            (pl.col("team_ctx_offensive_plays_l3") + pl.col("opp_ctx_offensive_plays_l3"))
            .cast(pl.Float32)
            .alias("matchup_pace_l3")
        )
        pace_exprs.append(
            (pl.col("team_ctx_offensive_plays_l3") - pl.col("opp_ctx_offensive_plays_l3"))
            .cast(pl.Float32)
            .alias("pace_diff_l3")
        )
    
    # Defensive matchup features
    if "opp_ctx_red_zone_play_rate_prev" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_red_zone_play_rate_prev").cast(pl.Float32).alias("opp_def_red_zone_play_rate_prev")
        )
    
    if "opp_ctx_red_zone_play_rate_l3" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_red_zone_play_rate_l3").cast(pl.Float32).alias("opp_def_red_zone_play_rate_l3")
        )
    
    if "opp_ctx_td_per_play_prev" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_td_per_play_prev").cast(pl.Float32).alias("opp_def_td_per_play_prev")
        )
    
    if "opp_ctx_td_per_play_l3" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_td_per_play_l3").cast(pl.Float32).alias("opp_def_td_per_play_l3")
        )
    
    if pace_exprs:
        df = df.with_columns(pace_exprs)
    
    return df


def add_market_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all market-derived features.
    
    This is the main entry point that applies all market feature functions.
    """
    before_cols = len(df.columns)
    
    df = add_team_implied_totals(df)
    df = add_pace_matchup_features(df)
    df = add_market_interaction_features(df)
    
    added = len(df.columns) - before_cols
    logger.info("    Total market features added: %d", added)
    
    return df

