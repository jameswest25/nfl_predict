"""Player aggregation functions for player-game level.

This module contains the core aggregation functions that combine
play-level data into player-game level statistics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass

__all__ = [
    "aggregate_passers",
    "aggregate_rushers",
    "aggregate_receivers",
    "merge_multi_role_players",
    "METADATA_COLUMNS",
]

logger = logging.getLogger(__name__)

# Metadata columns to carry through aggregation
METADATA_COLUMNS = [
    "stadium_key",
    "stadium_name",
    "stadium_tz",
    "roof",
    "surface",
    "home_team",
    "away_team",
    "game_start_utc",
    "season_type",
]


def _metadata_exprs(frame: pl.DataFrame) -> list[pl.Expr]:
    """Build metadata column aggregation expressions."""
    return [
        pl.col(col).first().alias(col)
        for col in METADATA_COLUMNS
        if col in frame.columns
    ]


def aggregate_passers(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate passing stats per (passer, game)."""
    
    # Filter to plays with a passer
    passers = df.filter(pl.col("passer_player_id").is_not_null())
    
    if passers.is_empty():
        return pl.DataFrame()
    
    # Aggregate per passer-game
    return (
        passers
        .group_by(["season", "week", "game_id", "game_date", "passer_player_id", "passer_player_name"])
        .agg([
            # Passing stats
            pl.col("passing_yards").fill_null(0).sum().cast(pl.Float64).alias("passing_yards"),
            pl.col("pass_attempt").fill_null(0).sum().cast(pl.Int64).alias("pass_attempt"),
            pl.col("completion").fill_null(0).sum().cast(pl.Int64).alias("completion"),
            pl.when(pl.col("pass_touchdown") == 1).then(1).otherwise(0).sum().cast(pl.Int64).alias("passing_td"),
            
            # Passer can also rush (e.g., Lamar Jackson)
            pl.col("rushing_yards").fill_null(0).sum().cast(pl.Float64).alias("rushing_yards"),
            pl.col("carry").fill_null(0).sum().cast(pl.Int64).alias("carry"),
            pl.when(pl.col("rush_touchdown") == 1).then(1).otherwise(0).sum().cast(pl.Int64).alias("rushing_td_count"),
            pl.col("red_zone_carry").fill_null(0).sum().cast(pl.Int64).alias("red_zone_carry"),
            pl.col("goal_to_go_carry").fill_null(0).sum().cast(pl.Int64).alias("goal_to_go_carry"),
            # Situational rushing: short-yardage (<=2 yds to go) and 3rd-down carries
            (
                pl.when((pl.col("carry").fill_null(0) > 0) & (pl.col("ydstogo").fill_null(100) <= 2))
                .then(1)
                .otherwise(0)
            )
            .sum()
            .cast(pl.Int64)
            .alias("short_yard_carry"),
            (
                pl.when((pl.col("carry").fill_null(0) > 0) & (pl.col("down").fill_null(0) == 3))
                .then(1)
                .otherwise(0)
            )
            .sum()
            .cast(pl.Int64)
            .alias("third_down_carry"),
            # No receiving for passers typically
            pl.lit(0.0).cast(pl.Float64).alias("receiving_yards"),
            pl.lit(0).cast(pl.Int64).alias("target"),
            pl.lit(0).cast(pl.Int64).alias("reception"),
            pl.lit(0).cast(pl.Int64).alias("receiving_td_count"),
            pl.lit(0).cast(pl.Int64).alias("red_zone_target"),
            pl.lit(0).cast(pl.Int64).alias("goal_to_go_target"),
            pl.col("yards_gained")
            .fill_null(0)
            .gt(0)
            .cast(pl.Int8)
            .sum()
            .cast(pl.Int64)
            .alias("rush_success_plays"),
            
            # Touchdowns (scored by player, not thrown)
            pl.when(pl.col("touchdown_player_id") == pl.col("passer_player_id"))
              .then(pl.col("touchdown").fill_null(0))
              .otherwise(0)
              .sum()
              .cast(pl.Int64)
              .alias("touchdowns"),
            
            # Team context
            pl.col("posteam").first().alias("team"),
            pl.col("defteam").first().alias("opponent"),
            *_metadata_exprs(passers),
        ])
        .rename({
            "passer_player_id": "player_id",
            "passer_player_name": "player_name",
        })
        .with_columns([
            # anytime_td: any non-passing TD credited to the passer (rush/receive/return)
            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
            # td_count: total TDs scored by the player
            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
        ])
    )


def aggregate_rushers(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate rushing stats per (rusher, game)."""
    
    rushers = df.filter(pl.col("rusher_player_id").is_not_null())
    
    if rushers.is_empty():
        return pl.DataFrame()
    
    return (
        rushers
        .group_by(["season", "week", "game_id", "game_date", "rusher_player_id", "rusher_player_name"])
        .agg([
            # Rushing stats
            pl.col("rushing_yards").fill_null(0).sum().cast(pl.Float64).alias("rushing_yards"),
            pl.col("carry").fill_null(0).sum().cast(pl.Int64).alias("carry"),
            pl.when(pl.col("rush_touchdown") == 1).then(1).otherwise(0).sum().cast(pl.Int64).alias("rushing_td_count"),
            pl.col("red_zone_carry").fill_null(0).sum().cast(pl.Int64).alias("red_zone_carry"),
            pl.col("goal_to_go_carry").fill_null(0).sum().cast(pl.Int64).alias("goal_to_go_carry"),
            (
                pl.when((pl.col("carry").fill_null(0) > 0) & (pl.col("ydstogo").fill_null(100) <= 2))
                .then(1)
                .otherwise(0)
            )
            .sum()
            .cast(pl.Int64)
            .alias("short_yard_carry"),
            (
                pl.when((pl.col("carry").fill_null(0) > 0) & (pl.col("down").fill_null(0) == 3))
                .then(1)
                .otherwise(0)
            )
            .sum()
            .cast(pl.Int64)
            .alias("third_down_carry"),
            
            # Initialize other stats to 0
            pl.lit(0.0).cast(pl.Float64).alias("passing_yards"),
            pl.lit(0).cast(pl.Int64).alias("pass_attempt"),
            pl.lit(0).cast(pl.Int64).alias("completion"),
            pl.lit(0).cast(pl.Int64).alias("passing_td"),
            pl.lit(0.0).cast(pl.Float64).alias("receiving_yards"),
            pl.lit(0).cast(pl.Int64).alias("target"),
            pl.lit(0).cast(pl.Int64).alias("reception"),
            pl.lit(0).cast(pl.Int64).alias("receiving_td_count"),
            pl.lit(0).cast(pl.Int64).alias("red_zone_target"),
            pl.lit(0).cast(pl.Int64).alias("goal_to_go_target"),
            pl.lit(0).cast(pl.Int64).alias("rush_success_plays"),
            
            # Touchdowns
            pl.when(pl.col("touchdown_player_id") == pl.col("rusher_player_id"))
              .then(pl.col("touchdown").fill_null(0))
              .otherwise(0)
              .sum()
              .cast(pl.Int64)
              .alias("touchdowns"),
            
            # Team context
            pl.col("posteam").first().alias("team"),
            pl.col("defteam").first().alias("opponent"),
            *_metadata_exprs(rushers),
        ])
        .rename({
            "rusher_player_id": "player_id",
            "rusher_player_name": "player_name",
        })
        .with_columns([
            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
        ])
    )


def aggregate_receivers(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate receiving stats per (receiver, game)."""
    
    receivers = df.filter(pl.col("receiver_player_id").is_not_null())
    
    if receivers.is_empty():
        return pl.DataFrame()
    
    # Build dynamic aggregation list based on available columns
    agg_exprs = [
            # Receiving stats
            pl.col("receiving_yards").fill_null(0).sum().cast(pl.Float64).alias("receiving_yards"),
            pl.col("target").fill_null(0).sum().cast(pl.Int64).alias("target"),
            pl.col("reception").fill_null(0).sum().cast(pl.Int64).alias("reception"),
            pl.when(pl.col("pass_touchdown") == 1).then(1).otherwise(0).sum().cast(pl.Int64).alias("receiving_td_count"),
            pl.col("red_zone_target").fill_null(0).sum().cast(pl.Int64).alias("red_zone_target"),
            pl.col("goal_to_go_target").fill_null(0).sum().cast(pl.Int64).alias("goal_to_go_target"),
    ]
    
    # Add air yards if available
    if "air_yards" in receivers.columns:
        # Air yards on completions only (for efficiency_rec_yards_air)
        agg_exprs.append(
            pl.when(pl.col("complete_pass") == 1)
            .then(pl.col("air_yards"))
            .otherwise(0)
            .fill_null(0)
            .sum()
            .cast(pl.Float64)
            .alias("air_yards_total")
        )
        # Air yards on ALL targets (for usage_target_yards - depth of target)
        agg_exprs.append(
            pl.col("air_yards")
            .fill_null(0)
            .sum()
            .cast(pl.Float64)
            .alias("air_yards_all_targets")
        )
    
    # Add yards after catch if available (only for completions)
    if "yards_after_catch" in receivers.columns:
        agg_exprs.append(
            pl.col("yards_after_catch")
            .fill_null(0)
            .sum()
            .cast(pl.Float64)
            .alias("yards_after_catch_total")
        )
    
    return (
        receivers
        .group_by(["season", "week", "game_id", "game_date", "receiver_player_id", "receiver_player_name"])
        .agg(agg_exprs + [
            
            # Initialize other stats to 0
            pl.lit(0.0).cast(pl.Float64).alias("passing_yards"),
            pl.lit(0).cast(pl.Int64).alias("pass_attempt"),
            pl.lit(0).cast(pl.Int64).alias("completion"),
            pl.lit(0).cast(pl.Int64).alias("passing_td"),
            pl.lit(0.0).cast(pl.Float64).alias("rushing_yards"),
            pl.lit(0).cast(pl.Int64).alias("carry"),
            pl.lit(0).cast(pl.Int64).alias("rushing_td_count"),
            pl.lit(0).cast(pl.Int64).alias("red_zone_carry"),
            pl.lit(0).cast(pl.Int64).alias("goal_to_go_carry"),
            pl.lit(0).cast(pl.Int64).alias("rush_success_plays"),
            
            # Touchdowns
            pl.when(pl.col("touchdown_player_id") == pl.col("receiver_player_id"))
              .then(pl.col("touchdown").fill_null(0))
              .otherwise(0)
              .sum()
              .cast(pl.Int64)
              .alias("touchdowns"),
            
            # Team context
            pl.col("posteam").first().alias("team"),
            pl.col("defteam").first().alias("opponent"),
            *_metadata_exprs(receivers),
        ])
        .rename({
            "receiver_player_id": "player_id",
            "receiver_player_name": "player_name",
        })
        .with_columns([
            (pl.col("touchdowns") > 0).cast(pl.Int8).alias("anytime_td"),
            pl.col("touchdowns").cast(pl.Int64).alias("td_count"),
        ])
    )


def merge_multi_role_players(df: pl.DataFrame, *, label_version: str | None = None) -> pl.DataFrame:
    """Merge stats for players who had multiple roles in same game.
    
    Example: QB who passed and rushed, WR who received and rushed.
    """
    from utils.feature.core.labels import DEFAULT_LABEL_VERSION, compute_td_labels
    
    label_version = label_version or DEFAULT_LABEL_VERSION
    
    # Build aggregation expressions based on available columns
    agg_exprs = [
            # Sum all yardage
            pl.col("passing_yards").fill_null(0).sum().alias("passing_yards"),
            pl.col("rushing_yards").fill_null(0).sum().alias("rushing_yards"),
            pl.col("receiving_yards").fill_null(0).sum().alias("receiving_yards"),
            
            # Sum counts
            pl.col("pass_attempt").fill_null(0).sum().alias("pass_attempt"),
            pl.col("completion").fill_null(0).sum().alias("completion"),
            pl.col("carry").fill_null(0).sum().alias("carry"),
            pl.col("target").fill_null(0).sum().alias("target"),
            pl.col("reception").fill_null(0).sum().alias("reception"),
            
            # Sum red-zone / goal-to-go opportunity counts
            pl.col("red_zone_target").fill_null(0).sum().alias("red_zone_target"),
            pl.col("red_zone_carry").fill_null(0).sum().alias("red_zone_carry"),
            pl.col("goal_to_go_target").fill_null(0).sum().alias("goal_to_go_target"),
            pl.col("goal_to_go_carry").fill_null(0).sum().alias("goal_to_go_carry"),
            pl.col("rush_success_plays").fill_null(0).sum().alias("rush_success_plays"),
            
            # Sum TD counts
            pl.col("passing_td").fill_null(0).sum().alias("passing_td"),
            pl.col("rushing_td_count").fill_null(0).sum().alias("rushing_td_count"),
            pl.col("receiving_td_count").fill_null(0).sum().alias("receiving_td_count"),
            pl.col("touchdowns").fill_null(0).sum().alias("touchdowns"),
            
            # Team context (take first, should be same)
            pl.col("team").first().alias("team"),
            pl.col("opponent").first().alias("opponent"),
    ]
    
    # Add air_yards if present
    if "air_yards_total" in df.columns:
        agg_exprs.append(pl.col("air_yards_total").fill_null(0).sum().alias("air_yards_total"))
    if "air_yards_all_targets" in df.columns:
        agg_exprs.append(pl.col("air_yards_all_targets").fill_null(0).sum().alias("air_yards_all_targets"))
    if "yards_after_catch_total" in df.columns:
        agg_exprs.append(pl.col("yards_after_catch_total").fill_null(0).sum().alias("yards_after_catch_total"))
    
    # Metadata columns (take first)
    for col in METADATA_COLUMNS:
        if col in df.columns:
            agg_exprs.append(pl.col(col).first().alias(col))
    
    # Group and aggregate
    merged = (
        df
        .group_by(["season", "week", "game_id", "game_date", "player_id", "player_name"])
        .agg(agg_exprs)
    )
    
    # Fill nulls with 0 for numeric columns
    numeric_cols = [
        "passing_yards", "rushing_yards", "receiving_yards",
        "pass_attempt", "completion", "carry", "target", "reception",
        "passing_td", "rushing_td_count", "receiving_td_count", "touchdowns",
        "td_count", "red_zone_target", "red_zone_carry", "goal_to_go_target", "goal_to_go_carry",
        "rush_success_plays"
    ]
    
    for col in numeric_cols:
        if col in merged.columns:
            merged = merged.with_columns(pl.col(col).fill_null(0))

    merged = merged.with_columns([
        pl.col("touchdowns").fill_null(0).alias("touchdowns"),
        pl.col("rushing_td_count").fill_null(0).alias("rushing_td_count"),
        pl.col("receiving_td_count").fill_null(0).alias("receiving_td_count"),
        pl.col("passing_td").fill_null(0).alias("passing_td"),
        pl.col("red_zone_target").fill_null(0).alias("red_zone_target"),
        pl.col("red_zone_carry").fill_null(0).alias("red_zone_carry"),
        pl.col("goal_to_go_target").fill_null(0).alias("goal_to_go_target"),
        pl.col("goal_to_go_carry").fill_null(0).alias("goal_to_go_carry"),
    ])
    
    # Recompute touchdown labels after merging
    merged = compute_td_labels(merged, version=label_version)
    
    return merged
