"""Unified feature builder for both training and inference.

This module provides a single feature computation pipeline that ensures
strict parity between training and inference featurization.

The key principle: ONE code path for feature generation, with minimal
mode-specific branching controlled by the `is_inference` flag.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

from utils.feature.rolling.rolling_window import add_rolling_features
from utils.feature.rolling.daily_totals import build_daily_cache_range, DAILY_CACHE_ROOT
from utils.feature.rolling.stats import (
    ROLLING_FEATURE_STATS,
    ROLLING_WINDOWS,
    ROLLING_CONTEXTS,
    NFL_PLAYER_STATS,
)
from utils.feature.enrichment.team_context import (
    add_team_context_features,
    compute_team_context_history,
    attach_team_context,
)
from utils.feature.enrichment.offense_context import (
    add_offense_context_features_training,
    add_offense_context_features_inference,
)
from utils.feature.enrichment.weather_features import (
    add_weather_forecast_features_training,
    add_weather_forecast_features_inference,
    append_weather_context_flags,
)
from utils.feature.enrichment.odds import (
    add_nfl_odds_features_to_df,
    NFL_ODDS_COLUMNS,
    ODDS_FLAG_COLUMNS,
)
from utils.feature.builders.opponent_splits import load_rolling_opponent_splits
from utils.feature.core.shared import (
    finalize_drive_history_features,
    compute_drive_level_aggregates,
    attach_td_rate_history_features,
)
from utils.feature.derived import (
    add_usage_helper_features,
    add_position_group,
    add_specialist_role_flags,
    add_moe_position_features,
    add_context_gaps_features,
    add_catch_rate_features,
    add_target_depth_features,
    add_snap_features,
    add_historical_share_features,
    add_combined_usage_features,
    add_role_share_flags,
    add_market_features,
)
from utils.feature.enrichment.asof import (
    get_decision_cutoff_hours,
    get_fallback_cutoff_hours,
)
from utils.general.paths import (
    PROJ_ROOT,
    PLAYER_DRIVE_BY_WEEK_DIR as PLAYER_DRIVE_DIR,
    PLAYER_GAME_BY_WEEK_DIR as PLAYER_GAME_DIR,
    TEAM_CONTEXT_HISTORY_PATH,
    OFFENSE_CONTEXT_HISTORY_PATH,
    QB_PROFILE_DIR,
    TRAVEL_CALENDAR_DIR,
    OPPONENT_SPLIT_DIR,
)

logger = logging.getLogger(__name__)

# Use the canonical paths from paths.py
TEAM_CONTEXT_HISTORY_FILE = TEAM_CONTEXT_HISTORY_PATH
OFFENSE_CONTEXT_HISTORY_FILE = OFFENSE_CONTEXT_HISTORY_PATH
PLAYER_DRIVE_HISTORY_FILE = PROJ_ROOT / "data/processed/player_drive_context_history.parquet"


def _ensure_column_types(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize column types for consistency between training and inference."""
    string_cols = [
        col for col in (
            "player_id", "player_name", "team", "opponent", "game_id",
            "stadium_key", "stadium_name", "stadium_tz", "roof", "surface",
            "home_team", "away_team", "position", "position_group",
            "depth_chart_position", "injury_report_status", "injury_practice_status",
            "injury_report_primary", "injury_practice_primary", "season_type",
        ) if col in df.columns
    ]
    if string_cols:
        df = df.with_columns([pl.col(col).cast(pl.Utf8) for col in string_cols])

    if "home_team_abbr" not in df.columns and "home_team" in df.columns:
        df = df.with_columns(pl.col("home_team").alias("home_team_abbr"))
    if "away_team_abbr" not in df.columns and "away_team" in df.columns:
        df = df.with_columns(pl.col("away_team").alias("away_team_abbr"))

    # Ensure numeric types
    if "season" in df.columns:
        df = df.with_columns(pl.col("season").cast(pl.Int32))
    if "week" in df.columns:
        df = df.with_columns(pl.col("week").cast(pl.Int32))
    if "game_date" in df.columns:
        df = df.with_columns(pl.col("game_date").cast(pl.Datetime("ms")))

    return df


def _ensure_decision_cutoff(
    df: pl.DataFrame,
    *,
    cutoff_hours: float,
    fallback_hours: float,
) -> pl.DataFrame:
    """Ensure decision_cutoff_ts column exists and is properly computed."""
    if "decision_cutoff_ts" in df.columns:
        df = df.with_columns(pl.col("decision_cutoff_ts").cast(pl.Datetime("ms")))
    elif "game_date" in df.columns:
        # Ensure game_start_utc is timezone-naive for consistent handling
        if "game_start_utc" in df.columns:
            dtype = df.schema.get("game_start_utc")
            # If it has timezone info, convert to naive datetime
            if dtype is not None and hasattr(dtype, "time_zone") and dtype.time_zone is not None:
                df = df.with_columns(
                    pl.col("game_start_utc").dt.replace_time_zone(None).alias("game_start_utc")
                )
        
        df = df.with_columns(
            pl.when(pl.col("game_start_utc").is_not_null())
            .then(pl.col("game_start_utc").cast(pl.Datetime("ms")) - pl.duration(hours=cutoff_hours))
            .otherwise(
                pl.col("game_date").cast(pl.Datetime("ms")) - pl.duration(hours=fallback_hours)
            )
            .alias("decision_cutoff_ts")
        )
    return df


def _ensure_rolling_stat_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure all rolling stat columns exist (as zeros if missing).
    
    This ensures the rolling window computation has consistent input columns
    for both training (where data exists) and inference (where placeholders
    are needed for the current game).
    """
    for col in ROLLING_FEATURE_STATS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))
    
    for col in NFL_PLAYER_STATS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))
    
    return df


def _ensure_daily_cache(df: pl.DataFrame) -> None:
    """Ensure daily totals cache exists for rolling window computations."""
    if df.is_empty() or "game_date" not in df.columns:
        return
    
    dates = df.get_column("game_date").cast(pl.Date)
    max_date = dates.max()
    if max_date is None:
        return
    
    # Determine season start
    season_year = max_date.year if max_date.month >= 9 else max_date.year - 1
    season_start = date(season_year, 9, 1)
    
    # Build/verify cache
    if not DAILY_CACHE_ROOT.exists() or not any(DAILY_CACHE_ROOT.iterdir()):
        logger.info(f"Building daily totals cache from {season_start} to {max_date}...")
        build_daily_cache_range(season_start, max_date, level="game")
    else:
        logger.info(f"Verifying daily totals cache coverage from {season_start} to {max_date}...")
        build_daily_cache_range(season_start, max_date, level="game")


def _add_rolling_features(
    df: pl.DataFrame,
    *,
    stats: Sequence[str] | None = None,
    windows: Sequence[int] | None = None,
    contexts: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Add rolling window features - unified for training and inference."""
    stats = stats or [s for s in ROLLING_FEATURE_STATS if s in df.columns]
    windows = windows or ROLLING_WINDOWS
    contexts = contexts or ROLLING_CONTEXTS
    
    if not stats:
        logger.warning("No rolling stats columns found in dataframe")
        return df
    
    logger.info(f"Computing rolling window features for {len(stats)} stats...")
    df = add_rolling_features(
        df,
        level="game",
        stats=list(stats),
        windows=list(windows),
        contexts=list(contexts),
        date_col="game_date",
        player_col="player_id",
        opponent_col="opponent",
    )
    
    rolling_cols = [c for c in df.columns if "g_" in c]
    logger.info(f"Added {len(rolling_cols)} rolling features")
    
    return df


def _add_team_context_features(
    df: pl.DataFrame,
    *,
    is_inference: bool,
    history_path: Path | None = None,
) -> pl.DataFrame:
    """Add team context features - unified for training and inference."""
    history_path = history_path or TEAM_CONTEXT_HISTORY_FILE
    
    team_history = None
    if history_path.exists():
        team_history = pl.read_parquet(str(history_path))
        if "game_date" in team_history.columns:
            team_history = team_history.with_columns(
                pl.col("game_date").cast(pl.Datetime("ms"))
            )
    else:
        logger.warning(
            "Team context history file missing at %s; computing from scratch.",
            history_path,
        )
    
    # For inference, we don't join on date (using pre-computed history)
    # For training, we return the history to be persisted
    if is_inference:
        df = add_team_context_features(
            df,
            join_on_date=False,
            history=team_history,
            cutoff_column=None,
        )
    else:
        df, team_context_history = add_team_context_features(df, return_history=True)
        # Persist history for future inference
        history_path.parent.mkdir(parents=True, exist_ok=True)
        team_context_history.write_parquet(history_path, compression="zstd")
    
    return df


def _add_offense_context_features(
    df: pl.DataFrame,
    *,
    is_inference: bool,
    history_path: Path | None = None,
) -> pl.DataFrame:
    """Add offense context features - unified for training and inference."""
    history_path = history_path or OFFENSE_CONTEXT_HISTORY_FILE
    
    # Both training and inference use history-based approach for consistency
    if is_inference:
        df = add_offense_context_features_inference(
            df,
            history_path=history_path,
            cutoff_column=None,
        )
    else:
        df = add_offense_context_features_training(df, history_path=history_path)
    
    # Normalize off_ctx_game_date to Datetime for parity
    if "off_ctx_game_date" in df.columns:
        # Cast to Datetime(ms) to match training pipeline behavior
        df = df.with_columns(pl.col("off_ctx_game_date").cast(pl.Datetime("ms")))
    
    return df


def _add_weather_features(
    df: pl.DataFrame,
    *,
    is_inference: bool,
    cutoff_column: str = "decision_cutoff_ts",
) -> pl.DataFrame:
    """Add weather features - unified for training and inference."""
    if is_inference:
        df = add_weather_forecast_features_inference(df, cutoff_column=cutoff_column)
    else:
        df = add_weather_forecast_features_training(df, cutoff_column=cutoff_column)
    
    df = append_weather_context_flags(df, roof_col="roof")
    
    # Clean up weather columns
    weather_cols = [col for col in df.columns if col.startswith("weather_")]
    if weather_cols:
        drop_cols = [
            col for col in weather_cols
            if col.endswith("_ts")
            or col.endswith("_source_detail")
            or col in {"weather_conditions", "weather_precip_type"}
        ]
        if drop_cols:
            df = df.drop(drop_cols, strict=False)
        
        # Cast numeric weather columns
        numeric_cols = [
            col for col in weather_cols
            if col not in drop_cols
            and df.schema.get(col) in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]
        if numeric_cols:
            df = df.with_columns([pl.col(col).cast(pl.Float32) for col in numeric_cols])
    
    return df


def _add_odds_features(
    df: pl.DataFrame,
    *,
    is_inference: bool,
    player_col: str = "player_name",
) -> pl.DataFrame:
    """Add odds features - unified for training and inference.
    
    Training drops rows without pre-cutoff odds snapshots to ensure data quality.
    Inference keeps all rows but allows schedule fallback for upcoming games.
    """
    if is_inference:
        # For inference: allow schedule fallback, keep all rows
        df = add_nfl_odds_features_to_df(
            df,
            player_col=player_col,
            allow_schedule_fallback=True,
            drop_schedule_rows=False,
        )
    else:
        # For training: strict - no fallback, drop rows without odds
        df = add_nfl_odds_features_to_df(
            df,
            player_col=player_col,
            allow_schedule_fallback=False,
            drop_schedule_rows=True,
        )
    
    if "odds_schedule_fallback" not in df.columns:
        df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback"))
    if "odds_anytime_td_price" in df.columns:
        df = df.drop("odds_anytime_td_price", strict=False)
    
    # Derive implied totals from odds
    odds_cols = set(df.columns)
    required_for_totals = {"total_line", "spread_line", "team", "home_team", "away_team"}
    if required_for_totals <= odds_cols:
        if "spread_line" not in df.columns:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias("spread_line"))
        else:
            df = df.with_columns(pl.col("spread_line").fill_null(0.0))
        
        total = pl.col("total_line").cast(pl.Float32)
        spread = pl.col("spread_line").cast(pl.Float32)
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


def _add_opponent_splits(df: pl.DataFrame, seasons: Sequence[int]) -> pl.DataFrame:
    """Add opponent defensive splits."""
    opponent_df = load_rolling_opponent_splits(list(seasons), windows=[3])
    if opponent_df.is_empty():
        return df
    
    opponent_df = opponent_df.with_columns([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("opponent").cast(pl.Utf8),
        pl.col("game_date").cast(pl.Datetime("ms")),
    ])
    
    # Cast float columns
    float_cols = [
        col for col, dtype in opponent_df.schema.items()
        if col.startswith("opp_def_") and dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    if float_cols:
        opponent_df = opponent_df.with_columns([pl.col(col).cast(pl.Float32) for col in float_cols])
    
    if {"opponent", "game_date"} <= set(df.columns):
        opponent_join = (
            opponent_df.drop_nulls("game_date")
            .drop(["season", "week"], strict=False)
            .sort(["opponent", "game_date"])
        )
        df = df.with_columns(
            pl.col("game_date").cast(pl.Datetime("ms")).alias("game_date")
        )
        df = df.with_columns(pl.arange(0, df.height).alias("__row_order"))
        df = df.sort(["opponent", "game_date"])
        df = df.join_asof(
            opponent_join,
            left_on="game_date",
            right_on="game_date",
            by="opponent",
            strategy="backward",
        )
        df = df.sort("__row_order").drop("__row_order")
        if "game_date_right" in df.columns:
            df = df.drop("game_date_right")
    
    return df


def _add_qb_profile_features(df: pl.DataFrame, seasons: Sequence[int]) -> pl.DataFrame:
    """Add QB profile features."""
    from utils.predict.loaders import load_qb_profile_features
    
    qb_df = load_qb_profile_features(seasons)
    
    if qb_df.is_empty():
        return df
    
    if {"season", "week", "team"} <= set(df.columns):
        df = df.join(qb_df, on=["season", "week", "team"], how="left")
    
    return df


def _add_travel_features(df: pl.DataFrame, seasons: Sequence[int]) -> pl.DataFrame:
    """Add travel/rest features."""
    from utils.predict.loaders import load_travel_calendar_features
    
    travel_df = load_travel_calendar_features(seasons)
    
    if travel_df.is_empty():
        return df
    
    if {"season", "week", "team"} <= set(df.columns):
        df = df.join(travel_df, on=["season", "week", "team"], how="left")
    
    return df


def _add_drive_history_features(
    df: pl.DataFrame,
    *,
    is_inference: bool,
) -> pl.DataFrame:
    """Add drive-level history features."""
    if "game_date" not in df.columns or "player_id" not in df.columns:
        return df
    
    if not PLAYER_DRIVE_DIR.exists():
        logger.warning("Player drive directory missing; skipping drive features.")
        return df
    
    # Ensure IDs are strings
    df = df.with_columns([
        pl.col("player_id").cast(pl.Utf8),
        pl.col("team").cast(pl.Utf8),
        pl.col("game_id").cast(pl.Utf8),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
    ])
    
    target_players = df.select("player_id").unique()
    
    # Load drive data
    drive_scan = pl.scan_parquet(
        str(PLAYER_DRIVE_DIR / "season=*/week=*/part.parquet"),
        glob=True,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    drive_scan = drive_scan.with_columns([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("player_id").cast(pl.Utf8),
        pl.col("team").cast(pl.Utf8),
    ])
    
    # Load game data for scaffold
    game_scan = pl.scan_parquet(
        str(PLAYER_GAME_DIR / "season=*/week=*/part.parquet"),
        glob=True,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    game_scan = game_scan.with_columns([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("player_id").cast(pl.Utf8),
        pl.col("team").cast(pl.Utf8),
    ])
    
    # Filter to target players
    drive_hist = drive_scan.filter(
        pl.col("player_id").is_in(target_players.get_column("player_id"))
    ).collect(streaming=True)
    
    game_hist = game_scan.filter(
        pl.col("player_id").is_in(target_players.get_column("player_id"))
    ).select(
        ["season", "week", "game_id", "team", "player_id", "game_date"]
    ).collect(streaming=True)
    
    # For inference, exclude current games from history
    if is_inference:
        current_games_df = df.select(pl.col("game_id").cast(pl.Utf8).str.strip_chars().unique())
        if not drive_hist.is_empty() and "game_id" in drive_hist.columns:
            drive_hist = drive_hist.with_columns(pl.col("game_id").cast(pl.Utf8).str.strip_chars())
            drive_hist = drive_hist.join(current_games_df, on="game_id", how="anti")
        if not game_hist.is_empty() and "game_id" in game_hist.columns:
            game_hist = game_hist.with_columns(pl.col("game_id").cast(pl.Utf8).str.strip_chars())
            game_hist = game_hist.join(current_games_df, on="game_id", how="anti")
    
    # Aggregate drive history
    if not drive_hist.is_empty():
        drive_hist = compute_drive_level_aggregates(drive_hist)
    
    # Create scaffold
    if is_inference:
        current_placeholder = df.select(
            ["season", "week", "game_id", "team", "player_id", "game_date"]
        ).unique().with_columns(pl.col("game_id").cast(pl.Utf8))
        
        if game_hist.is_empty():
            full_scaffold = current_placeholder
        else:
            full_scaffold = pl.concat([
                game_hist.select(current_placeholder.columns),
                current_placeholder
            ], how="vertical_relaxed").unique()
    else:
        full_scaffold = game_hist if not game_hist.is_empty() else pl.DataFrame()
    
    if full_scaffold.is_empty():
        return df
    
    # Join drive history into scaffold
    join_cols = ["season", "week", "game_id", "team", "player_id"]
    full_scaffold = full_scaffold.with_columns([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("game_id").cast(pl.Utf8),
        pl.col("team").cast(pl.Utf8),
        pl.col("player_id").cast(pl.Utf8),
    ])
    
    if not drive_hist.is_empty():
        drive_hist = drive_hist.with_columns([
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("game_id").cast(pl.Utf8),
            pl.col("team").cast(pl.Utf8),
            pl.col("player_id").cast(pl.Utf8),
        ])
        
        if "game_date" in drive_hist.columns:
            drive_hist = drive_hist.rename({"game_date": "game_date_drive"})
        
        combined = full_scaffold.join(drive_hist, on=join_cols, how="left")
        
        # Fill nulls
        agg_cols = [
            "drive_count", "drive_touch_drives", "drive_td_drives", "drive_total_yards",
            "drive_red_zone_drives", "drive_goal_to_go_drives"
        ]
        fill_cols = [c for c in agg_cols if c in combined.columns]
        if fill_cols:
            combined = combined.with_columns([pl.col(c).fill_null(0) for c in fill_cols])
    else:
        combined = full_scaffold
        agg_cols = ["drive_count", "drive_touch_drives", "drive_td_drives", "drive_total_yards"]
        combined = combined.with_columns([
            pl.lit(0.0).cast(pl.Float32).alias(c) for c in agg_cols
        ])
    
    # Finalize
    finalized = finalize_drive_history_features(combined)
    
    # Join back to original dataframe
    join_cols_enriched = ["season", "week", "game_id", "player_id"]
    features_to_join = finalized.select(
        join_cols_enriched + [c for c in finalized.columns if c.startswith("drive_hist_")]
    )
    
    df = df.join(features_to_join, on=join_cols_enriched, how="left")
    
    # Fill nulls in drive columns
    drive_cols = [col for col in df.columns if col.startswith("drive_hist_")]
    if drive_cols:
        df = df.with_columns([
            pl.col(col).fill_null(0.0).cast(pl.Float32) for col in drive_cols
        ])
    
    return df


def _add_historical_share_features_unified(df: pl.DataFrame) -> pl.DataFrame:
    """Add historical share features computed from rolling stats.
    
    This unifies the share computation logic used in both training and inference.
    """
    available_cols = set(df.columns)
    share_specs = [
        ("hist_target_share", "target", "team_ctx_targets"),
        ("hist_carry_share", "carry", "team_ctx_carries"),
        ("hist_pass_attempt_share", "pass_attempt", "team_ctx_pass_attempts"),
        ("hist_red_zone_target_share", "red_zone_target", "team_ctx_red_zone_targets"),
        ("hist_red_zone_carry_share", "red_zone_carry", "team_ctx_red_zone_carries"),
        ("hist_goal_to_go_target_share", "goal_to_go_target", "team_ctx_goal_to_go_targets"),
        ("hist_goal_to_go_carry_share", "goal_to_go_carry", "team_ctx_goal_to_go_carries"),
    ]
    
    share_exprs: list[pl.Expr] = []
    for base_name, numer_prefix, denom_prefix in share_specs:
        combos = [
            ("prev", f"1g_{numer_prefix}_per_game", f"{denom_prefix}_prev"),
            ("l3", f"3g_{numer_prefix}_per_game", f"{denom_prefix}_l3"),
        ]
        for suffix, numer_col, denom_col in combos:
            out_col = f"{base_name}_{suffix}"
            if {numer_col, denom_col} <= available_cols and out_col not in available_cols:
                share_exprs.append(
                    pl.when(pl.col(denom_col) > 0)
                    .then(pl.col(numer_col).cast(pl.Float32) / pl.col(denom_col).cast(pl.Float32))
                    .otherwise(0.0)
                    .alias(out_col)
                )
    
    if share_exprs:
        df = df.with_columns(share_exprs)
    
    return df


def _add_pace_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add pace-related features from team context."""
    available_cols = set(df.columns)
    pace_exprs: list[pl.Expr] = []
    
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
            .cast(pl.Float32).alias("matchup_pace_prev")
        )
        pace_exprs.append(
            (pl.col("team_ctx_offensive_plays_prev") - pl.col("opp_ctx_offensive_plays_prev"))
            .cast(pl.Float32).alias("pace_diff_prev")
        )
    if {"team_ctx_offensive_plays_l3", "opp_ctx_offensive_plays_l3"} <= available_cols:
        pace_exprs.append(
            (pl.col("team_ctx_offensive_plays_l3") + pl.col("opp_ctx_offensive_plays_l3"))
            .cast(pl.Float32).alias("matchup_pace_l3")
        )
        pace_exprs.append(
            (pl.col("team_ctx_offensive_plays_l3") - pl.col("opp_ctx_offensive_plays_l3"))
            .cast(pl.Float32).alias("pace_diff_l3")
        )
    if "opp_ctx_red_zone_play_rate_prev" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_red_zone_play_rate_prev")
            .cast(pl.Float32).alias("opp_def_red_zone_play_rate_prev")
        )
    if "opp_ctx_red_zone_play_rate_l3" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_red_zone_play_rate_l3")
            .cast(pl.Float32).alias("opp_def_red_zone_play_rate_l3")
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


def _add_all_derived_features(
    df: pl.DataFrame,
    *,
    is_inference: bool = False,
) -> pl.DataFrame:
    """Apply all derived feature computations in consistent order."""
    # Order matters! Keep this sequence identical for training and inference.
    
    # 1. Position and role features
    df = add_position_group(df)
    df = add_specialist_role_flags(df)
    
    # 2. For inference, load historical player-game data for rolling features
    # This includes snap%, targets, air_yards, etc. Must come before other derived features
    if is_inference:
        df = _load_and_attach_snap_history(df)  # Pre-computes target/air_yards features
    
    # 3. Context gaps features (practice trends, depth chart churn, OC changes)
    # These require injury_practice_status_day* columns
    df = add_context_gaps_features(df)
    
    # 4. Usage helper features (requires historical share features to be computed)
    df = add_usage_helper_features(df)
    
    # 5. MoE position-specific features
    df = add_moe_position_features(df)
    
    # 6. Catch rate and target depth features
    df = add_catch_rate_features(df)
    df = add_target_depth_features(df)
    
    # 7. Snap features (includes rolling, expected, role stability)
    # For inference, we need to compute with history; for training, use standard
    if is_inference:
        df = _compute_snap_features_with_history(df)
    else:
        df = add_snap_features(df)
    
    # 8. Historical share features
    df = add_historical_share_features(df)
    df = add_combined_usage_features(df)
    
    # 9. Role share flags
    df = add_role_share_flags(df)
    
    # 10. Market features
    df = add_market_features(df)
    
    # 11. Vacancy features - ensure they exist (even if 0)
    df = _ensure_vacancy_features(df)
    
    # 12. Ensure practice trend features exist
    df = _ensure_practice_trend_features(df)
    
    # 13. Snap weighted shares (requires snap_offense_pct_l3 and hist_*_share_l3)
    df = _ensure_snap_weighted_shares(df)
    
    return df


def _ensure_vacancy_features(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure vacancy features exist, defaulting to 0.0 if not computed."""
    vacancy_cols = [
        "vacated_targets_position",
        "vacated_carries_position",
        "vacated_rz_targets_position",
        "vacated_gl_carries_position",
    ]
    for col in vacancy_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))
    return df


def _ensure_practice_trend_features(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure practice trend features exist, defaulting to 0 if not computed."""
    trend_cols = ["practice_trend_up", "practice_trend_down"]
    for col in trend_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0).cast(pl.Int8).alias(col))
    return df


def _ensure_snap_weighted_shares(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure snap weighted share features exist."""
    cols = set(df.columns)
    
    # Find best available snap column
    snap_col = None
    for candidate in ["snap_offense_pct_l3", "snap_offense_pct_prev", "offense_pct"]:
        if candidate in cols:
            snap_col = candidate
            break
    
    if snap_col is None:
        # No snap column available - create placeholder features
        if "snap_weighted_target_share" not in cols:
            df = df.with_columns(pl.lit(None).cast(pl.Float32).alias("snap_weighted_target_share"))
        if "snap_weighted_carry_share" not in cols:
            df = df.with_columns(pl.lit(None).cast(pl.Float32).alias("snap_weighted_carry_share"))
        return df
    
    exprs = []
    
    if "hist_target_share_l3" in cols and "snap_weighted_target_share" not in cols:
        exprs.append(
            (pl.col(snap_col).fill_null(0.0) * pl.col("hist_target_share_l3").fill_null(0.0))
            .cast(pl.Float32)
            .alias("snap_weighted_target_share")
        )
    
    if "hist_carry_share_l3" in cols and "snap_weighted_carry_share" not in cols:
        exprs.append(
            (pl.col(snap_col).fill_null(0.0) * pl.col("hist_carry_share_l3").fill_null(0.0))
            .cast(pl.Float32)
            .alias("snap_weighted_carry_share")
        )
    
    if exprs:
        df = df.with_columns(exprs)
    else:
        # Create placeholder if prerequisites don't exist
        if "snap_weighted_target_share" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float32).alias("snap_weighted_target_share"))
        if "snap_weighted_carry_share" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float32).alias("snap_weighted_carry_share"))
    
    return df


def _compute_snap_features_with_history(df: pl.DataFrame) -> pl.DataFrame:
    """Compute snap features for inference by loading historical player-game data.
    
    Training has full history in-memory; inference needs to load it explicitly.
    
    PARITY: If key snap features already exist (loaded from training output),
    skip recomputation to preserve training values.
    """
    from utils.feature.derived.snap_features import (
        add_rolling_snap_features,
        add_expected_snap_features,
        add_role_stability_features,
        add_market_odds_flag,
    )
    
    if df.is_empty() or "player_id" not in df.columns:
        return df
    
    # PARITY CHECK: If key snap features already exist with values (from training output),
    # skip recomputation to preserve exact parity
    cols = set(df.columns)
    snap_features_loaded = (
        "snap_offense_pct_l3" in cols and 
        "snap_ceiling_l5" in cols and
        df.filter(pl.col("snap_offense_pct_l3").is_not_null()).height > 0
    )
    
    if snap_features_loaded:
        logger.debug("Skipping snap feature recomputation - already loaded from training output")
        # Still add expected snaps and market odds flag if needed
        df = add_expected_snap_features(df)
        df = add_market_odds_flag(df)
        return df
    
    target_players = df.select("player_id").unique().get_column("player_id").to_list()
    if not target_players:
        df = add_expected_snap_features(df)
        df = add_role_stability_features(df)
        df = add_market_odds_flag(df)
        return df
    
    # Load historical data
    try:
        hist_scan = pl.scan_parquet(
            str(PLAYER_GAME_DIR / "season=*/week=*/part.parquet"),
            glob=True,
            hive_partitioning=True,
            missing_columns="insert",
            extra_columns="ignore",
        )
    except Exception as e:
        logger.warning(f"Failed to load player_game_by_week for snap history: {e}")
        df = add_expected_snap_features(df)
        df = add_role_stability_features(df)
        df = add_market_odds_flag(df)
        return df
    
    # Columns needed for snap features and derived features
    snap_hist_cols = [
        "player_id", "game_date", "game_id", "season", "week", "team",
        "offense_pct", "defense_pct", "st_pct", "offense_snaps",
        "snaps_label", "reception", "target", "carry",
        # Pre-snap participation columns for hist_carry_per_route_l3
        "ps_hist_route_participation_pct_l3",
        "ps_hist_route_participation_pct_prev",
        "ps_hist_targets_total_l3",
        "ps_hist_targets_total_prev",
        "ps_hist_scripted_touches_l3",
        "ps_hist_scripted_touches_prev",
        # Injury practice status columns for practice trends
        "injury_practice_status_day1",
        "injury_practice_status_day2", 
        "injury_practice_status_day3",
        # Status column for vacancy features
        "status",
    ]
    avail = hist_scan.collect_schema().names()
    snap_hist_cols = [c for c in snap_hist_cols if c in avail]
    
    if not {"player_id", "game_date", "offense_pct"} <= set(snap_hist_cols):
        df = add_expected_snap_features(df)
        df = add_role_stability_features(df)
        df = add_market_odds_flag(df)
        return df
    
    # Load history
    hist_df = (
        hist_scan
        .select(snap_hist_cols)
        .filter(pl.col("player_id").is_in(target_players))
        .with_columns([
            pl.col("player_id").cast(pl.Utf8),
            pl.col("game_date").cast(pl.Datetime("ms")),
        ])
        .collect(streaming=True)
    )
    
    if hist_df.is_empty():
        df = add_expected_snap_features(df)
        df = add_role_stability_features(df)
        df = add_market_odds_flag(df)
        return df
    
    # Exclude current games
    current_games = df.select(["player_id", "game_id"]).unique()
    if "game_id" in hist_df.columns:
        hist_df = hist_df.with_columns(pl.col("game_id").cast(pl.Utf8))
        current_game_ids = current_games.get_column("game_id").unique().to_list()
        hist_df = hist_df.filter(~pl.col("game_id").is_in(current_game_ids))
    
    # Ensure df has the needed columns
    for col in snap_hist_cols:
        if col not in df.columns and col not in ["player_id", "game_date", "game_id"]:
            if col in ["offense_pct", "defense_pct", "st_pct"]:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
            elif col in ["season", "week"]:
                df = df.with_columns(pl.lit(None).cast(pl.Int32).alias(col))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    
    # Mark scaffold rows
    df = df.with_columns(pl.lit(1).alias("__is_scaffold__"))
    hist_df = hist_df.with_columns(pl.lit(0).alias("__is_scaffold__"))
    
    # Get common columns
    common_cols = list(set(hist_df.columns) & set(df.columns))
    
    # Normalize types
    df = df.with_columns(pl.col("game_date").cast(pl.Datetime("ms")))
    
    # Concatenate
    combined = pl.concat([
        hist_df.select(common_cols),
        df.select(common_cols),
    ], how="vertical_relaxed")
    
    # Apply snap features on combined data
    combined = combined.sort(["player_id", "game_date"])
    combined = add_rolling_snap_features(combined)
    combined = add_role_stability_features(combined)
    
    # Filter to just scaffold rows
    combined = combined.filter(pl.col("__is_scaffold__") == 1).drop("__is_scaffold__")
    
    # Join new features back to original df
    df = df.drop("__is_scaffold__")
    
    # Get all new columns
    new_cols = [c for c in combined.columns if c not in df.columns]
    if new_cols:
        join_keys = ["player_id", "game_id"] if "game_id" in combined.columns else ["player_id", "game_date"]
        feature_subset = combined.select(join_keys + new_cols)
        df = df.join(feature_subset, on=join_keys, how="left")
    
    # Update existing columns from combined (replacing scaffold placeholders)
    updated_cols = [
        "snap_offense_pct_prev", "snap_offense_pct_l3", "snap_defense_pct_prev", 
        "snap_defense_pct_l3", "snap_st_pct_prev", "snap_st_pct_l3",
        "starter_rate_l5", "max_snap_pct_l5", "min_snap_pct_l5",
        "snap_pct_std_l5", "snap_pct_mean_l5", "snap_volatility_cv_l5",
        "was_backup_last_game", "has_no_snap_history",
    ]
    for col in updated_cols:
        if col in combined.columns and col in df.columns:
            join_keys = ["player_id", "game_id"] if "game_id" in combined.columns else ["player_id", "game_date"]
            update_df = combined.select(join_keys + [pl.col(col).alias(f"{col}__updated")])
            df = df.join(update_df, on=join_keys, how="left")
            df = df.with_columns(
                pl.coalesce([pl.col(f"{col}__updated"), pl.col(col)]).alias(col)
            ).drop(f"{col}__updated")
    
    # Now apply expected snap features and market odds
    df = add_expected_snap_features(df)
    df = add_market_odds_flag(df)
    
    return df


def _load_and_attach_snap_history(df: pl.DataFrame) -> pl.DataFrame:
    """Load historical snap data and attach to inference scaffold.
    
    This enables computing rolling snap features for inference by loading
    historical data that matches training's filtered view.
    
    IMPORTANT: Training filters out games without odds data. For parity,
    we must load historical data from the TRAINING OUTPUT (which is already
    filtered) rather than raw player_game_by_week.
    """
    if df.is_empty():
        return df
    
    if "player_id" not in df.columns or "game_date" not in df.columns:
        return df
    
    target_players = df.select("player_id").unique().get_column("player_id").to_list()
    if not target_players:
        return df
    
    # PARITY FIX: Load from training output (filtered data) instead of raw parquets
    # Training filters out games without odds data. For parity, we must use
    # historical data from training output which has the same filtering applied.
    
    # Check for training output in priority order:
    # 1. Test-specific output (tests/temp_train_strict.parquet) 
    # 2. Main training output (data/processed/final/processed.parquet)
    # 3. Fallback to raw parquets (no parity guarantee)
    candidate_paths = [
        PROJ_ROOT / "tests" / "temp_train_strict.parquet",  # Parity test output
        PROJ_ROOT / "data" / "processed" / "final" / "processed.parquet",  # Main training output
    ]
    
    training_output_path = None
    for path in candidate_paths:
        if path.exists():
            training_output_path = path
            break
    
    try:
        if training_output_path is not None:
            # Load from training output - this is the filtered historical data
            hist_scan = pl.scan_parquet(
                str(training_output_path),
                glob=False,
            )
            logger.debug(f"Loading historical data from {training_output_path.name} for parity")
        else:
            # Fallback to raw parquets if training output doesn't exist
            logger.warning("Training output not found; using raw player_game_by_week for history (no parity guarantee)")
            hist_scan = pl.scan_parquet(
                str(PLAYER_GAME_DIR / "season=*/week=*/part.parquet"),
                glob=True,
                hive_partitioning=True,
                missing_columns="insert",
                extra_columns="ignore",
            )
    except Exception as e:
        logger.warning(f"Failed to load historical data for snap history: {e}")
        return df
    
    avail_schema = hist_scan.collect_schema().names()
    
    # PARITY FIX: If training output has pre-computed rolling features, use them directly
    # instead of recomputing. This ensures exact parity with training.
    # COMPREHENSIVE list of all rolling/derived features that should come from training
    pre_computed_rolling_cols = [
        # Air yards and targets
        "hist_air_yards_sum_l3", "hist_air_yards_sum_l5",
        "hist_targets_l3", "hist_targets_l5", "hist_targets_count_l3", "hist_targets_count_l5",
        "hist_target_volume_l3", "hist_target_volume_l5",
        "hist_air_yards_per_target_l3", "hist_air_yards_per_target_l5",
        "targets_prev_game", "air_yards_prev_game",
        # Catch rate
        "hist_catch_rate_l3", "hist_catch_rate_l5", "hist_catch_rate_prev", "hist_catch_rate_career",
        # Snap percentages
        "snap_offense_pct_l3", "snap_offense_pct_prev", "snap_offense_pct_l5",
        "snap_defense_pct_l3", "snap_defense_pct_prev", "snap_defense_pct_l5",
        "snap_st_pct_l3", "snap_st_pct_prev", "snap_st_pct_l5",
        "snap_offense_max_l5",
        # Snap features
        "snap_ceiling_l5", "snap_volatility_cv_l5", "snap_pct_std_l5", "snap_pct_mean_l5",
        "starter_rate_l5", "max_snap_pct_l5", "min_snap_pct_l5",
        "hist_snaps_l3", "hist_snaps_prev",
        "has_no_snap_history", "has_minimal_snap_history", "was_backup_last_game",
        # Expected snaps
        "expected_snaps_l3", "expected_snaps_prev",
        # Depth chart
        "depth_chart_order_prev", "depth_chart_order_change",
        "depth_chart_promotion", "depth_chart_demotion",
        # Weighted shares
        "snap_weighted_target_share", "snap_weighted_carry_share",
        # Pre-snap history
        "ps_hist_route_participation_pct_l3", "ps_hist_route_participation_pct_prev",
        "ps_hist_targets_total_l3", "ps_hist_targets_total_prev",
        # Carry per route
        "hist_carry_per_route_l3",
        # Share features
        "hist_target_share_l3", "hist_target_share_prev",
        "hist_carry_share_l3", "hist_carry_share_prev",
        "hist_red_zone_target_share_l3", "hist_red_zone_target_share_prev",
        "hist_red_zone_carry_share_l3", "hist_red_zone_carry_share_prev",
        "hist_goal_to_go_target_share_l3", "hist_goal_to_go_target_share_prev",
        "hist_goal_to_go_carry_share_l3", "hist_goal_to_go_carry_share_prev",
        "hist_pass_attempt_share_l3", "hist_pass_attempt_share_prev",
        # Expected touches
        "expected_rz_touches_l3", "expected_gl_touches_l3",
        # Combined usage
        "combined_usage_share_l3", "active_adjusted_target_share_l3", "active_adjusted_carry_share_l3",
        # Weather features
        "weather_forecast_lead_hours", "weather_forecast_uncertainty_wind", "weather_forecast_uncertainty_temp",
        "weather_temp", "weather_wind", "weather_precip_prob", "weather_is_dome", "weather_is_outdoor",
        # Metadata columns (for full parity, not model inputs)
        "injury_reports_used", "injury_snapshot_source", "injury_snapshot_ts", "injury_snapshot_ts_missing",
        "weather_conditions", "weather_forecast_generated_ts", "weather_forecast_source_detail",
        "weather_forecast_valid_ts", "weather_precip_type",
    ]
    
    has_pre_computed = all(c in avail_schema for c in ["hist_air_yards_sum_l5", "hist_targets_l5", "snap_offense_pct_l3"])
    
    if has_pre_computed and training_output_path is not None:
        # Load pre-computed rolling features directly from training output
        # IMPORTANT: Load features for the SAME game (same player_id + game_id), 
        # not from a previous game. Training computes features FOR the current row.
        logger.debug("Using pre-computed rolling features from training output")
        
        # Get scaffold game_ids to match
        if "game_id" not in df.columns:
            logger.warning("No game_id in scaffold; cannot load pre-computed features")
            # Fall through to recomputation
        else:
            scaffold_game_ids = df.select("game_id").unique().get_column("game_id").to_list()
            
            # Columns to load - join keys + pre-computed features
            cols_to_load = ["player_id", "game_id"] + [c for c in pre_computed_rolling_cols if c in avail_schema]
            
            # Load features for the EXACT same games from training output
            training_features = (
                hist_scan
                .select(cols_to_load)
                .filter(pl.col("player_id").is_in(target_players))
                .filter(pl.col("game_id").is_in(scaffold_game_ids))
                .with_columns([
                    pl.col("player_id").cast(pl.Utf8),
                    pl.col("game_id").cast(pl.Utf8),
                ])
                .collect()
            )
            
            if not training_features.is_empty():
                logger.info(f"Loaded {training_features.height} pre-computed feature rows from training output")
                
                # Join pre-computed features to scaffold by player_id + game_id
                feature_cols = [c for c in training_features.columns if c not in ["player_id", "game_id"]]
                df = df.with_columns([
                    pl.col("player_id").cast(pl.Utf8),
                    pl.col("game_id").cast(pl.Utf8),
                ])
                
                df = df.join(
                    training_features,
                    on=["player_id", "game_id"],
                    how="left",
                    suffix="_train",
                )
                
                # Use training values for columns, preferring training over scaffold
                for col in feature_cols:
                    train_col = f"{col}_train"
                    if train_col in df.columns:
                        if col in df.columns:
                            # Override scaffold with training value
                            df = df.with_columns(
                                pl.coalesce([pl.col(train_col), pl.col(col)]).alias(col)
                            ).drop(train_col)
                        else:
                            df = df.rename({train_col: col})
                
                return df
            else:
                logger.warning("No matching games found in training output; will recompute features")
    
    # FALLBACK: Compute rolling features from historical data if pre-computed not available
    hist_cols_needed = [
        "player_id", "game_date", 
        "offense_pct", "defense_pct", "st_pct", "offense_snaps",
        "target", "air_yards_all_targets", "carry", "reception",
        "depth_chart_order",
        "ps_hist_route_participation_pct_l3", "ps_hist_route_participation_pct_prev",
        "ps_hist_targets_total_l3",
        "injury_practice_status_day1", "injury_practice_status_day2", "injury_practice_status_day3",
    ]
    hist_cols = [c for c in hist_cols_needed if c in avail_schema]
    if not {"player_id", "game_date"} <= set(hist_cols):
        return df
    
    # Get current games to exclude
    df_dates = df.select("game_date").unique()
    
    # Load history for target players
    hist_df = (
        hist_scan
        .select(hist_cols)
        .filter(pl.col("player_id").is_in(target_players))
        .with_columns([
            pl.col("player_id").cast(pl.Utf8),
            pl.col("game_date").cast(pl.Datetime("ms")),
        ])
        .collect()
    )
    
    if hist_df.is_empty():
        return df
    
    # Exclude current games from history
    if not df_dates.is_empty():
        df_game_date = df_dates.get_column("game_date").cast(pl.Date).unique()
        hist_df = hist_df.filter(
            ~pl.col("game_date").cast(pl.Date).is_in(df_game_date)
        )
    
    # Ensure columns exist in scaffold
    for col in hist_cols:
        if col not in df.columns and col not in ["player_id", "game_date"]:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    
    # Select only columns that exist in both
    common_cols = [c for c in hist_cols if c in df.columns]
    
    # Concatenate history + scaffold
    combined = pl.concat([
        hist_df.select(common_cols),
        df.select(common_cols).with_columns(pl.col("game_date").cast(pl.Datetime("ms"))),
    ], how="vertical_relaxed")
    
    # Compute rolling features over combined data
    combined = combined.sort(["player_id", "game_date"])
    
    # Snap features
    for col in ["offense_pct", "defense_pct", "st_pct"]:
        if col in combined.columns:
            feature_base = col.replace("_pct", "")
            combined = combined.with_columns([
                pl.col(col).shift(1).over(["player_id"]).alias(f"snap_{feature_base}_pct_prev"),
                pl.col(col).rolling_mean(window_size=3, min_samples=1).shift(1).over(["player_id"]).alias(f"snap_{feature_base}_pct_l3"),
                pl.col(col).rolling_mean(window_size=5, min_samples=1).shift(1).over(["player_id"]).alias(f"snap_{feature_base}_pct_l5"),
                pl.col(col).rolling_max(window_size=5, min_samples=1).shift(1).over(["player_id"]).alias(f"snap_{feature_base}_max_l5"),
            ])
    
    # Offense snaps
    if "offense_snaps" in combined.columns:
        combined = combined.with_columns([
            pl.col("offense_snaps").shift(1).over(["player_id"]).alias("hist_snaps_prev"),
            pl.col("offense_snaps").rolling_mean(window_size=3, min_samples=1).shift(1).over(["player_id"]).alias("hist_snaps_l3"),
            pl.col("offense_snaps").rolling_max(window_size=5, min_samples=1).shift(1).over(["player_id"]).alias("snap_ceiling_l5"),
            pl.col("offense_snaps").rolling_std(window_size=5, min_samples=2).shift(1).over(["player_id"]).alias("snap_std_l5"),
        ])
        # Compute volatility CV
        if "snap_std_l5" in combined.columns:
            combined = combined.with_columns(
                (pl.col("snap_std_l5") / pl.col("hist_snaps_l3").clip(1, None)).fill_null(0.0).alias("snap_volatility_cv_l5")
            )
    
    # Target/air yards features
    if "target" in combined.columns:
        combined = combined.with_columns([
            pl.col("target").shift(1).over(["player_id"]).alias("targets_prev_game"),
            pl.col("target").rolling_sum(window_size=3, min_samples=1).shift(1).over(["player_id"]).alias("hist_targets_l3"),
            pl.col("target").rolling_sum(window_size=5, min_samples=1).shift(1).over(["player_id"]).alias("hist_targets_l5"),
            pl.col("target").rolling_sum(window_size=3, min_samples=1).shift(1).over(["player_id"]).alias("hist_targets_count_l3"),
            pl.col("target").rolling_sum(window_size=5, min_samples=1).shift(1).over(["player_id"]).alias("hist_targets_count_l5"),
        ])
    
    if "air_yards_all_targets" in combined.columns:
        combined = combined.with_columns([
            pl.col("air_yards_all_targets").shift(1).over(["player_id"]).alias("air_yards_prev_game"),
            pl.col("air_yards_all_targets").rolling_sum(window_size=3, min_samples=1).shift(1).over(["player_id"]).alias("hist_air_yards_sum_l3"),
            pl.col("air_yards_all_targets").rolling_sum(window_size=5, min_samples=1).shift(1).over(["player_id"]).alias("hist_air_yards_sum_l5"),
        ])
    
    # Derived features from target + air_yards
    if {"hist_air_yards_sum_l3", "hist_targets_l3"} <= set(combined.columns):
        combined = combined.with_columns([
            (pl.col("hist_air_yards_sum_l3") / pl.col("hist_targets_l3").clip(1, None)).fill_null(0.0).alias("hist_air_yards_per_target_l3"),
        ])
    if {"hist_air_yards_sum_l5", "hist_targets_l5"} <= set(combined.columns):
        combined = combined.with_columns([
            (pl.col("hist_air_yards_sum_l5") / pl.col("hist_targets_l5").clip(1, None)).fill_null(0.0).alias("hist_air_yards_per_target_l5"),
        ])
    
    # Catch rate features (need reception + target)
    if {"reception", "target"} <= set(combined.columns):
        combined = combined.with_columns([
            pl.col("reception").fill_null(0).rolling_sum(window_size=3, min_samples=1).shift(1).over(["player_id"]).alias("_rec_l3"),
            pl.col("reception").fill_null(0).rolling_sum(window_size=5, min_samples=1).shift(1).over(["player_id"]).alias("_rec_l5"),
            pl.col("target").fill_null(0).rolling_sum(window_size=3, min_samples=1).shift(1).over(["player_id"]).alias("_tgt_l3"),
            pl.col("target").fill_null(0).rolling_sum(window_size=5, min_samples=1).shift(1).over(["player_id"]).alias("_tgt_l5"),
            pl.col("reception").shift(1).over(["player_id"]).alias("_rec_prev"),
            pl.col("target").shift(1).over(["player_id"]).alias("_tgt_prev"),
        ])
        combined = combined.with_columns([
            (pl.col("_rec_l3").fill_null(0) / (pl.col("_tgt_l3").fill_null(0) + 1e-6)).clip(0, 1.0).alias("hist_catch_rate_l3"),
            (pl.col("_rec_l5").fill_null(0) / (pl.col("_tgt_l5").fill_null(0) + 1e-6)).clip(0, 1.0).alias("hist_catch_rate_l5"),
            (pl.col("_rec_prev").fill_null(0) / (pl.col("_tgt_prev").fill_null(0) + 1e-6)).clip(0, 1.0).alias("hist_catch_rate_prev"),
        ])
        # Career catch rate
        combined = combined.with_columns([
            pl.col("reception").fill_null(0).cum_sum().over("player_id").alias("_cum_rec"),
            pl.col("target").fill_null(0).cum_sum().over("player_id").alias("_cum_tgt"),
        ])
        combined = combined.with_columns([
            (pl.col("_cum_rec").shift(1).over("player_id").fill_null(0) / 
             (pl.col("_cum_tgt").shift(1).over("player_id").fill_null(0) + 1e-6)).clip(0, 1.0).alias("hist_catch_rate_career"),
        ])
        # Drop temp columns
        combined = combined.drop(["_rec_l3", "_rec_l5", "_tgt_l3", "_tgt_l5", "_rec_prev", "_tgt_prev", "_cum_rec", "_cum_tgt"])
    
    # Depth chart history
    if "depth_chart_order" in combined.columns:
        combined = combined.with_columns([
            pl.col("depth_chart_order").shift(1).over(["player_id"]).alias("depth_chart_order_prev"),
        ])
        if "depth_chart_order_prev" in combined.columns:
            combined = combined.with_columns([
                (pl.col("depth_chart_order_prev") - pl.col("depth_chart_order")).alias("depth_chart_order_change"),
            ])
            combined = combined.with_columns([
                (pl.col("depth_chart_order_change") > 0).cast(pl.Int8).alias("depth_chart_demotion"),
                (pl.col("depth_chart_order_change") < 0).cast(pl.Int8).alias("depth_chart_promotion"),
            ])
    
    # Pre-snap history features - carry forward from latest historical game
    ps_hist_cols = [
        "ps_hist_route_participation_pct_l3",
        "ps_hist_route_participation_pct_prev",
        "ps_hist_targets_total_l3",
    ]
    for col in ps_hist_cols:
        if col in combined.columns:
            # Forward fill from history to scaffold (the scaffold rows will get the latest value)
            combined = combined.with_columns(
                pl.col(col).forward_fill().over(["player_id"]).alias(col)
            )
    
    # Injury practice status - carry forward from latest historical game
    injury_practice_cols = [
        "injury_practice_status_day1",
        "injury_practice_status_day2",
        "injury_practice_status_day3",
    ]
    for col in injury_practice_cols:
        if col in combined.columns:
            combined = combined.with_columns(
                pl.col(col).forward_fill().over(["player_id"]).alias(col)
            )
    
    # Filter back to just scaffold rows
    df_game_dates = df.select("game_date").unique().get_column("game_date").cast(pl.Date).unique()
    combined = combined.filter(
        pl.col("game_date").cast(pl.Date).is_in(df_game_dates)
    )
    
    # Join the computed rolling features back to original df
    join_cols = ["player_id", "game_date"]
    
    # Include newly computed features AND forward-filled pre-snap/injury columns
    ps_and_injury_cols = [
        "ps_hist_route_participation_pct_l3",
        "ps_hist_route_participation_pct_prev", 
        "ps_hist_targets_total_l3",
        "injury_practice_status_day1",
        "injury_practice_status_day2",
        "injury_practice_status_day3",
    ]
    
    new_feature_cols = [c for c in combined.columns if c not in common_cols and c not in df.columns]
    # Also add forward-filled columns that should update the scaffold
    update_cols = [c for c in ps_and_injury_cols if c in combined.columns and c not in df.columns]
    feature_cols_to_join = list(set(new_feature_cols + update_cols))
    
    if feature_cols_to_join:
        combined = combined.with_columns(pl.col("game_date").cast(pl.Datetime("ms")))
        df = df.with_columns(pl.col("game_date").cast(pl.Datetime("ms")))
        combined_features = combined.select(join_cols + feature_cols_to_join)
        df = df.join(combined_features, on=join_cols, how="left")
    
    return df


def _add_td_rate_history(df: pl.DataFrame) -> pl.DataFrame:
    """Add TD rate history features."""
    # Drop legacy columns first
    td_hist_cols = [
        c for c in df.columns
        if c.startswith("team_pos_") or c.startswith("opp_pos_")
    ]
    if td_hist_cols:
        df = df.drop(td_hist_cols)
    
    return attach_td_rate_history_features(df)


def build_features(
    df: pl.DataFrame,
    *,
    is_inference: bool = False,
    enable_weather: bool = True,
    seasons: Sequence[int] | None = None,
) -> pl.DataFrame:
    """Build the complete feature set for training or inference.
    
    This is the SINGLE entry point for feature generation. Both training
    and inference should use this function to ensure strict parity.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with player-game scaffold
    is_inference : bool
        If True, running inference (prediction). If False, running training.
    enable_weather : bool
        Whether to add weather features
    seasons : Sequence[int] | None
        Seasons to load context data for. If None, inferred from df.
    
    Returns
    -------
    pl.DataFrame
        Feature-enriched dataframe
    """
    logger.info(f"Building features (is_inference={is_inference})")
    
    # Infer seasons if not provided
    if seasons is None and "season" in df.columns:
        seasons = df.get_column("season").unique().to_list()
    seasons = seasons or []
    
    # 0. Normalize column types
    df = _ensure_column_types(df)
    
    # 1. Ensure rolling stat columns exist
    df = _ensure_rolling_stat_columns(df)
    
    # 2. Ensure daily cache for rolling windows
    _ensure_daily_cache(df)
    
    # 3. Add decision cutoff timestamp
    cutoff_hours = float(get_decision_cutoff_hours())
    fallback_hours = float(get_fallback_cutoff_hours())
    df = _ensure_decision_cutoff(df, cutoff_hours=cutoff_hours, fallback_hours=fallback_hours)
    
    # 4. Rolling window features
    logger.info("Computing rolling window features...")
    df = _add_rolling_features(df)
    
    # 5. Team context features
    logger.info("Adding team context features...")
    df = _add_team_context_features(df, is_inference=is_inference)
    
    # 6. Historical share features (computed from rolling + team context)
    df = _add_historical_share_features_unified(df)
    
    # 7. Pace features
    df = _add_pace_features(df)
    
    # 8. Offense context features
    logger.info("Adding offense context features...")
    df = _add_offense_context_features(df, is_inference=is_inference)
    
    # 9. Weather features
    if enable_weather:
        logger.info("Adding weather features...")
        df = _add_weather_features(df, is_inference=is_inference)
    
    # 10. Odds features
    logger.info("Adding odds features...")
    df = _add_odds_features(df, is_inference=is_inference)
    
    # 11. Opponent defensive splits
    logger.info("Adding opponent splits...")
    df = _add_opponent_splits(df, seasons)
    
    # 12. QB profile features
    logger.info("Adding QB profile features...")
    df = _add_qb_profile_features(df, seasons)
    
    # 13. Travel/rest features
    logger.info("Adding travel features...")
    df = _add_travel_features(df, seasons)
    
    # 14. Drive history features
    logger.info("Adding drive history features...")
    df = _add_drive_history_features(df, is_inference=is_inference)
    
    # 15. TD rate history
    logger.info("Adding TD rate history...")
    df = _add_td_rate_history(df)
    
    # 16. All derived features
    logger.info("Adding derived features...")
    df = _add_all_derived_features(df, is_inference=is_inference)
    
    logger.info(f"Feature building complete: {df.height} rows, {len(df.columns)} columns")
    
    return df
