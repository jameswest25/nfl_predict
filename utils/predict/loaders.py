"""Data loaders for prediction context features.

This module provides partition-based loading functions for various
feature caches used in inference.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import polars as pl

from utils.general.paths import (
    PROJ_ROOT,
    QB_PROFILE_DIR,
    TRAVEL_CALENDAR_DIR,
    PLAYER_GAME_BY_WEEK_DIR as PLAYER_GAME_DIR,
)

logger = logging.getLogger(__name__)


def _collect_partition_paths(base_dir: Path, seasons: Iterable[int]) -> list[Path]:
    """Collect partition file paths for given seasons."""
    paths: list[Path] = []
    for season in sorted({int(s) for s in seasons}):
        season_dir = base_dir / f"season={season}"
        if not season_dir.exists():
            continue
        paths.extend(sorted(season_dir.glob("week=*/part.parquet")))
    return paths


def load_qb_profile_features(seasons: Iterable[int]) -> pl.DataFrame:
    """Load QB profile features from partition cache.
    
    Parameters
    ----------
    seasons : Iterable[int]
        Seasons to load
    
    Returns
    -------
    pl.DataFrame
        QB profile features indexed by (season, week, team)
    """
    paths = _collect_partition_paths(QB_PROFILE_DIR, seasons)
    if not paths:
        return pl.DataFrame()
    
    scan = pl.scan_parquet(
        paths,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    df = scan.collect(streaming=True)
    
    if df.is_empty():
        return df
    
    # Normalize column names
    if "qb_id" in df.columns:
        df = df.rename({"qb_id": "qb_profile_id"})
    
    # Keep one QB per team (highest dropbacks)
    if "qb_profile_dropbacks_prev" in df.columns:
        df = df.with_columns(
            pl.col("qb_profile_dropbacks_prev")
            .fill_null(-1.0)
            .alias("__qb_profile_dropbacks_rank")
        ).sort(
            ["season", "week", "team", "__qb_profile_dropbacks_rank"],
            descending=[False, False, False, True],
        )
        df = df.unique(subset=["season", "week", "team"], keep="first").drop("__qb_profile_dropbacks_rank")
    else:
        df = df.unique(subset=["season", "week", "team"], keep="first")
    
    # Type normalization
    df = df.with_columns([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("team").cast(pl.Utf8),
    ])
    
    float_cols = [
        col for col, dtype in df.schema.items()
        if col.startswith("qb_profile_") and dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    if float_cols:
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in float_cols])
    
    for ts_col in ("qb_profile_data_as_of", "qb_profile_team_data_as_of"):
        if ts_col in df.columns:
            df = df.with_columns(pl.col(ts_col).cast(pl.Datetime("ms")))
    
    # Drop timestamp columns that shouldn't be features
    drop_cols = [
        col for col in ("qb_profile_data_as_of", "qb_profile_team_data_as_of", "game_local_start")
        if col in df.columns
    ]
    if drop_cols:
        df = df.drop(drop_cols)
    
    return df


def load_travel_calendar_features(seasons: Iterable[int]) -> pl.DataFrame:
    """Load travel/rest features from partition cache.
    
    Parameters
    ----------
    seasons : Iterable[int]
        Seasons to load
    
    Returns
    -------
    pl.DataFrame
        Travel features indexed by (season, week, team)
    """
    paths = _collect_partition_paths(TRAVEL_CALENDAR_DIR, seasons)
    if not paths:
        return pl.DataFrame()
    
    scan = pl.scan_parquet(
        paths,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    df = scan.collect(streaming=True)
    
    if df.is_empty():
        return df
    
    keep_cols = [
        "season", "week", "team",
        "rest_days", "rest_hours", "rest_days_rolling3",
        "travel_km", "travel_miles", "travel_km_rolling3",
        "timezone_change_hours", "time_diff_from_home_hours",
        "game_timezone_offset", "team_timezone_offset", "local_start_hour",
        "consecutive_road_games", "consecutive_home_games",
        "is_short_week", "is_long_rest", "bye_week_flag",
        "west_to_east_early", "east_to_west_late",
    ]
    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df.select(existing_cols)
    
    # Rename to travel_ prefix
    rename_map = {
        "rest_days": "travel_rest_days",
        "rest_hours": "travel_rest_hours",
        "rest_days_rolling3": "travel_rest_days_l3",
        "travel_km": "travel_distance_km",
        "travel_miles": "travel_distance_miles",
        "travel_km_rolling3": "travel_distance_km_l3",
        "timezone_change_hours": "travel_timezone_change_hours",
        "time_diff_from_home_hours": "travel_time_diff_from_home_hours",
        "game_timezone_offset": "travel_game_timezone_offset",
        "team_timezone_offset": "travel_team_timezone_offset",
        "local_start_hour": "travel_local_start_hour",
        "consecutive_road_games": "travel_consecutive_road_games",
        "consecutive_home_games": "travel_consecutive_home_games",
        "is_short_week": "travel_short_week_flag",
        "is_long_rest": "travel_long_rest_flag",
        "bye_week_flag": "travel_bye_week_flag",
        "west_to_east_early": "travel_west_to_east_early_flag",
        "east_to_west_late": "travel_east_to_west_late_flag",
    }
    df = df.rename(rename_map)
    
    # Type normalization
    df = df.with_columns([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("team").cast(pl.Utf8),
    ])
    
    float_cols = [
        col for col, dtype in df.schema.items()
        if col.startswith("travel_") and col not in {
            "travel_short_week_flag", "travel_long_rest_flag",
            "travel_bye_week_flag", "travel_west_to_east_early_flag",
            "travel_east_to_west_late_flag",
        } and dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    if float_cols:
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in float_cols])
    
    flag_cols = [
        col for col in [
            "travel_short_week_flag", "travel_long_rest_flag",
            "travel_bye_week_flag", "travel_west_to_east_early_flag",
            "travel_east_to_west_late_flag",
        ] if col in df.columns
    ]
    if flag_cols:
        df = df.with_columns([pl.col(col).cast(pl.Int8) for col in flag_cols])
    
    return df


def load_injury_history_features(seasons: Iterable[int], weeks: Iterable[int]) -> pl.DataFrame:
    """Load injury history features from player-game cache.
    
    Parameters
    ----------
    seasons : Iterable[int]
        Seasons to load
    weeks : Iterable[int]
        Weeks to filter
    
    Returns
    -------
    pl.DataFrame
        Injury history features indexed by (season, week, player_id)
    """
    try:
        hist_scan = pl.scan_parquet(
            str(PLAYER_GAME_DIR / "season=*/week=*/part.parquet"),
            glob=True,
            hive_partitioning=True,
            missing_columns="insert",
            extra_columns="ignore",
        )
    except Exception as e:
        logger.warning(f"Failed to load player_game_by_week for injury history: {e}")
        return pl.DataFrame()

    hist_scan = hist_scan.with_columns([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("player_id").cast(pl.Utf8),
    ])

    injury_hist_cols = [
        "recent_inactivity_count",
        "injury_hours_since_last_report",
        "injury_hours_until_game_at_last_report",
        "injury_hours_between_last_reports",
        "rest_days_since_last_game",
        "injury_player_inactive_rate_prior",
        "injury_depth_slot_inactive_rate_prior",
        "injury_practice_pattern_inactive_rate_prior",
        "injury_snapshot_valid",
        "injury_transaction_days_since",
        "injury_last_transaction_note",
        "injury_report_count",
        "depth_chart_mobility",
        "depth_chart_position",
        "was_inactive_last_game",
        "injury_inactive_probability",
        "snap_offense_pct_prev",
        "snap_offense_pct_l3",
        "snap_offense_snaps_prev",
        "snap_defense_pct_prev",
        "snap_defense_pct_l3",
        "snap_defense_snaps_prev",
        "snap_st_pct_prev",
        "snap_st_pct_l3",
        "snap_st_snaps_prev",
    ]

    hist_scan = hist_scan.filter(
        pl.col("season").is_in(list(seasons)) & pl.col("week").is_in(list(weeks))
    )

    select_cols = ["season", "week", "player_id"] + [
        c for c in injury_hist_cols if c in hist_scan.collect_schema().names()
    ]
    hist_df = hist_scan.select(select_cols).collect()
    
    if hist_df.is_empty():
        return hist_df

    # Type normalization
    cast_exprs = [
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("player_id").cast(pl.Utf8),
    ]
    for col in injury_hist_cols:
        if col in hist_df.columns and col not in ("injury_last_transaction_note", "depth_chart_position"):
            cast_exprs.append(pl.col(col).cast(pl.Float32))
    
    hist_df = hist_df.with_columns(cast_exprs)
    
    return hist_df


def load_ps_baselines(season: int) -> pl.DataFrame:
    """Load pre-snap participation baselines from prior seasons.
    
    Parameters
    ----------
    season : int
        Current season (will load from season-3 to season-1)
    
    Returns
    -------
    pl.DataFrame
        Player baselines indexed by player_id
    """
    if season is None:
        return pl.DataFrame()
    
    candidate_seasons = [yr for yr in range(season - 3, season) if yr > 2000]
    paths = _collect_partition_paths(PLAYER_GAME_DIR, candidate_seasons)
    if not paths:
        return pl.DataFrame()
    
    PS_BASELINE_COLUMNS = [
        "ps_route_participation_plays",
        "ps_team_dropbacks",
        "ps_route_participation_pct",
        "ps_targets_total",
        "ps_targets_slot_count",
        "ps_targets_wide_count",
        "ps_targets_inline_count",
        "ps_targets_backfield_count",
        "ps_targets_slot_share",
        "ps_targets_wide_share",
        "ps_targets_inline_share",
        "ps_targets_backfield_share",
        "ps_total_touches",
        "ps_scripted_touches",
        "ps_scripted_touch_share",
        "ps_tracking_team_dropbacks",
        "ps_tracking_has_game_data",
    ]
    
    scan = pl.scan_parquet(
        paths,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    required_cols = ["player_id", "game_date", *PS_BASELINE_COLUMNS]
    missing = [col for col in required_cols if col not in scan.collect_schema().names()]
    if missing:
        required_cols = [col for col in required_cols if col in scan.collect_schema().names()]
    if "player_id" not in required_cols or not required_cols:
        return pl.DataFrame()
    
    df = (
        scan.select(required_cols)
        .filter(pl.col("player_id").is_not_null() & pl.col("game_date").is_not_null())
        .with_columns(pl.col("game_date").cast(pl.Datetime("ms")))
        .sort(["player_id", "game_date"])
        .group_by("player_id", maintain_order=True)
        .agg([
            pl.col(col).last().alias(col) 
            for col in required_cols if col not in {"player_id", "game_date"}
        ])
        .collect()
    )
    
    return df
