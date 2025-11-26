from __future__ import annotations

"""Feature pipeline orchestrator for NFL player-level predictions.

Stages
------
1. **build_play_level**         – derive cleaned play-level slices
2. **build_player_drive_level** – aggregate per drive
3. **build_player_game_level**  – aggregate per game and join context

This wrapper triggers each stage, optionally enforces a schema after each
write, and returns the final game-level DataFrame for downstream usage.
"""

import logging
import os
from datetime import date, datetime, timedelta
from typing import Sequence

import polars as pl
import polars.selectors as cs

from utils.feature.play_level import build_play_level
from utils.feature.drive_level import build_drive_level
from utils.feature.game_level import build_game_level
from utils.feature.player_drive_level import build_player_drive_level
from utils.feature.player_game_level import build_player_game_level
from utils.feature.opponent_splits import build_opponent_splits
from utils.feature.labels import DEFAULT_LABEL_VERSION, get_label_spec
from utils.feature.targets import validate_target_columns
from utils.feature.daily_totals import build_daily_cache_range
from utils.feature.rolling_window import add_rolling_features
from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
from utils.feature.odds import add_nfl_odds_features_to_df
from utils.feature.team_context import add_team_context_features, compute_team_context_history
from utils.feature.offense_context import (
    add_offense_context_features_training,
    _append_offense_context_columns,
)
from utils.feature.leak_guard import (
    DEFAULT_LEAK_POLICY,
    build_schema_snapshot,
    enforce_leak_guard,
    write_schema_snapshot,
)
from utils.feature.weather_features import (
    add_weather_forecast_features_training,
    append_weather_context_flags,
)
from utils.feature.asof import (
    decision_cutoff_horizons,
    decision_cutoff_override,
    get_decision_cutoff_hours,
    fallback_cutoff_hours,
    drop_missing_snapshots_enabled,
)
from utils.feature.asof_metadata import build_asof_metadata, load_asof_metadata
from utils.general.constants import LEAK_PRONE_COLUMNS, format_cutoff_label
from utils.general.paths import (
    PLAY_BY_WEEK_DIR as PLAY_DIR,
    DRIVE_BY_WEEK_DIR as DRIVE_DIR,
    GAME_BY_WEEK_DIR as GAME_DIR,
    PLAYER_DRIVE_BY_WEEK_DIR as PLAYER_DRIVE_DIR,
    PLAYER_GAME_BY_WEEK_DIR as PLAYER_GAME_DIR,
    FINAL_FEATURES_PARQUET as FINAL_OUT,
    DATA_CLEANED as CLEAN_DIR,
    TEAM_CONTEXT_HISTORY_PATH as TEAM_CONTEXT_PATH,
    OFFENSE_CONTEXT_HISTORY_PATH as OFF_CONTEXT_PATH,
    PLAYER_DRIVE_HISTORY_PATH,
    FEATURE_AUDIT_DIR,
)

__all__ = ["build_feature_matrix", "refresh_context_histories"]

# Configure logging to see RollingWindow debug output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

ENABLE_WEATHER_FEATURES = True


def _get_clean_date_bounds() -> tuple[date | None, date | None]:
    """Return the earliest and latest available cleaned NFL dates."""
    if not CLEAN_DIR.exists():
        return None, None

    dates: list[date] = []
    for partition in CLEAN_DIR.glob("date=*"):
        try:
            _, value = partition.name.split("=", 1)
            dates.append(date.fromisoformat(value))
        except (ValueError, IndexError):
            continue

    if not dates:
        return None, None
    return min(dates), max(dates)


def _finalize_drive_history_features(drive_features: pl.DataFrame) -> pl.DataFrame:
    """Apply shared smoothing/lag logic to drive-level aggregates."""
    if drive_features.is_empty():
        return drive_features

    drive_features = drive_features.with_columns(
        [
            pl.col("game_date").cast(pl.Datetime("ms")),
            pl.col("drive_count").cast(pl.Float32),
        ]
    )
    drive_features = drive_features.sort(["player_id", "team", "game_date"])

    group_cols = ["player_id", "team"]
    numeric_cols = ["drive_count", "drive_touch_drives", "drive_td_drives", "drive_total_yards"]
    if "drive_red_zone_drives" in drive_features.columns:
        numeric_cols.append("drive_red_zone_drives")
    if "drive_goal_to_go_drives" in drive_features.columns:
        numeric_cols.append("drive_goal_to_go_drives")

    drive_features = drive_features.with_columns(
        [
            pl.col(col).cast(pl.Float32)
            for col in numeric_cols
            if col != "drive_count"
        ]
    )

    drive_features = drive_features.with_columns(
        pl.col("game_date").shift(1).over(group_cols).alias("data_as_of")
    )
    cumulative_cols = [f"{col}_cumulative" for col in numeric_cols]
    drive_features = drive_features.with_columns(
        [
            pl.col(col).cum_sum().over(group_cols).alias(f"{col}_cumulative")
            for col in numeric_cols
        ]
    )

    def _prev_name(col: str) -> str:
        return f"drive_hist_{col.removeprefix('drive_')}_prev"

    def _l3_name(col: str) -> str:
        return f"drive_hist_{col.removeprefix('drive_')}_l3"

    drive_features = drive_features.with_columns(
        [
            pl.col(f"{col}_cumulative")
            .shift(1)
            .over(group_cols)
            .fill_null(0.0)
            .alias(_prev_name(col))
            for col in numeric_cols
        ]
    )
    drive_features = drive_features.with_columns(
        [
            pl.col(col)
            .rolling_sum(window_size=3, min_periods=1)
            .shift(1)
            .over(group_cols)
            .fill_null(0.0)
            .alias(_l3_name(col))
            for col in numeric_cols
        ]
    )
    drive_features = drive_features.drop(cumulative_cols)

    rate_specs = [
        ("drive_touch_drives", "drive_count", "touch_rate"),
        ("drive_td_drives", "drive_count", "td_rate"),
        ("drive_total_yards", "drive_count", "yards_per_drive"),
    ]
    if "drive_red_zone_drives" in numeric_cols:
        rate_specs.append(("drive_red_zone_drives", "drive_count", "red_zone_rate"))
    if "drive_goal_to_go_drives" in numeric_cols:
        rate_specs.append(("drive_goal_to_go_drives", "drive_count", "goal_to_go_rate"))

    rate_exprs: list[pl.Expr] = []
    rate_feature_names: list[str] = []
    for numerator, denominator, label in rate_specs:
        prev_num = _prev_name(numerator)
        prev_den = _prev_name(denominator)
        l3_num = _l3_name(numerator)
        l3_den = _l3_name(denominator)
        prev_label = f"drive_hist_{label}_prev"
        l3_label = f"drive_hist_{label}_l3"
        rate_exprs.append(
            pl.when(pl.col(prev_den) > 0)
            .then(pl.col(prev_num) / pl.col(prev_den))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias(prev_label)
        )
        rate_exprs.append(
            pl.when(pl.col(l3_den) > 0)
            .then(pl.col(l3_num) / pl.col(l3_den))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias(l3_label)
        )
        rate_feature_names.extend([prev_label, l3_label])

    if rate_exprs:
        drive_features = drive_features.with_columns(rate_exprs)

    feature_cols = (
        [_prev_name(col) for col in numeric_cols]
        + [_l3_name(col) for col in numeric_cols]
    )
    feature_cols.extend(rate_feature_names)
    feature_cols = list(dict.fromkeys(feature_cols))

    drive_features = drive_features.select(
        [
            "season",
            "week",
            "game_id",
            "team",
            "player_id",
            "game_date",
            "data_as_of",
            *feature_cols,
        ]
    )

    return drive_features.with_columns(
        [pl.col(col).fill_null(0.0).cast(pl.Float32) for col in feature_cols]
    )

# ---------------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------------

def _build_feature_matrix_internal(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    chunk_days: int = 7,
    recompute_intermediate: bool = True,
    build_rolling: bool = True,
    output_path: Path | None = None,
    primary_output_path: Path | None = None,
    cutoff_label: str | None = None,
    label_version: str | None = None,
) -> pl.DataFrame:
    """Run full feature pipeline up to *inclusive* `end_date` for a single cutoff window.

    Parameters
    ----------
    end_date : date | None
        Last `game_date` to include in rolling windows (default: today).
    save_output : bool
        If True (default) – call `update_daily_stats` after game stage.
    """
    cutoff_label = cutoff_label or "default"
    target_output_path = output_path or FINAL_OUT
    primary_output = primary_output_path or target_output_path
    current_cutoff_hours = float(get_decision_cutoff_hours())
    clean_min_date, clean_max_date = (None, None)
    label_version = label_version or DEFAULT_LABEL_VERSION
    if start_date is None or end_date is None:
        clean_min_date, clean_max_date = _get_clean_date_bounds()

    end_date = end_date or clean_max_date or date.today()
    start_date = start_date or clean_min_date or end_date          # “one-shot” default

    if clean_max_date and end_date > clean_max_date:
        logging.info(
            "Clamping end_date %s to latest cleaned date %s to avoid empty slices",
            end_date,
            clean_max_date,
        )
        end_date = clean_max_date

    if start_date > end_date:
        raise ValueError("start_date must be ≤ end_date")

    logging.info(
        "Feature pipeline: generating features %s → %s in %s-day chunks",
        start_date, end_date, chunk_days,
    )

    # ------------------------------------------------------------------
    # Modular build (play → player-drive → player-game) from cleaned daily Parquets
    # Note: drive_level and game_level are team-aggregated (kept for legacy/analysis)
    # ------------------------------------------------------------------
    cur = start_date
    if recompute_intermediate:
        while cur <= end_date:
            chunk_end = min(cur + timedelta(days=chunk_days - 1), end_date)
            logging.info("🔹  Processing chunk %s → %s", cur, chunk_end)

            # 1. Play-level selection (from cleaned daily data)
            build_play_level(start_date=cur, end_date=chunk_end)

            # 2. Team-level aggregations (SKIPPED - not needed for player predictions)
            # build_drive_level(start_date=cur, end_date=chunk_end)
            # build_game_level(start_date=cur, end_date=chunk_end)
            
            # 3. Opponent splits (defensive weaknesses)
            build_opponent_splits(start_date=cur, end_date=chunk_end)
            
            # 4. Player-drive aggregations (for rolling window features)
            build_player_drive_level(start_date=cur, end_date=chunk_end)
            
            # 5. Player-game aggregations (for rolling window features)
            build_player_game_level(start_date=cur, end_date=chunk_end, label_version=label_version)

            # advance to next slice
            cur += timedelta(days=chunk_days)

    # ------------------------------------------------------------------
    # Build daily cache for rolling windows (if enabled)
    # ------------------------------------------------------------------
    if build_rolling:
        logging.info("🔹 Building daily totals cache for rolling windows...")
        build_daily_cache_range(
            start_date=start_date,
            end_date=end_date,
            level="game",
        )
        logging.info("✅  Daily cache built")

    # ------------------------------------------------------------------
    # Build final monolithic feature matrix from player-game level data
    # This is what training will use
    # ------------------------------------------------------------------
    logging.info("Building final feature matrix from player-game aggregations...")
    
    player_game_scan = pl.scan_parquet(
        str(PLAYER_GAME_DIR / "season=*/week=*/part.parquet"),
        glob=True,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    player_game_scan = player_game_scan.drop(cs.starts_with("weather_"))
    pg_schema = player_game_scan.collect_schema()
    pg_categorical_cols = [
        name
        for name, dtype in zip(pg_schema.names(), pg_schema.dtypes())
        if "Categorical" in str(dtype)
    ]
    if pg_categorical_cols:
        player_game_scan = player_game_scan.with_columns([pl.col(col).cast(pl.Utf8) for col in pg_categorical_cols])

    if "game_start_utc" in pg_schema.names():
        player_game_scan = player_game_scan.with_columns(
            pl.col("game_start_utc")
            .dt.cast_time_unit("ms")
            .dt.replace_time_zone("UTC")
            .alias("game_start_utc")
        )

    player_game_scan = player_game_scan.with_columns([
        pl.col("player_id").cast(pl.Utf8),
        pl.col("player_name").cast(pl.Utf8),
        pl.col("team").cast(pl.Utf8),
        pl.col("opponent").cast(pl.Utf8),
        pl.col("game_id").cast(pl.Utf8),
    ])

    df_player_game_all = player_game_scan.collect(streaming=True)

    logging.info("Attaching as-of metadata...")
    seasons_available = df_player_game_all.get_column("season").unique().to_list() if "season" in df_player_game_all.columns else []
    game_ids = df_player_game_all.get_column("game_id").unique().to_list() if "game_id" in df_player_game_all.columns else []
    asof_meta = pl.DataFrame()
    try:
        asof_meta = load_asof_metadata()
    except Exception as exc:
        logger.warning("Failed to load existing as-of metadata: %s", exc)
        asof_meta = pl.DataFrame()

    if asof_meta.is_empty() or (game_ids and asof_meta.filter(pl.col("game_id").is_in(game_ids)).height < len(game_ids)):
        try:
            asof_meta = build_asof_metadata(seasons_available or [int(date.today().year)], force=True)
        except Exception as exc:
            logger.warning("Unable to build as-of metadata: %s", exc)
            asof_meta = pl.DataFrame()

    if not asof_meta.is_empty():
        asof_meta = asof_meta.filter(pl.col("game_id").is_in(game_ids))
        asof_meta = asof_meta.with_columns(
            [
                pl.col("cutoff_ts").alias("decision_cutoff_ts"),
                pl.col("cutoff_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("injury_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("roster_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("odds_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("forecast_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
            ]
        )
        if asof_meta.height:
            coverage_frame = asof_meta.select(
                [
                    pl.col("injury_snapshot_ts").is_not_null().mean().alias("injury_snapshot_coverage"),
                    pl.col("roster_snapshot_ts").is_not_null().mean().alias("roster_snapshot_coverage"),
                    pl.col("odds_snapshot_ts").is_not_null().mean().alias("odds_snapshot_coverage"),
                    pl.col("forecast_snapshot_ts").is_not_null().mean().alias("forecast_snapshot_coverage"),
                ]
            )
            if coverage_frame.height:
                coverage_dict = {k: float(v) for k, v in coverage_frame.to_dicts()[0].items()}
                summary = ", ".join(f"{k.replace('_coverage', '')}: {v*100:.1f}%" for k, v in coverage_dict.items())
                logging.info("As-of snapshot coverage for requested games — %s", summary)
                weak_signals = {k: v for k, v in coverage_dict.items() if v < 0.8}
                if weak_signals:
                    logging.warning(
                        "Snapshot coverage below 80%% for: %s. Recent rebuilds may be missing data; "
                        "run collectors or backfills before production scoring.",
                        ", ".join(f"{k.replace('_coverage', '')} ({v*100:.1f}%)" for k, v in weak_signals.items()),
                    )
            if "decision_cutoff_hours" in asof_meta.columns and "odds_snapshot_ts" in asof_meta.columns:
                horizon_stats = (
                    asof_meta.group_by("decision_cutoff_hours")
                    .agg(
                        pl.col("odds_snapshot_ts").is_not_null().mean().alias("odds_snapshot_coverage"),
                        pl.count().alias("game_count"),
                    )
                    .sort("decision_cutoff_hours")
                )
                for row in horizon_stats.to_dicts():
                    logging.info(
                        "Odds snapshot coverage @ %sh before kickoff: %.1f%% (%d games)",
                        row["decision_cutoff_hours"],
                        row["odds_snapshot_coverage"] * 100.0,
                        row["game_count"],
                    )
        df_player_game_all = df_player_game_all.join(asof_meta, on="game_id", how="left")
        if "odds_snapshot_ts_right" in df_player_game_all.columns and "odds_snapshot_ts" in df_player_game_all.columns:
            df_player_game_all = (
                df_player_game_all.with_columns(
                    pl.coalesce(pl.col("odds_snapshot_ts"), pl.col("odds_snapshot_ts_right")).alias("odds_snapshot_ts")
                )
                .drop("odds_snapshot_ts_right")
            )
        required_snapshot_cols = [
            "decision_cutoff_ts",
            "injury_snapshot_ts",
            "roster_snapshot_ts",
            "odds_snapshot_ts",
            "forecast_snapshot_ts",
        ]
        missing_flag_exprs = [
            pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_missing")
            for col in required_snapshot_cols
            if col in df_player_game_all.columns
        ]
        if missing_flag_exprs:
            df_player_game_all = df_player_game_all.with_columns(missing_flag_exprs)

        drop_missing = drop_missing_snapshots_enabled()
        non_null_filters = [
            pl.col(col).is_not_null() for col in required_snapshot_cols if col in df_player_game_all.columns
        ]
        if drop_missing and non_null_filters:
            before_rows = df_player_game_all.height
            df_player_game_all = df_player_game_all.filter(pl.all_horizontal(non_null_filters))
            removed_rows = before_rows - df_player_game_all.height
            if removed_rows > 0:
                logging.warning(
                    "Dropped %d rows lacking full pre-cutoff snapshots (required columns: %s).",
                    removed_rows,
                    ", ".join(required_snapshot_cols),
                )
    
    if "decision_cutoff_ts" in df_player_game_all.columns:
        guard_columns = [
            ("injury_snapshot_ts", "injury"),
            ("roster_snapshot_ts", "roster"),
            ("odds_snapshot_ts", "odds"),
            ("forecast_snapshot_ts", "weather"),
        ]
        audit_paths = {
            "roster": FEATURE_AUDIT_DIR / "roster_cutoff_violations.parquet",
            "odds": FEATURE_AUDIT_DIR / "odds_cutoff_violations.parquet",
        }
        for column_name, label in guard_columns:
            if column_name in df_player_game_all.columns:
                violation_mask = pl.col(column_name).is_not_null() & (
                    pl.col(column_name) > pl.col("decision_cutoff_ts")
                )
                violations = df_player_game_all.filter(violation_mask)
                if violations.height:
                    audit_path = audit_paths.get(label)
                    if audit_path:
                        audit_path.parent.mkdir(parents=True, exist_ok=True)
                        cols_to_save = [
                            col
                            for col in ("game_id", "player_id", "team", "decision_cutoff_ts", column_name)
                            if col in violations.columns
                        ]
                        violations.select(cols_to_save).write_parquet(audit_path, compression="zstd")
                before_guard = df_player_game_all.height
                df_player_game_all = df_player_game_all.filter(
                    pl.col(column_name).is_null() | (pl.col(column_name) <= pl.col("decision_cutoff_ts"))
                )
                dropped_guard = before_guard - df_player_game_all.height
                if dropped_guard > 0:
                    logging.warning(
                        "Dropped %d %s rows with snapshots after cutoff.",
                        dropped_guard,
                        label,
                    )

    # ------------------------------------------------------------------
    # Join drive-level aggregates for richer usage context
    # ------------------------------------------------------------------
    drive_scan = pl.scan_parquet(
        str(PLAYER_DRIVE_DIR / "season=*/week=*/part.parquet"),
        glob=True,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    drive_schema = drive_scan.collect_schema()
    drive_categorical_cols = [
        name
        for name, dtype in zip(drive_schema.names(), drive_schema.dtypes())
        if "Categorical" in str(dtype)
    ]
    if drive_categorical_cols:
        drive_scan = drive_scan.with_columns([pl.col(col).cast(pl.Utf8) for col in drive_categorical_cols])
    drive_scan = drive_scan.with_columns(
        [
            pl.col("team").cast(pl.Utf8) if "team" in drive_schema.names() else pl.lit(None).cast(pl.Utf8).alias("team"),
            pl.col("player_id").cast(pl.Utf8) if "player_id" in drive_schema.names() else pl.lit(None).cast(pl.Utf8).alias("player_id"),
            pl.col("game_id").cast(pl.Utf8) if "game_id" in drive_schema.names() else pl.lit(None).cast(pl.Utf8).alias("game_id"),
            pl.col("season").cast(pl.Int32) if "season" in drive_schema.names() else pl.lit(None).cast(pl.Int32).alias("season"),
            pl.col("week").cast(pl.Int32) if "week" in drive_schema.names() else pl.lit(None).cast(pl.Int32).alias("week"),
        ]
    )
    drive_schema_names = set(drive_schema.names())

    agg_exprs: list[pl.Expr] = [
        pl.count().alias("drive_count"),
        pl.col("game_date").max().alias("game_date"),
        pl.when(
            (
                pl.col("carry").fill_null(0)
                + pl.col("target").fill_null(0)
                + pl.col("pass_attempt").fill_null(0)
                + pl.col("reception").fill_null(0)
            )
            > 0
        )
        .then(1)
        .otherwise(0)
        .sum()
        .cast(pl.Float32)
        .alias("drive_touch_drives"),
        pl.when(pl.col("touchdown").fill_null(0) > 0)
        .then(1)
        .otherwise(0)
        .sum()
        .cast(pl.Float32)
        .alias("drive_td_drives"),
        (
            pl.col("passing_yards").fill_null(0)
            + pl.col("rushing_yards").fill_null(0)
            + pl.col("receiving_yards").fill_null(0)
        )
        .sum()
        .cast(pl.Float32)
        .alias("drive_total_yards"),
    ]

    if {"red_zone_carry", "red_zone_target"} <= drive_schema_names:
        agg_exprs.append(
            pl.when(
                (
                    pl.col("red_zone_carry").fill_null(0)
                    + pl.col("red_zone_target").fill_null(0)
                )
                > 0
            )
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Float32)
            .alias("drive_red_zone_drives")
        )

    if {"goal_to_go_carry", "goal_to_go_target"} <= drive_schema_names:
        agg_exprs.append(
            pl.when(
                (
                    pl.col("goal_to_go_carry").fill_null(0)
                    + pl.col("goal_to_go_target").fill_null(0)
                )
                > 0
            )
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Float32)
            .alias("drive_goal_to_go_drives")
        )

    drive_features = (
        drive_scan.group_by(["season", "week", "game_id", "team", "player_id"])
        .agg(agg_exprs)
        .collect(streaming=True)
    )
    drive_features = _finalize_drive_history_features(drive_features)
    if not drive_features.is_empty():
        df_player_game_all = df_player_game_all.join(
            drive_features.drop("game_date"),
            on=["season", "week", "game_id", "team", "player_id"],
            how="left",
        )

        PLAYER_DRIVE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        drive_features.write_parquet(PLAYER_DRIVE_HISTORY_PATH, compression="zstd")
    
    if df_player_game_all.is_empty():
        logging.warning("No player-game rows found for final matrix; skipping.")
        return pl.DataFrame()
    
    # Normalize key categorical columns
    string_cols = [
        col for col in (
            "player_id",
            "player_name",
            "team",
            "opponent",
            "game_id",
            "stadium_key",
            "stadium_name",
            "stadium_tz",
            "roof",
            "surface",
            "home_team",
            "away_team",
            "position",
            "position_group",
            "depth_chart_position",
            "injury_report_status",
            "injury_practice_status",
            "injury_report_primary",
            "injury_practice_primary",
            "season_type",
        ) if col in df_player_game_all.columns
    ]
    if string_cols:
        df_player_game_all = df_player_game_all.with_columns([pl.col(col).cast(pl.Utf8) for col in string_cols])

    if "home_team_abbr" not in df_player_game_all.columns and "home_team" in df_player_game_all.columns:
        df_player_game_all = df_player_game_all.with_columns(pl.col("home_team").alias("home_team_abbr"))
    if "away_team_abbr" not in df_player_game_all.columns and "away_team" in df_player_game_all.columns:
        df_player_game_all = df_player_game_all.with_columns(pl.col("away_team").alias("away_team_abbr"))

    logging.info("Enriching team context features...")
    df_player_game_all, team_context_history = add_team_context_features(
        df_player_game_all, return_history=True
    )
    TEAM_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_context_history.write_parquet(TEAM_CONTEXT_PATH, compression="zstd")

    logging.info("Adding pace and defensive matchup features...")
    pace_exprs: list[pl.Expr] = []
    available_cols = set(df_player_game_all.columns)
    if "team_ctx_offensive_plays_prev" in available_cols:
        pace_exprs.append(pl.col("team_ctx_offensive_plays_prev").cast(pl.Float32).alias("team_pace_prev"))
    if "team_ctx_offensive_plays_l3" in available_cols:
        pace_exprs.append(pl.col("team_ctx_offensive_plays_l3").cast(pl.Float32).alias("team_pace_l3"))
    if "opp_ctx_offensive_plays_prev" in available_cols:
        pace_exprs.append(pl.col("opp_ctx_offensive_plays_prev").cast(pl.Float32).alias("opp_pace_prev"))
    if "opp_ctx_offensive_plays_l3" in available_cols:
        pace_exprs.append(pl.col("opp_ctx_offensive_plays_l3").cast(pl.Float32).alias("opp_pace_l3"))
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
        df_player_game_all = df_player_game_all.with_columns(pace_exprs)

    logging.info("Enriching offense situational context (OC/QB)...")
    df_player_game_all = add_offense_context_features_training(
        df_player_game_all,
        history_path=OFF_CONTEXT_PATH,
    )

    if ENABLE_WEATHER_FEATURES:
        logging.info("Enriching weather forecast features...")
        df_player_game_all = add_weather_forecast_features_training(
            df_player_game_all,
            cutoff_column="decision_cutoff_ts",
        )
    else:
        logging.info("Skipping weather forecast enrichment (disabled until coverage improves).")
    weather_cols = [col for col in df_player_game_all.columns if col.startswith("weather_")]
    if weather_cols:
        drop_weather_cols = [
            col
            for col in weather_cols
            if col.endswith("_ts")
            or col.endswith("_source_detail")
            or col in {"weather_conditions", "weather_precip_type"}
        ]
        if drop_weather_cols:
            df_player_game_all = df_player_game_all.drop(drop_weather_cols, strict=False)
        weather_numeric_cols = [
            col
            for col in weather_cols
            if col not in drop_weather_cols
            and any(
                df_player_game_all.schema.get(col) == dtype
                for dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32)
            )
        ]
        if weather_numeric_cols:
            df_player_game_all = df_player_game_all.with_columns(
                [pl.col(col).cast(pl.Float32) for col in weather_numeric_cols]
            )
        weather_flag_cols = [
            col
            for col in weather_cols
            if col not in drop_weather_cols
            and (
                col.endswith("_flag")
                or col.endswith("_is_backfill")
                or col.endswith("_is_historical")
            )
        ]
        if weather_flag_cols:
            df_player_game_all = df_player_game_all.with_columns(
                [pl.col(col).cast(pl.Int8) for col in weather_flag_cols]
            )
        df_player_game_all = append_weather_context_flags(
            df_player_game_all,
            roof_col="roof",
        )

    logging.info("Enriching odds features (NFL)...")
    rows_before_odds = df_player_game_all.height
    df_player_game_all = add_nfl_odds_features_to_df(
        df_player_game_all,
        player_col="player_name",
        allow_schedule_fallback=False,
        drop_schedule_rows=True,
    )

    rows_after_odds = df_player_game_all.height
    logging.info(
        "Odds features attached; retained %d/%d rows (%.2f%%) with pre-cutoff snapshots",
        rows_after_odds,
        rows_before_odds,
        (rows_after_odds / rows_before_odds * 100.0) if rows_before_odds else 0.0,
    )
    if "odds_schedule_fallback" not in df_player_game_all.columns:
        df_player_game_all = df_player_game_all.with_columns(
            pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")
        )
    if "odds_anytime_td_price" in df_player_game_all.columns:
        df_player_game_all = df_player_game_all.drop("odds_anytime_td_price", strict=False)
        logging.info("Player prop odds enrichment disabled to avoid live snapshot leakage.")

    numeric_casts = []
    if "depth_chart_order" in df_player_game_all.columns:
        numeric_casts.append(pl.col("depth_chart_order").cast(pl.Int16))
    for col in ("is_home", "game_start_hour_utc", "game_day_of_week"):
        if col in df_player_game_all.columns:
            numeric_casts.append(pl.col(col).cast(pl.Int8))
    for col in (
        "injury_is_listed",
        "injury_practice_dnp_count",
        "injury_practice_limited_count",
        "injury_practice_full_count",
    ):
        if col in df_player_game_all.columns:
            numeric_casts.append(pl.col(col).cast(pl.Int8))
    if numeric_casts:
        df_player_game_all = df_player_game_all.with_columns(numeric_casts)

    if (
        "injury_inactive_probability" in df_player_game_all.columns
        and "injury_inactive_probability_source" in df_player_game_all.columns
        and "injury_inactive_probability_model" not in df_player_game_all.columns
    ):
        df_player_game_all = df_player_game_all.with_columns(
            pl.when(pl.col("injury_inactive_probability_source") == "model")
            .then(pl.col("injury_inactive_probability"))
            .otherwise(pl.lit(None))
            .alias("injury_inactive_probability_model")
        )

    # Validate target columns exist
    df_player_game_all = validate_target_columns(df_player_game_all, strict=False)
    
    # ------------------------------------------------------------------
    # Add rolling window features (if enabled)
    # ------------------------------------------------------------------
    if build_rolling:
        logging.info("🔹 Computing rolling window features...")

        rolling_stats = ROLLING_FEATURE_STATS  # Curated, supported stats only
        rolling_windows = ROLLING_WINDOWS      # Trimmed windows: 1,3,5,season
        rolling_contexts = ROLLING_CONTEXTS    # Contexts supported by cache: vs_any

        missing_stats = [s for s in rolling_stats if s not in df_player_game_all.columns]
        if missing_stats:
            logging.warning(
                "Rolling feature stats missing from dataset, will skip: %s",
                ", ".join(sorted(missing_stats)),
            )
            rolling_stats = [s for s in rolling_stats if s not in missing_stats]
        if not rolling_stats:
            logging.warning("No rolling stats available; skipping rolling feature computation.")
            logging.info("✅  Added 0 rolling features")
        else:
            df_player_game_all = add_rolling_features(
                df_player_game_all,
                level="game",
                stats=rolling_stats,
                windows=rolling_windows,
                contexts=rolling_contexts,
                date_col="game_date",
                player_col="player_id",
                opponent_col="opponent",
            )
        
            rolling_cols = [c for c in df_player_game_all.columns if "g_" in c or "season" in c]
            logging.info(f"✅  Added {len(rolling_cols)} rolling features")
    
    logging.info("Deriving historical usage share features...")
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
    available_cols = set(df_player_game_all.columns)
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
        df_player_game_all = df_player_game_all.with_columns(share_exprs)
        logging.info("Added %d historical share features", len(share_exprs))
    leakage_cols = [
        "target_share",
        "carry_share",
        "pass_attempt_share",
        "red_zone_target_share",
        "red_zone_carry_share",
        "goal_to_go_target_share",
        "goal_to_go_carry_share",
    ]
    df_player_game_all = df_player_game_all.drop(leakage_cols, strict=False)
    # Keep raw outcome columns (targets, carries, etc.) in the feature matrix so they can
    # serve as supervised targets. These columns are excluded from model inputs later via
    # training configuration, so we avoid dropping them here to retain label availability.

    available_cols = set(df_player_game_all.columns)
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
        df_player_game_all = df_player_game_all.with_columns(market_exprs)
    
    # Leak guard + schema snapshot before persisting
    spec = get_label_spec(label_version)
    allow_exact = {spec.primary, *spec.labels.keys(), *spec.aliases.keys()}
    df_player_game_all, leak_result = enforce_leak_guard(
        df_player_game_all,
        policy=DEFAULT_LEAK_POLICY,
        allow_exact=allow_exact,
        drop_banned=True,
        drop_non_allowlisted=False,
        raise_on_banned=True,
    )

    schema = build_schema_snapshot(
        df_player_game_all,
        policy=DEFAULT_LEAK_POLICY,
        metadata={
            "cutoff_label": cutoff_label,
            "label_version": spec.name,
        },
        banned=leak_result.banned,
    )
    schema_dir = FEATURE_AUDIT_DIR / "schema" / "anytime_td"
    schema_path = schema_dir / f"{cutoff_label}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    write_schema_snapshot(schema_path, schema)
    logging.info("    Schema snapshot written → %s", schema_path)

    # Write final feature matrix
    df_player_game_all = df_player_game_all.with_columns(
        pl.lit(current_cutoff_hours).cast(pl.Float32).alias("decision_horizon_hours")
    )

    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_player_game_all.write_parquet(target_output_path, compression="zstd")

    if primary_output != target_output_path:
        primary_output.parent.mkdir(parents=True, exist_ok=True)
        df_player_game_all.write_parquet(primary_output, compression="zstd")

    logging.info("✅  Final feature matrix (%s cutoff) written → %s", cutoff_label, target_output_path)
    if primary_output != target_output_path:
        logging.info("    Primary feature matrix updated → %s", primary_output)
    logging.info("    Rows: %d, Columns: %d", len(df_player_game_all), len(df_player_game_all.columns))
    logging.info("    Date range: %s to %s", 
                 df_player_game_all["game_date"].min(), 
                 df_player_game_all["game_date"].max())
    
    label_candidates = {spec.primary, "anytime_td"}
    for label_col in label_candidates:
        if label_col in df_player_game_all.columns:
            td_count = df_player_game_all[label_col].sum()
            td_rate = td_count / len(df_player_game_all) * 100
            logging.info(
                "    Label %s rate: %.2f%% (%d/%d)",
                label_col,
                td_rate,
                td_count,
                len(df_player_game_all),
            )
    
    return df_player_game_all


def build_feature_matrix(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    chunk_days: int = 7,
    recompute_intermediate: bool = True,
    build_rolling: bool = True,
    cutoff_hours_list: Sequence[float] | None = None,
    label_version: str | None = None,
) -> pl.DataFrame:
    """Build feature matrices for one or more decision cutoff horizons.

    Returns the feature frame associated with the primary (first) horizon.
    """
    horizons = [float(h) for h in (cutoff_hours_list or decision_cutoff_horizons())]
    if not horizons:
        horizons = [float(get_decision_cutoff_hours())]
    fallback_default = float(fallback_cutoff_hours())

    primary_result: pl.DataFrame | None = None

    for idx, hours in enumerate(horizons):
        label = format_cutoff_label(hours)
        output_path = FINAL_OUT.with_name(f"{FINAL_OUT.stem}_{label}{FINAL_OUT.suffix}")
        primary_output = FINAL_OUT if idx == 0 else None
        logging.info("===== Building feature matrix for cutoff %.2f hours (%s) =====", hours, label)
        with decision_cutoff_override(cutoff_hours=hours, fallback_hours=fallback_default):
            df = _build_feature_matrix_internal(
                start_date=start_date,
                end_date=end_date,
                chunk_days=chunk_days,
                recompute_intermediate=recompute_intermediate if idx == 0 else False,
                build_rolling=build_rolling,
                output_path=output_path,
                primary_output_path=primary_output,
                cutoff_label=label,
                label_version=label_version,
            )
        if idx == 0:
            primary_result = df

    if primary_result is None:
        raise RuntimeError("Feature matrix generation produced no outputs.")
    return primary_result


def refresh_context_histories(*, end_date: date | None = None) -> None:
    """Recompute team, offense, and drive context caches up to `end_date`."""
    logging.info(
        "Refreshing context histories%s",
        f" through {end_date}" if end_date else "",
    )

    player_game_scan = pl.scan_parquet(
        str(PLAYER_GAME_DIR / "season=*/week=*/part.parquet"),
        glob=True,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    player_game_scan = player_game_scan.with_columns(
        [
            pl.col("player_id").cast(pl.Utf8),
            pl.col("team").cast(pl.Utf8),
            pl.col("opponent").cast(pl.Utf8),
            pl.col("game_date").cast(pl.Date),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
        ]
    )
    if end_date is not None:
        player_game_scan = player_game_scan.filter(pl.col("game_date") <= pl.lit(end_date))

    df_player_game = player_game_scan.collect(streaming=True)
    if df_player_game.is_empty():
        logging.warning("No player-game rows available; skipping context refresh.")
        return

    df_player_game = df_player_game.with_columns(
        pl.col("game_date").cast(pl.Datetime("ms"))
    )

    logging.info("Recomputing team context history...")
    team_history = compute_team_context_history(df_player_game)
    TEAM_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_history.write_parquet(TEAM_CONTEXT_PATH, compression="zstd")
    logging.info(
        "Team context history updated → %s (%d rows)",
        TEAM_CONTEXT_PATH,
        len(team_history),
    )

    logging.info("Recomputing offense context history...")
    seasons = df_player_game.get_column("season").unique().to_list()
    _, offense_history = _append_offense_context_columns(df_player_game, seasons)
    OFF_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    offense_history.write_parquet(OFF_CONTEXT_PATH, compression="zstd")
    logging.info(
        "Offense context history updated → %s (%d rows)",
        OFF_CONTEXT_PATH,
        len(offense_history),
    )

    logging.info("Recomputing drive history...")
    drive_scan = pl.scan_parquet(
        str(PLAYER_DRIVE_DIR / "season=*/week=*/part.parquet"),
        glob=True,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    drive_scan = drive_scan.with_columns(
        [
            pl.col("player_id").cast(pl.Utf8),
            pl.col("team").cast(pl.Utf8),
            pl.col("game_id").cast(pl.Utf8),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("game_date").cast(pl.Datetime("ms")),
        ]
    )
    if end_date is not None:
        drive_scan = drive_scan.filter(
            pl.col("game_date").cast(pl.Date) <= pl.lit(end_date)
        )
    drive_df = drive_scan.collect(streaming=True)
    if drive_df.is_empty():
        logging.warning("No player-drive rows available; drive history not refreshed.")
    else:
        agg_exprs: list[pl.Expr] = [
            pl.count().alias("drive_count"),
            pl.col("game_date").max().alias("game_date"),
            pl.when(
                (
                    pl.col("carry").fill_null(0)
                    + pl.col("target").fill_null(0)
                    + pl.col("pass_attempt").fill_null(0)
                    + pl.col("reception").fill_null(0)
                )
                > 0
            )
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Float32)
            .alias("drive_touch_drives"),
            pl.when(pl.col("touchdown").fill_null(0) > 0)
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Float32)
            .alias("drive_td_drives"),
            (
                pl.col("passing_yards").fill_null(0)
                + pl.col("rushing_yards").fill_null(0)
                + pl.col("receiving_yards").fill_null(0)
            )
            .sum()
            .cast(pl.Float32)
            .alias("drive_total_yards"),
        ]
        if {"red_zone_carry", "red_zone_target"} <= set(drive_df.columns):
            agg_exprs.append(
                pl.when(
                    (
                        pl.col("red_zone_carry").fill_null(0)
                        + pl.col("red_zone_target").fill_null(0)
                    )
                    > 0
                )
                .then(1)
                .otherwise(0)
                .sum()
                .cast(pl.Float32)
                .alias("drive_red_zone_drives")
            )
        if {"goal_to_go_carry", "goal_to_go_target"} <= set(drive_df.columns):
            agg_exprs.append(
                pl.when(
                    (
                        pl.col("goal_to_go_carry").fill_null(0)
                        + pl.col("goal_to_go_target").fill_null(0)
                    )
                    > 0
                )
                .then(1)
                .otherwise(0)
                .sum()
                .cast(pl.Float32)
                .alias("drive_goal_to_go_drives")
            )

        drive_agg = drive_df.group_by(["season", "week", "game_id", "team", "player_id"]).agg(agg_exprs)
        drive_agg = _finalize_drive_history_features(drive_agg)
        if drive_agg.is_empty():
            logging.warning("Aggregated drive history empty; nothing written.")
        else:
            PLAYER_DRIVE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            drive_agg.write_parquet(PLAYER_DRIVE_HISTORY_PATH, compression="zstd")
            logging.info(
                "Drive history updated → %s (%d rows)",
                PLAYER_DRIVE_HISTORY_PATH,
                len(drive_agg),
            )

    logging.info("Refreshing as-of metadata...")
    try:
        build_asof_metadata(seasons, force=True)
    except Exception as exc:
        logging.warning("Failed to rebuild as-of metadata: %s", exc)

    logging.info("Context histories refreshed.")
