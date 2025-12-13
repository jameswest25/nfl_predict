"""Feature pipeline orchestrator for NFL player-level predictions.

Stages
------
1. **build_play_level**         – derive cleaned play-level slices
2. **build_player_drive_level** – aggregate per drive
3. **build_player_game_level**  – aggregate per game and join context

This wrapper triggers each stage, optionally enforces a schema after each
write, and returns the final game-level DataFrame for downstream usage.

The heavy feature engineering logic has been extracted to utility modules in
utils/feature/ for easier maintenance:
- usage_features.py      - usage model helpers and position baselines
- position_features.py   - MoE and position-specific features
- catch_rate_features.py - catch rate efficiency features
- target_depth_features.py - air yards and route features
- snap_features.py       - rolling snaps, expected snaps, role stability
- historical_share_features.py - historical usage shares
- market_features.py     - market/odds-derived features
- asof_enrichment.py     - as-of metadata enrichment
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import polars as pl
import polars.selectors as cs

# Stage builders
from utils.feature.builders.play_level import build_play_level
from utils.feature.builders.player_drive_level import build_player_drive_level
from utils.feature.builders.player_game_level import build_player_game_level
from utils.feature.builders.opponent_splits import build_opponent_splits
from utils.feature.rolling.daily_totals import build_daily_cache_range
from utils.feature.rolling.rolling_window import add_rolling_features
from utils.feature.rolling.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS

# Context enrichment
from utils.feature.enrichment.team_context import add_team_context_features, compute_team_context_history
from utils.feature.enrichment.offense_context import (
    add_offense_context_features_training,
    _append_offense_context_columns,
)
from utils.feature.enrichment.weather_features import (
    add_weather_forecast_features_training,
    append_weather_context_flags,
)
from utils.feature.enrichment.odds import add_nfl_odds_features_to_df
from utils.feature.core.shared import (
    finalize_drive_history_features,
    attach_td_rate_history_features,
)

# Feature enrichment utilities
from utils.feature.derived.usage_features import add_usage_helper_features
from utils.feature.derived.position_features import (
    add_position_group,
    add_specialist_role_flags,
    add_moe_position_features,
)
from utils.feature.derived.catch_rate_features import add_catch_rate_features
from utils.feature.derived.target_depth_features import add_target_depth_features
from utils.feature.derived.snap_features import (
    add_rolling_snap_features,
    add_expected_snap_features,
    add_role_stability_features,
    add_market_odds_flag,
)
from utils.feature.derived.historical_share_features import (
    add_historical_share_features,
    add_combined_usage_features,
    add_role_share_flags,
    drop_leakage_columns,
)
from utils.feature.derived.market_features import add_market_features
from utils.feature.enrichment.asof_enrichment import (
    attach_asof_metadata,
    apply_snapshot_guards,
    load_and_build_asof_metadata,
)

# Labels and validation
from utils.feature.core.labels import DEFAULT_LABEL_VERSION, get_label_spec
from utils.feature.core.targets import validate_target_columns

# Leak guard and schema
from utils.feature.core.leak_guard import (
    DEFAULT_LEAK_POLICY,
    build_schema_snapshot,
    enforce_leak_guard,
    write_schema_snapshot,
)
from utils.feature.enrichment.asof import (
    decision_cutoff_horizons,
    decision_cutoff_override,
    get_decision_cutoff_hours,
    fallback_cutoff_hours,
    drop_missing_snapshots_enabled,
)
from utils.feature.enrichment.asof_metadata import build_asof_metadata, load_asof_metadata

# Paths and constants
from utils.general.constants import LEAK_PRONE_COLUMNS, NFL_TARGET_COLUMNS, format_cutoff_label
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

ENABLE_WEATHER_FEATURES = True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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


def _load_player_game_data() -> pl.LazyFrame:
    """Load player-game data from partitioned parquet files."""
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
        player_game_scan = player_game_scan.with_columns(
            [pl.col(col).cast(pl.Utf8) for col in pg_categorical_cols]
        )

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
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
    ])

    return player_game_scan


def _load_drive_features(target_players: pl.DataFrame) -> pl.DataFrame:
    """Load and aggregate drive-level features."""
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
        drive_scan = drive_scan.with_columns(
            [pl.col(col).cast(pl.Utf8) for col in drive_categorical_cols]
        )
    drive_scan = drive_scan.with_columns([
            pl.col("team").cast(pl.Utf8) if "team" in drive_schema.names() else pl.lit(None).cast(pl.Utf8).alias("team"),
            pl.col("player_id").cast(pl.Utf8) if "player_id" in drive_schema.names() else pl.lit(None).cast(pl.Utf8).alias("player_id"),
            pl.col("game_id").cast(pl.Utf8) if "game_id" in drive_schema.names() else pl.lit(None).cast(pl.Utf8).alias("game_id"),
            pl.col("season").cast(pl.Int32) if "season" in drive_schema.names() else pl.lit(None).cast(pl.Int32).alias("season"),
            pl.col("week").cast(pl.Int32) if "week" in drive_schema.names() else pl.lit(None).cast(pl.Int32).alias("week"),
    ])
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

    # Build scaffold from player-game data for dense history
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
        pl.col("game_id").cast(pl.Utf8),
    ])
    
    scaffold = game_scan.filter(
        pl.col("player_id").is_in(target_players.get_column("player_id"))
    ).select(
        ["season", "week", "game_id", "team", "player_id", "game_date"]
    ).collect(streaming=True)
    
    join_keys = ["season", "week", "game_id", "team", "player_id"]
    
    if not scaffold.is_empty():
        scaffold = scaffold.unique(subset=join_keys)

    if not drive_features.is_empty():
        drive_features = drive_features.with_columns([
            pl.col("player_id").cast(pl.Utf8),
            pl.col("game_id").cast(pl.Utf8),
            pl.col("team").cast(pl.Utf8),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
        ])
        drive_features = scaffold.join(
            drive_features,
            on=join_keys,
            how="left",
            suffix="_drive"
        )
        if "game_date_drive" in drive_features.columns:
             drive_features = drive_features.with_columns(
                 pl.col("game_date").fill_null(pl.col("game_date_drive"))
             ).drop("game_date_drive")
        
        agg_cols = [
            "drive_count", "drive_touch_drives", "drive_td_drives", "drive_total_yards",
            "drive_red_zone_drives", "drive_goal_to_go_drives"
        ]
        existing_aggs = [c for c in agg_cols if c in drive_features.columns]
        if existing_aggs:
             drive_features = drive_features.with_columns([
                 pl.col(c).fill_null(0) for c in existing_aggs
             ])
    else:
        drive_features = scaffold
        agg_cols = ["drive_count", "drive_touch_drives", "drive_td_drives", "drive_total_yards"]
        drive_features = drive_features.with_columns([
            pl.lit(0.0).cast(pl.Float32).alias(c) for c in agg_cols
        ])

    return finalize_drive_history_features(drive_features)


def _normalize_column_types(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize column types for consistency."""
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

    return df


def _cast_numeric_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Cast numeric columns to appropriate types."""
    numeric_casts = []
    
    if "depth_chart_order" in df.columns:
        numeric_casts.append(pl.col("depth_chart_order").cast(pl.Int16))
    for col in ("is_home", "game_start_hour_utc", "game_day_of_week"):
        if col in df.columns:
            numeric_casts.append(pl.col(col).cast(pl.Int8))
    for col in (
        "injury_is_listed",
        "injury_practice_dnp_count",
        "injury_practice_limited_count",
        "injury_practice_full_count",
    ):
        if col in df.columns:
            numeric_casts.append(pl.col(col).cast(pl.Int8))
    
    if numeric_casts:
        df = df.with_columns(numeric_casts)
    
    return df


def _add_injury_availability_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add injury availability probability features."""
    if (
        "injury_inactive_probability" in df.columns
        and "injury_inactive_probability_source" in df.columns
        and "injury_inactive_probability_model" not in df.columns
    ):
        df = df.with_columns(
            pl.when(pl.col("injury_inactive_probability_source") == "model")
            .then(pl.col("injury_inactive_probability"))
            .otherwise(pl.lit(None))
            .alias("injury_inactive_probability_model")
        )

    if "injury_inactive_probability_model" in df.columns:
        df = df.with_columns([
                pl.col("injury_inactive_probability_model")
                .fill_null(0.0)
                .cast(pl.Float32)
                .clip(0.0, 1.0)
                .alias("availability_inactive_prob"),
                (
                    pl.col("injury_inactive_probability_model")
                    .fill_null(0.0)
                    .cast(pl.Float32)
                    .clip(0.0, 1.0)
                    * (1.0 - pl.col("injury_inactive_probability_model")
                               .fill_null(0.0)
                               .cast(pl.Float32)
                               .clip(0.0, 1.0))
                )
                .alias("availability_uncertainty"),
        ])
    
    return df


def _process_weather_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Process and normalize weather columns."""
    weather_cols = [col for col in df.columns if col.startswith("weather_")]
    if not weather_cols:
        return df
    
        drop_weather_cols = [
            col
            for col in weather_cols
            if col.endswith("_ts")
            or col.endswith("_source_detail")
            or col in {"weather_conditions", "weather_precip_type"}
        ]
        if drop_weather_cols:
            df = df.drop(drop_weather_cols, strict=False)
    
        weather_numeric_cols = [
            col
            for col in weather_cols
            if col not in drop_weather_cols
            and any(
            df.schema.get(col) == dtype
                for dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32)
            )
        ]
        if weather_numeric_cols:
            df = df.with_columns(
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
            df = df.with_columns(
                [pl.col(col).cast(pl.Int8) for col in weather_flag_cols]
            )
    
    df = append_weather_context_flags(df, roof_col="roof")
    
    return df


def _drop_td_history_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop legacy team/position TD rate columns."""
    td_hist_cols = [
        c
        for c in df.columns
        if c.startswith("team_pos_") or c.startswith("opp_pos_")
    ]
    if td_hist_cols:
        df = df.drop(td_hist_cols)
    return df


def _apply_leak_guard(
    df: pl.DataFrame,
    *,
    label_version: str,
    cutoff_label: str,
) -> pl.DataFrame:
    """Apply leak guard and write schema snapshot."""
    spec = get_label_spec(label_version)
    allow_exact = {spec.primary, *spec.labels.keys(), *spec.aliases.keys()}
    leak_allowlist = allow_exact | set(NFL_TARGET_COLUMNS)
    
    drop_before_guard = [
        col
        for col in LEAK_PRONE_COLUMNS
        if col in df.columns and col not in leak_allowlist
    ]
    if drop_before_guard:
        logger.info(
            "    Dropping %d leak-prone columns prior to schema validation",
            len(drop_before_guard),
        )
        df = df.drop(drop_before_guard, strict=False)

    df, leak_result = enforce_leak_guard(
        df,
        policy=DEFAULT_LEAK_POLICY,
        allow_exact=allow_exact,
        drop_banned=True,
        drop_non_allowlisted=False,
        raise_on_banned=True,
    )

    schema = build_schema_snapshot(
        df,
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
    logger.info("    Schema snapshot written → %s", schema_path)
    
    return df


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
    """Run full feature pipeline up to *inclusive* `end_date` for a single cutoff window."""
    cutoff_label = cutoff_label or "default"
    target_output_path = output_path or FINAL_OUT
    primary_output = primary_output_path or target_output_path
    current_cutoff_hours = float(get_decision_cutoff_hours())
    label_version = label_version or DEFAULT_LABEL_VERSION
    
    # Determine date range
    clean_min_date, clean_max_date = (None, None)
    if start_date is None or end_date is None:
        clean_min_date, clean_max_date = _get_clean_date_bounds()

    end_date = end_date or clean_max_date or date.today()
    start_date = start_date or clean_min_date or end_date

    if clean_max_date and end_date > clean_max_date:
        logger.info(
            "Clamping end_date %s to latest cleaned date %s to avoid empty slices",
            end_date,
            clean_max_date,
        )
        end_date = clean_max_date

    if start_date > end_date:
        raise ValueError("start_date must be ≤ end_date")

    logger.info(
        "Feature pipeline: generating features %s → %s in %s-day chunks",
        start_date, end_date, chunk_days,
    )

    # ------------------------------------------------------------------
    # Stage 1: Build intermediate levels
    # ------------------------------------------------------------------
    cur = start_date
    if recompute_intermediate:
        while cur <= end_date:
            chunk_end = min(cur + timedelta(days=chunk_days - 1), end_date)
            logger.info("🔹  Processing chunk %s → %s", cur, chunk_end)

            build_play_level(start_date=cur, end_date=chunk_end)
            build_opponent_splits(start_date=cur, end_date=chunk_end)
            build_player_drive_level(start_date=cur, end_date=chunk_end)
            build_player_game_level(start_date=cur, end_date=chunk_end, label_version=label_version)

            cur += timedelta(days=chunk_days)

    # ------------------------------------------------------------------
    # Stage 2: Build daily cache for rolling windows
    # ------------------------------------------------------------------
    if build_rolling:
        logger.info("🔹 Building daily totals cache for rolling windows...")
        build_daily_cache_range(start_date=start_date, end_date=end_date, level="game")
        logger.info("✅  Daily cache built")

    # ------------------------------------------------------------------
    # Stage 3: Load player-game data
    # ------------------------------------------------------------------
    logger.info("Building final feature matrix from player-game aggregations...")
    
    player_game_scan = _load_player_game_data()
    df = player_game_scan.collect(streaming=True)
    
    # Clean up duplicate columns from upstream joins
    right_cols = [c for c in df.columns if c.endswith("_right")]
    if right_cols:
        logger.info("Dropping %d duplicate join columns from player-game data", len(right_cols))
        df = df.drop(right_cols)

    # ------------------------------------------------------------------
    # Stage 4: Attach as-of metadata
    # ------------------------------------------------------------------
    logger.info("Attaching as-of metadata...")
    seasons_available = df.get_column("season").unique().to_list() if "season" in df.columns else []
    game_ids = df.get_column("game_id").unique().to_list() if "game_id" in df.columns else []
    
    asof_meta = load_and_build_asof_metadata(
        seasons=seasons_available or [int(date.today().year)],
        game_ids=game_ids,
    )
    
    if not asof_meta.is_empty():
        df = attach_asof_metadata(
            df,
            asof_meta=asof_meta,
            drop_missing_snapshots=drop_missing_snapshots_enabled(),
        )
        df = apply_snapshot_guards(df, audit_dir=FEATURE_AUDIT_DIR)

    # ------------------------------------------------------------------
    # Stage 5: Join drive-level features
    # ------------------------------------------------------------------
    target_players = df.select("player_id").unique()
    drive_features = _load_drive_features(target_players)
    
    if not drive_features.is_empty():
        df = df.join(
            drive_features.drop("game_date"),
            on=["season", "week", "game_id", "player_id"],
            how="left",
        )
        PLAYER_DRIVE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        drive_features.write_parquet(PLAYER_DRIVE_HISTORY_PATH, compression="zstd")

    if df.is_empty():
        logger.warning("No player-game rows found for final matrix; skipping.")
        return pl.DataFrame()

    # ------------------------------------------------------------------
    # Stage 6: Normalize and cast columns
    # ------------------------------------------------------------------
    df = _normalize_column_types(df)
    df = _cast_numeric_columns(df)
    df = _add_injury_availability_features(df)

    # ------------------------------------------------------------------
    # Stage 7: Add feature groups (using utility modules)
    # ------------------------------------------------------------------
    
    # Position and specialist features
    df = add_position_group(df)
    df = add_specialist_role_flags(df)

    # Team and offense context
    logger.info("Enriching team context features...")
    df, team_context_history = add_team_context_features(df, return_history=True)
    TEAM_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_context_history.write_parquet(TEAM_CONTEXT_PATH, compression="zstd")

    logger.info("Enriching offense situational context (OC/QB)...")
    df = add_offense_context_features_training(df, history_path=OFF_CONTEXT_PATH)
            
    # Weather features
    if ENABLE_WEATHER_FEATURES:
        logger.info("Enriching weather forecast features...")
        df = add_weather_forecast_features_training(df, cutoff_column="decision_cutoff_ts")
    df = _process_weather_columns(df)

    # Odds features
    logger.info("Enriching odds features (NFL)...")
    rows_before_odds = df.height
    df = add_nfl_odds_features_to_df(
        df,
        player_col="player_name",
        allow_schedule_fallback=False,
        drop_schedule_rows=True,
    )
    rows_after_odds = df.height
    logger.info(
        "Odds features attached; retained %d/%d rows (%.2f%%) with pre-cutoff snapshots",
        rows_after_odds,
        rows_before_odds,
        (rows_after_odds / rows_before_odds * 100.0) if rows_before_odds else 0.0,
    )
    
    if "odds_schedule_fallback" not in df.columns:
        df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback"))
    if "odds_anytime_td_price" in df.columns:
        df = df.drop("odds_anytime_td_price", strict=False)

    # Validate targets
    df = validate_target_columns(df, strict=False)

    # TD rate history
    df = _drop_td_history_columns(df)
    df = attach_td_rate_history_features(df)

    # Rolling window features
    if build_rolling:
        logger.info("🔹 Computing rolling window features...")
        rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in df.columns]
        if rolling_stats:
            df = add_rolling_features(
                df,
                level="game",
                stats=rolling_stats,
                windows=ROLLING_WINDOWS,
                contexts=ROLLING_CONTEXTS,
                date_col="game_date",
                player_col="player_id",
                opponent_col="opponent",
            )
            rolling_cols = [c for c in df.columns if "g_" in c or "season" in c]
            logger.info(f"✅  Added {len(rolling_cols)} rolling features")

    # Basic rolling snap features (no dependencies on historical shares)
    df = add_rolling_snap_features(df)
    df = add_role_stability_features(df)
    df = add_market_odds_flag(df)

    # Historical share features (must come before expected snap features)
    df = add_historical_share_features(df)
    df = add_combined_usage_features(df)

    # Expected snap features (depends on historical shares for RZ/GL touches)
    df = add_expected_snap_features(df)

    # Market features (implied totals, pace, interactions)
    df = add_market_features(df)

    # Usage helper features
    df = add_usage_helper_features(df)

    # MoE position-specific features
    df = add_moe_position_features(df)
    
    # Catch rate features
    df = add_catch_rate_features(df)

    # Target depth features
    df = add_target_depth_features(df)

    # Role share flags
    df = add_role_share_flags(df)
    df = drop_leakage_columns(df)

    # ------------------------------------------------------------------
    # Stage 8: Apply leak guard and write output
    # ------------------------------------------------------------------
    df = _apply_leak_guard(df, label_version=label_version, cutoff_label=cutoff_label)

    df = df.with_columns(
        pl.lit(current_cutoff_hours).cast(pl.Float32).alias("decision_horizon_hours")
    )

    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target_output_path, compression="zstd")

    if primary_output != target_output_path:
        primary_output.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(primary_output, compression="zstd")

    logger.info("✅  Final feature matrix (%s cutoff) written → %s", cutoff_label, target_output_path)
    if primary_output != target_output_path:
        logger.info("    Primary feature matrix updated → %s", primary_output)
    logger.info("    Rows: %d, Columns: %d", len(df), len(df.columns))
    logger.info("    Date range: %s to %s", df["game_date"].min(), df["game_date"].max())
    
    spec = get_label_spec(label_version)
    label_candidates = {spec.primary, "anytime_td"}
    for label_col in label_candidates:
        if label_col in df.columns:
            td_count = df[label_col].sum()
            td_rate = td_count / len(df) * 100
            logger.info(
                "    Label %s rate: %.2f%% (%d/%d)",
                label_col,
                td_rate,
                td_count,
                len(df),
            )
    
    return df


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
        logger.info("===== Building feature matrix for cutoff %.2f hours (%s) =====", hours, label)
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
    logger.info(
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
    player_game_scan = player_game_scan.with_columns([
            pl.col("player_id").cast(pl.Utf8),
            pl.col("team").cast(pl.Utf8),
            pl.col("opponent").cast(pl.Utf8),
            pl.col("game_date").cast(pl.Date),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
    ])
    if end_date is not None:
        player_game_scan = player_game_scan.filter(pl.col("game_date") <= pl.lit(end_date))

    df_player_game = player_game_scan.collect(streaming=True)
    if df_player_game.is_empty():
        logger.warning("No player-game rows available; skipping context refresh.")
        return

    df_player_game = df_player_game.with_columns(
        pl.col("game_date").cast(pl.Datetime("ms"))
    )

    logger.info("Recomputing team context history...")
    team_history = compute_team_context_history(df_player_game)
    TEAM_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_history.write_parquet(TEAM_CONTEXT_PATH, compression="zstd")
    logger.info(
        "Team context history updated → %s (%d rows)",
        TEAM_CONTEXT_PATH,
        len(team_history),
    )

    logger.info("Recomputing offense context history...")
    seasons = df_player_game.get_column("season").unique().to_list()
    _, offense_history = _append_offense_context_columns(df_player_game, seasons)
    OFF_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    offense_history.write_parquet(OFF_CONTEXT_PATH, compression="zstd")
    logger.info(
        "Offense context history updated → %s (%d rows)",
        OFF_CONTEXT_PATH,
        len(offense_history),
    )

    logger.info("Recomputing drive history...")
    drive_scan = pl.scan_parquet(
        str(PLAYER_DRIVE_DIR / "season=*/week=*/part.parquet"),
        glob=True,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    drive_scan = drive_scan.with_columns([
            pl.col("player_id").cast(pl.Utf8),
            pl.col("team").cast(pl.Utf8),
            pl.col("game_id").cast(pl.Utf8),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("game_date").cast(pl.Datetime("ms")),
    ])
    if end_date is not None:
        drive_scan = drive_scan.filter(
            pl.col("game_date").cast(pl.Date) <= pl.lit(end_date)
        )
    drive_df = drive_scan.collect(streaming=True)
    
    if drive_df.is_empty():
        logger.warning("No player-drive rows available; drive history not refreshed.")
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
        drive_agg = finalize_drive_history_features(drive_agg)
        if drive_agg.is_empty():
            logger.warning("Aggregated drive history empty; nothing written.")
        else:
            PLAYER_DRIVE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            drive_agg.write_parquet(PLAYER_DRIVE_HISTORY_PATH, compression="zstd")
            logger.info(
                "Drive history updated → %s (%d rows)",
                PLAYER_DRIVE_HISTORY_PATH,
                len(drive_agg),
            )

    logger.info("Refreshing as-of metadata...")
    try:
        build_asof_metadata(seasons, force=True)
    except Exception as exc:
        logger.warning("Failed to rebuild as-of metadata: %s", exc)

    logger.info("Context histories refreshed.")
