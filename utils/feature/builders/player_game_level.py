"""Player-game level aggregation for NFL predictions.

Aggregates play-by-play data to player-game level for all skill positions.
Each player gets one row per game with their total stats.

Target columns created:
- anytime_td_offense (binary): Did player score an offensive TD (rush/rec/pass thrown)?
- anytime_td_all (binary): Did player score any TD (including defensive/special teams if present)?
- anytime_td_rush / anytime_td_rec / anytime_td_pass_thrown: Type-specific TD flags.
- anytime_td (binary): Legacy alias → current label version primary.
- td_count_offense / td_count_all: TD counts aligned to label semantics.
- td_count (int): Legacy alias for the primary td_count.
- passing_td (int): Number of passing TDs (for QBs)
- receiving_yards (float): Total receiving yards
- rushing_yards (float): Total rushing yards
- passing_yards (float): Total passing yards
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import datetime as dt
import logging
from typing import Any, Dict, List, Optional

import polars as pl

try:  # Optional heavy dependencies
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None
    pd = None

from utils.collect.arrival_log import log_feed_arrivals
from utils.collect.nfl_schedules import get_schedule
from utils.feature.enrichment.asof import decision_cutoff_hours_default, fallback_cutoff_hours, get_decision_cutoff_hours, get_fallback_cutoff_hours
from utils.feature.builders.player_drive_level import _load_play_level_data
from utils.feature.core.labels import DEFAULT_LABEL_VERSION, compute_td_labels

# Import from refactored modules
from utils.feature.builders.data_loaders import (
    load_rosters_for_years as _load_rosters_for_years,
    load_roster_snapshots_for_years as _load_roster_snapshots_for_years,
    load_nflverse_depth_charts as _load_nflverse_depth_charts,
    join_nflverse_depth_charts as _join_nflverse_depth_charts,
    load_injuries_for_years as _load_injuries_for_years,
    load_injury_transactions_for_years as _load_injury_transactions_for_years,
    normalize_player_name as _normalize_player_name,
    normalize_player_name_expr as _normalize_player_name_expr,
    load_snap_counts_for_years as _load_snap_counts_for_years,
    load_player_market_features as _load_player_market_features,
    load_qb_profile_features as _load_qb_profile_features,
    load_travel_calendar_features as _load_travel_calendar_features,
)
from utils.feature.builders.injury_enrichment import (
    compute_injury_history_rates as _compute_injury_history_rates,
    apply_injury_availability_model as _apply_injury_availability_model,
    INJURY_DEFAULTS,
    INJURY_NUMERIC_COLS,
    INJURY_STRING_COLS,
    fill_missing_injury_columns,
    fill_injury_string_columns,
    compute_roster_deltas,
)
from utils.feature.builders.player_aggregation import (
    aggregate_passers as _aggregate_passers,
    aggregate_rushers as _aggregate_rushers,
    aggregate_receivers as _aggregate_receivers,
    merge_multi_role_players as _merge_multi_role_players,
    METADATA_COLUMNS,
)
from utils.general.paths import (
    PLAYER_MARKET_PROCESSED_DIR,
    QB_PROFILE_DIR,
    TRAVEL_CALENDAR_DIR,
    OPPONENT_SPLIT_DIR,
    PLAY_BY_WEEK_DIR as PLAY_DIR,
    PLAYER_GAME_BY_WEEK_DIR as PLAYER_GAME_DIR,
    ROSTER_CACHE_DIR,
    ROSTER_SNAPSHOT_DIR,
    INJURY_CACHE_DIR,
    INJURY_TRANSACTION_CACHE_DIR,
    SNAP_CACHE_DIR,
    SCHEDULE_CACHE_PATH,
)
from utils.feature.builders.opponent_splits import load_rolling_opponent_splits

try:
    from nfl_data_py import import_weekly_rosters, import_injuries, import_snap_counts
except Exception:  # pragma: no cover - optional dependency
    import_weekly_rosters = None
    import_injuries = None
    import_snap_counts = None

logger = logging.getLogger(__name__)

# Note: Paths imported from utils.general.paths (single source of truth)
# Constants METADATA_COLUMNS imported from player_aggregation module
REQUIRED_ROSTER_COLS = ['season', 'week', 'team', 'player_id']

PLAYER_MARKET_FLOAT_COLUMNS = [
    "market_anytime_td_prob",
    "market_anytime_td_consensus",
    "market_anytime_td_spread",
    "market_anytime_td_book_count",
    "market_anytime_td_prob_2h",
    "market_anytime_td_consensus_2h",
    "market_anytime_td_spread_2h",
    "market_anytime_td_book_count_2h",
    "market_anytime_td_prob_6h",
    "market_anytime_td_consensus_6h",
    "market_anytime_td_spread_6h",
    "market_anytime_td_book_count_6h",
    "market_anytime_td_prob_24h",
    "market_anytime_td_consensus_24h",
    "market_anytime_td_spread_24h",
    "market_anytime_td_book_count_24h",
    "market_anytime_td_prob_open",
    "market_anytime_td_consensus_open",
    "market_anytime_td_spread_open",
    "market_anytime_td_book_count_open",
    "market_anytime_td_delta",
    "market_anytime_td_delta_24h",
    "market_anytime_td_delta_6h",
    "market_anytime_td_delta_2h",
    "market_player_tds_line",
    "market_player_tds_prob_over",
    "market_player_tds_consensus_over",
    "market_anytime_td_residual",
    "market_anytime_td_prob_l3",
]
PLAYER_MARKET_COLUMNS = PLAYER_MARKET_FLOAT_COLUMNS + [
    "market_anytime_td_missing",
]


def _collect_partition_paths(base_dir: Path, seasons: list[int]) -> list[Path]:
    paths: List[Path] = []
    for season in sorted({int(s) for s in seasons}):
        season_dir = base_dir / f"season={season}"
        if not season_dir.exists():
            continue
        paths.extend(sorted(season_dir.glob("week=*/part.parquet")))
    return paths


SKILL_POSITIONS = {"WR", "TE", "RB", "HB", "FB"}
ALIGNMENT_ORDER = ["slot", "wide", "inline", "backfield"]
SLOT_ROUTE_CUES = {
    "SLANT",
    "IN/DIG",
    "SHALLOW CROSS/DRAG",
    "SCREEN",
    "QUICK OUT",
    "HITCH/CURL",
    "TEXAS/ANGLE",
}
PRE_SNAP_BASE_COLUMNS = [
    "ps_route_participation_plays",
    "ps_team_dropbacks",
    "ps_tracking_team_dropbacks",
    "ps_tracking_has_game_data",
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
]
PRE_SNAP_ROLLING_TARGETS = [
    "ps_route_participation_pct",
    "ps_route_participation_plays",
    "ps_targets_total",
    "ps_targets_slot_count",
    "ps_targets_wide_count",
    "ps_targets_inline_count",
    "ps_targets_backfield_count",
    "ps_scripted_touch_share",
    "ps_scripted_touches",
    "ps_total_touches",
    "ps_targets_slot_share",
    "ps_targets_wide_share",
    "ps_targets_inline_share",
    "ps_targets_backfield_share",
    "ps_tracking_team_dropbacks",
    "ps_tracking_has_game_data",
]


def _ps_game_alias(name: str) -> str:
    """Return the ps_game_ alias for a ps_* column."""
    if not name.startswith("ps_"):
        return name
    return f"ps_game_{name[len('ps_'):]}"
def _safe_split(col: pl.Expr) -> pl.Expr:
    return (
        pl.when(col.is_not_null())
        .then(col.str.split(";"))
        .otherwise(pl.lit([]))
    )


def _compute_pre_snap_usage(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame()

    # Ensure critical situational columns exist for scripted-play heuristics.
    numeric_defaults = {
        "game_seconds_remaining": pl.Float32,
        "half_seconds_remaining": pl.Float32,
        "quarter_seconds_remaining": pl.Float32,
        "score_differential": pl.Float32,
    }
    for col, dtype in numeric_defaults.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    if "no_huddle" not in df.columns:
        df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("no_huddle"))
    if "qtr" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Int8).alias("qtr"))
    if "down" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Int8).alias("down"))

    base_cols = [
        "season",
        "week",
        "game_id",
        "posteam",
        "play_id",
        "order_sequence",
        "qtr",
        "down",
        "game_seconds_remaining",
        "half_seconds_remaining",
        "quarter_seconds_remaining",
        "score_differential",
        "no_huddle",
        "qb_dropback",
        "target",
        "carry",
        "receiver_player_id",
        "rusher_player_id",
        "route",
        "pass_location",
        "offense_players",
        "offense_positions",
    ]
    available_cols = [c for c in base_cols if c in df.columns]
    if not available_cols:
        return pl.DataFrame()

    working = df.select(available_cols).with_columns(
        [
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("game_id").cast(pl.Utf8),
            pl.col("posteam").cast(pl.Utf8).str.to_uppercase().alias("team"),
            pl.col("offense_players").cast(pl.Utf8),
            pl.col("offense_positions").cast(pl.Utf8),
            pl.col("route").cast(pl.Utf8),
            pl.col("pass_location").cast(pl.Utf8),
            pl.col("qtr").cast(pl.Int8),
            pl.col("down").cast(pl.Int8),
            pl.col("game_seconds_remaining").cast(pl.Float32),
            pl.col("half_seconds_remaining").cast(pl.Float32),
            pl.col("quarter_seconds_remaining").cast(pl.Float32),
            pl.col("score_differential").cast(pl.Float32),
            pl.col("no_huddle").cast(pl.Int8),
        ]
    )

    pass_plays = working.filter(pl.col("qb_dropback").fill_null(0) == 1)
    pass_plays = pass_plays.filter(
        pl.col("team").is_not_null()
        & pl.col("offense_players").is_not_null()
        & pl.col("offense_positions").is_not_null()
    )
    if pass_plays.is_empty():
        return pl.DataFrame()

    pass_lists = pass_plays.with_columns(
        [
            _safe_split(pl.col("offense_players")).alias("off_player_ids"),
            _safe_split(pl.col("offense_positions")).alias("off_positions_list"),
        ]
    ).filter(
        pl.col("off_player_ids").list.len() == pl.col("off_positions_list").list.len()
    )

    if pass_lists.is_empty():
        return pl.DataFrame()

    pass_exploded = (
        pass_lists
        .explode(["off_player_ids", "off_positions_list"])
        .with_columns(
            [
                pl.col("off_player_ids").cast(pl.Utf8).alias("player_id"),
                pl.col("off_positions_list")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.to_uppercase()
                .alias("off_position"),
            ]
        )
        .filter(
            pl.col("player_id").is_not_null()
            & pl.col("off_position").is_in(list(SKILL_POSITIONS))
        )
    )

    if pass_exploded.is_empty():
        return pl.DataFrame()

    route_participation = (
        pass_exploded.select(
            [
                "season",
                "week",
                "team",
                "game_id",
                "play_id",
                "player_id",
            ]
        )
        .unique()
        .group_by(["season", "week", "team", "player_id"])
        .agg(pl.len().alias("ps_route_participation_plays"))
    )

    dropbacks = (
        pass_lists.select(["season", "week", "team", "game_id"])
        .unique()
        .group_by(["season", "week", "team"])
        .agg(pl.len().alias("ps_team_dropbacks"))
    )

    route_participation = route_participation.join(
        dropbacks, on=["season", "week", "team"], how="left"
    ).with_columns(
        [
            pl.when(pl.col("ps_team_dropbacks") > 0)
            .then(pl.col("ps_route_participation_plays") / pl.col("ps_team_dropbacks"))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("ps_route_participation_pct")
        ]
    )

    targets = pass_lists.filter(
        (pl.col("target").fill_null(0) == 1)
        & pl.col("receiver_player_id").is_not_null()
    ).with_columns(
        [
            pl.col("receiver_player_id").cast(pl.Utf8).alias("player_id"),
            pl.col("route").fill_null("").str.strip_chars().str.to_uppercase(),
            pl.col("pass_location").fill_null("").str.to_uppercase(),
        ]
    )

    target_alignment = targets.join(
        pass_exploded.select(
            ["season", "week", "team", "game_id", "play_id", "player_id", "off_position"]
        ),
        left_on=["season", "week", "team", "game_id", "play_id", "player_id"],
        right_on=["season", "week", "team", "game_id", "play_id", "player_id"],
        how="inner",
    )

    if not target_alignment.is_empty():
        alignment_expr = (
            pl.when(pl.col("off_position") == "TE")
            .then(pl.lit("inline"))
            .when(pl.col("off_position").is_in(["RB", "HB", "FB"]))
            .then(pl.lit("backfield"))
            .when(pl.col("off_position") == "WR")
            .then(
                pl.when(
                    (pl.col("pass_location") == "MIDDLE")
                    | pl.col("route").is_in(list(SLOT_ROUTE_CUES))
                )
                .then(pl.lit("slot"))
                .otherwise(pl.lit("wide"))
            )
            .otherwise(pl.lit("wide"))
        )

        target_alignment = target_alignment.with_columns(
            alignment_expr.alias("ps_alignment")
        )

        target_counts = target_alignment.group_by(
            ["season", "week", "team", "player_id", "ps_alignment"]
        ).agg(pl.len().alias("target_count"))

        alignment_pivot = target_counts.pivot(
            columns="ps_alignment",
            values="target_count",
            index=["season", "week", "team", "player_id"],
        ).with_columns(
            [
                pl.col(col).fill_null(0).cast(pl.Float32).alias(f"ps_targets_{col}_count")
                for col in ALIGNMENT_ORDER
                if col in target_counts["ps_alignment"].unique().to_list()
            ]
        )

        totals = target_alignment.group_by(
            ["season", "week", "team", "player_id"]
        ).agg(pl.len().alias("ps_targets_total"))

        alignment_stats = totals.join(
            alignment_pivot, on=["season", "week", "team", "player_id"], how="left"
        )

        for alignment in ALIGNMENT_ORDER:
            count_col = f"ps_targets_{alignment}_count"
            if count_col not in alignment_stats.columns:
                alignment_stats = alignment_stats.with_columns(
                    pl.lit(0).cast(pl.Float32).alias(count_col)
                )

        alignment_stats = alignment_stats.with_columns(
            [
                pl.when(pl.col("ps_targets_total") > 0)
                .then(pl.col(f"ps_targets_{alignment}_count") / pl.col("ps_targets_total"))
                .otherwise(0.0)
                .cast(pl.Float32)
                .alias(f"ps_targets_{alignment}_share")
                for alignment in ALIGNMENT_ORDER
            ]
        )
        drop_raw = [col for col in ALIGNMENT_ORDER if col in alignment_stats.columns]
        if drop_raw:
            alignment_stats = alignment_stats.drop(drop_raw)
    else:
        alignment_stats = pl.DataFrame()

    # Scripted plays (early-drive heuristic)
    scripted_source = (
        df.select(
            [
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                pl.col("game_id").cast(pl.Utf8),
                pl.col("posteam").cast(pl.Utf8).str.to_uppercase().alias("team"),
                "play_id",
                "qtr",
                "down",
                "game_seconds_remaining",
                "half_seconds_remaining",
                "quarter_seconds_remaining",
                "score_differential",
                "no_huddle",
                "target",
                "carry",
                "receiver_player_id",
                "rusher_player_id",
                "order_sequence",
            ]
        )
        .filter(pl.col("team").is_not_null())
        .sort(["game_id", "team", "order_sequence", "play_id"])
        .with_columns(
            pl.col("order_sequence")
            .rank(method="ordinal")
            .over(["game_id", "team"])
            .cast(pl.Int32)
            .sub(1)
            .alias("offense_play_rank")
        )
        .with_columns(
            [
                pl.when(pl.col("game_seconds_remaining").is_not_null())
                .then(3600.0 - pl.col("game_seconds_remaining"))
                .when(pl.col("half_seconds_remaining").is_not_null())
                .then(1800.0 - pl.col("half_seconds_remaining"))
                .otherwise(None)
                .alias("__game_elapsed"),
                (
                    (pl.col("qtr").is_in([2, 4]))
                    & (pl.col("quarter_seconds_remaining").fill_null(900.0) <= 120.0)
                ).alias("__two_minute_like"),
                (
                    (pl.col("no_huddle").fill_null(0) == 1)
                    & (pl.col("quarter_seconds_remaining").fill_null(900.0) < 240.0)
                ).alias("__hurry_up"),
                pl.col("down").fill_null(1).is_in([1, 2]).alias("__early_down"),
            ]
        )
        .with_columns(
            (
                (pl.col("offense_play_rank") < 20)
                & (
                    pl.when(pl.col("__game_elapsed").is_not_null())
                    .then(pl.col("__game_elapsed") <= 1200.0)
                    .otherwise(True)
                )
                & (~pl.col("__two_minute_like"))
                & (~pl.col("__hurry_up"))
                & pl.col("__early_down")
            )
            .cast(pl.Int8)
            .alias("is_scripted_play")
        )
        .drop(["__game_elapsed", "__two_minute_like", "__hurry_up", "__early_down"])
    )

    target_touches = (
        scripted_source.filter(pl.col("target").fill_null(0) == 1)
        .filter(pl.col("receiver_player_id").is_not_null())
        .select(
            [
                "season",
                "week",
                "team",
                pl.col("receiver_player_id").cast(pl.Utf8).alias("player_id"),
                "is_scripted_play",
                pl.lit(1).cast(pl.Int8).alias("is_target_touch"),
                pl.lit(0).cast(pl.Int8).alias("is_carry_touch"),
            ]
        )
    )

    rush_touches = (
        scripted_source.filter(pl.col("carry").fill_null(0) == 1)
        .filter(pl.col("rusher_player_id").is_not_null())
        .select(
            [
                "season",
                "week",
                "team",
                pl.col("rusher_player_id").cast(pl.Utf8).alias("player_id"),
                "is_scripted_play",
                pl.lit(0).cast(pl.Int8).alias("is_target_touch"),
                pl.lit(1).cast(pl.Int8).alias("is_carry_touch"),
            ]
        )
    )

    touches = pl.concat([target_touches, rush_touches], how="diagonal_relaxed")

    if not touches.is_empty():
        scripted_stats = touches.group_by(
            ["season", "week", "team", "player_id"]
        ).agg(
            [
                pl.len().cast(pl.Float32).alias("ps_total_touches"),
                pl.col("is_scripted_play").sum().cast(pl.Float32).alias(
                    "ps_scripted_touches"
                ),
                # Decompose scripted touches into target/carry components.
                (
                    (pl.col("is_scripted_play") * pl.col("is_target_touch"))
                    .sum()
                    .cast(pl.Float32)
                ).alias("ps_scripted_target_touches"),
                (
                    (pl.col("is_scripted_play") * pl.col("is_carry_touch"))
                    .sum()
                    .cast(pl.Float32)
                ).alias("ps_scripted_carry_touches"),
            ]
        )
        scripted_stats = scripted_stats.with_columns(
            [
                pl.when(pl.col("ps_total_touches") > 0)
                .then(pl.col("ps_scripted_touches") / pl.col("ps_total_touches"))
                .otherwise(0.0)
                .cast(pl.Float32)
                .alias("ps_scripted_touch_share"),
                pl.when(pl.col("ps_total_touches") > 0)
                .then(pl.col("ps_scripted_target_touches") / pl.col("ps_total_touches"))
                .otherwise(0.0)
                .cast(pl.Float32)
                .alias("ps_scripted_target_share"),
                pl.when(pl.col("ps_total_touches") > 0)
                .then(pl.col("ps_scripted_carry_touches") / pl.col("ps_total_touches"))
                .otherwise(0.0)
                .cast(pl.Float32)
                .alias("ps_scripted_carry_share"),
            ]
        )
    else:
        scripted_stats = pl.DataFrame()

    frames = []
    if not route_participation.is_empty():
        frames.append(route_participation)
    if not alignment_stats.is_empty():
        frames.append(alignment_stats)
    if not scripted_stats.is_empty():
        frames.append(scripted_stats)

    if not frames:
        return pl.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.join(
            frame,
            on=["season", "week", "team", "player_id"],
            how="outer",
            suffix="_ps",
        )
        for key in ["season", "week", "team", "player_id"]:
            suffixed = f"{key}_ps"
            if suffixed in merged.columns:
                merged = merged.with_columns(
                    pl.coalesce(pl.col(key), pl.col(suffixed)).alias(key)
                ).drop(suffixed)

    merged = merged.with_columns(
        [
            (
                pl.col("ps_team_dropbacks").cast(pl.Float32)
                if "ps_team_dropbacks" in merged.columns
                else pl.lit(0).cast(pl.Float32)
            ).alias("ps_tracking_team_dropbacks"),
            (
                pl.when(
                    (
                        pl.col("ps_team_dropbacks").fill_null(0) > 0
                    )
                    | (pl.col("ps_total_touches").fill_null(0) > 0)
                )
                .then(1)
                .otherwise(0)
                .cast(pl.Float32)
            ).alias("ps_tracking_has_game_data"),
        ]
    )

    numeric_cols = [
        col
        for col in merged.columns
        if col not in {"season", "week", "team", "player_id"}
    ]
    merged = merged.with_columns(
        [pl.col(col).fill_null(0).cast(pl.Float32).alias(col) for col in numeric_cols]
    )
    # Expose realized same-game stats under a clear namespace for downstream label use.
    game_alias_exprs = []
    for col in numeric_cols:
        if not col.startswith("ps_"):
            continue
        alias = _ps_game_alias(col)
        if alias not in merged.columns:
            game_alias_exprs.append(pl.col(col).alias(alias))
    if game_alias_exprs:
        merged = merged.with_columns(game_alias_exprs)
    return merged


def _load_schedule_team_rows(seasons: list[int]) -> pl.DataFrame:
    """Return one row per (team, game) with opponent metadata for the requested seasons."""
    if not seasons:
        return pl.DataFrame()

    if not SCHEDULE_CACHE_PATH.exists():
        try:
            from utils.collect.nfl_schedules import cache_schedules
            cache_schedules([int(s) for s in seasons])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Schedule cache missing at %s and refresh failed: %s", SCHEDULE_CACHE_PATH, exc)
            return pl.DataFrame()

    try:
        schedule = pl.read_parquet(SCHEDULE_CACHE_PATH)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read schedule cache: %s", exc)
        return pl.DataFrame()

    if schedule.is_empty():
        return pl.DataFrame()

    season_list = [int(s) for s in seasons]
    schedule = schedule.filter(pl.col("season").is_in(season_list))
    if schedule.is_empty():
        try:
            from utils.collect.nfl_schedules import cache_schedules
            cache_schedules(season_list)
            schedule = pl.read_parquet(SCHEDULE_CACHE_PATH).filter(pl.col("season").is_in(season_list))
        except Exception as exc:
            logger.warning("Schedule cache lacks seasons %s and refresh failed: %s", seasons, exc)
            return pl.DataFrame()
        if schedule.is_empty():
            return pl.DataFrame()

    schedule = schedule.with_columns(
        [
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("gameday").cast(pl.Date).alias("game_date"),
            pl.when(pl.col("game_type").is_not_null())
            .then(pl.col("game_type"))
            .otherwise(pl.lit("REG"))
            .alias("season_type"),
            pl.col("start_time_utc")
            .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
            .alias("game_start_utc")
            if "start_time_utc" in schedule.columns
            else pl.lit(None).cast(pl.Datetime(time_unit="ms", time_zone="UTC")).alias("game_start_utc"),
        ]
    )

    common_cols = [
        "season",
        "week",
        "game_id",
        "game_date",
        "season_type",
        "roof",
        "surface",
        "stadium",
        "game_start_utc",
    ]

    # Home team perspective
    home = schedule.select(
        common_cols
        + [
            pl.col("home_team").alias("team"),
            pl.col("away_team").alias("opponent"),
            pl.col("home_team").alias("home_team"),
            pl.col("away_team").alias("away_team"),
        ]
    ).with_columns(
        [
            pl.lit(1).cast(pl.Int8).alias("is_home"),
            pl.col("stadium").cast(pl.Utf8).alias("stadium_name"),
            pl.lit(None).cast(pl.Utf8).alias("stadium_key"),
            pl.lit(None).cast(pl.Utf8).alias("stadium_tz"),
        ]
    )

    # Away team perspective
    away = schedule.select(
        common_cols
        + [
            pl.col("away_team").alias("team"),
            pl.col("home_team").alias("opponent"),
            pl.col("home_team").alias("home_team"),
            pl.col("away_team").alias("away_team"),
        ]
    ).with_columns(
        [
            pl.lit(0).cast(pl.Int8).alias("is_home"),
            pl.col("stadium").cast(pl.Utf8).alias("stadium_name"),
            pl.lit(None).cast(pl.Utf8).alias("stadium_key"),
            pl.lit(None).cast(pl.Utf8).alias("stadium_tz"),
        ]
    )

    combined = pl.concat([home, away], how="diagonal_relaxed")
    combined = combined.with_columns(
        [
            pl.col("team").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            pl.col("opponent").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            pl.col("home_team").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            pl.col("away_team").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
        ]
    )

    return combined


def _append_zero_usage_players(
    df_merged: pl.DataFrame,
    seasons: list[int],
) -> pl.DataFrame:
    """Ensure each active/inactive rostered player appears even without recorded usage."""
    if not seasons:
        return df_merged

    try:
        roster_pl = _load_rosters_for_years(seasons)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Unable to load rosters for zero-usage scaffolding: %s", exc)
        return df_merged

    if roster_pl.is_empty():
        return df_merged

    schedule_team = _load_schedule_team_rows(seasons)
    if schedule_team.is_empty():
        return df_merged

    roster_pl = roster_pl.filter(pl.col("player_id").is_not_null())
    if roster_pl.is_empty():
        return df_merged

    roster_pl = roster_pl.with_columns(
        [
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("team").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            pl.col("player_id").cast(pl.Utf8),
            pl.col("player_name").cast(pl.Utf8),
            pl.col("status").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
        ]
    )

    if "position_group" in roster_pl.columns:
        roster_pl = roster_pl.filter(
            pl.col("position_group")
            .fill_null("")
            .str.to_uppercase()
            .is_in(["QB", "RB", "WR", "TE"])
        )
    elif "position" in roster_pl.columns:
        roster_pl = roster_pl.filter(
            pl.col("position")
            .fill_null("")
            .str.to_uppercase()
            .is_in(["QB", "RB", "WR", "TE", "FB"])
        )

    roster_pl = roster_pl.filter(pl.col("status").is_in(["ACT", "INA", "ACTIVATED", "QUESTIONABLE"]))

    roster_game = roster_pl.join(schedule_team, on=["season", "week", "team"], how="inner")
    if roster_game.is_empty():
        return df_merged

    for col, dtype in (
        ("status_abbr", pl.Utf8),
        ("status_detail", pl.Utf8),
        ("depth_chart_order", pl.Int32),
    ):
        if col not in roster_game.columns:
            roster_game = roster_game.with_columns(pl.lit(None).cast(dtype).alias(col))

    snapshot_pl = _load_roster_snapshots_for_years(seasons)
    cutoff_hours = float(get_decision_cutoff_hours())
    fallback_hours = float(get_fallback_cutoff_hours())
    roster_game = roster_game.with_columns(
        pl.when(pl.col("game_start_utc").is_not_null())
        .then(pl.col("game_start_utc") - pl.duration(hours=cutoff_hours))
        .otherwise(
            pl.when(pl.col("game_date").is_not_null())
            .then(
                pl.col("game_date")
                .cast(pl.Datetime("ms", "UTC"))
                - pl.duration(hours=fallback_hours)
            )
            .otherwise(None)
        )
        .alias("_decision_cutoff_ts")
    )

    if not snapshot_pl.is_empty():
        snapshot_pl = snapshot_pl.select(
            [
                pl.col("game_id"),
                pl.col("player_id"),
                pl.col("team"),
                pl.col("snapshot_ts").alias("snapshot_ts_snap"),
                pl.col("status").alias("status_snap"),
                pl.col("status_abbr").alias("status_abbr_snap"),
                pl.col("status_detail").alias("status_detail_snap"),
                pl.col("depth_chart_order").alias("depth_chart_order_snap"),
            ]
        )
        roster_game = roster_game.join(
            snapshot_pl,
            on=["game_id", "player_id"],
            how="left",
        )
        roster_game = roster_game.with_columns(
            pl.when(
                pl.col("snapshot_ts_snap").is_not_null()
                & pl.col("_decision_cutoff_ts").is_not_null()
                & (pl.col("snapshot_ts_snap") <= pl.col("_decision_cutoff_ts"))
            )
            .then(pl.col("snapshot_ts_snap"))
            .otherwise(None)
            .alias("roster_status_snapshot_ts")
        )
        roster_game = roster_game.with_columns(
            [
                pl.when(pl.col("roster_status_snapshot_ts").is_not_null())
                .then(pl.col("status_snap"))
                .otherwise(pl.col("status"))
                .alias("status"),
                pl.when(pl.col("roster_status_snapshot_ts").is_not_null())
                .then(pl.col("status_abbr_snap"))
                .otherwise(pl.col("status_abbr"))
                .alias("status_abbr"),
                pl.when(pl.col("roster_status_snapshot_ts").is_not_null())
                .then(pl.col("status_detail_snap"))
                .otherwise(pl.col("status_detail"))
                .alias("status_detail"),
                pl.when(pl.col("roster_status_snapshot_ts").is_not_null())
                .then(pl.col("depth_chart_order_snap"))
                .otherwise(pl.col("depth_chart_order"))
                .alias("depth_chart_order"),
            ]
        ).drop(
            [
                "snapshot_ts_snap",
                "status_snap",
                "status_abbr_snap",
                "status_detail_snap",
                "depth_chart_order_snap",
            ]
        )
    else:
        roster_game = roster_game.with_columns(
            pl.lit(None).cast(pl.Datetime("ms", "UTC")).alias("roster_status_snapshot_ts")
        )

    window_weeks = (
        df_merged.select(["season", "week"])
        .drop_nulls()
        .unique()
        .with_columns([pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)])
    )
    if window_weeks.is_empty():
        return df_merged

    roster_game = roster_game.join(window_weeks, on=["season", "week"], how="inner")
    if roster_game.is_empty():
        return df_merged

    key_cols = ["season", "week", "team", "player_id", "game_id"]
    cast_exprs = []
    for col in key_cols:
        if col not in df_merged.columns:
            continue
        if col in ("season", "week"):
            cast_exprs.append(pl.col(col).cast(pl.Int32))
        else:
            cast_exprs.append(pl.col(col).cast(pl.Utf8))
    if cast_exprs:
        df_merged = df_merged.with_columns(cast_exprs)

    existing_keys = df_merged.select(key_cols).unique()
    missing = roster_game.join(existing_keys, on=key_cols, how="anti")
    if missing.is_empty():
        return df_merged

    zero_exprs = [
        pl.lit(0.0).cast(pl.Float64).alias("passing_yards"),
        pl.lit(0.0).cast(pl.Float64).alias("rushing_yards"),
        pl.lit(0.0).cast(pl.Float64).alias("receiving_yards"),
        pl.lit(0.0).cast(pl.Float64).alias("air_yards_total"),
        pl.lit(0.0).cast(pl.Float64).alias("air_yards_all_targets"),
        pl.lit(0.0).cast(pl.Float64).alias("yards_after_catch_total"),
        pl.lit(0).cast(pl.Int64).alias("pass_attempt"),
        pl.lit(0).cast(pl.Int64).alias("completion"),
        pl.lit(0).cast(pl.Int64).alias("carry"),
        pl.lit(0).cast(pl.Int64).alias("target"),
        pl.lit(0).cast(pl.Int64).alias("reception"),
        pl.lit(0).cast(pl.Int64).alias("red_zone_target"),
        pl.lit(0).cast(pl.Int64).alias("red_zone_carry"),
        pl.lit(0).cast(pl.Int64).alias("goal_to_go_target"),
        pl.lit(0).cast(pl.Int64).alias("goal_to_go_carry"),
        pl.lit(0).cast(pl.Int64).alias("passing_td"),
        pl.lit(0).cast(pl.Int64).alias("rushing_td_count"),
        pl.lit(0).cast(pl.Int64).alias("receiving_td_count"),
        pl.lit(0).cast(pl.Int64).alias("touchdowns"),
        pl.lit(0).cast(pl.Int8).alias("anytime_td"),
        pl.lit(0).cast(pl.Int64).alias("td_count"),
        pl.lit(None).cast(pl.Utf8).alias("touchdown_player_id"),
        pl.lit(0).cast(pl.Int8).alias("touchdown"),
        pl.lit(0.0).cast(pl.Float32).alias("snap_offense_pct_prev"),
        pl.lit(0.0).cast(pl.Float32).alias("snap_offense_pct_l3"),
        pl.lit(0).cast(pl.Int32).alias("snap_offense_snaps_prev"),
        pl.lit(0.0).cast(pl.Float32).alias("snap_defense_pct_prev"),
        pl.lit(0.0).cast(pl.Float32).alias("snap_defense_pct_l3"),
        pl.lit(0).cast(pl.Int32).alias("snap_defense_snaps_prev"),
        pl.lit(0.0).cast(pl.Float32).alias("snap_st_pct_prev"),
        pl.lit(0.0).cast(pl.Float32).alias("snap_st_pct_l3"),
        pl.lit(0).cast(pl.Int32).alias("snap_st_snaps_prev"),
    ]

    missing = missing.with_columns(zero_exprs)

    # Ensure season_type and metadata defaults
    missing = missing.with_columns(
        [
            pl.when(pl.col("season_type").is_null())
            .then(pl.lit("REG"))
            .otherwise(pl.col("season_type"))
            .alias("season_type"),
            pl.col("game_start_utc")
            .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
            .alias("game_start_utc"),
            pl.col("game_date").cast(pl.Date),
        ]
    )

    if "snap_zero_usage_stub" not in df_merged.columns:
        # Removed explicit stub feature per user preference for natural feature learning
        pass
    
    # No longer adding snap_zero_usage_stub column

    combined = pl.concat([df_merged, missing], how="diagonal_relaxed")
    if "_decision_cutoff_ts" in combined.columns:
        combined = combined.drop("_decision_cutoff_ts")
    logger.info("Added %d zero-usage roster rows", len(missing))

    # ------------------------------------------------------------------
    # Depth Chart Mobility
    # ------------------------------------------------------------------
    if "depth_chart_order" in combined.columns:
        combined = combined.sort(["player_id", "season", "week"])
        combined = combined.with_columns(
            (pl.col("depth_chart_order").shift(1).over("player_id") - pl.col("depth_chart_order"))
            .fill_null(0)
            .alias("depth_chart_mobility")
        )

    # ------------------------------------------------------------------
    # Injury Recovery / Pitch Count Signals
    # ------------------------------------------------------------------
    if "status" in combined.columns:
        combined = combined.with_columns(
            (pl.col("status") == "INA").cast(pl.Int8).alias("is_inactive")
        )
        # Recent inactivity count (proxy for "rust" or "pitch count" risk)
        combined = combined.with_columns(
            pl.col("is_inactive")
            .rolling_sum(window_size=5, min_periods=1)
            .shift(1)
            .over("player_id")
            .fill_null(0)
            .alias("recent_inactivity_count")
        )
        
        # Was inactive last game? (Strong signal for limited snaps)
        combined = combined.with_columns(
            pl.col("is_inactive")
            .shift(1)
            .over("player_id")
            .fill_null(0)
            .alias("was_inactive_last_game")
        )

    # ------------------------------------------------------------------
    # Role flags (red-zone / goal-line) derived from historical shares
    # ------------------------------------------------------------------
    combined = _append_role_flags(combined)
    combined = _append_moe_position_features(combined)

    return combined


def _append_role_flags(df: pl.DataFrame, quantile: float = 0.7) -> pl.DataFrame:
    """Create role classification flags using team/position quantiles."""
    share_specs = [
        ("hist_red_zone_target_share_l3", "role_primary_red_zone_target"),
        ("hist_goal_to_go_carry_share_l3", "role_goal_line_back"),
    ]
    group_cols = [col for col in ("team", "position") if col in df.columns]
    if not group_cols:
        return df

    for share_col, flag_name in share_specs:
        if share_col not in df.columns:
            continue
        thresholds = (
            df.select(group_cols + [share_col])
            .drop_nulls()
            .group_by(group_cols)
            .agg(
                pl.col(share_col)
                .quantile(quantile, interpolation="higher")
                .alias(f"__{flag_name}_threshold")
            )
        )
        fallback_expr = df.select(
            pl.col(share_col).fill_null(0.0).quantile(quantile, interpolation="higher")
        )
        fallback = fallback_expr.item() if fallback_expr.height else 0.0

        if thresholds.is_empty():
            df = df.with_columns(
                pl.when(pl.col(share_col).is_not_null())
                .then(pl.col(share_col).fill_null(0.0) >= pl.lit(fallback))
                .otherwise(0)
                .cast(pl.Int8)
                .alias(flag_name)
            )
            continue

        df = df.join(thresholds, on=group_cols, how="left")
        threshold_col = f"__{flag_name}_threshold"
        df = df.with_columns(
            pl.when(pl.col(share_col).is_not_null())
            .then(
                pl.col(share_col).fill_null(0.0)
                >= pl.col(threshold_col).fill_null(fallback)
            )
            .otherwise(0)
            .cast(pl.Int8)
            .alias(flag_name)
        )
        flagged = df.filter(pl.col(flag_name) == 1).height
        logger.info(
            "Role flag %s: %d rows flagged using %.0fth percentile threshold.",
            flag_name,
            flagged,
            quantile * 100,
        )
        df = df.drop(threshold_col, strict=False)

    return df


def _append_moe_position_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add position-specific features optimized for Mixture of Experts (MoE) modeling.
    
    These features help MoE models by providing explicit position-role signals:
    - QB: is_likely_starter - binary indicator for starting QB
    - RB: rb_workload_tier - binned usage tier (0-3)
    - WR: wr_depth_tier - binned snap % tier (0-3 for WR4+/WR3/WR2/WR1)
    - TE: te_is_receiving_te - binary indicator for receiving-focused TE
    """
    new_cols = []
    
    # QB Feature: is_likely_starter (snap_pct_prev > 0.8)
    if "snap_offense_pct_prev" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase() == "QB")
            .then(
                (pl.col("snap_offense_pct_prev").fill_null(0.0) > 0.8).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("qb_is_likely_starter")
        )
        logger.info("Added qb_is_likely_starter feature for MoE QB model.")
    
    # RB Feature: rb_workload_tier (binned combined usage share)
    if "combined_usage_share_l3" in df.columns and "position" in df.columns:
        # Tier 0: <10%, Tier 1: 10-25%, Tier 2: 25-40%, Tier 3: >40%
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase().is_in(["RB", "HB", "FB"]))
            .then(
                pl.when(pl.col("combined_usage_share_l3").fill_null(0.0) > 0.4)
                .then(pl.lit(3))
                .when(pl.col("combined_usage_share_l3").fill_null(0.0) > 0.25)
                .then(pl.lit(2))
                .when(pl.col("combined_usage_share_l3").fill_null(0.0) > 0.10)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_workload_tier")
        )
        logger.info("Added rb_workload_tier feature for MoE RB model.")
    
    # RB Feature: rb_is_receiving_back (target share > carry share)
    if "hist_target_share_l3" in df.columns and "hist_carry_share_l3" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase().is_in(["RB", "HB", "FB"]))
            .then(
                (pl.col("hist_target_share_l3").fill_null(0.0) > 
                 pl.col("hist_carry_share_l3").fill_null(0.0)).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_receiving_back")
        )
        logger.info("Added rb_is_receiving_back feature for MoE RB model.")
    
    # WR Feature: wr_depth_tier (binned snap_offense_pct_prev)
    # Tier 0: <40% (WR4+), Tier 1: 40-70% (WR3), Tier 2: 70-85% (WR2), Tier 3: >85% (WR1)
    if "snap_offense_pct_prev" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase() == "WR")
            .then(
                pl.when(pl.col("snap_offense_pct_prev").fill_null(0.0) > 0.85)
                .then(pl.lit(3))
                .when(pl.col("snap_offense_pct_prev").fill_null(0.0) > 0.70)
                .then(pl.lit(2))
                .when(pl.col("snap_offense_pct_prev").fill_null(0.0) > 0.40)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("wr_depth_tier")
        )
        logger.info("Added wr_depth_tier feature for MoE WR model.")
    
    # WR Feature: wr_is_slot_receiver
    if "ps_targets_slot_share_l3" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase() == "WR")
            .then(
                (pl.col("ps_targets_slot_share_l3").fill_null(0.0) > 0.4).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("wr_is_slot_receiver")
        )
        logger.info("Added wr_is_slot_receiver feature for MoE WR model.")
    
    # TE Feature: te_is_receiving_te (target share > 10%)
    if "hist_target_share_l3" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase() == "TE")
            .then(
                (pl.col("hist_target_share_l3").fill_null(0.0) > 0.10).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_receiving_te")
        )
        logger.info("Added te_is_receiving_te feature for MoE TE model.")
    
    # TE Feature: te_target_concentration (target share / snap share)
    if "hist_target_share_l3" in df.columns and "snap_offense_pct_l3" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase() == "TE")
            .then(
                pl.col("hist_target_share_l3").fill_null(0.0) / 
                (pl.col("snap_offense_pct_l3").fill_null(0.01) + 0.01)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_target_concentration")
        )
        logger.info("Added te_target_concentration feature for MoE TE model.")
    
    # =========================================================================
    # Additional RB Features (High-Impact for MoE)
    # =========================================================================
    
    # RB Feature: rb_is_lead_back (carry share > 40%) - 0.536 correlation with snaps!
    if "hist_carry_share_l3" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase().is_in(["RB", "HB", "FB"]))
            .then(
                (pl.col("hist_carry_share_l3").fill_null(0.0) > 0.40).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_lead_back")
        )
        logger.info("Added rb_is_lead_back feature for MoE RB model.")
    
    # RB Feature: rb_is_committee_back (carry share 15-40%)
    if "hist_carry_share_l3" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase().is_in(["RB", "HB", "FB"]))
            .then(
                ((pl.col("hist_carry_share_l3").fill_null(0.0) >= 0.15) & 
                 (pl.col("hist_carry_share_l3").fill_null(0.0) <= 0.40)).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_committee_back")
        )
        logger.info("Added rb_is_committee_back feature for MoE RB model.")
    
    # RB Feature: rb_snap_tier (binned snap percentage for RBs)
    # Tier 0: <20%, Tier 1: 20-40%, Tier 2: 40-60%, Tier 3: >60%
    if "snap_offense_pct_prev" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase().is_in(["RB", "HB", "FB"]))
            .then(
                pl.when(pl.col("snap_offense_pct_prev").fill_null(0.0) > 0.60)
                .then(pl.lit(3))
                .when(pl.col("snap_offense_pct_prev").fill_null(0.0) > 0.40)
                .then(pl.lit(2))
                .when(pl.col("snap_offense_pct_prev").fill_null(0.0) > 0.20)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_snap_tier")
        )
        logger.info("Added rb_snap_tier feature for MoE RB model.")
    
    # RB Feature: rb_goal_line_indicator (goal line carry share > 20%)
    if "hist_goal_to_go_carry_share_l3" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase().is_in(["RB", "HB", "FB"]))
            .then(
                (pl.col("hist_goal_to_go_carry_share_l3").fill_null(0.0) > 0.20).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("rb_is_goal_line_rb")
        )
        logger.info("Added rb_is_goal_line_rb feature for MoE RB model.")
    
    # =========================================================================
    # Additional TE Features (High-Impact for MoE)
    # =========================================================================
    
    # TE Feature: te_is_te1 (snap_pct > 65%) - 0.554 correlation with snaps!
    if "snap_offense_pct_l3" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase() == "TE")
            .then(
                (pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.65).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_te1")
        )
        logger.info("Added te_is_te1 feature for MoE TE model.")
    
    # TE Feature: te_is_blocking_te (high snap, low target)
    if "snap_offense_pct_l3" in df.columns and "hist_target_share_l3" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase() == "TE")
            .then(
                ((pl.col("snap_offense_pct_l3").fill_null(0.0) > 0.40) & 
                 (pl.col("hist_target_share_l3").fill_null(0.0) < 0.08)).cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_is_blocking_te")
        )
        logger.info("Added te_is_blocking_te feature for MoE TE model.")
    
    # TE Feature: te_snap_tier (binned snap percentage for TEs)
    # Tier 0: <30%, Tier 1: 30-50%, Tier 2: 50-70%, Tier 3: >70%
    if "snap_offense_pct_prev" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase() == "TE")
            .then(
                pl.when(pl.col("snap_offense_pct_prev").fill_null(0.0) > 0.70)
                .then(pl.lit(3))
                .when(pl.col("snap_offense_pct_prev").fill_null(0.0) > 0.50)
                .then(pl.lit(2))
                .when(pl.col("snap_offense_pct_prev").fill_null(0.0) > 0.30)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
            )
            .otherwise(pl.lit(None).cast(pl.Int8))
            .alias("te_snap_tier")
        )
        logger.info("Added te_snap_tier feature for MoE TE model.")
    
    # TE Feature: te_route_participation (from pre-snap data)
    if "ps_game_route_participation_pct" in df.columns and "position" in df.columns:
        new_cols.append(
            pl.when(pl.col("position").str.to_uppercase() == "TE")
            .then(
                pl.col("ps_game_route_participation_pct").fill_null(0.0).cast(pl.Float32)
            )
            .otherwise(pl.lit(None).cast(pl.Float32))
            .alias("te_route_participation")
        )
        logger.info("Added te_route_participation feature for MoE TE model.")
    
    if new_cols:
        df = df.with_columns(new_cols)
        logger.info(f"Added {len(new_cols)} MoE position-specific features.")
    
    return df


def build_player_game_level(
    *,
    start_date: date,
    end_date: date,
    label_version: str | None = None,
) -> None:
    """Aggregate play-level data to player-game level.
    
    Creates one row per (player, game) with accumulated stats.
    Handles all skill positions: QB, RB, WR, TE.
    
    Parameters
    ----------
    start_date : date
        Start date (inclusive)
    end_date : date
        End date (inclusive)
    """
    PLAYER_GAME_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load play-level data
    force_utf8_candidates = (
        "posteam", "defteam", "team", "opponent",
        "passer_player_id", "rusher_player_id", "receiver_player_id",
        "passer_player_name", "rusher_player_name", "receiver_player_name",
        "passer_id", "rusher_id", "receiver_id",
        "posteam_type", "touchdown_player_id",
    )
    all_plays = _load_play_level_data()
    if all_plays.is_empty():
        logger.warning("No play-level data available in %s", PLAY_DIR)
        return

    df = all_plays.filter(
        (pl.col("game_date").cast(pl.Utf8) >= start_date.isoformat()) &
        (pl.col("game_date").cast(pl.Utf8) <= end_date.isoformat())
    )
    if df.is_empty():
        logger.warning("No play-level data found for date range %s to %s", start_date, end_date)
        return
    force_utf8_cols = [col for col in force_utf8_candidates if col in df.columns]
    if force_utf8_cols:
        df = df.with_columns([pl.col(col).cast(pl.Utf8) for col in force_utf8_cols])
    df = df.with_columns(
        [
            pl.col(col).cast(pl.Utf8)
            for col, dtype in zip(df.columns, df.dtypes)
            if isinstance(dtype, pl.Categorical)
        ]
    )
    
    logger.info("Loaded %d plays for player-game aggregation", len(df))

    seasons_in_window: list[int] = []
    if "season" in df.columns:
        try:
            seasons_in_window = sorted(
                {
                    int(s)
                    for s in df["season"].drop_nulls().unique().to_list()
                    if s is not None
                }
            )
        except Exception:
            seasons_in_window = []

    # Ensure situational columns exist for downstream aggregation
    situational_cols = ["red_zone_target", "red_zone_carry", "goal_to_go_target", "goal_to_go_carry"]
    missing_cols = [c for c in situational_cols if c not in df.columns]
    if missing_cols:
        df = df.with_columns([pl.lit(0).cast(pl.Int8).alias(c) for c in missing_cols])

    # Team-level totals for share features
    # Build player-game rows by aggregating three player roles separately
    player_games = []
    
    # --- 1. PASSERS (QBs) ---
    passer_stats = _aggregate_passers(df)
    if not passer_stats.is_empty():
        player_games.append(passer_stats)
        logger.info("Aggregated %d passer-game rows", len(passer_stats))
    
    # --- 2. RUSHERS (RBs, QBs, WRs on end-arounds) ---
    rusher_stats = _aggregate_rushers(df)
    if not rusher_stats.is_empty():
        player_games.append(rusher_stats)
        logger.info("Aggregated %d rusher-game rows", len(rusher_stats))
    
    # --- 3. RECEIVERS (WRs, TEs, RBs on screens) ---
    receiver_stats = _aggregate_receivers(df)
    if not receiver_stats.is_empty():
        player_games.append(receiver_stats)
        logger.info("Aggregated %d receiver-game rows", len(receiver_stats))
    
    if not player_games:
        logger.warning("No player-game stats aggregated")
        return
    
    # Combine all player roles
    df_all = pl.concat(player_games, how="diagonal")
    
    # Merge rows where same player had multiple roles in same game
    # (e.g., QB who also rushed, WR who also rushed)
    df_merged = _merge_multi_role_players(
        df_all, label_version=label_version or DEFAULT_LABEL_VERSION
    )

    # IMPORTANT: Ensure zero-usage rostered players are present *before* we
    # derive any TD label columns so that they receive consistent anytime_td_*
    # and td_count_* labels (all zeros when they have no TD components).
    df_merged = _append_zero_usage_players(df_merged, seasons_in_window)

    # Recompute standardized TD labels from core components so that
    # anytime_td_* and td_count_* columns follow the canonical semantics
    # (including the skill-only variants) for *all* players, including
    # the appended zero-usage scaffold rows.
    df_merged = compute_td_labels(df_merged, version=label_version or DEFAULT_LABEL_VERSION)

    # Scripted TD flag: did this player score a TD on a game where they had scripted touches?
    if "pre_snap_scripted_td" not in df_merged.columns:
        if "ps_game_scripted_touches" in df_merged.columns:
            has_skill_td: pl.Expr | None = None
            if "anytime_td_skill" in df_merged.columns:
                has_skill_td = pl.col("anytime_td_skill")
            elif "anytime_td" in df_merged.columns:
                has_skill_td = pl.col("anytime_td")
            if has_skill_td is not None:
                df_merged = df_merged.with_columns(
                    (
                        (has_skill_td.fill_null(0) == 1)
                        & (pl.col("ps_game_scripted_touches").fill_null(0) > 0)
                    )
                    .cast(pl.Int8)
                    .alias("pre_snap_scripted_td")
                )
        # If scripted stats are unavailable, create a safe default column of zeros.
        if "pre_snap_scripted_td" not in df_merged.columns:
            df_merged = df_merged.with_columns(
                pl.lit(0).cast(pl.Int8).alias("pre_snap_scripted_td")
            )
    
    pre_snap_usage = _compute_pre_snap_usage(df)
    if not pre_snap_usage.is_empty():
        df_merged = df_merged.join(
            pre_snap_usage, on=["season", "week", "team", "player_id"], how="left"
        )
    for col in PRE_SNAP_BASE_COLUMNS:
        if col in df_merged.columns:
            df_merged = df_merged.with_columns(
                pl.col(col).fill_null(0).cast(pl.Float32).alias(col)
            )
        else:
            df_merged = df_merged.with_columns(
                pl.lit(0).cast(pl.Float32).alias(col)
            )
    ps_game_aliases: list[pl.Expr] = []
    for col in PRE_SNAP_BASE_COLUMNS:
        alias = _ps_game_alias(col)
        if alias not in df_merged.columns and col in df_merged.columns:
            ps_game_aliases.append(pl.col(col).alias(alias))
    if ps_game_aliases:
        df_merged = df_merged.with_columns(ps_game_aliases)

    if PRE_SNAP_ROLLING_TARGETS:
        df_merged = df_merged.sort(["player_id", "season", "week"])
        df_merged = df_merged.with_columns(
            [
                pl.col(col)
                .shift(1)
                .over("player_id")
                .fill_null(0)
                .alias(f"{col}_prev")
                for col in PRE_SNAP_ROLLING_TARGETS
            ]
        )
        df_merged = df_merged.with_columns(
            [
                pl.col(col)
                .rolling_mean(window_size=3, min_periods=1)
                .over("player_id")
                .shift(1)
                .fill_null(0)
                .alias(f"{col}_l3")
                for col in PRE_SNAP_ROLLING_TARGETS
            ]
        )
        hist_aliases: list[pl.Expr] = []
        for base in PRE_SNAP_ROLLING_TARGETS:
            for suffix in ("prev", "l3"):
                name = f"{base}_{suffix}"
                if name in df_merged.columns:
                    hist_aliases.append(pl.col(name).alias(f"ps_hist_{name[len('ps_'):]}"))
        if hist_aliases:
            df_merged = df_merged.with_columns(hist_aliases)
    
    logger.info("Final player-game rows: %d (after merging multi-role players)", len(df_merged))
    # Enrich with roster metadata (position, depth chart, injury status)
    seasons_available: list[int] = []
    schedule_team = pl.DataFrame()
    if len(df_merged) > 0:
        seasons_available = [int(s) for s in df_merged['season'].unique().drop_nulls().to_list()]
        if seasons_available:
            schedule_team = _load_schedule_team_rows(seasons_available)
            roster_pl = _load_rosters_for_years(seasons_available)
            desired_cols = [
                'season', 'week', 'team', 'player_id',
                'position', 'position_group', 'depth_chart_position', 'depth_chart_order',
                'status'  # Critical for is_inactive derivation - was missing, causing 60% null!
            ]
            roster_cols = [c for c in desired_cols if c in roster_pl.columns]
            if not roster_cols:
                raise ValueError("Roster dataset missing all desired enrichment columns.")

            roster_pl = roster_pl.select(roster_cols).unique(subset=[c for c in ['season', 'week', 'team', 'player_id'] if c in roster_cols])
            # Align key dtypes prior to join
            cast_exprs_roster = []
            if 'season' in roster_pl.columns:
                cast_exprs_roster.append(pl.col('season').cast(pl.Int32))
            if 'week' in roster_pl.columns:
                cast_exprs_roster.append(pl.col('week').cast(pl.Int32))
            if 'team' in roster_pl.columns:
                cast_exprs_roster.append(pl.col('team').cast(pl.Utf8))
            if 'player_id' in roster_pl.columns:
                cast_exprs_roster.append(pl.col('player_id').cast(pl.Utf8))
            if cast_exprs_roster:
                roster_pl = roster_pl.with_columns(cast_exprs_roster)

            cast_exprs_df = []
            if 'season' in df_merged.columns:
                cast_exprs_df.append(pl.col('season').cast(pl.Int32))
            if 'week' in df_merged.columns:
                cast_exprs_df.append(pl.col('week').cast(pl.Int32))
            if 'team' in df_merged.columns:
                cast_exprs_df.append(pl.col('team').cast(pl.Utf8))
            if 'player_id' in df_merged.columns:
                cast_exprs_df.append(pl.col('player_id').cast(pl.Utf8))
            if cast_exprs_df:
                df_merged = df_merged.with_columns(cast_exprs_df)

            join_keys = [k for k in ['season', 'week', 'team', 'player_id'] if k in roster_pl.columns and k in df_merged.columns]
            if not join_keys:
                raise ValueError("Roster enrichment failed: join keys missing from either roster or player-game data.")

            df_merged = df_merged.join(roster_pl, on=join_keys, how="left")

            # Fill positional metadata when the canonical roster feed is missing
            # entries for certain players (common for late signings or call-ups).
            position_sources = [
                pl.col(col).cast(pl.Utf8)
                for col in (
                    "position",
                    "position_right",
                    "depth_chart_position",
                    "depth_chart_position_right",
                    "ngs_position",
                )
                if col in df_merged.columns
            ]
            if position_sources:
                df_merged = df_merged.with_columns(
                    pl.coalesce(position_sources)
                    .str.strip_chars()
                    .str.to_uppercase()
                    .alias("position")
                )

            depth_chart_sources = [
                pl.col(col).cast(pl.Utf8)
                for col in ("depth_chart_position", "depth_chart_position_right")
                if col in df_merged.columns
            ]
            if depth_chart_sources:
                df_merged = df_merged.with_columns(
                    pl.coalesce(depth_chart_sources)
                    .str.strip_chars()
                    .str.to_uppercase()
                    .alias("depth_chart_position")
                )

            logger.info("Roster metadata joined for %d rows (seasons: %s)", len(df_merged), seasons_available)
        else:
            logger.info("No season identifiers found for roster enrichment; skipping.")

    # ------------------------------------------------------------------
    # Team-level usage totals and per-player share labels for usage models
    # ------------------------------------------------------------------
    if {"season", "week", "team", "game_id", "target", "carry"} <= set(df_merged.columns):
        usage_team = (
            df_merged
            .group_by(["season", "week", "team", "game_id"])
            .agg(
                [
                    pl.col("target").fill_null(0).sum().cast(pl.Float32).alias("team_targets_total"),
                    pl.col("carry").fill_null(0).sum().cast(pl.Float32).alias("team_carries_total"),
                ]
            )
        )
        if not usage_team.is_empty():
            df_merged = df_merged.join(
                usage_team,
                on=["season", "week", "team", "game_id"],
                how="left",
            )
            eps = 1e-6
            df_merged = df_merged.with_columns(
                [
                    (
                        pl.col("target").fill_null(0).cast(pl.Float32)
                        / (pl.col("team_targets_total").fill_null(0.0) + eps)
                    )
                    .clip(0.0, 1.0)
                    .alias("target_share_label"),
                    (
                        pl.col("carry").fill_null(0).cast(pl.Float32)
                        / (pl.col("team_carries_total").fill_null(0.0) + eps)
                    )
                    .clip(0.0, 1.0)
                    .alias("carry_share_label"),
                ]
            )

    if "rush_success_plays" not in df_merged.columns:
        df_merged = df_merged.with_columns(pl.lit(0).cast(pl.Int64).alias("rush_success_plays"))

    success_eps = 1e-6
    df_merged = df_merged.with_columns(
        [
            (
                pl.col("reception").fill_null(0).cast(pl.Float32)
                / (pl.col("target").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .clip(0.0, 1.0)
            .alias("rec_success_rate_label"),
            (
                pl.col("rush_success_plays").fill_null(0).cast(pl.Float32)
                / (pl.col("carry").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .clip(0.0, 1.0)
            .alias("rush_success_rate_label"),
            (
                pl.col("receiving_yards").fill_null(0).cast(pl.Float32)
                / (pl.col("target").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .alias("yards_per_target_label"),
            # Yards per catch (only for completed receptions)
            (
                pl.col("receiving_yards").fill_null(0).cast(pl.Float32)
                / (pl.col("reception").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .alias("yards_per_catch_label"),
            # Air yards per catch (depth of target on completions)
            (
                pl.col("air_yards_total").fill_null(0).cast(pl.Float32)
                / (pl.col("reception").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .alias("air_yards_per_catch_label"),
            # Yards after catch per catch (YAC efficiency)
            (
                pl.col("yards_after_catch_total").fill_null(0).cast(pl.Float32)
                / (pl.col("reception").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .alias("yac_per_catch_label"),
            # Air yards per target (depth of target, including incompletions)
            (
                pl.col("air_yards_all_targets").fill_null(0).cast(pl.Float32)
                / (pl.col("target").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .alias("air_yards_per_target_label"),
            (
                pl.col("rushing_yards").fill_null(0).cast(pl.Float32)
                / (pl.col("carry").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .alias("yards_per_carry_label"),
            (
                pl.col("receiving_td_count").fill_null(0).cast(pl.Float32)
                / (pl.col("target").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .clip(0.0, 1.0)
            .alias("td_per_target_label"),
            (
                pl.col("rushing_td_count").fill_null(0).cast(pl.Float32)
                / (pl.col("carry").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .clip(0.0, 1.0)
            .alias("td_per_carry_label"),
            (
                pl.col("rushing_td_count").fill_null(0).cast(pl.Float32)
                / (pl.col("red_zone_carry").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .clip(0.0, 1.0)
            .alias("td_per_rz_carry_label"),
            (
                pl.col("rushing_td_count").fill_null(0).cast(pl.Float32)
                / (pl.col("goal_to_go_carry").fill_null(0).cast(pl.Float32) + success_eps)
            )
            .clip(0.0, 1.0)
            .alias("td_per_gl_carry_label"),
        ]
    )

    # Enrich with weekly injury reports
    if seasons_available:
        try:
            injuries_raw = _load_injuries_for_years(seasons_available)
        except Exception as exc:  # pragma: no cover - passthrough for optional dependency
            logger.warning("Injury enrichment skipped: %s", exc)
        else:
            if injuries_raw.is_empty():
                logger.info("No injury reports available for seasons %s; skipping enrichment.", seasons_available)
            else:
                injury_pl = injuries_raw
                if "player_id" not in injury_pl.columns and "gsis_id" in injury_pl.columns:
                    injury_pl = injury_pl.rename({"gsis_id": "player_id"})
                available_cols = [c for c in [
                    "season",
                    "week",
                    "team",
                    "player_id",
                    "report_status",
                    "report_status_abbr",
                    "report_status_description",
                    "practice_status",
                    "report_primary_injury",
                    "practice_primary_injury",
                    "date_modified",
                    "reported_at",
                ] if c in injury_pl.columns]
                injury_pl = injury_pl.select(available_cols)

                if "player_id" not in injury_pl.columns:
                    logger.warning("Injury dataset missing player identifiers; skipping enrichment.")
                else:
                    logger.info("Loaded %d raw injury rows", len(injury_pl))
                    injury_pl = injury_pl.filter(pl.col("player_id").is_not_null())
                    cast_exprs: list[pl.Expr] = [pl.col("player_id").cast(pl.Utf8)]
                    if "season" in injury_pl.columns:
                        cast_exprs.append(pl.col("season").cast(pl.Int32))
                    if "week" in injury_pl.columns:
                        cast_exprs.append(pl.col("week").cast(pl.Int32))
                    if "team" in injury_pl.columns:
                        cast_exprs.append(
                            pl.col("team")
                            .cast(pl.Utf8)
                            .str.strip_chars()
                            .str.to_uppercase()
                            .alias("team")
                        )
                    logger.info("Injury rows after player_id filter: %d", len(injury_pl))

                    status_source_exprs: list[pl.Expr] = []
                    if "report_status_description" in injury_pl.columns:
                        status_source_exprs.append(pl.col("report_status_description").cast(pl.Utf8))
                    if "report_status_abbr" in injury_pl.columns:
                        status_source_exprs.append(pl.col("report_status_abbr").cast(pl.Utf8))
                    if "report_status" in injury_pl.columns:
                        status_source_exprs.append(pl.col("report_status").cast(pl.Utf8))
                    if status_source_exprs:
                        cast_exprs.append(
                            pl.coalesce(status_source_exprs)
                            .str.strip_chars()
                            .str.replace_all(r"\\s+", " ", literal=False)
                            .str.to_uppercase()
                            .alias("report_status_clean")
                        )
                    if "practice_status" in injury_pl.columns:
                        cast_exprs.append(
                            pl.col("practice_status")
                            .cast(pl.Utf8)
                            .str.strip_chars()
                            .str.replace_all(r"\\s+", " ", literal=False)
                            .str.to_uppercase()
                            .alias("practice_status_clean")
                        )
                    if "report_primary_injury" in injury_pl.columns:
                        cast_exprs.append(
                            pl.col("report_primary_injury")
                            .cast(pl.Utf8)
                            .str.strip_chars()
                            .alias("report_primary_injury")
                        )
                    if "practice_primary_injury" in injury_pl.columns:
                        cast_exprs.append(
                            pl.col("practice_primary_injury")
                            .cast(pl.Utf8)
                            .str.strip_chars()
                            .alias("practice_primary_injury")
                        )
                    if "date_modified" in injury_pl.columns:
                        cast_exprs.append(
                            pl.col("date_modified")
                            .cast(pl.Utf8)
                            .alias("date_modified_ts")
                        )
                    else:
                        cast_exprs.append(pl.lit(None).cast(pl.Utf8).alias("date_modified_ts"))
                    injury_pl = injury_pl.with_columns(cast_exprs)

                    timestamp_sources: list[pl.Expr] = []
                    if "reported_at" in injury_pl.columns:
                        timestamp_sources.append(pl.col("reported_at").cast(pl.Utf8))
                    if "date_modified_ts" in injury_pl.columns:
                        timestamp_sources.append(pl.col("date_modified_ts"))
                    if timestamp_sources:
                        injury_pl = injury_pl.with_columns(
                            pl.coalesce(timestamp_sources).alias("_injury_reported_at_raw")
                        )
                    else:
                        injury_pl = injury_pl.with_columns(
                            pl.lit(None).cast(pl.Utf8).alias("_injury_reported_at_raw")
                        )
                    injury_pl = injury_pl.with_columns(
                        pl.col("_injury_reported_at_raw")
                        .str.strptime(pl.Datetime("ns", "UTC"), strict=False)
                        .alias("reported_at")
                    ).drop("_injury_reported_at_raw")

                    if not schedule_team.is_empty():
                        schedule_cols = [col for col in ["season", "week", "team", "game_date", "season_type", "game_start_utc"] if col in schedule_team.columns]
                        injury_pl = injury_pl.join(
                            schedule_team.select(schedule_cols).unique(),
                            on=[c for c in ["season", "week", "team"] if c in injury_pl.columns],
                            how="left",
                        )
                    else:
                        injury_pl = injury_pl.with_columns(
                            [
                                pl.lit(None).cast(pl.Date).alias("game_date"),
                                pl.lit(None).cast(pl.Datetime("ms", "UTC")).alias("game_start_utc"),
                                pl.lit("REG").alias("season_type"),
                            ]
                        )

                    cutoff_hours = float(get_decision_cutoff_hours())
                    fallback_hours = float(get_fallback_cutoff_hours())
                    injury_pl = injury_pl.with_columns(
                        pl.when(pl.col("game_start_utc").is_not_null())
                        .then(pl.col("game_start_utc") - pl.duration(hours=cutoff_hours))
                        .otherwise(
                            pl.when(pl.col("game_date").is_not_null())
                            .then(
                                pl.col("game_date")
                                .cast(pl.Datetime("ms", "UTC"))
                                - pl.duration(hours=fallback_hours)
                            )
                            .otherwise(None)
                        )
                        .alias("decision_cutoff_ts")
                    )

                    injury_pl = injury_pl.filter(
                        pl.col("decision_cutoff_ts").is_not_null()
                        & pl.col("reported_at").is_not_null()
                        & (pl.col("reported_at") <= pl.col("decision_cutoff_ts"))
                    )
                    logger.info(
                        "Injury rows after cutoff filter: %d",
                        len(injury_pl),
                    )

                    if "practice_status_clean" in injury_pl.columns:
                        injury_pl = injury_pl.with_columns([
                            pl.col("practice_status_clean")
                            .fill_null("")
                            .str.contains("DID NOT PARTICIPATE")
                            .cast(pl.Int8)
                            .alias("practice_dnp_flag"),
                            pl.col("practice_status_clean")
                            .fill_null("")
                            .str.contains("LIMITED")
                            .cast(pl.Int8)
                            .alias("practice_limited_flag"),
                            pl.col("practice_status_clean")
                            .fill_null("")
                            .str.contains("FULL")
                            .cast(pl.Int8)
                            .alias("practice_full_flag"),
                        ])

                    sort_cols = [col for col in ["season", "week", "player_id", "date_modified_ts"] if col in injury_pl.columns]
                    if sort_cols:
                        injury_pl = injury_pl.sort(sort_cols, nulls_last=True)

                    group_keys = [k for k in ["season", "week", "player_id"] if k in injury_pl.columns]
                    if not group_keys:
                        logger.warning("Injury dataset missing required grouping keys; skipping enrichment.")
                    else:
                        agg_exprs: list[pl.Expr] = [pl.len().alias("_injury_rows")]
                        if "team" in injury_pl.columns:
                            agg_exprs.append(
                                pl.col("team").last().alias("injury_report_team")
                            )
                        if "report_status_clean" in injury_pl.columns:
                            agg_exprs.append(
                                pl.col("report_status_clean").last().alias("injury_report_status")
                            )
                            agg_exprs.append(
                                pl.col("report_status_clean").drop_nulls().alias("_injury_report_status_seq")
                            )
                        if "practice_status_clean" in injury_pl.columns:
                            agg_exprs.append(
                                pl.col("practice_status_clean").last().alias("injury_practice_status")
                            )
                            agg_exprs.append(
                                pl.col("practice_status_clean").drop_nulls().alias("_injury_practice_status_seq")
                            )
                        if "report_primary_injury" in injury_pl.columns:
                            agg_exprs.append(
                                pl.col("report_primary_injury").last().alias("injury_report_primary")
                            )
                        if "practice_primary_injury" in injury_pl.columns:
                            agg_exprs.append(
                                pl.col("practice_primary_injury").last().alias("injury_practice_primary")
                            )
                        if "practice_dnp_flag" in injury_pl.columns:
                            agg_exprs.append(
                                pl.col("practice_dnp_flag")
                                .fill_null(0)
                                .sum()
                                .alias("injury_practice_dnp_count")
                            )
                        if "practice_limited_flag" in injury_pl.columns:
                            agg_exprs.append(
                                pl.col("practice_limited_flag")
                                .fill_null(0)
                                .sum()
                                .alias("injury_practice_limited_count")
                            )
                        if "practice_full_flag" in injury_pl.columns:
                            agg_exprs.append(
                                pl.col("practice_full_flag")
                                .fill_null(0)
                                .sum()
                                .alias("injury_practice_full_count")
                            )
                        if "reported_at" in injury_pl.columns:
                            agg_exprs.append(
                                pl.col("reported_at")
                                .drop_nulls()
                                .alias("_injury_reported_at_seq")
                            )
                        if "decision_cutoff_ts" in injury_pl.columns:
                            agg_exprs.append(pl.col("decision_cutoff_ts").last().alias("_injury_decision_cutoff_ts"))
                        if "game_start_utc" in injury_pl.columns:
                            agg_exprs.append(pl.col("game_start_utc").last().alias("_injury_game_start_utc"))

                        injury_features = injury_pl.group_by(group_keys).agg(agg_exprs)

                        injury_features = injury_features.with_columns([
                            pl.when(pl.col("_injury_rows") > 0)
                            .then(1)
                            .otherwise(0)
                            .cast(pl.Float32)
                            .alias("injury_is_listed")
                        ]).drop("_injury_rows")

                        if "injury_report_team" in injury_features.columns:
                            injury_features = injury_features.drop("injury_report_team")
    
                        if "_injury_practice_status_seq" in injury_features.columns:
                            injury_features = injury_features.with_columns([
                                pl.col("_injury_practice_status_seq").list.get(0, null_on_oob=True).alias("injury_practice_status_day1"),
                                pl.col("_injury_practice_status_seq").list.get(1, null_on_oob=True).alias("injury_practice_status_day2"),
                                pl.col("_injury_practice_status_seq").list.get(2, null_on_oob=True).alias("injury_practice_status_day3"),
                                pl.col("_injury_practice_status_seq").list.join(">").alias("injury_practice_status_sequence"),
                            ]).drop("_injury_practice_status_seq")
                        if "_injury_report_status_seq" in injury_features.columns:
                            injury_features = injury_features.with_columns([
                                pl.col("_injury_report_status_seq").list.join(">").alias("injury_report_status_sequence"),
                                pl.col("_injury_report_status_seq").list.last().alias("injury_game_designation"),
                            ]).drop("_injury_report_status_seq")
                        if "injury_game_designation" in injury_features.columns:
                            injury_features = injury_features.with_columns(
                                pl.col("injury_game_designation")
                                .cast(pl.Utf8)
                                .alias("injury_game_designation")
                            )
                            injury_features = injury_features.with_columns(
                                pl.col("injury_game_designation")
                                .is_in(["OUT", "DOUBTFUL", "INACTIVE"])
                                .cast(pl.Float32)
                                .alias("injury_is_inactive_designation")
                            )

                        report_prob = (
                            pl.when(pl.col("injury_report_status") == "OUT")
                            .then(0.98)
                            .when(pl.col("injury_report_status") == "DOUBTFUL")
                            .then(0.85)
                            .when(pl.col("injury_report_status") == "QUESTIONABLE")
                            .then(0.55)
                            .when(pl.col("injury_report_status") == "SUSPENDED")
                            .then(0.9)
                            .otherwise(0.1)
                        )
                        practice_prob = (
                            pl.when(pl.col("injury_practice_status") == "DID NOT PARTICIPATE")
                            .then(0.8)
                            .when(pl.col("injury_practice_status") == "LIMITED")
                            .then(0.5)
                            .when(pl.col("injury_practice_status") == "FULL")
                            .then(0.1)
                            .otherwise(0.15)
                        )

                        injury_features = injury_features.with_columns(
                            [
                                report_prob.cast(pl.Float32).alias("_prob_report"),
                                practice_prob.cast(pl.Float32).alias("_prob_practice"),
                            ]
                        )

                        injury_features = injury_features.with_columns(
                            pl.max_horizontal(pl.col("_prob_report"), pl.col("_prob_practice")).alias(
                                "injury_inactive_probability"
                            )
                        )

                        if "injury_practice_dnp_count" in injury_features.columns:
                            injury_features = injury_features.with_columns(
                                pl.when(pl.col("injury_practice_dnp_count").fill_null(0) >= 2)
                                .then(
                                    pl.when((pl.col("injury_inactive_probability") + 0.1) > 1.0)
                                    .then(1.0)
                                    .otherwise(pl.col("injury_inactive_probability") + 0.1)
                                )
                                .otherwise(pl.col("injury_inactive_probability"))
                                .alias("injury_inactive_probability")
                            )

                        injury_features = injury_features.with_columns(
                            pl.col("injury_inactive_probability").fill_null(0.1).cast(pl.Float32)
                        ).drop(["_prob_report", "_prob_practice"])

                        if "_injury_practice_status_seq" in injury_features.columns:
                            injury_features = injury_features.with_columns(
                                pl.col("_injury_practice_status_seq").list.len().alias("injury_practice_report_count")
                            )
                            injury_features = injury_features.with_columns(
                                pl.col("_injury_practice_status_seq").list.slice(-3, 3).alias("_injury_practice_last3")
                            )
                            injury_features = injury_features.with_columns([
                                pl.col("_injury_practice_last3")
                                .list.eval(
                                    pl.element()
                                    .fill_null("")
                                    .str.contains("DID NOT PARTICIPATE")
                                    .cast(pl.Int8)
                                )
                                .list.sum()
                                .alias("injury_practice_dnp_last3"),
                                pl.col("_injury_practice_last3")
                                .list.eval(
                                    pl.element()
                                    .fill_null("")
                                    .str.contains("LIMITED")
                                    .cast(pl.Int8)
                                )
                                .list.sum()
                                .alias("injury_practice_limited_last3"),
                                pl.col("_injury_practice_last3")
                                .list.eval(
                                    pl.element()
                                    .fill_null("")
                                    .str.contains("FULL")
                                    .cast(pl.Int8)
                                )
                                .list.sum()
                                .alias("injury_practice_full_last3"),
                            ]).drop("_injury_practice_last3")

                        if "_injury_reported_at_seq" in injury_features.columns:
                            injury_features = injury_features.with_columns(
                                pl.col("_injury_reported_at_seq").list.len().alias("injury_report_count")
                            )
                            injury_features = injury_features.with_columns(
                                pl.col("_injury_reported_at_seq").list.get(-1, null_on_oob=True).alias("_injury_last_report_at"),
                                pl.col("_injury_reported_at_seq").list.get(-2, null_on_oob=True).alias("_injury_prev_report_at"),
                            )
                            injury_features = injury_features.with_columns([
                                pl.when(
                                    pl.col("_injury_decision_cutoff_ts").is_not_null()
                                    & pl.col("_injury_last_report_at").is_not_null()
                                )
                                .then(
                                    (pl.col("_injury_decision_cutoff_ts") - pl.col("_injury_last_report_at"))
                                    .dt.total_minutes()
                                )
                                .otherwise(None)
                                .alias("injury_hours_since_last_report"),
                                pl.when(
                                    pl.col("_injury_game_start_utc").is_not_null()
                                    & pl.col("_injury_last_report_at").is_not_null()
                                )
                                .then(
                                    (pl.col("_injury_game_start_utc") - pl.col("_injury_last_report_at"))
                                    .dt.total_minutes()
                                )
                                .otherwise(None)
                                .alias("injury_hours_until_game_at_last_report"),
                                pl.when(
                                    pl.col("_injury_prev_report_at").is_not_null()
                                    & pl.col("_injury_last_report_at").is_not_null()
                                )
                                .then(
                                    (pl.col("_injury_last_report_at") - pl.col("_injury_prev_report_at"))
                                    .dt.total_minutes()
                                )
                                .otherwise(None)
                                .alias("injury_hours_between_last_reports"),
                            ])
                            injury_features = injury_features.with_columns([
                                (pl.col("injury_hours_since_last_report") / 60.0).cast(pl.Float32).alias("injury_hours_since_last_report"),
                                (pl.col("injury_hours_until_game_at_last_report") / 60.0).cast(pl.Float32).alias("injury_hours_until_game_at_last_report"),
                                (pl.col("injury_hours_between_last_reports") / 60.0).cast(pl.Float32).alias("injury_hours_between_last_reports"),
                            ])

                        injury_features = injury_features.drop(
                            [
                                col
                                for col in [
                                    "_injury_practice_status_seq",
                                    "_injury_report_status_seq",
                                    "_injury_reported_at_seq",
                                    "_injury_decision_cutoff_ts",
                                    "_injury_game_start_utc",
                                    "_injury_last_report_at",
                                    "_injury_prev_report_at",
                                    "practice_dnp_flag",
                                    "practice_limited_flag",
                                    "practice_full_flag",
                                    "decision_cutoff_ts",
                                    "date_modified_ts",
                                ]
                                if col in injury_features.columns
                            ]
                        )

                        for col in [
                            "injury_practice_dnp_count",
                            "injury_practice_limited_count",
                            "injury_practice_full_count",
                        ]:
                            if col in injury_features.columns:
                                injury_features = injury_features.with_columns(
                                    pl.col(col).cast(pl.Float32)
                                )
                        if "injury_is_listed" in injury_features.columns:
                            injury_features = injury_features.with_columns(
                                pl.col("injury_is_listed").cast(pl.Float32)
                            )

                        injury_features = injury_features.with_columns(
                            pl.lit(1).cast(pl.Int8).alias("injury_snapshot_valid")
                        )
                        logger.debug(
                            "Built injury features preview: %s",
                            injury_features.select(
                                ["season", "week", "player_id", "injury_report_status"]
                            ).head(5),
                        )

                        if "injury_report_status" in injury_features.columns:
                            non_null_status = injury_features.filter(pl.col("injury_report_status").is_not_null()).height
                            logger.info(
                                "Injury status coverage: %d/%d rows with report status",
                                non_null_status,
                                len(injury_features),
                            )

                        join_keys = [k for k in ["season", "week", "player_id"] if k in df_merged.columns and k in injury_features.columns]
                        if join_keys:
                            logger.info(
                                "Prepared injury features for join (keys %s, rows=%d)",
                                join_keys,
                                len(injury_features),
                            )
                        else:
                            logger.warning("Injury enrichment failed: join keys not present in player-game frame.")

        injury_defaults: dict[str, object] = {
            "injury_report_status": "UNKNOWN",
            "injury_practice_status": "UNKNOWN",
            "injury_report_primary": "UNKNOWN",
            "injury_practice_primary": "UNKNOWN",
            "injury_practice_status_day1": "UNKNOWN",
            "injury_practice_status_day2": "UNKNOWN",
            "injury_practice_status_day3": "UNKNOWN",
            "injury_practice_status_sequence": "UNKNOWN",
            "injury_report_status_sequence": "UNKNOWN",
            "injury_game_designation": "UNKNOWN",
            "injury_is_inactive_designation": 0.0,
            "injury_inactive_probability": 0.1,
            "injury_inactive_probability_model": None,
            "injury_practice_report_count": 0.0,
            "injury_report_count": 0.0,
            "injury_practice_dnp_last3": 0.0,
            "injury_practice_limited_last3": 0.0,
            "injury_practice_full_last3": 0.0,
            "injury_hours_since_last_report": None,
            "injury_hours_until_game_at_last_report": None,
            "injury_hours_between_last_reports": None,
            "roster_hours_since_last_game": None,
            "rest_days_since_last_game": None,
            "roster_depth_chart_order_delta": None,
            "roster_status_changed": 0.0,
            "injury_player_inactive_rate_prior": None,
            "injury_depth_slot_inactive_rate_prior": None,
            "injury_practice_pattern_inactive_rate_prior": None,
            "injury_snapshot_valid": 0,
        "injury_transaction_days_since": None,
        "injury_last_transaction_note": "UNKNOWN",
        }
        missing_cols = [col for col in injury_defaults if col not in df_merged.columns]
        if missing_cols:
            df_merged = df_merged.with_columns([
                pl.lit(injury_defaults[col]).alias(col) for col in missing_cols
            ])

        numeric_cols = [
            "injury_practice_dnp_count",
            "injury_practice_limited_count",
            "injury_practice_full_count",
            "injury_is_listed",
            "injury_is_inactive_designation",
            "injury_inactive_probability",
            "injury_practice_report_count",
            "injury_report_count",
            "injury_practice_dnp_last3",
            "injury_practice_limited_last3",
            "injury_practice_full_last3",
            "injury_hours_since_last_report",
            "injury_hours_until_game_at_last_report",
            "injury_hours_between_last_reports",
            "roster_hours_since_last_game",
            "rest_days_since_last_game",
            "roster_depth_chart_order_delta",
            "roster_status_changed",
            "injury_player_inactive_rate_prior",
            "injury_depth_slot_inactive_rate_prior",
            "injury_practice_pattern_inactive_rate_prior",
            "injury_inactive_probability_model",
            "injury_snapshot_valid",
        "injury_transaction_days_since",
        ]
        for col in numeric_cols:
            if col not in df_merged.columns:
                df_merged = df_merged.with_columns(pl.lit(None).cast(pl.Float32).alias(col))
            else:
                df_merged = df_merged.with_columns(pl.col(col).cast(pl.Float32))
        if "injury_snapshot_valid" in df_merged.columns:
            df_merged = df_merged.with_columns(pl.col("injury_snapshot_valid").cast(pl.Int8))

        # --- Roster deltas & historical injury propensities ---
        sort_keys = [k for k in ["player_id", "season", "week"] if k in df_merged.columns]
        if sort_keys:
            df_merged = df_merged.sort(sort_keys)

            if "game_start_utc" in df_merged.columns:
                df_merged = df_merged.with_columns(
                    (pl.col("game_start_utc") - pl.col("game_start_utc").shift(1).over("player_id"))
                    .dt.total_minutes()
                    .alias("roster_minutes_since_last_game")
                )
                df_merged = df_merged.with_columns(
                    (pl.col("roster_minutes_since_last_game") / 60.0).cast(pl.Float32).alias("roster_hours_since_last_game")
                )
                df_merged = df_merged.with_columns(
                    (pl.col("roster_hours_since_last_game") / 24.0)
                    .cast(pl.Float32)
                    .alias("rest_days_since_last_game")
                )
            else:
                df_merged = df_merged.with_columns(
                    [
                        pl.lit(None).cast(pl.Float32).alias("roster_hours_since_last_game"),
                        pl.lit(None).cast(pl.Float32).alias("rest_days_since_last_game"),
                    ]
                )

            if "depth_chart_order" in df_merged.columns:
                df_merged = df_merged.with_columns(
                    (
                        pl.col("depth_chart_order").cast(pl.Float32)
                        - pl.col("depth_chart_order").cast(pl.Float32).shift(1).over("player_id")
                    ).alias("roster_depth_chart_order_delta")
                )
            else:
                df_merged = df_merged.with_columns(pl.lit(None).cast(pl.Float32).alias("roster_depth_chart_order_delta"))

            df_merged = df_merged.with_columns(pl.lit(0).cast(pl.Int8).alias("roster_status_changed"))

        else:
            df_merged = df_merged.with_columns(
                [
                    pl.lit(None).cast(pl.Float32).alias("roster_hours_since_last_game"),
                    pl.lit(None).cast(pl.Float32).alias("rest_days_since_last_game"),
                    pl.lit(None).cast(pl.Float32).alias("roster_depth_chart_order_delta"),
                    pl.lit(0).cast(pl.Int8).alias("roster_status_changed"),
                ]
            )

        df_merged = _compute_injury_history_rates(df_merged)
        df_merged = _apply_injury_availability_model(df_merged)

        string_cols_fill = [
            "injury_report_status",
            "injury_practice_status",
            "injury_report_primary",
            "injury_practice_primary",
            "injury_practice_status_day1",
            "injury_practice_status_day2",
            "injury_practice_status_day3",
            "injury_practice_status_sequence",
            "injury_report_status_sequence",
            "injury_game_designation",
            "injury_last_transaction_note",
        ]
        df_merged = df_merged.with_columns(
            [
                pl.col(col).fill_null("UNKNOWN").cast(pl.Utf8).alias(col)
                for col in string_cols_fill
                if col in df_merged.columns
            ]
        )

        if "injury_game_designation" in df_merged.columns:
            df_merged = df_merged.with_columns(
                pl.col("injury_game_designation").cast(pl.Utf8).alias("injury_game_designation")
            )

        # Snap count enrichment
        try:
            snap_counts = _load_snap_counts_for_years(seasons_available)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Snap count enrichment skipped: %s", exc)
        else:
            if not snap_counts.is_empty():
                desired_snap_cols = [
                    "season",
                    "week",
                    "team",
                    "player_id",
                    "player",
                    "offense_snaps",
                    "offense_pct",
                    "defense_snaps",
                    "defense_pct",
                    "st_snaps",
                    "st_pct",
                ]
                snap_cols_available = [c for c in desired_snap_cols if c in snap_counts.columns]
                snap_counts = snap_counts.select(snap_cols_available)
                if "player" in snap_counts.columns:
                    snap_counts = snap_counts.rename({"player": "player_name"})
                snap_counts = snap_counts.with_columns(
                    [
                        pl.col("season").cast(pl.Int32),
                        pl.col("week").cast(pl.Int32),
                        pl.col("team").cast(pl.Utf8),
                        pl.col("player_name").cast(pl.Utf8),
                    ]
                )
                if "player_id" in snap_counts.columns:
                    snap_counts = snap_counts.with_columns(pl.col("player_id").cast(pl.Utf8))
                snap_counts = snap_counts.with_columns(
                    [
                        pl.col(col).cast(pl.Float32)
                        for col in ("offense_snaps", "offense_pct", "defense_snaps", "defense_pct", "st_snaps", "st_pct")
                        if col in snap_counts.columns
                    ]
                )
                
                # Add normalized player name for more robust matching
                snap_counts = snap_counts.with_columns(_normalize_player_name_expr())
                
                df_merged = df_merged.with_columns(
                    [
                        pl.col("player_name").cast(pl.Utf8),
                        pl.col("player_id").cast(pl.Utf8),
                        pl.col("team").cast(pl.Utf8),
                        pl.col("season").cast(pl.Int32),
                        pl.col("week").cast(pl.Int32),
                    ]
                )
                
                # Add normalized player name to df_merged as well
                df_merged = df_merged.with_columns(_normalize_player_name_expr())
                
                join_keys: list[str] = []
                if "player_id" in snap_counts.columns and "player_id" in df_merged.columns:
                    join_keys = [k for k in ["season", "week", "player_id"] if k in df_merged.columns and k in snap_counts.columns]
                    if "team" in df_merged.columns and "team" in snap_counts.columns:
                        join_keys.append("team")
                if not join_keys:
                    # Use normalized name for more robust matching across name formats
                    join_keys = [k for k in ["season", "week", "team", "__normalized_player_name"] if k in df_merged.columns and k in snap_counts.columns]
                    if join_keys:
                        logger.info("Snap counts using normalized name-based join for cross-format matching.")
                if join_keys:
                    df_merged = df_merged.join(snap_counts.drop("player_name"), on=join_keys, how="left")
                else:
                    logger.warning("Snap count join skipped: required keys missing.")
                
                # Clean up temporary normalized name column
                if "__normalized_player_name" in df_merged.columns:
                    df_merged = df_merged.drop("__normalized_player_name")

        snap_feature_cols = ["offense_snaps", "offense_pct", "defense_snaps", "defense_pct", "st_snaps", "st_pct"]
        missing_snap_cols = [col for col in snap_feature_cols if col not in df_merged.columns]
        if missing_snap_cols:
            df_merged = df_merged.with_columns(
                [pl.lit(0.0).cast(pl.Float32).alias(col) for col in missing_snap_cols]
            )
        df_merged = df_merged.with_columns(
            [
                pl.col(col).cast(pl.Float32).fill_null(0.0).alias(col)
                for col in snap_feature_cols
                if col in df_merged.columns
            ]
        )
        # Persist an explicit snaps label for stage-1 modeling
        if "offense_snaps" in df_merged.columns:
            df_merged = df_merged.with_columns(
                pl.col("offense_snaps").fill_null(0.0).cast(pl.Float32).alias("snaps_label")
            )
        else:
            df_merged = df_merged.with_columns(pl.lit(0.0).cast(pl.Float32).alias("snaps_label"))
        if {"player_id", "game_date"} <= set(df_merged.columns):
            df_merged = df_merged.sort(["player_id", "game_date"])
            snap_pct_cols = [col for col in ("offense_pct", "defense_pct", "st_pct") if col in df_merged.columns]
            snap_count_cols = [col for col in ("offense_snaps", "defense_snaps", "st_snaps") if col in df_merged.columns]
            lag_exprs: list[pl.Expr] = []
            rolling_exprs: list[pl.Expr] = []
            for col in snap_pct_cols:
                feature_base = col.replace("_pct", "")
                lag_exprs.append(
                    pl.col(col)
                    .shift(1)
                    .over(["player_id"])
                    .alias(f"snap_{feature_base}_pct_prev")
                )
                rolling_exprs.append(
                    pl.col(col)
                    .rolling_mean(window_size=3, min_periods=1)
                    .shift(1)
                    .over(["player_id"])
                    .alias(f"snap_{feature_base}_pct_l3")
                )
            for col in snap_count_cols:
                feature_base = col.replace("_snaps", "")
                lag_exprs.append(
                    pl.col(col)
                    .shift(1)
                    .over(["player_id"])
                    .alias(f"snap_{feature_base}_snaps_prev")
                )
            if lag_exprs:
                df_merged = df_merged.with_columns(lag_exprs)
            if rolling_exprs:
                df_merged = df_merged.with_columns(rolling_exprs)

    # ---------------------------------------------------------------------------
    # Join official NFL depth chart data from nflverse
    # ---------------------------------------------------------------------------
    df_merged = _join_nflverse_depth_charts(df_merged)

    # ---------------------------------------------------------------------------
    # Add zero-usage / no-history flags for snap prediction
    # ---------------------------------------------------------------------------
    # Players with no snap history are typically zero-usage appends from rosters
    # who haven't appeared in play-by-play data yet. This is a strong signal
    # that they likely won't play.
    if "snap_offense_pct_l3" in df_merged.columns:
        df_merged = df_merged.with_columns(
            pl.col("snap_offense_pct_l3")
            .is_null()
            .cast(pl.Int8)
            .alias("has_no_snap_history")
        )
    else:
        df_merged = df_merged.with_columns(
            pl.lit(0).cast(pl.Int8).alias("has_no_snap_history")
        )

    leak_cols = [
        "target_share",
        "carry_share",
        "pass_attempt_share",
        "red_zone_target_share",
        "red_zone_carry_share",
        "goal_to_go_target_share",
        "goal_to_go_carry_share",
    ]
    df_merged = df_merged.drop(leak_cols, strict=False)

    # Normalize roster numeric columns
    if "depth_chart_order" in df_merged.columns:
        df_merged = df_merged.with_columns(
            pl.col("depth_chart_order").cast(pl.Int16)
        )

    # Derived matchup/scheduling features
    derived_exprs: list[pl.Expr] = []
    if {"team", "home_team"} <= set(df_merged.columns):
        derived_exprs.append(
            pl.when(pl.col("team") == pl.col("home_team"))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_home")
        )
    if "game_start_utc" in df_merged.columns:
        derived_exprs.append(
            pl.when(pl.col("game_start_utc").is_not_null())
            .then(pl.col("game_start_utc").dt.hour())
            .otherwise(None)
            .cast(pl.Int8)
            .alias("game_start_hour_utc")
        )
    if "game_date" in df_merged.columns:
        derived_exprs.append(
            pl.col("game_date").dt.weekday().cast(pl.Int8).alias("game_day_of_week")
        )
    if derived_exprs:
        df_merged = df_merged.with_columns(derived_exprs)
    df_merged = _append_role_flags(df_merged)
    df_merged = _append_moe_position_features(df_merged)

    if "offense_pct" in df_merged.columns:
        df_merged = df_merged.with_columns(
            pl.when(pl.col("offense_pct").fill_null(0.0) > 0.0)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_active")
        )
        df_merged = df_merged.with_columns(
            pl.when(pl.col("is_active") == 1)
            .then(pl.col("offense_pct"))
            .otherwise(pl.lit(None))
            .alias("offense_pct_active")
        )
    else:
        df_merged = df_merged.with_columns(
            [
                pl.lit(0).cast(pl.Int8).alias("is_active"),
                pl.lit(None).cast(pl.Float32).alias("offense_pct_active"),
            ]
        )

    # Ensure consistent string types for identifiers before writing
    str_cols = [c for c in (
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
        "season_type",
        "position",
        "position_group",
        "depth_chart_position",
        "injury_report_status",
        "injury_practice_status",
        "injury_report_primary",
        "injury_practice_primary",
    ) if c in df_merged.columns]
    if str_cols:
        df_merged = df_merged.with_columns([pl.col(c).cast(pl.Utf8) for c in str_cols])

    if "game_start_utc" in df_merged.columns:
        df_merged = df_merged.with_columns(
            pl.col("game_start_utc").cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
        )

    numeric_casts: list[pl.Expr] = []
    if "weight" in df_merged.columns:
        numeric_casts.append(pl.col("weight").cast(pl.Float32))
    if "height" in df_merged.columns:
        numeric_casts.append(pl.col("height").cast(pl.Float32))
    if "rookie_year" in df_merged.columns:
        numeric_casts.append(pl.col("rookie_year").cast(pl.Int32))
    market_numeric_cols = [col for col in df_merged.columns if col.startswith("market_")]
    for col in market_numeric_cols:
        numeric_casts.append(pl.col(col).cast(pl.Float32))
    if numeric_casts:
        df_merged = df_merged.with_columns(numeric_casts)

    # Write weekly partitions
    dedup_keys = [col for col in ["season", "week", "game_id", "team", "player_id"] if col in df_merged.columns]
    if dedup_keys:
        sum_cols = [
            "passing_yards",
            "rushing_yards",
            "receiving_yards",
            "pass_attempt",
            "completion",
            "carry",
            "target",
            "reception",
            "red_zone_target",
            "red_zone_carry",
            "goal_to_go_target",
            "goal_to_go_carry",
            "passing_td",
            "rushing_td_count",
            "receiving_td_count",
            "touchdowns",
            "td_count",
        ]
        max_cols = [
            "anytime_td",
            "offense_snaps",
            "offense_pct",
            "defense_snaps",
            "defense_pct",
            "st_snaps",
            "st_pct",
        ]

        agg_exprs: list[pl.Expr] = []
        for col in sum_cols:
            if col in df_merged.columns:
                agg_exprs.append(pl.col(col).fill_null(0).sum().alias(col))
        for col in max_cols:
            if col in df_merged.columns:
                agg_exprs.append(pl.col(col).fill_null(0).max().alias(col))

        def _player_name_expr() -> pl.Expr:
            if "player_name" not in df_merged.columns:
                return pl.lit(None).alias("player_name")
            preferred = pl.col("player_name").filter(
                pl.col("player_name").is_not_null()
                & (~pl.col("player_name").str.contains("\\."))
            ).first()
            fallback = pl.col("player_name").filter(pl.col("player_name").is_not_null()).first()
            return pl.coalesce(preferred, fallback, pl.col("player_name").first()).alias("player_name")

        string_cols = [
            col
            for col in df_merged.columns
            if col not in dedup_keys + sum_cols + max_cols
        ]

        string_exprs: list[pl.Expr] = []
        for col in string_cols:
            if col == "player_name":
                string_exprs.append(_player_name_expr())
            else:
                string_exprs.append(pl.col(col).drop_nulls().first().alias(col))

        df_merged = (
            df_merged
            .group_by(dedup_keys, maintain_order=True)
            .agg(agg_exprs + string_exprs)
        )

        if (
            "injury_features" in locals()
            and isinstance(injury_features, pl.DataFrame)
            and not injury_features.is_empty()
        ):
            post_join_keys = [k for k in ["season", "week", "player_id"] if k in df_merged.columns and k in injury_features.columns]
            if post_join_keys:
                injury_cols_override = [
                    "injury_report_status",
                    "injury_practice_status",
                    "injury_report_primary",
                    "injury_practice_primary",
                    "injury_practice_dnp_count",
                    "injury_practice_limited_count",
                    "injury_practice_full_count",
                    "injury_is_listed",
                    "injury_practice_status_day1",
                    "injury_practice_status_day2",
                    "injury_practice_status_day3",
                    "injury_practice_status_sequence",
                    "injury_report_status_sequence",
                    "injury_game_designation",
                    "injury_is_inactive_designation",
                    "injury_inactive_probability",
                    "injury_practice_report_count",
                    "injury_report_count",
                    "injury_practice_dnp_last3",
                    "injury_practice_limited_last3",
                    "injury_practice_full_last3",
                    "injury_inactive_probability_model",
                ]
                existing_injury_cols = [col for col in injury_cols_override if col in df_merged.columns]
                if existing_injury_cols:
                    df_merged = df_merged.drop(existing_injury_cols)
                df_merged = df_merged.join(
                    injury_features,
                    on=post_join_keys,
                    how="left",
                )
                logger.debug(
                    "Injury join delivered %d rows (sample: %s)",
                    len(df_merged),
                    df_merged.select(
                        ["season", "week", "player_id", "injury_report_status"]
                    ).head(5),
                )

        tx_seasons = (
            df_merged.select(pl.col("season").drop_nulls().unique())
            .to_series()
            .drop_nulls()
            .to_list()
            if "season" in df_merged.columns
            else []
        )
        injury_transactions = _load_injury_transactions_for_years(tx_seasons)
        if not injury_transactions.is_empty():
            logger.debug(
                "Loaded %d injury transactions covering seasons %s",
                injury_transactions.height,
                tx_seasons,
            )
            injury_transactions = injury_transactions.filter(pl.col("player_id").is_not_null())
            if not injury_transactions.is_empty():
                injury_transactions = injury_transactions.with_columns(
                    [
                        pl.col("transaction_date")
                        .cast(pl.Datetime("ms", "UTC"))
                        .alias("transaction_ts"),
                        pl.col("notes").cast(pl.Utf8).alias("transaction_notes"),
                    ]
                ).select(
                    [
                        "player_id",
                        "transaction_ts",
                        "transaction_notes",
                    ]
                )

                df_merged = df_merged.with_columns(
                    pl.coalesce(
                        pl.col("game_start_utc"),
                        pl.col("game_date").cast(pl.Datetime("ms", "UTC")),
                    ).alias("game_start_ts")
                )

                df_merged = df_merged.sort(["player_id", "game_start_ts"], maintain_order=True)
                injury_transactions = injury_transactions.sort(["player_id", "transaction_ts"])

                df_merged = df_merged.join_asof(
                    injury_transactions,
                    left_on="game_start_ts",
                    right_on="transaction_ts",
                    by="player_id",
                    strategy="backward",
                )

                df_merged = df_merged.with_columns(
                    [
                        pl.when(pl.col("transaction_ts").is_not_null())
                        .then(
                            (
                                (pl.col("game_start_ts") - pl.col("transaction_ts"))
                                / pl.duration(days=1)
                            ).cast(pl.Float32)
                        )
                        .otherwise(None)
                        .alias("injury_transaction_days_since"),
                        pl.col("transaction_notes")
                        .fill_null("UNKNOWN")
                        .cast(pl.Utf8)
                        .alias("injury_last_transaction_note"),
                    ]
                )

                drop_cols = [col for col in ["transaction_ts", "transaction_notes"] if col in df_merged.columns]
                if drop_cols:
                    df_merged = df_merged.drop(drop_cols)

        if "game_start_ts" in df_merged.columns:
            df_merged = df_merged.drop("game_start_ts")

    weather_cols_pg = [col for col in df_merged.columns if col.startswith("weather_")]
    if weather_cols_pg:
        drop_weather_pg = [
            col
            for col in weather_cols_pg
            if col.endswith("_ts")
            or col.endswith("_source_detail")
            or col in {"weather_conditions", "weather_precip_type"}
        ]
        if drop_weather_pg:
            df_merged = df_merged.drop(drop_weather_pg, strict=False)
        weather_numeric_cols_pg = [
            col
            for col in weather_cols_pg
            if col not in drop_weather_pg
            and df_merged.schema.get(col)
            in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32)
        ]
        if weather_numeric_cols_pg:
            df_merged = df_merged.with_columns(
                [pl.col(col).cast(pl.Float32) for col in weather_numeric_cols_pg]
            )
        weather_flag_cols_pg = [
            col
            for col in weather_cols_pg
            if col not in drop_weather_pg
            and (
                col.endswith("_flag")
                or col.endswith("_is_backfill")
                or col.endswith("_is_historical")
            )
        ]
        if weather_flag_cols_pg:
            df_merged = df_merged.with_columns(
                [pl.col(col).cast(pl.Int8) for col in weather_flag_cols_pg]
            )

    if seasons_available:
        # Add previous season to ensure rolling windows have history for week 1
        extended_seasons = sorted(list(set(seasons_available + [min(seasons_available) - 1])))
        opponent_split_pl = load_rolling_opponent_splits(extended_seasons)
        
        if not opponent_split_pl.is_empty():
            if {"season", "week", "opponent"} <= set(df_merged.columns):
                if "game_date" in opponent_split_pl.columns:
                    opponent_split_pl = opponent_split_pl.drop("game_date")
                
                df_merged = df_merged.join(
                    opponent_split_pl,
                    on=["season", "week", "opponent"],
                    how="left",
                )
            else:
                logger.warning("Opponent split join skipped – missing season/week/opponent columns.")

        qb_profile_pl = _load_qb_profile_features(seasons_available)
        if not qb_profile_pl.is_empty():
            if {"season", "week", "team"} <= set(df_merged.columns):
                df_merged = df_merged.join(
                    qb_profile_pl,
                    on=["season", "week", "team"],
                    how="left",
                )
            else:
                logger.warning("QB profile join skipped – missing season/week/team columns.")

        travel_pl = _load_travel_calendar_features(seasons_available)
        if not travel_pl.is_empty():
            if {"season", "week", "team"} <= set(df_merged.columns):
                df_merged = df_merged.join(
                    travel_pl,
                    on=["season", "week", "team"],
                    how="left",
                )
            else:
                logger.warning("Travel calendar join skipped – missing season/week/team columns.")

        player_market_pl = _load_player_market_features(seasons_available)
        if not player_market_pl.is_empty():
            join_keys = [
                key
                for key in ["season", "week", "game_id", "player_id"]
                if key in df_merged.columns and key in player_market_pl.columns
            ]
            if join_keys:
                df_merged = df_merged.join(player_market_pl, on=join_keys, how="left")
            else:
                logger.warning("Player market join skipped – missing required keys.")

    drop_after_join = [
        col
        for col in (
            "qb_profile_data_as_of",
            "qb_profile_team_data_as_of",
            "game_local_start",
        )
        if col in df_merged.columns
    ]
    if drop_after_join:
        df_merged = df_merged.drop(drop_after_join)
    missing_market_cols = [col for col in PLAYER_MARKET_FLOAT_COLUMNS if col not in df_merged.columns]
    if missing_market_cols:
        df_merged = df_merged.with_columns(
            [pl.lit(None).cast(pl.Float32).alias(col) for col in missing_market_cols]
        )

    if {"market_anytime_td_prob", "injury_inactive_probability"} <= set(df_merged.columns):
        df_merged = df_merged.with_columns(
            (
                pl.col("market_anytime_td_prob") - pl.col("injury_inactive_probability")
            )
            .alias("market_anytime_td_residual")
        )

    if {"player_id", "game_date", "market_anytime_td_prob"} <= set(df_merged.columns):
        df_merged = df_merged.sort(["player_id", "game_date"])
        df_merged = df_merged.with_columns(
            pl.col("market_anytime_td_prob")
            .rolling_mean(window_size=3, min_periods=1)
            .over("player_id")
            .alias("market_anytime_td_prob_l3")
        )

    if "season" not in df_merged.columns or "week" not in df_merged.columns:
        logger.error("Missing season/week columns, cannot partition")
        return
    
    # IMPORTANT: Coalesce status columns before dropping _right columns
    # The original `status` column comes from zero-usage roster rows (filtered to ACT/INA)
    # The `status_right` column comes from roster enrichment (has status for all rows)
    # We need to keep status_right values where status is null
    if "status_right" in df_merged.columns and "status" in df_merged.columns:
        df_merged = df_merged.with_columns(
            pl.coalesce([pl.col("status"), pl.col("status_right")]).alias("status")
        )
        logger.info("Coalesced status columns: filled null status values from roster enrichment")
        
        # Recompute is_inactive now that status is fully populated
        df_merged = df_merged.with_columns(
            (pl.col("status") == "INA").cast(pl.Int8).alias("is_inactive")
        )
    
    # Clean up duplicate columns from joins (columns ending with _right)
    right_cols = [c for c in df_merged.columns if c.endswith("_right")]
    if right_cols:
        logger.info("Dropping %d duplicate join columns: %s", len(right_cols), right_cols[:5])
        df_merged = df_merged.drop(right_cols)
    
    for (s, w), sub in df_merged.group_by(["season", "week"], maintain_order=True):
        out_dir = PLAYER_GAME_DIR / f"season={int(s)}" / f"week={int(w)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "part.parquet"
        out_file.unlink(missing_ok=True)
        sub.write_parquet(out_file, compression="zstd")
        logger.info("Wrote %d player-games to %s", len(sub), out_file)

    if "market_anytime_td_prob" in df_merged.columns:
        df_merged = df_merged.with_columns(
            pl.col("market_anytime_td_prob").is_null().cast(pl.Int8).alias("market_anytime_td_missing")
        )


# Note: Aggregation and enrichment functions have been refactored to:
# - utils.feature.builders.player_aggregation (aggregate_passers, aggregate_rushers, aggregate_receivers, merge_multi_role_players)
# - utils.feature.builders.data_loaders (load_rosters_for_years, load_injuries_for_years, etc.)
# - utils.feature.builders.injury_enrichment (compute_injury_history_rates, apply_injury_availability_model)
