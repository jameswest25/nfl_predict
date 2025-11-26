from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import polars as pl

logger = logging.getLogger(__name__)

COORD_DEFAULT = "UNKNOWN_COORDINATOR"
QB_DEFAULT = "UNKNOWN_QB"
EPS = 1e-3

COORDINATOR_MAP_PATH = Path("data/processed/offensive_coordinators.parquet")


def _build_is_unavailable_expr(df: pl.DataFrame) -> pl.Expr:
    """Return a boolean expression marking players ruled out before cutoff."""
    exprs: list[pl.Expr] = []
    if "injury_game_designation" in df.columns:
        exprs.append(
            pl.col("injury_game_designation")
            .cast(pl.Utf8)
            .str.to_uppercase()
            .is_in(["OUT", "DOUBTFUL", "INACTIVE"])
        )
    if "injury_report_status" in df.columns:
        exprs.append(
            pl.col("injury_report_status")
            .cast(pl.Utf8)
            .str.to_uppercase()
            .is_in(["OUT", "DOUBTFUL"])
        )
    if not exprs:
        return pl.lit(False)
    return pl.any_horizontal(exprs)


def _load_offensive_coordinator_map(seasons: Iterable[int]) -> pl.DataFrame:
    """Fetch offensive coordinator assignments per team/season."""
    seasons_list = sorted({int(season) for season in seasons})
    if COORDINATOR_MAP_PATH.exists():
        try:
            data = pl.read_parquet(str(COORDINATOR_MAP_PATH))
            if not data.is_empty():
                return (
                    data.filter(pl.col("season").is_in(seasons_list))
                    .select(["season", "team", "offensive_coordinator"])
                    .with_columns(pl.col("offensive_coordinator").cast(pl.Utf8).str.to_uppercase())
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read offensive coordinator map (%s); falling back to defaults.", exc)

    try:
        from utils.collect.offensive_coordinators import build_offensive_coordinator_map, DEFAULT_OUTPUT
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Unable to import coordinator builder (%s). Using fallback labels.", exc
        )
        return pl.DataFrame(
            {
                "season": seasons_list,
                "team": [""] * len(seasons_list),
                "offensive_coordinator": [COORD_DEFAULT] * len(seasons_list),
            }
        )

    output_path = build_offensive_coordinator_map(seasons_list, output_path=COORDINATOR_MAP_PATH)
    if not Path(output_path).exists():
        return pl.DataFrame(
            {
                "season": seasons_list,
                "team": [""] * len(seasons_list),
                "offensive_coordinator": [COORD_DEFAULT] * len(seasons_list),
            }
        )

    data = pl.read_parquet(str(output_path))
    if data.is_empty():
        return pl.DataFrame(
            {
                "season": seasons_list,
                "team": [""] * len(seasons_list),
                "offensive_coordinator": [COORD_DEFAULT] * len(seasons_list),
            }
        )
    return (
        data.filter(pl.col("season").is_in(seasons_list))
        .select(["season", "team", "offensive_coordinator"])
        .with_columns(pl.col("offensive_coordinator").cast(pl.Utf8).str.to_uppercase())
    )


def _compute_primary_qb(df: pl.DataFrame) -> pl.DataFrame:
    qb_df = (
        df.select(
            [
                "season",
                "week",
                "team",
                "player_id",
                pl.col("pass_attempt").fill_null(0.0).alias("pass_attempt"),
            ]
        )
        .sort(["season", "week", "team", "pass_attempt"], descending=[False, False, False, True])
        .group_by(["season", "week", "team"])
        .agg(pl.col("player_id").first().alias("primary_qb_id"))
        .with_columns(pl.col("primary_qb_id").fill_null(QB_DEFAULT))
    )
    return qb_df


def _append_offense_context_columns(
    player_df: pl.DataFrame,
    seasons: Iterable[int],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    seasons_list = sorted({int(s) for s in seasons})
    coord_map = _load_offensive_coordinator_map(seasons_list)

    df = player_df.with_columns(
        pl.col("team").cast(pl.Utf8).alias("team"),
        pl.col("season").cast(pl.Int32),
    )

    df = df.join(coord_map, on=["season", "team"], how="left")
    df = df.with_columns(
        pl.when(pl.col("offensive_coordinator").is_null() | (pl.col("offensive_coordinator") == ""))
        .then(pl.concat_str([pl.lit("TEAM_"), pl.col("team")], separator="").str.to_uppercase())
        .otherwise(pl.col("offensive_coordinator").cast(pl.Utf8).str.to_uppercase())
        .alias("offensive_coordinator")
    )

    primary_qb = _compute_primary_qb(df)

    # Team-level usage per week
    team_usage = (
        df.group_by(["season", "week", "team"])
        .agg(
            [
                pl.col("game_date").min().alias("game_date"),
                pl.col("offensive_coordinator").first().alias("offensive_coordinator"),
                pl.col("red_zone_target").sum().alias("off_ctx_team_red_zone_targets"),
                pl.col("goal_to_go_carry").sum().alias("off_ctx_team_goal_to_go_carries"),
                pl.col("touchdowns").sum().alias("off_ctx_team_touchdowns"),
            ]
        )
        .join(primary_qb, on=["season", "week", "team"], how="left")
        .with_columns(pl.col("primary_qb_id").fill_null(QB_DEFAULT))
    )

    team_usage = team_usage.sort(["season", "week", "team"])
    team_usage = team_usage.with_columns(
        pl.col("game_date")
        .cast(pl.Datetime("ms"))
        .shift(1)
        .over(["season", "team"])
        .alias("data_as_of")
    )

    team_usage = team_usage.with_columns(
        [
            pl.col("off_ctx_team_red_zone_targets")
            .shift(1)
            .over(["season", "team"])
            .alias("off_ctx_team_red_zone_targets_prev"),
            pl.col("off_ctx_team_goal_to_go_carries")
            .shift(1)
            .over(["season", "team"])
            .alias("off_ctx_team_goal_to_go_carries_prev"),
            pl.col("off_ctx_team_touchdowns")
            .shift(1)
            .over(["season", "team"])
            .alias("off_ctx_team_touchdowns_prev"),
        ]
    )

    # Rolling team-level metrics (previous games only)
    team_usage = team_usage.with_columns(
        [
            pl.col("off_ctx_team_red_zone_targets_prev")
            .rolling_mean(window_size=3, min_periods=1)
            .over(["season", "team"])
            .alias("off_ctx_team_red_zone_targets_l3"),
            pl.col("off_ctx_team_goal_to_go_carries_prev")
            .rolling_mean(window_size=3, min_periods=1)
            .over(["season", "team"])
            .alias("off_ctx_team_goal_to_go_carries_l3"),
            pl.col("off_ctx_team_touchdowns_prev")
            .rolling_mean(window_size=3, min_periods=1)
            .over(["season", "team"])
            .alias("off_ctx_team_touchdowns_l3"),
        ]
    )

    # Coordinator aggregations (previous games)
    team_usage = team_usage.with_columns(
        [
            pl.col("off_ctx_team_red_zone_targets_prev")
            .rolling_mean(window_size=3, min_periods=1)
            .over(["season", "offensive_coordinator"])
            .alias("off_ctx_coord_red_zone_targets_l3"),
            pl.col("off_ctx_team_goal_to_go_carries_prev")
            .rolling_mean(window_size=3, min_periods=1)
            .over(["season", "offensive_coordinator"])
            .alias("off_ctx_coord_goal_to_go_carries_l3"),
            pl.col("off_ctx_team_touchdowns_prev")
            .rolling_mean(window_size=3, min_periods=1)
            .over(["season", "offensive_coordinator"])
            .alias("off_ctx_coord_touchdowns_l3"),
        ]
    )

    # Quarterback aggregations (previous games)
    team_usage = team_usage.with_columns(
        [
            pl.col("off_ctx_team_red_zone_targets_prev")
            .rolling_mean(window_size=3, min_periods=1)
            .over(["season", "primary_qb_id"])
            .alias("off_ctx_qb_red_zone_targets_l3"),
            pl.col("off_ctx_team_goal_to_go_carries_prev")
            .rolling_mean(window_size=3, min_periods=1)
            .over(["season", "primary_qb_id"])
            .alias("off_ctx_qb_goal_to_go_carries_l3"),
            pl.col("off_ctx_team_touchdowns_prev")
            .rolling_mean(window_size=3, min_periods=1)
            .over(["season", "primary_qb_id"])
            .alias("off_ctx_qb_touchdowns_l3"),
        ]
    )

    # Vacated usage calculation (from players marked OUT/INA)
    if "status" in df.columns:
        # Check if required rolling columns exist
        required_rolling = ["3g_target_per_game", "3g_carry_per_game"]
        
        # Optional extended rolling features for smarter vacated usage
        extended_rolling = ["3g_red_zone_target_per_game", "3g_goal_to_go_carry_per_game"]
        
        if not all(col in df.columns for col in required_rolling):
            logger.warning(
                "Missing rolling features %s for vacated usage calculation. Filling with 0.", 
                [c for c in required_rolling if c not in df.columns]
            )
            df = df.with_columns([
                pl.lit(0.0).alias("vacated_targets_position"),
                pl.lit(0.0).alias("vacated_carries_position"),
                pl.lit(0.0).alias("vacated_rz_targets_position"),
                pl.lit(0.0).alias("vacated_gl_carries_position"),
            ])
        else:
            df = df.with_columns(_build_is_unavailable_expr(df).alias("is_unavailable"))
            
            # Build aggregation expressions
            agg_exprs = [
                pl.col("3g_target_per_game").sum().alias("vacated_targets_position"),
                pl.col("3g_carry_per_game").sum().alias("vacated_carries_position")
            ]
            
            # Add extended features if available
            if "3g_red_zone_target_per_game" in df.columns:
                agg_exprs.append(pl.col("3g_red_zone_target_per_game").sum().alias("vacated_rz_targets_position"))
            
            if "3g_goal_to_go_carry_per_game" in df.columns:
                agg_exprs.append(pl.col("3g_goal_to_go_carry_per_game").sum().alias("vacated_gl_carries_position"))

            # Calculate team-level vacated targets/carries by position group
            vacated_usage = (
                df.filter(pl.col("is_unavailable"))
                .group_by(["season", "week", "team", "position"])
                .agg(agg_exprs)
            )
            
            if not vacated_usage.is_empty():
                logger.info("Calculated vacated usage for %d position groups", len(vacated_usage))
            
            # Join vacated usage back to active players
            df = df.join(
                vacated_usage,
                on=["season", "week", "team", "position"],
                how="left"
            ).with_columns([
                pl.col("vacated_targets_position").fill_null(0.0),
                pl.col("vacated_carries_position").fill_null(0.0),
                pl.col("vacated_rz_targets_position").fill_null(0.0) if "vacated_rz_targets_position" in vacated_usage.columns else pl.lit(0.0).alias("vacated_rz_targets_position"),
                pl.col("vacated_gl_carries_position").fill_null(0.0) if "vacated_gl_carries_position" in vacated_usage.columns else pl.lit(0.0).alias("vacated_gl_carries_position"),
            ])
    else:
        df = df.with_columns([
            pl.lit(0.0).alias("vacated_targets_position"),
            pl.lit(0.0).alias("vacated_carries_position"),
            pl.lit(0.0).alias("vacated_rz_targets_position"),
            pl.lit(0.0).alias("vacated_gl_carries_position"),
        ])

    # Join back to player frame
    df = df.join(
        team_usage.select(
            [
                "season",
                "week",
                "team",
                "offensive_coordinator",
                "primary_qb_id",
                "data_as_of",
                "off_ctx_team_red_zone_targets_prev",
                "off_ctx_team_goal_to_go_carries_prev",
                "off_ctx_team_touchdowns_prev",
                "off_ctx_team_red_zone_targets_l3",
                "off_ctx_team_goal_to_go_carries_l3",
                "off_ctx_team_touchdowns_l3",
                "off_ctx_coord_red_zone_targets_l3",
                "off_ctx_coord_goal_to_go_carries_l3",
                "off_ctx_coord_touchdowns_l3",
                "off_ctx_qb_red_zone_targets_l3",
                "off_ctx_qb_goal_to_go_carries_l3",
                "off_ctx_qb_touchdowns_l3",
                "game_date",
            ]
        ),
        on=["season", "week", "team"],
        how="left",
        suffix="_offctx",
    )
    rename_map = {"data_as_of": "off_ctx_data_as_of"}
    if "game_date_offctx" in df.columns:
        rename_map["game_date_offctx"] = "off_ctx_game_date"
    df = df.rename(rename_map)

    df = df.sort(["team", "player_id", "game_date"])
    if "off_ctx_data_as_of" in df.columns:
        df = df.with_columns(pl.col("off_ctx_data_as_of").cast(pl.Datetime("ms")))
    
    player_prev_red_zone = (
        pl.col("red_zone_target")
        .fill_null(0.0)
        .cast(pl.Float32)
        .shift(1)
        .over(["team", "player_id"])
    )
    player_l3_red_zone = (
        pl.col("red_zone_target")
        .fill_null(0.0)
        .cast(pl.Float32)
        .rolling_mean(window_size=3, min_periods=1)
        .shift(1)
        .over(["team", "player_id"])
    )
    player_prev_goal = (
        pl.col("goal_to_go_carry")
        .fill_null(0.0)
        .cast(pl.Float32)
        .shift(1)
        .over(["team", "player_id"])
    )
    player_l3_goal = (
        pl.col("goal_to_go_carry")
        .fill_null(0.0)
        .cast(pl.Float32)
        .rolling_mean(window_size=3, min_periods=1)
        .shift(1)
        .over(["team", "player_id"])
    )
    
    share_exprs: list[pl.Expr] = []
    if "off_ctx_team_red_zone_targets_prev" in df.columns:
        share_exprs.append(
            pl.when(pl.col("off_ctx_team_red_zone_targets_prev") > 0)
            .then(player_prev_red_zone / pl.col("off_ctx_team_red_zone_targets_prev"))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("off_ctx_player_red_zone_share_prev")
        )
    if "off_ctx_team_red_zone_targets_l3" in df.columns:
        share_exprs.append(
            pl.when(pl.col("off_ctx_team_red_zone_targets_l3") > 0)
            .then(player_l3_red_zone / pl.col("off_ctx_team_red_zone_targets_l3"))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("off_ctx_player_red_zone_share_l3")
        )
        share_exprs.append(
            pl.when(pl.col("off_ctx_team_red_zone_targets_l3") > 0)
            .then(player_l3_red_zone / pl.col("off_ctx_team_red_zone_targets_l3"))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("off_ctx_player_red_zone_share")
        )
    if "off_ctx_team_goal_to_go_carries_prev" in df.columns:
        share_exprs.append(
            pl.when(pl.col("off_ctx_team_goal_to_go_carries_prev") > 0)
            .then(player_prev_goal / pl.col("off_ctx_team_goal_to_go_carries_prev"))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("off_ctx_player_goal_to_go_share_prev")
        )
    if "off_ctx_team_goal_to_go_carries_l3" in df.columns:
        share_exprs.append(
            pl.when(pl.col("off_ctx_team_goal_to_go_carries_l3") > 0)
            .then(player_l3_goal / pl.col("off_ctx_team_goal_to_go_carries_l3"))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("off_ctx_player_goal_to_go_share_l3")
        )
        share_exprs.append(
            pl.when(pl.col("off_ctx_team_goal_to_go_carries_l3") > 0)
            .then(player_l3_goal / pl.col("off_ctx_team_goal_to_go_carries_l3"))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("off_ctx_player_goal_to_go_share")
        )
    if share_exprs:
        df = df.with_columns(share_exprs)

    # Ensure all context columns exist even if null
    context_columns = [
        "off_ctx_team_red_zone_targets_prev",
        "off_ctx_team_goal_to_go_carries_prev",
        "off_ctx_team_touchdowns_prev",
        "off_ctx_team_red_zone_targets_l3",
        "off_ctx_team_goal_to_go_carries_l3",
        "off_ctx_team_touchdowns_l3",
        "off_ctx_coord_red_zone_targets_l3",
        "off_ctx_coord_goal_to_go_carries_l3",
        "off_ctx_coord_touchdowns_l3",
        "off_ctx_qb_red_zone_targets_l3",
        "off_ctx_qb_goal_to_go_carries_l3",
        "off_ctx_qb_touchdowns_l3",
        "off_ctx_player_red_zone_share",
        "off_ctx_player_red_zone_share_prev",
        "off_ctx_player_red_zone_share_l3",
        "off_ctx_player_goal_to_go_share",
        "off_ctx_player_goal_to_go_share_prev",
        "off_ctx_player_goal_to_go_share_l3",
    ]
    df = df.with_columns(
        [
            pl.col(col).fill_null(0.0).cast(pl.Float32)
            for col in context_columns
            if col in df.columns
        ]
    )

    history = team_usage.select(
        [
            "season",
            "week",
            "team",
            "game_date",
            "data_as_of",
            "offensive_coordinator",
            "primary_qb_id",
            "off_ctx_team_red_zone_targets_prev",
            "off_ctx_team_goal_to_go_carries_prev",
            "off_ctx_team_touchdowns_prev",
            "off_ctx_team_red_zone_targets_l3",
            "off_ctx_team_goal_to_go_carries_l3",
            "off_ctx_team_touchdowns_l3",
            "off_ctx_coord_red_zone_targets_l3",
            "off_ctx_coord_goal_to_go_carries_l3",
            "off_ctx_coord_touchdowns_l3",
            "off_ctx_qb_red_zone_targets_l3",
            "off_ctx_qb_goal_to_go_carries_l3",
            "off_ctx_qb_touchdowns_l3",
        ]
    ).sort(["team", "game_date"])

    return df, history


def add_offense_context_features_training(
    df: pl.DataFrame,
    *,
    history_path: Path,
) -> pl.DataFrame:
    seasons = df.get_column("season").unique().to_list()
    df_with_ctx, history = _append_offense_context_columns(df, seasons)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.write_parquet(history_path, compression="zstd")
    logger.info("Offense context history written → %s (%d rows)", history_path, len(history))
    return df_with_ctx


def add_offense_context_features_inference(
    df: pl.DataFrame,
    *,
    history_path: Path,
    cutoff_column: str | None = None,
) -> pl.DataFrame:
    if not history_path.exists():
        logger.warning("Offense context history missing at %s; skipping join.", history_path)
        return df
    history = pl.read_parquet(str(history_path))
    history = history.sort(["team", "game_date"])
    if "data_as_of" in history.columns:
        history = history.with_columns(pl.col("data_as_of").cast(pl.Datetime("ms")))

    join_columns = [
        "team",
        "game_date",
        "offensive_coordinator",
        "primary_qb_id",
        "off_ctx_team_red_zone_targets_prev",
        "off_ctx_team_goal_to_go_carries_prev",
        "off_ctx_team_touchdowns_prev",
        "off_ctx_team_red_zone_targets_l3",
        "off_ctx_team_goal_to_go_carries_l3",
        "off_ctx_team_touchdowns_l3",
        "off_ctx_coord_red_zone_targets_l3",
        "off_ctx_coord_goal_to_go_carries_l3",
        "off_ctx_coord_touchdowns_l3",
        "off_ctx_qb_red_zone_targets_l3",
        "off_ctx_qb_goal_to_go_carries_l3",
        "off_ctx_qb_touchdowns_l3",
    ]
    join_columns = [col for col in join_columns if col in history.columns]
    target_ts_col = cutoff_column if cutoff_column and cutoff_column in df.columns else "game_date"
    history_join = history.select(join_columns + ["data_as_of"]).rename({"data_as_of": "__off_ctx_data_as_of"})

    if "game_date" in df.columns and df["game_date"].dtype != history["game_date"].dtype:
        history_join = history_join.with_columns(pl.col("game_date").cast(df["game_date"].dtype))

    joined = df.with_columns(
        pl.col(target_ts_col).cast(pl.Datetime("ms")).alias("__off_ctx_target_ts")
        if target_ts_col in df.columns
        else pl.col("game_date").cast(pl.Datetime("ms")).alias("__off_ctx_target_ts")
    ).join_asof(
        history_join,
        left_on="__off_ctx_target_ts",
        right_on="__off_ctx_data_as_of",
        by="team",
        strategy="backward",
        suffix="_offctx",
    ).rename({"__off_ctx_data_as_of": "off_ctx_data_as_of"}).drop("__off_ctx_target_ts")

    context_columns = [
        "off_ctx_team_red_zone_targets_prev",
        "off_ctx_team_goal_to_go_carries_prev",
        "off_ctx_team_touchdowns_prev",
        "off_ctx_team_red_zone_targets_l3",
        "off_ctx_team_goal_to_go_carries_l3",
        "off_ctx_team_touchdowns_l3",
        "off_ctx_coord_red_zone_targets_l3",
        "off_ctx_coord_goal_to_go_carries_l3",
        "off_ctx_coord_touchdowns_l3",
        "off_ctx_qb_red_zone_targets_l3",
        "off_ctx_qb_goal_to_go_carries_l3",
        "off_ctx_qb_touchdowns_l3",
    ]

    joined = joined.with_columns(
        [
            pl.col(col).fill_null(0.0).cast(pl.Float32)
            for col in context_columns
            if col in joined.columns
        ]
    )
    if "off_ctx_data_as_of" in joined.columns:
        joined = joined.with_columns(pl.col("off_ctx_data_as_of").cast(pl.Datetime("ms")))
        if cutoff_column and cutoff_column in df.columns:
            potential = (
                joined.filter(
                    pl.col("off_ctx_data_as_of").is_not_null()
                    & pl.col(cutoff_column).is_not_null()
                    & (pl.col("off_ctx_data_as_of") >= pl.col(cutoff_column))
                ).height
            )
            if potential > 0:
                logger.debug(
                    "off_ctx_data_as_of >= cutoff detected for %d rows; earlier filtering enforced.",
                    potential,
                )
    
    share_exprs: list[pl.Expr] = []
    if {
        "1g_red_zone_target_per_game",
        "off_ctx_team_red_zone_targets_prev",
    }.issubset(set(joined.columns)):
        share_exprs.append(
            pl.when(pl.col("off_ctx_team_red_zone_targets_prev") > 0)
            .then(
                pl.col("1g_red_zone_target_per_game")
                / pl.col("off_ctx_team_red_zone_targets_prev")
            )
            .otherwise(0.0)
            .clip(0.0, 1.5)
            .cast(pl.Float32)
            .alias("off_ctx_player_red_zone_share_prev")
        )
    if {
        "3g_red_zone_target_per_game",
        "off_ctx_team_red_zone_targets_l3",
    }.issubset(set(joined.columns)):
        expr = (
            pl.when(pl.col("off_ctx_team_red_zone_targets_l3") > 0)
            .then(
                pl.col("3g_red_zone_target_per_game")
                / pl.col("off_ctx_team_red_zone_targets_l3")
            )
            .otherwise(0.0)
            .clip(0.0, 1.5)
            .cast(pl.Float32)
        )
        share_exprs.append(expr.alias("off_ctx_player_red_zone_share_l3"))
        share_exprs.append(expr.alias("off_ctx_player_red_zone_share"))
    if {
        "1g_goal_to_go_carry_per_game",
        "off_ctx_team_goal_to_go_carries_prev",
    }.issubset(set(joined.columns)):
        share_exprs.append(
            pl.when(pl.col("off_ctx_team_goal_to_go_carries_prev") > 0)
            .then(
                pl.col("1g_goal_to_go_carry_per_game")
                / pl.col("off_ctx_team_goal_to_go_carries_prev")
            )
            .otherwise(0.0)
            .clip(0.0, 1.5)
            .cast(pl.Float32)
            .alias("off_ctx_player_goal_to_go_share_prev")
        )
    if {
        "3g_goal_to_go_carry_per_game",
        "off_ctx_team_goal_to_go_carries_l3",
    }.issubset(set(joined.columns)):
        expr_goal = (
            pl.when(pl.col("off_ctx_team_goal_to_go_carries_l3") > 0)
            .then(
                pl.col("3g_goal_to_go_carry_per_game")
                / pl.col("off_ctx_team_goal_to_go_carries_l3")
            )
            .otherwise(0.0)
            .clip(0.0, 1.5)
            .cast(pl.Float32)
        )
        share_exprs.append(expr_goal.alias("off_ctx_player_goal_to_go_share_l3"))
        share_exprs.append(expr_goal.alias("off_ctx_player_goal_to_go_share"))
    
    if share_exprs:
        joined = joined.with_columns(share_exprs)

    return joined

