from __future__ import annotations

"""Play-level feature builder.

Reads cleaned daily Parquets and writes weekly play-level Parquets at
  data/processed/play_by_week/season=YYYY/week=W/part.parquet
"""

from pathlib import Path
from datetime import date
import polars as pl

CLEAN_DIR   = Path("data/cleaned")
PLAY_OUTDIR = Path("data/processed/play_by_week")

PLAY_COLS = [
    # Game identifiers
    "game_id",
    "season",
    "week",
    "game_date",
    "utc_ts",
    "game_start_utc",
    "yardline_100",
    "goal_to_go",
    "qtr",
    "down",
    "ydstogo",
    "series",
    "series_success",
    "series_result",
    "drive",
    "play_id",
    "order_sequence",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "quarter_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "wp",
    "vegas_wp",
    "xpass",
    "pass_oe",
    
    # Team context
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    
    # Play context
    "shotgun",
    "no_huddle",
    "qb_dropback",
    "qb_scramble",
    "qb_spike",
    "qb_kneel",
    "pass",
    "rush",
    "play_outcome",
    "success",
    
    # Player IDs and names
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "receiver_player_id",
    "receiver_player_name",
    
    # Personnel and alignment
    "offense_personnel",
    "offense_formation",
    "defense_personnel",
    "defense_man_zone_type",
    "defense_coverage_type",
    "defenders_in_box",
    "number_of_pass_rushers",
    "was_pressure",
    "offense_players",
    "offense_positions",
    "route",
    "pass_location",
    
    # Yardage outcomes
    "yards_gained",
    "rushing_yards",
    "passing_yards",
    "receiving_yards",
    "epa",
    
    # TD outcomes
    "pass_touchdown",
    "rush_touchdown",
    "touchdown",
    "touchdown_player_id",
    
    # Venue metadata
    "stadium_key",
    "stadium_name",
    "stadium_tz",
    "roof",
    "surface",
    
    # Derived stats for rolling windows
    "target",        # Passing target (for receivers)
    "reception",     # Completed reception
    "carry",         # Rushing attempt
    "rush_attempt",
    "pass_attempt",  # QB pass attempt
    "completion",    # Completed pass (for QBs)
    "red_zone_target",   # Red-zone targets (<= 20 yard line)
    "red_zone_carry",    # Red-zone rushing attempts
    "goal_to_go_target", # Goal-to-go targets
    "goal_to_go_carry",  # Goal-to-go carries
]

def build_play_level(*, start_date: date, end_date: date) -> None:
    PLAY_OUTDIR.mkdir(parents=True, exist_ok=True)

    scan = pl.scan_parquet(
        str(CLEAN_DIR / "date=*/part.parquet"),
        glob=True,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )

    # Filter by the hive partition 'date' if present; else fallback to 'game_date'
    schema_names = set(scan.collect_schema().names())
    if "season_type" in schema_names:
        scan = scan.filter(pl.col("season_type") == "REG")

    if "date" in schema_names:
        scan = scan.filter(
            (pl.col("date").cast(pl.Utf8) >= start_date.isoformat()) &
            (pl.col("date").cast(pl.Utf8) <= end_date.isoformat())
        )
    elif "game_date" in schema_names:
        scan = scan.filter(
            (pl.col("game_date").cast(pl.Utf8) >= start_date.isoformat()) &
            (pl.col("game_date").cast(pl.Utf8) <= end_date.isoformat())
        )

    cols = [c for c in PLAY_COLS if c in schema_names]
    df = scan.select(cols).collect(streaming=True)
    if df.is_empty():
        return
    
    # Normalize game_date to Date type
    if 'game_date' in df.columns and df['game_date'].dtype != pl.Date:
        df = df.with_columns(pl.col('game_date').dt.date().alias('game_date'))

    # Situational enrichments and numeric coercions
    df = df.with_columns(
        [
            pl.col("score_differential").cast(pl.Float32).fill_null(0.0).alias("score_differential"),
            (
                pl.col("wp").cast(pl.Float32).clip(0.0, 1.0).alias("wp")
                if "wp" in df.columns
                else pl.lit(None).alias("wp")
            ),
            (
                pl.col("vegas_wp").cast(pl.Float32).clip(0.0, 1.0).alias("vegas_wp")
                if "vegas_wp" in df.columns
                else pl.lit(None).alias("vegas_wp")
            ),
            (
                pl.col("game_seconds_remaining").cast(pl.Float32).fill_null(0.0).alias("game_seconds_remaining")
                if "game_seconds_remaining" in df.columns
                else pl.lit(None).alias("game_seconds_remaining")
            ),
            (
                pl.col("half_seconds_remaining").cast(pl.Float32).fill_null(0.0).alias("half_seconds_remaining")
                if "half_seconds_remaining" in df.columns
                else pl.lit(None).alias("half_seconds_remaining")
            ),
            (
                pl.col("quarter_seconds_remaining").cast(pl.Float32).fill_null(0.0).alias("quarter_seconds_remaining")
                if "quarter_seconds_remaining" in df.columns
                else pl.lit(None).alias("quarter_seconds_remaining")
            ),
            (
                pl.col("ydstogo").cast(pl.Float32).fill_null(0.0).alias("ydstogo")
                if "ydstogo" in df.columns
                else pl.lit(None).alias("ydstogo")
            ),
            (
                pl.col("defenders_in_box").cast(pl.Float32).alias("defenders_in_box")
                if "defenders_in_box" in df.columns
                else pl.lit(None).alias("defenders_in_box")
            ),
            (
                pl.col("number_of_pass_rushers").cast(pl.Float32).alias("number_of_pass_rushers")
                if "number_of_pass_rushers" in df.columns
                else pl.lit(None).alias("number_of_pass_rushers")
            ),
            (
                pl.col("defense_man_zone_type").cast(pl.Utf8).alias("defense_man_zone_type")
                if "defense_man_zone_type" in df.columns
                else pl.lit(None).alias("defense_man_zone_type")
            ),
            (
                pl.col("defense_coverage_type").cast(pl.Utf8).alias("defense_coverage_type")
                if "defense_coverage_type" in df.columns
                else pl.lit(None).alias("defense_coverage_type")
            ),
        ]
    )

    situ_exprs: list[pl.Expr] = []
    score_diff = pl.col("score_differential").fill_null(0.0)
    ydstogo = pl.col("ydstogo").fill_null(0.0)
    yardline = pl.col("yardline_100").cast(pl.Float32).fill_null(100.0)
    half_secs_expr = (
        pl.col("half_seconds_remaining").fill_null(9999.0)
        if "half_seconds_remaining" in df.columns
        else None
    )
    game_secs_expr = (
        pl.col("game_seconds_remaining").fill_null(9999.0)
        if "game_seconds_remaining" in df.columns
        else None
    )

    if half_secs_expr is not None:
        two_minute = (half_secs_expr <= 120.0)
        situ_exprs.append(two_minute.cast(pl.Int8).alias("situ_two_minute"))
    else:
        two_minute = pl.lit(False)

    if game_secs_expr is not None:
        situ_exprs.append((game_secs_expr <= 240.0).cast(pl.Int8).alias("situ_final_four_minutes"))

    situ_exprs.extend(
        [
            ((pl.col("no_huddle").fill_null(0) == 1) | two_minute)
            .cast(pl.Int8)
            .alias("situ_hurry_up"),
            (yardline <= 20.0).cast(pl.Int8).alias("situ_red_zone"),
            ((yardline <= 5.0) | (pl.col("goal_to_go").fill_null(0) == 1))
            .cast(pl.Int8)
            .alias("situ_goal_line"),
            (pl.col("down").fill_null(0) == 3).cast(pl.Int8).alias("situ_third_down"),
            (pl.col("down").fill_null(0) == 4).cast(pl.Int8).alias("situ_fourth_down"),
            (ydstogo <= 2.0).cast(pl.Int8).alias("situ_short_yardage"),
            (ydstogo >= 10.0).cast(pl.Int8).alias("situ_long_yardage"),
            (score_diff > 0.0).cast(pl.Int8).alias("situ_leading"),
            (score_diff < 0.0).cast(pl.Int8).alias("situ_trailing"),
            ((score_diff >= -8.0) & (score_diff < 0.0)).cast(pl.Int8).alias("situ_trailing_one_score"),
            ((score_diff <= 8.0) & (score_diff > 0.0)).cast(pl.Int8).alias("situ_leading_one_score"),
            (score_diff == 0.0).cast(pl.Int8).alias("situ_tied"),
            score_diff.abs().alias("situ_score_diff_abs"),
            pl.when(pl.col("xpass").is_not_null())
            .then(pl.col("xpass") >= 0.60)
            .otherwise(False)
            .cast(pl.Int8)
            .alias("situ_pass_expected"),
            pl.when(pl.col("xpass").is_not_null())
            .then(pl.col("xpass") <= 0.40)
            .otherwise(False)
            .cast(pl.Int8)
            .alias("situ_run_expected"),
            pl.when(pl.col("defenders_in_box").is_not_null())
            .then(pl.col("defenders_in_box") <= 6.0)
            .otherwise(False)
            .cast(pl.Int8)
            .alias("situ_light_box"),
            pl.when(pl.col("defenders_in_box").is_not_null())
            .then(pl.col("defenders_in_box") >= 8.0)
            .otherwise(False)
            .cast(pl.Int8)
            .alias("situ_heavy_box"),
            pl.when(pl.col("defense_man_zone_type").is_not_null())
            .then(
                pl.col("defense_man_zone_type")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .str.strip_chars()
                == "man"
            )
            .otherwise(False)
            .cast(pl.Int8)
            .alias("situ_man_coverage"),
            pl.when(pl.col("defense_man_zone_type").is_not_null())
            .then(
                pl.col("defense_man_zone_type")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .str.strip_chars()
                == "zone"
            )
            .otherwise(False)
            .cast(pl.Int8)
            .alias("situ_zone_coverage"),
            pl.when(pl.col("number_of_pass_rushers").is_not_null())
            .then(pl.col("number_of_pass_rushers") >= 5.0)
            .otherwise(False)
            .cast(pl.Int8)
            .alias("situ_blitz"),
            pl.col("was_pressure").fill_null(False).cast(pl.Int8).alias("situ_pressure"),
        ]
    )
    df = df.with_columns(situ_exprs)
    
    # Add derived stats if missing (for backward compatibility with old cleaned data)
    if "target" not in df.columns:
        df = df.with_columns([
            # Targets: passing play to a receiver
            ((pl.col("receiver_player_id").is_not_null()) & (pl.col("pass").fill_null(0) == 1)).cast(pl.Int8).alias("target"),
            
            # Receptions: completed pass (yards gained or TD)
            ((pl.col("receiving_yards").fill_null(0) > 0) | (pl.col("pass_touchdown").fill_null(0) == 1)).cast(pl.Int8).alias("reception"),
            
            # Carries: rushing attempt
            ((pl.col("rusher_player_id").is_not_null()) & (pl.col("rush").fill_null(0) == 1)).cast(pl.Int8).alias("carry"),
            
            # Pass attempts: QB pass attempt
            ((pl.col("passer_player_id").is_not_null()) & (pl.col("pass").fill_null(0) == 1)).cast(pl.Int8).alias("pass_attempt"),
            
            # Completions: completed pass (yards or TD)
            ((pl.col("passing_yards").fill_null(0) > 0) | (pl.col("pass_touchdown").fill_null(0) == 1)).cast(pl.Int8).alias("completion"),
            
            # Touchdown attribution columns
            pl.when(pl.col("rush_touchdown").fill_null(0) == 1)
              .then(pl.col("rusher_player_id"))
              .when((pl.col("pass_touchdown").fill_null(0) == 1) & pl.col("receiver_player_id").is_not_null())
              .then(pl.col("receiver_player_id"))
              .otherwise(None)
              .alias("touchdown_player_id"),
            pl.col("touchdown_player_id").is_not_null().cast(pl.Int8).alias("touchdown"),
        ])

    if "red_zone_target" not in df.columns:
        if {"target", "yardline_100"} <= schema_names:
            df = df.with_columns(
                (
                    (pl.col("target").fill_null(0) == 1)
                    & (pl.col("yardline_100").cast(pl.Float64) <= 20)
                ).cast(pl.Int8).alias("red_zone_target")
            )
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("red_zone_target"))

    if "red_zone_carry" not in df.columns:
        if {"carry", "yardline_100"} <= schema_names:
            df = df.with_columns(
                (
                    (pl.col("carry").fill_null(0) == 1)
                    & (pl.col("yardline_100").cast(pl.Float64) <= 20)
                ).cast(pl.Int8).alias("red_zone_carry")
            )
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("red_zone_carry"))

    if "goal_to_go_target" not in df.columns:
        if {"target", "goal_to_go"} <= schema_names:
            df = df.with_columns(
                (
                    (pl.col("target").fill_null(0) == 1)
                    & (pl.col("goal_to_go").fill_null(0) == 1)
                ).cast(pl.Int8).alias("goal_to_go_target")
            )
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("goal_to_go_target"))

    if "goal_to_go_carry" not in df.columns:
        if {"carry", "goal_to_go"} <= schema_names:
            df = df.with_columns(
                (
                    (pl.col("carry").fill_null(0) == 1)
                    & (pl.col("goal_to_go").fill_null(0) == 1)
                ).cast(pl.Int8).alias("goal_to_go_carry")
            )
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("goal_to_go_carry"))

    for (s, w), sub in df.group_by(["season","week"], maintain_order=True):
        out_dir = PLAY_OUTDIR / f"season={int(s)}" / f"week={int(w)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "part.parquet").unlink(missing_ok=True)
        sub.write_parquet(out_dir / "part.parquet", compression="zstd")


