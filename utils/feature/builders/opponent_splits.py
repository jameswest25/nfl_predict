from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import polars as pl

try:
    from nfl_data_py import import_weekly_rosters
except ImportError:
    import_weekly_rosters = None

logger = logging.getLogger(__name__)

PLAY_DIR = Path("data/processed/play_by_week")
OPPONENT_SPLIT_DIR = Path("data/processed/opponent_splits")
ROSTER_CACHE_DIR = Path("cache/feature/rosters")


def _load_rosters_for_years(years: Iterable[int]) -> pl.DataFrame:
    """Load weekly roster data for the given years with caching."""
    if not years:
        return pl.DataFrame()

    unique_years = sorted(set(int(y) for y in years))
    frames: list[pl.DataFrame] = []
    
    ROSTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for year in unique_years:
        cache_path = ROSTER_CACHE_DIR / f"roster_{year}.parquet"
        if cache_path.exists():
            roster_pl = pl.read_parquet(cache_path)
        else:
            if import_weekly_rosters is None:
                logger.warning("nfl_data_py not available; cannot fetch rosters.")
                continue
            try:
                roster_pd = import_weekly_rosters(years=[year])
                roster_pl = pl.from_pandas(roster_pd)
                # Ensure minimal columns exist
                if "gsis_id" in roster_pl.columns:
                    roster_pl = roster_pl.rename({"gsis_id": "player_id"})
                
                # Cache it
                roster_pl.write_parquet(cache_path, compression="zstd")
            except Exception as exc:
                logger.warning("Failed to load roster for %s: %s", year, exc)
                continue
        
        if not roster_pl.is_empty():
            frames.append(roster_pl)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="diagonal_relaxed")


def _load_play_level(start_date: date, end_date: date) -> pl.DataFrame:
    """Load play-level data for the requested window."""
    if not PLAY_DIR.exists():
        logger.warning("Play-level directory %s missing; run build_play_level first.", PLAY_DIR)
        return pl.DataFrame()

    # Read parquet files manually and concat with diagonal_relaxed to handle schema inconsistencies
    parquet_files = list(PLAY_DIR.glob("season=*/week=*/part.parquet"))
    if not parquet_files:
        logger.warning("No play-level parquet files found in %s", PLAY_DIR)
        return pl.DataFrame()
    
    # Read all files and concat with relaxed schema handling
    frames = []
    for f in parquet_files:
        try:
            df = pl.read_parquet(f)
            # Cast categorical columns to string to avoid schema conflicts
            for col in df.columns:
                if df[col].dtype == pl.Categorical:
                    df = df.with_columns(pl.col(col).cast(pl.Utf8))
            frames.append(df)
        except Exception as e:
            logger.warning("Failed to read %s: %s", f, e)
            continue
    
    if not frames:
        return pl.DataFrame()
    
    # Concat with relaxed schema handling
    scan = pl.concat(frames, how="diagonal_relaxed").lazy()

    if "game_date" not in scan.collect_schema().names():
        logger.warning("Play-level data missing game_date column; opponent splits skipped.")
        return pl.DataFrame()

    scan = scan.filter(
        (pl.col("game_date").cast(pl.Utf8) >= start_date.isoformat())
        & (pl.col("game_date").cast(pl.Utf8) <= end_date.isoformat())
    )
    cols_required = [
        "season",
        "week",
        "game_id",
        "game_date",
        "defteam",
        "yards_gained",
        "success",
        "epa",
        "rush_attempt",
        "pass_attempt",
        "pass",
        "rush",
        "was_pressure",
        "defense_man_zone_type",
        "defense_coverage_type",
        "defenders_in_box",
        "number_of_pass_rushers",
        "situ_two_minute",
        "situ_red_zone",
        "situ_third_down",
        "situ_fourth_down",
        "situ_blitz",
        "situ_light_box",
        "situ_heavy_box",
        "situ_pressure",
        "situ_pass_expected",
        "situ_run_expected",
        "situ_goal_line",
        "receiver_player_id",  # Added for position splits
        "pass_location",       # Added for directional splits
    ]
    available = set(scan.collect_schema().names())
    missing = [col for col in cols_required if col not in available]
    if missing:
        logger.warning(
            "Opponent splits missing source columns: %s",
            ", ".join(sorted(missing)),
        )
    select_cols = [col for col in cols_required if col in available]
    if not select_cols:
        return pl.DataFrame()

    df = scan.select(select_cols).collect()
    if df.is_empty():
        return df

    df = df.with_columns(
        [
            pl.col("defteam").cast(pl.Utf8),
            pl.col("success").cast(pl.Float32).fill_null(0.0),
            pl.col("epa").cast(pl.Float32).fill_null(0.0),
            pl.col("yards_gained").cast(pl.Float32).fill_null(0.0),
            pl.col("rush_attempt").fill_null(0).cast(pl.Int8),
            pl.col("pass_attempt").fill_null(0).cast(pl.Int8),
            pl.col("was_pressure").fill_null(False).cast(pl.Int8),
            pl.col("defenders_in_box").cast(pl.Float32),
            pl.col("number_of_pass_rushers").cast(pl.Float32),
        ]
    )
    return df


def _build_pass_metrics(df: pl.DataFrame, roster: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "pass_attempt" not in df.columns:
        return pl.DataFrame()

    pass_df = df.filter(pl.col("pass_attempt") == 1)
    if pass_df.is_empty():
        return pl.DataFrame()
    
    # Join roster to get receiver positions if available
    if not roster.is_empty() and "receiver_player_id" in pass_df.columns:
        # Prepare roster for join
        roster_key = roster.select(
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("player_id").cast(pl.Utf8).alias("receiver_player_id"),
            pl.col("position").cast(pl.Utf8)
        ).unique(subset=["season", "week", "receiver_player_id"])
        
        pass_df = pass_df.join(
            roster_key, 
            on=["season", "week", "receiver_player_id"], 
            how="left"
        )
    else:
        pass_df = pass_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("position"))

    agg_exprs = [
        pl.count().alias("opp_def_pass_plays"),
        pl.col("success").mean().alias("opp_def_pass_success_allowed"),
        pl.col("epa").mean().alias("opp_def_pass_epa_allowed"),
        (pl.col("yards_gained") >= 20.0).mean().alias("opp_def_pass_explosive_rate"),
        pl.when(pl.col("situ_two_minute") == 1).then(1).otherwise(0).mean().alias("opp_def_pass_two_minute_rate"),
        pl.when(pl.col("situ_third_down") == 1).then(1).otherwise(0).mean().alias("opp_def_pass_third_down_rate"),
        pl.when(pl.col("situ_fourth_down") == 1).then(1).otherwise(0).mean().alias("opp_def_pass_fourth_down_rate"),
        pl.when(pl.col("situ_red_zone") == 1).then(1).otherwise(0).mean().alias("opp_def_pass_red_zone_rate"),
        pl.when(pl.col("situ_goal_line") == 1).then(1).otherwise(0).mean().alias("opp_def_pass_goal_line_rate"),
        pl.when(pl.col("situ_pass_expected") == 1).then(1).otherwise(0).mean().alias("opp_def_pass_expected_rate"),
        pl.when(pl.col("situ_blitz") == 1).then(1).otherwise(0).mean().alias("opp_def_blitz_rate"),
        pl.when(pl.col("situ_pressure") == 1).then(1).otherwise(0).mean().alias("opp_def_pressure_rate"),
        pl.when(pl.col("defenders_in_box").is_not_null())
        .then(pl.col("defenders_in_box"))
        .otherwise(None)
        .mean()
        .alias("opp_def_pass_box_avg"),
        pl.when(pl.col("defense_man_zone_type").is_not_null())
        .then(
            pl.col("defense_man_zone_type")
            .str.to_lowercase()
            .str.strip_chars()
            == "man"
        )
        .otherwise(False)
        .mean()
        .alias("opp_def_man_rate"),
        pl.when(pl.col("defense_man_zone_type").is_not_null())
        .then(
            pl.col("defense_man_zone_type")
            .str.to_lowercase()
            .str.strip_chars()
            == "zone"
        )
        .otherwise(False)
        .mean()
        .alias("opp_def_zone_rate"),
        pl.when(pl.col("defense_coverage_type").is_not_null())
        .then(
            pl.col("defense_coverage_type")
            .str.to_lowercase()
            .str.contains("two|2", literal=False)
        )
        .otherwise(False)
        .mean()
        .alias("opp_def_two_high_rate"),
        pl.col("number_of_pass_rushers")
        .fill_null(0.0)
        .mean()
        .alias("opp_def_pass_rushers_avg"),
    ]
    
    # Add position-specific splits
    for pos in ["WR", "TE", "RB"]:
        agg_exprs.append(
            pl.when(pl.col("position") == pos)
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias(f"opp_def_pass_epa_allowed_vs_{pos}")
        )
        agg_exprs.append(
            pl.when(pl.col("position") == pos)
            .then(pl.col("success"))
            .otherwise(None)
            .mean()
            .alias(f"opp_def_pass_success_allowed_vs_{pos}")
        )

    # Add directional splits (left, middle, right)
    for loc in ["left", "middle", "right"]:
        agg_exprs.append(
            pl.when(pl.col("pass_location") == loc)
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias(f"opp_def_pass_epa_allowed_{loc}")
        )
        agg_exprs.append(
            pl.when(pl.col("pass_location") == loc)
            .then(pl.col("success"))
            .otherwise(None)
            .mean()
            .alias(f"opp_def_pass_success_allowed_{loc}")
        )

    return (
        pass_df.group_by(["season", "week", "defteam"])
        .agg(agg_exprs)
        .with_columns(
            [
                pl.col("opp_def_pass_success_allowed").fill_null(0.0),
                pl.col("opp_def_pass_epa_allowed").fill_null(0.0),
            ] + [
                pl.col(f"opp_def_pass_epa_allowed_vs_{pos}").fill_null(0.0) for pos in ["WR", "TE", "RB"]
            ] + [
                pl.col(f"opp_def_pass_success_allowed_vs_{pos}").fill_null(0.0) for pos in ["WR", "TE", "RB"]
            ] + [
                pl.col(f"opp_def_pass_epa_allowed_{loc}").fill_null(0.0) for loc in ["left", "middle", "right"]
            ] + [
                pl.col(f"opp_def_pass_success_allowed_{loc}").fill_null(0.0) for loc in ["left", "middle", "right"]
            ]
        )
    )


def _build_rush_metrics(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "rush_attempt" not in df.columns:
        return pl.DataFrame()

    rush_df = df.filter(pl.col("rush_attempt") == 1)
    if rush_df.is_empty():
        return pl.DataFrame()

    return (
        rush_df.group_by(["season", "week", "defteam"])
        .agg(
            [
                pl.count().alias("opp_def_rush_plays"),
                pl.col("success").mean().alias("opp_def_rush_success_allowed"),
                pl.col("epa").mean().alias("opp_def_rush_epa_allowed"),
                (pl.col("yards_gained") <= 0).mean().alias("opp_def_rush_stuff_rate"),
                pl.when(pl.col("yards_gained") >= 10.0).then(1).otherwise(0).mean().alias("opp_def_rush_explosive_rate"),
                pl.when(pl.col("situ_red_zone") == 1).then(1).otherwise(0).mean().alias("opp_def_rush_red_zone_rate"),
                pl.when(pl.col("situ_goal_line") == 1).then(1).otherwise(0).mean().alias("opp_def_rush_goal_line_rate"),
                pl.when(pl.col("situ_light_box") == 1).then(1).otherwise(0).mean().alias("opp_def_light_box_rate"),
                pl.when(pl.col("situ_heavy_box") == 1).then(1).otherwise(0).mean().alias("opp_def_heavy_box_rate"),
                pl.when(pl.col("situ_run_expected") == 1).then(1).otherwise(0).mean().alias("opp_def_run_expected_rate"),
            ]
        )
        .with_columns(
            [
                pl.col("opp_def_rush_success_allowed").fill_null(0.0),
                pl.col("opp_def_rush_epa_allowed").fill_null(0.0),
            ]
        )
    )


def build_opponent_splits(*, start_date: date, end_date: date) -> None:
    """Aggregate defensive opponent context features."""
    df = _load_play_level(start_date, end_date)
    if df.is_empty():
        logger.warning("Opponent splits: no play-level data found for %s – %s", start_date, end_date)
        return

    # Load rosters for the relevant seasons
    seasons = df.select(pl.col("season").unique()).to_series().to_list()
    roster = _load_rosters_for_years(seasons)

    base = (
        df.select(["season", "week", "defteam", "game_date"])
        .group_by(["season", "week", "defteam"])
        .agg(
            [
                pl.col("game_date").max().alias("game_date"),
            ]
        )
    )

    pass_metrics = _build_pass_metrics(df, roster)
    rush_metrics = _build_rush_metrics(df)

    combined = base
    if not pass_metrics.is_empty():
        combined = combined.join(pass_metrics, on=["season", "week", "defteam"], how="left")
    if not rush_metrics.is_empty():
        combined = combined.join(rush_metrics, on=["season", "week", "defteam"], how="left")

    if combined.is_empty():
        logger.warning("Opponent splits aggregation produced no rows.")
        return

    metric_cols = [col for col in combined.columns if col not in {"season", "week", "defteam", "game_date"}]
    if metric_cols:
        combined = combined.with_columns([pl.col(col).cast(pl.Float32).fill_null(0.0) for col in metric_cols])

    OPPONENT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for (season, week), sub in combined.group_by(["season", "week"], maintain_order=True):
        out_dir = OPPONENT_SPLIT_DIR / f"season={int(season)}" / f"week={int(week)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part.parquet"
        out_path.unlink(missing_ok=True)
        sub.write_parquet(out_path, compression="zstd")
        logger.info(
            "Opponent splits written for season %s week %s → %s (%d rows)",
            season,
            week,
            out_path,
            len(sub),
        )


def load_rolling_opponent_splits(
    seasons: list[int],
    windows: list[int] = [3],
) -> pl.DataFrame:
    """
    Load all opponent splits, sort by date, and compute rolling averages per team.
    
    Returns a DataFrame keyed by (opponent, game_date) containing:
      - 'opp_def_pass_epa_allowed_prev'
      - 'opp_def_pass_epa_allowed_l3'
      - etc.
    
    These are safe to join via asof join on game_date (strategy=backward).
    """
    if not seasons:
        return pl.DataFrame()
        
    # 1. Collect all paths
    paths = []
    for season in seasons:
        season_dir = OPPONENT_SPLIT_DIR / f"season={season}"
        if season_dir.exists():
            paths.extend(season_dir.glob("week=*/part.parquet"))
            
    if not paths:
        logger.warning("No opponent split partitions found for seasons %s", seasons)
        return pl.DataFrame()
        
    # 2. Read files manually and concat with diagonal_relaxed to handle schema inconsistencies
    frames = []
    for path in paths:
        try:
            sub = pl.read_parquet(path)
            # Cast columns that may have inconsistent types
            for col in sub.columns:
                if sub[col].dtype == pl.Categorical:
                    sub = sub.with_columns(pl.col(col).cast(pl.Utf8))
            # Standardize season to Int32
            if "season" in sub.columns:
                sub = sub.with_columns(pl.col("season").cast(pl.Int32))
            frames.append(sub)
        except Exception as e:
            logger.warning("Failed to read opponent split %s: %s", path, e)
            continue
    
    if not frames:
        return pl.DataFrame()
    
    df = pl.concat(frames, how="diagonal_relaxed")
    
    # Rename defteam -> opponent for consistency
    if "defteam" in df.columns:
        df = df.rename({"defteam": "opponent"})
    if df.is_empty():
        return df
        
    # 3. Sort for rolling
    df = df.with_columns(pl.col("game_date").cast(pl.Datetime("ms")))
    df = df.sort(["opponent", "game_date"])
    
    # 4. Compute rolling features
    # We want "prev" (lag 1) and "l3" (lag 1, then rolling 3)
    
    metric_cols = [
        col for col in df.columns 
        if col.startswith("opp_def_") 
        and df[col].dtype in (pl.Float32, pl.Float64)
    ]
    
    rolling_exprs = []
    
    for col in metric_cols:
        # Previous game (Lag 1)
        rolling_exprs.append(
            pl.col(col).shift(1).over("opponent").alias(f"{col}_prev")
        )
        
        # Last 3 games average (Lag 1, then roll 3)
        for w in windows:
            rolling_exprs.append(
                pl.col(col)
                .shift(1)
                .rolling_mean(window_size=w, min_periods=1)
                .over("opponent")
                .alias(f"{col}_l{w}")
            )
            
    df_rolling = df.with_columns(rolling_exprs)
    
    # 5. Keep only keys + rolling cols
    keep_cols = ["season", "week", "opponent", "game_date"] + [
        expr.meta.output_name() for expr in rolling_exprs
    ]
    
    # Filter out rows where "prev" is null (first game of history for a team)
    # Or keep them as null/0? 
    # Better to fill with null so model knows it's missing, or 0 if that's neutral.
    # EPA 0 is neutral. Rates 0 is... bad?
    # Let's fill with None (null) and let XGBoost handle it, or 0 if we must.
    # Existing code filled with 0.
    
    return df_rolling.select(keep_cols)


__all__ = ["build_opponent_splits", "load_rolling_opponent_splits"]
