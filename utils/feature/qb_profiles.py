from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Sequence

import polars as pl

from utils.general.paths import QB_PROFILE_DIR, QB_PROFILE_TEAM_DIR

logger = logging.getLogger(__name__)

PLAY_DIR = Path("data/processed/play_by_week")


QB_PROFILE_FEATURES: list[str] = [
    "qb_profile_dropbacks",
    "qb_profile_pass_attempts",
    "qb_profile_completion_rate",
    "qb_profile_pass_success_rate",
    "qb_profile_pass_yards_per_attempt",
    "qb_profile_pass_epa_per_dropback",
    "qb_profile_scramble_rate",
    "qb_profile_shotgun_rate",
    "qb_profile_no_huddle_rate",
    "qb_profile_pressure_rate",
    "qb_profile_deep_attempt_rate",
    "qb_profile_short_attempt_rate",
    "qb_profile_avg_air_yards",
    "qb_profile_median_air_yards",
    "qb_profile_avg_xpass",
    "qb_profile_rush_attempts",
    "qb_profile_rush_yards_per_attempt",
    "qb_profile_rush_success_rate",
    "qb_profile_rush_touchdown_rate",
    "qb_profile_games",
]

TEAM_FEATURE_PREFIX = "qb_profile_team_"


def _ensure_columns(df: pl.DataFrame, defaults: dict[str, pl.Expr]) -> pl.DataFrame:
    for col, expr in defaults.items():
        if col not in df.columns:
            df = df.with_columns(expr.alias(col))
    return df


def _load_play_level(start_date: date, end_date: date) -> pl.DataFrame:
    if not PLAY_DIR.exists():
        logger.warning("Play-level directory %s missing; quarterback profiles skipped.", PLAY_DIR)
        return pl.DataFrame()

    scan = pl.scan_parquet(
        str(PLAY_DIR / "season=*/week=*/part.parquet"),
        glob=True,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    schema = scan.collect_schema().names()
    if "game_date" not in schema:
        logger.warning("Play-level data missing 'game_date'; quarterback profiles skipped.")
        return pl.DataFrame()

    scan = scan.filter(
        (pl.col("game_date").cast(pl.Utf8) >= start_date.isoformat())
        & (pl.col("game_date").cast(pl.Utf8) <= end_date.isoformat())
    )

    columns_needed = [
        "season",
        "week",
        "game_id",
        "game_date",
        "posteam",
        "passer_player_id",
        "passing_yards",
        "qb_dropback",
        "pass_attempt",
        "completion",
        "air_yards",
        "epa",
        "success",
        "shotgun",
        "no_huddle",
        "qb_scramble",
        "situ_pressure",
        "xpass",
        "rusher_player_id",
        "rushing_yards",
        "rush_touchdown",
        "carry",
    ]
    available_cols = [col for col in columns_needed if col in schema]
    df = scan.select(available_cols).collect(streaming=True)
    if df.is_empty():
        return df

    defaults = {
        "qb_dropback": pl.lit(0.0).cast(pl.Float32),
        "pass_attempt": pl.lit(0.0).cast(pl.Float32),
        "completion": pl.lit(0.0).cast(pl.Float32),
        "air_yards": pl.lit(None).cast(pl.Float32),
        "epa": pl.lit(0.0).cast(pl.Float32),
        "success": pl.lit(0.0).cast(pl.Float32),
        "shotgun": pl.lit(0.0).cast(pl.Float32),
        "no_huddle": pl.lit(0.0).cast(pl.Float32),
        "qb_scramble": pl.lit(0.0).cast(pl.Float32),
        "situ_pressure": pl.lit(0).cast(pl.Int8),
        "xpass": pl.lit(None).cast(pl.Float32),
        "rush_touchdown": pl.lit(0.0).cast(pl.Float32),
        "rushing_yards": pl.lit(0.0).cast(pl.Float32),
        "carry": pl.lit(0.0).cast(pl.Float32),
    }
    df = _ensure_columns(df, defaults)
    df = df.with_columns(
        [
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("game_date").cast(pl.Date),
            pl.col("posteam").cast(pl.Utf8).alias("team"),
        ]
    )
    return df


def _prepare_pass_frame(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    pass_df = df.filter(pl.col("passer_player_id").is_not_null())
    if pass_df.is_empty():
        return pass_df
    return pass_df.with_columns(
        [
            pl.col("qb_dropback").fill_null(0.0).cast(pl.Float32).alias("dropback_flag"),
            pl.col("pass_attempt").fill_null(0.0).cast(pl.Float32).alias("pass_attempt_flag"),
            pl.col("completion").fill_null(0.0).cast(pl.Float32).alias("completion_flag"),
            pl.col("passing_yards").fill_null(0.0).cast(pl.Float32).alias("pass_yards"),
            pl.col("epa").fill_null(0.0).cast(pl.Float32).alias("pass_epa"),
            pl.col("success").fill_null(0.0).cast(pl.Float32).alias("pass_success"),
            pl.col("air_yards").cast(pl.Float32).alias("air_yards"),
            pl.when(pl.col("air_yards").is_not_null())
            .then((pl.col("air_yards") >= 15.0).cast(pl.Float32))
            .otherwise(0.0)
            .alias("deep_flag"),
            pl.when(pl.col("air_yards").is_not_null())
            .then((pl.col("air_yards") <= 0.0).cast(pl.Float32))
            .otherwise(0.0)
            .alias("short_flag"),
            pl.col("qb_scramble").fill_null(0.0).cast(pl.Float32).alias("scramble_flag"),
            pl.col("shotgun").fill_null(0.0).cast(pl.Float32).alias("shotgun_flag"),
            pl.col("no_huddle").fill_null(0.0).cast(pl.Float32).alias("no_huddle_flag"),
            pl.col("situ_pressure").fill_null(0).cast(pl.Float32).alias("pressure_flag"),
            pl.col("xpass").cast(pl.Float32).alias("xpass"),
        ]
    )


def _prepare_rush_frame(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    rush_df = df.filter(pl.col("rusher_player_id").is_not_null())
    if rush_df.is_empty():
        return rush_df
    rush_df = rush_df.with_columns(
        [
            pl.col("carry").fill_null(0.0).cast(pl.Float32).alias("rush_attempt_flag"),
            pl.col("rushing_yards").fill_null(0.0).cast(pl.Float32).alias("rush_yards"),
            pl.col("rush_touchdown").fill_null(0.0).cast(pl.Float32).alias("rush_td_flag"),
            pl.col("success").fill_null(0.0).cast(pl.Float32).alias("rush_success"),
        ]
    )
    keep_cols = [
        "season",
        "week",
        "team",
        "game_id",
        "game_date",
        "rusher_player_id",
        "rush_attempt_flag",
        "rush_yards",
        "rush_td_flag",
        "rush_success",
    ]
    existing = [col for col in keep_cols if col in rush_df.columns]
    return rush_df.select(existing)


def _aggregate_pass_metrics(pass_df: pl.DataFrame, group_cols: Sequence[str]) -> pl.DataFrame:
    if pass_df.is_empty():
        return pl.DataFrame()
    return (
        pass_df.group_by(group_cols, maintain_order=True)
        .agg(
            [
                pl.col("game_date").max().alias("game_date"),
                pl.col("game_id").n_unique().cast(pl.Float32).alias("games"),
                pl.col("dropback_flag").sum().alias("dropbacks"),
                pl.col("pass_attempt_flag").sum().alias("pass_attempts"),
                pl.col("completion_flag").sum().alias("completions"),
                pl.col("pass_yards").sum().alias("pass_yards"),
                pl.col("pass_epa").sum().alias("pass_epa"),
                pl.col("pass_success").mean().alias("pass_success_rate"),
                pl.col("air_yards").mean().alias("avg_air_yards"),
                pl.col("air_yards").median().alias("median_air_yards"),
                pl.col("deep_flag").mean().alias("deep_attempt_rate"),
                pl.col("short_flag").mean().alias("short_attempt_rate"),
                pl.col("scramble_flag").sum().alias("scrambles"),
                pl.col("shotgun_flag").mean().alias("shotgun_rate"),
                pl.col("no_huddle_flag").mean().alias("no_huddle_rate"),
                pl.col("pressure_flag").mean().alias("pressure_rate"),
                pl.col("xpass").mean().alias("avg_xpass"),
            ]
        )
        .with_columns(
            [
                pl.col("dropbacks").cast(pl.Float32),
                pl.col("pass_attempts").cast(pl.Float32),
                pl.col("completions").cast(pl.Float32),
                pl.col("pass_yards").cast(pl.Float32),
                pl.col("pass_epa").cast(pl.Float32),
                pl.col("games").cast(pl.Float32),
                pl.col("scrambles").cast(pl.Float32),
            ]
        )
    )


def _aggregate_rush_metrics(rush_df: pl.DataFrame, group_cols: Sequence[str]) -> pl.DataFrame:
    if rush_df.is_empty():
        return pl.DataFrame()
    return (
        rush_df.group_by(group_cols, maintain_order=True)
        .agg(
            [
                pl.col("rush_attempt_flag").sum().alias("rush_attempts"),
                pl.col("rush_yards").sum().alias("rush_yards"),
                pl.col("rush_td_flag").sum().alias("rush_tds"),
                pl.col("rush_success").mean().alias("rush_success_rate"),
            ]
        )
        .with_columns(
            [
                pl.col("rush_attempts").cast(pl.Float32),
                pl.col("rush_yards").cast(pl.Float32),
                pl.col("rush_tds").cast(pl.Float32),
            ]
        )
    )


def _derive_rates(stats: pl.DataFrame, prefix: str) -> pl.DataFrame:
    if stats.is_empty():
        return stats

    def _safe_rate(numerator: str, denominator: str, alias: str) -> pl.Expr:
        return (
            pl.when(pl.col(denominator) > 0)
            .then(pl.col(numerator) / pl.col(denominator))
            .otherwise(None)
            .cast(pl.Float32)
            .alias(alias)
        )

    return stats.with_columns(
        [
            _safe_rate("completions", "pass_attempts", "completion_rate"),
            _safe_rate("pass_yards", "pass_attempts", "pass_yards_per_attempt"),
            _safe_rate("pass_epa", "dropbacks", "pass_epa_per_dropback"),
            _safe_rate("scrambles", "dropbacks", "scramble_rate"),
            _safe_rate("rush_attempts", "dropbacks", "rush_attempt_rate"),
            _safe_rate("rush_yards", "rush_attempts", "rush_yards_per_attempt"),
            _safe_rate("rush_tds", "rush_attempts", "rush_touchdown_rate"),
        ]
        + [
            pl.col("pass_success_rate").cast(pl.Float32),
            pl.col("deep_attempt_rate").cast(pl.Float32),
            pl.col("short_attempt_rate").cast(pl.Float32),
            pl.col("shotgun_rate").cast(pl.Float32),
            pl.col("no_huddle_rate").cast(pl.Float32),
            pl.col("pressure_rate").cast(pl.Float32),
            pl.col("avg_air_yards").cast(pl.Float32),
            pl.col("median_air_yards").cast(pl.Float32),
            pl.col("avg_xpass").cast(pl.Float32),
            pl.col("rush_success_rate").cast(pl.Float32),
        ]
    ).rename(
        lambda name: f"{prefix}{name}" if name not in {"season", "week", "team", "passer_player_id", "game_date"} else name
    )


def _add_lagged_features(
    df: pl.DataFrame,
    metrics: Sequence[str],
    group_cols: Sequence[str],
    prefix: str,
) -> pl.DataFrame:
    if df.is_empty():
        return df
    sort_cols = list(group_cols) + ["week"]
    df = df.sort(sort_cols)
    data_as_of_name = f"{prefix}data_as_of"
    df = df.with_columns(
        pl.col("game_date")
        .cast(pl.Datetime("ms"))
        .shift(1)
        .over(group_cols)
        .alias(data_as_of_name)
    )
    lag_exprs: list[pl.Expr] = []
    for metric in metrics:
        prev_name = f"{prefix}{metric}_prev"
        l3_name = f"{prefix}{metric}_l3"
        lag_exprs.extend(
            [
                pl.col(f"{prefix}{metric}")
                .shift(1)
                .over(group_cols)
                .alias(prev_name),
                pl.col(f"{prefix}{metric}")
                .shift(1)
                .rolling_mean(window_size=3, min_periods=1)
                .over(group_cols)
                .alias(l3_name),
            ]
        )
    df = df.with_columns(lag_exprs)
    keep_cols = {"season", "week", "team", data_as_of_name}
    if "passer_player_id" in df.columns:
        keep_cols.add("passer_player_id")
    if "qb_id" in df.columns:
        keep_cols.add("qb_id")
    lag_cols = [f"{prefix}{metric}_prev" for metric in metrics] + [f"{prefix}{metric}_l3" for metric in metrics]
    keep_cols.update(lag_cols)
    return df.select([col for col in df.columns if col in keep_cols])


def _write_partitioned(df: pl.DataFrame, base_dir: Path) -> None:
    if df.is_empty():
        return
    for (season, week), sub in df.group_by(["season", "week"], maintain_order=True):
        out_dir = base_dir / f"season={int(season)}" / f"week={int(week)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "part.parquet"
        out_file.unlink(missing_ok=True)
        sub.write_parquet(out_file, compression="zstd")
        logger.debug("Wrote QB profile partition → %s (%d rows)", out_file, len(sub))


def build_qb_profiles(*, start_date: date, end_date: date) -> None:
    """Build quarterback style profiles and team fallback metrics."""
    raw = _load_play_level(start_date, end_date)
    if raw.is_empty():
        logger.warning("QB profiles: no play-level data available for %s – %s", start_date, end_date)
        return

    pass_df = _prepare_pass_frame(raw)
    rush_df = _prepare_rush_frame(raw)

    qb_group_cols = ["season", "week", "team", "passer_player_id"]
    qb_pass_stats = _aggregate_pass_metrics(pass_df, qb_group_cols)
    qb_rush_stats = _aggregate_rush_metrics(
        rush_df.rename({"rusher_player_id": "passer_player_id"}),
        qb_group_cols,
    )
    qb_stats = qb_pass_stats.join(qb_rush_stats, on=qb_group_cols, how="left").fill_null(0.0)

    qb_stats = _derive_rates(qb_stats, prefix="qb_profile_")
    qb_stats = qb_stats.rename({"passer_player_id": "qb_id"})
    qb_metrics = [name.split("qb_profile_")[1] for name in QB_PROFILE_FEATURES]
    qb_profiles = _add_lagged_features(
        qb_stats,
        metrics=qb_metrics,
        group_cols=["season", "team", "qb_id"],
        prefix="qb_profile_",
    )

    team_group_cols = ["season", "week", "team"]
    team_pass_stats = _aggregate_pass_metrics(pass_df, team_group_cols)
    team_rush_stats = _aggregate_rush_metrics(rush_df, team_group_cols)
    team_stats = team_pass_stats.join(team_rush_stats, on=team_group_cols, how="left").fill_null(0.0)
    team_stats = _derive_rates(team_stats, prefix=TEAM_FEATURE_PREFIX)
    team_metrics = qb_metrics
    team_profiles = _add_lagged_features(
        team_stats,
        metrics=team_metrics,
        group_cols=["season", "team"],
        prefix=TEAM_FEATURE_PREFIX,
    )

    combined = qb_profiles.join(team_profiles, on=["season", "week", "team"], how="left")

    if combined.is_empty():
        logger.warning("QB profiles aggregation produced zero rows.")
        return

    _write_partitioned(combined, QB_PROFILE_DIR)
    _write_partitioned(team_profiles, QB_PROFILE_TEAM_DIR)

    logger.info(
        "QB profiles built for %d rows across %d seasons (%s → %s).",
        len(combined),
        len(combined.select("season").unique()),
        start_date,
        end_date,
    )


__all__ = ["build_qb_profiles", "QB_PROFILE_FEATURES", "QB_PROFILE_DIR", "QB_PROFILE_TEAM_DIR"]

