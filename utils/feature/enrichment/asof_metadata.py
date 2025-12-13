from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import polars as pl

from utils.feature.enrichment.asof import decision_cutoff_hours_for_season_type, fallback_cutoff_hours
from utils.feature.enrichment.odds import NFL_ODDS_COLUMNS
from utils.general.paths import (
    ASOF_METADATA_PATH,
    ODDS_SNAPSHOT_DIR,
    PROJ_ROOT,
    ROSTER_SNAPSHOT_DIR,
    WEATHER_FORECAST_DIR,
)
from utils.collect.nfl_schedules import get_schedule

logger = logging.getLogger(__name__)

INJURY_CACHE_DIR = PROJ_ROOT / "cache" / "feature" / "injuries"

ASOF_METADATA_COLUMNS: Sequence[str] = (
    "game_id",
    "season",
    "week",
    "game_date",
    "game_start_utc",
    "cutoff_ts",
    "injury_snapshot_ts",
    "injury_snapshot_source",
    "injury_reports_used",
    "roster_snapshot_ts",
    "roster_snapshot_is_approximate",
    "roster_snapshot_lag_hours",
    "odds_snapshot_ts",
    "odds_snapshot_source",
    "odds_snapshot_generated_utc",
    "forecast_snapshot_ts",
    "forecast_snapshot_source",
    "metadata_generated_utc",
)


def _compute_cutoff(
    game_row: pl.DataFrame,
) -> pl.DataFrame:
    return game_row.with_columns(
        pl.when(pl.col("game_start_utc").is_not_null())
        .then(
            pl.col("game_start_utc")
            - pl.duration(
                hours=pl.col("decision_cutoff_hours").fill_null(fallback_cutoff_hours())
            )
        )
        .otherwise(
            pl.col("game_date")
            .cast(pl.Datetime("ms", "UTC"))
            - pl.duration(hours=fallback_cutoff_hours())
        )
        .alias("cutoff_ts")
    )


def _load_schedule(seasons: Iterable[int]) -> pl.DataFrame:
    schedule = get_schedule(sorted(set(seasons)))
    if schedule.empty:
        return pl.DataFrame(
            {
                "game_id": [],
                "season": [],
                "week": [],
                "game_date": [],
                "start_time_utc": [],
                "season_type": [],
            }
        )

    schedule = schedule.copy()
    if "gameday" in schedule.columns:
        gameday_series = schedule["gameday"]
    elif "game_date" in schedule.columns:
        gameday_series = schedule["game_date"]
    else:
        gameday_series = pd.NaT
    schedule["gameday"] = pd.to_datetime(gameday_series, errors="coerce")
    if "start_time_utc" in schedule.columns:
        schedule["start_time_utc"] = pd.to_datetime(schedule["start_time_utc"], utc=True, errors="coerce")
    else:
        schedule["start_time_utc"] = pd.NaT

    schedule["game_date"] = schedule["gameday"].dt.date
    if "season_type" in schedule.columns:
        season_type_series = schedule["season_type"]
    elif "game_type" in schedule.columns:
        season_type_series = schedule["game_type"]
    else:
        season_type_series = "REG"
    schedule["season_type"] = season_type_series
    schedule = schedule.drop_duplicates(subset=["game_id"])

    df = pl.from_pandas(
        schedule[
            [
                "game_id",
                "season",
                "week",
                "game_date",
                "start_time_utc",
                "season_type",
                "home_team",
                "away_team",
            ]
        ]
    )
    df = df.with_columns(
        [
            pl.col("game_date").cast(pl.Date),
            pl.col("start_time_utc")
            .cast(pl.Datetime("ms", "UTC"))
            .alias("game_start_utc"),
            pl.col("season_type").cast(pl.Utf8),
        ]
    )
    df = df.with_columns(
        pl.col("season_type")
        .map_elements(
            lambda st: decision_cutoff_hours_for_season_type((st or "").upper()),
            return_dtype=pl.Float64,
        )
        .alias("decision_cutoff_hours")
    )
    df = _compute_cutoff(df)
    return df


def _load_injury_metadata(base: pl.DataFrame) -> pl.DataFrame:
    if not INJURY_CACHE_DIR.exists():
        logger.warning("Injury cache directory missing at %s", INJURY_CACHE_DIR)
        return pl.DataFrame(
            {
                "game_id": pl.Series([], dtype=pl.Utf8),
                "injury_snapshot_ts": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
                "injury_snapshot_source": pl.Series([], dtype=pl.Utf8),
                "injury_reports_used": pl.Series([], dtype=pl.Int32),
            }
        )

    frames: list[pl.DataFrame] = []
    for season in sorted(base.get_column("season").unique().to_list()):
        cache_path = INJURY_CACHE_DIR / f"injury_{season}.parquet"
        if cache_path.exists():
            try:
                frame = pl.read_parquet(cache_path)
                required_cols = ("game_id", "season", "week", "reported_at")
                missing = [col for col in required_cols if col not in frame.columns]
                for col in missing:
                    frame = frame.with_columns(pl.lit(None).alias(col))
                frame = frame.select(list(required_cols))
                frame = frame.with_columns(
                    [
                        pl.col("game_id").cast(pl.Utf8),
                        pl.col("season").cast(pl.Int32),
                        pl.col("week").cast(pl.Int32),
                        pl.col("reported_at").cast(pl.Datetime("ms", "UTC")),
                    ]
                )
                frames.append(frame)
            except Exception as exc:
                logger.warning("Failed to read injury cache %s: %s", cache_path, exc)
        else:
            logger.debug("Injury cache missing for season %s at %s", season, cache_path)

    if not frames:
        return pl.DataFrame(
            {
                "game_id": pl.Series([], dtype=pl.Utf8),
                "injury_snapshot_ts": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
                "injury_snapshot_source": pl.Series([], dtype=pl.Utf8),
                "injury_reports_used": pl.Series([], dtype=pl.Int32),
            }
        )

    injuries = pl.concat(frames, how="vertical_relaxed")
    if "game_id" not in injuries.columns or "reported_at" not in injuries.columns:
        return pl.DataFrame(
            {
                "game_id": pl.Series([], dtype=pl.Utf8),
                "injury_snapshot_ts": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
                "injury_snapshot_source": pl.Series([], dtype=pl.Utf8),
                "injury_reports_used": pl.Series([], dtype=pl.Int32),
            }
        )

    injuries = injuries.with_columns(
        pl.col("reported_at").cast(pl.Datetime("ms", "UTC"))
    ).filter(pl.col("reported_at").is_not_null())

    merged = injuries.join(
        base.select(["game_id", "cutoff_ts"]),
        on="game_id",
        how="inner",
    )
    snapshot = (
        merged.filter(pl.col("reported_at") <= pl.col("cutoff_ts"))
        .group_by("game_id")
        .agg(
            [
                pl.col("reported_at").max().alias("injury_snapshot_ts"),
                pl.count().alias("injury_reports_used"),
            ]
        )
        .with_columns(
            pl.when(pl.col("injury_snapshot_ts").is_null())
            .then(pl.lit("none"))
            .otherwise(pl.lit("historical"))
            .alias("injury_snapshot_source")
        )
    )
    return snapshot.select(
        ["game_id", "injury_snapshot_ts", "injury_snapshot_source", "injury_reports_used"]
    )


def _load_roster_metadata(base: pl.DataFrame) -> pl.DataFrame:
    if base.is_empty():
        return pl.DataFrame(
            {
                "game_id": pl.Series([], dtype=pl.Utf8),
                "roster_snapshot_ts": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
                "roster_snapshot_is_approximate": pl.Series([], dtype=pl.Boolean),
                "roster_snapshot_lag_hours": pl.Series([], dtype=pl.Float64),
            }
        )

    snapshot_paths = list(ROSTER_SNAPSHOT_DIR.glob("season=*/week=*/part.parquet"))
    if not snapshot_paths:
        logger.warning("Roster snapshot directory is empty at %s", ROSTER_SNAPSHOT_DIR)
        return (
            base.select(["game_id", "cutoff_ts"])
            .with_columns(
                pl.col("cutoff_ts").alias("roster_snapshot_ts"),
                pl.lit(True).alias("roster_snapshot_is_approximate"),
                pl.lit(0.0).cast(pl.Float64).alias("roster_snapshot_lag_hours"),
            )
            .select(
                [
                    "game_id",
                    "roster_snapshot_ts",
                    "roster_snapshot_is_approximate",
                    "roster_snapshot_lag_hours",
                ]
            )
        )

    frames: list[pl.DataFrame] = []
    for path in snapshot_paths:
        try:
            df = pl.read_parquet(path)
        except Exception as exc:
            logger.warning("Failed to read roster snapshot %s: %s", path, exc)
            continue

        required = ["game_id", "team", "snapshot_ts"]
        for col in required:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))

        df = df.select(required).with_columns(
            [
                pl.col("game_id").cast(pl.Utf8),
                pl.col("team").cast(pl.Utf8),
                pl.col("snapshot_ts").cast(pl.Datetime("ms", "UTC")),
            ]
        )
        frames.append(df)

    if not frames:
        logger.warning("No roster snapshot frames could be loaded; falling back to cutoff timestamps.")
        return (
            base.select(["game_id", "cutoff_ts"])
            .with_columns(
                pl.col("cutoff_ts").alias("roster_snapshot_ts"),
                pl.lit(True).alias("roster_snapshot_is_approximate"),
                pl.lit(0.0).cast(pl.Float64).alias("roster_snapshot_lag_hours"),
            )
            .select(
                [
                    "game_id",
                    "roster_snapshot_ts",
                    "roster_snapshot_is_approximate",
                    "roster_snapshot_lag_hours",
                ]
            )
        )

    snapshots = pl.concat(frames, how="vertical_relaxed").filter(
        pl.col("game_id").is_not_null() & pl.col("snapshot_ts").is_not_null()
    )
    if snapshots.is_empty():
        return (
            base.select(["game_id", "cutoff_ts"])
            .with_columns(
                pl.col("cutoff_ts").alias("roster_snapshot_ts"),
                pl.lit(True).alias("roster_snapshot_is_approximate"),
                pl.lit(0.0).cast(pl.Float64).alias("roster_snapshot_lag_hours"),
            )
            .select(
                [
                    "game_id",
                    "roster_snapshot_ts",
                    "roster_snapshot_is_approximate",
                    "roster_snapshot_lag_hours",
                ]
            )
        )

    base_keys = base.select(["game_id", "cutoff_ts"])
    merged = snapshots.join(base_keys, on="game_id", how="inner")
    valid = merged.filter(pl.col("snapshot_ts") <= pl.col("cutoff_ts"))

    latest_snapshot = (
        valid.group_by("game_id")
        .agg(pl.col("snapshot_ts").max().alias("roster_snapshot_ts"))
    )

    result = base_keys.join(latest_snapshot, on="game_id", how="left")
    result = result.with_columns(
        pl.col("roster_snapshot_ts").is_not_null().alias("_has_snapshot")
    )
    result = result.with_columns(
        [
            pl.when(pl.col("_has_snapshot"))
            .then(pl.col("roster_snapshot_ts"))
            .otherwise(pl.col("cutoff_ts"))
            .alias("roster_snapshot_ts"),
            (~pl.col("_has_snapshot")).alias("roster_snapshot_is_approximate"),
        ]
    ).drop("_has_snapshot")
    result = result.with_columns(
        (
            (pl.col("cutoff_ts") - pl.col("roster_snapshot_ts"))
            .dt.total_seconds()
            / 3600.0
        ).cast(pl.Float64).alias("roster_snapshot_lag_hours")
    )
    return result.select(
        [
            "game_id",
            "roster_snapshot_ts",
            "roster_snapshot_is_approximate",
            "roster_snapshot_lag_hours",
        ]
    )


def _load_odds_metadata(base: pl.DataFrame) -> pl.DataFrame:
    odds_base_path = ODDS_SNAPSHOT_DIR
    glob_path = odds_base_path / "season=*/week=*/part.parquet"
    if not odds_base_path.exists():
        logger.warning("Odds snapshot directory missing at %s", odds_base_path)
        return pl.DataFrame(
            {
                "game_id": pl.Series([], dtype=pl.Utf8),
                "odds_snapshot_ts": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
                "odds_snapshot_source": pl.Series([], dtype=pl.Utf8),
                "odds_snapshot_generated_utc": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
            }
        )

    paths = list(odds_base_path.glob("season=*/week=*/part.parquet"))
    if not paths:
        return pl.DataFrame(
            {
                "game_id": pl.Series([], dtype=pl.Utf8),
                "odds_snapshot_ts": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
                "odds_snapshot_source": pl.Series([], dtype=pl.Utf8),
                "odds_snapshot_generated_utc": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
            }
        )

    frames: list[pl.DataFrame] = []
    for path in paths:
        try:
            df = pl.read_parquet(path)
        except Exception as exc:
            logger.warning("Failed to read odds snapshot %s: %s", path, exc)
            continue

        # Ensure all expected numeric columns exist and share the same dtype
        numeric_exprs: list[pl.Expr] = []
        missing_numeric: list[str] = []
        for col in NFL_ODDS_COLUMNS:
            if col in df.columns:
                numeric_exprs.append(pl.col(col).cast(pl.Float64))
            else:
                missing_numeric.append(col)
        if numeric_exprs:
            df = df.with_columns(numeric_exprs)
        if missing_numeric:
            df = df.with_columns([pl.lit(None).cast(pl.Float64).alias(col) for col in missing_numeric])

        # Standardise snapshot timestamp columns if present
        ts_columns = [
            "odds_snapshot_ts",
            "odds_snapshot_ts_2h",
            "odds_snapshot_ts_6h",
            "odds_snapshot_ts_24h",
            "odds_snapshot_ts_open",
            "odds_snapshot_generated_utc",
        ]
        ts_exprs = [
            pl.col(col).cast(pl.Utf8)
            for col in ts_columns
            if col in df.columns
        ]
        if ts_exprs:
            df = df.with_columns(ts_exprs)

        frames.append(df)

    if not frames:
        return pl.DataFrame(
            {
                "game_id": [],
                "odds_snapshot_ts": [],
                "odds_snapshot_source": [],
                "odds_snapshot_generated_utc": [],
            }
        )

    odds_df = pl.concat(frames, how="diagonal_relaxed")
    if odds_df.is_empty():
        return pl.DataFrame(
            {
                "game_id": [],
                "odds_snapshot_ts": [],
                "odds_snapshot_source": [],
                "odds_snapshot_generated_utc": [],
            }
        )

    columns = odds_df.columns
    ts_col = "odds_snapshot_ts" if "odds_snapshot_ts" in columns else None
    generated_col = (
        "odds_snapshot_generated_utc"
        if "odds_snapshot_generated_utc" in columns
        else None
    )

    if ts_col is None:
        logger.debug("Odds snapshots missing cutoff timestamp column; metadata will be empty.")
        return pl.DataFrame(
            {
                "game_id": [],
                "odds_snapshot_ts": [],
                "odds_snapshot_source": [],
                "odds_snapshot_generated_utc": [],
            }
        )

    odds_df = odds_df.with_columns(
        [
            pl.col(ts_col).cast(pl.Datetime("ms", "UTC")).alias("odds_snapshot_ts"),
            pl.col(generated_col).cast(pl.Datetime("ms", "UTC")).alias("odds_snapshot_generated_utc")
            if generated_col
            else pl.lit(None).cast(pl.Datetime("ms", "UTC")).alias("odds_snapshot_generated_utc"),
        ]
    )

    odds_meta = (
        odds_df.select(
            [
                "game_id",
                "odds_snapshot_ts",
                "odds_snapshot_generated_utc",
            ]
        )
        .group_by("game_id")
        .agg(
            [
                pl.col("odds_snapshot_ts").max().alias("odds_snapshot_ts"),
                pl.col("odds_snapshot_generated_utc").max().alias("odds_snapshot_generated_utc"),
            ]
        )
        .with_columns(
            pl.when(pl.col("odds_snapshot_ts").is_null())
            .then(pl.lit("none"))
            .otherwise(pl.lit("historical"))
            .alias("odds_snapshot_source")
        )
    )
    return odds_meta.select(
        ["game_id", "odds_snapshot_ts", "odds_snapshot_source", "odds_snapshot_generated_utc"]
    )


def _load_forecast_metadata(base: pl.DataFrame) -> pl.DataFrame:
    if not WEATHER_FORECAST_DIR.exists():
        logger.warning("Weather forecast directory missing at %s", WEATHER_FORECAST_DIR)
        return pl.DataFrame(
            {
                "game_id": pl.Series([], dtype=pl.Utf8),
                "forecast_snapshot_ts": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
                "forecast_snapshot_source": pl.Series([], dtype=pl.Utf8),
            }
        )

    paths = list(WEATHER_FORECAST_DIR.glob("season=*/week=*/part.parquet"))
    if not paths:
        return pl.DataFrame(
            {
                "game_id": pl.Series([], dtype=pl.Utf8),
                "forecast_snapshot_ts": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
                "forecast_snapshot_source": pl.Series([], dtype=pl.Utf8),
            }
        )

    scan = pl.scan_parquet(
        [str(path) for path in paths],
        hive_partitioning=True,
    )
    forecast = scan.collect()
    if forecast.is_empty():
        return pl.DataFrame(
            {
                "game_id": pl.Series([], dtype=pl.Utf8),
                "forecast_snapshot_ts": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
                "forecast_snapshot_source": pl.Series([], dtype=pl.Utf8),
            }
        )

    required_cols = [
        "game_id",
        "forecast_generated_ts",
        "forecast_provider",
        "forecast_is_backfill",
        "forecast_is_historical",
        "forecast_source_detail",
    ]
    missing = [col for col in required_cols if col not in forecast.columns]
    if missing:
        forecast = forecast.select([col for col in forecast.columns if col in required_cols])
        for col in missing:
            if col == "forecast_source_detail":
                forecast = forecast.with_columns(pl.lit(None).alias(col))
            elif col in {"forecast_is_backfill", "forecast_is_historical"}:
                forecast = forecast.with_columns(pl.lit(False).alias(col))

    forecast = forecast.with_columns(
        [
            pl.col("game_id").cast(pl.Utf8),
            pl.col("forecast_generated_ts").cast(pl.Datetime("ms", "UTC")),
            pl.col("forecast_source_detail").cast(pl.Utf8).alias("forecast_source_detail"),
        ]
    )

    joined = (
        forecast.join(base.select(["game_id", "cutoff_ts"]), on="game_id", how="inner")
        .filter(pl.col("forecast_generated_ts") <= pl.col("cutoff_ts"))
        .sort(["game_id", "forecast_generated_ts"])
    )

    if joined.is_empty():
        return pl.DataFrame(
            {
                "game_id": pl.Series([], dtype=pl.Utf8),
                "forecast_snapshot_ts": pl.Series([], dtype=pl.Datetime("ms", "UTC")),
                "forecast_snapshot_source": pl.Series([], dtype=pl.Utf8),
            }
        )

    agg = joined.group_by("game_id").agg(
        [
            pl.col("forecast_generated_ts").last().alias("forecast_snapshot_ts"),
            pl.col("forecast_provider").last().alias("forecast_provider"),
            pl.col("forecast_is_backfill").last().alias("forecast_is_backfill"),
            pl.col("forecast_is_historical").last().alias("forecast_is_historical"),
            pl.col("forecast_source_detail").last().alias("forecast_source_detail"),
        ]
    )

    agg = agg.with_columns(
        [
            pl.col("forecast_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
            pl.when(pl.col("forecast_snapshot_ts").is_null())
            .then(pl.lit("none"))
            .otherwise(
                pl.when(pl.col("forecast_is_backfill") | pl.col("forecast_is_historical"))
                .then(pl.lit("backfill"))
                .otherwise(
                    pl.col("forecast_source_detail")
                    .fill_null(pl.col("forecast_provider"))
                    .fill_null("forecast")
                )
            )
            .alias("forecast_snapshot_source"),
        ]
    )

    return agg.select(["game_id", "forecast_snapshot_ts", "forecast_snapshot_source"])


def build_asof_metadata(
    seasons: Iterable[int],
    *,
    force: bool = True,
) -> pl.DataFrame:
    """Build the central as-of metadata table."""
    seasons = list({int(season) for season in seasons})
    if not seasons:
        raise ValueError("No seasons provided for as-of metadata build.")

    base = _load_schedule(seasons)
    if base.is_empty():
        logger.warning("Schedule empty for seasons %s; metadata not generated.", seasons)
        return base

    injury_meta = _load_injury_metadata(base)
    roster_meta = _load_roster_metadata(base)
    odds_meta = _load_odds_metadata(base)
    forecast_meta = _load_forecast_metadata(base)

    metadata = (
        base.join(injury_meta, on="game_id", how="left")
        .join(roster_meta, on="game_id", how="left")
        .join(odds_meta, on="game_id", how="left")
        .join(forecast_meta, on="game_id", how="left")
    )

    if "forecast_snapshot_ts" in metadata.columns:
        metadata = metadata.with_columns(
            pl.col("forecast_snapshot_ts").cast(pl.Datetime("ms", "UTC"))
        )
    else:
        metadata = metadata.with_columns(
            pl.lit(None).cast(pl.Datetime("ms", "UTC")).alias("forecast_snapshot_ts")
        )

    if "forecast_snapshot_source" in metadata.columns:
        metadata = metadata.with_columns(
            pl.col("forecast_snapshot_source").fill_null("none")
        )
    else:
        metadata = metadata.with_columns(pl.lit("none").alias("forecast_snapshot_source"))

    metadata = metadata.with_columns(
        pl.lit(dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)).alias("metadata_generated_utc")
    )

    metadata = (
        metadata.select(ASOF_METADATA_COLUMNS)
        .unique(subset=["game_id"], keep="last")
    )

    if force or not ASOF_METADATA_PATH.exists():
        ASOF_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_parquet(ASOF_METADATA_PATH, compression="zstd")
        logger.info(
            "As-of metadata written → %s (%d rows)",
            ASOF_METADATA_PATH,
            len(metadata),
        )
    else:
        logger.info("As-of metadata build completed (not persisted).")

    return metadata


def load_asof_metadata() -> pl.DataFrame:
    if not ASOF_METADATA_PATH.exists():
        logger.warning("As-of metadata file missing at %s", ASOF_METADATA_PATH)
        return pl.DataFrame()
    return pl.read_parquet(str(ASOF_METADATA_PATH))

