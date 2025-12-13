"""Injury feature enrichment for predictions.

This module handles loading and attaching injury report features
for inference.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import polars as pl

from utils.general.paths import PROJ_ROOT
from utils.collect.espn_injuries import collect_espn_injuries
from utils.feature.enrichment.asof import (
    get_decision_cutoff_hours,
    get_fallback_cutoff_hours,
)

logger = logging.getLogger(__name__)

INJURY_CACHE_DIR = PROJ_ROOT / "cache" / "feature" / "injuries"


def ensure_injury_cache(seasons: Iterable[int]) -> None:
    """Ensure injury caches exist and are fresh for the requested seasons.
    
    For the current season, refreshes the cache if it's older than 12 hours.
    """
    import datetime as dt
    
    cache_dir = INJURY_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    current_year = dt.datetime.now().year
    stale_threshold = dt.timedelta(hours=12)
    now = dt.datetime.now()
    
    missing: list[int] = []
    stale: list[int] = []
    
    for season in {int(s) for s in seasons if s is not None}:
        cache_path = cache_dir / f"injury_{int(season)}.parquet"
        if not cache_path.exists():
            missing.append(int(season))
        elif int(season) >= current_year:
            mtime = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
            if now - mtime > stale_threshold:
                stale.append(int(season))
    
    if missing:
        try:
            logger.info("Collecting ESPN injury reports for missing seasons %s", missing)
            collect_espn_injuries(missing, overwrite=False)
        except Exception as exc:
            logger.warning("Failed to collect ESPN injury cache: %s", exc)
    
    if stale:
        try:
            logger.info("Refreshing stale ESPN injury reports for seasons %s", stale)
            collect_espn_injuries(stale, overwrite=True)
        except Exception as exc:
            logger.warning("Failed to refresh ESPN injury cache: %s", exc)


def load_raw_injuries(seasons: Iterable[int]) -> pl.DataFrame | None:
    """Load raw injury data from cache."""
    frames: list[pl.DataFrame] = []
    for season in sorted(set(seasons)):
        cache_path = INJURY_CACHE_DIR / f"injury_{season}.parquet"
        if cache_path.exists():
            frames.append(pl.read_parquet(cache_path))
        else:
            logger.warning("Injury cache missing for season %s at %s", season, cache_path)
    
    if not frames:
        return None
    
    out = pl.concat(frames, how="vertical_relaxed")
    if "gsis_id" in out.columns and "player_id" not in out.columns:
        out = out.rename({"gsis_id": "player_id"})
    
    return out


def attach_injury_features(enriched: pl.DataFrame) -> pl.DataFrame:
    """Attach injury report features to prediction dataframe.
    
    Parameters
    ----------
    enriched : pl.DataFrame
        Player-game dataframe
    
    Returns
    -------
    pl.DataFrame
        Dataframe with injury features added
    """
    required = {"season", "week", "player_id"}
    if not required <= set(enriched.columns):
        return enriched

    seasons = enriched.get_column("season").unique().to_list()
    raw = load_raw_injuries(seasons)
    
    if raw is None or raw.is_empty():
        # Even if we have no raw injury reports, try to backfill from history
        return _attach_injury_history_from_player_games(enriched)

    keys = enriched.select(["season", "week"]).unique()
    meta_cols = [c for c in ["season", "week", "team", "game_date", "game_start_utc"] if c in enriched.columns]
    game_meta = (
        enriched.select(meta_cols)
        .with_columns([
            pl.col("team").cast(pl.Utf8) if "team" in meta_cols else pl.lit(None).cast(pl.Utf8),
            pl.col("game_date").cast(pl.Date) if "game_date" in meta_cols else pl.lit(None).cast(pl.Date),
            pl.col("game_start_utc").cast(pl.Datetime("ms", "UTC")) if "game_start_utc" in meta_cols else pl.lit(None).cast(pl.Datetime("ms", "UTC")),
        ])
        .unique()
    )
    
    available_cols = [c for c in [
        "season", "week", "team", "player_id",
        "report_status", "practice_status",
        "report_primary_injury", "practice_primary_injury",
        "reported_at",
    ] if c in raw.columns]
    
    injury_pl = raw.select(available_cols)
    if "player_id" not in injury_pl.columns:
        return enriched

    cast_exprs: list[pl.Expr] = [
        pl.col("player_id").cast(pl.Utf8),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
    ]
    if "team" in injury_pl.columns:
        cast_exprs.append(
            pl.col("team").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("team")
        )
    if "report_status" in injury_pl.columns:
        cast_exprs.append(
            pl.col("report_status").cast(pl.Utf8).str.strip_chars()
            .str.replace_all(r"\s+", " ", literal=False).str.to_uppercase()
            .alias("report_status_clean")
        )
    if "practice_status" in injury_pl.columns:
        cast_exprs.append(
            pl.col("practice_status").cast(pl.Utf8).str.strip_chars()
            .str.replace_all(r"\s+", " ", literal=False).str.to_uppercase()
            .alias("practice_status_clean")
        )
    if "report_primary_injury" in injury_pl.columns:
        cast_exprs.append(
            pl.col("report_primary_injury").cast(pl.Utf8).str.strip_chars()
            .alias("report_primary_injury")
        )
    if "practice_primary_injury" in injury_pl.columns:
        cast_exprs.append(
            pl.col("practice_primary_injury").cast(pl.Utf8).str.strip_chars()
            .alias("practice_primary_injury")
        )
    if "reported_at" in injury_pl.columns:
        cast_exprs.append(
            pl.col("reported_at").cast(pl.Datetime(time_unit="ns", time_zone="UTC"))
            .alias("reported_at")
        )
    else:
        cast_exprs.append(pl.lit(None).cast(pl.Datetime("ns", "UTC")).alias("reported_at"))

    injury_pl = injury_pl.with_columns(cast_exprs)
    injury_pl = injury_pl.join(keys, on=["season", "week"], how="inner")
    
    if "team" in injury_pl.columns and not game_meta.is_empty():
        join_cols = [c for c in ["season", "week", "team"] if c in injury_pl.columns and c in game_meta.columns]
        injury_pl = injury_pl.join(game_meta, on=join_cols, how="left")
    else:
        injury_pl = injury_pl.with_columns([
            pl.lit(None).cast(pl.Date).alias("game_date"),
            pl.lit(None).cast(pl.Datetime("ms", "UTC")).alias("game_start_utc"),
        ])
    
    # Filter by decision cutoff
    cutoff_hours = float(get_decision_cutoff_hours())
    fallback_hours = float(get_fallback_cutoff_hours())

    injury_pl = injury_pl.with_columns(
        pl.when(pl.col("game_start_utc").is_not_null())
        .then(pl.col("game_start_utc") - pl.duration(hours=cutoff_hours))
        .otherwise(
            pl.when(pl.col("game_date").is_not_null())
            .then(pl.col("game_date").cast(pl.Datetime("ms", "UTC")) - pl.duration(hours=fallback_hours))
            .otherwise(None)
        )
        .alias("decision_cutoff_ts")
    )
    
    injury_pl = injury_pl.filter(
        pl.col("decision_cutoff_ts").is_null()
        | pl.col("reported_at").is_null()
        | (pl.col("reported_at") <= pl.col("decision_cutoff_ts"))
    )
    injury_pl = injury_pl.drop([col for col in ["decision_cutoff_ts"] if col in injury_pl.columns])
    injury_pl = injury_pl.filter(pl.col("player_id").is_not_null())
    injury_pl = injury_pl.sort(["player_id", "season", "week", "reported_at"])
    
    # Add practice flags
    injury_pl = injury_pl.with_columns([
        pl.col("practice_status_clean").fill_null("")
        .str.contains("DID NOT PARTICIPATE").cast(pl.Int8).alias("practice_dnp_flag"),
        pl.col("practice_status_clean").fill_null("")
        .str.contains("LIMITED").cast(pl.Int8).alias("practice_limited_flag"),
        pl.col("practice_status_clean").fill_null("")
        .str.contains("FULL").cast(pl.Int8).alias("practice_full_flag"),
    ])

    # Aggregate by player-week
    aggregated = (
        injury_pl
        .group_by(["season", "week", "player_id"])
        .agg([
            pl.len().alias("_injury_rows"),
            pl.col("report_status_clean").last().alias("injury_report_status"),
            pl.col("practice_status_clean").last().alias("injury_practice_status"),
            pl.col("report_primary_injury").last().alias("injury_report_primary"),
            pl.col("practice_primary_injury").last().alias("injury_practice_primary"),
            pl.col("practice_dnp_flag").fill_null(0).sum().alias("injury_practice_dnp_count"),
            pl.col("practice_limited_flag").fill_null(0).sum().alias("injury_practice_limited_count"),
            pl.col("practice_full_flag").fill_null(0).sum().alias("injury_practice_full_count"),
            pl.col("practice_status_clean").drop_nulls().alias("_injury_practice_status_seq"),
            pl.col("report_status_clean").drop_nulls().alias("_injury_report_status_seq"),
        ])
    )

    if aggregated.is_empty():
        return enriched

    aggregated = aggregated.with_columns([
        pl.when(pl.col("_injury_rows") > 0).then(1.0).otherwise(None).alias("injury_is_listed"),
    ])
    aggregated = aggregated.with_columns([
        pl.col("_injury_practice_status_seq").list.get(0, null_on_oob=True).alias("injury_practice_status_day1"),
        pl.col("_injury_practice_status_seq").list.get(1, null_on_oob=True).alias("injury_practice_status_day2"),
        pl.col("_injury_practice_status_seq").list.get(2, null_on_oob=True).alias("injury_practice_status_day3"),
        pl.col("_injury_practice_status_seq").list.join(">").alias("injury_practice_status_sequence"),
        pl.col("_injury_report_status_seq").list.join(">").alias("injury_report_status_sequence"),
        pl.col("_injury_report_status_seq").list.get(-1, null_on_oob=True).alias("injury_game_designation"),
    ]).drop(["_injury_practice_status_seq", "_injury_report_status_seq", "_injury_rows"])
    
    aggregated = aggregated.with_columns([
        pl.col("injury_game_designation")
        .is_in(["OUT", "DOUBTFUL", "INACTIVE"])
        .cast(pl.Float32).alias("injury_is_inactive_designation")
    ])
    
    # Compute inactive probability
    report_prob = (
        pl.when(pl.col("injury_report_status") == "OUT").then(0.98)
        .when(pl.col("injury_report_status") == "DOUBTFUL").then(0.85)
        .when(pl.col("injury_report_status") == "QUESTIONABLE").then(0.55)
        .when(pl.col("injury_report_status") == "SUSPENDED").then(0.9)
        .otherwise(0.1)
    )
    practice_prob = (
        pl.when(pl.col("injury_practice_status") == "DID NOT PARTICIPATE").then(0.8)
        .when(pl.col("injury_practice_status") == "LIMITED").then(0.5)
        .when(pl.col("injury_practice_status") == "FULL").then(0.1)
        .otherwise(0.15)
    )
    aggregated = aggregated.with_columns([
        report_prob.cast(pl.Float32).alias("_prob_report"),
        practice_prob.cast(pl.Float32).alias("_prob_practice"),
    ])
    aggregated = aggregated.with_columns(
        pl.max_horizontal(pl.col("_prob_report"), pl.col("_prob_practice")).alias("injury_inactive_probability")
    )
    aggregated = aggregated.with_columns(
        pl.when(pl.col("injury_practice_dnp_count").fill_null(0) >= 2)
        .then(
            pl.when((pl.col("injury_inactive_probability") + 0.1) > 1.0)
            .then(1.0).otherwise(pl.col("injury_inactive_probability") + 0.1)
        )
        .otherwise(pl.col("injury_inactive_probability"))
        .alias("injury_inactive_probability")
    )
    aggregated = aggregated.with_columns([
        pl.col("injury_is_listed").cast(pl.Float32),
        pl.col("injury_practice_dnp_count").cast(pl.Float32),
        pl.col("injury_practice_limited_count").cast(pl.Float32),
        pl.col("injury_practice_full_count").cast(pl.Float32),
    ]).drop(["_prob_report", "_prob_practice"])
    aggregated = aggregated.with_columns(
        pl.col("injury_inactive_probability").fill_null(0.1).cast(pl.Float32)
    )

    enriched = enriched.join(aggregated, on=["season", "week", "player_id"], how="left")

    enriched = enriched.with_columns([
        pl.col("injury_is_inactive_designation").fill_null(0.0).alias("injury_is_inactive_designation"),
        pl.col("injury_practice_dnp_count").fill_null(0.0).alias("injury_practice_dnp_count"),
        pl.col("injury_practice_limited_count").fill_null(0.0).alias("injury_practice_limited_count"),
        pl.col("injury_practice_full_count").fill_null(0.0).alias("injury_practice_full_count"),
    ])

    # Attach historical injury priors
    enriched = _attach_injury_history_from_player_games(enriched)
    
    return enriched


def _attach_injury_history_from_player_games(enriched: pl.DataFrame) -> pl.DataFrame:
    """Backfill injury history features from player_game_by_week artifacts."""
    from utils.predict.loaders import load_injury_history_features
    
    required = {"season", "week", "player_id"}
    if not required <= set(enriched.columns):
        return enriched

    seasons = enriched.get_column("season").unique().to_list()
    weeks = enriched.get_column("week").unique().to_list()

    hist_df = load_injury_history_features(seasons, weeks)
    if hist_df.is_empty():
        return enriched

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

    # Join and prefer existing enriched values
    joined = enriched.join(hist_df, on=["season", "week", "player_id"], how="left", suffix="_hist")

    fill_exprs: list[pl.Expr] = []
    for col in injury_hist_cols:
        hist_col = f"{col}_hist"
        if col in joined.columns and hist_col in joined.columns:
            if col == "injury_last_transaction_note":
                fill_exprs.append(
                    pl.when(pl.col(col).is_null() | (pl.col(col) == "UNKNOWN"))
                    .then(pl.col(hist_col)).otherwise(pl.col(col)).alias(col)
                )
            else:
                fill_exprs.append(
                    pl.when(pl.col(col).is_null())
                    .then(pl.col(hist_col)).otherwise(pl.col(col)).alias(col)
                )
        elif hist_col in joined.columns and col not in joined.columns:
            fill_exprs.append(pl.col(hist_col).alias(col))

    if fill_exprs:
        joined = joined.with_columns(fill_exprs)

    # Drop helper columns
    drop_cols = [c for c in joined.columns if c.endswith("_hist")]
    if drop_cols:
        joined = joined.drop(drop_cols)

    return joined
