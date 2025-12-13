"""As-of metadata enrichment for NFL player predictions.

This module handles enriching player-game data with as-of snapshot metadata,
which tracks when various data sources (injuries, rosters, odds, forecasts)
were captured relative to the decision cutoff time.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass

__all__ = [
    "attach_asof_metadata",
    "apply_snapshot_guards",
    "log_snapshot_coverage",
]

logger = logging.getLogger(__name__)


def attach_asof_metadata(
    df: pl.DataFrame,
    *,
    asof_meta: pl.DataFrame,
    drop_missing_snapshots: bool = False,
) -> pl.DataFrame:
    """Attach as-of metadata to player-game dataframe.
    
    Args:
        df: Player-game dataframe with game_id column
        asof_meta: As-of metadata with snapshot timestamps per game
        drop_missing_snapshots: Whether to drop rows missing full pre-cutoff snapshots
    
    Returns:
        Dataframe with as-of snapshot columns attached
    """
    if asof_meta.is_empty():
        return df
    
    logger.info("Attaching as-of metadata...")
    
    # Get game_ids to filter metadata
    game_ids = df.get_column("game_id").unique().to_list() if "game_id" in df.columns else []
    if not game_ids:
        return df
    
    asof_meta = asof_meta.filter(pl.col("game_id").is_in(game_ids))
    
    # Prepare metadata columns
    asof_meta = asof_meta.with_columns([
        pl.col("cutoff_ts").alias("decision_cutoff_ts"),
        pl.col("cutoff_ts").cast(pl.Datetime("ms", "UTC")),
        pl.col("injury_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
        pl.col("roster_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
        pl.col("odds_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
        pl.col("forecast_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
    ])
    
    asof_meta = asof_meta.with_columns(
        pl.col("injury_snapshot_ts").is_null().cast(pl.Int8).alias("injury_snapshot_ts_missing")
    )
    
    # Log coverage
    log_snapshot_coverage(asof_meta)
    
    # Join to main dataframe
    df = df.join(asof_meta, on="game_id", how="left")
    
    # Handle duplicate columns from join
    if "odds_snapshot_ts_right" in df.columns and "odds_snapshot_ts" in df.columns:
        df = (
            df.with_columns(
                pl.coalesce(pl.col("odds_snapshot_ts"), pl.col("odds_snapshot_ts_right")).alias("odds_snapshot_ts")
            )
            .drop("odds_snapshot_ts_right")
        )
    
    # Add missing flags for required snapshot columns
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
        if col in df.columns
    ]
    if missing_flag_exprs:
        df = df.with_columns(missing_flag_exprs)
    
    # Optionally drop rows missing full snapshots
    if drop_missing_snapshots:
        non_null_filters = [
            pl.col(col).is_not_null() for col in required_snapshot_cols if col in df.columns
        ]
        if non_null_filters:
            before_rows = df.height
            df = df.filter(pl.all_horizontal(non_null_filters))
            removed_rows = before_rows - df.height
            if removed_rows > 0:
                logger.warning(
                    "Dropped %d rows lacking full pre-cutoff snapshots (required columns: %s).",
                    removed_rows,
                    ", ".join(required_snapshot_cols),
                )
    
    return df


def log_snapshot_coverage(asof_meta: pl.DataFrame) -> None:
    """Log coverage statistics for snapshot columns."""
    if asof_meta.is_empty() or asof_meta.height == 0:
        return
    
    coverage_frame = asof_meta.select([
        pl.col("injury_snapshot_ts").is_not_null().mean().alias("injury_snapshot_coverage"),
        pl.col("roster_snapshot_ts").is_not_null().mean().alias("roster_snapshot_coverage"),
        pl.col("odds_snapshot_ts").is_not_null().mean().alias("odds_snapshot_coverage"),
        pl.col("forecast_snapshot_ts").is_not_null().mean().alias("forecast_snapshot_coverage"),
    ])
    
    if coverage_frame.height:
        coverage_dict = {k: float(v) for k, v in coverage_frame.to_dicts()[0].items()}
        summary = ", ".join(f"{k.replace('_coverage', '')}: {v*100:.1f}%" for k, v in coverage_dict.items())
        logger.info("As-of snapshot coverage for requested games — %s", summary)
        
        weak_signals = {k: v for k, v in coverage_dict.items() if v < 0.8}
        if weak_signals:
            logger.warning(
                "Snapshot coverage below 80%% for: %s. Recent rebuilds may be missing data; "
                "run collectors or backfills before production scoring.",
                ", ".join(f"{k.replace('_coverage', '')} ({v*100:.1f}%)" for k, v in weak_signals.items()),
            )
    
    # Log per-horizon coverage if available
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
            logger.info(
                "Odds snapshot coverage @ %sh before kickoff: %.1f%% (%d games)",
                row["decision_cutoff_hours"],
                row["odds_snapshot_coverage"] * 100.0,
                row["game_count"],
            )


def apply_snapshot_guards(
    df: pl.DataFrame,
    *,
    audit_dir: Path | None = None,
) -> pl.DataFrame:
    """Apply guards to ensure snapshots are before decision cutoff.
    
    Drops rows where snapshot timestamps are after the decision cutoff,
    as this would represent data leakage.
    
    Args:
        df: DataFrame with decision_cutoff_ts and snapshot columns
        audit_dir: Optional directory to write violation audit files
    
    Returns:
        DataFrame with violating rows removed
    """
    if "decision_cutoff_ts" not in df.columns:
        return df
    
    guard_columns = [
        ("injury_snapshot_ts", "injury"),
        ("roster_snapshot_ts", "roster"),
        ("odds_snapshot_ts", "odds"),
        ("forecast_snapshot_ts", "weather"),
    ]
    
    audit_paths = {
        "roster": audit_dir / "roster_cutoff_violations.parquet" if audit_dir else None,
        "odds": audit_dir / "odds_cutoff_violations.parquet" if audit_dir else None,
    }
    
    for column_name, label in guard_columns:
        if column_name not in df.columns:
            continue
        
        violation_mask = pl.col(column_name).is_not_null() & (
            pl.col(column_name) > pl.col("decision_cutoff_ts")
        )
        violations = df.filter(violation_mask)
        
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
        
        before_guard = df.height
        df = df.filter(
            pl.col(column_name).is_null() | (pl.col(column_name) <= pl.col("decision_cutoff_ts"))
        )
        dropped_guard = before_guard - df.height
        
        if dropped_guard > 0:
            logger.warning(
                "Dropped %d %s rows with snapshots after cutoff.",
                dropped_guard,
                label,
            )
    
    return df


def load_and_build_asof_metadata(
    seasons: list[int],
    game_ids: list[str] | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """Load existing or build new as-of metadata.
    
    Args:
        seasons: List of seasons to include
        game_ids: Optional list of game_ids to filter to
        force: Whether to force rebuilding metadata
    
    Returns:
        As-of metadata dataframe
    """
    from utils.feature.enrichment.asof_metadata import build_asof_metadata, load_asof_metadata
    
    asof_meta = pl.DataFrame()
    
    try:
        asof_meta = load_asof_metadata()
    except Exception as exc:
        logger.warning("Failed to load existing as-of metadata: %s", exc)
        asof_meta = pl.DataFrame()
    
    # Check if we need to rebuild
    needs_rebuild = asof_meta.is_empty()
    if not needs_rebuild and game_ids:
        covered_games = asof_meta.filter(pl.col("game_id").is_in(game_ids)).height
        needs_rebuild = covered_games < len(game_ids)
    
    if needs_rebuild or force:
        try:
            asof_meta = build_asof_metadata(seasons or [int(date.today().year)], force=True)
        except Exception as exc:
            logger.warning("Unable to build as-of metadata: %s", exc)
            asof_meta = pl.DataFrame()
    
    return asof_meta

