"""Player-drive level aggregation for NFL rolling window features.

Aggregates play-by-play data to player-drive level.
Each player gets one row per drive they were involved in.
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import logging
from typing import Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from nfl_data_py import import_weekly_rosters
except Exception:  # pragma: no cover - optional dependency
    import_weekly_rosters = None

from utils.feature.stats import PLAYER_DRIVE_STATS
from utils.general.paths import PROJ_ROOT

logger = logging.getLogger(__name__)

PLAY_DIR = PROJ_ROOT / "data" / "processed" / "play_by_week"
PLAYER_DRIVE_DIR = PROJ_ROOT / "data" / "processed" / "player_drive_by_week"

_PLAY_DATA_CACHE: Optional[pl.DataFrame] = None
def _load_play_level_data() -> pl.DataFrame:
    global _PLAY_DATA_CACHE
    if _PLAY_DATA_CACHE is not None:
        return _PLAY_DATA_CACHE

    tables: list[pa.Table] = []
    for path in sorted(PLAY_DIR.glob("season=*/week=*/part.parquet")):
        pf = pq.ParquetFile(path)
        table = pf.read()
        schema = table.schema
        if "season" in schema.names and schema.field(schema.get_field_index("season")).type != pa.int32():
            table = table.set_column(
                schema.get_field_index("season"),
                "season",
                table.column("season").cast(pa.int32()),
            )
            schema = table.schema
        if "week" in schema.names and schema.field(schema.get_field_index("week")).type != pa.int32():
            table = table.set_column(
                schema.get_field_index("week"),
                "week",
                table.column("week").cast(pa.int32()),
            )
            schema = table.schema
        for idx, col_name in enumerate(schema.names):
            field = schema.field(idx)
            if pa.types.is_dictionary(field.type) or pa.types.is_large_string(field.type):
                table = table.set_column(
                    idx,
                    col_name,
                    table.column(col_name).cast(pa.string()),
                )
                schema = table.schema
            elif col_name in STRING_COLUMNS and field.type != pa.string():
                table = table.set_column(
                    idx,
                    col_name,
                    table.column(col_name).cast(pa.string()),
                )
                schema = table.schema
        tables.append(table)

    if not tables:
        _PLAY_DATA_CACHE = pl.DataFrame()
        return _PLAY_DATA_CACHE

    merged_table = pa.concat_tables(tables, promote=True)
    df = pl.from_arrow(merged_table)
    _PLAY_DATA_CACHE = df
    return df

# Note: PLAYER_DRIVE_STATS imported from utils.feature.stats (single source of truth)

STRING_COLUMNS = (
    "posteam",
    "defteam",
    "team",
    "opponent",
    "passer_player_id",
    "rusher_player_id",
    "receiver_player_id",
    "passer_player_name",
    "rusher_player_name",
    "receiver_player_name",
    "passer_id",
    "rusher_id",
    "receiver_id",
    "posteam_type",
    "touchdown_player_id",
    "game_id",
    "drive",
)


def build_player_drive_level(*, start_date: date, end_date: date) -> None:
    """Aggregate play-level data to player-drive level.
    
    Creates one row per (player, drive) with aggregated stats.
    
    Parameters
    ----------
    start_date : date
        Start date (inclusive)
    end_date : date
        End date (inclusive)
    """
    PLAYER_DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    
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
    df = df.with_columns(
        [
            pl.col(col).cast(pl.Utf8)
            for col, dtype in zip(df.columns, df.dtypes)
            if isinstance(dtype, pl.Categorical)
        ]
    )
    
    logger.info("Loaded %d plays for player-drive aggregation", len(df))
    
    # Build player-drive rows for each player role
    player_drives = []
    
    # --- 1. PASSERS (QBs) ---
    passer_drives = _aggregate_passer_drives(df)
    if not passer_drives.is_empty():
        player_drives.append(passer_drives)
        logger.info("Aggregated %d passer-drive rows", len(passer_drives))
    
    # --- 2. RUSHERS (RBs, QBs, WRs) ---
    rusher_drives = _aggregate_rusher_drives(df)
    if not rusher_drives.is_empty():
        player_drives.append(rusher_drives)
        logger.info("Aggregated %d rusher-drive rows", len(rusher_drives))
    
    # --- 3. RECEIVERS (WRs, TEs, RBs) ---
    receiver_drives = _aggregate_receiver_drives(df)
    if not receiver_drives.is_empty():
        player_drives.append(receiver_drives)
        logger.info("Aggregated %d receiver-drive rows", len(receiver_drives))
    
    if not player_drives:
        logger.warning("No player-drive stats aggregated")
        return
    
    # Combine all roles
    df_all = pl.concat(player_drives, how="diagonal")
    
    # Merge multi-role players (e.g., QB who also rushed)
    df_merged = _merge_multi_role_players(df_all)
    
    logger.info("Final player-drive rows: %d (after merging multi-role players)", len(df_merged))

    if import_weekly_rosters is not None and len(df_merged) > 0:
        try:
            seasons_available = [int(s) for s in df_merged['season'].unique().drop_nulls().to_list()]
            if seasons_available:
                roster_pd = import_weekly_rosters(years=seasons_available)
                roster_pl = pl.from_pandas(roster_pd)
                desired_cols = [
                    'season', 'week', 'team', 'player_id',
                    'position', 'position_group', 'depth_chart_position', 'depth_chart_order'
                ]
                roster_cols = [c for c in desired_cols if c in roster_pl.columns]
                if roster_cols:
                    roster_pl = roster_pl.select(roster_cols).unique(subset=[c for c in ['season','week','team','player_id'] if c in roster_cols])
                    join_keys = [k for k in ['season','week','team','player_id'] if k in roster_pl.columns and k in df_merged.columns]
                    if join_keys:
                        df_schema = df_merged.schema
                        roster_schema = roster_pl.schema
                        string_like_dtypes = {pl.Utf8, pl.Categorical, pl.Null}
                        if hasattr(pl, "Object"):
                            string_like_dtypes.add(pl.Object)
                        keys_to_cast = [
                            key for key in join_keys
                            if (df_schema.get(key) in string_like_dtypes)
                            or (roster_schema.get(key) in string_like_dtypes)
                        ]
                        if keys_to_cast:
                            df_merged = df_merged.with_columns(
                                [pl.col(key).cast(pl.Utf8) for key in keys_to_cast if key in df_merged.columns]
                            )
                            roster_pl = roster_pl.with_columns(
                                [pl.col(key).cast(pl.Utf8) for key in keys_to_cast if key in roster_pl.columns]
                            )

                        df_merged = df_merged.join(roster_pl, on=join_keys, how="left")
                        logger.info("Roster metadata joined for %d drive rows", len(df_merged))
        except Exception as exc:
            logger.warning("Roster enrichment on drives skipped: %s", exc)

    # Ensure critical identifier columns use consistent string types
    str_cols = [c for c in ("player_id", "player_name", "team", "opponent", "game_id") if c in df_merged.columns]
    if str_cols:
        df_merged = df_merged.with_columns([pl.col(c).cast(pl.Utf8) for c in str_cols])

    # Write weekly partitions
    if "season" not in df_merged.columns or "week" not in df_merged.columns:
        logger.error("Missing season/week columns, cannot partition")
        return
    
    for (s, w), sub in df_merged.group_by(["season", "week"], maintain_order=True):
        out_dir = PLAYER_DRIVE_DIR / f"season={int(s)}" / f"week={int(w)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "part.parquet"
        out_file.unlink(missing_ok=True)
        sub.write_parquet(out_file, compression="zstd")
        logger.info("Wrote %d player-drives to %s", len(sub), out_file)


def _aggregate_passer_drives(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate passing stats per (passer, drive)."""
    
    passers = df.filter(pl.col("passer_player_id").is_not_null())
    
    if passers.is_empty():
        return pl.DataFrame()
    
    return (
        passers
        .group_by(["season", "week", "game_id", "game_date", "drive", 
                   "passer_player_id", "passer_player_name", "posteam", "defteam"])
        .agg([
            pl.col("passing_yards").fill_null(0).sum().cast(pl.Float64).alias("passing_yards"),
            pl.col("pass_attempt").fill_null(0).sum().cast(pl.Int64).alias("pass_attempt"),
            pl.col("completion").fill_null(0).sum().cast(pl.Int64).alias("completion"),
            
            # QBs can also rush
            pl.col("rushing_yards").fill_null(0).sum().cast(pl.Float64).alias("rushing_yards"),
            pl.col("carry").fill_null(0).sum().cast(pl.Int64).alias("carry"),
            
            # QBs don't receive
            pl.lit(0.0).cast(pl.Float64).alias("receiving_yards"),
            pl.lit(0).cast(pl.Int64).alias("target"),
            pl.lit(0).cast(pl.Int64).alias("reception"),
            
            # TDs - General
            pl.when(pl.col("touchdown_player_id") == pl.col("passer_player_id"))
              .then(pl.col("touchdown").fill_null(0))
              .otherwise(0)
              .sum()
              .cast(pl.Int64)
              .alias("touchdown"),
            
            # TDs - Specific Types
            pl.when(pl.col("pass_touchdown") == 1).then(1).otherwise(0).sum().cast(pl.Int64).alias("passing_td"),
            pl.when(pl.col("rush_touchdown") == 1).then(1).otherwise(0).sum().cast(pl.Int64).alias("rushing_td_count"),
            pl.lit(0).cast(pl.Int64).alias("receiving_td_count"),  # QBs don't receive
        ])
        .rename({
            "passer_player_id": "player_id",
            "passer_player_name": "player_name",
        })
        .with_columns([
            pl.col("posteam").alias("team"),
            pl.col("defteam").alias("opponent"),
            (pl.col("touchdown") > 0).cast(pl.Int8).alias("anytime_td"),
            pl.col("touchdown").cast(pl.Int64).alias("td_count"),
        ])
    )


def _aggregate_rusher_drives(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate rushing stats per (rusher, drive)."""
    
    rushers = df.filter(pl.col("rusher_player_id").is_not_null())
    
    if rushers.is_empty():
        return pl.DataFrame()
    
    return (
        rushers
        .group_by(["season", "week", "game_id", "game_date", "drive",
                   "rusher_player_id", "rusher_player_name", "posteam", "defteam"])
        .agg([
            pl.col("rushing_yards").fill_null(0).sum().cast(pl.Float64).alias("rushing_yards"),
            pl.col("carry").fill_null(0).sum().cast(pl.Int64).alias("carry"),
            
            # Rushers don't pass or receive typically
            pl.lit(0.0).cast(pl.Float64).alias("passing_yards"),
            pl.lit(0).cast(pl.Int64).alias("pass_attempt"),
            pl.lit(0).cast(pl.Int64).alias("completion"),
            pl.lit(0.0).cast(pl.Float64).alias("receiving_yards"),
            pl.lit(0).cast(pl.Int64).alias("target"),
            pl.lit(0).cast(pl.Int64).alias("reception"),
            
            # TDs - General
            pl.when(pl.col("touchdown_player_id") == pl.col("rusher_player_id"))
              .then(pl.col("touchdown").fill_null(0))
              .otherwise(0)
              .sum()
              .cast(pl.Int64)
              .alias("touchdown"),
            
            # TDs - Specific Types
            pl.lit(0).cast(pl.Int64).alias("passing_td"),  # Rushers don't pass
            pl.when(pl.col("rush_touchdown") == 1).then(1).otherwise(0).sum().cast(pl.Int64).alias("rushing_td_count"),
            pl.lit(0).cast(pl.Int64).alias("receiving_td_count"),  # Rushers don't receive
        ])
        .rename({
            "rusher_player_id": "player_id",
            "rusher_player_name": "player_name",
        })
        .with_columns([
            pl.col("posteam").alias("team"),
            pl.col("defteam").alias("opponent"),
            (pl.col("touchdown") > 0).cast(pl.Int8).alias("anytime_td"),
            pl.col("touchdown").cast(pl.Int64).alias("td_count"),
        ])
    )


def _aggregate_receiver_drives(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate receiving stats per (receiver, drive)."""
    
    receivers = df.filter(pl.col("receiver_player_id").is_not_null())
    
    if receivers.is_empty():
        return pl.DataFrame()
    
    return (
        receivers
        .group_by(["season", "week", "game_id", "game_date", "drive",
                   "receiver_player_id", "receiver_player_name", "posteam", "defteam"])
        .agg([
            pl.col("receiving_yards").fill_null(0).sum().cast(pl.Float64).alias("receiving_yards"),
            pl.col("target").fill_null(0).sum().cast(pl.Int64).alias("target"),
            pl.col("reception").fill_null(0).sum().cast(pl.Int64).alias("reception"),
            
            # Receivers don't pass or rush typically
            pl.lit(0.0).cast(pl.Float64).alias("passing_yards"),
            pl.lit(0).cast(pl.Int64).alias("pass_attempt"),
            pl.lit(0).cast(pl.Int64).alias("completion"),
            pl.lit(0.0).cast(pl.Float64).alias("rushing_yards"),
            pl.lit(0).cast(pl.Int64).alias("carry"),
            
            # TDs - General
            pl.when(pl.col("touchdown_player_id") == pl.col("receiver_player_id"))
              .then(pl.col("touchdown").fill_null(0))
              .otherwise(0)
              .sum()
              .cast(pl.Int64)
              .alias("touchdown"),
            
            # TDs - Specific Types
            pl.lit(0).cast(pl.Int64).alias("passing_td"),  # Receivers don't pass
            pl.lit(0).cast(pl.Int64).alias("rushing_td_count"),  # Receivers don't rush
            pl.when(pl.col("pass_touchdown") == 1).then(1).otherwise(0).sum().cast(pl.Int64).alias("receiving_td_count"),
        ])
        .rename({
            "receiver_player_id": "player_id",
            "receiver_player_name": "player_name",
        })
        .with_columns([
            pl.col("posteam").alias("team"),
            pl.col("defteam").alias("opponent"),
            (pl.col("touchdown") > 0).cast(pl.Int8).alias("anytime_td"),
            pl.col("touchdown").cast(pl.Int64).alias("td_count"),
        ])
    )


def _merge_multi_role_players(df: pl.DataFrame) -> pl.DataFrame:
    """Merge stats for players who had multiple roles in same drive."""
    
    merged = (
        df
        .group_by(["season", "week", "game_id", "game_date", "drive", 
                   "player_id", "player_name"])
        .agg([
            # Sum all stats
            pl.col("passing_yards").sum().alias("passing_yards"),
            pl.col("rushing_yards").sum().alias("rushing_yards"),
            pl.col("receiving_yards").sum().alias("receiving_yards"),
            pl.col("pass_attempt").sum().alias("pass_attempt"),
            pl.col("completion").sum().alias("completion"),
            pl.col("target").sum().alias("target"),
            pl.col("reception").sum().alias("reception"),
            pl.col("carry").sum().alias("carry"),
            pl.col("touchdown").sum().alias("touchdown"),
            
            # TDs - Specific Types
            pl.col("passing_td").sum().alias("passing_td"),
            pl.col("rushing_td_count").sum().alias("rushing_td_count"),
            pl.col("receiving_td_count").sum().alias("receiving_td_count"),
            
            # Team context (take first)
            pl.col("team").first().alias("team"),
            pl.col("opponent").first().alias("opponent"),
        ])
    )
    
    # Fill nulls with 0
    for col in PLAYER_DRIVE_STATS:
        if col in merged.columns:
            merged = merged.with_columns(pl.col(col).fill_null(0))
    
    return merged.with_columns([
        (pl.col("touchdown") > 0).cast(pl.Int8).alias("anytime_td"),
        pl.col("touchdown").cast(pl.Int64).alias("td_count"),
    ])

