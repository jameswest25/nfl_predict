"""NFL daily totals cache for rolling window computations.

Builds daily aggregated stats per player-context combination:
- vs_any: All games for this player
- vs_team: Games vs specific opponent

Cache structure per day:
  player_id, date, ctx, opponent → stat_num columns
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Callable
import polars as pl
import logging

from utils.feature.stats import NFL_PLAYER_STATS

logger = logging.getLogger(__name__)

# Cache root directory
DAILY_CACHE_ROOT = Path("cache/feature/daily_totals")

# Schema for ID columns (numeric stat columns are dynamic)
DAILY_ID_SCHEMA = {
    "ctx": pl.Utf8,           # Context identifier
    "player_id": pl.Utf8,     # Player identifier
    "season": pl.Int32,       # NFL Season (crucial for rolling window)
    "team": pl.Utf8,          # Player team for that game
    "date": pl.Date,          # Game date
    "opponent": pl.Utf8,      # Opponent team (nullable for vs_any/with_team)
}


def write_daily_totals(
    day: date,
    raw_loader: Callable[[date], pl.DataFrame],
    level: str = "game",  # "game" or "drive"
) -> None:
    """Write daily totals cache for one calendar date.
    
    Creates file: cache/feature/daily_totals/level=<level>/date=<YYYY-MM-DD>/part.parquet
    
    Contains TWO slices identified by 'ctx' column:
      - ctx='vs_any':  per (player_id, date) - all games
      - ctx='vs_team': per (player_id, opponent, date) - vs specific opponent
    
    Parameters
    ----------
    day : date
        The date to process
    raw_loader : Callable
        Function that loads player-level data for a given date
        Should return DataFrame with: player_id, game_date, opponent, <stats>
    level : str
        Aggregation level: "game" or "drive"
    """
    out_dir = DAILY_CACHE_ROOT / f"level={level}" / f"date={day.isoformat()}"
    out_file = out_dir / "part.parquet"
    
    # Skip if already exists
    if out_file.exists():
        return

    # Load raw data for this day
    df = raw_loader(day)
    if df.is_empty():
        return

    # Normalize date column
    if "game_date" in df.columns:
        df = df.with_columns(pl.col("game_date").dt.date().alias("date"))
    elif "date" not in df.columns:
        logger.warning(f"No date column found for {day}, skipping")
        return
    
    # Filter to this specific day
    df = df.filter(pl.col("date") == day)
    if df.is_empty():
        return
    
    # Validate required columns
    required = ["player_id", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"Missing required columns {missing} for {day}, skipping")
        return
    
    # Build stat expressions (numerators)
    # Each stat becomes <stat>_num
    stat_exprs = []
    for stat in NFL_PLAYER_STATS:
        if stat in df.columns:
            stat_exprs.append(pl.col(stat).fill_null(0).sum().alias(f"{stat}_num"))
    
    # Add denominator (count of rows aggregated)
    stat_exprs.append(pl.len().alias("denom"))
    
    if not stat_exprs:
        logger.warning(f"No stats found in data for {day}")
        return
    
    df = df.with_columns([
        pl.col("team").cast(pl.Utf8).alias("team") if "team" in df.columns else pl.lit(None).cast(pl.Utf8).alias("team"),
        pl.col("opponent").cast(pl.Utf8) if "opponent" in df.columns else pl.lit(None).cast(pl.Utf8).alias("opponent"),
        pl.col("season").cast(pl.Int32) if "season" in df.columns else pl.col("date").dt.year().cast(pl.Int32).alias("season"),
    ])

    # --- CONTEXT 1: vs_any (all games for player on this date with current team) ---
    vs_any = (
        df
        .group_by(["player_id", "season", "team", "date"])
        .agg(stat_exprs)
        .with_columns([
            pl.lit("vs_any").alias("ctx"),
            pl.lit(None).cast(pl.Utf8).alias("opponent"),
        ])
    )
    
    # --- CONTEXT 2: vs_team (games vs specific opponent) ---
    if "opponent" in df.columns:
        vs_team = (
            df
            .group_by(["player_id", "season", "team", "opponent", "date"])
            .agg(stat_exprs)
            .with_columns([
                pl.lit("vs_team").alias("ctx"),
            ])
        )

        contexts = [vs_any, vs_team]
    else:
        # No opponent column, only vs_any context
        logger.warning(f"No opponent column for {day}, only vs_any context")
        contexts = [vs_any]

    # --- CONTEXT 3: with_team (career with current team only) ---
    if "team" in df.columns:
        with_team = (
            df
            .group_by(["player_id", "season", "team", "date"])
            .agg(stat_exprs)
            .with_columns([
                pl.lit("with_team").alias("ctx"),
                pl.lit(None).cast(pl.Utf8).alias("opponent"),
            ])
        )
        contexts.append(with_team)

    # Combine contexts
    daily = pl.concat(contexts, how="diagonal_relaxed")
    
    # Ensure schema consistency
    for col, dtype in DAILY_ID_SCHEMA.items():
        if col not in daily.columns:
            daily = daily.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            daily = daily.with_columns(pl.col(col).cast(dtype))

    # Column ordering: ID cols first, then stats
    id_cols = list(DAILY_ID_SCHEMA.keys())
    stat_cols = [c for c in daily.columns if c not in id_cols]
    daily = daily.select(id_cols + stat_cols)
    
    # Write to disk
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(out_file, compression="zstd")
    logger.info(f"Wrote daily totals for {day} ({level} level): {len(daily)} rows, {len(stat_cols)} stat columns")


def build_daily_cache_range(
    start_date: date,
    end_date: date,
    level: str = "game",
) -> None:
    """Build daily totals cache for a date range.
    
    Parameters
    ----------
    start_date : date
        Start date (inclusive)
    end_date : date  
        End date (inclusive)
    level : str
        Aggregation level: "game" or "drive"
    """
    logger.info(f"Building daily totals cache: {start_date} → {end_date} ({level} level)")
    
    # Determine source directory based on level
    if level == "game":
        source_dir = Path("data/processed/player_game_by_week")
    elif level == "drive":
        source_dir = Path("data/processed/player_drive_by_week")
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'game' or 'drive'")
    
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return
    
    # Load function: loads player data for a given date
    def load_player_data(day: date) -> pl.DataFrame:
        """Load player-level data for a specific date from weekly parquets."""
        try:
            # Scan all weekly parquets
            df = pl.scan_parquet(
                str(source_dir / "season=*/week=*/part.parquet"),
                glob=True,
                hive_partitioning=True,
                missing_columns="insert",
                extra_columns="ignore",
            )

            schema = df.collect_schema()
            categorical_cols = [
                name
                for name, dtype in zip(schema.names(), schema.dtypes())
                if "Categorical" in str(dtype)
            ]
            if categorical_cols:
                df = df.with_columns([pl.col(col).cast(pl.Utf8) for col in categorical_cols])
            
            # Filter to specific date
            df = df.filter(
                pl.col("game_date").dt.date() == day
            ).collect()
            
            return df
        except Exception as e:
            logger.warning(f"Could not load data for {day}: {e}")
            return pl.DataFrame()
    
    # Write daily totals for each day in range
    current = start_date
    days_written = 0
    days_skipped = 0
    
    while current <= end_date:
        out_file = DAILY_CACHE_ROOT / f"level={level}" / f"date={current.isoformat()}" / "part.parquet"
        
        if out_file.exists():
            days_skipped += 1
        else:
            write_daily_totals(current, load_player_data, level=level)
            if out_file.exists():
                days_written += 1
        
        current += timedelta(days=1)
    
    logger.info(f"Daily cache build complete: {days_written} written, {days_skipped} skipped")


def load_daily_cache(
    player_ids: list[str],
    start_date: date,
    end_date: date,
    level: str = "game",
    context: str = "vs_any",
) -> pl.DataFrame:
    """Load daily totals cache for specific players and date range.
    
    Parameters
    ----------
    player_ids : list[str]
        Player IDs to load
    start_date : date
        Start date (inclusive)
    end_date : date
        End date (inclusive)
    level : str
        Aggregation level: "game" or "drive"
    context : str
        Context to filter: "vs_any", "vs_team", or None for all
    
    Returns
    -------
    pl.DataFrame
        Cached daily totals with columns:
        - player_id, date, ctx, opponent
        - <stat>_num for each stat
        - denom
    """
    cache_dir = DAILY_CACHE_ROOT / f"level={level}"
    
    if not cache_dir.exists():
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        return pl.DataFrame()
    
    try:
        # Scan all parquet files in date range
        df = pl.scan_parquet(
            str(cache_dir / "date=*/part.parquet"),
            glob=True,
            hive_partitioning=True,
        )
        
        # Filter to date range and players
        df = df.filter(
            (pl.col("date") >= start_date) &
            (pl.col("date") <= end_date) &
            (pl.col("player_id").is_in(player_ids))
        )
        
        # Filter to context if specified
        if context:
            df = df.filter(pl.col("ctx") == context)
        
        return df.collect()
        
    except Exception as e:
        logger.error(f"Error loading daily cache: {e}")
        return pl.DataFrame()


if __name__ == "__main__":
    # Test: build cache for existing data
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Build cache for Sept 2024 data
    build_daily_cache_range(
        start_date=date(2024, 9, 1),
        end_date=date(2024, 9, 30),
        level="game"
    )

