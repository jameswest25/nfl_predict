"""Data loading functions for player-game level aggregation.

These functions load and prepare external data sources like rosters,
injuries, snap counts, depth charts, etc. for enrichment.
"""

from __future__ import annotations

from pathlib import Path
import datetime as dt
import logging
from typing import Any

import polars as pl

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from utils.general.paths import (
    PLAYER_MARKET_PROCESSED_DIR,
    QB_PROFILE_DIR,
    TRAVEL_CALENDAR_DIR,
    ROSTER_CACHE_DIR,
    ROSTER_SNAPSHOT_DIR,
    INJURY_CACHE_DIR,
    INJURY_TRANSACTION_CACHE_DIR,
    SNAP_CACHE_DIR,
)

try:
    from nfl_data_py import import_weekly_rosters, import_injuries, import_snap_counts
except Exception:  # pragma: no cover - optional dependency
    import_weekly_rosters = None
    import_injuries = None
    import_snap_counts = None

logger = logging.getLogger(__name__)

REQUIRED_ROSTER_COLS = ['season', 'week', 'team', 'player_id']

# Cache for nflverse depth charts
_DEPTH_CHART_CACHE: dict[int, pl.DataFrame] = {}


def _collect_partition_paths(base_dir: Path, seasons: list[int]) -> list[Path]:
    """Collect all partition parquet paths for the given seasons."""
    paths: list[Path] = []
    for season in sorted({int(s) for s in seasons}):
        season_dir = base_dir / f"season={season}"
        if not season_dir.exists():
            continue
        paths.extend(sorted(season_dir.glob("week=*/part.parquet")))
    return paths


def load_rosters_for_years(years: list[int]) -> pl.DataFrame:
    """Load weekly roster data for the given years with caching and validation."""
    from utils.collect.arrival_log import log_feed_arrivals
    from utils.collect.nfl_schedules import get_schedule

    if not years:
        raise ValueError("No seasons provided for roster load.")

    unique_years = sorted(set(int(y) for y in years))
    frames: list[pl.DataFrame] = []

    for year in unique_years:
        cache_path = ROSTER_CACHE_DIR / f"roster_{year}.parquet"
        if cache_path.exists():
            roster_pl = pl.read_parquet(cache_path)
        else:
            if import_weekly_rosters is None:
                raise RuntimeError(
                    "nfl_data_py.import_weekly_rosters is unavailable; cannot fetch roster data."
                )
            roster_pd = import_weekly_rosters(years=[year])
            roster_pl = pl.from_pandas(roster_pd)
            missing = [c for c in REQUIRED_ROSTER_COLS if c not in roster_pl.columns]
            if missing:
                raise ValueError(f"Roster data for season {year} missing required columns: {missing}")
            roster_pl.write_parquet(cache_path, compression="zstd")
            _log_roster_arrivals(year, roster_pd, get_schedule, log_feed_arrivals)
        frames.append(roster_pl)

    if not frames:
        raise ValueError(f"No roster data available for seasons {unique_years}")

    roster_pl = pl.concat(frames, how="diagonal_relaxed")
    missing = [c for c in REQUIRED_ROSTER_COLS if c not in roster_pl.columns]
    if missing:
        raise ValueError(f"Cached roster data missing required columns: {missing}")

    return roster_pl


def load_roster_snapshots_for_years(years: list[int]) -> pl.DataFrame:
    """Load time-aware roster snapshots for the requested seasons."""
    if not years:
        return pl.DataFrame()

    paths: list[Path] = []
    for season in sorted({int(y) for y in years}):
        season_dir = ROSTER_SNAPSHOT_DIR / f"season={season}"
        if not season_dir.exists():
            continue
        paths.extend(season_dir.glob("week=*/part.parquet"))

    frames: list[pl.DataFrame] = []
    for path in paths:
        try:
            df = pl.read_parquet(path)
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("Failed to read roster snapshot %s: %s", path, exc)
            continue

        if df.is_empty():
            continue

        available_cols = {
            "season": pl.Int32,
            "week": pl.Int32,
            "team": pl.Utf8,
            "game_id": pl.Utf8,
            "player_id": pl.Utf8,
            "espn_id": pl.Utf8,
            "status": pl.Utf8,
            "status_abbr": pl.Utf8,
            "status_detail": pl.Utf8,
            "snapshot_ts": pl.Datetime("ms", "UTC"),
            "depth_chart_order": pl.Int32,
        }
        missing_cols = [col for col in available_cols if col not in df.columns]
        if missing_cols:
            df = df.with_columns([pl.lit(None).alias(col) for col in missing_cols])

        df = df.select(list(available_cols)).with_columns(
            [
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                pl.col("team").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
                pl.col("game_id").cast(pl.Utf8),
                pl.col("player_id").cast(pl.Utf8),
                pl.col("status").cast(pl.Utf8),
                pl.col("status_abbr").cast(pl.Utf8),
                pl.col("status_detail").cast(pl.Utf8),
                pl.col("snapshot_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("depth_chart_order").cast(pl.Int32),
            ]
        )
        frames.append(df)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="diagonal_relaxed").filter(pl.col("player_id").is_not_null())


def load_nflverse_depth_charts(years: list[int]) -> pl.DataFrame:
    """Load official NFL depth chart data from nflverse via nfl_data_py.
    
    Returns a DataFrame with columns:
        - season, week, team, player_id, position, depth_chart_order, formation
    """
    from nfl_data_py import import_depth_charts
    
    frames: list[pl.DataFrame] = []
    for year in sorted(set(years)):
        if year in _DEPTH_CHART_CACHE:
            frames.append(_DEPTH_CHART_CACHE[year])
            continue
        
        try:
            dc_pd = import_depth_charts([year])
            if dc_pd.empty:
                logger.warning("No depth chart data available for %s", year)
                continue
            
            # Filter to offensive skill positions only
            dc_pd = dc_pd[dc_pd['formation'] == 'Offense']
            dc_pd = dc_pd[dc_pd['position'].isin(['QB', 'RB', 'WR', 'TE', 'FB'])]
            
            # Rename and select relevant columns
            dc_pd = dc_pd.rename(columns={
                'club_code': 'team',
                'gsis_id': 'player_id',
                'depth_team': 'depth_chart_order',
            })
            
            dc_pl = pl.from_pandas(dc_pd[['season', 'week', 'team', 'player_id', 'position', 'depth_chart_order']])
            dc_pl = dc_pl.with_columns([
                pl.col('season').cast(pl.Int32),
                pl.col('week').cast(pl.Int32),
                pl.col('team').str.strip_chars().str.to_uppercase(),
                pl.col('player_id').cast(pl.Utf8),
                pl.col('position').cast(pl.Utf8),
                pl.col('depth_chart_order').cast(pl.Int16),
            ])
            
            _DEPTH_CHART_CACHE[year] = dc_pl
            frames.append(dc_pl)
            logger.info("Loaded %d depth chart entries for season %s", dc_pl.height, year)
            
        except Exception as exc:
            logger.warning("Failed to load depth charts for %s: %s", year, exc)
            continue
    
    if not frames:
        return pl.DataFrame()
    
    return pl.concat(frames, how="diagonal_relaxed")


def join_nflverse_depth_charts(df: pl.DataFrame) -> pl.DataFrame:
    """Join official nflverse depth chart data to the player-game DataFrame.
    
    This populates depth_chart_order with real depth chart rankings from NFL data.
    """
    required_cols = {"season", "week", "team", "player_id"}
    if not required_cols.issubset(set(df.columns)):
        logger.warning("Missing columns for depth chart join: %s", required_cols - set(df.columns))
        return df
    
    seasons = df.get_column("season").unique().to_list()
    seasons = [int(s) for s in seasons if s is not None]
    
    if not seasons:
        return df
    
    depth_charts = load_nflverse_depth_charts(seasons)
    if depth_charts.is_empty():
        logger.warning("No depth chart data loaded for seasons %s", seasons)
        return df
    
    # Join depth charts to player-game data
    # Use left join to preserve all player-game rows
    df = df.join(
        depth_charts.select(["season", "week", "team", "player_id", "depth_chart_order"]).rename(
            {"depth_chart_order": "_nfl_depth_order"}
        ),
        on=["season", "week", "team", "player_id"],
        how="left",
    )
    
    # Update depth_chart_order: prefer nflverse data, fall back to existing if any
    if "depth_chart_order" in df.columns:
        df = df.with_columns(
            pl.coalesce([pl.col("_nfl_depth_order"), pl.col("depth_chart_order")])
            .cast(pl.Int16)
            .alias("depth_chart_order")
        )
    else:
        df = df.with_columns(
            pl.col("_nfl_depth_order").cast(pl.Int16).alias("depth_chart_order")
        )
    
    df = df.drop("_nfl_depth_order")
    
    # Log coverage
    total = df.height
    with_depth = df.filter(pl.col("depth_chart_order").is_not_null()).height
    logger.info("Depth chart coverage: %d/%d rows (%.1f%%)", with_depth, total, 100 * with_depth / total if total > 0 else 0)
    
    return df


def _log_roster_arrivals(
    season: int,
    roster_pd: "pd.DataFrame",
    get_schedule: Any,
    log_feed_arrivals: Any,
) -> None:
    """Log feed arrival metrics when roster data is freshly collected."""
    try:
        if roster_pd.empty:
            return
        if "team" not in roster_pd.columns or "week" not in roster_pd.columns:
            return
        schedule = get_schedule([season])
        if schedule.empty:
            return
        schedule = schedule.copy()
        if "start_time_utc" in schedule.columns:
            schedule["start_time_utc"] = pd.to_datetime(schedule["start_time_utc"], utc=True, errors="coerce")
        else:
            schedule["start_time_utc"] = pd.NaT
        team_frames: list["pd.DataFrame"] = []
        for col in ("home_team", "away_team"):
            if col not in schedule.columns:
                continue
            frame = schedule[["game_id", "season", "week", "start_time_utc", col]].rename(columns={col: "team"})
            team_frames.append(frame)
        if not team_frames:
            return
        schedule_team = pd.concat(team_frames, ignore_index=True)
        schedule_team["team"] = schedule_team["team"].astype(str)
        collected_at = dt.datetime.now(dt.timezone.utc)
        rows: list[dict[str, object]] = []
        for (team, week), group in roster_pd.groupby(["team", "week"], dropna=True):
            team_sched = schedule_team[schedule_team["team"] == team]
            if team_sched.empty:
                continue
            sched_week = team_sched[team_sched["week"] == week]
            if sched_week.empty:
                continue
            row_sched = sched_week.iloc[0]
            game_id = row_sched.get("game_id")
            start_time = row_sched.get("start_time_utc")
            if isinstance(start_time, pd.Timestamp):
                if start_time.tzinfo is None:
                    start_time = start_time.tz_localize("UTC")
                else:
                    start_time = start_time.tz_convert("UTC")
            else:
                start_time = pd.NaT
            rows.append(
                {
                    "season": season,
                    "week": int(week) if week is not None and not pd.isna(week) else None,
                    "game_id": str(game_id) if game_id is not None else None,
                    "team": str(team),
                    "game_start_ts": None if pd.isna(start_time) else start_time.to_pydatetime(),
                    "feed_timestamp": collected_at,
                    "collected_at": collected_at,
                    "metadata": {
                        "players_collected": int(len(group)),
                        "positions": sorted(group.get("position", pd.Series(dtype=str)).dropna().unique().tolist()),
                    },
                }
            )
        if rows:
            log_feed_arrivals("rosters", rows, snapshot_label="weekly")
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.warning("Failed to log roster arrival metrics for season %s: %s", season, exc)


def load_injuries_for_years(years: list[int]) -> pl.DataFrame:
    """Load injury report data for the given years with caching and validation."""
    if not years:
        raise ValueError("No seasons provided for injury load.")

    if import_injuries is None:
        raise RuntimeError(
            "nfl_data_py.import_injuries is unavailable; cannot fetch injury data."
        )

    unique_years = sorted(set(int(y) for y in years))
    frames: list[pl.DataFrame] = []

    for year in unique_years:
        cache_path = INJURY_CACHE_DIR / f"injury_{year}.parquet"
        if cache_path.exists():
            injury_pl = pl.read_parquet(cache_path)
        else:
            try:
                injury_pd = import_injuries(years=[year])
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Failed to import injury reports for season %s: %s", year, exc)
                continue
            if injury_pd is None or injury_pd.empty:
                logger.info("No injury reports returned for season %s.", year)
                continue
            injury_pl = pl.from_pandas(injury_pd)
            if "gsis_id" not in injury_pl.columns:
                logger.warning("Injury data for season %s missing 'gsis_id'; skipping cache write.", year)
                continue
            injury_pl.write_parquet(cache_path, compression="zstd")
        frames.append(injury_pl)

    if not frames:
        return pl.DataFrame()

    injury_pl = pl.concat(frames, how="diagonal_relaxed")
    return injury_pl


def load_injury_transactions_for_years(years: list[int]) -> pl.DataFrame:
    """Load injury transaction records scraped from prosportstransactions."""
    if not years:
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for year in sorted(set(int(y) for y in years)):
        cache_path = INJURY_TRANSACTION_CACHE_DIR / f"transactions_{year}.parquet"
        if not cache_path.exists():
            logger.debug("No injury transaction cache for season %s", year)
            continue
        try:
            tx_pl = pl.read_parquet(cache_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read injury transaction cache %s: %s", cache_path, exc)
            continue
        frames.append(tx_pl)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="diagonal_relaxed")


def normalize_player_name(name: str) -> str:
    """Convert full name to abbreviated format for matching.
    
    Examples:
        'Saquon Barkley' -> 'S.Barkley'
        'Christian McCaffrey' -> 'C.McCaffrey'
        'C.McCaffrey' -> 'C.McCaffrey' (already abbreviated)
    """
    if not name or (pd is not None and pd.isna(name)):
        return name
    name = str(name).strip()
    # If already abbreviated (contains a period), return as-is
    if '.' in name:
        return name
    parts = name.split()
    if len(parts) >= 2:
        first_initial = parts[0][0] + '.'
        last_name = parts[-1]
        return first_initial + last_name
    return name


def normalize_player_name_expr() -> pl.Expr:
    """Create a Polars expression to normalize player names."""
    return (
        pl.when(pl.col("player_name").str.contains(r"\."))
        .then(pl.col("player_name"))  # Already abbreviated
        .otherwise(
            pl.concat_str([
                pl.col("player_name").str.split(" ").list.first().str.slice(0, 1),
                pl.lit("."),
                pl.col("player_name").str.split(" ").list.last(),
            ])
        )
        .alias("__normalized_player_name")
    )


def load_snap_counts_for_years(years: list[int]) -> pl.DataFrame:
    """Load snap count data for the given years with caching.
    
    Note: Snap counts from nfl_data_py use pfr_player_id, but our feature matrix
    uses gsis_id (player_id). We load the ID mapping to convert between them.
    """
    if not years:
        raise ValueError("No seasons provided for snap count load.")

    if import_snap_counts is None:
        raise RuntimeError(
            "nfl_data_py.import_snap_counts is unavailable; cannot fetch snap counts."
        )

    unique_years = sorted(set(int(y) for y in years))
    frames: list[pl.DataFrame] = []

    for year in unique_years:
        cache_path = SNAP_CACHE_DIR / f"snap_counts_{year}.parquet"
        if cache_path.exists():
            snap_pl = pl.read_parquet(cache_path)
        else:
            try:
                snap_pd = import_snap_counts(years=[year])
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Failed to import snap counts for season %s: %s", year, exc)
                continue
            if snap_pd is None or snap_pd.empty:
                logger.info("No snap counts returned for season %s.", year)
                continue
            snap_pl = pl.from_pandas(snap_pd)
            snap_pl.write_parquet(cache_path, compression="zstd")
        frames.append(snap_pl)

    if not frames:
        return pl.DataFrame()

    snap_pl = pl.concat(frames, how="diagonal_relaxed")
    
    # Convert pfr_player_id to gsis_id (player_id) using the ID mapping
    # This fixes name mismatch issues (e.g., "A.St. Brown" vs "Amon-Ra St. Brown")
    if "pfr_player_id" in snap_pl.columns and "player_id" not in snap_pl.columns:
        try:
            import nfl_data_py as nfl
            ids_pd = nfl.import_ids()
            if ids_pd is not None and not ids_pd.empty:
                ids_pl = pl.from_pandas(ids_pd[["gsis_id", "pfr_id"]].dropna())
                ids_pl = ids_pl.rename({"gsis_id": "player_id", "pfr_id": "pfr_player_id"})
                ids_pl = ids_pl.with_columns([
                    pl.col("player_id").cast(pl.Utf8),
                    pl.col("pfr_player_id").cast(pl.Utf8),
                ])
                snap_pl = snap_pl.with_columns(pl.col("pfr_player_id").cast(pl.Utf8))
                snap_pl = snap_pl.join(ids_pl, on="pfr_player_id", how="left")
                matched = snap_pl.filter(pl.col("player_id").is_not_null()).height
                total = snap_pl.height
                logger.info("Snap counts: mapped %d/%d rows from pfr_player_id to gsis_id (%.1f%%)", 
                           matched, total, 100 * matched / total if total > 0 else 0)
        except Exception as exc:
            logger.warning("Failed to load ID mapping for snap counts: %s", exc)
    
    return snap_pl


def load_player_market_features(seasons: list[int]) -> pl.DataFrame:
    """Load processed player market priors for the given seasons."""
    if not seasons:
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for season in sorted({int(s) for s in seasons}):
        base_dir = PLAYER_MARKET_PROCESSED_DIR / f"season={season}"
        if not base_dir.exists():
            continue
        week_dirs = sorted(base_dir.glob("week=*"))
        for week_dir in week_dirs:
            part_path = week_dir / "part.parquet"
            if not part_path.exists():
                continue
            try:
                frame = pl.read_parquet(part_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load player market features from %s: %s", part_path, exc)
                continue
            if not frame.is_empty():
                frames.append(frame)
    if not frames:
        return pl.DataFrame()

    market_pl = pl.concat(frames, how="diagonal_relaxed")
    numeric_cols = [col for col in market_pl.columns if col.startswith("market_")]
    cast_exprs = [pl.col(col).cast(pl.Float32) for col in numeric_cols]
    if cast_exprs:
        market_pl = market_pl.with_columns(cast_exprs)

    # Downstream joins expect one row per (season, week, game_id, player_id).  The
    # player market files can contain multiple rows for a single player due to
    # selection-name spelling variations or bookmaker fragments.  Collapse those
    # duplicates up-front so we do not explode the feature matrix.
    group_cols = ["season", "week", "game_id", "player_id"]
    if not set(group_cols).issubset(market_pl.columns):
        return market_pl

    market_pl = market_pl.filter(pl.col("player_id").is_not_null())
    other_cols = [
        col for col in market_pl.columns if col not in set(group_cols + numeric_cols)
    ]
    agg_exprs: list[pl.Expr] = []
    if numeric_cols:
        agg_exprs.extend(
            pl.col(col).median().alias(col)  # median is more robust to outliers
            for col in numeric_cols
        )
    if other_cols:
        agg_exprs.extend(pl.col(col).first().alias(col) for col in other_cols)

    if agg_exprs:
        market_pl = (
            market_pl.group_by(group_cols, maintain_order=True)
            .agg(agg_exprs)
            .sort(group_cols)
        )

    return market_pl


def load_qb_profile_features(seasons: list[int]) -> pl.DataFrame:
    """Load QB profile features for the given seasons."""
    if not seasons:
        return pl.DataFrame()
    paths = _collect_partition_paths(QB_PROFILE_DIR, seasons)
    if not paths:
        return pl.DataFrame()
    scan = pl.scan_parquet(
        paths,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    df = scan.collect(streaming=True)
    if df.is_empty():
        return df
    if "qb_id" in df.columns:
        df = df.rename({"qb_id": "qb_profile_id"})
    if "qb_profile_dropbacks_prev" in df.columns:
        df = df.with_columns(
            pl.col("qb_profile_dropbacks_prev")
            .fill_null(-1.0)
            .alias("__qb_profile_dropbacks_rank")
        ).sort(
            ["season", "week", "team", "__qb_profile_dropbacks_rank"],
            descending=[False, False, False, True],
        )
        df = df.unique(subset=["season", "week", "team"], keep="first").drop("__qb_profile_dropbacks_rank")
    else:
        df = df.unique(subset=["season", "week", "team"], keep="first")
    df = df.with_columns(
        [
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("team").cast(pl.Utf8),
        ]
    )
    float_cols = [
        col
        for col, dtype in df.schema.items()
        if col.startswith("qb_profile_")
        and dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32)
    ]
    if float_cols:
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in float_cols])
    for ts_col in ("qb_profile_data_as_of", "qb_profile_team_data_as_of"):
        if ts_col in df.columns:
            df = df.with_columns(pl.col(ts_col).cast(pl.Datetime("ms")))
    return df


def load_travel_calendar_features(seasons: list[int]) -> pl.DataFrame:
    """Load travel calendar features for the given seasons."""
    from utils.general.paths import TRAVEL_CALENDAR_DIR

    if not seasons:
        return pl.DataFrame()
    paths = _collect_partition_paths(TRAVEL_CALENDAR_DIR, seasons)
    if not paths:
        return pl.DataFrame()
    scan = pl.scan_parquet(
        paths,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    df = scan.collect(streaming=True)
    if df.is_empty():
        return df
    keep_cols = [
        "season",
        "week",
        "team",
        "rest_days",
        "rest_hours",
        "rest_days_rolling3",
        "travel_km",
        "travel_miles",
        "travel_km_rolling3",
        "timezone_change_hours",
        "time_diff_from_home_hours",
        "game_timezone_offset",
        "team_timezone_offset",
        "local_start_hour",
        "consecutive_road_games",
        "consecutive_home_games",
        "is_short_week",
        "is_long_rest",
        "bye_week_flag",
        "west_to_east_early",
        "east_to_west_late",
    ]
    existing = [col for col in keep_cols if col in df.columns]
    df = df.select(existing)
    rename_map = {
        "rest_days": "travel_rest_days",
        "rest_hours": "travel_rest_hours",
        "rest_days_rolling3": "travel_rest_days_l3",
        "travel_km": "travel_distance_km",
        "travel_miles": "travel_distance_miles",
        "travel_km_rolling3": "travel_distance_km_l3",
        "timezone_change_hours": "travel_timezone_change_hours",
        "time_diff_from_home_hours": "travel_time_diff_from_home_hours",
        "game_timezone_offset": "travel_game_timezone_offset",
        "team_timezone_offset": "travel_team_timezone_offset",
        "local_start_hour": "travel_local_start_hour",
        "consecutive_road_games": "travel_consecutive_road_games",
        "consecutive_home_games": "travel_consecutive_home_games",
        "is_short_week": "travel_short_week_flag",
        "is_long_rest": "travel_long_rest_flag",
        "bye_week_flag": "travel_bye_week_flag",
        "west_to_east_early": "travel_west_to_east_early_flag",
        "east_to_west_late": "travel_east_to_west_late_flag",
    }
    df = df.rename(rename_map)
    df = df.with_columns(
        [
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("team").cast(pl.Utf8),
        ]
    )
    float_cols = [
        col
        for col, dtype in df.schema.items()
        if col.startswith("travel_")
        and col
        not in {
            "travel_short_week_flag",
            "travel_long_rest_flag",
            "travel_bye_week_flag",
            "travel_west_to_east_early_flag",
            "travel_east_to_west_late_flag",
        }
        and dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32)
    ]
    if float_cols:
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in float_cols])
    flag_cols = [
        col
        for col in [
            "travel_short_week_flag",
            "travel_long_rest_flag",
            "travel_bye_week_flag",
            "travel_west_to_east_early_flag",
            "travel_east_to_west_late_flag",
        ]
        if col in df.columns
    ]
    if flag_cols:
        df = df.with_columns([pl.col(col).cast(pl.Int8) for col in flag_cols])
    return df

