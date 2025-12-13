from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Final
import time
import sys
import requests

import pandas as pd
import numpy as np
from urllib.error import HTTPError
try:
    # nfl_data_py provides nflfastR tables (PBP, schedules, rosters)
    from nfl_data_py import import_pbp_data, import_schedules
    import nfl_data_py  # type: ignore
    if not hasattr(nfl_data_py, "Error"):
        nfl_data_py.Error = HTTPError  # type: ignore[attr-defined]
except Exception:
    import_pbp_data = None
    import_schedules = None

from utils.general.paths import (
    PROJ_ROOT, DATA_RAW, NFL_PBP_RAW_DIR, STADIUM_COORDS_FILE, COLLECT_CACHE_DIR
)
from utils.collect import (
    # Parquet I/O
    write_data, load_data, get_latest_date,
    
    # Weather Cache
    attach_weather,
    
    # Dtype Optimization
    OptimizationConfig, maybe_optimize,
    
    # Schema Guard
    ExpectedSchema,
    
    # NFL-specific
    cache_schedules,
    collect_weather_forecasts,
)
from utils.collect.audit import run_collect_audit
from utils.feature.enrichment.odds import collect_odds_snapshots

# Module logger
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration Loading (using centralized config module)
# -----------------------------------------------------------------------------
from utils.general.config import get_collect_params

collect_config = get_collect_params()

# -----------------------------------------------------------------------------
# Configuration constants (for easy tuning)
# -----------------------------------------------------------------------------
class Config:
    # Async/Network settings
    MAX_CONCURRENT_FEEDS = collect_config.get('max_concurrent_feeds', 20)
    MAX_FEED_RETRIES = collect_config.get('max_feed_retries', 3)
    FEED_TIMEOUT_SECONDS = collect_config.get('feed_timeout_seconds', 30)
    SEMAPHORE_LIMIT = collect_config.get('semaphore_limit', 40)
    
    # Batch processing
    DEFAULT_BATCH_DAYS = collect_config.get('default_batch_days', 7)
    
    # Cache settings
    WEATHER_CACHE_COMPRESSION = collect_config.get('weather_cache_compression', True)
    
    # Performance optimizations
    ENABLE_DTYPE_OPTIMIZATION = collect_config.get('enable_dtype_optimization', True)
    ENABLE_CATEGORICAL_COLUMNS = collect_config.get('enable_categorical_columns', True)

# -----------------------------------------------------------------------------
# Constants - using centralized paths
# -----------------------------------------------------------------------------
# Load NFL stadium → {lat, lon, tz} map one time
with STADIUM_COORDS_FILE.open() as fh:
    STADIUM_COORDS: Final[dict] = json.load(fh)

STADIUM_ALIAS_MAP: Final[dict] = {}
for _code, _meta in STADIUM_COORDS.items():
    if isinstance(_meta, dict):
        STADIUM_ALIAS_MAP[_code.upper()] = _code
        for _alias in _meta.get("aliases", []) or []:
            STADIUM_ALIAS_MAP[_alias.upper()] = _code

NFLVERSE_RELEASE_BASE_URL: Final[str] = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
NFLFASTR_GITHUB_BASE_URL: Final[str] = "https://github.com/nflverse/nflfastR-data/raw/master/data"
NFLFASTR_CACHE_DIR: Final[Path] = COLLECT_CACHE_DIR / "nflfastr"
NFLFASTR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _normalize_venue_name(raw: str | None) -> str | None:
    if not isinstance(raw, str):
        return None
    return STADIUM_ALIAS_MAP.get(raw.strip().upper())

# Initialize dtype optimization config
dtype_config = OptimizationConfig(
    enabled=Config.ENABLE_DTYPE_OPTIMIZATION,
    categorical_enabled=Config.ENABLE_CATEGORICAL_COLUMNS
)

# -----------------------------------------------------------------------------
# Helper functions (minimal ones that coordinate between modules)
# -----------------------------------------------------------------------------

def _get_default_start_date() -> dt.date:
    """Get default start date based on existing data"""
    try:
        latest = get_latest_date()
        if latest:
            # Start from the day after latest data
            return latest + dt.timedelta(days=1)
    except Exception as e:
        logger.warning(f"Could not determine latest date: {e}")
    
    # Default fallback
    return dt.date.today() - dt.timedelta(days=Config.DEFAULT_BATCH_DAYS)


def _download_pbp_from_github(seasons: list[int]) -> pd.DataFrame:
    """
    Download nflfastR play-by-play CSVs directly from the nflverse GitHub mirror.

    Parameters
    ----------
    seasons : list[int]
        Seasons to fetch.

    Returns
    -------
    pd.DataFrame
        Combined play-by-play rows for the requested seasons, or an empty frame if unavailable.
    """

    if not seasons:
        return pd.DataFrame()

    session = requests.Session()
    frames: list[pd.DataFrame] = []

    for season in seasons:
        url = f"{NFLFASTR_GITHUB_BASE_URL}/play_by_play_{season}.csv.gz"
        cache_path = NFLFASTR_CACHE_DIR / f"play_by_play_{season}.csv.gz"

        if cache_path.exists():
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info("Using cached nflfastR play-by-play for season %s (%.1f MB)", season, size_mb)
        else:
            sources = [
                ("nflverse release", NFLVERSE_RELEASE_BASE_URL),
                ("nflfastR legacy", NFLFASTR_GITHUB_BASE_URL),
            ]
            downloaded = False
            for source_name, base_url in sources:
                url = f"{base_url}/play_by_play_{season}.csv.gz"
                logger.info("Attempting download of season %s play-by-play from %s", season, source_name)
                try:
                    with session.get(url, timeout=120, stream=True) as response:
                        if response.status_code == 404:
                            logger.info("%s does not publish play-by-play for season %s (404).", source_name, season)
                            continue
                        response.raise_for_status()
                        temp_path = cache_path.with_name(cache_path.name + ".tmp")
                        with temp_path.open("wb") as fh:
                            for chunk in response.iter_content(chunk_size=1_048_576):
                                if chunk:
                                    fh.write(chunk)
                        temp_path.replace(cache_path)
                        size_mb = cache_path.stat().st_size / (1024 * 1024)
                        logger.info(
                            "Saved %s to cache from %s (%.1f MB)",
                            cache_path.name,
                            source_name,
                            size_mb,
                        )
                        downloaded = True
                        break
                except Exception as exc:
                    logger.error(
                        "Error downloading play-by-play for season %s from %s: %s",
                        season,
                        source_name,
                        exc,
                    )
                    continue

            if not downloaded:
                logger.warning("No remote source provided play-by-play for season %s", season)
                continue

        try:
            df_season = pd.read_csv(cache_path, compression="gzip", low_memory=False)
        except Exception as exc:
            logger.error("Could not load cached play-by-play CSV for season %s: %s", season, exc)
            continue

        if "season" not in df_season.columns:
            df_season["season"] = season
        else:
            df_season["season"] = df_season["season"].fillna(season)

        frames.append(df_season)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("✅ Loaded %s rows from nflfastR GitHub for seasons %s", len(combined), seasons)
    return combined

# -------------------------------------------------------------
# Weather back-fill utility (for periods collected without the
# visual_crossing_weather module present)
# -------------------------------------------------------------


def backfill_weather(start: str | None = None, end: str | None = None):
    """Attach missing weather fields for a historical date range.

    Only rows where *all* weather columns are currently null are
    processed, so already-enriched data is left untouched.
    """

    logger.info("Starting weather back-fill…")

    # -----------------------------
    # Validate & parse date inputs
    # -----------------------------
    def _parse(d: str | None):
        if d is None:
            return None
        try:
            return dt.datetime.strptime(d, "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError(f"Invalid date '{d}'. Please use YYYY-MM-DD format.") from e

    start_dt = _parse(start)
    end_dt = _parse(end)

    df = load_data()  # load full dataset; we'll filter manually below
    if df.empty:
        logger.warning("No PBP data found in the requested window – nothing to back-fill.")
        return

    from utils.collect.schema_guard import ExpectedSchema

    weather_cols = list(ExpectedSchema.WEATHER_COLUMNS)
    # Optional date-range filtering
    if 'game_date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

        if start_dt:
            df = df[df['game_date'].dt.date >= start_dt]
        if end_dt:
            df = df[df['game_date'].dt.date <= end_dt]

    missing_mask = df[weather_cols].isna().all(axis=1)

    rows_to_fix = missing_mask.sum()
    if rows_to_fix == 0:
        logger.info("All rows already have weather data – no back-fill needed.")
        return

    logger.info("Back-filling weather for %d rows (%s – %s)…", rows_to_fix,
                start_dt.isoformat() if start_dt else "min",
                end_dt.isoformat() if end_dt else "max")

    fix_df = df.loc[missing_mask].copy()
    fix_df = attach_weather(fix_df)

    # Overwrite weather columns for the affected rows
    df.loc[missing_mask, weather_cols] = fix_df[weather_cols]

    write_data(df)
    logger.info("✅ Weather back-fill completed – dataset updated.")

# -----------------------------------------------------------------------------
# Main collection functions
# -----------------------------------------------------------------------------

def _last_n_seasons(n: int = 4) -> list[int]:
    year = dt.date.today().year
    return list(range(year - n + 1, year + 1))

def collect_pbp(
    start: str = None,
    end: str = None,
    batch_days: int = None,
    seasons: list[int] | None = None,
    limit_weeks: int | None = None,
    force_full_refresh: bool = False,
    include_player_props: bool = False,
) -> pd.DataFrame:
    """
    Fetch NFL play-by-play (nflfastR via nfl_data_py) intelligently.
    
    By default, only fetches the CURRENT season to get new data incrementally.
    Use force_full_refresh=True to re-download all historical seasons.
    
    Parameters
    ----------
    start : str, optional
        Start date (not used in seasons mode)
    end : str, optional
        End date (not used in seasons mode)
    batch_days : int, optional
        Batch size (not used in seasons mode)
    seasons : list[int], optional
        Specific seasons to collect. If None, collects current season only.
    limit_weeks : int, optional
        Limit to first N weeks per season (for testing)
    force_full_refresh : bool, default False
        If True, collects all historical seasons (last 4 years)
        If False, only collects current season (incremental update)
        
    Returns
    -------
    pd.DataFrame
        Collected play-by-play data
    """
    if import_pbp_data is None:
        logger.error("nfl_data_py is not installed. Please 'pip install nfl-data-py'.")
        return pd.DataFrame()

    start_time = time.time()

    # Smart season selection
    if seasons is None:
        if force_full_refresh:
            # Full historical collection
            try:
                seasons_window = int(collect_config.get('seasons_window_years', 4))
            except Exception:
                seasons_window = 4
            seasons = _last_n_seasons(max(seasons_window, 1))
            logger.info("🔄 Full refresh mode: Loading NFL PBP for seasons: %s", seasons)
        else:
            # Incremental: only current season
            # NFL season runs Sept-Feb, so determine the correct season year
            current_year = dt.date.today().year
            current_month = dt.date.today().month
            
            # NFL season year is based on the year it starts (Sept)
            # Jan-Aug: use previous year (e.g., Jan 2025 = 2024 season)
            # Sept-Dec: use current year (e.g., Sept 2024 = 2024 season)
            if current_month < 3:
                # Jan-Feb: Previous season's playoffs
                seasons = [current_year - 1]
                logger.info("📥 Incremental update: Loading previous season (playoffs): %s", seasons)
            elif current_month < 9:
                # Mar-Aug: Offseason, collect last 2 completed seasons
                seasons = [current_year - 1, current_year - 2]
                logger.info("📥 Incremental update: Loading last 2 seasons (offseason): %s", seasons)
            else:
                # Sept-Dec: Current season in progress
                seasons = [current_year]
                logger.info("📥 Incremental update: Loading current season: %s", seasons)
    else:
        logger.info("📥 Loading specified NFL PBP seasons: %s", seasons)
    
    # Try to load data, with fallback for unavailable seasons
    pbp = None
    github_fallback_attempted = False
    try:
        pbp = import_pbp_data(years=seasons, downcast=True)
    except Exception as e:
        is_http_404 = isinstance(e, HTTPError) and getattr(e, "code", None) == 404
        if is_http_404:
            logger.info(
                "PBP data not yet published via nfl_data_py for seasons %s (HTTP 404). "
                "Attempting direct download from nflfastR GitHub.",
                seasons,
            )
        else:
            logger.warning(
                "Failed to import PBP data for seasons %s via nfl_data_py: %s. "
                "Attempting direct download from nflfastR GitHub.",
                seasons,
                e,
            )
        github_fallback_attempted = True
        pbp = _download_pbp_from_github(seasons)
        if pbp is None or pbp.empty:
            logger.warning(
                "nflfastR GitHub fallback returned no data for seasons %s", seasons
            )

    if (pbp is None or len(pbp) == 0) and seasons and seasons[0] >= 2024:
        fallback_seasons = [s - 1 for s in seasons]
        if github_fallback_attempted:
            logger.info(
                "Attempting fallback seasons %s because nfl_data_py and GitHub both lacked %s.",
                fallback_seasons,
                seasons,
            )
        else:
            logger.info("Retrying with fallback seasons: %s", fallback_seasons)
        try:
            pbp = import_pbp_data(years=fallback_seasons, downcast=True)
            logger.info("✅ Successfully loaded fallback seasons: %s", fallback_seasons)
        except Exception as e2:
            is_fallback_404 = isinstance(e2, HTTPError) and getattr(e2, "code", None) == 404
            if is_fallback_404:
                logger.info("Fallback seasons %s are also unavailable (HTTP 404).", fallback_seasons)
            else:
                logger.error("Failed to import fallback PBP data: %s", e2)
            return pd.DataFrame()
    elif pbp is None or len(pbp) == 0:
        logger.error("No fallback available for seasons %s", seasons)
        return pd.DataFrame()

    if pbp is None or len(pbp) == 0:
        logger.warning("No PBP data returned")
        return pd.DataFrame()

    # Ensure expected columns exist
    for col in ["game_id","season","week","home_team","away_team"]:
        if col not in pbp.columns:
            pbp[col] = pd.NA

    # Schedules for start times → utc_ts per game and neutral-site venue resolution
    game_start_map = {}
    venue_lookup: dict[str, str] = {}
    season_type_map: dict[str, str] = {}
    try:
        sched = import_schedules(years=seasons)
        if "game_id" in sched.columns:
            # Try standardized UTC start time fields
            ts_col = None
            for cand in ("start_time_utc","game_start_time_utc","gameday_utc"):
                if cand in sched.columns:
                    ts_col = cand
                    break
            if ts_col is None:
                # Fallback: combine gameday + eastern time if present
                date_col = next((c for c in ("gameday","game_date","game_day") if c in sched.columns), None)
                time_col = next((c for c in ("game_time_eastern","gametime","game_time") if c in sched.columns), None)
                if date_col and time_col:
                    dt_str = (sched[date_col].astype(str) + " " + sched[time_col].astype(str)).str.strip()
                    sched["_start_eastern"] = pd.to_datetime(dt_str, errors="coerce").dt.tz_localize("US/Eastern", nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
                    ts_col = "_start_eastern"
            if ts_col is not None:
                game_start_map = sched.set_index("game_id")[ts_col].to_dict()

            venue_cols = [c for c in ("site", "stadium", "location") if c in sched.columns]
            for row in sched.itertuples(index=False):
                game_id = getattr(row, "game_id", None)
                if not game_id:
                    continue
                venue_key = None
                for col in venue_cols:
                    venue_key = _normalize_venue_name(getattr(row, col, None))
                    if venue_key:
                        break
                if venue_key is None:
                    venue_key = _normalize_venue_name(getattr(row, "home_team", None))
                if venue_key:
                    venue_lookup[game_id] = venue_key

            season_type_col = next((c for c in ("season_type", "game_type") if c in sched.columns), None)
            if season_type_col:
                season_type_map = sched.set_index("game_id")[season_type_col].to_dict()
    except Exception as e:
        logger.warning("Schedules import failed or missing start times: %s", e)

    # Attach game_date and utc_ts (constant per game start)
    if "game_date" not in pbp.columns:
        date_from_sched = None
        try:
            date_from_sched = sched.set_index("game_id")[next(c for c in ("gameday","game_date") if c in sched.columns)]
        except Exception:
            pass
        if date_from_sched is not None:
            pbp["game_date"] = pbp["game_id"].map(date_from_sched)
        else:
            pbp["game_date"] = pd.to_datetime("today").date()

    # Capture game start timestamps (UTC)
    pbp["game_start_utc"] = pd.to_datetime(
        pbp["game_id"].map(game_start_map), utc=True, errors="coerce"
    )

    if pbp["game_start_utc"].isna().any():
        game_date_utc = pd.to_datetime(pbp["game_date"], errors="coerce", utc=True)
        fallback_start = game_date_utc.add(pd.to_timedelta(17, unit="h"))
        pbp["game_start_utc"] = pbp["game_start_utc"].fillna(fallback_start)

    # Default utc_ts to game start; refine per play when possible
    pbp["utc_ts"] = pbp["game_start_utc"]

    if {"qtr", "quarter_seconds_remaining"}.issubset(pbp.columns):
        qtr = pd.to_numeric(pbp["qtr"], errors="coerce")
        sec_remaining = pd.to_numeric(pbp["quarter_seconds_remaining"], errors="coerce")

        elapsed_seconds = pd.Series(np.nan, index=pbp.index, dtype="float64")

        reg_mask = (
            qtr.between(1, 4, inclusive="both")
            & sec_remaining.notna()
        )
        if reg_mask.any():
            sec_reg = sec_remaining.clip(lower=0, upper=900)
            elapsed_seconds.loc[reg_mask] = (
                (qtr.loc[reg_mask] - 1) * 900
                + (900 - sec_reg.loc[reg_mask])
            )

        ot_mask = (qtr >= 5) & sec_remaining.notna()
        if ot_mask.any():
            sec_ot = sec_remaining.clip(lower=0, upper=600)
            elapsed_seconds.loc[ot_mask] = (
                4 * 900
                + (qtr.loc[ot_mask] - 5) * 600
                + (600 - sec_ot.loc[ot_mask])
            )

        valid_mask = (
            elapsed_seconds.notna()
            & pbp["game_start_utc"].notna()
        )
        if valid_mask.any():
            pbp.loc[valid_mask, "utc_ts"] = (
                pbp.loc[valid_mask, "game_start_utc"]
                + pd.to_timedelta(elapsed_seconds.loc[valid_mask], unit="s")
            )

    # Stadium metadata: resolve neutral / international venues
    if "game_id" in pbp.columns:
        pbp["stadium_key"] = pbp["game_id"].map(venue_lookup)
    fallback_keys = pbp["home_team"].map(lambda val: _normalize_venue_name(val) or val)
    pbp["stadium_key"] = pbp["stadium_key"].fillna(fallback_keys)
    pbp["stadium_name"] = pbp["stadium_key"].map(lambda key: (STADIUM_COORDS.get(key) or {}).get("stadium"))
    pbp["stadium_tz"] = pbp["stadium_key"].map(lambda key: (STADIUM_COORDS.get(key) or {}).get("tz"))

    if "season_type" in pbp.columns:
        if season_type_map:
            pbp["season_type"] = pbp["season_type"].fillna(pbp["game_id"].map(season_type_map))
        pbp["season_type"] = pbp["season_type"].fillna("REG")
    elif season_type_map:
        pbp["season_type"] = pbp["game_id"].map(season_type_map).fillna("REG")
    else:
        pbp["season_type"] = "REG"

    # Optimize dtypes optionally
    if Config.ENABLE_DTYPE_OPTIMIZATION:
        try:
            pbp = maybe_optimize(pbp, dtype_config)
        except Exception:
            pass

    # Attach weather via existing system (best-effort)
    try:
        pbp = attach_weather(pbp)
    except Exception as e:
        logger.warning("Weather attachment skipped: %s", e)

    # Write monolithic files for back-compat
    # Skip monolithic write - weekly Parquet is the canonical store

    # Weekly play-of-record partitions are not persisted; we now derive weekly
    # outputs from cleaned daily. Only write raw daily with season/week/date.
    # Raw day-level Parquets under data/raw:
    #   data/raw/pbp_by_day/season=YYYY/week=WW/date=YYYY-MM-DD/part.parquet
    raw_day_base = PROJ_ROOT / "data" / "raw" / "pbp_by_day"
    raw_day_base.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for (season, week), dfw in pbp.groupby(["season","week"], dropna=False):
        if pd.isna(season) or pd.isna(week):
            continue
        if limit_weeks is not None and int(week) > int(limit_weeks):
            continue
        # Write raw day Parquets for this week's plays
        try:
            if 'game_date' in dfw.columns:
                dfw['game_date'] = pd.to_datetime(dfw['game_date'], errors='coerce').dt.date
                for day, dfd in dfw.groupby('game_date'):
                    if pd.isna(day):
                        continue
                    day_dir = raw_day_base / f"season={int(season)}" / f"week={int(week)}" / f"date={day}"
                    day_dir.mkdir(parents=True, exist_ok=True)
                    (day_dir / "part.parquet").unlink(missing_ok=True)
                    dfd.to_parquet(day_dir / "part.parquet", index=False, compression="zstd")
        except Exception as e:
            logger.warning("Failed writing raw day Parquets: %s", e)

    odds_start: dt.date | None = None
    odds_end: dt.date | None = None
    if "game_date" in pbp.columns:
        try:
            game_dates = pd.to_datetime(pbp["game_date"], errors="coerce").dropna()
            if not game_dates.empty:
                odds_start = game_dates.min().date()
                odds_end = game_dates.max().date()
        except Exception:
            pass

    pbp_seasons: list[int] = []
    if "season" in pbp.columns:
        try:
            season_series = pd.to_numeric(pbp["season"], errors="coerce").dropna().astype(int)
            pbp_seasons = sorted(set(season_series.tolist()))
        except Exception:
            pbp_seasons = []

    try:
        collect_odds_snapshots(
            seasons=pbp_seasons or seasons,
            start_date=odds_start,
            end_date=odds_end,
            include_player_props=include_player_props,
        )
    except Exception as exc:
        logger.warning("Odds snapshot collection skipped: %s", exc)

    try:
        collect_weather_forecasts(
            seasons=pbp_seasons or seasons,
            limit_weeks=limit_weeks,
        )
    except Exception as exc:
        logger.warning("Weather forecast snapshot collection skipped: %s", exc)

    elapsed = time.time() - start_time
    logger.info("✅ PBP collection done in %.2fs; rows=%s; raw daily rows written=%s", elapsed, len(pbp), wrote)
    try:
        # Run audit on in-memory PBP slice to avoid scanning legacy CSV
        run_collect_audit(df=pbp)
    except Exception as e:
        logger.warning("Collect audit skipped: %s", e)
    return pbp


# Legacy alias removed - use collect_pbp() directly
# collect_statcast was a baseball-era alias that no longer applies to NFL


def _cli():
    """Command line interface for the NFL data collection script"""
    parser = argparse.ArgumentParser(description="Collect NFL Play-by-Play Data")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD) - not used with seasons mode")
    parser.add_argument("--end", help="End date (YYYY-MM-DD) - not used with seasons mode")
    parser.add_argument("--seasons", type=str, default="",
                       help="Comma-separated seasons (e.g., 2024,2023,2022). Default: last 4 from config.")
    parser.add_argument("--limit-weeks", type=int, default=None,
                       help="Write only weeks <= this value per season (for testing).")
    parser.add_argument("--backfill-weather", action="store_true",
                       help="Back-fill weather data for rows currently missing it")
    parser.add_argument("--audit-only", action="store_true",
                       help="Run audit on existing collected data without collecting new data")
    parser.add_argument("--cache-schedules", action="store_true",
                       help="Pre-cache NFL schedules for specified seasons")
    parser.add_argument(
        "--include-player-props",
        action="store_true",
        help="When collecting odds snapshots, also fetch player prop markets (anytime TD, TD totals).",
    )
    parser.add_argument("--collect-injuries", action="store_true",
                       help="Fetch ESPN injury reports for the requested seasons")
    parser.add_argument("--overwrite", action="store_true",
                       help="Force refresh of cached data (injuries, weather) even if it already exists")
    parser.add_argument("--collect-offense-coordinators", action="store_true",
                       help="Build or refresh the offensive coordinator mapping for the requested seasons")
    parser.add_argument("--collect-weather-forecasts", action="store_true",
                       help="Fetch and persist weather forecast snapshots for the requested seasons")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(PROJ_ROOT / "logs" / "collect.log", mode='a')
        ]
    )
    
    # Ensure log directory exists
    (PROJ_ROOT / "logs").mkdir(exist_ok=True)
    
    if args.backfill_weather:
        backfill_weather(start=args.start, end=args.end)
    elif args.cache_schedules:
        sel_seasons = None
        if args.seasons:
            try:
                sel_seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
            except Exception:
                sel_seasons = _last_n_seasons(4)
        else:
            sel_seasons = _last_n_seasons(4)
        cache_schedules(sel_seasons)
    elif args.collect_injuries:
        if args.seasons:
            try:
                sel_seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
            except Exception:
                logger.error("Unable to parse --seasons argument: %s", args.seasons)
                return
        else:
            sel_seasons = [_last_n_seasons(1)[-1]]

        from utils.collect.espn_injuries import collect_espn_injuries
        from utils.collect.injury_transactions import collect_injury_transactions

        written = collect_espn_injuries(sel_seasons, overwrite=args.overwrite)
        if written:
            logger.info("Injury caches written: %s", ", ".join(str(p) for p in written))
        else:
            logger.warning("No injury data collected.")
        tx_written = collect_injury_transactions(sel_seasons)
        if tx_written:
            logger.info(
                "Injury transaction caches written: %s",
                ", ".join(str(p) for p in tx_written),
            )
    elif args.collect_offense_coordinators:
        if args.seasons:
            try:
                sel_seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
            except Exception:
                logger.error("Unable to parse --seasons argument: %s", args.seasons)
                return
        else:
            sel_seasons = _last_n_seasons(4)

        from utils.collect.offensive_coordinators import build_offensive_coordinator_map

        path = build_offensive_coordinator_map(sel_seasons)
        logger.info("Offensive coordinator map written to %s", path)
    elif args.collect_weather_forecasts:
        if args.seasons:
            try:
                sel_seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
            except Exception:
                logger.error("Unable to parse --seasons argument: %s", args.seasons)
                return
        else:
            sel_seasons = _last_n_seasons(1)

        written = collect_weather_forecasts(
            seasons=sel_seasons,
            limit_weeks=args.limit_weeks,
        )
        if written:
            logger.info("Weather forecast snapshots written: %s", ", ".join(str(p) for p in written))
        else:
            logger.warning("No weather forecast snapshots collected.")
    elif args.audit_only:
        # Load existing data and audit
        logger.info("Running audit on existing NFL data...")
        try:
            # Try to load some recent data for audit
            import glob
            recent_files = sorted(glob.glob(str(NFL_PBP_RAW_DIR / "season=*/week=*/date=*/part.parquet")))
            if recent_files:
                sample_df = pd.read_parquet(recent_files[-1])  # Most recent file
                run_collect_audit(df=sample_df)
            else:
                logger.warning("No data files found to audit")
        except Exception as e:
            logger.error(f"Audit failed: {e}")
    else:
        sel_seasons = None
        if args.seasons:
            try:
                sel_seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
            except Exception:
                sel_seasons = None
        collect_pbp(
            start=args.start,
            end=args.end,
            seasons=sel_seasons,
            limit_weeks=args.limit_weeks,
            include_player_props=args.include_player_props,
        )

def collect_nfl_data(start_date: dt.date | None = None, end_date: dt.date | None = None, 
                     seasons: list[int] | None = None,
                     force_full_refresh: bool = False) -> pd.DataFrame:
    """Convenience wrapper for NFL data collection with date objects.
    
    **Smart Collection Strategy:**
    - By default: Only collects CURRENT season (fast incremental updates)
    - With force_full_refresh=True: Re-downloads all historical seasons
    
    This ensures `python main.py collect` is fast and only gets new data.
    
    Parameters
    ----------
    start_date : date, optional
        Start date for collection (not used with seasons approach)
    end_date : date, optional
        End date for collection (not used with seasons approach)
    seasons : list[int], optional
        Specific NFL seasons to collect (e.g., [2024, 2023])
        If None, defaults to current season only
    force_full_refresh : bool, default False
        If True, collects all historical seasons (4 years)
        If False, only collects current season (incremental)
    
    Returns
    -------
    pd.DataFrame
        Collected play-by-play data
        
    Examples
    --------
    # Incremental update (fast, recommended for daily runs)
    >>> collect_nfl_data()
    
    # Full historical refresh (slow, ~180K plays)
    >>> collect_nfl_data(force_full_refresh=True)
    
    # Specific seasons
    >>> collect_nfl_data(seasons=[2024, 2023])
    """
    start_str = start_date.isoformat() if start_date else None
    end_str = end_date.isoformat() if end_date else None
    
    return collect_pbp(
        start=start_str,
        end=end_str,
        seasons=seasons,
        force_full_refresh=force_full_refresh,
        include_player_props=False,
    )


if __name__ == "__main__":
    _cli()
