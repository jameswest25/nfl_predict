"""Prediction scaffold builder.

This module handles building the player-game scaffold for inference,
including loading schedules, rosters, and creating the base dataframe.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from utils.general.paths import PROJ_ROOT, STADIUM_COORDS_FILE
from utils.collect.nfl_schedules import get_schedule as get_cached_schedule

logger = logging.getLogger(__name__)

# Import nfl_data_py (with graceful degradation)
try:
    from nfl_data_py import import_schedules, import_weekly_rosters
except Exception as exc:
    import_schedules = None
    import_weekly_rosters = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

# Position filters
ALLOWED_POSITIONS = {"QB", "RB", "WR", "TE"}

# Status exclusions
EXCLUDED_ROSTER_STATUSES = {
    "IR", "IR-R", "IR-N", "IR-C", "PUP", "NFI", "DNR", "SUS", "EXEMPT",
    "RET", "COVID", "PS", "WAV", "CUT", "RES", "RESERVE", "RES-RET", "DEV"
}
EXCLUDED_GAME_STATUSES = {"OUT", "DOUBTFUL", "INACTIVE", "SUSPENDED"}

# Load stadium coordinates
with STADIUM_COORDS_FILE.open() as fh:
    _STADIUM_COORDS = json.load(fh)

_STADIUM_ALIAS_MAP: dict[str, str] = {}
for code, meta in _STADIUM_COORDS.items():
    if isinstance(meta, dict):
        _STADIUM_ALIAS_MAP[code.upper()] = code
        for alias in (meta.get("aliases") or []):
            _STADIUM_ALIAS_MAP[str(alias).upper()] = code


def _normalize_stadium(raw: str | None) -> str | None:
    """Normalize stadium name to canonical key."""
    if not raw:
        return None
    return _STADIUM_ALIAS_MAP.get(raw.strip().upper())


def ensure_imports_ready() -> None:
    """Ensure nfl_data_py is available."""
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            f"nfl_data_py is required for the prediction pipeline. "
            f"Import failed with: {_IMPORT_ERROR}"
        )


def load_schedule(start: date, end: date) -> pd.DataFrame:
    """Load NFL game schedule for date range.
    
    Parameters
    ----------
    start : date
        Start date (inclusive)
    end : date
        End date (inclusive)
    
    Returns
    -------
    pd.DataFrame
        Schedule with game metadata
    """
    ensure_imports_ready()
    
    seasons = sorted({start.year, end.year})
    sched = import_schedules(years=seasons)
    sched["gameday"] = pd.to_datetime(sched["gameday"]).dt.date

    if "season_type" in sched.columns:
        valid_season = sched["season_type"].isin(["REG", "POST"])
    elif "game_type" in sched.columns:
        valid_season = sched["game_type"].isin(["REG", "POST"])
    else:
        valid_season = True

    mask = (
        (sched["gameday"] >= start)
        & (sched["gameday"] <= end)
        & valid_season
    )

    keep_cols = [
        "game_id", "gameday", "week", "season", "home_team", "away_team",
        "start_time_utc", "season_type", "game_type", "roof", "surface",
        "stadium", "location", "site",
    ]
    available = [col for col in keep_cols if col in sched.columns]
    games = sched.loc[mask, available].copy()
    games.rename(columns={"gameday": "game_date"}, inplace=True)
    
    if "game_type" in games.columns and "season_type" not in games.columns:
        games["season_type"] = games["game_type"]
    
    if "start_time_utc" in games.columns:
        games["start_time_utc"] = pd.to_datetime(
            games["start_time_utc"], errors="coerce", utc=True
        )
    else:
        games["start_time_utc"] = pd.NaT
    
    games.sort_values(["game_date", "game_id"], inplace=True)

    # Try to augment with cached schedule data
    try:
        cached_sched = get_cached_schedule(list(seasons))
    except Exception as exc:
        logger.warning("Falling back to nfl_data_py schedule only: %s", exc)
        cached_sched = pd.DataFrame()

    if not cached_sched.empty:
        cached_cols = ["game_id", "start_time_utc", "roof", "surface", "stadium", "location", "site"]
        cache_available = [col for col in cached_cols if col in cached_sched.columns]
        cache_frame = cached_sched[cache_available].copy()
        
        if "start_time_utc" in cache_frame.columns:
            cache_frame["start_time_utc"] = pd.to_datetime(
                cache_frame["start_time_utc"], errors="coerce", utc=True
            )
        
        rename_map = {col: f"{col}__cache" for col in cache_available if col != "game_id"}
        cache_frame = cache_frame.rename(columns=rename_map)
        games = games.merge(cache_frame, on="game_id", how="left")

        for col in ("start_time_utc", "roof", "surface", "stadium", "location", "site"):
            cache_col = f"{col}__cache"
            if cache_col in games.columns:
                if col == "start_time_utc":
                    games[col] = pd.to_datetime(games[col], errors="coerce", utc=True)
                    games[col] = games[col].where(games[col].notna(), games[cache_col])
                else:
                    games[col] = games[col].where(games[col].notna(), games[cache_col])
                games.drop(columns=[cache_col], inplace=True)

    if games.empty:
        raise ValueError(f"No scheduled games found between {start} and {end}.")
    
    return games.reset_index(drop=True)


def load_rosters(seasons: Iterable[int]) -> pd.DataFrame:
    """Load weekly roster data for given seasons.
    
    Parameters
    ----------
    seasons : Iterable[int]
        Seasons to load
    
    Returns
    -------
    pd.DataFrame
        Roster data with player info
    """
    ensure_imports_ready()
    
    roster = import_weekly_rosters(years=list(seasons))
    
    if "gsis_id" in roster.columns:
        roster["player_id"] = roster["gsis_id"]
    elif "player_id" not in roster.columns:
        raise ValueError("Roster data missing 'player_id'/'gsis_id' identifier.")

    name_col = None
    for cand in ("full_name", "display_name", "player_name"):
        if cand in roster.columns:
            name_col = cand
            break
    if name_col is None:
        roster["full_name"] = (
            roster.get("first_name", "").fillna("") + " " + 
            roster.get("last_name", "").fillna("")
        )
        name_col = "full_name"

    keep_cols = [
        "player_id", name_col, "position", "team", "season", "week", 
        "status", "depth_chart_position", "depth_chart_order"
    ]
    available = [c for c in keep_cols if c in roster.columns]
    roster = roster[available].copy()
    roster.rename(columns={name_col: "player_name"}, inplace=True)
    
    roster["player_id"] = roster["player_id"].astype(str)
    roster["player_name"] = roster["player_name"].fillna("Unknown Player").str.strip()
    roster["position"] = roster["position"].fillna("UNK")
    roster["team"] = roster["team"].fillna("UNK")
    roster["status"] = roster["status"].fillna("UNK")
    
    if "depth_chart_position" not in roster.columns:
        roster["depth_chart_position"] = ""
    roster["depth_chart_position"] = roster["depth_chart_position"].fillna("")
    
    return roster


def build_scaffold(games: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Build player-game scaffold for prediction.
    
    Parameters
    ----------
    games : pd.DataFrame
        Schedule data from load_schedule()
    roster : pd.DataFrame
        Roster data from load_rosters()
    
    Returns
    -------
    pd.DataFrame
        Scaffold with one row per player-game
    """
    records: list[dict] = []
    roster_key = roster.set_index(["season", "week", "team"])

    def _is_player_available(row: pd.Series) -> bool:
        status = str(row.get("status", "") or "").strip().upper()
        if status and (status in EXCLUDED_ROSTER_STATUSES or status.startswith("IR") or status.startswith("RESERVE")):
            return False
        game_status = str(
            row.get("injury_game_status", "") or row.get("injury_game_designation", "") or ""
        ).strip().upper()
        if game_status and any(flag in game_status for flag in EXCLUDED_GAME_STATUSES):
            return False
        return True

    for _, game in games.iterrows():
        season = int(game["season"])
        week = int(game["week"])
        game_id = game["game_id"]
        game_date = game["game_date"]
        home = game["home_team"]
        away = game["away_team"]
        
        raw_venue = game.get("stadium") or game.get("location") or game.get("site")
        stadium_key = _normalize_stadium(raw_venue) or _normalize_stadium(home) or home
        venue_meta = _STADIUM_COORDS.get(stadium_key, {}) if stadium_key else {}
        stadium_name = venue_meta.get("stadium") or raw_venue or stadium_key
        stadium_tz = venue_meta.get("tz")
        roof = game.get("roof") or venue_meta.get("roof")
        surface = game.get("surface") or venue_meta.get("surface")
        
        start_utc = game.get("start_time_utc")
        start_hour_utc = None
        if pd.notna(start_utc):
            start_utc = pd.to_datetime(start_utc, utc=True)
            start_hour_utc = int(start_utc.hour)
        
        season_type = game.get("season_type") or "REG"
        game_day_of_week = (
            int(pd.to_datetime(game_date).weekday()) if pd.notna(game_date) else None
        )

        for team, opponent in ((home, away), (away, home)):
            try:
                players = roster_key.loc[(season, week, team)]
            except KeyError:
                continue

            if isinstance(players, pd.Series):
                players = players.to_frame().T

            for _, row in players.iterrows():
                position = str(row.get("position", "") or "").upper()
                if position not in ALLOWED_POSITIONS:
                    continue
                if not _is_player_available(row):
                    continue

                records.append({
                    "player_id": str(row["player_id"]),
                    "player_name": row["player_name"],
                    "position": position,
                    "position_group": row.get("position_group", ""),
                    "team": team,
                    "opponent": opponent,
                    "season": season,
                    "week": week,
                    "game_id": game_id,
                    "game_date": datetime.combine(game_date, datetime.min.time()),
                    "status": row.get("status", "UNK"),
                    "depth_chart_position": row.get("depth_chart_position", ""),
                    "depth_chart_order": row.get("depth_chart_order"),
                    "injury_game_designation": row.get("injury_game_designation", ""),
                    "home_team": home,
                    "away_team": away,
                    "is_home": 1 if team == home else 0,
                    "season_type": season_type,
                    "stadium_key": stadium_key,
                    "stadium_name": stadium_name,
                    "stadium_tz": stadium_tz,
                    "roof": roof,
                    "surface": surface,
                    "game_start_utc": start_utc,
                    "game_start_hour_utc": start_hour_utc,
                    "game_day_of_week": game_day_of_week,
                })

    if not records:
        raise ValueError("Unable to build prediction scaffold – roster data missing for scheduled games.")

    df = pd.DataFrame.from_records(records).drop_duplicates(subset=["player_id", "game_id"])
    
    # Apply status filters
    if "status" in df.columns:
        df["status"] = df["status"].fillna("").astype(str).str.strip().str.upper()
        df = df[~df["status"].isin(EXCLUDED_ROSTER_STATUSES)]
    
    if "injury_game_designation" in df.columns:
        df["injury_game_designation"] = df["injury_game_designation"].fillna("").astype(str).str.strip()
        df = df[
            ~df["injury_game_designation"].str.upper().apply(
                lambda s: any(flag in s for flag in EXCLUDED_GAME_STATUSES) if s else False
            )
        ]
    
    # Type conversions
    if "game_start_utc" in df.columns:
        df["game_start_utc"] = pd.to_datetime(df["game_start_utc"], utc=True, errors="coerce")
    if "depth_chart_order" in df.columns:
        df["depth_chart_order"] = pd.to_numeric(df["depth_chart_order"], errors="coerce")
    for col in ("is_home", "game_start_hour_utc", "game_day_of_week"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df
