from __future__ import annotations

import json
import logging
from datetime import date
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import polars as pl
from zoneinfo import ZoneInfo

from utils.collect.nfl_schedules import get_schedule
from utils.general.paths import STADIUM_COORDS_JSON, TRAVEL_CALENDAR_DIR

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

NEUTRAL_SITE_OVERRIDES: dict[str, dict[str, object]] = {
    "wembley stadium": {"stadium": "Wembley Stadium", "lat": 51.5560, "lon": -0.2796, "tz": "Europe/London"},
    "tottenham hotspur stadium": {"stadium": "Tottenham Hotspur Stadium", "lat": 51.6043, "lon": -0.0664, "tz": "Europe/London"},
    "twickenham stadium": {"stadium": "Twickenham Stadium", "lat": 51.4550, "lon": -0.3417, "tz": "Europe/London"},
    "estadio azteca": {"stadium": "Estadio Azteca", "lat": 19.3030, "lon": -99.1505, "tz": "America/Mexico_City"},
    "allianz arena": {"stadium": "Allianz Arena", "lat": 48.2188, "lon": 11.6247, "tz": "Europe/Berlin"},
    "deutsche bank park": {"stadium": "Deutsche Bank Park", "lat": 50.0688, "lon": 8.6452, "tz": "Europe/Berlin"},
    "nissan stadium": {"stadium": "Nissan Stadium (Nagoya)", "lat": 35.510, "lon": 139.606, "tz": "Asia/Tokyo"},
}


def _load_stadium_metadata() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Return team and stadium lookup tables with coordinates and timezones."""
    path = Path(STADIUM_COORDS_JSON)
    if not path.exists():
        raise FileNotFoundError(f"Stadium coordinate file not found at {path}")
    with path.open("r") as handle:
        team_map: dict[str, dict[str, object]] = json.load(handle)
    stadium_map: dict[str, dict[str, object]] = {}
    for team, info in team_map.items():
        stadium_name = info.get("stadium")
        if stadium_name:
            stadium_map[stadium_name.strip().lower()] = {
                "stadium": stadium_name,
                "lat": info.get("lat"),
                "lon": info.get("lon"),
                "tz": info.get("tz"),
            }
    stadium_map.update(NEUTRAL_SITE_OVERRIDES)
    return team_map, stadium_map


def _lookup_stadium(stadium_name: str | None, home_team: str, team_map: dict[str, dict[str, object]], stadium_map: dict[str, dict[str, object]]) -> dict[str, object]:
    if stadium_name:
        info = stadium_map.get(stadium_name.strip().lower())
        if info:
            return info
    return team_map.get(home_team, {"stadium": None, "lat": None, "lon": None, "tz": None})


def _haversine_km(lat1: float | None, lon1: float | None, lat2: float | None, lon2: float | None) -> float | None:
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return EARTH_RADIUS_KM * c


def _tz_offset_hours(tz_name: str | None, ts_utc: pd.Timestamp) -> float | None:
    if not tz_name or pd.isna(ts_utc):
        return None
    try:
        tz = ZoneInfo(tz_name)
        offset = ts_utc.astimezone(tz).utcoffset()
        if offset is None:
            return None
        return offset.total_seconds() / 3600.0
    except Exception:
        return None


def _local_start_time(tz_name: str | None, ts_utc: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    if not tz_name or pd.isna(ts_utc):
        return pd.NaT
    try:
        return ts_utc.astimezone(ZoneInfo(tz_name))
    except Exception:
        return pd.NaT


def _build_team_level_rows(schedule: pd.DataFrame, team_map: dict[str, dict[str, object]], stadium_map: dict[str, dict[str, object]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in schedule.iterrows():
        start_time = pd.to_datetime(row.get("start_time_utc"), utc=True)
        game_date = pd.to_datetime(row.get("gameday")).date() if pd.notna(row.get("gameday")) else start_time.date()
        week = int(row.get("week"))
        season = int(row.get("season"))
        game_type = row.get("game_type")
        game_id = row.get("game_id")
        stadium_name = row.get("stadium")
        location = row.get("location")
        weekday = row.get("weekday")
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        stadium_info = _lookup_stadium(stadium_name, home_team, team_map, stadium_map)
        for team, opponent, is_home, team_score, opp_score in (
            (home_team, away_team, True, row.get("home_score"), row.get("away_score")),
            (away_team, home_team, False, row.get("away_score"), row.get("home_score")),
        ):
            team_info = team_map.get(team, {})
            records.append(
                {
                    "season": season,
                    "week": week,
                    "game_type": game_type,
                    "game_id": game_id,
                    "team": team,
                    "opponent": opponent,
                    "is_home": bool(is_home),
                    "team_score": team_score,
                    "opponent_score": opp_score,
                    "start_time_utc": start_time,
                    "game_date": game_date,
                    "weekday": weekday,
                    "location": location,
                    "stadium": stadium_name,
                    "home_team": home_team,
                    "stadium_lat": stadium_info.get("lat"),
                    "stadium_lon": stadium_info.get("lon"),
                    "stadium_tz": stadium_info.get("tz"),
                    "team_home_lat": team_info.get("lat"),
                    "team_home_lon": team_info.get("lon"),
                    "team_home_tz": team_info.get("tz"),
                }
            )
    df = pd.DataFrame.from_records(records)
    df.sort_values(["team", "start_time_utc"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _annotate_sequences(df: pd.DataFrame) -> pd.DataFrame:
    df["season_game_number"] = df.groupby("team").cumcount() + 1
    df["prev_start_time"] = df.groupby("team")["start_time_utc"].shift(1)
    df["rest_days"] = (df["start_time_utc"] - df["prev_start_time"]).dt.total_seconds() / (3600.0 * 24.0)
    df["rest_hours"] = df["rest_days"] * 24.0
    df["prev_stadium_lat"] = df.groupby("team")["stadium_lat"].shift(1)
    df["prev_stadium_lon"] = df.groupby("team")["stadium_lon"].shift(1)
    df["travel_km"] = [
        _haversine_km(prev_lat, prev_lon, lat, lon)
        for prev_lat, prev_lon, lat, lon in zip(
            df["prev_stadium_lat"],
            df["prev_stadium_lon"],
            df["stadium_lat"],
            df["stadium_lon"],
        )
    ]
    df["travel_miles"] = df["travel_km"] * 0.621371

    df["team_timezone_offset"] = [
        _tz_offset_hours(tz, ts) for tz, ts in zip(df["team_home_tz"], df["start_time_utc"])
    ]
    df["game_timezone_offset"] = [
        _tz_offset_hours(tz, ts) for tz, ts in zip(df["stadium_tz"], df["start_time_utc"])
    ]
    df["prev_game_timezone_offset"] = df.groupby("team")["game_timezone_offset"].shift(1)
    df["timezone_change_hours"] = df["game_timezone_offset"] - df["prev_game_timezone_offset"]
    df["time_diff_from_home_hours"] = df["game_timezone_offset"] - df["team_timezone_offset"]

    df["game_local_start"] = [
        _local_start_time(tz, ts) for tz, ts in zip(df["stadium_tz"], df["start_time_utc"])
    ]
    df["local_start_hour"] = [
        np.nan
        if (ts is None or ts is pd.NaT)
        else ts.hour + ts.minute / 60.0
        for ts in df["game_local_start"]
    ]

    df["is_home_int"] = df["is_home"].astype(int)
    consecutive_road = np.zeros(len(df), dtype=np.int32)
    consecutive_home = np.zeros(len(df), dtype=np.int32)
    for team, group in df.groupby("team", sort=False):
        idx = group.index.to_numpy()
        road_counter = 0
        home_counter = 0
        for position, row_idx in enumerate(idx):
            if df.at[row_idx, "is_home"]:
                home_counter += 1
                road_counter = 0
            else:
                road_counter += 1
                home_counter = 0
            consecutive_road[row_idx] = road_counter
            consecutive_home[row_idx] = home_counter
    df["consecutive_road_games"] = consecutive_road
    df["consecutive_home_games"] = consecutive_home

    df["is_short_week"] = df["rest_days"].le(5.0)
    df["is_long_rest"] = df["rest_days"].ge(10.0)
    df["bye_week_flag"] = df["rest_days"].ge(13.0)

    df["west_to_east_early"] = (df["time_diff_from_home_hours"] < -1.5) & (df["local_start_hour"] < 13.0)
    df["east_to_west_late"] = (df["time_diff_from_home_hours"] > 1.5) & (df["local_start_hour"] > 19.0)

    df["travel_km_rolling3"] = df.groupby("team")["travel_km"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["rest_days_rolling3"] = df.groupby("team")["rest_days"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    return df


def build_travel_calendar(seasons: Sequence[int]) -> None:
    """Assemble travel distance, rest, and calendar-based context features for each team."""
    if not seasons:
        raise ValueError("At least one season must be supplied.")
    schedule = get_schedule(list(seasons))
    if schedule is None or schedule.empty:
        logger.warning("No schedule data available for seasons %s", seasons)
        return

    team_map, stadium_map = _load_stadium_metadata()
    team_rows = _build_team_level_rows(schedule, team_map, stadium_map)
    if team_rows.empty:
        logger.warning("Travel calendar builder produced no team rows.")
        return

    annotated = _annotate_sequences(team_rows)

    keep_cols = [
        "season",
        "week",
        "game_type",
        "team",
        "opponent",
        "is_home",
        "team_score",
        "opponent_score",
        "start_time_utc",
        "game_date",
        "weekday",
        "location",
        "stadium",
        "stadium_lat",
        "stadium_lon",
        "stadium_tz",
        "team_home_lat",
        "team_home_lon",
        "team_home_tz",
        "season_game_number",
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
        "game_local_start",
        "local_start_hour",
        "consecutive_road_games",
        "consecutive_home_games",
        "is_short_week",
        "is_long_rest",
        "bye_week_flag",
        "west_to_east_early",
        "east_to_west_late",
    ]

    travel_df = pl.from_pandas(annotated[keep_cols])
    travel_df = travel_df.with_columns(
        [
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("is_home").cast(pl.Int8),
            pl.col("team_score").cast(pl.Float32),
            pl.col("opponent_score").cast(pl.Float32),
            pl.col("rest_days").cast(pl.Float32),
            pl.col("rest_hours").cast(pl.Float32),
            pl.col("rest_days_rolling3").cast(pl.Float32),
            pl.col("travel_km").cast(pl.Float32),
            pl.col("travel_miles").cast(pl.Float32),
            pl.col("travel_km_rolling3").cast(pl.Float32),
            pl.col("timezone_change_hours").cast(pl.Float32),
            pl.col("time_diff_from_home_hours").cast(pl.Float32),
            pl.col("game_timezone_offset").cast(pl.Float32),
            pl.col("team_timezone_offset").cast(pl.Float32),
            pl.col("local_start_hour").cast(pl.Float32),
            pl.col("consecutive_road_games").cast(pl.Int32),
            pl.col("consecutive_home_games").cast(pl.Int32),
            pl.col("is_short_week").cast(pl.Int8),
            pl.col("is_long_rest").cast(pl.Int8),
            pl.col("bye_week_flag").cast(pl.Int8),
            pl.col("west_to_east_early").cast(pl.Int8),
            pl.col("east_to_west_late").cast(pl.Int8),
        ]
    )

    for (season, week), frame in travel_df.group_by(["season", "week"], maintain_order=True):
        out_dir = TRAVEL_CALENDAR_DIR / f"season={int(season)}" / f"week={int(week)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "part.parquet"
        frame.write_parquet(out_file, compression="zstd")
    logger.info(
        "Travel calendar features written for seasons %s (%d rows).",
        seasons,
        len(travel_df),
    )


__all__ = ["build_travel_calendar"]

