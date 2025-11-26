"""Utilities to fetch NFL injury reports from ESPN's public API.

The nflverse injuries feed currently stops at the 2024 season, so we backfill
2025 (and future) data directly from ESPN.  We normalize the output so it
resembles the structure produced by ``nfl_data_py.import_injuries`` and write
the result into the same cache directory (`cache/feature/injuries`) that the
feature pipeline expects.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from nfl_data_py import import_players, import_schedules

from utils.general.paths import PROJ_ROOT
from utils.collect.arrival_log import log_feed_arrivals

logger = logging.getLogger(__name__)

ESPN_BASE = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/"
    "{team}/injuries?lang=en&region=us"
)

TEAM_SLUG_OVERRIDE = {
    "LA": "lar",
    "LAR": "lar",
    "WAS": "wsh",
}

# Reuse the feature-cache directory so the downstream loader can pick the files
# up without any additional changes.
INJURY_CACHE_DIR = PROJ_ROOT / "cache" / "feature" / "injuries"
INJURY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class InjuryRecord:
    season: int
    team: str
    week: int
    game_type: str
    game_date: dt.date
    game_id: str | None
    game_start_utc: dt.datetime | None
    espn_id: str
    player_id: str | None
    status: str | None
    status_abbr: str | None
    status_description: str | None
    injury_type_id: str | None
    short_comment: str | None
    long_comment: str | None
    source_description: str | None
    source_state: str | None
    reported_at: dt.datetime | None
    injury_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "season": self.season,
            "team": self.team,
            "week": self.week,
            "game_type": self.game_type,
            "game_date": self.game_date,
            "game_id": self.game_id,
            "game_start_utc": self.game_start_utc,
            "espn_id": self.espn_id,
            "player_id": self.player_id,
            "report_status": self.status,
            "report_status_abbr": self.status_abbr,
            "report_status_description": self.status_description,
            "injury_type_id": self.injury_type_id,
            "short_comment": self.short_comment,
            "long_comment": self.long_comment,
            "source_description": self.source_description,
            "source_state": self.source_state,
            "reported_at": self.reported_at,
            "injury_id": self.injury_id,
        }


def _team_list_from_schedule(schedule: pd.DataFrame) -> list[str]:
    home = schedule["home_team"].unique().tolist()
    away = schedule["away_team"].unique().tolist()
    return sorted({team for team in home + away if isinstance(team, str)})


def _build_schedule_lookup(season: int) -> dict[str, pd.DataFrame]:
    schedule = import_schedules([season])
    if schedule.empty:
        raise ValueError(f"No schedule data available for season {season}")

    schedule = schedule.rename(columns={"gameday": "game_date"})
    schedule["game_date"] = pd.to_datetime(schedule["game_date"]).dt.date
    if "start_time_utc" in schedule.columns:
        schedule["start_time_utc"] = pd.to_datetime(schedule["start_time_utc"], utc=True, errors="coerce")
    else:
        schedule["start_time_utc"] = pd.NaT

    frames: list[pd.DataFrame] = []
    for col in ("home_team", "away_team"):
        tmp = schedule[["game_id", "season", "week", "game_type", "game_date", "start_time_utc", col]].rename(
            columns={col: "team"}
        )
        frames.append(tmp)
    schedule_long = pd.concat(frames, ignore_index=True)
    schedule_long = schedule_long.sort_values(["team", "game_date", "week"])

    lookup: dict[str, pd.DataFrame] = {}
    for team, group in schedule_long.groupby("team"):
        lookup[team] = group.reset_index(drop=True)
    return lookup


def _load_player_mappings() -> tuple[dict[str, str], dict[str, str]]:
    players = import_players()
    players = players.dropna(subset=["espn_id"])
    players["espn_id"] = players["espn_id"].astype(str)
    espn_to_gsis = dict(zip(players["espn_id"], players["gsis_id"]))
    espn_to_name = dict(zip(players["espn_id"], players["display_name"]))
    return espn_to_gsis, espn_to_name


def _team_to_slug(team: str) -> str:
    return TEAM_SLUG_OVERRIDE.get(team.upper(), team.lower())


def _iter_team_injury_refs(team_slug: str) -> Iterable[str]:
    page = 1
    while True:
        url = f"{ESPN_BASE.format(team=team_slug)}&page={page}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("items", [])
        if not items:
            break
        for item in items:
            href = item.get("$ref")
            if href:
                yield href
        if page >= payload.get("pageCount", page):
            break
        page += 1


def _parse_athlete_id(ref: str) -> str:
    # .../athletes/<id>/injuries/<injury_id>
    try:
        return ref.split("/athletes/")[1].split("/")[0]
    except Exception:  # pragma: no cover - defensive parsing
        return ""


def _parse_injury_id(ref: str) -> str:
    # trailing segment after /injuries/
    try:
        return ref.split("/injuries/")[1].split("?")[0]
    except Exception:  # pragma: no cover - defensive parsing
        return ""


def _assign_week(
    team: str,
    injury_date: dt.date,
    schedule_lookup: dict[str, pd.DataFrame],
) -> tuple[int, str, dt.date, str | None, dt.datetime | None]:
    team_sched = schedule_lookup.get(team)
    if team_sched is None or team_sched.empty:
        return 0, "UNK", injury_date, None, None

    future_mask = team_sched["game_date"] >= injury_date
    if future_mask.any():
        row = team_sched.loc[future_mask].iloc[0]
    else:
        # If all games are in the past (e.g., postseason complete) take the last entry.
        row = team_sched.iloc[-1]

    game_id_val = row.get("game_id") if "game_id" in row.index else None
    if pd.isna(game_id_val):
        game_id_val = None
    elif game_id_val is not None:
        game_id_val = str(game_id_val)

    start_ts = row.get("start_time_utc")
    if isinstance(start_ts, pd.Timestamp):
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        start_ts = start_ts.to_pydatetime()
    else:
        start_ts = None

    return int(row["week"]), str(row["game_type"]), row["game_date"], game_id_val, start_ts


def collect_espn_injuries(
    seasons: Iterable[int],
    overwrite: bool = False,
) -> list[Path]:
    """Fetch and cache injury reports for the given seasons.

    Parameters
    ----------
    seasons
        Seasons (year) to retrieve.
    overwrite
        When ``True`` we refresh the cache even if a file already exists.

    Returns
    -------
    list[Path]
        Paths written to disk.
    """
    written_paths: list[Path] = []
    espn_to_gsis, _ = _load_player_mappings()

    for season in seasons:
        cache_path = INJURY_CACHE_DIR / f"injury_{season}.parquet"
        if cache_path.exists() and not overwrite:
            logger.info("Injury cache already present for %s at %s", season, cache_path)
            written_paths.append(cache_path)
            continue

        schedule_lookup = _build_schedule_lookup(season)
        teams = _team_list_from_schedule(import_schedules([season]))
        logger.info("Collecting ESPN injuries for %s (%d teams)", season, len(teams))

        records: list[dict[str, object]] = []
        for team in teams:
            try:
                refs = list(_iter_team_injury_refs(_team_to_slug(team)))
            except Exception as exc:  # pragma: no cover - network failure fallback
                logger.warning("Failed to list injuries for team %s: %s", team, exc)
                continue

            if not refs:
                continue

            for ref in refs:
                try:
                    detail = requests.get(ref, timeout=15).json()
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to fetch injury detail %s: %s", ref, exc)
                    continue

                athlete_id = _parse_athlete_id(ref)
                if not athlete_id:
                    continue

                reported_at = detail.get("date")
                injury_dt = (
                    pd.to_datetime(reported_at, utc=True).to_pydatetime().date()
                    if reported_at
                    else dt.date.today()
                )

                week, game_type, game_date, game_id, game_start = _assign_week(
                    team, injury_dt, schedule_lookup
                )
                type_info = detail.get("type") or {}
                source_info = detail.get("source") or {}

                record = InjuryRecord(
                    season=season,
                    team=team,
                    week=week,
                    game_type=game_type,
                    game_date=game_date,
                    game_id=game_id,
                    game_start_utc=game_start,
                    espn_id=athlete_id,
                    player_id=espn_to_gsis.get(athlete_id),
                    status=detail.get("status"),
                    status_abbr=type_info.get("abbreviation"),
                    status_description=type_info.get("description"),
                    injury_type_id=type_info.get("id"),
                    short_comment=detail.get("shortComment"),
                    long_comment=detail.get("longComment"),
                    source_description=source_info.get("description"),
                    source_state=source_info.get("state"),
                    reported_at=(
                        pd.to_datetime(reported_at, utc=True).to_pydatetime()
                        if reported_at
                        else None
                    ),
                    injury_id=_parse_injury_id(ref),
                )
                records.append(record.to_dict())

        if not records:
            logger.warning("No ESPN injury records collected for season %s", season)
            continue

        df = pd.DataFrame.from_records(records)
        df["game_start_utc"] = pd.to_datetime(df.get("game_start_utc"), utc=True, errors="coerce")
        df["practice_status"] = None
        df["report_primary_injury"] = None
        df["practice_primary_injury"] = None

        df = df.sort_values(["team", "week", "reported_at"]).reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, compression="zstd", index=False)
        logger.info("Wrote ESPN injuries cache for %s → %s (%d rows)", season, cache_path, len(df))

        try:
            collected_at = dt.datetime.now(dt.timezone.utc)
            grouped = df.dropna(subset=["game_id"]).groupby(["game_id", "team"], dropna=False)
            arrival_rows = []
            for (game_id_val, team_val), group in grouped:
                if pd.isna(game_id_val):
                    continue
                season_val = int(group["season"].mode().iloc[0]) if not group["season"].empty else season
                week_val = (
                    int(group["week"].mode().iloc[0])
                    if "week" in group.columns and not group["week"].empty
                    else None
                )
                game_start = group.get("game_start_utc")
                game_start_val = (
                    pd.to_datetime(game_start.dropna().iloc[-1], utc=True)
                    if game_start is not None and not game_start.dropna().empty
                    else pd.NaT
                )
                reported_at = group.get("reported_at")
                latest_report = (
                    pd.to_datetime(reported_at.dropna().iloc[-1], utc=True)
                    if reported_at is not None and not reported_at.dropna().empty
                    else pd.NaT
                )
                earliest_report = (
                    pd.to_datetime(reported_at.dropna().iloc[0], utc=True)
                    if reported_at is not None and not reported_at.dropna().empty
                    else pd.NaT
                )
                arrival_rows.append(
                    {
                        "season": season_val,
                        "week": week_val,
                        "game_id": str(game_id_val),
                        "team": str(team_val) if isinstance(team_val, str) else None,
                        "game_start_ts": None if pd.isna(game_start_val) else game_start_val.to_pydatetime(),
                        "feed_timestamp": None if pd.isna(latest_report) else latest_report.to_pydatetime(),
                        "feed_timestamp_min": None if pd.isna(earliest_report) else earliest_report.to_pydatetime(),
                        "collected_at": collected_at,
                        "metadata": {
                            "season_type": group.get("game_type", pd.Series(dtype=str)).dropna().unique().tolist(),
                            "rows": int(len(group)),
                        },
                    }
                )
            if arrival_rows:
                log_feed_arrivals("injuries", arrival_rows, snapshot_label="espn")
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.warning("Failed to log injury arrival metrics for season %s: %s", season, exc)

        written_paths.append(cache_path)

    return written_paths


__all__ = ["collect_espn_injuries"]

