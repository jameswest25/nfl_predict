"""Collect time-aware roster snapshots from ESPN.

The standard weekly roster feed in ``nfl_data_py`` only exposes one record per
week and carries no timestamp, which makes it impossible to know whether a
player's status was known at the model's decision cutoff.  This collector hits
ESPN's public team roster endpoints, records the roster at the moment of
collection, and stores those snapshots under
``data/processed/roster_snapshots/season=YYYY/week=WW`` for downstream
as-of joins.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import polars as pl
import requests
from nfl_data_py import import_schedules

from utils.collect.arrival_log import log_feed_arrivals
from utils.collect import espn_injuries as espn_common
from utils.general.paths import ROSTER_SNAPSHOT_DIR

logger = logging.getLogger(__name__)

ESPN_ROSTER_BASE = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/"
    "{team}/athletes?lang=en&region=us"
)


@dataclass(frozen=True)
class RosterSnapshotRecord:
    snapshot_ts: dt.datetime
    season: int
    week: int
    game_type: str
    game_date: dt.date
    game_id: str | None
    game_start_utc: dt.datetime | None
    team: str
    espn_id: str
    player_id: str | None
    player_name: str | None
    position: str | None
    depth_chart_order: int | None
    status: str | None
    status_abbr: str | None
    status_detail: str | None
    jersey: str | None
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_ts": self.snapshot_ts,
            "season": self.season,
            "week": self.week,
            "game_type": self.game_type,
            "game_date": self.game_date,
            "game_id": self.game_id,
            "game_start_utc": self.game_start_utc,
            "team": self.team,
            "espn_id": self.espn_id,
            "player_id": self.player_id,
            "player_name": self.player_name,
            "position": self.position,
            "depth_chart_order": self.depth_chart_order,
            "status": self.status,
            "status_abbr": self.status_abbr,
            "status_detail": self.status_detail,
            "jersey": self.jersey,
            "source": self.source,
        }


def _iter_team_roster_refs(team_slug: str) -> Iterable[str]:
    page = 1
    while True:
        url = f"{ESPN_ROSTER_BASE.format(team=team_slug)}&page={page}"
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
        page_count = payload.get("pageCount", page)
        if page >= page_count:
            break
        page += 1


def collect_espn_roster_snapshots(
    seasons: Iterable[int],
    *,
    snapshot_ts: dt.datetime | None = None,
) -> list[Path]:
    """Collect roster snapshots for the supplied seasons."""
    snapshot_ts = snapshot_ts or dt.datetime.now(dt.timezone.utc)
    espn_to_gsis, espn_to_name = espn_common._load_player_mappings()  # type: ignore[attr-defined]
    written_paths: list[Path] = []

    for season in sorted(set(int(s) for s in seasons)):
        schedule_lookup = espn_common._build_schedule_lookup(season)  # type: ignore[attr-defined]
        schedule = import_schedules([season])
        if schedule.empty:
            logger.warning("No schedule available for season %s; skipping roster snapshot.", season)
            continue
        teams = espn_common._team_list_from_schedule(schedule)  # type: ignore[attr-defined]
        if not teams:
            logger.warning("No teams resolved for season %s roster snapshot.", season)
            continue

        logger.info("Collecting ESPN roster snapshot for %s (%d teams)", season, len(teams))
        records: list[dict[str, object]] = []

        for team in teams:
            try:
                refs = list(_iter_team_roster_refs(espn_common._team_to_slug(team)))  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("Failed to fetch roster list for %s: %s", team, exc)
                continue

            for ref in refs:
                try:
                    detail = requests.get(ref, timeout=15).json()
                except Exception as exc:
                    logger.warning("Failed roster detail %s: %s", ref, exc)
                    continue

                athlete_id = espn_common._parse_athlete_id(ref)  # type: ignore[attr-defined]
                if not athlete_id:
                    continue

                position_info = detail.get("position") or {}
                status_info = detail.get("status") or {}
                status_type = status_info.get("type") or {}

                injury_date = snapshot_ts.date()
                week, game_type, game_date, game_id, game_start = espn_common._assign_week(  # type: ignore[attr-defined]
                    team, injury_date, schedule_lookup
                )

                record = RosterSnapshotRecord(
                    snapshot_ts=snapshot_ts,
                    season=season,
                    week=week,
                    game_type=game_type,
                    game_date=game_date,
                    game_id=game_id,
                    game_start_utc=game_start,
                    team=team,
                    espn_id=athlete_id,
                    player_id=espn_to_gsis.get(athlete_id),
                    player_name=espn_to_name.get(athlete_id),
                    position=position_info.get("abbreviation") or position_info.get("name"),
                    depth_chart_order=detail.get("depthChartOrder"),
                    status=status_type.get("name") or status_type.get("description"),
                    status_abbr=status_type.get("abbreviation"),
                    status_detail=status_info.get("displayValue") or status_type.get("detail"),
                    jersey=str(detail.get("jersey")) if detail.get("jersey") is not None else None,
                    source="espn",
                )
                records.append(record.to_dict())

        if not records:
            logger.warning("No roster snapshot records collected for season %s", season)
            continue

        roster_df = pl.from_pandas(pd.DataFrame.from_records(records))
        roster_df = roster_df.with_columns(
            [
                pl.col("snapshot_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("game_start_utc").cast(pl.Datetime("ms", "UTC")),
                pl.col("game_id").cast(pl.Utf8),
                pl.col("team").str.strip_chars().str.to_uppercase(),
                pl.col("player_id").cast(pl.Utf8),
                pl.col("espn_id").cast(pl.Utf8),
                pl.col("position").cast(pl.Utf8),
                pl.col("status").cast(pl.Utf8),
                pl.col("status_abbr").cast(pl.Utf8),
                pl.col("status_detail").cast(pl.Utf8),
            ]
        )

        for (season_val, week_val), part in roster_df.group_by(["season", "week"], maintain_order=True):
            season_int = int(season_val) if season_val is not None else season
            week_int = int(week_val) if week_val is not None else 0
            out_dir = ROSTER_SNAPSHOT_DIR / f"season={season_int}" / f"week={week_int:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "part.parquet"
            if out_path.exists():
                existing = pl.read_parquet(out_path)
                combined = pl.concat([existing, part], how="diagonal_relaxed")
                combined = combined.unique(
                    subset=["snapshot_ts", "team", "espn_id", "game_id"], keep="last"
                )
            else:
                combined = part
            combined.write_parquet(out_path, compression="zstd")
            written_paths.append(out_path)

        try:
            rows = []
            for (team, week), group in roster_df.group_by(["team", "week"], maintain_order=True):
                rows.append(
                    {
                        "season": season,
                        "week": int(week) if week is not None else None,
                        "game_id": group.get_column("game_id").drop_nulls().first() if "game_id" in group.columns else None,
                        "team": team,
                        "game_start_ts": group.get_column("game_start_utc").drop_nulls().first() if "game_start_utc" in group.columns else None,
                        "feed_timestamp": snapshot_ts,
                        "collected_at": snapshot_ts,
                        "metadata": {
                            "players": int(group.height),
                            "statuses": sorted(
                                group.get_column("status")
                                .cast(pl.Utf8)
                                .drop_nulls()
                                .unique()
                                .to_list()
                            ),
                        },
                    }
                )
            if rows:
                log_feed_arrivals("roster_snapshots", rows, snapshot_label="espn")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to log roster snapshot arrivals: %s", exc)

    return written_paths


__all__ = ["collect_espn_roster_snapshots"]

