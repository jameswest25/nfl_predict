"""Collect NFL depth chart data from ESPN's public API.

This collector fetches real depth chart rankings from ESPN's depthcharts
endpoint and writes them to roster_snapshots for use in feature engineering.
The key field is `depth_chart_order` (rank), which distinguishes starters
from backups.
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
from nfl_data_py import import_players, import_schedules

from utils.general.paths import ROSTER_SNAPSHOT_DIR

logger = logging.getLogger(__name__)

# ESPN team abbreviation -> ESPN team ID mapping
ESPN_TEAM_IDS = {
    'ATL': '1', 'BUF': '2', 'CHI': '3', 'CIN': '4', 'CLE': '5', 'DAL': '6',
    'DEN': '7', 'DET': '8', 'GB': '9', 'TEN': '10', 'IND': '11', 'KC': '12',
    'LV': '13', 'LAR': '14', 'MIA': '15', 'MIN': '16', 'NE': '17', 'NO': '18',
    'NYG': '19', 'NYJ': '20', 'PHI': '21', 'ARI': '22', 'PIT': '23', 'LAC': '24',
    'SF': '25', 'SEA': '26', 'TB': '27', 'WSH': '28', 'CAR': '29', 'JAX': '30',
    'BAL': '33', 'HOU': '34',
    # Aliases
    'LA': '14', 'WAS': '28', 'OAK': '13',
}

DEPTH_CHART_URL = (
    "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
    "seasons/{season}/teams/{team_id}/depthcharts?lang=en&region=us"
)


@dataclass(frozen=True)
class DepthChartRecord:
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
    depth_chart_position: str | None
    depth_chart_name: str | None
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
            "depth_chart_position": self.depth_chart_position,
            "depth_chart_name": self.depth_chart_name,
            "source": self.source,
        }


def _load_player_mappings() -> tuple[dict[str, str], dict[str, str]]:
    """Load ESPN ID -> GSIS ID and ESPN ID -> name mappings."""
    players = import_players()
    players = players.dropna(subset=["espn_id"])
    players["espn_id"] = players["espn_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    espn_to_gsis = dict(zip(players["espn_id"], players["gsis_id"]))
    espn_to_name = dict(zip(players["espn_id"], players["display_name"]))
    return espn_to_gsis, espn_to_name


def _build_schedule_lookup(season: int) -> dict[str, pd.DataFrame]:
    """Build a lookup of team -> schedule DataFrame for week assignment."""
    schedule = import_schedules([season])
    if schedule.empty:
        return {}
    
    schedule = schedule.rename(columns={"gameday": "game_date"})
    schedule["game_date"] = pd.to_datetime(schedule["game_date"]).dt.date
    if "start_time_utc" in schedule.columns:
        schedule["start_time_utc"] = pd.to_datetime(schedule["start_time_utc"], utc=True, errors="coerce")
    else:
        schedule["start_time_utc"] = pd.NaT
    
    frames = []
    for col in ("home_team", "away_team"):
        tmp = schedule[["game_id", "season", "week", "game_type", "game_date", "start_time_utc", col]].rename(
            columns={col: "team"}
        )
        frames.append(tmp)
    schedule_long = pd.concat(frames, ignore_index=True).sort_values(["team", "game_date", "week"])
    
    return {team: group.reset_index(drop=True) for team, group in schedule_long.groupby("team")}


def _assign_week(
    team: str,
    ref_date: dt.date,
    schedule_lookup: dict[str, pd.DataFrame],
) -> tuple[int, str, dt.date, str | None, dt.datetime | None]:
    """Assign the current week based on the reference date."""
    team_sched = schedule_lookup.get(team)
    if team_sched is None or team_sched.empty:
        return 0, "REG", ref_date, None, None
    
    future_mask = team_sched["game_date"] >= ref_date
    if future_mask.any():
        row = team_sched.loc[future_mask].iloc[0]
    else:
        row = team_sched.iloc[-1]
    
    game_id_val = row.get("game_id")
    if pd.isna(game_id_val):
        game_id_val = None
    else:
        game_id_val = str(game_id_val)
    
    start_ts = row.get("start_time_utc")
    if isinstance(start_ts, pd.Timestamp) and not pd.isna(start_ts):
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        start_ts = start_ts.to_pydatetime()
    else:
        start_ts = None
    
    return int(row["week"]), str(row["game_type"]), row["game_date"], game_id_val, start_ts


def _fetch_depth_chart(team_id: str, season: int) -> list[dict]:
    """Fetch depth chart data for a single team."""
    url = DEPTH_CHART_URL.format(season=season, team_id=team_id)
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        logger.warning("Failed to fetch depth chart for team %s: HTTP %s", team_id, resp.status_code)
        return []
    return resp.json().get("items", [])


def _extract_espn_id(ref: str) -> str | None:
    """Extract ESPN athlete ID from a reference URL."""
    try:
        return ref.split("/athletes/")[1].split("?")[0]
    except (IndexError, AttributeError):
        return None


def collect_espn_depth_charts(
    seasons: Iterable[int],
    *,
    snapshot_ts: dt.datetime | None = None,
) -> list[Path]:
    """Collect depth chart data for the supplied seasons.
    
    Parameters
    ----------
    seasons
        Seasons (years) to collect depth charts for.
    snapshot_ts
        Optional timestamp for the snapshot. Defaults to now.
    
    Returns
    -------
    list[Path]
        Paths to written parquet files.
    """
    snapshot_ts = snapshot_ts or dt.datetime.now(dt.timezone.utc)
    espn_to_gsis, espn_to_name = _load_player_mappings()
    written_paths: list[Path] = []
    
    for season in sorted(set(int(s) for s in seasons)):
        logger.info("Collecting ESPN depth charts for season %s", season)
        
        schedule_lookup = _build_schedule_lookup(season)
        if not schedule_lookup:
            logger.warning("No schedule data for season %s", season)
            continue
        
        records: list[dict[str, object]] = []
        
        for team_abbrev, team_id in ESPN_TEAM_IDS.items():
            # Skip aliases
            if team_abbrev in ('LA', 'WAS', 'OAK'):
                continue
            
            logger.debug("Fetching depth chart for %s (ID: %s)", team_abbrev, team_id)
            
            try:
                depth_charts = _fetch_depth_chart(team_id, season)
            except Exception as exc:
                logger.warning("Error fetching depth chart for %s: %s", team_abbrev, exc)
                continue
            
            if not depth_charts:
                continue
            
            # Assign week based on current date
            ref_date = snapshot_ts.date()
            week, game_type, game_date, game_id, game_start = _assign_week(
                team_abbrev, ref_date, schedule_lookup
            )
            
            for dc in depth_charts:
                dc_name = dc.get("name", "Unknown")
                positions = dc.get("positions", {})
                
                for pos_key, pos_data in positions.items():
                    pos_abbrev = pos_data.get("position", {}).get("abbreviation", pos_key.upper())
                    athletes = pos_data.get("athletes", [])
                    
                    for ath in athletes:
                        rank = ath.get("rank")
                        ath_ref = ath.get("athlete", {}).get("$ref", "")
                        espn_id = _extract_espn_id(ath_ref)
                        
                        if not espn_id:
                            continue
                        
                        # Try to fetch athlete name if not in mapping
                        player_name = espn_to_name.get(espn_id)
                        if not player_name:
                            try:
                                ath_resp = requests.get(ath_ref, timeout=10)
                                if ath_resp.status_code == 200:
                                    player_name = ath_resp.json().get("displayName")
                            except Exception:
                                pass
                        
                        record = DepthChartRecord(
                            snapshot_ts=snapshot_ts,
                            season=season,
                            week=week,
                            game_type=game_type,
                            game_date=game_date,
                            game_id=game_id,
                            game_start_utc=game_start,
                            team=team_abbrev,
                            espn_id=espn_id,
                            player_id=espn_to_gsis.get(espn_id),
                            player_name=player_name,
                            position=pos_abbrev,
                            depth_chart_order=rank,
                            depth_chart_position=pos_key.upper(),
                            depth_chart_name=dc_name,
                            source="espn_depthchart",
                        )
                        records.append(record.to_dict())
        
        if not records:
            logger.warning("No depth chart records collected for season %s", season)
            continue
        
        logger.info("Collected %d depth chart records for season %s", len(records), season)
        
        # Convert to polars and write
        roster_df = pl.from_pandas(pd.DataFrame.from_records(records))
        roster_df = roster_df.with_columns([
            pl.col("snapshot_ts").cast(pl.Datetime("ms", "UTC")),
            pl.col("game_start_utc").cast(pl.Datetime("ms", "UTC")),
            pl.col("game_id").cast(pl.Utf8),
            pl.col("team").str.strip_chars().str.to_uppercase(),
            pl.col("player_id").cast(pl.Utf8),
            pl.col("espn_id").cast(pl.Utf8),
            pl.col("position").cast(pl.Utf8),
            pl.col("depth_chart_order").cast(pl.Int32),
        ])
        
        # Group by season/week and write
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
                    subset=["team", "espn_id", "depth_chart_position"], keep="last"
                )
            else:
                combined = part
            
            combined.write_parquet(out_path, compression="zstd")
            written_paths.append(out_path)
            logger.info("Wrote depth chart snapshot: %s (%d rows)", out_path, combined.height)
    
    return written_paths


if __name__ == "__main__":
    import argparse
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Collect ESPN depth charts")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025], help="Seasons to collect")
    args = parser.parse_args()
    
    paths = collect_espn_depth_charts(args.seasons)
    print(f"\nWrote {len(paths)} files:")
    for p in paths:
        print(f"  {p}")

