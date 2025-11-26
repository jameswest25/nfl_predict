from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from nfl_data_py import import_players

from utils.general.paths import PROJ_ROOT

logger = logging.getLogger(__name__)

TRANSACTION_SOURCE = "https://raw.githubusercontent.com/JaseZiv/NFL-Injuries/main/data/nfl_injuries.json"
TRANSACTION_CACHE_DIR = PROJ_ROOT / "cache" / "feature" / "injury_transactions"
TRANSACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEAM_NICKNAME_TO_ABBR = {
    "49ERS": "SF",
    "BEARS": "CHI",
    "BENGALS": "CIN",
    "BILLS": "BUF",
    "BRONCOS": "DEN",
    "BROWNS": "CLE",
    "BUCCANEERS": "TB",
    "CARDINALS": "ARI",
    "CHARGERS": "LAC",
    "CHIEFS": "KC",
    "COLTS": "IND",
    "COMMANDERS": "WAS",
    "COWBOYS": "DAL",
    "DOLPHINS": "MIA",
    "EAGLES": "PHI",
    "FALCONS": "ATL",
    "GIANTS": "NYG",
    "JAGUARS": "JAX",
    "JETS": "NYJ",
    "LIONS": "DET",
    "PACKERS": "GB",
    "PANTHERS": "CAR",
    "PATRIOTS": "NE",
    "RAIDERS": "LV",
    "RAMS": "LAR",
    "RAVENS": "BAL",
    "SAINTS": "NO",
    "SEAHAWKS": "SEA",
    "STEELERS": "PIT",
    "TEXANS": "HOU",
    "TITANS": "TEN",
    "VIKINGS": "MIN",
    # Historical nicknames
    "REDSKINS": "WAS",
    "REDSKINS*": "WAS",
    "OILERS": "TEN",
    "RAIDERS (OAKLAND)": "OAK",
    "RAIDERS (LA)": "LA",
    "BROWNS (OLD)": "CLE",
    "FOOTBALL TEAM": "WAS",
    "BALTIMORE COLTS": "IND",
    "ST. LOUIS RAMS": "STL",
    "SAN DIEGO CHARGERS": "SD",
}


def _normalize_name(raw: str | None) -> list[str]:
    if not raw:
        return []
    # split alternate names separated by "/" or ","
    raw_parts = re.split(r"[\/,]", raw)
    cleaned: list[str] = []
    for part in raw_parts:
        part = re.sub(r"\(.*?\)", "", part)  # drop paren aliases
        part = part.replace("'", "").strip()
        if not part:
            continue
        cleaned.append(part)
    return cleaned or []


def _normalization_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return key


def _load_player_lookup() -> dict[str, str]:
    players = import_players()
    players = players.dropna(subset=["display_name", "gsis_id"])
    players["norm"] = players["display_name"].apply(_normalization_key)
    return dict(zip(players["norm"], players["gsis_id"]))


def _map_team(team: str | None) -> str | None:
    if not team:
        return None
    team = team.strip().upper()
    return TEAM_NICKNAME_TO_ABBR.get(team, team[:3])


def collect_injury_transactions(
    seasons: Iterable[int] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Fetch the nightly prosportstransactions injury JSON and persist season-specific parquet slices.
    """
    try:
        resp = requests.get(TRANSACTION_SOURCE, timeout=30)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("Failed to download injury transactions: %s", exc)
        return []

    try:
        records = resp.json()
    except Exception as exc:  # pragma: no cover
        logger.error("Invalid JSON in injury transactions feed: %s", exc)
        return []

    if not isinstance(records, list):
        logger.error("Unexpected injury transaction payload type: %s", type(records))
        return []

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("Injury transaction feed returned zero rows.")
        return []

    df = df.rename(
        columns={
            "Date": "date",
            "Team": "team",
            "Acquired": "acquired",
            "Relinquished": "player_raw",
            "Notes": "notes",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["season"] = df["date"].dt.year.astype(int)
    if seasons:
        seasons = list({int(s) for s in seasons})
        df = df[df["season"].isin(seasons)]

    df = df[df["season"] >= 1999]  # filter to modern era
    if df.empty:
        logger.warning("No injury transactions remain after filtering seasons.")
        return []

    player_lookup = _load_player_lookup()

    mapped_player_ids: list[str | None] = []
    mapped_names: list[str] = []
    for raw in df["player_raw"].fillna(""):
        candidates = _normalize_name(raw)
        mapped_id = None
        clean_name = raw.strip()
        for candidate in candidates:
            norm = _normalization_key(candidate)
            mapped_id = player_lookup.get(norm)
            if mapped_id:
                clean_name = candidate
                break
        mapped_player_ids.append(mapped_id)
        mapped_names.append(clean_name)

    df["player_id"] = mapped_player_ids
    df["player_name"] = mapped_names
    df["team_abbr"] = df["team"].apply(_map_team)
    df["transaction_date"] = df["date"].dt.tz_localize("UTC")

    cols = [
        "season",
        "transaction_date",
        "team_abbr",
        "player_id",
        "player_name",
        "notes",
    ]
    df = df[cols]

    written: list[Path] = []
    for season, grp in df.groupby("season"):
        cache_path = TRANSACTION_CACHE_DIR / f"transactions_{season}.parquet"
        if cache_path.exists() and not overwrite:
            logger.info("Transaction cache already present for %s", season)
            written.append(cache_path)
            continue
        grp.sort_values("transaction_date", inplace=True)
        grp.to_parquet(cache_path, compression="zstd", index=False)
        written.append(cache_path)
        logger.info(
            "Wrote injury transaction cache for %s → %s (%d rows)",
            season,
            cache_path,
            len(grp),
        )

    return written


__all__ = ["collect_injury_transactions"]

