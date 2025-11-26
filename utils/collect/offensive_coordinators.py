from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/processed/offensive_coordinators.parquet")
MANUAL_CONFIG = Path("config/offensive_coordinators.yaml")


def _load_manual_overrides() -> pd.DataFrame:
    if not MANUAL_CONFIG.exists():
        logger.info("Manual offensive coordinator config not found at %s", MANUAL_CONFIG)
        return pd.DataFrame(columns=["season", "team", "offensive_coordinator"])

    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        logger.warning("Unable to import yaml while reading %s (%s)", MANUAL_CONFIG, exc)
        return pd.DataFrame(columns=["season", "team", "offensive_coordinator"])

    data = yaml.safe_load(MANUAL_CONFIG.read_text())
    if not isinstance(data, dict):
        logger.warning("Manual coordinator config malformed (expected dict) – ignoring.")
        return pd.DataFrame(columns=["season", "team", "offensive_coordinator"])

    rows: list[dict[str, object]] = []
    for season_str, team_map in data.items():
        try:
            season = int(season_str)
        except (TypeError, ValueError):
            logger.warning("Skipping manual coordinator entry with invalid season: %s", season_str)
            continue
        if not isinstance(team_map, dict):
            continue
        for team, coordinator in team_map.items():
            if not coordinator:
                continue
            rows.append(
                {
                    "season": season,
                    "team": str(team).upper(),
                    "offensive_coordinator": str(coordinator).strip(),
                    "source": "manual",
                }
            )
    return pd.DataFrame(rows, columns=["season", "team", "offensive_coordinator", "source"])


def _derive_from_schedules(seasons: Iterable[int]) -> pd.DataFrame:
    try:
        import nfl_data_py as ndp  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.error("nfl_data_py unavailable; unable to derive coordinator map (%s)", exc)
        return pd.DataFrame(columns=["season", "team", "offensive_coordinator"])

    seasons = sorted({int(season) for season in seasons})
    if not seasons:
        return pd.DataFrame(columns=["season", "team", "offensive_coordinator"])

    schedules = ndp.import_schedules(seasons)
    if schedules.empty:
        logger.warning("import_schedules returned empty frame for seasons %s", seasons)
        return pd.DataFrame(columns=["season", "team", "offensive_coordinator"])

    # Combine home and away coach assignments so each team-season combination has entries.
    home = schedules[["season", "home_team", "home_coach"]].rename(
        columns={"home_team": "team", "home_coach": "coach"}
    )
    away = schedules[["season", "away_team", "away_coach"]].rename(
        columns={"away_team": "team", "away_coach": "coach"}
    )
    combined = pd.concat([home, away], ignore_index=True)
    combined = combined.dropna(subset=["team", "coach"])
    if combined.empty:
        return pd.DataFrame(columns=["season", "team", "offensive_coordinator"])

    combined["team"] = combined["team"].astype(str).str.upper()
    combined["coach"] = combined["coach"].astype(str).str.strip()

    # Determine the most frequent coach per team/season (assume primary play-caller).
    counts = (
        combined.groupby(["season", "team", "coach"])
        .size()
        .reset_index(name="games")
    )
    counts.sort_values(["season", "team", "games"], ascending=[True, True, False], inplace=True)
    primary = (
        counts.groupby(["season", "team"], as_index=False)
        .first()
        .rename(columns={"coach": "offensive_coordinator"})
    )
    primary["source"] = "head_coach_fallback"
    return primary[["season", "team", "offensive_coordinator", "source"]]


def build_offensive_coordinator_map(
    seasons: Iterable[int],
    *,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    """
    Build an offensive coordinator (play-caller) mapping for the requested seasons.

    Manual overrides from ``config/offensive_coordinators.yaml`` are merged on top of
    schedule-derived head coach fallbacks.
    """
    seasons = sorted({int(season) for season in seasons})
    if not seasons:
        raise ValueError("No seasons provided for coordinator map.")

    fallback = _derive_from_schedules(seasons)
    manual = _load_manual_overrides()

    if manual.empty and fallback.empty:
        logger.warning("Unable to derive offensive coordinator data for seasons %s", seasons)
        return output_path

    if manual.empty:
        merged = fallback
    else:
        merged = fallback.merge(
            manual,
            how="outer",
            on=["season", "team"],
            suffixes=("_fallback", "_manual"),
        )
        merged["offensive_coordinator"] = merged["offensive_coordinator_manual"].combine_first(
            merged["offensive_coordinator_fallback"]
        )
        merged["source"] = merged["source_manual"].combine_first(merged["source_fallback"])
        merged = merged[["season", "team", "offensive_coordinator", "source"]]

    merged = merged.dropna(subset=["offensive_coordinator"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info(
        "Offensive coordinator map written → %s (%d rows, seasons %s)",
        output_path,
        len(merged),
        seasons,
    )
    return output_path


__all__ = ["build_offensive_coordinator_map", "DEFAULT_OUTPUT", "MANUAL_CONFIG"]

