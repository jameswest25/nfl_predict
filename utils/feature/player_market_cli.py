import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import polars as pl

from utils.general.paths import PLAYER_MARKET_PROCESSED_DIR, PLAYER_ODDS_RAW_DIR, FEATURE_CACHE_DIR
from utils.feature.odds import SNAPSHOT_SEQUENCE, SNAPSHOT_SUFFIXES, DELTA_SEQUENCE, _american_to_prob
from utils.general.constants import TEAM_ABBR_TO_NAME, TEAM_NAME_TO_ABBR

logger = logging.getLogger(__name__)

ROSTER_CACHE_TEMPLATE = FEATURE_CACHE_DIR / "rosters" / "roster_{season}.parquet"
NAME_NORMALIZER = re.compile(r"[^A-Z0-9]")

GROUP_KEYS: Tuple[str, ...] = (
    "season",
    "week",
    "game_id",
    "event_id",
    "player_id",
    "selection_name",
    "home_team",
    "away_team",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate player prop markets into model features.")
    parser.add_argument("--seasons", type=str, default="", help="Comma-separated list of seasons to process")
    parser.add_argument("--weeks", type=str, default="", help="Optional comma-separated list of weeks (1-23)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files")
    return parser.parse_args()


def _discover_seasons() -> List[int]:
    if not PLAYER_ODDS_RAW_DIR.exists():
        return []
    seasons: List[int] = []
    for path in sorted(PLAYER_ODDS_RAW_DIR.glob("season=*")):
        try:
            seasons.append(int(path.name.split("=")[1]))
        except Exception:
            continue
    return seasons


def _discover_weeks(season: int) -> List[int]:
    base_dir = PLAYER_ODDS_RAW_DIR / f"season={season}"
    if not base_dir.exists():
        return []
    weeks: List[int] = []
    for path in sorted(base_dir.glob("week=*")):
        try:
            weeks.append(int(path.name.split("=")[1]))
        except Exception:
            continue
    return weeks


def _normalize_player_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    cleaned = NAME_NORMALIZER.sub("", str(name).upper())
    return cleaned or None


def _resolve_team_abbr(raw_team: Optional[str], home_team: Optional[str], away_team: Optional[str]) -> Optional[str]:
    if raw_team:
        team = raw_team.strip().upper()
        if team in TEAM_ABBR_TO_NAME:
            return team
        if team in TEAM_NAME_TO_ABBR:
            return TEAM_NAME_TO_ABBR[team]
        for abbr, name in TEAM_ABBR_TO_NAME.items():
            if team == name.upper():
                return abbr
            if team in name.upper():
                return abbr
    for candidate in (home_team, away_team):
        if isinstance(candidate, str) and candidate:
            cand = candidate.strip().upper()
            if cand in TEAM_ABBR_TO_NAME:
                return cand
    return None


def _build_roster_lookup(seasons: Iterable[int]) -> Dict[Tuple[int, str, str], str]:
    lookup: Dict[Tuple[int, str, str], str] = {}
    for season in seasons:
        roster_path = Path(str(ROSTER_CACHE_TEMPLATE).format(season=season))
        if not roster_path.exists():
            logger.warning("Roster cache missing for season %s (expected %s)", season, roster_path)
            continue
        roster = pl.read_parquet(roster_path, columns=["season", "team", "player_id", "player_name"])
        if roster.is_empty():
            continue
        roster = roster.with_columns(
            [
                pl.col("season").cast(pl.Int32),
                pl.col("team").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
                pl.col("player_id").cast(pl.Utf8),
                pl.col("player_name").cast(pl.Utf8),
            ]
        )
        for row in roster.iter_rows(named=True):
            norm_name = _normalize_player_name(row.get("player_name"))
            team = row.get("team")
            player_id = row.get("player_id")
            if not norm_name or not player_id or not team:
                continue
            lookup[(int(row["season"]), team, norm_name)] = player_id
    return lookup


def _assign_player_id_mapper(lookup: Dict[Tuple[int, str, str], str]):
    def _mapper(row: Dict[str, Any]) -> Optional[str]:
        season = int(row.get("season")) if row.get("season") is not None else None
        if season is None:
            return None
        name_key = row.get("name_key")
        if not name_key:
            return None
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        team_hint = row.get("selection_team")
        team_candidates: List[str] = []
        resolved_hint = _resolve_team_abbr(team_hint, home_team, away_team) if team_hint else None
        if resolved_hint:
            team_candidates.append(resolved_hint)
        for candidate in (home_team, away_team):
            if isinstance(candidate, str):
                team_candidates.append(candidate.strip().upper())
        seen: set[str] = set()
        for team in team_candidates:
            if not team:
                continue
            team_up = team.upper()
            if team_up in seen:
                continue
            seen.add(team_up)
            key = (season, team_up, name_key)
            if key in lookup:
                return lookup[key]
        fallback_key = next((pid for (s, _, nk), pid in lookup.items() if s == season and nk == name_key), None)
        return fallback_key

    return _mapper


def _load_raw_player_odds(season: int, week: int) -> Optional[pl.DataFrame]:
    week_dir = PLAYER_ODDS_RAW_DIR / f"season={season}" / f"week={week:02d}"
    if not week_dir.exists():
        return None
    matches = list(week_dir.glob("book=*/part.parquet"))
    if not matches:
        return None
    df = pl.scan_parquet(str(week_dir / "book=*" / "part.parquet")).collect()
    if df.is_empty():
        return None
    df = df.with_columns(
        [
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("game_id").cast(pl.Utf8),
            pl.col("event_id").cast(pl.Utf8),
            pl.col("bookmaker_key").cast(pl.Utf8),
            pl.col("market_key").cast(pl.Utf8),
            pl.col("selection_name").cast(pl.Utf8),
            pl.col("selection_team").cast(pl.Utf8),
            pl.col("home_team").cast(pl.Utf8),
            pl.col("away_team").cast(pl.Utf8),
            pl.col("outcome_type").cast(pl.Utf8),
            pl.col("snapshot_label").cast(pl.Utf8),
            pl.col("price").cast(pl.Float64),
            pl.col("point").cast(pl.Float64),
            pl.col("snapshot_ts").cast(pl.Utf8),
        ]
    ).with_columns(
        [
            pl.col("home_team").str.strip_chars().str.to_uppercase(),
            pl.col("away_team").str.strip_chars().str.to_uppercase(),
            pl.col("selection_team").str.strip_chars().str.to_uppercase(),
        ]
    )
    return df


def _aggregate_anytime_features(df_any: pl.DataFrame) -> pl.DataFrame:
    if df_any.is_empty():
        return df_any
    metrics: Optional[pl.DataFrame] = None
    for label in SNAPSHOT_SEQUENCE:
        suffix = SNAPSHOT_SUFFIXES[label]
        df_label = df_any.filter(pl.col("snapshot_label") == label)
        if df_label.is_empty():
            continue
        alias_prob = f"market_anytime_td_prob{suffix}"
        alias_consensus = f"market_anytime_td_consensus{suffix}"
        alias_spread = f"market_anytime_td_spread{suffix}"
        alias_books = f"market_anytime_td_book_count{suffix}"
        agg = df_label.group_by(GROUP_KEYS).agg(
            [
                pl.col("implied_prob").mean().alias(alias_prob),
                pl.col("implied_prob").median().alias(alias_consensus),
                (pl.col("implied_prob").max() - pl.col("implied_prob").min()).alias(alias_spread),
                pl.len().alias(alias_books),
            ]
        )
        if metrics is None:
            metrics = agg
        else:
            metrics = metrics.join(agg, on=GROUP_KEYS, how="full", suffix="_dup")
            drop_cols = [f"{key}_dup" for key in GROUP_KEYS if f"{key}_dup" in metrics.columns]
            if drop_cols:
                metrics = metrics.drop(drop_cols)
    if metrics is None:
        return pl.DataFrame()
    base_col = "market_anytime_td_prob"
    if base_col in metrics.columns:
        for key in DELTA_SEQUENCE:
            suffix = SNAPSHOT_SUFFIXES[key]
            compare_col = base_col if suffix == "" else f"{base_col}{suffix}"
            if compare_col not in metrics.columns:
                continue
            delta_name = "market_anytime_td_delta" if key == "open" else f"market_anytime_td_delta{suffix}"
            metrics = metrics.with_columns(
                pl.when(pl.col(base_col).is_not_null() & pl.col(compare_col).is_not_null())
                .then(pl.col(base_col) - pl.col(compare_col))
                .otherwise(None)
                .alias(delta_name)
            )
    return metrics


def _aggregate_tds_features(df_tds: pl.DataFrame) -> pl.DataFrame:
    if df_tds.is_empty():
        return df_tds
    over_df = df_tds.filter(pl.col("outcome_type").str.contains("over", literal=False))
    under_df = df_tds.filter(pl.col("outcome_type").str.contains("under", literal=False))
    result: Optional[pl.DataFrame] = None
    if not over_df.is_empty():
        agg_over = over_df.group_by(GROUP_KEYS).agg(
            [
                pl.col("point").median().alias("market_player_tds_line"),
                pl.col("implied_prob").mean().alias("market_player_tds_prob_over"),
                pl.col("implied_prob").median().alias("market_player_tds_consensus_over"),
            ]
        )
        result = agg_over
    if not under_df.is_empty():
        agg_under = under_df.group_by(GROUP_KEYS).agg(
            [
                pl.col("implied_prob").mean().alias("market_player_tds_prob_under"),
                pl.col("implied_prob").median().alias("market_player_tds_consensus_under"),
            ]
        )
        if result is None:
            result = agg_under
        else:
            result = result.join(agg_under, on=GROUP_KEYS, how="full", suffix="_dup")
            drop_cols = [f"{key}_dup" for key in GROUP_KEYS if f"{key}_dup" in result.columns]
            if drop_cols:
                result = result.drop(drop_cols)
    return result if result is not None else pl.DataFrame()


def _process_player_market(
    df_raw: pl.DataFrame,
    roster_lookup: Dict[Tuple[int, str, str], str],
) -> pl.DataFrame:
    if df_raw.is_empty():
        return df_raw
    df = df_raw.with_columns(
        pl.col("selection_name")
        .map_elements(_normalize_player_name, return_dtype=pl.Utf8)
        .alias("name_key")
    )
    mapper = _assign_player_id_mapper(roster_lookup)
    df = df.with_columns(
        pl.struct(
            [
                "season",
                "home_team",
                "away_team",
                "selection_team",
                "name_key",
            ]
        ).map_elements(mapper, return_dtype=pl.Utf8).alias("player_id")
    )
    df = df.with_columns(
        pl.col("price").map_elements(_american_to_prob, return_dtype=pl.Float64).alias("implied_prob")
    )
    anytime_df = df.filter(pl.col("market_key") == "player_anytime_td").filter(pl.col("implied_prob").is_not_null())
    tds_df = df.filter(pl.col("market_key") == "player_tds_over").filter(pl.col("implied_prob").is_not_null())
    anytime_features = _aggregate_anytime_features(anytime_df)
    tds_features = _aggregate_tds_features(tds_df)
    frames = [frame for frame in (anytime_features, tds_features) if frame is not None and not frame.is_empty()]
    if not frames:
        return pl.DataFrame()
    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on=GROUP_KEYS, how="full", suffix="_dup")
        drop_cols = [f"{key}_dup" for key in GROUP_KEYS if f"{key}_dup" in result.columns]
        if drop_cols:
            result = result.drop(drop_cols)
    return result


def _write_processed(season: int, week: int, df: pl.DataFrame, overwrite: bool) -> Path:
    out_dir = PLAYER_MARKET_PROCESSED_DIR / f"season={season}" / f"week={week:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part.parquet"
    if out_path.exists() and not overwrite:
        logger.info(
            "Skipping season %s week %s – processed file exists (use --overwrite to refresh).", season, week
        )
        return out_path
    df = df.with_columns([pl.col(col).cast(pl.Float32) for col in df.columns if col.startswith("market_")])
    df.write_parquet(out_path, compression="zstd")
    return out_path


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = _discover_seasons()
    if not seasons:
        logger.info("No seasons discovered – nothing to process.")
        return

    week_filter: Optional[List[int]] = None
    if args.weeks:
        week_filter = [int(w.strip()) for w in args.weeks.split(",") if w.strip()]

    roster_lookup = _build_roster_lookup(seasons)
    if not roster_lookup:
        logger.warning("Roster lookup is empty – player_id mapping may fail.")

    processed_paths: List[Path] = []
    for season in seasons:
        weeks = week_filter if week_filter else _discover_weeks(season)
        if not weeks:
            logger.info("No player odds weeks found for season %s", season)
            continue
        for week in weeks:
            raw = _load_raw_player_odds(season, week)
            if raw is None or raw.is_empty():
                logger.info("No player odds raw data for season %s week %s", season, week)
                continue
            processed = _process_player_market(raw, roster_lookup)
            if processed.is_empty():
                logger.info("Processed player market features empty for season %s week %s", season, week)
                continue
            out_path = _write_processed(season, week, processed, args.overwrite)
            processed_paths.append(out_path)
            mapped_ratio = float(processed.filter(pl.col("player_id").is_not_null()).height) / float(processed.height)
            logger.info(
                "Season %s Week %s player market features written (%s rows). Player mapping rate: %.2f%%",
                season,
                week,
                processed.height,
                mapped_ratio * 100.0,
            )

    if processed_paths:
        logger.info("Player market features written: %d partitions", len(processed_paths))
    else:
        logger.info("No player market features produced.")


if __name__ == "__main__":
    main()


