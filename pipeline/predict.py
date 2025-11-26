"""
NFL prediction pipeline
=======================

Generates anytime-touchdown probabilities for upcoming games by:

1. Loading the upcoming schedule and weekly rosters (via nfl_data_py)
2. Building a player-level scaffold for offensive skill positions
3. Enriching that scaffold with rolling-window features using the same
   caches that power the training pipeline
4. Sequentially running the model chain:
   Availability -> Team Pace -> Usage -> Efficiency -> Anytime TD
5. Writing an actionable slate under `output/predictions/`
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Any

from functools import lru_cache

import json
import logging
import joblib
import numpy as np
import pandas as pd
import yaml
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from nfl_data_py import import_schedules, import_weekly_rosters
except Exception as exc:  # pragma: no cover - dependency guard
    import_schedules = None
    import_weekly_rosters = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

import polars as pl

from utils.feature.rolling_window import add_rolling_features
from utils.general.paths import (
    PROJ_ROOT,
    STADIUM_COORDS_FILE,
    QB_PROFILE_DIR,
    TRAVEL_CALENDAR_DIR,
    OPPONENT_SPLIT_DIR,
    WEATHER_FORECAST_DIR,
    ODDS_SNAPSHOT_DIR,
)
from utils.collect.weather_forecasts import collect_weather_forecasts
from utils.feature.odds import collect_odds_snapshots
from utils.feature.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
from utils.feature.odds import (
    add_nfl_odds_features_to_df,
    NFL_ODDS_COLUMNS,
    ODDS_FLAG_COLUMNS,
)
from utils.feature.team_context import add_team_context_features
from utils.feature.offense_context import add_offense_context_features_inference
from utils.feature.opponent_splits import load_rolling_opponent_splits
from utils.feature.weather_features import (
    add_weather_forecast_features_inference,
    append_weather_context_flags,
)
from utils.feature.asof_metadata import load_asof_metadata, build_asof_metadata
from utils.feature.asof import (
    decision_cutoff_hours_default,
    fallback_cutoff_hours,
    decision_cutoff_override,
    get_decision_cutoff_hours,
    get_fallback_cutoff_hours,
)
from utils.collect.nfl_schedules import get_schedule as get_cached_schedule
from utils.collect.espn_injuries import collect_espn_injuries
from utils.train.injury_availability import (
    load_latest_artifact as load_injury_artifact,
    predict_probabilities as predict_injury_probabilities,
    MODEL_PROB_COL,
    PROB_LOW_COL,
    PROB_HIGH_COL,
)
from utils.feature.player_game_level import _compute_injury_history_rates

MODEL_DIR = Path("output/models")
METRICS_DIR = Path("output/metrics")
PREDICTION_DIR = Path("output/predictions")
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
TEAM_CONTEXT_HISTORY_PATH = PROJ_ROOT / "data/processed/team_context_history.parquet"
OFFENSE_CONTEXT_HISTORY_PATH = PROJ_ROOT / "data/processed/offense_context_history.parquet"
PLAYER_DRIVE_HISTORY_PATH = PROJ_ROOT / "data/processed/player_drive_context_history.parquet"
PLAYER_GAME_DIR = PROJ_ROOT / "data/processed/player_game_by_week"
CONFIG_PATH = PROJ_ROOT / "config/training.yaml"

logger = logging.getLogger(__name__)

ENABLE_WEATHER_FEATURES = True

ALLOWED_POSITIONS = {"QB", "RB", "WR", "TE"}
# Status strings in nfl_data_py rosters are inconsistent; we exclude obvious
# non-participants (various injured reserve designations, suspended/exempt).
EXCLUDED_ROSTER_STATUSES = {
    "IR", "IR-R", "IR-N", "IR-C", "PUP", "NFI", "DNR", "SUS", "EXEMPT",
    "RET", "COVID", "PS", "WAV", "CUT", "RES", "RESERVE", "RES-RET", "DEV"
}
EXCLUDED_GAME_STATUSES = {"OUT", "DOUBTFUL", "INACTIVE", "SUSPENDED"}

INJURY_CACHE_DIR = PROJ_ROOT / "cache" / "feature" / "injuries"
INJURY_MODEL_FALLBACK_INTERVAL_THRESHOLD = 0.65


ODDS_COLUMNS = NFL_ODDS_COLUMNS + ODDS_FLAG_COLUMNS

META_COLUMNS = [
    "player_id",
    "player_name",
    "position",
    "team",
    "opponent",
    "season",
    "week",
    "game_id",
    "game_date",
    "depth_chart_position",
    "depth_chart_order",
    "injury_game_designation",
    "injury_report_status",
    "injury_practice_status",
    "injury_report_primary",
    "injury_practice_primary",
    "injury_is_listed",
    "injury_practice_dnp_count",
    "injury_practice_limited_count",
    "injury_practice_full_count",
    "is_home",
    "game_start_hour_utc",
    "game_day_of_week",
    "season_type",
    "stadium_key",
    "stadium_name",
    "stadium_tz",
    "roof",
    "surface",
    "home_team",
    "away_team",
    *ODDS_COLUMNS,
]

BASE_GAME_COLS = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "pass_attempt",
    "completion",
    "carry",
    "target",
    "reception",
    "passing_td",
    "rushing_td_count",
    "receiving_td_count",
    "touchdowns",
    "td_count",
    "hist_target_share_prev",
    "hist_target_share_l3",
    "hist_carry_share_prev",
    "hist_carry_share_l3",
    "hist_pass_attempt_share_prev",
    "hist_pass_attempt_share_l3",
    "hist_red_zone_target_share_prev",
    "hist_red_zone_target_share_l3",
    "hist_red_zone_carry_share_prev",
    "hist_red_zone_carry_share_l3",
    "hist_goal_to_go_target_share_prev",
    "hist_goal_to_go_target_share_l3",
    "hist_goal_to_go_carry_share_prev",
    "hist_goal_to_go_carry_share_l3",
]

PS_BASELINE_COLUMNS = [
    "ps_route_participation_plays",
    "ps_team_dropbacks",
    "ps_route_participation_pct",
    "ps_targets_total",
    "ps_targets_slot_count",
    "ps_targets_wide_count",
    "ps_targets_inline_count",
    "ps_targets_backfield_count",
    "ps_targets_slot_share",
    "ps_targets_wide_share",
    "ps_targets_inline_share",
    "ps_targets_backfield_share",
    "ps_total_touches",
    "ps_scripted_touches",
    "ps_scripted_touch_share",
    "ps_tracking_team_dropbacks",
    "ps_tracking_has_game_data",
    "ps_route_participation_pct_prev",
    "ps_route_participation_pct_l3",
    "ps_scripted_touch_share_prev",
    "ps_scripted_touch_share_l3",
    "ps_targets_slot_share_prev",
    "ps_targets_slot_share_l3",
    "ps_targets_wide_share_prev",
    "ps_targets_wide_share_l3",
    "ps_targets_inline_share_prev",
    "ps_targets_inline_share_l3",
    "ps_targets_backfield_share_prev",
    "ps_targets_backfield_share_l3",
    "ps_route_participation_plays_l3",
    "ps_targets_total_l3",
    "ps_targets_slot_count_l3",
    "ps_targets_wide_count_l3",
    "ps_targets_inline_count_l3",
    "ps_targets_backfield_count_l3",
    # Historical namespace (preferred)
    "ps_hist_route_participation_pct_prev",
    "ps_hist_route_participation_pct_l3",
    "ps_hist_route_participation_plays_prev",
    "ps_hist_route_participation_plays_l3",
    "ps_hist_targets_total_prev",
    "ps_hist_targets_total_l3",
    "ps_hist_targets_slot_count_prev",
    "ps_hist_targets_slot_count_l3",
    "ps_hist_targets_wide_count_prev",
    "ps_hist_targets_wide_count_l3",
    "ps_hist_targets_inline_count_prev",
    "ps_hist_targets_inline_count_l3",
    "ps_hist_targets_backfield_count_prev",
    "ps_hist_targets_backfield_count_l3",
    "ps_hist_targets_slot_share_prev",
    "ps_hist_targets_slot_share_l3",
    "ps_hist_targets_wide_share_prev",
    "ps_hist_targets_wide_share_l3",
    "ps_hist_targets_inline_share_prev",
    "ps_hist_targets_inline_share_l3",
    "ps_hist_targets_backfield_share_prev",
    "ps_hist_targets_backfield_share_l3",
    "ps_hist_total_touches_prev",
    "ps_hist_total_touches_l3",
    "ps_hist_scripted_touches_prev",
    "ps_hist_scripted_touches_l3",
    "ps_hist_scripted_touch_share_prev",
    "ps_hist_scripted_touch_share_l3",
    "ps_hist_tracking_team_dropbacks_prev",
    "ps_hist_tracking_team_dropbacks_l3",
    "ps_hist_tracking_has_game_data_prev",
    "ps_hist_tracking_has_game_data_l3",
]

PS_L3_FALLBACK_MAP = {
    "ps_route_participation_plays_l3": "ps_route_participation_plays",
    "ps_targets_total_l3": "ps_targets_total",
    "ps_targets_slot_count_l3": "ps_targets_slot_count",
    "ps_targets_wide_count_l3": "ps_targets_wide_count",
    "ps_targets_inline_count_l3": "ps_targets_inline_count",
    "ps_targets_backfield_count_l3": "ps_targets_backfield_count",
    "ps_hist_route_participation_plays_l3": "ps_hist_route_participation_plays_prev",
    "ps_hist_targets_total_l3": "ps_hist_targets_total_prev",
    "ps_hist_targets_slot_count_l3": "ps_hist_targets_slot_count_prev",
    "ps_hist_targets_wide_count_l3": "ps_hist_targets_wide_count_prev",
    "ps_hist_targets_inline_count_l3": "ps_hist_targets_inline_count_prev",
    "ps_hist_targets_backfield_count_l3": "ps_hist_targets_backfield_count_prev",
    "ps_hist_total_touches_l3": "ps_hist_total_touches_prev",
    "ps_hist_scripted_touches_l3": "ps_hist_scripted_touches_prev",
    "ps_hist_route_participation_pct_l3": "ps_hist_route_participation_pct_prev",
    "ps_hist_scripted_touch_share_l3": "ps_hist_scripted_touch_share_prev",
    "ps_hist_targets_slot_share_l3": "ps_hist_targets_slot_share_prev",
    "ps_hist_targets_wide_share_l3": "ps_hist_targets_wide_share_prev",
    "ps_hist_targets_inline_share_l3": "ps_hist_targets_inline_share_prev",
    "ps_hist_targets_backfield_share_l3": "ps_hist_targets_backfield_share_prev",
    "ps_hist_tracking_team_dropbacks_l3": "ps_hist_tracking_team_dropbacks_prev",
    "ps_hist_tracking_has_game_data_l3": "ps_hist_tracking_has_game_data_prev",
}

OUTPUT_COLUMN_PREFIXES = (
    "pred_",
    "expected_",
    "hist_",
    "ps_",
    "snap_",
    "team_ctx_",
    "opp_ctx_",
    "opp_def_",
    "market_",
    "travel_",
    "injury_",
    "drive_hist_",
)
OUTPUT_COLUMN_WHITELIST = [
    "spread_line",
    "team_implied_total",
    "season",
    "week",
    "game_id",
    "position",
]

with STADIUM_COORDS_FILE.open() as fh:
    _STADIUM_COORDS = json.load(fh)

_STADIUM_ALIAS_MAP: dict[str, str] = {}
for code, meta in _STADIUM_COORDS.items():
    if isinstance(meta, dict):
        _STADIUM_ALIAS_MAP[code.upper()] = code
        for alias in (meta.get("aliases") or []):
            _STADIUM_ALIAS_MAP[str(alias).upper()] = code


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _ensure_injury_cache(seasons: Iterable[int]) -> None:
    """Ensure injury caches exist and are fresh for the requested seasons.
    
    For the current season, refreshes the cache if it's older than 12 hours
    to ensure we have the latest injury reports for predictions.
    """
    import datetime as dt
    
    cache_dir = PROJ_ROOT / "cache" / "feature" / "injuries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    current_year = dt.datetime.now().year
    stale_threshold = dt.timedelta(hours=12)
    now = dt.datetime.now()
    
    missing: list[int] = []
    stale: list[int] = []
    
    for season in {int(s) for s in seasons if s is not None}:
        cache_path = cache_dir / f"injury_{int(season)}.parquet"
        if not cache_path.exists():
            missing.append(int(season))
        elif int(season) >= current_year:
            # Check if current season's cache is stale
            mtime = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
            if now - mtime > stale_threshold:
                stale.append(int(season))
    
    if missing:
        try:
            logger.info("Collecting ESPN injury reports for missing seasons %s", missing)
            collect_espn_injuries(missing, overwrite=False)
        except Exception as exc:
            logger.warning("Failed to collect ESPN injury cache: %s", exc)
    
    if stale:
        try:
            logger.info("Refreshing stale ESPN injury reports for seasons %s", stale)
            collect_espn_injuries(stale, overwrite=True)
        except Exception as exc:
            logger.warning("Failed to refresh ESPN injury cache: %s", exc)


def _ensure_weather_cache(seasons: Iterable[int]) -> None:
    """Ensure weather forecast caches exist and are fresh for the requested seasons."""
    import datetime as dt
    
    # For inference, we always want the latest forecast for upcoming games.
    # collect_weather_forecasts handles staleness and lead-time logic internally.
    # We just trigger it for the current season to ensure we have data.
    
    current_year = dt.datetime.now().year
    active_seasons = [int(s) for s in seasons if int(s) >= current_year]
    
    if not active_seasons:
        return
        
    try:
        # We don't force refresh indiscriminately, but reliance on lead_hours 
        # means it will fetch if the current time window matches a new lead time.
        logger.info("Ensuring weather forecasts for seasons %s", active_seasons)
        collect_weather_forecasts(active_seasons, force_refresh=False)
    except Exception as exc:
        logger.warning("Failed to ensure weather cache: %s", exc)


def _ensure_odds_cache(seasons: Iterable[int], start_date=None, end_date=None) -> None:
    """Ensure odds snapshot caches exist and are fresh for the requested window."""
    import datetime as dt
    
    current_year = dt.datetime.now().year
    active_seasons = [int(s) for s in seasons if int(s) >= current_year]
    
    if not active_seasons:
        return

    try:
        logger.info("Ensuring odds snapshots for seasons %s (window: %s to %s)", active_seasons, start_date, end_date)
        collect_odds_snapshots(
            active_seasons,
            start_date=start_date,
            end_date=end_date,
            # We want to capture player props if available, though they are heavy
            include_player_props=True 
        )
    except Exception as exc:
        logger.warning("Failed to ensure odds cache: %s", exc)



def _inject_composed_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    def _ensure_availability_composites(frame: pd.DataFrame) -> None:
        has_active = "pred_availability_active" in frame.columns
        has_share = "pred_availability_snapshare" in frame.columns
        if not (has_active and has_share):
            return
        active = pd.to_numeric(frame["pred_availability_active"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        share = pd.to_numeric(frame["pred_availability_snapshare"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        combined = (active * share).astype(np.float32)
        if "pred_availability" not in frame.columns:
            frame["pred_availability"] = combined
        if "pred_availability_raw" not in frame.columns:
            if "pred_availability_active_raw" in frame.columns:
                raw_active = (
                    pd.to_numeric(frame["pred_availability_active_raw"], errors="coerce")
                    .fillna(0.0)
                    .clip(0.0, 1.0)
                )
                frame["pred_availability_raw"] = (raw_active * share).astype(np.float32)
            else:
                frame["pred_availability_raw"] = combined

    _ensure_availability_composites(df)

    def _safe_mul(a: str, b: str, out_col: str):
        if a in df.columns and b in df.columns:
             # Only overwrite if not present or to refresh
            df[out_col] = df[a] * df[b]

    def _select_column(candidates: list[str]) -> str | None:
        for name in candidates:
            if name in df.columns:
                return name
        return None

    # Availability x usage → expected opportunities
    _safe_mul("pred_availability", "pred_usage_targets", "expected_targets")
    _safe_mul("pred_availability_raw", "pred_usage_targets", "expected_targets_raw")
    _safe_mul("pred_availability", "pred_usage_carries", "expected_carries")
    _safe_mul("pred_availability_raw", "pred_usage_carries", "expected_carries_raw")

    if "pred_team_pace" in df.columns:
        team_pace = pd.to_numeric(df["pred_team_pace"], errors="coerce").clip(lower=0.0)
        df["expected_team_plays"] = team_pace
        pass_rate_col = _select_column(
            [
                "team_ctx_pass_rate_prev",
                "team_ctx_pass_rate_l3",
                "team_ctx_pass_rate_l5",
            ]
        )
        if pass_rate_col:
            pass_rate = (
                pd.to_numeric(df[pass_rate_col], errors="coerce")
                .fillna(0.5)
                .clip(lower=0.0, upper=1.0)
            )
            df["expected_team_pass_plays"] = team_pace * pass_rate
            df["expected_team_rush_plays"] = team_pace * (1.0 - pass_rate)

    if "expected_targets" in df.columns and "expected_carries" in df.columns:
        df["expected_opportunities"] = df["expected_targets"] + df["expected_carries"]
    if "expected_targets_raw" in df.columns and "expected_carries_raw" in df.columns:
        df["expected_opportunities_raw"] = df["expected_targets_raw"] + df["expected_carries_raw"]

    if "pred_efficiency_tds" in df.columns and "expected_opportunities" in df.columns:
        df["expected_td_signal"] = df["pred_efficiency_tds"] * df["expected_opportunities"]
    if "pred_efficiency_tds" in df.columns and "expected_opportunities_raw" in df.columns:
        df["expected_td_signal_raw"] = df["pred_efficiency_tds"] * df["expected_opportunities_raw"]

    def _expected_red_zone(base_col: str, share_col: str | None, out_col: str):
        if base_col in df.columns and share_col:
            share = pd.to_numeric(df[share_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
            df[out_col] = df[base_col] * share

    rz_target_share = _select_column(
        [
            "hist_red_zone_target_share_prev",
            "hist_red_zone_target_share_l3",
            "red_zone_target_share_prev",
            "red_zone_target_share_l3",
        ]
    )
    rz_carry_share = _select_column(
        [
            "hist_red_zone_carry_share_prev",
            "hist_red_zone_carry_share_l3",
            "red_zone_carry_share_prev",
            "red_zone_carry_share_l3",
        ]
    )
    team_rz_rate = _select_column(
        [
            "team_red_zone_play_rate_prev",
            "team_red_zone_play_rate_l3",
        ]
    )
    team_gtg_rate = _select_column(
        [
            "team_goal_to_go_play_rate_prev",
            "team_goal_to_go_play_rate_l3",
        ]
    )

    def _team_factor(rate_col: str | None, out_col: str) -> None:
        if rate_col and rate_col in df.columns:
            df[out_col] = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)

    _team_factor(team_rz_rate, "team_red_zone_rate")
    _team_factor(team_gtg_rate, "team_goal_to_go_rate")

    _expected_red_zone("expected_targets", rz_target_share, "expected_rz_targets")
    _expected_red_zone("expected_targets_raw", rz_target_share, "expected_rz_targets_raw")
    _expected_red_zone("expected_carries", rz_carry_share, "expected_rz_carries")
    _expected_red_zone("expected_carries_raw", rz_carry_share, "expected_rz_carries_raw")

    if "expected_opportunities" in df.columns and "team_red_zone_rate" in df.columns:
        df["team_based_rz_opportunities"] = df["expected_opportunities"] * df["team_red_zone_rate"]
    if "expected_opportunities_raw" in df.columns and "team_red_zone_rate" in df.columns:
        df["team_based_rz_opportunities_raw"] = df["expected_opportunities_raw"] * df["team_red_zone_rate"]

    def _combine_rz(base: str, team_col: str, out_col: str):
        if base in df.columns and team_col in df.columns:
            df[out_col] = df[base].fillna(0.0) + df[team_col].fillna(0.0)

    _combine_rz("expected_rz_opportunities", "team_based_rz_opportunities", "expected_rz_opportunities")
    _combine_rz("expected_rz_opportunities_raw", "team_based_rz_opportunities_raw", "expected_rz_opportunities_raw")

    if "expected_rz_targets" in df.columns and "expected_rz_carries" in df.columns:
        df["expected_rz_opportunities"] = df["expected_rz_targets"] + df["expected_rz_carries"]
    if "expected_rz_targets_raw" in df.columns and "expected_rz_carries_raw" in df.columns:
        df["expected_rz_opportunities_raw"] = df["expected_rz_targets_raw"] + df["expected_rz_carries_raw"]

    if "pred_efficiency_tds" in df.columns and "expected_rz_opportunities" in df.columns:
        df["expected_rz_td_signal"] = df["pred_efficiency_tds"] * df["expected_rz_opportunities"]
    if "pred_efficiency_tds" in df.columns and "expected_rz_opportunities_raw" in df.columns:
        df["expected_rz_td_signal_raw"] = df["pred_efficiency_tds"] * df["expected_rz_opportunities_raw"]

    if "pred_efficiency_rec_yards" in df.columns:
        df["expected_receiving_yards"] = (
            pd.to_numeric(df["pred_efficiency_rec_yards"], errors="coerce")
            .clip(lower=0.0)
        )
    if "pred_efficiency_rush_yards" in df.columns:
        df["expected_rushing_yards"] = (
            pd.to_numeric(df["pred_efficiency_rush_yards"], errors="coerce")
            .clip(lower=0.0)
        )
    if "expected_receiving_yards" in df.columns and "expected_rushing_yards" in df.columns:
        df["expected_total_yards"] = df["expected_receiving_yards"] + df["expected_rushing_yards"]

    if "expected_td_signal" in df.columns:
        df["expected_td_prob_poisson"] = 1.0 - np.exp(-df["expected_td_signal"].clip(lower=0.0))
    if "expected_td_signal_raw" in df.columns:
        df["expected_td_prob_poisson_raw"] = 1.0 - np.exp(-df["expected_td_signal_raw"].clip(lower=0.0))
    if "expected_rz_td_signal" in df.columns:
        df["expected_rz_td_prob_poisson"] = 1.0 - np.exp(-df["expected_rz_td_signal"].clip(lower=0.0))
    if "expected_rz_td_signal_raw" in df.columns:
        df["expected_rz_td_prob_poisson_raw"] = 1.0 - np.exp(-df["expected_rz_td_signal_raw"].clip(lower=0.0))

    return df

def _collect_partition_paths(base_dir: Path, seasons: Iterable[int]) -> list[Path]:
    paths: list[Path] = []
    for season in sorted({int(s) for s in seasons}):
        season_dir = base_dir / f"season={season}"
        if not season_dir.exists():
            continue
        paths.extend(sorted(season_dir.glob("week=*/part.parquet")))
    return paths


@lru_cache(maxsize=6)
def _load_ps_baselines(season: int) -> pl.DataFrame:
    """
    Load the latest available pre-snap participation features for seasons prior to `season`.
    These serve as carry-forward baselines when the current season has no tracking data yet.
    """
    if season is None:
        return pl.DataFrame()
    candidate_seasons = [yr for yr in range(season - 3, season) if yr > 2000]
    paths = _collect_partition_paths(PLAYER_GAME_DIR, candidate_seasons)
    if not paths:
        return pl.DataFrame()
    scan = pl.scan_parquet(
        paths,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    required_cols = ["player_id", "game_date", *PS_BASELINE_COLUMNS]
    missing = [col for col in required_cols if col not in scan.collect_schema().names()]
    if missing:
        required_cols = [col for col in required_cols if col in scan.collect_schema().names()]
    if "player_id" not in required_cols or not required_cols:
        return pl.DataFrame()
    df = (
        scan.select(required_cols)
        .filter(pl.col("player_id").is_not_null() & pl.col("game_date").is_not_null())
        .with_columns(pl.col("game_date").cast(pl.Datetime("ms")))
        .sort(["player_id", "game_date"])
        .group_by("player_id", maintain_order=True)
        .agg([pl.col(col).last().alias(col) for col in required_cols if col not in {"player_id", "game_date"}])
        .collect()
    )
    return df


def _load_opponent_split_features(seasons: Iterable[int]) -> pl.DataFrame:
    """
    Mirror the rolling opponent-level defensive splits that were used during training.
    """
    df = load_rolling_opponent_splits(list(seasons), windows=[3])
    if df.is_empty():
        return df
    df = df.with_columns(
        [
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("opponent").cast(pl.Utf8),
            pl.col("game_date").cast(pl.Datetime("ms")),
        ]
    )
    float_cols = [
        col
        for col, dtype in df.schema.items()
        if col.startswith("opp_def_")
        and dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32)
    ]
    if float_cols:
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in float_cols])
    return df


def _apply_ps_fallback(enriched: pl.DataFrame, *, season_hint: int | None) -> pl.DataFrame:
    if enriched.is_empty():
        return enriched

    def _safe_any(exprs: list[pl.Expr], *, alias: str, default: bool = False) -> pl.Expr:
        expr = pl.any_horizontal(exprs) if exprs else pl.lit(default)
        return expr.alias(alias)

    existing_cols = set(enriched.columns)
    has_actual_signals: list[pl.Expr] = []
    recent_hist_signals: list[pl.Expr] = []
    for col in (
        "ps_team_dropbacks",
        "ps_route_participation_plays",
        "ps_hist_route_participation_pct_prev",
        "ps_hist_total_touches_prev",
        "ps_hist_scripted_touches_prev",
        "ps_tracking_has_game_data",
        "ps_hist_tracking_has_game_data_prev",
    ):
        if col in existing_cols:
            has_actual_signals.append(pl.col(col).fill_null(0) > 0)
    for col in (
        "ps_hist_route_participation_pct_prev",
        "ps_hist_route_participation_pct_l3",
        "ps_hist_scripted_touch_share_prev",
        "ps_hist_scripted_touch_share_l3",
        "ps_hist_tracking_has_game_data_prev",
        "ps_hist_tracking_has_game_data_l3",
    ):
        if col in existing_cols:
            recent_hist_signals.append(pl.col(col).is_not_null())

    enriched = enriched.with_columns(
        [
            _safe_any(has_actual_signals, alias="__ps_has_actual"),
            _safe_any(recent_hist_signals, alias="__ps_has_recent_hist"),
        ]
    )

    missing_guard = "ps_team_dropbacks" in existing_cols
    baseline_used_signals: list[pl.Expr] = []
    helper_flag = "__ps_missing_flag"
    enriched = enriched.with_columns(
        pl.col("ps_team_dropbacks").fill_null(0).le(0).alias(helper_flag)
        if missing_guard
        else pl.lit(True).alias(helper_flag)
    )
    missing_flag = pl.col(helper_flag)

    if season_hint is not None:
        baseline = _load_ps_baselines(season_hint)
    else:
        baseline = pl.DataFrame()

    if not baseline.is_empty():
        available_cols = [
            col
            for col in PS_BASELINE_COLUMNS
            if col in enriched.columns and col in baseline.columns
        ]
        if available_cols:
            rename_map = {col: f"__ps_base_{col}" for col in available_cols if col in baseline.columns}
            baseline = baseline.rename(rename_map)
            enriched = enriched.join(baseline, on="player_id", how="left")

            fill_exprs: list[pl.Expr] = []
            drop_cols: list[str] = []
            for col in available_cols:
                base_col = f"__ps_base_{col}"
                if base_col in enriched.columns:
                    baseline_used_signals.append(missing_flag & pl.col(base_col).is_not_null())
                    fill_exprs.append(
                        pl.when(missing_flag & pl.col(base_col).is_not_null())
                        .then(pl.col(base_col))
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
                    drop_cols.append(base_col)
            if fill_exprs:
                enriched = enriched.with_columns(fill_exprs)
            if drop_cols:
                enriched = enriched.drop(drop_cols)

            l3_exprs: list[pl.Expr] = []
            for l3_col, src_col in PS_L3_FALLBACK_MAP.items():
                if l3_col in enriched.columns and src_col in enriched.columns:
                    l3_exprs.append(
                        pl.when(
                            pl.col(src_col).is_not_null()
                            & (pl.col(src_col).fill_null(0) > 0)
                            & (pl.col(l3_col).fill_null(0) == 0)
                        )
                        .then(pl.col(src_col))
                        .otherwise(pl.col(l3_col))
                        .alias(l3_col)
                    )
            if l3_exprs:
                enriched = enriched.with_columns(l3_exprs)

    enriched = enriched.with_columns(
        _safe_any(baseline_used_signals, alias="__ps_used_baseline", default=False)
    )

    enriched = enriched.with_columns(
        [
            pl.col("__ps_has_actual").cast(pl.Int8).alias("ps_tracking_has_actual"),
            pl.col("__ps_has_recent_hist").cast(pl.Int8).alias("ps_tracking_has_recent_hist"),
            pl.col("__ps_used_baseline").cast(pl.Int8).alias("ps_tracking_used_baseline"),
            pl.when(pl.col("__ps_has_actual"))
            .then(pl.lit("actual"))
            .when(pl.col("__ps_used_baseline"))
            .then(pl.lit("baseline"))
            .otherwise(pl.lit("none"))
            .alias("ps_baseline_source"),
        ]
    )
    enriched = enriched.drop([helper_flag, "__ps_has_actual", "__ps_has_recent_hist", "__ps_used_baseline"])
    return enriched


def _load_qb_profile_features(seasons: Iterable[int]) -> pl.DataFrame:
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


def _load_travel_calendar_features(seasons: Iterable[int]) -> pl.DataFrame:
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
    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df.select(existing_cols)
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


def _prepare_weather_columns(df: pl.DataFrame) -> pl.DataFrame:
    weather_cols = [col for col in df.columns if col.startswith("weather_")]
    if not weather_cols:
        return df
    drop_cols = [
        col
        for col in weather_cols
        if col.endswith("_ts")
        or col.endswith("_source_detail")
        or col in {"weather_conditions", "weather_precip_type"}
    ]
    if drop_cols:
        df = df.drop(drop_cols, strict=False)
    numeric_cols = [
        col
        for col in weather_cols
        if col not in drop_cols
        and df.schema.get(col)
        in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32)
    ]
    if numeric_cols:
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in numeric_cols])
    flag_cols = [
        col
        for col in weather_cols
        if col not in drop_cols
        and (
            col.endswith("_flag")
            or col.endswith("_is_backfill")
            or col.endswith("_is_historical")
        )
    ]
    if flag_cols:
        df = df.with_columns([pl.col(col).cast(pl.Int8) for col in flag_cols])
    return df


def _attach_additional_context_features(enriched: pl.DataFrame) -> pl.DataFrame:
    if enriched.is_empty():
        return enriched
    enriched = _prepare_weather_columns(enriched)
    required_keys = {"season", "week"}
    if not required_keys <= set(enriched.columns):
        return enriched
    seasons = enriched.select("season").unique()["season"].to_list()
    opponent_pl = _load_opponent_split_features(seasons)
    if not opponent_pl.is_empty() and {"opponent", "game_date"} <= set(enriched.columns):
        opponent_join = (
            opponent_pl.drop_nulls("game_date")
            .drop(["season", "week"], strict=False)
            .sort(["opponent", "game_date"])
        )
        enriched = enriched.with_columns(
            pl.col("game_date").cast(pl.Datetime("ms")).alias("game_date")
        )
        enriched = enriched.with_columns(
            pl.arange(0, enriched.height).alias("__row_order")
        )
        enriched = enriched.sort(["opponent", "game_date"])
        enriched = enriched.join_asof(
            opponent_join,
            left_on="game_date",
            right_on="game_date",
            by="opponent",
            strategy="backward",
        )
        enriched = enriched.sort("__row_order").drop("__row_order")
        if "game_date_right" in enriched.columns:
            enriched = enriched.drop("game_date_right")
    qb_pl = _load_qb_profile_features(seasons)
    if not qb_pl.is_empty() and {"team", "game_date"} <= set(enriched.columns):
        # Use as-of join for QB profiles to handle missing weeks
        # First, we need a date column in qb_pl for the as-of join
        if "qb_profile_data_as_of" in qb_pl.columns:
            qb_join = (
                qb_pl.drop_nulls("qb_profile_data_as_of")
                .drop(["season", "week"], strict=False)
                .with_columns(pl.col("qb_profile_data_as_of").cast(pl.Datetime("ms")).alias("__qb_join_date"))
                .sort(["team", "__qb_join_date"])
            )
            enriched = enriched.with_columns(
                pl.arange(0, enriched.height).alias("__qb_row_order")
            )
            enriched = enriched.sort(["team", "game_date"])
            enriched = enriched.join_asof(
                qb_join,
                left_on="game_date",
                right_on="__qb_join_date",
                by="team",
                strategy="backward",
            )
            enriched = enriched.sort("__qb_row_order").drop(["__qb_row_order", "__qb_join_date"], strict=False)
        else:
            # Fallback to exact week join if no date column available
            enriched = enriched.join(
                qb_pl,
                on=["season", "week", "team"],
                how="left",
            )
    travel_pl = _load_travel_calendar_features(seasons)
    if not travel_pl.is_empty() and {"team"} <= set(enriched.columns):
        # Try as-of join first if we have game_date columns
        if "game_date" in enriched.columns and "game_date" in travel_pl.columns:
            travel_join = (
                travel_pl.drop_nulls("game_date")
                .drop(["season", "week"], strict=False)
                .with_columns(pl.col("game_date").cast(pl.Datetime("ms")).alias("__travel_join_date"))
                .sort(["team", "__travel_join_date"])
            )
            enriched = enriched.with_columns(
                pl.arange(0, enriched.height).alias("__travel_row_order")
            )
            enriched = enriched.sort(["team", "game_date"])
            enriched = enriched.join_asof(
                travel_join,
                left_on="game_date",
                right_on="__travel_join_date",
                by="team",
                strategy="backward",
            )
            enriched = enriched.sort("__travel_row_order").drop(["__travel_row_order", "__travel_join_date"], strict=False)
        else:
            # Fallback to exact week join
            enriched = enriched.join(
                travel_pl,
                on=["season", "week", "team"],
                how="left",
            )
    drop_cols = [
        col
        for col in ("qb_profile_data_as_of", "qb_profile_team_data_as_of", "game_local_start")
        if col in enriched.columns
    ]
    if drop_cols:
        enriched = enriched.drop(drop_cols)
    return enriched


def _normalize_stadium(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return _STADIUM_ALIAS_MAP.get(raw.strip().upper())


def _ensure_imports_ready() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "nfl_data_py is required for the prediction pipeline. "
            f"Import failed with: {_IMPORT_ERROR}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFL anytime touchdown prediction pipeline.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Prediction slate start date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of consecutive days (from --date) to include. Default: 1.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output filename. Defaults to predictions_<start>_<end>.csv",
    )
    parser.add_argument(
        "--decision-cutoff-hours",
        type=float,
        default=None,
        help="Override decision cutoff window (hours before kickoff).",
    )
    parser.add_argument(
        "--fallback-cutoff-hours",
        type=float,
        default=None,
        help="Override fallback cutoff when kickoff time is unavailable.",
    )
    return parser.parse_args()


def _resolve_dates(start_str: str | None, days: int) -> Tuple[date, date]:
    start = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else date.today()
    end = start + timedelta(days=max(days, 1) - 1)
    return start, end


def _load_schedule(start: date, end: date) -> pd.DataFrame:
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
        "game_id",
        "gameday",
        "week",
        "season",
        "home_team",
        "away_team",
        "start_time_utc",
        "season_type",
        "game_type",
        "roof",
        "surface",
        "stadium",
        "location",
        "site",
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

    try:
        cached_sched = get_cached_schedule(list(seasons))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Falling back to nfl_data_py schedule only: %s", exc)
        cached_sched = pd.DataFrame()

    if not cached_sched.empty:
        cached_cols = [
            "game_id",
            "start_time_utc",
            "roof",
            "surface",
            "stadium",
            "location",
            "site",
        ]
        cache_available = [col for col in cached_cols if col in cached_sched.columns]
        cache_frame = cached_sched[cache_available].copy()
        if "start_time_utc" in cache_frame.columns:
            cache_frame["start_time_utc"] = pd.to_datetime(
                cache_frame["start_time_utc"], errors="coerce", utc=True
            )
        rename_map = {
            col: f"{col}__cache" for col in cache_available if col != "game_id"
        }
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


def _load_rosters(seasons: Iterable[int]) -> pd.DataFrame:
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
        roster["full_name"] = roster.get("first_name", "").fillna("") + " " + roster.get("last_name", "").fillna("")
        name_col = "full_name"

    # Ensure depth_chart_order is preserved if available
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


def _build_scaffold(games: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    roster_key = roster.set_index(["season", "week", "team"])

    def _is_player_available(row: pd.Series) -> bool:
        status = str(row.get("status", "") or "").strip().upper()
        if status and (status in EXCLUDED_ROSTER_STATUSES or status.startswith("IR") or status.startswith("RESERVE")):
            return False
        game_status = str(
            row.get("injury_game_status", "")
            or row.get("injury_game_designation", "")
            or ""
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
        stadium_key = (
            _normalize_stadium(raw_venue)
            or _normalize_stadium(home)
            or home
        )
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

                records.append(
                    {
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
                    }
                )

    if not records:
        raise ValueError("Unable to build prediction scaffold – roster data missing for scheduled games.")

    df = pd.DataFrame.from_records(records).drop_duplicates(subset=["player_id", "game_id"])
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
    if "game_start_utc" in df.columns:
        df["game_start_utc"] = pd.to_datetime(df["game_start_utc"], utc=True, errors="coerce")
    if "depth_chart_order" in df.columns:
        df["depth_chart_order"] = pd.to_numeric(df["depth_chart_order"], errors="coerce")
    for col in ("is_home", "game_start_hour_utc", "game_day_of_week"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def _compute_features(scaffold: pd.DataFrame) -> pd.DataFrame:
    pl_df = pl.from_pandas(scaffold)
    if "game_date" in pl_df.columns:
        pl_df = pl_df.with_columns(
            pl.col("game_date").cast(pl.Datetime(time_unit="ms"))
        )
    if "home_team_abbr" not in pl_df.columns and "home_team" in pl_df.columns:
        pl_df = pl_df.with_columns(pl.col("home_team").alias("home_team_abbr"))
    if "away_team_abbr" not in pl_df.columns and "away_team" in pl_df.columns:
        pl_df = pl_df.with_columns(pl.col("away_team").alias("away_team_abbr"))
    season_hint = None
    if "season" in pl_df.columns:
        pl_df = pl_df.with_columns(pl.col("season").cast(pl.Int32))
        try:
            season_hint = int(pl_df.get_column("season").max())
        except Exception:
            season_hint = None
    if "week" in pl_df.columns:
        pl_df = pl_df.with_columns(pl.col("week").cast(pl.Int32))

    seasons = pl_df.get_column("season").unique().to_list() if "season" in pl_df.columns else []
    game_ids = pl_df.get_column("game_id").unique().to_list() if "game_id" in pl_df.columns else []
    asof_meta = pl.DataFrame()
    try:
        asof_meta = load_asof_metadata()
    except Exception as exc:
        logger.warning("Failed to load as-of metadata: %s", exc)
        asof_meta = pl.DataFrame()

    if asof_meta.is_empty() or (game_ids and asof_meta.filter(pl.col("game_id").is_in(game_ids)).height < len(game_ids)):
        if seasons:
            try:
                asof_meta = build_asof_metadata(seasons, force=True)
            except Exception as exc:
                logger.warning("Unable to rebuild as-of metadata: %s", exc)
                asof_meta = pl.DataFrame()

    if not asof_meta.is_empty():
        asof_meta = asof_meta.filter(pl.col("game_id").is_in(game_ids))
        asof_meta = asof_meta.with_columns(
            [
                pl.col("cutoff_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("cutoff_ts").cast(pl.Datetime("ms", "UTC")).alias("decision_cutoff_ts"),
                pl.col("injury_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("roster_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("odds_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
                pl.col("forecast_snapshot_ts").cast(pl.Datetime("ms", "UTC")),
            ]
        )
        pl_df = pl_df.join(asof_meta, on="game_id", how="left")

    rolling_stats = [s for s in ROLLING_FEATURE_STATS if s in pl_df.columns]
    missing_stats = sorted(set(ROLLING_FEATURE_STATS) - set(rolling_stats))
    if missing_stats:
        logger.warning("Prediction rolling features skipping missing stats: %s", ", ".join(missing_stats))

    if rolling_stats:
        enriched = add_rolling_features(
            pl_df,
            level="game",
            stats=rolling_stats,
            windows=ROLLING_WINDOWS,
            contexts=ROLLING_CONTEXTS,
            date_col="game_date",
            player_col="player_id",
            opponent_col="opponent",
        )
    else:
        enriched = pl_df

    # Rename rolling features to match model expectations (snap counts)
    # The model was trained on features named like 'snap_offense_pct_prev'
    # but standard rolling logic produces '1g_offense_pct_per_game'.
    rename_map = {
        "1g_offense_pct_per_game": "snap_offense_pct_prev",
        "3g_offense_pct_per_game": "snap_offense_pct_l3",
        "1g_offense_snaps_per_game": "snap_offense_snaps_prev",
        "1g_defense_pct_per_game": "snap_defense_pct_prev",
        "3g_defense_pct_per_game": "snap_defense_pct_l3",
        "1g_defense_snaps_per_game": "snap_defense_snaps_prev",
        "1g_st_pct_per_game": "snap_st_pct_prev",
        "3g_st_pct_per_game": "snap_st_pct_l3",
        "1g_st_snaps_per_game": "snap_st_snaps_prev",

        # Pre-snap participation/target splits (historical projections)
        "1g_ps_route_participation_pct_per_game": "ps_hist_route_participation_pct_prev",
        "3g_ps_route_participation_pct_per_game": "ps_hist_route_participation_pct_l3",
        "1g_ps_route_participation_plays_per_game": "ps_hist_route_participation_plays_prev",
        "3g_ps_route_participation_plays_per_game": "ps_hist_route_participation_plays_l3",
        "1g_ps_targets_total_per_game": "ps_hist_targets_total_prev",
        "3g_ps_targets_total_per_game": "ps_hist_targets_total_l3",
        "1g_ps_targets_slot_count_per_game": "ps_hist_targets_slot_count_prev",
        "3g_ps_targets_slot_count_per_game": "ps_hist_targets_slot_count_l3",
        "1g_ps_targets_wide_count_per_game": "ps_hist_targets_wide_count_prev",
        "3g_ps_targets_wide_count_per_game": "ps_hist_targets_wide_count_l3",
        "1g_ps_targets_inline_count_per_game": "ps_hist_targets_inline_count_prev",
        "3g_ps_targets_inline_count_per_game": "ps_hist_targets_inline_count_l3",
        "1g_ps_targets_backfield_count_per_game": "ps_hist_targets_backfield_count_prev",
        "3g_ps_targets_backfield_count_per_game": "ps_hist_targets_backfield_count_l3",
        "1g_ps_targets_slot_share_per_game": "ps_hist_targets_slot_share_prev",
        "3g_ps_targets_slot_share_per_game": "ps_hist_targets_slot_share_l3",
        "1g_ps_targets_wide_share_per_game": "ps_hist_targets_wide_share_prev",
        "3g_ps_targets_wide_share_per_game": "ps_hist_targets_wide_share_l3",
        "1g_ps_targets_inline_share_per_game": "ps_hist_targets_inline_share_prev",
        "3g_ps_targets_inline_share_per_game": "ps_hist_targets_inline_share_l3",
        "1g_ps_targets_backfield_share_per_game": "ps_hist_targets_backfield_share_prev",
        "3g_ps_targets_backfield_share_per_game": "ps_hist_targets_backfield_share_l3",
        "1g_ps_total_touches_per_game": "ps_hist_total_touches_prev",
        "3g_ps_total_touches_per_game": "ps_hist_total_touches_l3",
        "1g_ps_scripted_touches_per_game": "ps_hist_scripted_touches_prev",
        "3g_ps_scripted_touches_per_game": "ps_hist_scripted_touches_l3",
        "1g_ps_scripted_touch_share_per_game": "ps_hist_scripted_touch_share_prev",
        "3g_ps_scripted_touch_share_per_game": "ps_hist_scripted_touch_share_l3",
    }
    existing = set(enriched.columns)
    valid_renames = {k: v for k, v in rename_map.items() if k in existing}
    if valid_renames:
        enriched = enriched.rename(valid_renames)

    enriched = _apply_ps_fallback(enriched, season_hint=season_hint)

    alias_specs = [
        ("1g_target_per_game", "target_prev"),
        ("3g_target_per_game", "target_l3"),
        ("1g_carry_per_game", "carry_prev"),
        ("3g_carry_per_game", "carry_l3"),
        ("1g_pass_attempt_per_game", "pass_attempt_prev"),
        ("3g_pass_attempt_per_game", "pass_attempt_l3"),
        ("1g_red_zone_target_per_game", "red_zone_target_prev"),
        ("3g_red_zone_target_per_game", "red_zone_target_l3"),
        ("1g_red_zone_carry_per_game", "red_zone_carry_prev"),
        ("3g_red_zone_carry_per_game", "red_zone_carry_l3"),
        ("1g_goal_to_go_target_per_game", "goal_to_go_target_prev"),
        ("3g_goal_to_go_target_per_game", "goal_to_go_target_l3"),
        ("1g_goal_to_go_carry_per_game", "goal_to_go_carry_prev"),
        ("3g_goal_to_go_carry_per_game", "goal_to_go_carry_l3"),
    ]
    alias_exprs = [
        pl.col(src).cast(pl.Float32).alias(alias)
        for src, alias in alias_specs
        if src in enriched.columns and alias not in enriched.columns
    ]
    if alias_exprs:
        enriched = enriched.with_columns(alias_exprs)

    cutoff_hours = float(get_decision_cutoff_hours())
    fallback_hours = float(get_fallback_cutoff_hours())
    if "decision_cutoff_ts" in enriched.columns:
        enriched = enriched.with_columns(pl.col("decision_cutoff_ts").cast(pl.Datetime("ms")))
    elif "game_date" in enriched.columns:
        enriched = enriched.with_columns(
            pl.when(pl.col("game_start_utc").is_not_null())
            .then(pl.col("game_start_utc") - pl.duration(hours=cutoff_hours))
            .otherwise(
                pl.col("game_date")
                .cast(pl.Datetime("ms"))
                - pl.duration(hours=fallback_hours)
            )
            .alias("decision_cutoff_ts")
        )
    team_history = None
    if TEAM_CONTEXT_HISTORY_PATH.exists():
        team_history = pl.read_parquet(str(TEAM_CONTEXT_HISTORY_PATH))
        if "game_date" in team_history.columns:
            team_history = team_history.with_columns(
                pl.col("game_date").cast(pl.Datetime("ms"))
            )
    else:
        logger.warning(
            "Team context history file missing at %s; skipping team context features.",
            TEAM_CONTEXT_HISTORY_PATH,
        )
    enriched = add_team_context_features(
        enriched,
        join_on_date=True,
        history=team_history,
        cutoff_column="decision_cutoff_ts",
    )

    available_cols = set(enriched.columns)
    share_specs = [
        ("hist_target_share", "target", "team_ctx_pass_attempts"),
        ("hist_carry_share", "carry", "team_ctx_carries"),
        ("hist_pass_attempt_share", "pass_attempt", "team_ctx_pass_attempts"),
        ("hist_red_zone_target_share", "red_zone_target", "team_ctx_red_zone_targets"),
        ("hist_red_zone_carry_share", "red_zone_carry", "team_ctx_red_zone_carries"),
        ("hist_goal_to_go_target_share", "goal_to_go_target", "team_ctx_goal_to_go_targets"),
        ("hist_goal_to_go_carry_share", "goal_to_go_carry", "team_ctx_goal_to_go_carries"),
    ]
    share_exprs: list[pl.Expr] = []
    for base_name, numer_prefix, denom_prefix in share_specs:
        combos = [
            ("prev", f"{numer_prefix}_prev", f"{denom_prefix}_prev"),
            ("l3", f"{numer_prefix}_l3", f"{denom_prefix}_l3"),
        ]
        for suffix, numer_col, denom_col in combos:
            if {numer_col, denom_col} <= available_cols:
                share_exprs.append(
                    pl.when(pl.col(denom_col) > 0)
                    .then(
                        pl.col(numer_col).cast(pl.Float32)
                        / pl.col(denom_col).cast(pl.Float32)
                    )
                    .otherwise(0.0)
                    .alias(f"{base_name}_{suffix}")
                )
    if share_exprs:
        enriched = enriched.with_columns(share_exprs)

    # Mirror training-time derived pace features (team/opp/matchup aggregates)
    pace_exprs: list[pl.Expr] = []
    available_cols = set(enriched.columns)
    if "team_ctx_offensive_plays_prev" in available_cols:
        pace_exprs.append(
            pl.col("team_ctx_offensive_plays_prev").cast(pl.Float32).alias("team_pace_prev")
        )
    if "team_ctx_offensive_plays_l3" in available_cols:
        pace_exprs.append(
            pl.col("team_ctx_offensive_plays_l3").cast(pl.Float32).alias("team_pace_l3")
        )
    if "opp_ctx_offensive_plays_prev" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_offensive_plays_prev").cast(pl.Float32).alias("opp_pace_prev")
        )
    if "opp_ctx_offensive_plays_l3" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_offensive_plays_l3").cast(pl.Float32).alias("opp_pace_l3")
        )
    if {"team_ctx_offensive_plays_prev", "opp_ctx_offensive_plays_prev"} <= available_cols:
        pace_exprs.append(
            (pl.col("team_ctx_offensive_plays_prev") + pl.col("opp_ctx_offensive_plays_prev"))
            .cast(pl.Float32)
            .alias("matchup_pace_prev")
        )
        pace_exprs.append(
            (pl.col("team_ctx_offensive_plays_prev") - pl.col("opp_ctx_offensive_plays_prev"))
            .cast(pl.Float32)
            .alias("pace_diff_prev")
        )
    if {"team_ctx_offensive_plays_l3", "opp_ctx_offensive_plays_l3"} <= available_cols:
        pace_exprs.append(
            (pl.col("team_ctx_offensive_plays_l3") + pl.col("opp_ctx_offensive_plays_l3"))
            .cast(pl.Float32)
            .alias("matchup_pace_l3")
        )
        pace_exprs.append(
            (pl.col("team_ctx_offensive_plays_l3") - pl.col("opp_ctx_offensive_plays_l3"))
            .cast(pl.Float32)
            .alias("pace_diff_l3")
        )
    if "opp_ctx_red_zone_play_rate_prev" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_red_zone_play_rate_prev")
            .cast(pl.Float32)
            .alias("opp_def_red_zone_play_rate_prev")
        )
    if "opp_ctx_red_zone_play_rate_l3" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_red_zone_play_rate_l3")
            .cast(pl.Float32)
            .alias("opp_def_red_zone_play_rate_l3")
        )
    if "opp_ctx_td_per_play_prev" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_td_per_play_prev").cast(pl.Float32).alias("opp_def_td_per_play_prev")
        )
    if "opp_ctx_td_per_play_l3" in available_cols:
        pace_exprs.append(
            pl.col("opp_ctx_td_per_play_l3").cast(pl.Float32).alias("opp_def_td_per_play_l3")
        )
    if pace_exprs:
        enriched = enriched.with_columns(pace_exprs)

    enriched = add_offense_context_features_inference(
        enriched,
        history_path=OFFENSE_CONTEXT_HISTORY_PATH,
        cutoff_column="decision_cutoff_ts",
    )
    if ENABLE_WEATHER_FEATURES:
        enriched = add_weather_forecast_features_inference(
            enriched,
            cutoff_column="decision_cutoff_ts",
        )
        enriched = append_weather_context_flags(
            enriched,
            roof_col="roof",
        )
    else:
        logger.info("Skipping weather forecast enrichment for inference (disabled until coverage improves).")
    enriched = _attach_additional_context_features(enriched)
    enriched = _attach_drive_history_features(enriched)
    enriched = _attach_injury_features(enriched)
    enriched = _compute_vacancy_features(enriched)
    enriched = _compute_history_dependent_features(enriched)
    enriched = add_nfl_odds_features_to_df(
        enriched,
        player_col="player_name",
        allow_schedule_fallback=True,
        drop_schedule_rows=False,
    )
    if "spread_line" not in enriched.columns:
        enriched = enriched.with_columns(pl.lit(0.0).cast(pl.Float32).alias("spread_line"))
    else:
        enriched = enriched.with_columns(pl.col("spread_line").fill_null(0.0))
        
    if "team_implied_total" not in enriched.columns:
        enriched = enriched.with_columns(pl.lit(22.5).cast(pl.Float32).alias("team_implied_total"))
    else:
        enriched = enriched.with_columns(pl.col("team_implied_total").fill_null(22.5))

    df = enriched.to_pandas()

    def _apply_historical_shares(frame: pd.DataFrame) -> None:
        share_specs = [
            ("hist_target_share", "target", "team_ctx_pass_attempts"),
            ("hist_carry_share", "carry", "team_ctx_carries"),
            ("hist_pass_attempt_share", "pass_attempt", "team_ctx_pass_attempts"),
            ("hist_red_zone_target_share", "red_zone_target", "team_ctx_red_zone_targets"),
            ("hist_red_zone_carry_share", "red_zone_carry", "team_ctx_red_zone_carries"),
            ("hist_goal_to_go_target_share", "goal_to_go_target", "team_ctx_goal_to_go_targets"),
            ("hist_goal_to_go_carry_share", "goal_to_go_carry", "team_ctx_goal_to_go_carries"),
        ]
        cols = set(frame.columns)

        for base_name, numer_prefix, denom_prefix in share_specs:
            combos = [
                ("prev", f"{numer_prefix}_prev", f"{denom_prefix}_prev"),
                ("l3", f"{numer_prefix}_l3", f"{denom_prefix}_l3"),
            ]
            for suffix, numer_col, denom_col in combos:
                if numer_col in cols and denom_col in cols:
                    numer = pd.to_numeric(frame[numer_col], errors="coerce")
                    denom = (
                        pd.to_numeric(frame[denom_col], errors="coerce")
                        .replace(0, np.nan)
                    )
                    frame[f"{base_name}_{suffix}"] = (
                        numer / denom
                    ).fillna(0.0).astype("float32")

    _apply_historical_shares(df)

    for col in BASE_GAME_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    if "depth_chart_order" in df.columns:
        df["depth_chart_order"] = pd.to_numeric(df["depth_chart_order"], errors="coerce")
    for col in ("is_home", "game_start_hour_utc", "game_day_of_week"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return df


def _load_raw_injuries_for_predictions(seasons: Iterable[int]) -> pl.DataFrame | None:
    frames: list[pl.DataFrame] = []
    for season in sorted(set(seasons)):
        cache_path = INJURY_CACHE_DIR / f"injury_{season}.parquet"
        if cache_path.exists():
            frames.append(pl.read_parquet(cache_path))
        else:
            logger.warning("Injury cache missing for season %s at %s", season, cache_path)
    if not frames:
        return None
    out = pl.concat(frames, how="vertical_relaxed")
    if "gsis_id" in out.columns and "player_id" not in out.columns:
        out = out.rename({"gsis_id": "player_id"})
    return out


def _attach_drive_history_features(enriched: pl.DataFrame) -> pl.DataFrame:
    if "game_date" not in enriched.columns or "player_id" not in enriched.columns:
        return enriched

    if not PLAYER_DRIVE_HISTORY_PATH.exists():
        logger.warning(
            "Drive history file missing at %s; skipping drive context features.",
            PLAYER_DRIVE_HISTORY_PATH,
        )
        return enriched

    drive_history = pl.read_parquet(str(PLAYER_DRIVE_HISTORY_PATH))
    if drive_history.is_empty():
        return enriched

    drive_history = drive_history.sort(["player_id", "team", "game_date"])
    drive_history = drive_history.with_columns(
        pl.col("game_date").cast(pl.Datetime("ms"))
    )
    if "data_as_of" in drive_history.columns:
        drive_history = drive_history.with_columns(pl.col("data_as_of").cast(pl.Datetime("ms")))

    join_keys = ["player_id", "team"] if "team" in enriched.columns else ["player_id"]
    target_col = "decision_cutoff_ts" if "decision_cutoff_ts" in enriched.columns else "game_date"
    if target_col in enriched.columns and enriched.schema.get(target_col) == pl.Datetime("ms", "UTC"):
        enriched = enriched.with_columns(
            pl.col(target_col).dt.replace_time_zone(None).alias(target_col)
        )
    drive_join = (
        drive_history
        .drop_nulls("data_as_of")
        .with_columns(pl.col("data_as_of").alias("__drive_ctx_data_as_of"))
        .sort(join_keys + ["__drive_ctx_data_as_of"])
    )
    enriched = enriched.with_columns(
        pl.col(target_col).cast(pl.Datetime("ms")).alias("__drive_ctx_target_ts")
        if target_col in enriched.columns
        else pl.col("game_date").cast(pl.Datetime("ms")).alias("__drive_ctx_target_ts")
    ).join_asof(
        drive_join,
        left_on="__drive_ctx_target_ts",
        right_on="__drive_ctx_data_as_of",
        by=join_keys,
        strategy="backward",
    ).rename({"__drive_ctx_data_as_of": "drive_ctx_data_as_of"}).drop("__drive_ctx_target_ts")

    drive_cols = [col for col in enriched.columns if col.startswith("drive_hist_")]
    if drive_cols:
        enriched = enriched.with_columns(
            [pl.col(col).fill_null(0.0).cast(pl.Float32) for col in drive_cols]
        )
    if "drive_ctx_data_as_of" in enriched.columns:
        enriched = enriched.with_columns(pl.col("drive_ctx_data_as_of").cast(pl.Datetime("ms")))

    return enriched


def _compute_vacancy_features(enriched: pl.DataFrame) -> pl.DataFrame:
    """
    Compute vacancy features for inference using current rolling stats.
    Vacancy features: Sum of historical usage (3g avg) from players who are OUT.
    """
    vacancy_cols = [
        "vacated_targets_position",
        "vacated_carries_position",
        "vacated_rz_targets_position",
        "vacated_gl_carries_position",
    ]
    for col in vacancy_cols:
        if col not in enriched.columns:
            enriched = enriched.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))
            
    if "injury_game_designation" not in enriched.columns:
        return enriched

    # Find OUT players
    out_mask = pl.col("injury_game_designation").cast(pl.Utf8).str.to_uppercase() == "OUT"
    
    agg_exprs = []
    if "3g_target_per_game" in enriched.columns:
        agg_exprs.append(pl.col("3g_target_per_game").sum().alias("vacated_targets_position"))
    elif "target_l3" in enriched.columns: 
        agg_exprs.append(pl.col("target_l3").sum().alias("vacated_targets_position"))
        
    if "3g_carry_per_game" in enriched.columns:
        agg_exprs.append(pl.col("3g_carry_per_game").sum().alias("vacated_carries_position"))
    elif "carry_l3" in enriched.columns:
        agg_exprs.append(pl.col("carry_l3").sum().alias("vacated_carries_position"))
        
    if "3g_red_zone_target_per_game" in enriched.columns:
        agg_exprs.append(pl.col("3g_red_zone_target_per_game").sum().alias("vacated_rz_targets_position"))
    elif "red_zone_target_l3" in enriched.columns:
        agg_exprs.append(pl.col("red_zone_target_l3").sum().alias("vacated_rz_targets_position"))
        
    if "3g_goal_to_go_carry_per_game" in enriched.columns:
        agg_exprs.append(pl.col("3g_goal_to_go_carry_per_game").sum().alias("vacated_gl_carries_position"))
    elif "goal_to_go_carry_l3" in enriched.columns:
        agg_exprs.append(pl.col("goal_to_go_carry_l3").sum().alias("vacated_gl_carries_position"))
        
    if not agg_exprs:
        return enriched
        
    vacated_usage = (
        enriched.filter(out_mask)
        .group_by(["season", "week", "team", "position"])
        .agg(agg_exprs)
    )
    
    if vacated_usage.is_empty():
        return enriched
        
    enriched = enriched.join(
        vacated_usage,
        on=["season", "week", "team", "position"],
        how="left",
        suffix="_vac"
    )
    
    for col in vacancy_cols:
        vac_col = f"{col}_vac"
        if vac_col in enriched.columns:
            enriched = enriched.with_columns(
                pl.col(vac_col).fill_null(0.0).cast(pl.Float32).alias(col)
            ).drop(vac_col)
            
    return enriched


def _apply_injury_availability_model_inference(enriched: pl.DataFrame) -> pl.DataFrame:
    """Attach injury availability model outputs if artifacts are available."""
    try:
        artifact = load_injury_artifact()
    except Exception as exc:
        logger.warning("Failed to load injury availability artifact: %s", exc)
        return enriched

    if artifact is None:
        logger.warning("Injury availability artifact missing; skipping probability enrichment.")
        return enriched

    try:
        proba_df = predict_injury_probabilities(enriched, artifact)
    except Exception as exc:
        logger.warning("Injury availability inference failed: %s", exc)
        return enriched

    if proba_df.is_empty():
        return enriched

    proba_cols = proba_df.columns
    enriched = enriched.drop([col for col in proba_cols if col in enriched.columns], strict=False)
    enriched = pl.concat([enriched, proba_df], how="horizontal")

    width_col = "injury_inactive_probability_interval_width"
    if {MODEL_PROB_COL, PROB_LOW_COL, PROB_HIGH_COL, width_col}.issubset(set(enriched.columns)) and "injury_inactive_probability" in enriched.columns:
        fallback_expr = pl.col(width_col) > INJURY_MODEL_FALLBACK_INTERVAL_THRESHOLD
        enriched = enriched.with_columns(
            [
                pl.when(fallback_expr)
                .then(pl.lit("heuristic"))
                .otherwise(pl.lit("model"))
                .alias("injury_inactive_probability_source"),
                pl.when(fallback_expr)
                .then(pl.col("injury_inactive_probability"))
                .otherwise(pl.col(MODEL_PROB_COL))
                .alias("injury_inactive_probability"),
                pl.when(fallback_expr)
                .then(pl.col("injury_inactive_probability"))
                .otherwise(pl.col(PROB_LOW_COL))
                .alias(PROB_LOW_COL),
                pl.when(fallback_expr)
                .then(pl.col("injury_inactive_probability"))
                .otherwise(pl.col(PROB_HIGH_COL))
                .alias(PROB_HIGH_COL),
            ]
        )
    elif "injury_inactive_probability_source" not in enriched.columns:
        enriched = enriched.with_columns(pl.lit("model").alias("injury_inactive_probability_source"))

    return enriched


def _compute_history_dependent_features(enriched: pl.DataFrame) -> pl.DataFrame:
    """
    Compute features requiring historical player state by loading processed history.
    - Injury history rates (cumulative)
    - Inactivity streaks
    """
    req_cols = [
        "season", "week", "player_id", "team", "depth_chart_position",
        "injury_practice_status_sequence", "injury_is_inactive_designation",
        "status"
    ]
    
    try:
        # Use manual file iteration to handle schema evolution (e.g. int32 vs dictionary encoded season)
        files = sorted(PLAYER_GAME_DIR.glob("season=*/week=*/part.parquet"))
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f, columns=req_cols))
            except Exception:
                continue
        
        if not dfs:
            return enriched
            
        history_pd = pd.concat(dfs, ignore_index=True)
        
        # Enforce types for critical join keys
        if "season" in history_pd.columns:
            history_pd["season"] = history_pd["season"].astype("int32")
        if "week" in history_pd.columns:
            history_pd["week"] = history_pd["week"].astype("int32")
            
        history = pl.from_pandas(history_pd)
    except Exception as e:
        logger.warning(f"Failed to load player history: {e}")
        return enriched

    if history.is_empty():
        return enriched

    targets = enriched.select(["season", "week"]).unique()
    if not targets.is_empty():
        history = history.join(targets, on=["season", "week"], how="anti")

    if "injury_is_inactive_designation" not in enriched.columns:
        enriched = enriched.with_columns(
            pl.when(pl.col("injury_game_designation").cast(pl.Utf8).str.to_uppercase() == "OUT")
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias("injury_is_inactive_designation")
        )
        
    if "status" not in enriched.columns:
        enriched = enriched.with_columns(pl.lit("ACT").alias("status"))
        
    common_cols = [c for c in history.columns if c in enriched.columns]
    current = enriched.select(common_cols)
    history_subset = history.select(common_cols)
    
    combined = pl.concat([history_subset, current], how="vertical_relaxed")
    
    combined = _compute_injury_history_rates(combined)
    
    if "status" in combined.columns:
        combined = combined.with_columns(
            (pl.col("status") == "INA").cast(pl.Int8).alias("is_inactive")
        )
        combined = combined.with_columns(pl.col("is_inactive").fill_null(0))
        
        combined = combined.sort(["player_id", "season", "week"])
        
        combined = combined.with_columns([
            pl.col("is_inactive")
            .rolling_sum(window_size=5, min_periods=1)
            .shift(1)
            .over("player_id")
            .fill_null(0)
            .alias("recent_inactivity_count"),
            
            pl.col("is_inactive")
            .shift(1)
            .over("player_id")
            .fill_null(0)
            .alias("was_inactive_last_game")
        ])
    
    features_to_join = [
        "injury_player_inactive_rate_prior",
        "injury_depth_slot_inactive_rate_prior",
        "injury_practice_pattern_inactive_rate_prior",
        "recent_inactivity_count",
        "was_inactive_last_game"
    ]
    
    features = combined.select(["season", "week", "player_id", "team"] + [f for f in features_to_join if f in combined.columns])
    
    enriched = enriched.join(
        features,
        on=["season", "week", "player_id", "team"],
        how="left"
    )
    enriched = _apply_injury_availability_model_inference(enriched)
    
    injury_defaults = {
        "injury_hours_since_last_report": 168.0,
        "injury_hours_until_game_at_last_report": 0.0,
        "injury_hours_between_last_reports": 0.0,
        "injury_report_count": 0.0,
        "injury_transaction_days_since": 365.0,
        "injury_snapshot_valid": 0,
        "injury_snapshot_ts_missing": 1,
    }
    for col, val in injury_defaults.items():
        if col not in enriched.columns:
             enriched = enriched.with_columns(pl.lit(val).alias(col))
    if "injury_last_transaction_note" not in enriched.columns:
        enriched = enriched.with_columns(pl.lit("UNKNOWN").alias("injury_last_transaction_note"))

    if "off_ctx_game_date" not in enriched.columns:
        enriched = enriched.with_columns(pl.col("game_date").alias("off_ctx_game_date"))
        
    return enriched


def _attach_injury_features(enriched: pl.DataFrame) -> pl.DataFrame:
    required = {"season", "week", "player_id"}
    if not required <= set(enriched.columns):
        return enriched

    seasons = enriched.get_column("season").unique().to_list()
    raw = _load_raw_injuries_for_predictions(seasons)
    if raw is None or raw.is_empty():
        return enriched

    keys = enriched.select(["season", "week"]).unique()
    meta_cols = [c for c in ["season", "week", "team", "game_date", "game_start_utc"] if c in enriched.columns]
    game_meta = (
        enriched.select(meta_cols)
        .with_columns(
            [
                pl.col("team").cast(pl.Utf8) if "team" in meta_cols else pl.lit(None).cast(pl.Utf8),
                pl.col("game_date").cast(pl.Date) if "game_date" in meta_cols else pl.lit(None).cast(pl.Date),
                pl.col("game_start_utc").cast(pl.Datetime("ms", "UTC")) if "game_start_utc" in meta_cols else pl.lit(None).cast(pl.Datetime("ms", "UTC")),
            ]
        )
        .unique()
    )
    available_cols = [c for c in [
        "season",
        "week",
        "team",
        "player_id",
        "report_status",
        "practice_status",
        "report_primary_injury",
        "practice_primary_injury",
        "reported_at",
    ] if c in raw.columns]
    injury_pl = raw.select(available_cols)
    if "player_id" not in injury_pl.columns:
        return enriched

    cast_exprs: list[pl.Expr] = [
        pl.col("player_id").cast(pl.Utf8),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
    ]
    if "team" in injury_pl.columns:
        cast_exprs.append(
            pl.col("team")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.to_uppercase()
            .alias("team")
        )
    if "report_status" in injury_pl.columns:
        cast_exprs.append(
            pl.col("report_status")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.replace_all(r"\s+", " ", literal=False)
            .str.to_uppercase()
            .alias("report_status_clean")
        )
    if "practice_status" in injury_pl.columns:
        cast_exprs.append(
            pl.col("practice_status")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.replace_all(r"\s+", " ", literal=False)
            .str.to_uppercase()
            .alias("practice_status_clean")
        )
    if "report_primary_injury" in injury_pl.columns:
        cast_exprs.append(
            pl.col("report_primary_injury")
            .cast(pl.Utf8)
            .str.strip_chars()
            .alias("report_primary_injury")
        )
    if "practice_primary_injury" in injury_pl.columns:
        cast_exprs.append(
            pl.col("practice_primary_injury")
            .cast(pl.Utf8)
            .str.strip_chars()
            .alias("practice_primary_injury")
        )
    if "reported_at" in injury_pl.columns:
        cast_exprs.append(
            pl.col("reported_at")
            .cast(pl.Datetime(time_unit="ns", time_zone="UTC"))
            .alias("reported_at")
        )
    else:
        cast_exprs.append(pl.lit(None).cast(pl.Datetime("ns", "UTC")).alias("reported_at"))

    injury_pl = injury_pl.with_columns(cast_exprs)
    injury_pl = injury_pl.join(keys, on=["season", "week"], how="inner")
    if "team" in injury_pl.columns and not game_meta.is_empty():
        injury_pl = injury_pl.join(
            game_meta,
            on=[c for c in ["season", "week", "team"] if c in injury_pl.columns and c in game_meta.columns],
            how="left",
        )
    else:
        injury_pl = injury_pl.with_columns(
            [
                pl.lit(None).cast(pl.Date).alias("game_date"),
                pl.lit(None).cast(pl.Datetime("ms", "UTC")).alias("game_start_utc"),
            ]
        )
    cutoff_hours = float(get_decision_cutoff_hours())
    fallback_hours = float(get_fallback_cutoff_hours())

    injury_pl = injury_pl.with_columns(
        pl.when(pl.col("game_start_utc").is_not_null())
        .then(pl.col("game_start_utc") - pl.duration(hours=cutoff_hours))
        .otherwise(
            pl.when(pl.col("game_date").is_not_null())
            .then(
                pl.col("game_date")
                .cast(pl.Datetime("ms", "UTC"))
                - pl.duration(hours=fallback_hours)
            )
            .otherwise(None)
        )
        .alias("decision_cutoff_ts")
    )
    injury_pl = injury_pl.filter(
        pl.col("decision_cutoff_ts").is_null()
        | pl.col("reported_at").is_null()
        | (pl.col("reported_at") <= pl.col("decision_cutoff_ts"))
    )
    injury_pl = injury_pl.drop([col for col in ["decision_cutoff_ts"] if col in injury_pl.columns])
    injury_pl = injury_pl.filter(pl.col("player_id").is_not_null())
    injury_pl = injury_pl.sort(["player_id", "season", "week", "reported_at"])
    injury_pl = injury_pl.with_columns([
        pl.col("practice_status_clean")
        .fill_null("")
        .str.contains("DID NOT PARTICIPATE")
        .cast(pl.Int8)
        .alias("practice_dnp_flag"),
        pl.col("practice_status_clean")
        .fill_null("")
        .str.contains("LIMITED")
        .cast(pl.Int8)
        .alias("practice_limited_flag"),
        pl.col("practice_status_clean")
        .fill_null("")
        .str.contains("FULL")
        .cast(pl.Int8)
        .alias("practice_full_flag"),
    ])

    aggregated = (
        injury_pl
        .group_by(["season", "week", "player_id"])
        .agg(
            [
                pl.len().alias("_injury_rows"),
                pl.col("report_status_clean").last().alias("injury_report_status"),
                pl.col("practice_status_clean").last().alias("injury_practice_status"),
                pl.col("report_primary_injury").last().alias("injury_report_primary"),
                pl.col("practice_primary_injury").last().alias("injury_practice_primary"),
                pl.col("practice_dnp_flag").fill_null(0).sum().alias("injury_practice_dnp_count"),
                pl.col("practice_limited_flag").fill_null(0).sum().alias("injury_practice_limited_count"),
                pl.col("practice_full_flag").fill_null(0).sum().alias("injury_practice_full_count"),
                pl.col("practice_status_clean").drop_nulls().alias("_injury_practice_status_seq"),
                pl.col("report_status_clean").drop_nulls().alias("_injury_report_status_seq"),
            ]
        )
    )

    if aggregated.is_empty():
        return enriched

    aggregated = aggregated.with_columns(
        [
            pl.when(pl.col("_injury_rows") > 0).then(1.0).otherwise(None).alias("injury_is_listed"),
        ]
    )
    aggregated = aggregated.with_columns(
        [
            pl.col("_injury_practice_status_seq").list.get(0, null_on_oob=True).alias("injury_practice_status_day1"),
            pl.col("_injury_practice_status_seq").list.get(1, null_on_oob=True).alias("injury_practice_status_day2"),
            pl.col("_injury_practice_status_seq").list.get(2, null_on_oob=True).alias("injury_practice_status_day3"),
            pl.col("_injury_practice_status_seq").list.join(">").alias("injury_practice_status_sequence"),
            pl.col("_injury_report_status_seq").list.join(">").alias("injury_report_status_sequence"),
            pl.col("_injury_report_status_seq").list.get(-1, null_on_oob=True).alias("injury_game_designation"),
        ]
    ).drop(["_injury_practice_status_seq", "_injury_report_status_seq", "_injury_rows"])
    aggregated = aggregated.with_columns(
        [
            pl.col("injury_game_designation")
            .is_in(["OUT", "DOUBTFUL", "INACTIVE"])
            .cast(pl.Float32)
            .alias("injury_is_inactive_designation")
        ]
    )
    report_prob = (
        pl.when(pl.col("injury_report_status") == "OUT")
        .then(0.98)
        .when(pl.col("injury_report_status") == "DOUBTFUL")
        .then(0.85)
        .when(pl.col("injury_report_status") == "QUESTIONABLE")
        .then(0.55)
        .when(pl.col("injury_report_status") == "SUSPENDED")
        .then(0.9)
        .otherwise(0.1)
    )
    practice_prob = (
        pl.when(pl.col("injury_practice_status") == "DID NOT PARTICIPATE")
        .then(0.8)
        .when(pl.col("injury_practice_status") == "LIMITED")
        .then(0.5)
        .when(pl.col("injury_practice_status") == "FULL")
        .then(0.1)
        .otherwise(0.15)
    )
    aggregated = aggregated.with_columns(
        [
            report_prob.cast(pl.Float32).alias("_prob_report"),
            practice_prob.cast(pl.Float32).alias("_prob_practice"),
        ]
    )
    aggregated = aggregated.with_columns(
        pl.max_horizontal(pl.col("_prob_report"), pl.col("_prob_practice")).alias("injury_inactive_probability")
    )
    aggregated = aggregated.with_columns(
        pl.when(pl.col("injury_practice_dnp_count").fill_null(0) >= 2)
        .then(
            pl.when((pl.col("injury_inactive_probability") + 0.1) > 1.0)
            .then(1.0)
            .otherwise(pl.col("injury_inactive_probability") + 0.1)
        )
        .otherwise(pl.col("injury_inactive_probability"))
        .alias("injury_inactive_probability")
    )
    aggregated = aggregated.with_columns(
        [
            pl.col("injury_is_listed").cast(pl.Float32),
            pl.col("injury_practice_dnp_count").cast(pl.Float32),
            pl.col("injury_practice_limited_count").cast(pl.Float32),
            pl.col("injury_practice_full_count").cast(pl.Float32),
        ]
    ).drop(["_prob_report", "_prob_practice"])
    aggregated = aggregated.with_columns(
        pl.col("injury_inactive_probability").fill_null(0.1).cast(pl.Float32)
    )

    return enriched.join(aggregated, on=["season", "week", "player_id"], how="left")


def _load_artifacts(problem_name: str) -> dict:
    path = MODEL_DIR / f"inference_artifacts_{problem_name}.joblib"
    if not path.exists():
        if problem_name == "anytime_td":
            alt = MODEL_DIR / "inference_artifacts_anytime_td_meta.joblib"
            if alt.exists():
                logger.warning(
                    "Using legacy anytime_td_meta inference artifacts for anytime_td."
                )
                return joblib.load(alt)
        raise FileNotFoundError(
            f"Inference artifacts not found at {path}. "
            "Run the training pipeline first."
        )
    return joblib.load(path)


def _prepare_feature_matrix(df: pd.DataFrame, artifacts: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = artifacts["feature_columns"]
    cat_levels = artifacts.get("category_levels", {})
    categorical_features = artifacts.get("categorical_features", [])

    meta = df.reindex(columns=META_COLUMNS, fill_value=None).copy()
    X = df.reindex(columns=feature_cols, fill_value=np.nan)

    # Convert datetime columns to numeric epoch micros (float) to mirror training prep
    datetime_cols = X.select_dtypes(include=["datetime", "datetimetz"]).columns
    for col in datetime_cols:
        col_as_dt = pd.to_datetime(X[col], utc=True, errors="coerce")
        numeric = col_as_dt.view("int64").astype("float64", copy=False)
        numeric[col_as_dt.isna().to_numpy()] = np.nan
        X[col] = numeric / 1_000_000.0

    for col, levels in cat_levels.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=levels)

    # Ensure categorical dtype for any recorded categorical columns (even if no levels cached)
    for col in categorical_features:
        if col in X.columns and not pd.api.types.is_categorical_dtype(X[col]):
            X[col] = pd.Categorical(X[col])

    numeric_cols = X.select_dtypes(include=["int64", "int32", "float64"]).columns
    if len(numeric_cols):
        X[numeric_cols] = X[numeric_cols].astype("float32")

    return X, meta


def _latest_model_path(problem: str) -> Path:
    names_to_try = [problem]
    if problem == "anytime_td":
        names_to_try.append("anytime_td_meta")
    for name in names_to_try:
        problem_dir = MODEL_DIR / name / "xgboost"
        if not problem_dir.exists():
            continue
        candidates = [d for d in problem_dir.iterdir() if d.is_dir()]
        if not candidates:
            continue
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        model_path = latest / "model.joblib"
        if model_path.exists():
            if name != problem:
                logger.warning("Using legacy model directory for %s located at %s", problem, latest)
            return model_path
    raise FileNotFoundError(f"No model directory found for {problem}.")


def _metrics_path(problem: str) -> Path:
    names_to_try = [problem]
    if problem == "anytime_td":
        names_to_try.append("anytime_td_meta")
    for name in names_to_try:
        problem_dir = METRICS_DIR / name / "xgboost"
        if not problem_dir.exists():
            continue
        runs = [d for d in problem_dir.iterdir() if d.is_dir()]
        if not runs:
            continue
        latest = max(runs, key=lambda p: p.stat().st_mtime)
        path = latest / "metrics.yaml"
        if path.exists():
            if name != problem:
                logger.warning("Using legacy metrics directory for %s located at %s", problem, latest)
            return path
    return Path()


def _load_threshold(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return 0.5
    with metrics_path.open("r") as fp:
        metrics = yaml.safe_load(fp)
    return float(metrics.get("decision_threshold", 0.5))

def _apply_guards_inline(features_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
     adj = preds.copy()
     
     # 1. Inactive / Out
     if "injury_is_inactive_designation" in features_df.columns:
         mask = pd.to_numeric(features_df["injury_is_inactive_designation"], errors="coerce").fillna(0).astype(bool).to_numpy()
         adj[mask] = 0.0
     
     if "injury_game_designation" in features_df.columns:
         des = features_df["injury_game_designation"].fillna("").astype(str).str.upper()
         out_mask = des.isin(["OUT", "INACTIVE"]).to_numpy()
         adj[out_mask] = 0.0
         
         doubtful_mask = (des == "DOUBTFUL").to_numpy()
         adj[doubtful_mask] *= 0.25
         
         q_mask = (des == "QUESTIONABLE").to_numpy()
         adj[q_mask] *= 0.85
     
     if "injury_practice_status" in features_df.columns:
         prac = features_df["injury_practice_status"].fillna("").astype(str).str.upper()
         dnp_mask = prac.str.contains("DID NOT PARTICIPATE", regex=False).to_numpy()
         adj[dnp_mask] *= 0.55
         
         lim_mask = prac.str.contains("LIMITED", regex=False).to_numpy()
         adj[lim_mask] *= 0.85
     
     return np.clip(adj, 0.0, 1.0)

def _apply_availability_floor(features_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Apply minimum availability floor for active players with missing history."""
    adj = preds.copy()
    
    if "snap_offense_pct_l3" in features_df.columns:
         history = pd.to_numeric(features_df["snap_offense_pct_l3"], errors="coerce").fillna(0)
         no_history_mask = (history == 0).to_numpy()
         
         # Determine active status
         if "injury_is_inactive_designation" in features_df.columns:
             inactive = pd.to_numeric(features_df["injury_is_inactive_designation"], errors="coerce").fillna(0).astype(bool).to_numpy()
             active_mask = ~inactive
         else:
             active_mask = np.ones(len(adj), dtype=bool)
             
         # Apply floor to active players with no history
         if "position" in features_df.columns:
             is_qb = (features_df["position"] == "QB").to_numpy()
             qb_floor_mask = no_history_mask & active_mask & is_qb
             other_floor_mask = no_history_mask & active_mask & (~is_qb)
             
             adj[qb_floor_mask] = np.maximum(adj[qb_floor_mask], 0.9) # Starters usually play 100%
             adj[other_floor_mask] = np.maximum(adj[other_floor_mask], 0.35)
         else:
             floor_mask = no_history_mask & active_mask
             adj[floor_mask] = np.maximum(adj[floor_mask], 0.35)
             
    return adj

def _predict_for_problem(features: pd.DataFrame, problem_config: dict, artifacts: dict, threshold: float = 0.5) -> np.ndarray:
    """Load the trained model and return probability or point forecasts based on task metadata."""
    problem_name = problem_config.get("name", "<unknown>")
    task_type = str(
        problem_config.get("task_type")
        or artifacts.get("task_type")
        or ""
    ).lower()
    output_mode = artifacts.get("output_mode") or problem_config.get("output_mode")

    X, _ = _prepare_feature_matrix(features, artifacts)
    model = joblib.load(_latest_model_path(problem_name))

    is_classification = task_type in {"classification", "binary", "multiclass"}
    if is_classification:
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)
            if preds.ndim > 1 and preds.shape[1] > 1:
                preds = preds[:, 1]
        else:
            preds = model.predict(X)
        preds = np.asarray(preds, dtype=float)
        preds = np.clip(preds, 0.0, 1.0)
    else:
        preds = np.asarray(model.predict(X), dtype=float)

    if output_mode and output_mode == "logit":
        preds = 1.0 / (1.0 + np.exp(-preds))

    return preds

def main() -> None:
    _ensure_imports_ready()
    args = _parse_args()
    start_date, end_date = _resolve_dates(args.date, args.days)
    config = _load_config()
    problems = config.get("problems", [])

    with decision_cutoff_override(
        cutoff_hours=args.decision_cutoff_hours,
        fallback_hours=args.fallback_cutoff_hours,
    ):
        active_cutoff = get_decision_cutoff_hours()
        print(
            f"Generating predictions for {start_date} → {end_date} "
            f"(cutoff {active_cutoff:.2f}h before kickoff)"
        )
        games = _load_schedule(start_date, end_date)
        roster = _load_rosters(games["season"].unique())
        scaffold = _build_scaffold(games, roster)

        _ensure_injury_cache(games["season"].unique())
        _ensure_weather_cache(games["season"].unique())
        _ensure_odds_cache(games["season"].unique(), start_date=start_date, end_date=end_date)

        features = _compute_features(scaffold)
        
        # Sequential Prediction Loop
        for problem in problems:
            p_name = problem["name"]
            p_target = problem.get("target_col")
            
            # We need to map problem name to prediction column name
            # Convention: pred_{p_name} unless it is specific (like team_pace)
            # Actually, training pipeline likely uses pred_{p_name} or pred_{target_col}?
            # Let's check train.py. 
            # In train.py: 
            #   predictions[f"pred_{problem['name']}"] = ...
            # So we should use pred_{p_name}.
            pred_col = f"pred_{p_name}"
            
            logger.info("Predicting %s -> %s", p_name, pred_col)
            
            try:
                artifacts = _load_artifacts(p_name)
            except FileNotFoundError:
                logger.warning("Skipping %s (artifacts not found).", p_name)
                continue

            preds = _predict_for_problem(features, problem, artifacts)
            
            # Apply guards for availability components immediately
            if p_name == "availability_active":
                features["pred_availability_active_raw"] = preds.copy()
                preds = _apply_guards_inline(features, preds)
                preds = _apply_availability_floor(features, preds)
                preds = np.clip(preds, 0.0, 1.0)
            elif p_name == "availability_snapshare":
                preds = np.clip(preds, 0.0, 1.0)
            elif p_name == "pre_snap_routes":
                preds = np.clip(preds, 0.0, 1.0)
            elif p_name == "pre_snap_scripted_touches":
                preds = np.clip(preds, 0.0, None)

            features[pred_col] = preds
            
            # Debug: Print prediction stats
            print(f"--- {pred_col} stats ---")
            print(features[pred_col].describe().T[['mean', 'min', '50%', 'max']])
            print("------------------------")
            
            if p_name == "availability":
                 print("--- Availability Feature Inputs (Head) ---")
                 cols = ["player_name", "snap_offense_pct_prev", "snap_offense_pct_l3", "injury_is_listed", "depth_chart_order", "depth_chart_position"]
                 print(features[[c for c in cols if c in features.columns]].head())
                 print("------------------------------------------")

            if p_name == "efficiency_tds":
                features = _inject_composed_features(features)

            # If team_pace, also map to pred_team_pace if p_name != team_pace (unlikely)
            # The feature injection expects 'pred_team_pace'?
            # _inject_composed_features uses 'pred_usage_targets', 'pred_efficiency_tds'.
            # It DOES NOT seem to use 'pred_team_pace' directly in injection, 
            # but the Usage models use it as input.
            
            # IMPORTANT: Inject composed features incrementally so next models can use them
        features = _inject_composed_features(features)

        # Debug: composed features
        debug_cols = ["expected_opportunities", "expected_td_signal", "pred_anytime_td"]
        print("\n--- Composed Feature Stats ---")
        print(features[[c for c in debug_cols if c in features.columns]].describe().T[['mean', 'min', '50%', 'max']])
        print("------------------------------\n")

        final_prob_col = "pred_anytime_td"
        if final_prob_col not in features.columns:
            if "pred_anytime_td_meta" in features.columns:
                logger.warning(
                    "pred_anytime_td not found; falling back to legacy pred_anytime_td_meta."
                )
                final_prob_col = "pred_anytime_td_meta"
            else:
                logger.error("Final prediction column %s not found in features.", final_prob_col)
                return

        # Prepare final output dataframe
        # We need to re-score the final model to get "implied_decimal_odds" and "prediction" boolean
        # actually we already have the probability in features[final_prob_col].
        
        threshold = _load_threshold(_metrics_path("anytime_td")) # Default to anytime_td metrics
        proba = features[final_prob_col].to_numpy()
        
        # Apply final guards (optional, but good for safety)
        # Using meta columns for guards
        meta_final = features[META_COLUMNS].copy() # Ensure we have meta cols
        proba = _apply_guards_inline(features, proba) 
        
        picks = proba >= threshold
        
        extra_cols = [
            col
            for col in features.columns
            if col not in META_COLUMNS
            and (
                col in OUTPUT_COLUMN_WHITELIST
                or any(col.startswith(prefix) for prefix in OUTPUT_COLUMN_PREFIXES)
            )
        ]
        out = features[META_COLUMNS + extra_cols].copy()
        out["prob_anytime_td"] = proba
        out["prediction"] = picks.astype(int)
        out["implied_decimal_odds"] = np.where(proba > 0, 1.0 / proba, np.nan)
        out["model_threshold"] = threshold
        
        out.sort_values(
            ["prob_anytime_td", "game_date", "team", "player_name"],
            ascending=[False, True, True, True],
            inplace=True,
        )

        output_name = (
            args.output
            if args.output
            else f"anytime_td_predictions_{start_date.isoformat()}_{end_date.isoformat()}.csv"
        )
        output_path = PREDICTION_DIR / output_name
        out.to_csv(output_path, index=False)

        positive_rate = out["prediction"].mean() * 100.0
        print(f"Wrote {len(out)} rows → {output_path}")
        print(f"Positive predictions: {out['prediction'].sum()} ({positive_rate:.2f}% of slate)")
        
        # Debug: Show top 5
        print("\nTop 5 Predictions:")
        print(out[["player_name", "team", "prob_anytime_td", "implied_decimal_odds"]].head(5))


if __name__ == "__main__":
    main()
