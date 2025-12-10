"""
Consolidated NFL Odds Module
============================

This module provides all odds-related functionality for the NFL prediction pipeline:

1. OddsAPIClient - HTTP client for The Odds API with caching and retry logic
2. Game-level odds features (moneyline, spread, totals) from nfl_odds.py
3. Player-level odds features (anytime TD props) from odds_features.py
4. Odds snapshot collection and persistence from odds_snapshots.py

All odds logic is consolidated here to prevent drift and ensure consistency.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import requests

from utils.general.paths import CACHE_ROOT, ODDS_SNAPSHOT_DIR, PLAYER_ODDS_RAW_DIR
from utils.general.constants import normalize_team as _normalize_team
from utils.general.config import get_odds_api_key, get_odds_snapshot_config

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

SPORT_KEY = "americanfootball_nfl"
DEFAULT_REGIONS = "us"  # US-facing books have the most NFL coverage

_ODDS_SNAPSHOT_CFG = get_odds_snapshot_config() or {}

SNAPSHOT_HOUR_UTC = int(_ODDS_SNAPSHOT_CFG.get("fixed_hour_utc", 12))  # default seed hour
ODDS_FALLBACK_HOURS = float(_ODDS_SNAPSHOT_CFG.get("fallback_hours_before_kickoff", 36.0))
ODDS_SNAPSHOT_MODE = str(_ODDS_SNAPSHOT_CFG.get("mode", "cutoff")).lower()

OPEN_START_HOUR_UTC = 8  # Earliest hour to probe for an "open" snapshot

# Earliest reliable date for odds snapshots in archive
ODDS_START_DATE = dt.date(2021, 9, 1)

# Snapshot suffix mapping for multiple time horizons
SNAPSHOT_SUFFIXES = OrderedDict([
    ("cutoff", ""),
    ("2h", "_2h"),
    ("6h", "_6h"),
    ("24h", "_24h"),
    ("open", "_open"),
])
SNAPSHOT_SEQUENCE = list(SNAPSHOT_SUFFIXES.keys())
DELTA_SEQUENCE = ["open", "24h", "6h", "2h"]
RELATIVE_HOURS = {"2h": 2, "6h": 6, "24h": 24}

OPEN_LOOKBACK_HOURS: Tuple[int, ...] = (168, 144, 120, 96, 72, 48, 36, 30, 24, 18, 12, 8, 6)

# Game-level metrics captured at each snapshot
SNAPSHOT_BASE_METRICS = [
    "moneyline_home", "moneyline_away", "spread_line", "spread_home_price",
    "spread_away_price", "total_line", "total_over_price", "total_under_price",
    "implied_prob_home", "implied_prob_away", "moneyline_vig",
    "moneyline_book_count", "moneyline_price_std", "spread_book_count",
    "spread_line_std", "spread_price_std", "total_book_count",
    "total_line_std", "total_price_std", "odds_hours_before_kickoff",
]

DELTA_METRICS = [
    "moneyline_home", "moneyline_away", "spread_line", "spread_home_price",
    "spread_away_price", "total_line", "total_over_price", "total_under_price",
    "implied_prob_home", "implied_prob_away", "moneyline_vig",
]

TEAM_TOTAL_METRICS = ["team_implied_total", "opp_implied_total"]

ADDITIONAL_COLUMNS = [
    "moneyline_move_max_abs", "moneyline_move_direction",
    "spread_line_move_max_abs", "spread_line_move_direction",
    "total_line_move_max_abs", "total_line_move_direction",
    "team_implied_total_move_max_abs", "team_implied_total_move_direction",
    "opp_implied_total_move_max_abs", "opp_implied_total_move_direction",
]

# Player market keys for props collection
PLAYER_MARKET_KEYS: Tuple[str, ...] = ("player_anytime_td", "player_tds_over")
PLAYER_MARKETS_PARAM = ",".join(PLAYER_MARKET_KEYS)
PLAYER_PROPS_START_TS = pd.Timestamp("2023-05-03T05:30:00Z", tz="UTC")
PLAYER_PROPS_START_DATE = PLAYER_PROPS_START_TS.date()

# Player odds columns
PLAYER_ODDS_NUMERIC_COLS = ["odds_anytime_td_price"]
ODDS_FLAG_COLUMNS = ["odds_expected", "odds_schedule_fallback"]
ODDS_DATETIME_COLUMNS = ["odds_snapshot_ts"]


def _unique_preserve(seq: Iterable[str]) -> list[str]:
    """Return unique items while preserving order."""
    return list(dict.fromkeys(seq))


def _build_odds_column_list() -> list[str]:
    """Build the complete list of NFL odds feature columns."""
    columns: list[str] = []

    # Snapshot-specific metrics
    for metric in SNAPSHOT_BASE_METRICS:
        for key in SNAPSHOT_SEQUENCE:
            suffix = SNAPSHOT_SUFFIXES[key]
            col_name = metric if key == "cutoff" else f"{metric}{suffix}"
            columns.append(col_name)

    # Deltas relative to cutoff snapshot
    for metric in DELTA_METRICS:
        for key in DELTA_SEQUENCE:
            suffix = SNAPSHOT_SUFFIXES[key]
            col_name = f"{metric}_delta" if key == "open" else f"{metric}_delta{suffix}"
            columns.append(col_name)

    # Percent change in vig
    for key in DELTA_SEQUENCE:
        suffix = SNAPSHOT_SUFFIXES[key]
        col_name = "moneyline_vig_pct_delta" if key == "open" else f"moneyline_vig_pct_delta{suffix}"
        columns.append(col_name)

    # Team and opponent implied totals at each snapshot
    for metric in TEAM_TOTAL_METRICS:
        for key in SNAPSHOT_SEQUENCE:
            suffix = SNAPSHOT_SUFFIXES[key]
            col_name = metric if key == "cutoff" else f"{metric}{suffix}"
            columns.append(col_name)

    # Team/opponent implied total deltas
    for metric in TEAM_TOTAL_METRICS:
        for key in DELTA_SEQUENCE:
            suffix = SNAPSHOT_SUFFIXES[key]
            col_name = f"{metric}_delta" if key == "open" else f"{metric}_delta{suffix}"
            columns.append(col_name)

    columns.extend(ADDITIONAL_COLUMNS)
    return _unique_preserve(columns)


NFL_ODDS_COLUMNS = _build_odds_column_list()

_NFL_EARLIEST_DATE = ODDS_START_DATE
_CANDIDATE_HOURS: tuple[int, ...] = (
    SNAPSHOT_HOUR_UTC,
    max(SNAPSHOT_HOUR_UTC + 2, OPEN_START_HOUR_UTC),
    max(SNAPSHOT_HOUR_UTC + 4, OPEN_START_HOUR_UTC + 2),
    18, 20, 22,
)

SNAPSHOT_TS_COLUMNS = {
    key: ("odds_snapshot_ts" if suffix == "" else f"odds_snapshot_ts{suffix}")
    for key, suffix in SNAPSHOT_SUFFIXES.items()
}

SCHEDULE_CACHE_PATH = Path("cache/collect/nfl_schedules.parquet")


# =============================================================================
# CACHE UTILITIES
# =============================================================================

CACHE_DIR = CACHE_ROOT / "odds_api"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_read(cache_key: str):
    """Read from disk cache."""
    fp = CACHE_DIR / f"{cache_key}.json"
    if fp.exists():
        try:
            with fp.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            fp.unlink(missing_ok=True)
    return None


def _cache_write(cache_key: str, data: Any):
    """Write to disk cache."""
    fp = CACHE_DIR / f"{cache_key}.json"
    try:
        with fp.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as _e:
        logger.debug("Failed to write cache %s: %s", fp, _e)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def snapshot_datetime(game_day: dt.date, hour: int = SNAPSHOT_HOUR_UTC) -> dt.datetime:
    """Create a UTC-aware datetime for a snapshot on a given game day and hour."""
    hour = int(max(min(hour, 23), 0))
    return dt.datetime(game_day.year, game_day.month, game_day.day, hour, 0, 0,
                       tzinfo=dt.timezone.utc)


def _current_utc() -> pd.Timestamp:
    """Return current UTC timestamp."""
    ts = pd.Timestamp.utcnow()
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _american_to_prob(odds: Optional[float]) -> Optional[float]:
    """Convert American odds to implied probability."""
    if odds is None:
        return None
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def _spread_to_home_prob(spread: Optional[float]) -> Optional[float]:
    """Approximate home win probability from the consensus spread."""
    if spread is None or (isinstance(spread, float) and math.isnan(spread)):
        return None
    try:
        spread_val = float(spread)
    except (TypeError, ValueError):
        return None
    sigma = 13.45
    mu = -spread_val
    z = mu / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _median(values: Iterable[float]) -> Optional[float]:
    """Compute median of values, ignoring None."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(np.median(vals))


def _std(values: Iterable[float]) -> Optional[float]:
    """Compute std of values."""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return None
    return float(np.std(vals))


def _diff(current: Optional[float], opening: Optional[float]) -> Optional[float]:
    """Compute difference between current and opening values."""
    if current is None or opening is None:
        return None
    try:
        return float(current) - float(opening)
    except (TypeError, ValueError):
        return None


def _pct_change(current: Optional[float], reference: Optional[float]) -> Optional[float]:
    """Compute percent change from reference to current."""
    if current is None or reference is None:
        return None
    try:
        reference = float(reference)
        if reference == 0:
            return None
        return (float(current) - reference) / abs(reference)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# Player name normalization helpers
_NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def _last_name(full_name: str) -> str:
    """Return the primary last name for fuzzy odds matching."""
    if not full_name:
        return ""
    clean = re.sub(r"[^A-Za-z\-']", " ", full_name)
    tokens = [tok for tok in clean.split() if tok]
    if not tokens:
        return ""
    last = tokens[-1].lower()
    if last.rstrip(".") in _NAME_SUFFIXES and len(tokens) >= 2:
        last = tokens[-2].lower()
    return last.rstrip(".")


# =============================================================================
# ODDS API CLIENT
# =============================================================================

class OddsAPIClient:
    """Thin wrapper around The Odds API V4 with basic retry & caching logic."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str, regions: str = DEFAULT_REGIONS, timeout: int = 10):
        self.api_key = api_key
        self.regions = regions
        self.timeout = timeout
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._last_request_monotonic: float = 0.0
        try:
            self._min_interval = float(os.getenv("ODDS_API_MIN_INTERVAL", "0.3"))
        except Exception:
            self._min_interval = 0.3
        self.metrics: Dict[str, int] = {
            'events_cache_hits': 0, 'events_api_calls': 0, 'odds_cache_hits': 0,
            'odds_live_calls': 0, 'odds_hist_calls': 0,
        }

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        url = f"{self.BASE_URL}{path}"
        params = params.copy()
        params.setdefault("apiKey", self.api_key)
        attempts = 0
        while True:
            attempts += 1
            try:
                with self._lock:
                    now = time.monotonic()
                    wait = self._min_interval - (now - self._last_request_monotonic)
                    if wait > 0:
                        time.sleep(wait)
                    self._last_request_monotonic = time.monotonic()
                resp = self._session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 401:
                    logger.error("[OddsAPI] 401 Unauthorized – no remaining quota or invalid key.")
                    resp.raise_for_status()
                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    try:
                        backoff = float(ra) if ra is not None else max(2.0, 3 * self._min_interval)
                    except Exception:
                        backoff = max(2.0, 3 * self._min_interval)
                    logger.warning("[OddsAPI] 429 rate-limit encountered. Sleeping %.1f s…", backoff)
                    if attempts <= 4:
                        time.sleep(backoff)
                        continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempts >= 3:
                    logger.error("[OddsAPI] Request failed after retries: %s", exc)
                    raise
                logger.warning("[OddsAPI] Request error – retrying (%d/3): %s", attempts, exc)

    @lru_cache(maxsize=1024)
    def get_historical_events(self, snapshot_dt: dt.datetime) -> Dict[str, Any]:
        """Return the historical events snapshot wrapper for the given timestamp."""
        iso_ts = snapshot_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.debug("Fetching historical event list for %s", iso_ts)
        cache_key = f"events_{iso_ts}"
        cached = _cache_read(cache_key)
        if cached is not None:
            self.metrics['events_cache_hits'] += 1
            return cached

        today = dt.datetime.utcnow().date()
        if snapshot_dt.date() > today:
            return {"data": []}

        try:
            data = self._get(
                f"/historical/sports/{SPORT_KEY}/events",
                {"date": iso_ts},
            )
            self.metrics['events_api_calls'] += 1
        except requests.HTTPError as exc:
            if (exc.response is not None and exc.response.status_code == 404
                    and snapshot_dt.date() == dt.datetime.utcnow().date()):
                data = {"data": []}
            else:
                raise
        _cache_write(cache_key, data)
        return data

    @lru_cache(maxsize=2048)
    def get_event_id(self, home_team: str, away_team: str, snapshot_dt: dt.datetime) -> Optional[str]:
        """Lookup the Odds-API event id for a game."""
        home_team_norm = _normalize_team(home_team)
        away_team_norm = _normalize_team(away_team)

        events_wrapper = self.get_historical_events(snapshot_dt)
        data: List[Dict[str, Any]] = events_wrapper.get("data", events_wrapper)

        for ev in data:
            if ev["home_team"] == home_team_norm and ev["away_team"] == away_team_norm:
                return ev["id"]
        for ev in data:
            if ev["home_team"] == away_team_norm and ev["away_team"] == home_team_norm:
                return ev["id"]

        if snapshot_dt.date() >= dt.datetime.utcnow().date():
            try:
                live_events_cache_key = f"live_events_{snapshot_dt.date()}"
                live_events = _cache_read(live_events_cache_key)
                if live_events is None:
                    live_events = self._get(
                        f"/sports/{SPORT_KEY}/events",
                        {"regions": self.regions},
                    )
                    _cache_write(live_events_cache_key, live_events)
                data_live = live_events.get("data", live_events)
                for ev in data_live:
                    if ev["home_team"] == home_team_norm and ev["away_team"] == away_team_norm:
                        return ev["id"]
                    if ev["home_team"] == away_team_norm and ev["away_team"] == home_team_norm:
                        return ev["id"]
            except Exception as _live_exc:
                logger.debug("Live events lookup failed: %s", _live_exc)

        logger.debug("No event id found for %s vs %s on %s", home_team, away_team, snapshot_dt.date())
        return None

    @lru_cache(maxsize=1024)
    def get_historical_event_odds(
        self, event_id: str, markets: str, snapshot_dt: dt.datetime, *, force_historical: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Fetch historical or live odds for an event."""
        iso_ts = snapshot_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        m_hash = hashlib.md5(markets.encode()).hexdigest()[:8]
        cache_key = f"odds_{'H_' if force_historical else ''}{event_id}_{iso_ts}_{m_hash}"
        cached = _cache_read(cache_key)
        if cached is not None:
            self.metrics['odds_cache_hits'] += 1
            return cached

        if snapshot_dt.date() >= dt.datetime.utcnow().date() and not force_historical:
            try:
                live_url = f"{self.BASE_URL}/sports/{SPORT_KEY}/events/{event_id}/odds"
                resp = self._session.get(
                    live_url,
                    params={"regions": self.regions, "markets": markets, "apiKey": self.api_key},
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    _cache_write(cache_key, data)
                    self.metrics['odds_live_calls'] += 1
                    return data
                else:
                    logger.debug("Live odds not yet available (status %s) for event %s", resp.status_code, event_id)
                    return None
            except Exception as _live_exc:
                logger.debug("Live odds request failed for event %s: %s", event_id, _live_exc)
                return None

        try:
            data = self._get(
                f"/historical/sports/{SPORT_KEY}/events/{event_id}/odds",
                {"regions": self.regions, "markets": markets, "date": iso_ts},
            )
            _cache_write(cache_key, data)
            self.metrics['odds_hist_calls'] += 1
            return data
        except requests.HTTPError as exc:
            logger.warning("Failed to fetch historical odds for %s markets=%s – %s", event_id, markets, exc)
            if snapshot_dt.date() < dt.datetime.utcnow().date():
                _cache_write(cache_key, {})
            return None

    @lru_cache(maxsize=2048)
    def get_event_info(self, home_team: str, away_team: str, snapshot_dt: dt.datetime) -> Optional[Dict[str, Any]]:
        """Return the event dict (including commence_time) for the supplied matchup."""
        home_team_norm = _normalize_team(home_team)
        away_team_norm = _normalize_team(away_team)

        events_wrapper = self.get_historical_events(snapshot_dt)
        data = events_wrapper.get("data", events_wrapper)

        for ev in data:
            if ev["home_team"] == home_team_norm and ev["away_team"] == away_team_norm:
                return ev
        for ev in data:
            if ev["home_team"] == away_team_norm and ev["away_team"] == home_team_norm:
                return ev

        if snapshot_dt.date() >= dt.datetime.utcnow().date():
            live_events_cache_key = f"live_events_{snapshot_dt.date()}"
            live_events = _cache_read(live_events_cache_key)
            if live_events is None:
                live_events = self._get(f"/sports/{SPORT_KEY}/events", {"regions": self.regions})
                _cache_write(live_events_cache_key, live_events)
            data_live = live_events.get("data", live_events)
            for ev in data_live:
                if ev["home_team"] == home_team_norm and ev["away_team"] == away_team_norm:
                    return ev
                if ev["home_team"] == away_team_norm and ev["away_team"] == home_team_norm:
                    return ev

        logger.debug("No event info found for %s vs %s on %s", home_team, away_team, snapshot_dt.date())
        return None


# =============================================================================
# GAME-LEVEL ODDS FEATURES (from nfl_odds.py)
# =============================================================================

# Import remaining functions from the original nfl_odds.py
# These are complex parsing and computation functions that remain unchanged

def _parse_moneyline(odds_json: Dict[str, Any], home_name: str, away_name: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse moneyline odds from bookmaker data."""
    home_prices, away_prices = [], []
    data_block = odds_json.get("data", odds_json)
    for book in data_block.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            for oc in mkt.get("outcomes", []):
                name = oc.get("name")
                price = oc.get("price")
                if price is None:
                    continue
                if name == home_name:
                    home_prices.append(float(price))
                elif name == away_name:
                    away_prices.append(float(price))
    return _median(home_prices), _median(away_prices)


def _parse_spreads(odds_json: Dict[str, Any], home_name: str, away_name: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Parse spread odds from bookmaker data."""
    lines, home_prices, away_prices = [], [], []
    data_block = odds_json.get("data", odds_json)
    for book in data_block.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt.get("key") != "spreads":
                continue
            for oc in mkt.get("outcomes", []):
                name = oc.get("name")
                point = oc.get("point")
                price = oc.get("price")
                if point is None or price is None:
                    continue
                if name == home_name:
                    lines.append(float(point))
                    home_prices.append(float(price))
                elif name == away_name:
                    away_prices.append(float(price))
    return _median(lines), _median(home_prices), _median(away_prices), _std(lines)


def _parse_totals(odds_json: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Parse total (over/under) odds from bookmaker data."""
    lines, over_prices, under_prices = [], [], []
    data_block = odds_json.get("data", odds_json)
    for book in data_block.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt.get("key") != "totals":
                continue
            for oc in mkt.get("outcomes", []):
                name = oc.get("name")
                point = oc.get("point")
                price = oc.get("price")
                if point is None or price is None:
                    continue
                lines.append(float(point))
                if name == "Over":
                    over_prices.append(float(price))
                elif name == "Under":
                    under_prices.append(float(price))
    return _median(lines), _median(over_prices), _median(under_prices), _std(lines)


def _build_default_row() -> Dict[str, Optional[float]]:
    """Build a default row with all odds columns set to None."""
    row: Dict[str, Optional[float]] = {col: None for col in NFL_ODDS_COLUMNS}
    row["odds_expected"] = 0
    row["odds_schedule_fallback"] = 0
    return row


def _extract_snapshot_metrics(odds_json: Optional[Dict[str, Any]], home_name: str, away_name: str) -> Dict[str, Optional[float]]:
    """Extract all metrics from a single odds snapshot."""
    if odds_json is None:
        return {}
    
    ml_home, ml_away = _parse_moneyline(odds_json, home_name, away_name)
    spread_line, spread_home, spread_away, spread_line_std = _parse_spreads(odds_json, home_name, away_name)
    total_line, total_over, total_under, total_line_std = _parse_totals(odds_json)
    
    prob_home = _american_to_prob(ml_home)
    prob_away = _american_to_prob(ml_away)
    
    vig = None
    if prob_home is not None and prob_away is not None:
        vig = (prob_home + prob_away) - 1.0
    
    data_block = odds_json.get("data", odds_json)
    ml_book_count = sum(1 for b in data_block.get("bookmakers", []) 
                        for m in b.get("markets", []) if m.get("key") == "h2h")
    spread_book_count = sum(1 for b in data_block.get("bookmakers", [])
                            for m in b.get("markets", []) if m.get("key") == "spreads")
    total_book_count = sum(1 for b in data_block.get("bookmakers", [])
                           for m in b.get("markets", []) if m.get("key") == "totals")
    
    return {
        "moneyline_home": ml_home,
        "moneyline_away": ml_away,
        "spread_line": spread_line,
        "spread_home_price": spread_home,
        "spread_away_price": spread_away,
        "total_line": total_line,
        "total_over_price": total_over,
        "total_under_price": total_under,
        "implied_prob_home": prob_home,
        "implied_prob_away": prob_away,
        "moneyline_vig": vig,
        "moneyline_book_count": float(ml_book_count) if ml_book_count else None,
        "spread_book_count": float(spread_book_count) if spread_book_count else None,
        "total_book_count": float(total_book_count) if total_book_count else None,
        "spread_line_std": spread_line_std,
        "total_line_std": total_line_std,
    }


def _compute_game_odds(
    client: OddsAPIClient,
    game_day: dt.date,
    home_team: str,
    away_team: str,
    *,
    open_snapshot_hour: int,
    return_snapshot_ts: bool = False,
    cutoff_hours: float | None = None,
    fallback_hours: float | None = None,
) -> Dict[str, Optional[float]] | Tuple[Dict[str, Optional[float]], Dict[str, Optional[pd.Timestamp]], Optional[str]]:
    """Compute complete game odds features from API data."""
    home_norm = _normalize_team(home_team)
    away_norm = _normalize_team(away_team)
    
    if game_day < ODDS_START_DATE:
        row = _build_default_row()
        if return_snapshot_ts:
            return row, {key: None for key in SNAPSHOT_SEQUENCE}, None
        return row
    
    # Find the event
    event_info = None
    event_id = None
    for hour in _CANDIDATE_HOURS:
        snap_dt = snapshot_datetime(game_day, hour)
        event_info = client.get_event_info(home_norm, away_norm, snap_dt)
        if event_info:
            event_id = event_info.get("id")
            break
    
    if event_id is None:
        row = _build_default_row()
        if return_snapshot_ts:
            return row, {key: None for key in SNAPSHOT_SEQUENCE}, None
        return row
    
    # Compute snapshot timestamps
    commence_ts = pd.to_datetime(event_info.get("commence_time"), utc=True, errors="coerce")
    snapshot_ts_map: Dict[str, Optional[pd.Timestamp]] = {}
    
    # Cutoff snapshot
    from utils.feature.asof import decision_cutoff_hours_default, fallback_cutoff_hours

    default_cutoff = decision_cutoff_hours_default()
    cutoff_hours_value = default_cutoff
    if cutoff_hours is not None:
        try:
            cutoff_hours_value = float(cutoff_hours)
        except (TypeError, ValueError):
            cutoff_hours_value = default_cutoff
    if not math.isfinite(cutoff_hours_value) or cutoff_hours_value <= 0:
        cutoff_hours_value = default_cutoff

    default_fallback = fallback_cutoff_hours()
    fallback_hours_value = default_fallback
    if fallback_hours is not None:
        try:
            fallback_hours_value = float(fallback_hours)
        except (TypeError, ValueError):
            fallback_hours_value = default_fallback
    if not math.isfinite(fallback_hours_value) or fallback_hours_value <= 0:
        fallback_hours_value = default_fallback

    if pd.notna(commence_ts):
        cutoff_ts = commence_ts - pd.Timedelta(hours=cutoff_hours_value)
    else:
        fallback_base = pd.Timestamp(game_day).tz_localize("UTC")
        cutoff_ts = fallback_base - pd.Timedelta(hours=fallback_hours_value)
    snapshot_ts_map["cutoff"] = cutoff_ts
    
    # Relative snapshots
    for key, hours in RELATIVE_HOURS.items():
        if pd.notna(commence_ts):
            snapshot_ts_map[key] = commence_ts - pd.Timedelta(hours=hours)
        else:
            snapshot_ts_map[key] = None
    
    # Open snapshot
    open_ts = None
    for lookback in OPEN_LOOKBACK_HOURS:
        candidate = snapshot_datetime(game_day, open_snapshot_hour) - dt.timedelta(hours=lookback)
        if client.get_event_info(home_norm, away_norm, candidate):
            ts = pd.Timestamp(candidate)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            open_ts = ts
            break
    snapshot_ts_map["open"] = open_ts
    
    # Fetch and parse snapshots
    markets = "h2h,spreads,totals"
    snapshots: Dict[str, Dict[str, Optional[float]]] = {}
    
    for key, ts in snapshot_ts_map.items():
        if ts is None:
            snapshots[key] = {}
            continue
        try:
            odds_json = client.get_historical_event_odds(event_id, markets, ts.to_pydatetime())
            snapshots[key] = _extract_snapshot_metrics(odds_json, home_norm, away_norm)
        except Exception:
            snapshots[key] = {}
    
    # Build result row
    row = _build_default_row()
    row["odds_expected"] = 1
    
    # Populate snapshot values
    for key in SNAPSHOT_SEQUENCE:
        suffix = SNAPSHOT_SUFFIXES[key]
        snap_data = snapshots.get(key, {})
        for metric in SNAPSHOT_BASE_METRICS:
            col_name = metric if key == "cutoff" else f"{metric}{suffix}"
            if metric in snap_data:
                row[col_name] = snap_data[metric]
    
    # Compute hours before kickoff for cutoff
    if pd.notna(commence_ts) and snapshot_ts_map.get("cutoff") is not None:
        hours_before = (commence_ts - snapshot_ts_map["cutoff"]).total_seconds() / 3600.0
        row["odds_hours_before_kickoff"] = hours_before
    
    # Compute deltas
    cutoff_data = snapshots.get("cutoff", {})
    for key in DELTA_SEQUENCE:
        suffix = SNAPSHOT_SUFFIXES[key]
        compare_data = snapshots.get(key, {})
        for metric in DELTA_METRICS:
            current = cutoff_data.get(metric)
            opening = compare_data.get(metric)
            delta = _diff(current, opening)
            col_name = f"{metric}_delta" if key == "open" else f"{metric}_delta{suffix}"
            row[col_name] = delta
        
        # Vig percent delta
        current_vig = cutoff_data.get("moneyline_vig")
        opening_vig = compare_data.get("moneyline_vig")
        vig_pct = _pct_change(current_vig, opening_vig)
        vig_col = "moneyline_vig_pct_delta" if key == "open" else f"moneyline_vig_pct_delta{suffix}"
        row[vig_col] = vig_pct
    
    if return_snapshot_ts:
        return row, snapshot_ts_map, event_id
    return row


# =============================================================================
# NFL GAME-LEVEL ODDS ENRICHMENT
# =============================================================================

def add_nfl_odds_features_to_df(
    df_pl: pl.DataFrame, *, api_key: Optional[str] = None, player_col: str = "player_name",
    verbose: bool = False, allow_schedule_fallback: bool = False, drop_schedule_rows: bool = False,
) -> pl.DataFrame:
    """Add NFL game-level odds features to a Polars DataFrame.
    
    This enriches the DataFrame with moneyline, spread, and totals data from The Odds API.
    """
    if df_pl.is_empty():
        return df_pl
    
    feature_cols = NFL_ODDS_COLUMNS + ODDS_FLAG_COLUMNS + ODDS_DATETIME_COLUMNS
    base_frame = df_pl.drop(feature_cols, strict=False)
    
    # ------------------------------------------------------------------
    # 1) Historical games: prefer stored odds snapshots for parity
    # ------------------------------------------------------------------
    use_snapshots = False
    max_game_date: Optional[dt.date] = None
    if "game_date" in df_pl.columns:
        try:
            max_game_date = df_pl.get_column("game_date").cast(pl.Date).max()
        except Exception:
            max_game_date = None
    today = dt.date.today()
    if (
        max_game_date is not None
        and max_game_date < today
        and {"season", "week", "home_team", "away_team"}.issubset(set(df_pl.columns))
        and ODDS_SNAPSHOT_DIR.exists()
    ):
        use_snapshots = True

    if use_snapshots:
        enriched = _attach_nfl_odds_from_snapshots(base_frame, df_pl)
        if drop_schedule_rows and "odds_expected" in enriched.columns:
            enriched = enriched.filter(pl.col("odds_expected") == 1)
        return enriched

    # ------------------------------------------------------------------
    # 2) Fallback to live Odds API for current / future games
    # ------------------------------------------------------------------
    if not api_key:
        empty_exprs = [pl.lit(None).cast(pl.Float32).alias(c) for c in NFL_ODDS_COLUMNS]
        empty_exprs += [pl.lit(None).cast(pl.Datetime("ms", "UTC")).alias(c) for c in ODDS_DATETIME_COLUMNS]
        flag_expr = [pl.lit(0).cast(pl.Int8).alias("odds_expected"), pl.lit(1).cast(pl.Int8).alias("odds_schedule_fallback")]
        result = base_frame.with_columns(empty_exprs + flag_expr)
        if drop_schedule_rows:
            return result.filter(pl.col("odds_expected") == 1)
        return result
    
    required_cols = {"game_date", "home_team", "away_team"}
    missing = required_cols - set(df_pl.columns)
    if missing:
        logger.warning("add_nfl_odds_features_to_df: DataFrame missing required columns %s", missing)
        empty_exprs = [pl.lit(None).cast(pl.Float32).alias(c) for c in NFL_ODDS_COLUMNS]
        empty_exprs += [pl.lit(None).cast(pl.Datetime("ms", "UTC")).alias(c) for c in ODDS_DATETIME_COLUMNS]
        flag_expr = [pl.lit(0).cast(pl.Int8).alias("odds_expected"), pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback")]
        return base_frame.with_columns(empty_exprs + flag_expr)
    
    client = OddsAPIClient(api_key)
    core_cols = ["game_date", "home_team", "away_team"]
    if "home_team_abbr" in df_pl.columns:
        core_cols.append("home_team_abbr")
    if "away_team_abbr" in df_pl.columns:
        core_cols.append("away_team_abbr")
    if player_col not in core_cols and player_col in df_pl.columns:
        core_cols.append(player_col)
    if "game_id" in df_pl.columns:
        core_cols.append("game_id")
    has_cutoff_ts = "decision_cutoff_ts" in df_pl.columns
    has_game_start = "game_start_utc" in df_pl.columns
    if has_cutoff_ts:
        core_cols.append("decision_cutoff_ts")
    if has_game_start:
        core_cols.append("game_start_utc")
    
    core_pd = df_pl.select(core_cols).to_pandas()
    game_cache: Dict[Tuple[dt.date, str, str, Optional[float]], Dict[str, Optional[float]]] = {}
    rows: Dict[int, Dict[str, Optional[float]]] = {}
    
    for idx, row in core_pd.iterrows():
        game_date_val = pd.to_datetime(row["game_date"]).date()
        home_team_val = row.get("home_team")
        away_team_val = row.get("away_team")
        cutoff_hours_override: Optional[float] = None
        if has_cutoff_ts and has_game_start:
            cutoff_ts = pd.to_datetime(row.get("decision_cutoff_ts"), utc=True, errors="coerce")
            kickoff_ts = pd.to_datetime(row.get("game_start_utc"), utc=True, errors="coerce")
            if pd.notna(cutoff_ts) and pd.notna(kickoff_ts):
                delta_hours = (kickoff_ts - cutoff_ts).total_seconds() / 3600.0
                if delta_hours > 0:
                    cutoff_hours_override = float(delta_hours)
        cache_key = (
            game_date_val,
            str(home_team_val),
            str(away_team_val),
            round(cutoff_hours_override, 4) if cutoff_hours_override is not None else None,
        )
        
        if cache_key in game_cache:
            rows[idx] = game_cache[cache_key].copy()
            continue
        
        try:
            data, snapshot_ts_map, _ = _compute_game_odds(
                client, game_date_val, str(home_team_val), str(away_team_val),
                open_snapshot_hour=SNAPSHOT_HOUR_UTC,
                return_snapshot_ts=True,
                cutoff_hours=cutoff_hours_override,
                fallback_hours=ODDS_FALLBACK_HOURS,
            )
            cutoff_snapshot_ts = snapshot_ts_map.get("cutoff") if snapshot_ts_map else None
            data["odds_snapshot_ts"] = cutoff_snapshot_ts
        except Exception as exc:
            logger.debug("add_nfl_odds_features_to_df: failed odds for %s – %s", cache_key, exc)
            data = _build_default_row()
            data["odds_snapshot_ts"] = None
        
        game_cache[cache_key] = data.copy()
        rows[idx] = data.copy()
    
    odds_df = pd.DataFrame.from_dict(rows, orient="index").reindex(core_pd.index)
    odds_df = odds_df.fillna(value=np.nan)
    
    odds_pl = pl.from_pandas(odds_df)
    cast_exprs = [pl.col(col).cast(pl.Float32) for col in NFL_ODDS_COLUMNS if col in odds_pl.columns]
    flag_exprs = [pl.col(flag).cast(pl.Int8) for flag in ODDS_FLAG_COLUMNS if flag in odds_pl.columns]
    datetime_exprs = [
        pl.col(col).cast(pl.Datetime("ms", "UTC")) for col in ODDS_DATETIME_COLUMNS if col in odds_pl.columns
    ]
    all_exprs = cast_exprs + flag_exprs + datetime_exprs
    if all_exprs:
        odds_pl = odds_pl.with_columns(all_exprs)
    
    enriched = base_frame.hstack(odds_pl)
    
    if drop_schedule_rows and "odds_expected" in enriched.columns:
        enriched = enriched.filter(pl.col("odds_expected") == 1)
    
    return enriched


def _attach_nfl_odds_from_snapshots(
    base_frame: pl.DataFrame,
    df_pl: pl.DataFrame,
) -> pl.DataFrame:
    """
    Attach NFL odds from stored snapshots for historical games.

    This is used for dates inside the training horizon to avoid live API
    drift and to guarantee train/predict parity. It joins against
    `data/raw/odds_snapshots/season=*/week=*/part.parquet` on
    (season, week, home_team, away_team).
    """
    # Ensure key columns are present
    key_cols = {"season", "week", "home_team", "away_team"}
    if not key_cols.issubset(set(df_pl.columns)):
        return base_frame

    if not ODDS_SNAPSHOT_DIR.exists():
        return base_frame

    try:
        seasons = df_pl.get_column("season").unique().to_list()
        weeks = df_pl.get_column("week").unique().to_list()
    except Exception:
        return base_frame

    if not seasons or not weeks:
        return base_frame

    # Load relevant snapshot partitions eagerly and concatenate with relaxed schema
    frames: list[pl.DataFrame] = []
    for season in seasons:
        for week in weeks:
            path = ODDS_SNAPSHOT_DIR / f"season={int(season)}" / f"week={_format_week(week)}" / "part.parquet"
            if path.exists():
                try:
                    frames.append(pl.read_parquet(path))
                except Exception as exc:
                    logger.warning("Failed to read odds snapshot %s: %s", path, exc)
    if not frames:
        return base_frame

    snap = pl.concat(frames, how="diagonal_relaxed")
    if snap.is_empty():
        return base_frame

    snap_cols_all = set(snap.columns)
    numeric_cols = [c for c in NFL_ODDS_COLUMNS if c in snap_cols_all]
    datetime_cols = [c for c in ODDS_DATETIME_COLUMNS if c in snap_cols_all]

    select_cols = ["season", "week", "home_team", "away_team"] + numeric_cols + datetime_cols
    select_cols = [c for c in select_cols if c in snap_cols_all]
    snap = snap.select(select_cols)
    if snap.is_empty():
        return base_frame

    # Normalise types
    cast_exprs: list[pl.Expr] = []
    for col in numeric_cols:
        cast_exprs.append(pl.col(col).cast(pl.Float32))
    for col in datetime_cols:
        cast_exprs.append(pl.col(col).cast(pl.Datetime("ms", "UTC")))
    if cast_exprs:
        snap = snap.with_columns(cast_exprs)

    # Odds flags: mark rows with any odds as expected
    has_any_odds = None
    for col in ("total_line", "moneyline_home", "moneyline_away"):
        if col in snap.columns:
            expr = pl.col(col).is_not_null()
            has_any_odds = expr if has_any_odds is None else (has_any_odds | expr)
    if has_any_odds is None:
        has_any_odds = pl.lit(False)

    snap = snap.with_columns(
        [
            has_any_odds.cast(pl.Int8).alias("odds_expected"),
            pl.lit(0).cast(pl.Int8).alias("odds_schedule_fallback"),
        ]
    )

    # Join back to base frame
    enriched = base_frame.join(
        snap,
        on=["season", "week", "home_team", "away_team"],
        how="left",
    )

    # Ensure all expected columns exist
    missing_numeric = [c for c in NFL_ODDS_COLUMNS if c not in enriched.columns]
    if missing_numeric:
        enriched = enriched.with_columns(
            [pl.lit(None).cast(pl.Float32).alias(c) for c in missing_numeric]
        )

    missing_dt = [c for c in ODDS_DATETIME_COLUMNS if c not in enriched.columns]
    if missing_dt:
        enriched = enriched.with_columns(
            [pl.lit(None).cast(pl.Datetime("ms", "UTC")).alias(c) for c in missing_dt]
        )

    for flag in ODDS_FLAG_COLUMNS:
        if flag not in enriched.columns:
            default_val = 0
            enriched = enriched.with_columns(
                pl.lit(default_val).cast(pl.Int8).alias(flag)
            )
    
    return enriched


# =============================================================================
# PLAYER-LEVEL ODDS FEATURES
# =============================================================================

def attach_player_odds_features(
    df: pd.DataFrame,
    api_key: str,
    player_col: str = "player_name",
    verbose: bool = False,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Attach player prop odds (anytime TD) to a pandas DataFrame."""
    def _fill_team_names(full_col: str, abbr_col: str):
        nonlocal df
        if full_col not in df.columns:
            if abbr_col in df.columns:
                df = df.copy()
                df[full_col] = df[abbr_col].apply(_normalize_team)
            else:
                raise ValueError(f"attach_player_odds_features: Missing '{full_col}' or '{abbr_col}' column")
        else:
            if abbr_col in df.columns:
                mask = df[full_col].isna() | (df[full_col].astype(str).str.strip() == '')
                if mask.any():
                    df.loc[mask, full_col] = df.loc[mask, abbr_col].astype(str).apply(_normalize_team)
    
    _fill_team_names('home_team', 'home_team_abbr')
    _fill_team_names('away_team', 'away_team_abbr')
    
    required_cols = {"game_date", "home_team", "away_team", player_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"attach_player_odds_features: DataFrame missing required columns: {missing}")
    
    client = OddsAPIClient(api_key)
    
    if verbose:
        logger.setLevel(logging.INFO)
        logger.info("attach_player_odds_features | processing rows=%d", len(df))
    
    odds_map: Dict[int, Dict[str, Any]] = {}
    _CANDIDATE_HOURS_UTC = [SNAPSHOT_HOUR_UTC, 14, 16, 18, 20, 22]
    
    try:
        throttle_workers = int(os.getenv("ODDS_API_MAX_WORKERS", "0"))
    except Exception:
        throttle_workers = 0
    
    def _snapshot_dt_pd(ts: pd.Timestamp, hour: int) -> dt.datetime:
        ts = pd.to_datetime(ts)
        return dt.datetime(ts.year, ts.month, ts.day, hour, 0, 0, tzinfo=dt.timezone.utc)
    
    def _process_one_game(game_key: Tuple[pd.Timestamp, str, str], game_df: pd.DataFrame):
        sample_row = game_df.iloc[0]
        home_team_val = _normalize_team(str(sample_row["home_team"]))
        away_team_val = _normalize_team(str(sample_row["away_team"]))
        game_day = pd.to_datetime(sample_row["game_date"]).date()
        
        if game_day < PLAYER_PROPS_START_DATE:
            for idx in game_df.index:
                odds_map[idx] = {"odds_anytime_td_price": None}
            return
        
        event_info = None
        event_found_hour = None
        for hour in _CANDIDATE_HOURS_UTC:
            snap_dt_candidate = _snapshot_dt_pd(sample_row["game_date"], hour)
            event_info = client.get_event_info(home_team_val, away_team_val, snap_dt_candidate)
            if event_info:
                event_found_hour = hour
                break
        
        if event_info is None:
            for idx in game_df.index:
                odds_map[idx] = {"odds_anytime_td_price": None}
            return
        
        event_id = event_info["id"]
        commence_ts = pd.to_datetime(event_info["commence_time"], utc=True)
        latest_safe_ts = (commence_ts - pd.Timedelta(minutes=15)).floor("5min")
        earliest_snap_dt = _snapshot_dt_pd(sample_row["game_date"], event_found_hour or SNAPSHOT_HOUR_UTC)
        
        selected_snap = latest_safe_ts
        if selected_snap < earliest_snap_dt:
            selected_snap = earliest_snap_dt

        selected_snap = pd.Timestamp(selected_snap)
        if selected_snap.tzinfo is None:
            selected_snap = selected_snap.tz_localize("UTC")
        else:
            selected_snap = selected_snap.tz_convert("UTC")

        props_expected = selected_snap >= PLAYER_PROPS_START_TS
        if not props_expected:
            for idx in game_df.index:
                odds_map[idx] = {"odds_anytime_td_price": None}
            return

        odds_props = client.get_historical_event_odds(
            event_id,
            PLAYER_MARKETS_PARAM,
            selected_snap.to_pydatetime(),
        )
        if not odds_props and logger.isEnabledFor(logging.WARNING):
            logger.warning(
                "Player props missing (provider gap) event_id=%s home=%s away=%s snapshot=%s",
                event_id,
                home_team_val,
                away_team_val,
                selected_snap.isoformat(),
            )
        
        player_td_prices = {}
        if odds_props:
            data = odds_props.get("data", odds_props)
            for bookmaker in data.get("bookmakers", []):
                for mkt in bookmaker.get("markets", []):
                    key = mkt.get("key")
                    if key != "player_anytime_td":
                        continue
                    for oc in mkt.get("outcomes", []):
                        label = str(oc.get("name") or "").strip().lower()
                        if label == "no":
                            continue
                        player_name = (
                            oc.get("description")
                            or oc.get("participant")
                            or oc.get("name")
                        )
                        price = oc.get("price")
                        if not player_name or price is None:
                            continue
                        norm_name = _last_name(player_name)
                        if not norm_name:
                            continue
                        player_td_prices.setdefault(norm_name, []).append(float(price))
        
        for idx, row in game_df.iterrows():
            p_name = str(row[player_col])
            p_last = _last_name(p_name)
            td_price = None
            if p_last in player_td_prices:
                td_price = float(pd.Series(player_td_prices[p_last]).median())
            odds_map[idx] = {"odds_anytime_td_price": td_price}
    
    game_groups = df.groupby(["game_date", "home_team", "away_team"], sort=False, observed=True)
    _workers_base = max(1, min(max_workers, len(game_groups)))
    _workers = min(_workers_base, throttle_workers) if throttle_workers > 0 else _workers_base
    
    total_games = len(game_groups)
    completed_games = 0
    with ThreadPoolExecutor(max_workers=_workers) as ex:
        futures = {ex.submit(_process_one_game, k, g.copy()): k for k, g in game_groups}
        for fut in as_completed(futures):
            completed_games += 1
            if verbose and (completed_games == total_games or completed_games % max(1, total_games // 10) == 0):
                logger.info("attach_player_odds_features | %d/%d games done", completed_games, total_games)
    
    for idx in df.index:
        if idx not in odds_map:
            odds_map[idx] = {"odds_anytime_td_price": df.loc[idx].get("odds_anytime_td_price")}
    
    odds_df = pd.DataFrame.from_dict(odds_map, orient="index").reindex(df.index)
    for _c in PLAYER_ODDS_NUMERIC_COLS:
        if _c in odds_df.columns:
            odds_df[_c] = pd.to_numeric(odds_df[_c], errors='coerce')
    
    result = pd.concat([df, odds_df], axis=1)
    game_dates = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    result["odds_expected"] = (game_dates >= PLAYER_PROPS_START_DATE).astype("int8")
    result["odds_schedule_fallback"] = 0
    
    miss = (result["odds_expected"] == 1) & result[PLAYER_ODDS_NUMERIC_COLS].isna().all(axis=1)
    if miss.any():
        result.loc[miss, "odds_expected"] = 2
        if logger.isEnabledFor(logging.WARNING):
            logger.warning("Odds expected but missing for %d rows (likely fetch gap).", int(miss.sum()))
    
    if verbose:
        logger.info("attach_player_odds_features | Completed. Non-null counts:\n%s", result[PLAYER_ODDS_NUMERIC_COLS].notna().sum())
    
    return result


def add_player_odds_features_to_df(
    df_pl: pl.DataFrame, *, api_key: str | None = None, player_col: str = "player_name",
    per_game_workers: int = 4, verbose: bool = False,
) -> pl.DataFrame:
    """Attach player odds features directly to an in-memory Polars DataFrame."""
    from utils.general.config import get_odds_api_key
    
    if df_pl.is_empty():
        return df_pl
    
    if not api_key:
        api_key = get_odds_api_key()
    
    if not api_key:
        logger.info("add_player_odds_features_to_df: no odds_api_key configured – adding empty columns and returning")
        empty_exprs = [pl.lit(None).cast(pl.Float32).alias(c) for c in PLAYER_ODDS_NUMERIC_COLS]
        flag_expr = [pl.lit(0).cast(pl.Int8).alias("odds_expected")]
        df_base = df_pl.drop(PLAYER_ODDS_NUMERIC_COLS + ODDS_FLAG_COLUMNS, strict=False)
        return df_base.with_columns(empty_exprs + flag_expr)
    
    try:
        need_cols = ["game_date", "home_team", "away_team", "home_team_abbr", "away_team_abbr"]
        if player_col not in need_cols:
            need_cols.append(player_col)
        present_cols = [c for c in need_cols if c in df_pl.columns]
        core_pd = df_pl.select(present_cols).to_pandas()
        
        if player_col not in core_pd.columns:
            logger.warning("add_player_odds_features_to_df: missing player column '%s' – returning original frame", player_col)
            return df_pl
        
        enriched_pd = attach_player_odds_features(
            core_pd, api_key=api_key, player_col=player_col, verbose=verbose, max_workers=per_game_workers,
        )
        
        want_cols = PLAYER_ODDS_NUMERIC_COLS + ODDS_FLAG_COLUMNS
        odds_pd = enriched_pd[[c for c in want_cols if c in enriched_pd.columns]].copy()
        
        for c in PLAYER_ODDS_NUMERIC_COLS:
            if c not in odds_pd.columns:
                odds_pd[c] = pd.Series([None] * len(enriched_pd))
        if "odds_expected" not in odds_pd.columns:
            odds_pd["odds_expected"] = 1
        
        odds_pl = pl.from_pandas(odds_pd).with_columns(
            [pl.col(c).cast(pl.Float32) for c in PLAYER_ODDS_NUMERIC_COLS] +
            [pl.col("odds_expected").cast(pl.Int8)]
        )
        
        df_base = df_pl.drop(want_cols, strict=False)
        return df_base.hstack(odds_pl)
    except Exception as exc:
        logger.warning("add_player_odds_features_to_df: failed to enrich in-memory – %s", exc)
        empty_exprs = [pl.lit(None).cast(pl.Float32).alias(c) for c in PLAYER_ODDS_NUMERIC_COLS]
        flag_expr = [pl.lit(0).cast(pl.Int8).alias("odds_expected")]
        df_base = df_pl.drop(PLAYER_ODDS_NUMERIC_COLS + ODDS_FLAG_COLUMNS, strict=False)
        return df_base.with_columns(empty_exprs + flag_expr)


# =============================================================================
# ODDS SNAPSHOT COLLECTION AND PERSISTENCE
# =============================================================================

def _format_week(value: object) -> str:
    try:
        return f"{int(value):02d}"
    except (TypeError, ValueError):
        return str(value)


def _normalize_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    df = schedule.copy()
    if "gameday" in df.columns:
        df["game_date"] = pd.to_datetime(df["gameday"], errors="coerce").dt.date
    elif "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    else:
        df["game_date"] = pd.NaT
    
    if "start_time_utc" in df.columns:
        df["start_time_utc"] = pd.to_datetime(df["start_time_utc"], errors="coerce", utc=True)
    else:
        df["start_time_utc"] = pd.NaT
    
    return df


def collect_odds_snapshots(
    seasons: Iterable[int], *, start_date: dt.date | None = None, end_date: dt.date | None = None,
    api_key: str | None = None, snapshot_labels: Sequence[str] = SNAPSHOT_SEQUENCE,
    include_player_props: bool = False,
) -> List[Path]:
    """Collect and persist betting market snapshots for the supplied seasons."""
    from utils.collect.arrival_log import log_feed_arrivals
    from utils.collect.nfl_schedules import get_schedule
    from utils.general.config import get_odds_api_key
    
    seasons = list({int(season) for season in seasons if season is not None})
    if not seasons:
        logger.info("collect_odds_snapshots: no seasons provided – skipping.")
        return []
    
    if not api_key:
        api_key = get_odds_api_key()
    if not api_key:
        logger.info("collect_odds_snapshots: no odds_api_key configured – skipping.")
        return []
    
    schedule = get_schedule(seasons)
    if schedule.empty:
        logger.info("collect_odds_snapshots: schedule empty for seasons %s.", seasons)
        return []
    
    schedule = _normalize_schedule(schedule)
    
    if start_date is None:
        start_date = schedule["game_date"].min()
    if end_date is None:
        end_date = schedule["game_date"].max()
    if start_date is None or end_date is None:
        logger.info("collect_odds_snapshots: schedule did not provide game dates.")
        return []
    
    date_mask = (schedule["game_date"] >= start_date) & (schedule["game_date"] <= end_date)
    schedule = schedule.loc[date_mask].copy()
    if schedule.empty:
        logger.info("collect_odds_snapshots: no games between %s and %s.", start_date, end_date)
        return []
    
    client = OddsAPIClient(api_key)
    written_paths: List[Path] = []
    snapshot_records: Dict[Tuple[int, object], List[Dict[str, object]]] = {}
    collected_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
    
    for game in schedule.itertuples(index=False):
        game_id = getattr(game, "game_id", None)
        game_day = getattr(game, "game_date", None)
        home_team = getattr(game, "home_team", None)
        away_team = getattr(game, "away_team", None)
        season = getattr(game, "season", None)
        week = getattr(game, "week", None)
        
        if not all([game_id, game_day, home_team, away_team, season, week]):
            continue
        
        try:
            game_day_date = pd.to_datetime(game_day).date()
        except Exception:
            continue
        
        try:
            feature_row, snapshot_ts, event_id = _compute_game_odds(
                client, game_day_date, str(home_team), str(away_team),
                open_snapshot_hour=SNAPSHOT_HOUR_UTC,
                return_snapshot_ts=True,
                fallback_hours=ODDS_FALLBACK_HOURS,
            )
        except Exception as exc:
            logger.debug("collect_odds_snapshots: odds computation failed for %s vs %s – %s", home_team, away_team, exc)
            feature_row = _build_default_row()
            snapshot_ts = {label: None for label in snapshot_labels}
            event_id = None
        
        start_time_utc = getattr(game, "start_time_utc", pd.NaT)
        start_time_iso = None
        if pd.notna(start_time_utc):
            start_time_iso = pd.to_datetime(start_time_utc, utc=True, errors="coerce")
            start_time_iso = start_time_iso.isoformat() if pd.notna(start_time_iso) else None
        
        record: Dict[str, object] = {
            "season": int(season) if season is not None else None,
            "week": week,
            "game_id": game_id,
            "game_date": game_day_date,
            "home_team": home_team,
            "away_team": away_team,
            "season_type": getattr(game, "season_type", None),
            "start_time_utc": start_time_iso,
            "odds_snapshot_generated_utc": collected_utc,
            "event_id": event_id,
        }
        record.update(feature_row)
        
        for label in snapshot_labels:
            column = SNAPSHOT_TS_COLUMNS.get(label)
            if not column:
                continue
            ts_value = snapshot_ts.get(label)
            if ts_value is None:
                record[column] = None
            else:
                if isinstance(ts_value, pd.Timestamp):
                    record[column] = ts_value.tz_convert("UTC").isoformat()
                else:
                    record[column] = pd.to_datetime(ts_value, utc=True, errors="coerce")
                    record[column] = record[column].isoformat() if pd.notna(record[column]) else None
        
        key = (int(record["season"]), week)
        snapshot_records.setdefault(key, []).append(record)
    
    for (season, week), records in snapshot_records.items():
        if not records:
            continue
        df_week = pl.DataFrame(records)
        if df_week.is_empty():
            continue
        
        df_week = df_week.unique(subset=["game_id"], keep="last")
        
        out_dir = ODDS_SNAPSHOT_DIR / f"season={season}" / f"week={_format_week(week)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part.parquet"
        
        df_week.write_parquet(out_path, compression="zstd")
        written_paths.append(out_path)
    
    if written_paths:
        logger.info("collect_odds_snapshots: wrote %d snapshot partitions for seasons %s.", len(written_paths), seasons)
    else:
        logger.info("collect_odds_snapshots: no odds snapshots written for seasons %s.", seasons)
    
    return written_paths


# =============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# =============================================================================

# For backwards compatibility with existing imports
attach_odds_features = attach_player_odds_features
add_odds_features_to_df = add_player_odds_features_to_df


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "CACHE_DIR",
    "DEFAULT_REGIONS",
    "DELTA_SEQUENCE",
    "NFL_ODDS_COLUMNS",
    "ODDS_FLAG_COLUMNS",
    "ODDS_START_DATE",
    "OPEN_START_HOUR_UTC",
    "PLAYER_ODDS_NUMERIC_COLS",
    "SNAPSHOT_HOUR_UTC",
    "SNAPSHOT_SEQUENCE",
    "SNAPSHOT_SUFFIXES",
    "SPORT_KEY",
    # Client
    "OddsAPIClient",
    # Helpers
    "snapshot_datetime",
    "_american_to_prob",
    "_build_default_row",
    "_cache_read",
    "_cache_write",
    "_compute_game_odds",
    # Game-level features
    "add_nfl_odds_features_to_df",
    # Player-level features
    "attach_player_odds_features",
    "add_player_odds_features_to_df",
    # Backwards compatibility
    "attach_odds_features",
    "add_odds_features_to_df",
    # Collection
    "collect_odds_snapshots",
]

