from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Iterable, Optional, Sequence

import pandas as pd
import polars as pl
import requests
import yaml
from zoneinfo import ZoneInfo

from utils.collect.arrival_log import log_feed_arrivals
from utils.collect.nfl_schedules import get_schedule
from utils.general.paths import (
    PROJ_ROOT,
    STADIUM_COORDS_FILE,
    WEATHER_FORECAST_DIR,
)
from utils.collect.visual_crossing_weather import (
    VISUAL_CROSSING_API_KEY,
    VISUAL_CROSSING_BASE_URL,
    CORE_WEATHER_FIELDS,
    RATE_LIMIT_DELAY,
    get_session,
)

logger = logging.getLogger(__name__)


@dataclass
class ForecastConfig:
    provider: str = "visual_crossing"
    lead_hours: Sequence[int] = (72, 24, 6, 2)
    rate_limit_per_minute: int = 45
    timeout_seconds: int = 30
    include_fields: Sequence[str] = tuple(CORE_WEATHER_FIELDS)


def _load_weather_config() -> ForecastConfig:
    cfg_path = PROJ_ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        logger.warning("Weather config not found at %s; using defaults.", cfg_path)
        return ForecastConfig()

    try:
        config = yaml.safe_load(cfg_path.read_text())
    except Exception as exc:
        logger.warning("Failed to parse weather config: %s", exc)
        return ForecastConfig()

    weather_cfg = (config or {}).get("weather", {})
    forecast_cfg = (weather_cfg or {}).get("forecast", {}) or {}
    leads = forecast_cfg.get("lead_hours") or ForecastConfig.lead_hours
    include_fields = forecast_cfg.get("include_fields") or CORE_WEATHER_FIELDS
    return ForecastConfig(
        provider=str(forecast_cfg.get("provider", "visual_crossing")),
        lead_hours=tuple(int(h) for h in leads),
        rate_limit_per_minute=int(forecast_cfg.get("rate_limit_per_minute", 45)),
        timeout_seconds=int(forecast_cfg.get("timeout_seconds", 30)),
        include_fields=tuple(include_fields),
    )


def _load_stadium_coords() -> dict[str, dict]:
    with STADIUM_COORDS_FILE.open() as fh:
        return json.load(fh)


def _resolve_stadium_meta(stadium_key: str | None, home_team: str | None, coords_map: dict[str, dict]) -> dict:
    if stadium_key and stadium_key in coords_map:
        return coords_map[stadium_key] or {}
    # fallback: use home team key
    if home_team:
        home_meta = coords_map.get(home_team.upper())
        if home_meta:
            return home_meta
    return {}


def _rate_limit_sleep(rate_limit_per_minute: int) -> float:
    if rate_limit_per_minute <= 0:
        return RATE_LIMIT_DELAY
    return max(RATE_LIMIT_DELAY, 60.0 / float(rate_limit_per_minute))


def _fetch_hourly_forecast(
    lat: float,
    lon: float,
    start_local: dt.datetime,
    end_local: dt.datetime,
    api_key: str,
    timeout_seconds: int,
    include_fields: Sequence[str],
) -> pd.DataFrame | None:
    """
    Fetch hourly forecast (or historical) data for the provided window.
    """
    session = get_session()
    location = f"{lat:.4f},{lon:.4f}"
    start_str = start_local.strftime("%Y-%m-%d")
    end_str = end_local.strftime("%Y-%m-%d")

    params = {
        "unitGroup": "us",
        "contentType": "json",
        "include": "hours",
        "options": "nonulls",
        "elements": ",".join({"datetime", "datetimeEpoch", *include_fields}),
        "key": api_key,
    }

    url = f"{VISUAL_CROSSING_BASE_URL}/{location}/{start_str}/{end_str}"
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout_seconds)
            if resp.status_code == 429:
                logger.warning(
                    "Visual Crossing rate limit reached (HTTP 429). "
                    "Attempt %d/%d – sleeping and retrying.",
                    attempt,
                    retries,
                )
                sleep(min(120, 15 * attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            days = data.get("days") or []
            rows: list[dict] = []
            for day in days:
                for hour in day.get("hours", []):
                    # Visual Crossing returns datetimeEpoch in seconds (UTC)
                    rows.append(hour)
            if not rows:
                return None
            frame = pd.DataFrame(rows)
            if "datetimeEpoch" in frame.columns:
                frame["datetime_utc"] = pd.to_datetime(frame["datetimeEpoch"], unit="s", utc=True)
            else:
                frame["datetime_utc"] = pd.NaT
            return frame
        except requests.RequestException as exc:
            logger.warning("Failed to fetch forecast (attempt %d/%d): %s", attempt, retries, exc)
            sleep(min(90, 10 * attempt))
        except ValueError as exc:
            logger.warning("Failed to parse forecast payload: %s", exc)
            break
    return None


def _select_hour(frame: pd.DataFrame, target_utc: dt.datetime) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    if "datetime_utc" not in frame.columns:
        return None
    # Compute absolute difference in seconds, prefer closest hour
    diffs = (frame["datetime_utc"] - target_utc).abs().dt.total_seconds()
    if diffs.isna().all():
        return None
    idx = diffs.idxmin()
    if pd.isna(idx):
        return None
    closest = frame.loc[idx]
    if pd.isna(closest.get("datetime_utc")):
        return None
    # Ensure the closest observation is within 3 hours
    if abs(diffs[idx]) > 3 * 3600:
        logger.debug(
            "Closest forecast hour for %s is %.1f hours away; treating as missing.",
            target_utc,
            diffs[idx] / 3600.0,
        )
        return None
    return closest


def collect_weather_forecasts(
    seasons: Sequence[int],
    *,
    lead_hours: Sequence[int] | None = None,
    limit_weeks: int | None = None,
    force_refresh: bool = False,
) -> list[Path]:
    """
    Collect forecast snapshots at fixed lead times for the provided seasons.

    Parameters
    ----------
    seasons : Sequence[int]
        NFL seasons to collect (e.g., [2025, 2024]).
    lead_hours : Sequence[int], optional
        Custom lead times (in hours before kickoff). Defaults to config.
    limit_weeks : int, optional
        If provided, only collect for weeks <= limit_weeks (useful for testing).
    force_refresh : bool, default False
        If True, overwrite existing snapshots for the same (game_id, lead_hours).

    Returns
    -------
    list[Path]
        Paths of parquet partitions written or updated.
    """
    if not seasons:
        logger.info("No seasons provided for weather forecast collection.")
        return []

    cfg = _load_weather_config()
    leads = sorted(set(int(h) for h in (lead_hours or cfg.lead_hours)), reverse=True)
    rate_sleep = _rate_limit_sleep(cfg.rate_limit_per_minute)

    if cfg.provider.lower() != "visual_crossing":
        logger.error("Unsupported weather provider: %s", cfg.provider)
        return []
    if not VISUAL_CROSSING_API_KEY:
        logger.warning("Visual Crossing API key missing; skipping forecast collection.")
        return []

    schedule = get_schedule(list(seasons))
    if schedule.empty:
        logger.warning("Schedule is empty; cannot collect forecasts.")
        return []

    if limit_weeks is not None:
        try:
            limit_weeks_int = int(limit_weeks)
            schedule = schedule[schedule["week"].astype("Int64") <= limit_weeks_int]
        except Exception:
            logger.warning("Invalid limit_weeks=%s; ignoring.", limit_weeks)

    schedule = schedule.dropna(subset=["game_id"]).copy()
    if schedule.empty:
        logger.info("No scheduled games found for requested seasons/weeks.")
        return []

    stadium_coords = _load_stadium_coords()
    now_utc = dt.datetime.now(dt.timezone.utc)

    rows: list[dict] = []
    games_processed = 0

    for game in schedule.itertuples(index=False):
        game_id = getattr(game, "game_id", None)
        season = int(getattr(game, "season", 0) or 0)
        week = getattr(game, "week", None)
        home_team = getattr(game, "home_team", None)
        away_team = getattr(game, "away_team", None)
        season_type = getattr(game, "season_type", None)

        if not game_id or pd.isna(game_id):
            continue

        game_start = getattr(game, "start_time_utc", None)
        if pd.isna(game_start):
            game_start = None
        if game_start is None:
            gameday = getattr(game, "gameday", None)
            if pd.isna(gameday):
                gameday = None
            if gameday is not None:
                game_start = pd.to_datetime(gameday).tz_localize("UTC") + pd.Timedelta(hours=21)
        if game_start is None:
            logger.debug("Skipping game %s – missing start time.", game_id)
            continue
        if not isinstance(game_start, pd.Timestamp):
            try:
                game_start = pd.to_datetime(game_start, utc=True)
            except Exception:
                continue
        else:
            game_start = game_start.tz_convert("UTC")

        venue_key = getattr(game, "stadium_key", None)
        if pd.isna(venue_key):
            venue_key = None
        venue_meta = _resolve_stadium_meta(
            venue_key,
            getattr(game, "home_team", None),
            stadium_coords,
        )
        lat = venue_meta.get("lat")
        lon = venue_meta.get("lon")
        tz_name = venue_meta.get("tz", "US/Eastern")
        stadium_name = venue_meta.get("stadium")

        if lat is None or lon is None:
            logger.debug("Skipping game %s – missing stadium coordinates.", game_id)
            continue

        try:
            local_tz = ZoneInfo(tz_name)
        except Exception:
            logger.debug("Invalid timezone %s for game %s; defaulting to US/Eastern.", tz_name, game_id)
            local_tz = ZoneInfo("US/Eastern")

        game_start_local = game_start.tz_convert(local_tz)

        pad_hours = max(leads) + 24 if leads else 96
        start_local = game_start_local - dt.timedelta(hours=pad_hours)
        end_local = game_start_local + dt.timedelta(hours=12)

        frame = _fetch_hourly_forecast(
            lat=lat,
            lon=lon,
            start_local=start_local,
            end_local=end_local,
            api_key=VISUAL_CROSSING_API_KEY,
            timeout_seconds=cfg.timeout_seconds,
            include_fields=cfg.include_fields,
        )

        sleep(rate_sleep)

        if frame is None:
            logger.debug("No forecast data returned for game %s.", game_id)
            continue

        row_hour = _select_hour(frame, game_start)
        if row_hour is None:
            logger.debug("Unable to align forecast hour for game %s.", game_id)
            continue

        forecast_valid_ts = pd.to_datetime(row_hour["datetime_utc"], utc=True)
        ingested_at = dt.datetime.now(dt.timezone.utc)

        for lead in leads:
            lead_delta = dt.timedelta(hours=float(lead))
            forecast_generated_ts = (game_start.to_pydatetime() - lead_delta).replace(tzinfo=dt.timezone.utc)

            # Only emit snapshots when we are at or after the expected generation time
            if forecast_generated_ts > now_utc + dt.timedelta(minutes=1):
                continue

            forecast_is_backfill = forecast_generated_ts < (ingested_at - dt.timedelta(minutes=30))
            forecast_is_historical = game_start.to_pydatetime() < now_utc

            precip_type_val = row_hour.get("preciptype")
            if isinstance(precip_type_val, list):
                precip_type_val = ",".join(sorted(set(str(x) for x in precip_type_val if x)))

        dew_point_val = row_hour.get("dewpoint")
        if dew_point_val is None:
            dew_point_val = row_hour.get("dew")
        rows.append(
                {
                    "game_id": str(game_id),
                    "season": int(season),
                    "week": int(week) if week is not None and not pd.isna(week) else None,
                    "season_type": str(season_type) if season_type is not None else None,
                    "home_team": str(home_team) if home_team is not None else None,
                    "away_team": str(away_team) if away_team is not None else None,
                    "stadium_key": venue_key or venue_meta.get("code"),
                    "stadium_name": stadium_name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "forecast_provider": cfg.provider,
                    "forecast_lead_hours": float(lead),
                    "forecast_generated_ts": forecast_generated_ts,
                    "forecast_valid_ts": forecast_valid_ts.to_pydatetime(),
                    "forecast_available_ts": ingested_at,
                    "forecast_is_backfill": bool(forecast_is_backfill),
                    "forecast_is_historical": bool(forecast_is_historical),
                    "forecast_source_detail": "historical_backfill" if forecast_is_historical else "live_pull",
                    "ingested_at": ingested_at,
                    "temp_f": row_hour.get("temp"),
                    "feels_like_f": row_hour.get("feelslike"),
                    "humidity_pct": row_hour.get("humidity"),
                "dew_point_f": dew_point_val,
                    "pressure_mb": row_hour.get("pressure"),
                    "wind_speed_mph": row_hour.get("windspeed"),
                    "wind_gust_mph": row_hour.get("windgust"),
                    "wind_direction_deg": row_hour.get("winddir"),
                    "cloud_cover_pct": row_hour.get("cloudcover"),
                    "visibility_miles": row_hour.get("visibility"),
                    "precip_amount_in": row_hour.get("precip"),
                    "precip_probability_pct": row_hour.get("precipprob"),
                    "snow_amount_in": row_hour.get("snow"),
                    "conditions": row_hour.get("conditions"),
                    "precip_type": precip_type_val,
                }
            )

        games_processed += 1

    if not rows:
        logger.info("No forecast snapshots collected for seasons %s.", seasons)
        return []

    df = pl.from_dicts(rows)

    # Ensure datatypes
    time_cols = [
        "forecast_generated_ts",
        "forecast_valid_ts",
        "forecast_available_ts",
        "ingested_at",
    ]
    for col in time_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Datetime("ms", "UTC")))

    df = df.with_columns(
        [
            pl.col("forecast_lead_hours").cast(pl.Float32),
            pl.col("temp_f").cast(pl.Float32),
            pl.col("feels_like_f").cast(pl.Float32),
            pl.col("humidity_pct").cast(pl.Float32),
            pl.col("dew_point_f").cast(pl.Float32),
            pl.col("pressure_mb").cast(pl.Float32),
            pl.col("wind_speed_mph").cast(pl.Float32),
            pl.col("wind_gust_mph").cast(pl.Float32),
            pl.col("wind_direction_deg").cast(pl.Float32),
            pl.col("cloud_cover_pct").cast(pl.Float32),
            pl.col("visibility_miles").cast(pl.Float32),
            pl.col("precip_amount_in").cast(pl.Float32),
            pl.col("precip_probability_pct").cast(pl.Float32),
            pl.col("snow_amount_in").cast(pl.Float32),
        ]
    )

    written: list[Path] = []
    for (season, week), part in df.group_by(["season", "week"], maintain_order=True):
        partition_dir = WEATHER_FORECAST_DIR / f"season={int(season)}" / f"week={int(week) if week is not None else 0}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        path = partition_dir / "part.parquet"

        if path.exists():
            existing = pl.read_parquet(str(path))
            combined = pl.concat([existing, part], how="vertical_relaxed")
            combined = combined.sort(["game_id", "forecast_lead_hours", "ingested_at"])
            combined = combined.unique(subset=["game_id", "forecast_lead_hours"], keep="last")
            combined.write_parquet(str(path), compression="zstd")
        else:
            part.write_parquet(str(path), compression="zstd")

        written.append(path)
        logger.info(
            "Weather forecasts written → %s (%d rows)",
            path,
            part.height,
        )
        _log_weather_arrivals(part)

    logger.info(
        "Collected %d forecast snapshots across %d games (seasons=%s).",
        df.height,
        games_processed,
        seasons,
    )

    return written


def _log_weather_arrivals(part: pl.DataFrame) -> None:
    """Log feed arrival metrics for weather forecasts."""
    try:
        if part.is_empty():
            return
        df_pd = part.to_pandas()
        if df_pd.empty:
            return
        rows_by_lead: dict[str, list[dict[str, object]]] = {}
        for _, row in df_pd.iterrows():
            game_id = row.get("game_id")
            if not isinstance(game_id, str):
                game_id = str(game_id) if game_id is not None else None
            season = row.get("season")
            week = row.get("week")
            lead = row.get("forecast_lead_hours")
            lead_label = f"{int(lead)}h" if lead is not None and not pd.isna(lead) else "unknown"
            game_start_ts = pd.to_datetime(row.get("forecast_valid_ts"), utc=True, errors="coerce")
            collected_at = pd.to_datetime(row.get("forecast_available_ts"), utc=True, errors="coerce")
            rows_by_lead.setdefault(lead_label, []).append(
                {
                    "season": int(season) if season is not None and not pd.isna(season) else None,
                    "week": int(week) if week is not None and not pd.isna(week) else None,
                    "game_id": game_id,
                    "team": None,
                    "game_start_ts": None if pd.isna(game_start_ts) else game_start_ts.to_pydatetime(),
                    "feed_timestamp": None if pd.isna(collected_at) else collected_at.to_pydatetime(),
                    "collected_at": None if pd.isna(collected_at) else collected_at.to_pydatetime(),
                    "metadata": {
                        "forecast_generated_ts": _safe_to_datetime(row.get("forecast_generated_ts")),
                        "forecast_lead_hours": lead,
                        "provider": row.get("forecast_provider"),
                    },
                }
            )
        for label, rows in rows_by_lead.items():
            if rows:
                log_feed_arrivals("weather", rows, snapshot_label=label)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.warning("Failed to log weather arrival metrics: %s", exc)


def _safe_to_datetime(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts.isoformat()


__all__ = ["collect_weather_forecasts"]

