from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional

import pandas as pd

from utils.general.paths import PROJ_ROOT

logger = logging.getLogger(__name__)

ARRIVAL_LOG_ROOT = PROJ_ROOT / "cache" / "collect" / "feed_arrivals"
ARRIVAL_LOG_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class FeedArrivalRecord:
    feed: str
    snapshot_label: str
    season: Optional[int]
    week: Optional[int]
    game_id: Optional[str]
    team: Optional[str]
    game_start_ts: Optional[dt.datetime]
    feed_timestamp: Optional[dt.datetime]
    feed_timestamp_min: Optional[dt.datetime]
    collected_at: dt.datetime
    metadata: MutableMapping[str, object]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        metadata = payload.pop("metadata", None) or {}
        # Serialise metadata as JSON so downstream tools can parse easily.
        try:
            payload["metadata_json"] = json.dumps(metadata, default=str)
        except TypeError:
            payload["metadata_json"] = json.dumps({k: str(v) for k, v in metadata.items()})
        payload["minutes_until_kickoff"] = _minutes_until_kickoff(
            self.game_start_ts, self.collected_at
        )
        payload["minutes_since_snapshot"] = _minutes_since_snapshot(
            self.feed_timestamp, self.collected_at
        )
        payload["minutes_since_first_snapshot"] = _minutes_since_snapshot(
            self.feed_timestamp_min, self.collected_at
        )
        return payload


def _minutes_until_kickoff(
    kickoff: Optional[dt.datetime], collected_at: dt.datetime
) -> Optional[float]:
    if kickoff is None:
        return None
    delta = kickoff - collected_at
    return round(delta.total_seconds() / 60.0, 3)


def _minutes_since_snapshot(
    snapshot_ts: Optional[dt.datetime], collected_at: dt.datetime
) -> Optional[float]:
    if snapshot_ts is None:
        return None
    delta = collected_at - snapshot_ts
    return round(delta.total_seconds() / 60.0, 3)


def _ensure_utc(value: Optional[dt.datetime]) -> Optional[dt.datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def log_feed_arrivals(
    feed: str,
    records: Iterable[Mapping[str, object]],
    *,
    snapshot_label: str = "default",
) -> None:
    """
    Persist arrival records for downstream latency analysis.

    Parameters
    ----------
    feed
        Feed identifier (e.g., "injuries", "odds", "weather", "rosters").
    records
        Iterable of dictionaries containing the following keys (optional where noted):
            - season (int, optional)
            - week (int, optional)
            - game_id (str, optional)
            - team (str, optional)
            - game_start_ts (datetime, optional, UTC recommended)
            - feed_timestamp (datetime, optional, UTC recommended)
            - feed_timestamp_min (datetime, optional)
            - collected_at (datetime, optional – defaults to now UTC)
            - metadata (mapping, optional) – additional contextual data
    snapshot_label
        Additional label describing the snapshot variant (e.g., "cutoff", "2h").
    """
    rows: list[dict[str, object]] = []
    now_utc = dt.datetime.now(dt.timezone.utc)

    for record in records:
        if not record:
            continue
        metadata = dict(record.get("metadata") or {})
        collected_at = _ensure_utc(record.get("collected_at")) or now_utc
        row = FeedArrivalRecord(
            feed=feed,
            snapshot_label=snapshot_label,
            season=_coerce_int(record.get("season")),
            week=_coerce_int(record.get("week")),
            game_id=_coerce_str(record.get("game_id")),
            team=_coerce_str(record.get("team")),
            game_start_ts=_ensure_utc(record.get("game_start_ts")),
            feed_timestamp=_ensure_utc(record.get("feed_timestamp")),
            feed_timestamp_min=_ensure_utc(record.get("feed_timestamp_min")),
            collected_at=collected_at,
            metadata=metadata,
        )
        rows.append(row.to_dict())

    if not rows:
        return

    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Ensure datetime columns are timezone-aware (UTC)
    datetime_cols = [
        "game_start_ts",
        "feed_timestamp",
        "feed_timestamp_min",
        "collected_at",
    ]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    outfile = (
        ARRIVAL_LOG_ROOT
        / feed
        / f"{snapshot_label}_{dt.datetime.now(dt.timezone.utc):%Y%m%d_%H%M%S_%f}.parquet"
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(outfile, compression="zstd", index=False)
        logger.info(
            "Feed arrival log written for %s/%s → %s (%d rows)",
            feed,
            snapshot_label,
            outfile,
            len(df),
        )
    except Exception as exc:
        logger.warning("Unable to write feed arrival log %s: %s", outfile, exc)


def _coerce_int(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return None


__all__ = ["log_feed_arrivals"]


