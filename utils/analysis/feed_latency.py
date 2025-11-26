from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt

from utils.general.paths import PROJ_ROOT

ARRIVAL_ROOT = PROJ_ROOT / "cache" / "collect" / "feed_arrivals"
OUTPUT_DIR = PROJ_ROOT / "output" / "metrics" / "feed_latency"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_arrival_logs() -> pd.DataFrame:
    if not ARRIVAL_ROOT.exists():
        logger.warning("Arrival log root missing at %s", ARRIVAL_ROOT)
        return pd.DataFrame()
    parquet_files = list(ARRIVAL_ROOT.glob("*/*.parquet"))
    if not parquet_files:
        logger.warning("No feed arrival parquet logs found under %s", ARRIVAL_ROOT)
        return pd.DataFrame()
    frames = []
    for path in parquet_files:
        try:
            frame = pd.read_parquet(path)
            frame["source_path"] = str(path)
            frames.append(frame)
        except Exception as exc:  # pragma: no cover - diagnostics
            logger.warning("Failed to load %s: %s", path, exc)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    time_cols = ["game_start_ts", "feed_timestamp", "feed_timestamp_min", "collected_at"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _plot_histogram(df: pd.DataFrame, feed: str, label: str) -> None:
    data = df["minutes_until_kickoff"].dropna()
    if data.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=40, color="#1f77b4", edgecolor="white")
    plt.title(f"{feed} ({label}) minutes until kickoff")
    plt.xlabel("Minutes until kickoff (positive = before)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / f"{feed}_{label}_lag_hist.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Feed latency histogram written → %s", plot_path)


def _summarise(df: pd.DataFrame, feed: str, label: str) -> dict[str, object]:
    stats = df["minutes_until_kickoff"].dropna()
    if stats.empty:
        return {
            "feed": feed,
            "snapshot_label": label,
            "count": 0,
        }
    quantiles = stats.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    return {
        "feed": feed,
        "snapshot_label": label,
        "count": int(stats.count()),
        "mean_minutes": stats.mean(),
        "std_minutes": stats.std(),
        "min_minutes": stats.min(),
        "max_minutes": stats.max(),
        "p10_minutes": quantiles.get(0.1),
        "p25_minutes": quantiles.get(0.25),
        "median_minutes": quantiles.get(0.5),
        "p75_minutes": quantiles.get(0.75),
        "p90_minutes": quantiles.get(0.9),
    }


def main() -> None:
    df = _load_arrival_logs()
    if df.empty:
        logger.warning("No arrival logs to summarise.")
        return
    df["minutes_until_kickoff"] = df["minutes_until_kickoff"].astype(float)
    summary_rows: list[dict[str, object]] = []
    grouped = df.groupby(["feed", "snapshot_label"], dropna=False)
    for (feed, label), group in grouped:
        label = label or "default"
        summary = _summarise(group, feed, label)
        summary_rows.append(summary)
        _plot_histogram(group, str(feed), str(label))
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = OUTPUT_DIR / "feed_latency_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Feed latency summary written → %s", summary_path)


if __name__ == "__main__":
    main()


