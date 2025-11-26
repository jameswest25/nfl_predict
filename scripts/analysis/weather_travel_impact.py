#!/usr/bin/env python3
"""
Generate weather and travel impact summaries/plots for anytime TD outcomes.

The script pulls the final feature matrix (`data/processed/final/processed.parquet`)
and produces:
  - Hit-rate tables grouped by weather/travel features
  - Seaborn bar plots saved under `docs/analysis/weather_travel/`
  - A JSON summary file for downstream reporting
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.general.paths import PROJ_ROOT, FINAL_FEATURES_PARQUET

OUTPUT_DIR = PROJ_ROOT / "docs" / "analysis" / "weather_travel"
SUMMARY_DIR = PROJ_ROOT / "output" / "metrics" / "weather_travel"

DEFAULT_WEATHER_FLAGS = [
    "weather_bad_passing_flag",
    "roof_is_indoor_flag",
    "weather_extreme_wind_flag",
    "weather_freezing_flag",
    "weather_heavy_precip_flag",
]

DEFAULT_TRAVEL_FEATURES = [
    "travel_rest_days",
    "travel_rest_days_l3",
    "rest_days_since_last_game",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TD hit rates vs weather/travel features.")
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=FINAL_FEATURES_PARQUET,
        help="Parquet file with final feature matrix.",
    )
    parser.add_argument(
        "--weather-flags",
        nargs="*",
        default=DEFAULT_WEATHER_FLAGS,
        help="Weather/roof binary columns to evaluate.",
    )
    parser.add_argument(
        "--travel-features",
        nargs="*",
        default=DEFAULT_TRAVEL_FEATURES,
        help="Travel/rest numeric columns to bucket.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="anytime_td",
        help="Target column to measure hit rate (default: anytime_td).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=4,
        help="Number of quantile bins for travel features.",
    )
    return parser.parse_args()


def load_frame(path: Path, target_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix missing at {path}")
    df = pd.read_parquet(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in feature matrix.")
    return df


def summarize_flags(df: pd.DataFrame, flags: Sequence[str], target: str) -> list[dict[str, float]]:
    summaries = []
    for flag in flags:
        if flag not in df.columns:
            continue
        subset = df[[flag, target]].dropna()
        if subset.empty:
            continue
        pivot = subset.groupby(flag)[target].agg(["mean", "count"]).reset_index()
        for _, row in pivot.iterrows():
            summaries.append(
                {
                    "feature": flag,
                    "bucket": int(row[flag]),
                    "hit_rate": float(row["mean"]),
                    "count": int(row["count"]),
                }
            )
    return summaries


def summarize_travel(df: pd.DataFrame, features: Sequence[str], target: str, bins: int) -> list[dict[str, float]]:
    summaries = []
    for feature in features:
        if feature not in df.columns:
            continue
        series = df[[feature, target]].dropna()
        if series.empty:
            continue
        try:
            series["bucket"] = pd.qcut(series[feature], bins, duplicates="drop")
        except ValueError:
            continue
        pivot = series.groupby("bucket")[target].agg(["mean", "count"]).reset_index()
        for _, row in pivot.iterrows():
            summaries.append(
                {
                    "feature": feature,
                    "bucket": str(row["bucket"]),
                    "hit_rate": float(row["mean"]),
                    "count": int(row["count"]),
                }
            )
    return summaries


def plot_flag(df: pd.DataFrame, flag: str, target: str, output_path: Path) -> None:
    subset = df[[flag, target]].dropna()
    if subset.empty:
        return
    plt.figure(figsize=(4, 3))
    sns.barplot(subset, x=flag, y=target, estimator="mean")
    plt.ylabel("TD Hit Rate")
    plt.xlabel(flag)
    plt.title(f"{flag} vs {target}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_travel(df: pd.DataFrame, feature: str, target: str, bins: int, output_path: Path) -> None:
    subset = df[[feature, target]].dropna()
    if subset.empty:
        return
    try:
        subset["bucket"] = pd.qcut(subset[feature], bins, duplicates="drop")
    except ValueError:
        return
    plt.figure(figsize=(5, 3))
    sns.barplot(subset, x="bucket", y=target, estimator="mean")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("TD Hit Rate")
    plt.xlabel(feature)
    plt.title(f"{feature} bins vs {target}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    df = load_frame(args.feature_path, args.target)

    flag_summary = summarize_flags(df, args.weather_flags, args.target)
    travel_summary = summarize_travel(df, args.travel_features, args.target, args.bins)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    for flag in args.weather_flags:
        if flag in df.columns:
            plot_flag(df, flag, args.target, OUTPUT_DIR / f"{timestamp}_{flag}.png")
    for feature in args.travel_features:
        if feature in df.columns:
            plot_travel(df, feature, args.target, args.bins, OUTPUT_DIR / f"{timestamp}_{feature}.png")

    summary = {
        "generated_at": timestamp,
        "feature_path": str(args.feature_path),
        "target": args.target,
        "weather_flags": flag_summary,
        "travel_features": travel_summary,
    }
    summary_path = SUMMARY_DIR / f"weather_travel_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote weather/travel impact summary → {summary_path}")
    print(f"Charts saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

