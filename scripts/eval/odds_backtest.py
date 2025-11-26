#!/usr/bin/env python3
"""
Compute baseline metrics for an odds-only anytime-TD signal.

This is a lightweight diagnostic runner intended to track whether horizon-aligned
odds snapshots remain predictive once the cutoff logic is tightened.

Example:
    PYTHONPATH=. ./venv/bin/python scripts/eval/odds_backtest.py \
        --start-date 2024-09-01 --end-date 2024-12-31 \
        --score-column market_anytime_td_prob \
        --tag odds_cutoff_v2 --horizon-hours 6
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from utils.general.paths import FINAL_FEATURES_PARQUET, PROJ_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Odds-only anytime TD baseline.")
    parser.add_argument("--start-date", type=str, help="ISO date (inclusive) filter for game_date.")
    parser.add_argument("--end-date", type=str, help="ISO date (inclusive) filter for game_date.")
    parser.add_argument(
        "--score-column",
        type=str,
        default="market_anytime_td_prob",
        help="Column to use as the odds-derived probability signal.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="anytime_td",
        help="Binary target column.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=float,
        default=None,
        help="Optional annotation for the decision horizon (purely informational).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJ_ROOT / "output" / "metrics" / "baselines" / "odds_horizon.json",
        help="Where to append the JSON summary.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional label to distinguish multiple runs (e.g., 'pre_cutoff_fix').",
    )
    return parser.parse_args()


def load_frame(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix missing at {path}")
    return pl.read_parquet(path)


def filter_frame(df: pl.DataFrame, start: str | None, end: str | None) -> pl.DataFrame:
    if "game_date" not in df.columns:
        return df
    expr = []
    if start:
        expr.append(pl.col("game_date") >= start)
    if end:
        expr.append(pl.col("game_date") <= end)
    if expr:
        df = df.filter(pl.all_horizontal(expr))
    return df


def evaluate(
    df: pl.DataFrame,
    score_col: str,
    target_col: str,
) -> dict[str, Any]:
    if score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' not found in frame.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in frame.")

    subset = (
        df.select([target_col, score_col])
        .drop_nulls()
        .to_pandas()
    )
    if subset.empty:
        raise ValueError("No rows remaining after dropping nulls for odds backtest.")

    y_true = subset[target_col].astype(float)
    y_pred = subset[score_col].astype(float).clip(0.0, 1.0)

    metrics = {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_pred)),
    }
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_pred, labels=[0, 1]))
    except ValueError:
        metrics["log_loss"] = None
    metrics["count"] = int(len(subset))
    return metrics


def save_summary(output_path: Path, summary: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    existing.append(summary)
    output_path.write_text(json.dumps(existing, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    frame = load_frame(FINAL_FEATURES_PARQUET)
    frame = filter_frame(frame, args.start_date, args.end_date)
    metrics = evaluate(frame, args.score_column, args.target_column)
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "score_column": args.score_column,
        "target_column": args.target_column,
        "horizon_hours": args.horizon_hours,
        "tag": args.tag,
        **metrics,
    }
    save_summary(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

