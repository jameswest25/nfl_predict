#!/usr/bin/env python3
"""
Standardized backtest runner for model problems/horizons.

This script is intentionally lightweight: it expects that the feature matrix
already contains both the target and the model prediction columns for each
problem/horizon.  It computes AUC, Brier, log-loss, and optional calibration
metrics per problem and dumps the results into
`output/metrics/baselines/<timestamp>_<tag>.json`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from utils.general.paths import PROJ_ROOT, FINAL_FEATURES_PARQUET


def parse_problem(arg: str) -> tuple[str, str]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError("Problem args must be name=prob_column.")
    name, column = arg.split("=", 1)
    return name.strip(), column.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest existing model predictions.")
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=FINAL_FEATURES_PARQUET,
        help="Feature matrix path (default: data/processed/final/processed.parquet).",
    )
    parser.add_argument(
        "--problem",
        action="append",
        type=parse_problem,
        required=True,
        help="Problem spec as name=prediction_column. Repeat per problem.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="anytime_td",
        help="Target column (default: anytime_td).",
    )
    parser.add_argument(
        "--season",
        type=int,
        nargs="*",
        help="Optional season filters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJ_ROOT / "output" / "metrics" / "baselines",
        help="Directory to store summary JSON.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional label to differentiate runs.",
    )
    return parser.parse_args()


def filter_seasons(df: pl.DataFrame, seasons: Sequence[int] | None) -> pl.DataFrame:
    if not seasons or "season" not in df.columns:
        return df
    return df.filter(pl.col("season").is_in([int(s) for s in seasons]))


def evaluate_problem(df: pl.DataFrame, target_col: str, prediction_col: str) -> dict[str, float]:
    subset = df.select([target_col, prediction_col]).drop_nulls()
    if subset.is_empty():
        raise ValueError(f"No rows for column {prediction_col} after drop_nulls.")
    pdf = subset.to_pandas()
    y_true = pdf[target_col].astype(float)
    y_prob = pdf[prediction_col].astype(float).clip(0.0, 1.0)
    metrics = {
        "count": int(len(pdf)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        metrics["log_loss"] = None
    return metrics


def main() -> None:
    args = parse_args()
    df = pl.read_parquet(args.feature_path)
    df = filter_seasons(df, args.season)

    summaries = []
    for name, pred_col in args.problem:
        if pred_col not in df.columns:
            raise ValueError(f"Prediction column '{pred_col}' missing for problem '{name}'.")
        metrics = evaluate_problem(df, args.target, pred_col)
        metrics.update(
            {
                "problem": name,
                "prediction_column": pred_col,
                "target_column": args.target,
                "seasons": args.season,
            }
        )
        summaries.append(metrics)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or "backtest"
    out_path = out_dir / f"{timestamp}_{tag}.json"
    out_path.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    print(f"Wrote backtest summary → {out_path}")


if __name__ == "__main__":
    main()

