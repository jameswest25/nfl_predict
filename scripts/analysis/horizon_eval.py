#!/usr/bin/env python3
"""
Aggregate cross-horizon availability / anytime-TD metrics.

This utility expects one or more prediction files (CSV or Parquet) that already
contain the model's probability output and the ground-truth target.  Each file
is tagged with a horizon label (e.g., "1.5h", "3h", "6h"), and the script
emits a consolidated metrics table under `output/metrics/horizon_eval/`.

Example
-------
PYTHONPATH=. ./venv/bin/python scripts/analysis/horizon_eval.py \
    --prediction 1.5h=output/predictions/h1p5.csv \
    --prediction 3h=output/predictions/h3.csv \
    --prediction 6h=output/predictions/h6.csv \
    --target anytime_td --prob-column pred_anytime_td
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from utils.general.paths import PROJ_ROOT

OUTPUT_DIR = PROJ_ROOT / "output" / "metrics" / "horizon_eval"


def parse_prediction_arg(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError("Prediction arguments must be label=path.")
    label, path = arg.split("=", 1)
    return label.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model quality across decision horizons.")
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction_arg,
        required=True,
        help="Prediction file argument as label=path. Repeat for each horizon.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="anytime_td",
        help="Binary target column name (default: anytime_td).",
    )
    parser.add_argument(
        "--prob-column",
        type=str,
        default="pred_anytime_td",
        help="Model probability column (default: pred_anytime_td).",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json"],
        default="csv",
        help="Summary output format (default: csv).",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins to use for calibration summaries.",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported prediction file format: {path}")


def calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, bins: int) -> dict[str, float]:
    bin_ids = np.clip((y_prob * bins).astype(int), 0, bins - 1)
    df = pd.DataFrame({"bin": bin_ids, "y_true": y_true, "y_prob": y_prob})
    agg = df.groupby("bin").agg({"y_true": "mean", "y_prob": "mean", "*": "count"})
    agg = agg.rename(columns={"y_true": "observed", "y_prob": "predicted", "*": "count"})
    mse = ((agg["observed"] - agg["predicted"]) ** 2).fillna(0.0)
    return {"calibration_mse": float(mse.mean()), "calibration_max": float(mse.max())}


def evaluate_file(label: str, df: pd.DataFrame, target_col: str, prob_col: str, bins: int) -> dict[str, float]:
    subset = df[[target_col, prob_col]].dropna()
    if subset.empty:
        raise ValueError(f"No valid rows for horizon '{label}' (after dropping nulls).")
    y_true = subset[target_col].astype(float).to_numpy()
    y_prob = subset[prob_col].astype(float).clip(0.0, 1.0).to_numpy()

    metrics = {
        "horizon": label,
        "count": int(len(subset)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        metrics["log_loss"] = None
    metrics.update(calibration_summary(y_true, y_prob, bins))
    return metrics


def write_summary(records: Iterable[dict[str, float]], fmt: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"horizon_eval_{timestamp}.{fmt}"

    records = list(records)
    if fmt == "csv":
        pd.DataFrame.from_records(records).to_csv(path, index=False)
    else:
        path.write_text(json.dumps(records, indent=2, sort_keys=True))
    return path


def main() -> None:
    args = parse_args()
    summaries = []
    for label, path in args.prediction:
        df = load_predictions(path)
        metrics = evaluate_file(label, df, args.target, args.prob_column, args.calibration_bins)
        metrics["source_path"] = str(path)
        summaries.append(metrics)

    out_path = write_summary(summaries, args.output_format)
    print(f"Wrote horizon evaluation summary → {out_path}")


if __name__ == "__main__":
    main()

