from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from utils.general.paths import PROJ_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_frame(feature_path: Path, seasons: list[int] | None = None) -> pd.DataFrame:
    scan = pl.scan_parquet(str(feature_path))
    schema = scan.collect_schema()
    available = schema.names()
    columns = [
        "season",
        "week",
        "injury_is_inactive_designation",
        "injury_inactive_probability",
    ]
    if "injury_inactive_probability_model" in available:
        columns.append("injury_inactive_probability_model")
    if "injury_inactive_probability_source" in available:
        columns.append("injury_inactive_probability_source")
    if "injury_inactive_probability_interval_width" in available:
        columns.append("injury_inactive_probability_interval_width")
    scan = scan.select([pl.col(col) for col in columns if col in available])
    if seasons:
        scan = scan.filter(pl.col("season").is_in(seasons))
    df = scan.collect(streaming=True).to_pandas()
    df.dropna(subset=["injury_is_inactive_designation"], inplace=True)
    df["injury_is_inactive_designation"] = df["injury_is_inactive_designation"].astype(int)
    return df


def _weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = df.groupby(["season", "week"])
    for (season, week), frame in grouped:
        if frame.empty:
            continue
        y_true = frame["injury_is_inactive_designation"].to_numpy()
        proba = frame["injury_inactive_probability"].to_numpy()
        try:
            auc = roc_auc_score(y_true, proba)
        except ValueError:
            auc = np.nan
        pr_auc = average_precision_score(y_true, proba)
        brier = brier_score_loss(y_true, proba)
        frac_positive = float(y_true.mean()) if y_true.size else np.nan
        fallback_rate = np.nan
        mean_interval = np.nan
        if "injury_inactive_probability_source" in frame.columns:
            source_series = frame["injury_inactive_probability_source"].astype(str).str.lower()
            fallback_rate = float((source_series == "heuristic").mean())
        if "injury_inactive_probability_interval_width" in frame.columns:
            mean_interval = float(frame["injury_inactive_probability_interval_width"].astype(float).mean())
        records.append(
            {
                "season": season,
                "week": week,
                "rows": len(frame),
                "auc": auc,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "inactive_rate": frac_positive,
                "fallback_rate": fallback_rate,
                "mean_interval_width": mean_interval,
            }
        )
    return pd.DataFrame(records)


def _calibration_table(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    y_true = df["injury_is_inactive_designation"].to_numpy()
    proba = df["injury_inactive_probability"].to_numpy()
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor injury availability model calibration.")
    parser.add_argument(
        "--feature-matrix",
        type=str,
        default="data/processed/final/processed.parquet",
        help="Feature matrix to evaluate.",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="",
        help="Comma separated seasons to include (default: latest season).",
    )
    args = parser.parse_args()
    feature_path = Path(args.feature_matrix).expanduser()
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    if not seasons:
        scan = pl.scan_parquet(str(feature_path)).select("season")
        seasons = [int(scan.collect().to_series().max())]
    frame = _load_frame(feature_path, seasons)
    if frame.empty:
        logger.warning("No rows found for seasons %s", seasons)
        return
    weekly = _weekly_metrics(frame)
    calib = _calibration_table(frame, bins=10)
    output_dir = PROJ_ROOT / "output" / "metrics" / "injury_availability" / "monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(output_dir / "weekly_metrics.csv", index=False)
    calib.to_csv(output_dir / "calibration_curve.csv", index=False)
    logger.info("Weekly metrics and calibration curve written to %s", output_dir)


if __name__ == "__main__":
    main()


