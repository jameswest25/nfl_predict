from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils.train.injury_availability import train_injury_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train injury availability model")
    parser.add_argument(
        "--feature-matrix",
        type=str,
        default="data/processed/final/processed.parquet",
        help="Path to the feature matrix parquet used for training.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Conformal risk level (alpha) for interval construction.",
    )
    parser.add_argument(
        "--train-seasons",
        type=str,
        default="",
        help="Comma-separated seasons to use for training (defaults to all except calibration/test).",
    )
    parser.add_argument(
        "--calibration-seasons",
        type=str,
        default="",
        help="Comma-separated seasons for calibration set (defaults to penultimate season).",
    )
    parser.add_argument(
        "--test-seasons",
        type=str,
        default="",
        help="Comma-separated seasons for test set (defaults to latest season).",
    )
    return parser.parse_args()


def _parse_season_arg(value: str) -> list[int] | None:
    if not value:
        return None
    return [int(s.strip()) for s in value.split(",") if s.strip()]


def main() -> None:
    args = _parse_args()
    feature_matrix_path = Path(args.feature_matrix).expanduser()
    artifact = train_injury_model(
        feature_matrix_path=feature_matrix_path,
        alpha=args.alpha,
        train_seasons=_parse_season_arg(args.train_seasons),
        calibration_seasons=_parse_season_arg(args.calibration_seasons),
        test_seasons=_parse_season_arg(args.test_seasons),
    )
    logger.info("Trained injury availability model with metrics: %s", artifact.metrics)


if __name__ == "__main__":
    main()


