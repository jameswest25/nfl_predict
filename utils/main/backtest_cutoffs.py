from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from pipeline.feature import build_feature_matrix
from pipeline.train import ModelTrainer
from utils.feature.asof import decision_cutoff_horizons, decision_cutoff_override
from utils.general.constants import format_cutoff_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _collect_metrics(trainer: ModelTrainer) -> Iterable[dict[str, object]]:
    results: list[dict[str, object]] = []
    problems = [p["name"] for p in trainer.problems] if isinstance(trainer.problems, list) else []
    model_names = trainer.config["training"]["models_to_train"]
    for problem in problems:
        for model_name in model_names:
            metrics_path = (
                trainer.paths.metric_dir / problem / model_name / trainer.run_id / "metrics.yaml"
            )
            if not metrics_path.exists():
                legacy_path = trainer.paths.metric_dir / f"{problem}_{model_name}.yaml"
                metrics_path = legacy_path if legacy_path.exists() else metrics_path
            if not metrics_path.exists():
                logger.warning("Metrics not found for %s/%s (expected %s)", problem, model_name, metrics_path)
                continue
            with metrics_path.open("r") as fh:
                metrics = yaml.safe_load(fh) or {}
            record = {"problem": problem, "model": model_name}
            record.update(metrics)
            results.append(record)
    return results


def main() -> None:
    horizons = decision_cutoff_horizons()
    if not horizons:
        logger.error("No cutoff horizons configured; aborting.")
        return

    summary_records: list[dict[str, object]] = []
    recompute_flag = True

    for hours in horizons:
        label = format_cutoff_label(hours)
        logger.info("=== Evaluating cutoff horizon %.2f hours (%s) ===", hours, label)

        build_feature_matrix(
            recompute_intermediate=recompute_flag,
            cutoff_hours_list=[hours],
        )
        recompute_flag = False

        with decision_cutoff_override(cutoff_hours=hours):
            trainer = ModelTrainer(
                overrides={
                    "run_tag": f"cutoff_{label}_bt",
                }
            )
            trainer.run()

            metrics_entries = _collect_metrics(trainer)
            for metrics in metrics_entries:
                record = {
                    "cutoff_hours": hours,
                    "cutoff_label": label,
                    **metrics,
                }
                summary_records.append(record)

    if not summary_records:
        logger.warning("No metrics collected; nothing to write.")
        return

    summary_df = pd.DataFrame(summary_records)
    summary_path = Path("output/metrics/cutoff_backtest_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Backtest summary metrics written → %s", summary_path)


if __name__ == "__main__":
    main()


