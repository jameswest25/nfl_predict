"""Mixture of Experts training wrapper (NFL).

This repo previously carried a standalone MoE training implementation here.
That implementation drifted from the main training pipeline (`pipeline/train.py`)
in several correctness-critical ways (feature selection, purged splits/OOF,
missing-value handling defaults like `fillna(0)`, and artifact naming).

This script now delegates to `pipeline/train.py` so MoE training uses the same
authoritative codepaths as global training (including horizon-safe artifacts).

Usage:
  python pipeline/train_moe.py --tasks usage_targets,usage_carries
  python pipeline/train_moe.py --all

Notes:
- Use `--decision-cutoff-hours` to target a specific horizon.
- Models and inference artifacts are written with the run_id (recommended), plus
  horizon-safe legacy copies for backwards compatibility.
"""

from __future__ import annotations

import argparse
import logging

import yaml

from pipeline.train import train as train_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _moe_tasks(cfg: dict) -> list[str]:
    out: list[str] = []
    for p in cfg.get("problems", []) or []:
        moe_cfg = p.get("per_position_training") or {}
        if moe_cfg.get("enabled"):
            out.append(str(p["name"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MoE models (delegates to pipeline/train.py)")
    parser.add_argument("--config", default="config/training.yaml", help="Path to training config")
    parser.add_argument("--tasks", help="Comma-separated list of tasks to train")
    parser.add_argument("--all", action="store_true", help="Train all MoE-enabled tasks")
    parser.add_argument("--run-tag", default="train_moe_wrapper", help="Run tag appended to run_id")
    parser.add_argument("--decision-cutoff-hours", type=float, default=None, help="Decision cutoff hours (horizon)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.all:
        tasks = _moe_tasks(cfg)
    elif args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        raise SystemExit("Must specify --tasks or --all")

    if not tasks:
        raise SystemExit("No tasks selected.")

    logger.info("Delegating MoE training to pipeline/train.py for tasks: %s", tasks)

    overrides: dict = {
        "problems": tasks,
        "run_tag": str(args.run_tag),
        "model_architecture": "moe",
    }
    if args.decision_cutoff_hours is not None:
        overrides["decision_cutoff_hours"] = float(args.decision_cutoff_hours)

    train_main(config_path=args.config, **overrides)


if __name__ == "__main__":
    main()
