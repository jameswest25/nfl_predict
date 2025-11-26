#!/usr/bin/env python3
"""
Compare two baseline JSON summaries and highlight differences.

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/backtest/compare_baselines.py \
        --current output/metrics/baselines/20251201_backtest.json \
        --previous output/metrics/baselines/20251115_backtest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff two baseline metric files.")
    parser.add_argument("--current", type=Path, required=True, help="Current baseline JSON.")
    parser.add_argument("--previous", type=Path, required=True, help="Previous baseline JSON.")
    parser.add_argument("--output", type=Path, help="Optional JSON file to write diff summary.")
    return parser.parse_args()


def load_baseline(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Baseline format must be list; got {type(data)} in {path}")
    mapping = {}
    for row in data:
        key = f"{row.get('problem')}::{row.get('prediction_column')}"
        mapping[key] = row
    return mapping


def diff_metrics(curr: Dict[str, Any], prev: Dict[str, Any]) -> Dict[str, float | None]:
    metrics = ["auc", "brier", "log_loss"]
    diff = {}
    for metric in metrics:
        curr_val = curr.get(metric)
        prev_val = prev.get(metric)
        if curr_val is None or prev_val is None:
            diff[f"delta_{metric}"] = None
        else:
            diff[f"delta_{metric}"] = float(curr_val) - float(prev_val)
    return diff


def main() -> None:
    args = parse_args()
    current = load_baseline(args.current)
    previous = load_baseline(args.previous)

    summaries: List[Dict[str, Any]] = []
    for key, curr_row in current.items():
        prev_row = previous.get(key)
        if not prev_row:
            summaries.append(
                {"key": key, "status": "new", **{k: curr_row.get(k) for k in ("auc", "brier", "log_loss")}}
            )
            continue
        diff = diff_metrics(curr_row, prev_row)
        entry = {
            "key": key,
            "problem": curr_row.get("problem"),
            "prediction_column": curr_row.get("prediction_column"),
            **diff,
        }
        summaries.append(entry)

    for key in set(previous) - set(current):
        summaries.append({"key": key, "status": "missing_in_current"})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, indent=2))

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

