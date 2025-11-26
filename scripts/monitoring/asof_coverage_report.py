#!/usr/bin/env python3
"""
Generate snapshot coverage dashboards from asof_metadata.parquet.

Outputs:
  - Markdown summary at docs/monitoring/asof_coverage.md
  - Historical metrics appended to output/metrics/coverage/history.parquet
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from utils.general.paths import PROJ_ROOT, ASOF_METADATA_PATH

DOC_PATH = PROJ_ROOT / "docs" / "monitoring" / "asof_coverage.md"
HISTORY_PATH = PROJ_ROOT / "output" / "metrics" / "coverage" / "history.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize as-of metadata snapshot coverage.")
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=ASOF_METADATA_PATH,
        help="Path to asof_metadata.parquet (default: data/processed/asof_metadata.parquet).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DOC_PATH,
        help="Markdown file to write (default: docs/monitoring/asof_coverage.md).",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=HISTORY_PATH,
        help="Parquet file to append historical metrics.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Asof metadata not found at {path}")
    return pl.read_parquet(path)


def compute_coverage(df: pl.DataFrame) -> pl.DataFrame:
    columns = [
        "injury_snapshot_ts",
        "roster_snapshot_ts",
        "odds_snapshot_ts",
        "forecast_snapshot_ts",
    ]
    exprs = [
        pl.col(col).is_not_null().mean().alias(f"{col}_coverage")
        for col in columns
        if col in df.columns
    ]
    return df.select(exprs)


def compute_horizon_breakdown(df: pl.DataFrame) -> pl.DataFrame:
    if "decision_cutoff_hours" not in df.columns:
        return pl.DataFrame()
    columns = [
        "injury_snapshot_ts",
        "roster_snapshot_ts",
        "odds_snapshot_ts",
        "forecast_snapshot_ts",
    ]
    agg_exprs = [
        pl.col(col).is_not_null().mean().alias(f"{col}_coverage")
        for col in columns
        if col in df.columns
    ]
    agg_exprs.append(pl.count().alias("game_count"))
    return (
        df.group_by("decision_cutoff_hours")
        .agg(agg_exprs)
        .sort("decision_cutoff_hours")
    )


def append_history(history_path: Path, snapshot: pl.DataFrame) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot.with_columns(
        pl.lit(datetime.utcnow().isoformat()).alias("generated_at")
    )
    if history_path.exists():
        existing = pl.read_parquet(history_path)
        snapshot = pl.concat([existing, snapshot], how="vertical_relaxed")
    snapshot.write_parquet(history_path, compression="zstd")


def write_markdown(out_path: Path, coverage: pl.DataFrame, horizons: pl.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# As-of Snapshot Coverage",
        "",
        f"_Generated: {datetime.utcnow().isoformat()}_",
        "",
    ]

    if coverage.height:
        coverage_dict = coverage.to_dicts()[0]
        lines.append("## Overall Coverage")
        for key, value in coverage_dict.items():
            lines.append(f"- {key.replace('_coverage', '')}: {value * 100:.1f}%")
        lines.append("")
    else:
        lines.append("No coverage metrics could be computed (missing columns).")

    if horizons.height:
        lines.append("## Coverage by Decision Cutoff Hours")
        lines.append("")
        lines.append("| Horizon (h) | Injury % | Roster % | Odds % | Weather % | Games |")
        lines.append("|-------------|----------|----------|--------|-----------|-------|")
        for row in horizons.to_dicts():
            lines.append(
                "| {h:.1f} | {inj:.1f}% | {ros:.1f}% | {odds:.1f}% | {wx:.1f}% | {count} |".format(
                    h=row["decision_cutoff_hours"],
                    inj=row.get("injury_snapshot_ts_coverage", 0.0) * 100.0,
                    ros=row.get("roster_snapshot_ts_coverage", 0.0) * 100.0,
                    odds=row.get("odds_snapshot_ts_coverage", 0.0) * 100.0,
                    wx=row.get("forecast_snapshot_ts_coverage", 0.0) * 100.0,
                    count=row["game_count"],
                )
            )
        lines.append("")
    else:
        lines.append("No horizon breakdown available (missing decision_cutoff_hours).")

    out_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    df = load_metadata(args.metadata_path)
    coverage = compute_coverage(df)
    horizons = compute_horizon_breakdown(df)

    append_history(args.history, coverage.with_columns(pl.lit(datetime.utcnow().isoformat()).alias("generated_at")))
    write_markdown(args.out, coverage, horizons)
    print(f"Wrote report → {args.out}")
    print(f"Updated history → {args.history}")


if __name__ == "__main__":
    main()

