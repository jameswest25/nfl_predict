from __future__ import annotations

"""Game-level feature builder.

Aggregates play-level rows to game-level per (season, week, game_id).
Writes to data/processed/game_by_week/season=YYYY/week=W/part.parquet
"""

from pathlib import Path
from datetime import date
import polars as pl

PLAY_DIR   = Path("data/processed/play_by_week")
GAME_DIR   = Path("data/processed/game_by_week")

def build_game_level(*, start_date: date, end_date: date) -> None:
    GAME_DIR.mkdir(parents=True, exist_ok=True)

    scan = pl.scan_parquet(
        str(PLAY_DIR / "season=*/week=*/part.parquet"),
        glob=True,
        hive_partitioning=True,
        missing_columns="insert",
        extra_columns="ignore",
    )
    scan = scan.filter(
        (pl.col("game_date").cast(pl.Utf8) >= start_date.isoformat()) &
        (pl.col("game_date").cast(pl.Utf8) <= end_date.isoformat())
    )

    df = (
        scan.group_by(["season","week","game_id","posteam","defteam"], maintain_order=True)
            .agg([
                pl.count().alias("n_plays"),
                pl.col("yards_gained").sum().alias("game_yards"),
                pl.col("touchdown").fill_null(0).sum().alias("game_tds"),
            ])
            .collect(streaming=True)
    )
    if df.is_empty():
        return
    for (s, w), sub in df.group_by(["season","week"], maintain_order=True):
        out_dir = GAME_DIR / f"season={int(s)}" / f"week={int(w)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "part.parquet").unlink(missing_ok=True)
        sub.write_parquet(out_dir / "part.parquet", compression="zstd")
