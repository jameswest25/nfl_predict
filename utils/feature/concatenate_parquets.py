#!/usr/bin/env python3
import sys, logging
from pathlib import Path
from datetime import datetime
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("merge_parquets")

BASE = Path("/Users/jameswest/Desktop/mlb_predict_refactor/data/processed/statcast")
NEW_BASE = BASE / "new_parquets"
OLD_BASE = BASE / "old_parquets"
OUT_BASES = {
    "pa_by_day": BASE / "pa_by_day",
    "pitch_by_day": BASE / "pitch_by_day",
    "game_by_day": BASE / "game_by_day",
}
JOIN_KEYS = {
    "pitch_by_day": ["game_pk", "at_bat_number", "pitch_number", "batter"],
    "pa_by_day":    ["game_pk", "at_bat_number", "batter"],
    "game_by_day":  ["game_pk", "batter"],
}
CUTOFF = datetime(2024, 3, 31)

def parse_date_folder(p: Path):
    # "date=YYYY-MM-DD"
    if not p.is_dir(): return None
    if not p.name.startswith("date="): return None
    try:
        return datetime.strptime(p.name.split("=", 1)[1], "%Y-%m-%d")
    except Exception:
        return None

def list_new_dates(data_type: str, side: str):
    root = NEW_BASE / data_type / f"side={side}"
    if not root.exists(): return []
    out = []
    for d in root.iterdir():
        dt = parse_date_folder(d)
        if dt and dt >= CUTOFF:
            out.append((d, dt))
    return sorted(out, key=lambda x: x[1])

def get_schema_map(schema: pl.Schema):
    # return {name: dtype}
    return {name: dtype for name, dtype in zip(schema.names(), schema.dtypes())}

def ensure_unique_right(lf: pl.LazyFrame, keys: list[str]) -> pl.LazyFrame:
    # Deduplicate the right side on join keys if needed, keeping the last occurrence.
    # (Prevents accidental 1:N joins from multiplying rows.)
    # We check n_unique(keys) vs height quickly:
    est = lf.select(pl.len().alias("n"), *[pl.col(k) for k in keys]).head(0)  # noop collect schema path
    # Polars can't n_unique across multiple columns lazily without collect;
    # do a cheap trick: group and count, then filter dup groups.
    dup_groups = (
        lf.group_by(keys).agg(pl.len().alias("__cnt__"))
          .filter(pl.col("__cnt__") > 1)
          .select(pl.len().alias("__n_dup_groups__"))
          .collect(streaming=True)
    )
    if dup_groups["__n_dup_groups__"][0] == 0:
        return lf
    log.warning("Right side has duplicate join keys; deduplicating with group_by(...).last()")
    # Use last() to keep the newest columns; adjust if you prefer first()
    return lf.group_by(keys).agg(pl.all().last())

def merge_one(data_type: str, side: str, date_folder: Path):
    date_str = date_folder.name  # "date=YYYY-MM-DD"
    new_file = date_folder / "part.parquet"
    old_file = OLD_BASE / data_type / f"side={side}" / date_str / "part.parquet"
    out_dir  = OUT_BASES[data_type] / f"side={side}" / date_str
    out_file = out_dir / "part.parquet"

    if not new_file.exists():
        log.warning(f"Missing new file: {new_file}")
        return False
    if not old_file.exists():
        log.warning(f"Missing old file: {old_file}")
        return False

    keys = JOIN_KEYS[data_type]

    # Collect schemas to compare/cast keys
    old_schema = pl.scan_parquet(str(old_file)).collect_schema()
    new_schema = pl.scan_parquet(str(new_file)).collect_schema()
    old_types = get_schema_map(old_schema)
    new_types = get_schema_map(new_schema)

    # Verify all keys exist
    for k in keys:
        if k not in old_types:
            raise RuntimeError(f"Join key '{k}' not in OLD schema for {old_file}")
        if k not in new_types:
            raise RuntimeError(f"Join key '{k}' not in NEW schema for {new_file}")

    # Right (new) should provide only columns that don't already exist in old
    keep_new_cols = [c for c in new_schema.names() if c not in old_schema.names() and c not in keys]
    if not keep_new_cols:
        log.info(f"No new columns to add for {data_type}/{side}/{date_str} — skipping")
        # Still copy old -> destination if destination missing?
        out_dir.mkdir(parents=True, exist_ok=True)
        if not out_file.exists():
            pl.read_parquet(str(old_file)).write_parquet(str(out_file), compression="zstd")
        return True

    # Build lazy frames
    old_lf = pl.scan_parquet(str(old_file))

    # Align key dtypes on the NEW side to OLD's dtypes to prevent join failures or upcasting
    cast_exprs = []
    for k in keys:
        if new_types[k] != old_types[k]:
            cast_exprs.append(pl.col(k).cast(old_types[k]))
    new_lf = pl.scan_parquet(str(new_file)).select(keys + keep_new_cols)
    if cast_exprs:
        new_lf = new_lf.with_columns(cast_exprs)

    # Ensure right side uniqueness on keys to keep the join 1:1
    new_lf = ensure_unique_right(new_lf, keys)

    # Perform left join: preserve every row from OLD; add NEW columns when present
    merged_lf = old_lf.join(new_lf, on=keys, how="left")

    # Materialize and write atomically
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp.part.parquet"
    df = merged_lf.collect(streaming=True)  # streaming may fall back internally; harmless to keep True
    # Optional: sanity check row count equals old
    old_rows = pl.scan_parquet(str(old_file)).select(pl.len()).collect().item()
    if df.height != old_rows:
        log.warning(f"Row count changed after join for {out_file}: old={old_rows}, new={df.height}")

    df.write_parquet(str(tmp), compression="zstd")
    if out_file.exists():
        out_file.unlink()
    tmp.rename(out_file)
    log.info(f"Wrote {out_file} with {df.height} rows and +{len(keep_new_cols)} new columns")
    return True

def main():
    total_ok = total_err = 0
    for data_type in ["pa_by_day", "pitch_by_day", "game_by_day"]:
        for side in ["batter", "pitcher"]:
            dates = list_new_dates(data_type, side)
            log.info(f"{data_type}/{side}: {len(dates)} dates to process")
            for date_folder, _dt in dates:
                try:
                    ok = merge_one(data_type, side, date_folder)
                    total_ok += int(ok)
                    total_err += int(not ok)
                except Exception as e:
                    log.exception(f"Failed {data_type}/{side}/{date_folder.name}: {e}")
                    total_err += 1
    log.info(f"Done. Success: {total_ok}, Errors: {total_err}")
    sys.exit(0 if total_err == 0 else 1)

if __name__ == "__main__":
    main()
