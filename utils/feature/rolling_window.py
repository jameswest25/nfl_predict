"""NFL Rolling Window Features - Direct port from MLB rolling_window.py

Uses the exact same vectorized Polars pattern as the working MLB implementation.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache/feature/daily_totals")
NFL_SEASON_START_MMDD = (9, 1)


class RollingWindow:
    """Rolling window computation for NFL player statistics."""
    
    def __init__(self, level: str = "game", *, season_start_mmdd: tuple[int, int] = NFL_SEASON_START_MMDD):
        if level not in {"game", "drive"}:
            raise ValueError(f"level must be 'game' or 'drive'")
        
        self.level = level
        self.season_start_mmdd = season_start_mmdd
        self.cache_root = CACHE_ROOT / f"level={level}"
        
        # Empty frame for when no cache exists
        self.EMPTY_TOTALS_LAZY = pl.DataFrame({
            "ctx": [], "player_id": [], "opponent": [], "date": [], "denom": [],
        }).lazy()
    
    def _season_anchor(self, d: date) -> date:
        m0, d0 = self.season_start_mmdd
        year = d.year if d.month >= m0 else d.year - 1
        return date(year, m0, d0)
    
    def _earliest_cache_date(self) -> date | None:
        if not self.cache_root.exists():
            return None
        dates = []
        for p in self.cache_root.glob("date=*"):
            try:
                d = date.fromisoformat(p.name.split("=", 1)[1])
                dates.append(d)
            except Exception:
                continue
        return min(dates) if dates else None
    
    def _scan_totals(
        self,
        lo: date,
        hi: date,
        player_ids: list[str] | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Load cache with same pattern as MLB version."""
        
        def _add_requested_columns_to_empty(lf: pl.LazyFrame) -> pl.LazyFrame:
            if not columns:
                return lf
            df = lf.collect()
            existing = set(df.columns)
            missing = [c for c in columns if c not in existing]
            if missing:
                df = df.with_columns([pl.lit(0).cast(pl.Float64).alias(c) for c in missing])
            return df.lazy()
        
        if hi < lo:
            return _add_requested_columns_to_empty(self.EMPTY_TOTALS_LAZY)
        
        dates = [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]
        paths = [
            self.cache_root / f"date={d.isoformat()}/part.parquet"
            for d in dates
            if (self.cache_root / f"date={d.isoformat()}/part.parquet").exists()
        ]
        
        if not paths:
            return _add_requested_columns_to_empty(self.EMPTY_TOTALS_LAZY)
        
        # Load and concatenate
        frames = []
        for p in paths:
            df_day = pl.read_parquet(p)
            if "team" not in df_day.columns:
                df_day = df_day.with_columns(pl.lit("").cast(pl.Utf8).alias("team"))
            if "opponent" not in df_day.columns:
                df_day = df_day.with_columns(pl.lit("").cast(pl.Utf8).alias("opponent"))
            df_day = df_day.filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
            if player_ids is not None:
                df_day = df_day.filter(pl.col("player_id").is_in(player_ids))
            if not df_day.is_empty():
                frames.append(df_day)
        
        if not frames:
            return _add_requested_columns_to_empty(self.EMPTY_TOTALS_LAZY)
        
        df_combined = pl.concat(frames, how="diagonal_relaxed")
        
        # Add missing columns
        if columns:
            existing = set(df_combined.columns)
            missing = [c for c in columns if c not in existing]
            if missing:
                # Special handling for season: if missing in cache, fallback to date-derived year
                if "season" in missing:
                    # Fill season with date year. Note: this is a fallback for old cache.
                    df_combined = df_combined.with_columns(
                        pl.col("date").dt.year().cast(pl.Int32).alias("season")
                    )
                    missing = [c for c in missing if c != "season"]
                
                if missing:
                    df_combined = df_combined.with_columns([pl.lit(0).cast(pl.Float64).alias(c) for c in missing])
            
            # Fill nulls
            fill_exprs = []
            for c in columns:
                if c not in ["date", "ctx", "player_id", "opponent", "season"]:
                    fill_exprs.append(pl.col(c).fill_null(0))
            if fill_exprs:
                df_combined = df_combined.with_columns(fill_exprs)
        
        lf = df_combined.lazy()
        
        # Filter null dates before timestamp calculation
        lf = lf.filter(pl.col("date").is_not_null())
        
        # Add timestamp (critical - same as MLB)
        lf = (
            lf.with_columns(
                ((pl.col("date") + pl.duration(days=1)) - pl.duration(milliseconds=1))
                .cast(pl.Datetime("ms"))
                .alias("__ts")
            )
            .sort(["player_id", "__ts"])
            .with_columns(pl.col("__ts").set_sorted())
        )
        
        return lf
    
    def compute(
        self,
        ds: pl.LazyFrame | pl.DataFrame,
        *,
        keys: Iterable[str],
        windows: Iterable[str],
        contexts: Iterable[str],
        date_col: str = "game_date",
    ) -> pl.LazyFrame:
        """Compute rolling features using MLB pattern."""
        
        if isinstance(ds, pl.DataFrame):
            ds = ds.lazy()
        
        keys = list(keys)
        windows = list(windows)
        contexts = list(contexts)
        
        schema = ds.collect_schema()
        schema_names = set(schema.names())

        missing_stats = [k for k in keys if k not in schema_names]
        if missing_stats:
            raise ValueError(
                f"RollingWindow.compute received missing stats in dataset: {missing_stats}. "
                "Ensure stats requested for rolling features exist in the base frame."
            )
        
        # Cast player_id
        ds = ds.with_columns(pl.col("player_id").cast(pl.Utf8))
        
        # Filter null dates in INPUT (critical!)
        ds = ds.filter(pl.col(date_col).is_not_null())
        
        # Normalize date to datetime
        if schema[date_col] == pl.Date:
            ds = ds.with_columns(pl.col(date_col).cast(pl.Datetime("ms")))
        
        # Add timestamp - CRITICAL
        ds = (
            ds.with_columns(
                ((pl.col(date_col) + pl.duration(days=1)) - pl.duration(milliseconds=1)).alias("__ts")
            )
            .sort(["player_id", "__ts"])
            .with_columns(pl.col("__ts").set_sorted())
        )
        
        # Get date range of INPUT data
        min_max = ds.select(
            pl.col(date_col).dt.date().min().alias("lo"),
            pl.col(date_col).dt.date().max().alias("hi"),
        ).collect()
        batch_min_date, batch_max_date = min_max.row(0)

        rolling_feature_names: list[str] = []
        for ctx in contexts:
            suffix = "" if ctx == "vs_any" else f"_{ctx}"
            for w in windows:
                for k in keys:
                    rolling_feature_names.append(f"{w}g_{k}_per_{self.level}{suffix}")

        if batch_min_date is None or batch_max_date is None:
            logger.warning("RollingWindow.compute received empty dataset; returning null rolling features.")
            result = ds
            for feat in rolling_feature_names:
                result = result.with_columns(pl.lit(None).cast(pl.Float32).alias(feat))
            return result.drop("__ts")
        
        # Determine lookback for CACHE scan
        # Load cache INCLUDING the batch dates - cumsum logic prevents leakage
        # Each row's rolling features = cumsum(up to row) - row_value = history BEFORE row
        if "lifetime" in windows:
            earliest = self._earliest_cache_date()
            lo_hist = earliest if earliest else batch_min_date - timedelta(days=365 * 10)
        elif "season" in windows:
            season_start = self._season_anchor(batch_min_date)
            lo_hist = season_start if season_start < batch_min_date else batch_min_date - timedelta(days=365)
        else:
            lo_hist = batch_min_date - timedelta(days=90)
        
        # Include the ENTIRE batch in cache so later games can reference earlier games
        # The cumsum - current_value logic ensures each game only sees PRIOR data
        hi_hist = batch_max_date
        
        logger.info(f"📅 Cache range: {lo_hist} → {hi_hist} (batch: {batch_min_date} to {batch_max_date})")
        
        # Get player list
        player_ids = ds.select(pl.col("player_id").unique()).collect().to_series().to_list()
        
        # Build column list
        scan_cols = ["ctx", "denom", "player_id", "team", "date", "opponent", "season"]
        scan_cols += [f"{k}_num" for k in keys]
        
        # Load cache
        totals_hist = self._scan_totals(lo_hist, hi_hist, player_ids, scan_cols)
        
        # DEBUG: Check what scan loaded
        scan_row_count = totals_hist.select(pl.len()).collect().item()
        logger.info(f"🔍 Scanned cache: {scan_row_count} rows from {lo_hist} to {hi_hist}")
        logger.info(f"🔍 About to group and aggregate cache...")
        
        # Group and aggregate (EXACT MLB pattern)
        # CRITICAL: Filter null dates BEFORE groupby (groupby includes nulls as a group!)
        base_cols = ["ctx", "player_id", "team", "opponent", "date", "season"]
        numeric_cols = [
            c for c in totals_hist.columns
            if c not in base_cols and c != "__ts"
        ]
        agg_exprs = [pl.col(col).sum().alias(col) for col in numeric_cols]
        totals_hist_df = (
            totals_hist
            .select(pl.all().exclude("__ts"))  # Drop __ts from scan first
            .filter(pl.col("date").is_not_null())  # Filter BEFORE groupby
            .group_by(base_cols, maintain_order=True)
            .agg(agg_exprs)
            .collect()
        )
        
        logger.info(f"✅  Grouped: {len(totals_hist_df)} rows")
        
        # If no historical data, return input unchanged with null rolling features
        if len(totals_hist_df) == 0:
            logger.warning("⚠️  No historical data for rolling windows - features will be null")
            result = ds
            for feat in rolling_feature_names:
                result = result.with_columns(pl.lit(None).cast(pl.Float32).alias(feat))
            
            return result.drop("__ts")
        
        # Filter nulls one more time on the EAGER df (groupby might have created them somehow?)
        totals_hist_df = totals_hist_df.filter(pl.col("date").is_not_null())
        
        # DEBUG
        null_dates = totals_hist_df["date"].is_null().sum()
        if null_dates > 0:
            logger.warning(f"⚠️  Still {null_dates} null dates in totals_hist_df after filtering!")
            # Drop them
            totals_hist_df = totals_hist_df.filter(pl.col("date").is_not_null())
        
        # Step 1: opponent first
        totals_hist_df = totals_hist_df.with_columns(
            [
                pl.col("opponent").fill_null("").cast(pl.Utf8),
                pl.col("team").fill_null("").cast(pl.Utf8),
            ]
        )
        
        # Step 2: timestamp (THIS is where it fails)
        totals_hist_df = totals_hist_df.with_columns(
            ((pl.col("date") + pl.duration(days=1)) - pl.duration(milliseconds=1))
            .cast(pl.Datetime("ms"))
            .alias("__ts")
        )
        
        totals_hist = totals_hist_df.lazy()
        
        # Double check: filter out any null dates that might have snuck through
        unified_df = (
            totals_hist
            .filter(pl.col("date").is_not_null())
            .sort(["player_id", "__ts"])
            .with_columns(
                pl.col("__ts").set_sorted(),
                pl.col("opponent").fill_null("").cast(pl.Utf8),
                pl.col("team").fill_null("").cast(pl.Utf8),
                pl.col("season").cast(pl.Int32).alias("season_id"),
            )
            .collect()
        )
        
        feature_names_by_ctx: dict[str, list[str]] = {ctx: [] for ctx in contexts}
        for ctx in contexts:
            suffix = "" if ctx == "vs_any" else f"_{ctx}"
            for w in windows:
                for k in keys:
                    feature_name = f"{w}g_{k}_per_{self.level}{suffix}"
                    feature_names_by_ctx[ctx].append(feature_name)
        
        context_frames: dict[str, pl.LazyFrame | None] = {}
        int_windows = [w for w in windows if isinstance(w, int)]
        use_season = "season" in windows
        use_lifetime = "lifetime" in windows
        
        for ctx in contexts:
            if ctx == "vs_any":
                join_cols = ["player_id"]
            elif ctx == "vs_team":
                join_cols = ["player_id", "opponent"]
            elif ctx == "with_team":
                join_cols = ["player_id", "team"]
            else:
                join_cols = ["player_id"]
            ctx_df = unified_df.filter(pl.col("ctx") == ctx)
            if ctx_df.is_empty():
                context_frames[ctx] = None
                continue
            
            feature_frames: list[pl.DataFrame] = []
            partitions = ctx_df.partition_by(join_cols, maintain_order=True)
            
            for group_df in partitions:
                group_sorted = group_df.sort("__ts")
                denom = group_sorted["denom"].to_numpy().astype(np.float64, copy=False)
                n = len(denom)
                if n == 0:
                    continue
                
                den_cum = denom.cumsum()
                prev_den = np.concatenate(([0.0], den_cum[:-1]))
                
                season_ids = group_sorted["season_id"].to_numpy().astype(np.int32, copy=False)
                if use_season:
                    den_prev_season = np.zeros(n, dtype=np.float64)
                    for season in np.unique(season_ids):
                        idx = np.where(season_ids == season)[0]
                        if idx.size == 0:
                            continue
                        cum = denom[idx].cumsum()
                        den_prev_season[idx] = np.concatenate(([0.0], cum[:-1]))
                else:
                    den_prev_season = None
                
                den_window_cache: dict[int, np.ndarray] = {}
                for w in int_windows:
                    if w <= n:
                        shifted = np.concatenate((np.zeros(w, dtype=np.float64), prev_den[:-w]))
                    else:
                        shifted = np.zeros(n, dtype=np.float64)
                    den_window_cache[w] = prev_den - shifted
                
                feature_arrays: dict[str, np.ndarray] = {}
                
                for k in keys:
                    num = group_sorted[f"{k}_num"].to_numpy().astype(np.float64, copy=False)
                    num_cum = num.cumsum()
                    prev_num = np.concatenate(([0.0], num_cum[:-1]))
                    
                    if use_season:
                        num_prev_season = np.zeros(n, dtype=np.float64)
                        for season in np.unique(season_ids):
                            idx = np.where(season_ids == season)[0]
                            if idx.size == 0:
                                continue
                            cum = num[idx].cumsum()
                            num_prev_season[idx] = np.concatenate(([0.0], cum[:-1]))
                    else:
                        num_prev_season = None
                    
                    for w in windows:
                        if w == "lifetime":
                            denom_use = prev_den
                            numer_use = prev_num
                        elif w == "season":
                            if den_prev_season is None or num_prev_season is None:
                                continue
                            denom_use = den_prev_season
                            numer_use = num_prev_season
                        else:
                            w_int = int(w)
                            denom_use = den_window_cache.get(w_int)
                            if denom_use is None:
                                denom_use = np.zeros(n, dtype=np.float64)
                            if w_int <= n:
                                shifted_num = np.concatenate((np.zeros(w_int, dtype=np.float64), prev_num[:-w_int]))
                            else:
                                shifted_num = np.zeros(n, dtype=np.float64)
                            numer_use = prev_num - shifted_num
                        
                        values = np.full(n, np.nan, dtype=np.float32)
                        if denom_use is not None:
                            mask = denom_use > 0
                            if mask.any():
                                values[mask] = (numer_use[mask] / denom_use[mask]).astype(np.float32)
                        
                        feat_name = f"{w}g_{k}_per_{self.level}" + ("" if ctx == "vs_any" else f"_{ctx}")
                        feature_arrays[feat_name] = values
                
                data: dict[str, pl.Series | np.ndarray] = {
                    "player_id": group_sorted["player_id"],
                    "__ts": group_sorted["__ts"],
                }
                if ctx == "vs_team":
                    data["opponent"] = group_sorted["opponent"]
                if ctx in ("with_team",):
                    data["team"] = group_sorted["team"]
                
                for feat_name in feature_names_by_ctx[ctx]:
                    values = feature_arrays.get(feat_name)
                    if values is None:
                        values = np.full(n, np.nan, dtype=np.float32)
                    data[feat_name] = values
                
                feature_df = pl.DataFrame(data)
                float_cols = [f for f in feature_names_by_ctx[ctx] if f in feature_df.columns]
                if float_cols:
                    feature_df = feature_df.with_columns([pl.col(f).cast(pl.Float32) for f in float_cols])
                    feature_df = feature_df.with_columns([pl.col(f).fill_nan(None) for f in float_cols])
                
                feature_frames.append(feature_df)
            
            if feature_frames:
                feat_ctx_df = pl.concat(feature_frames, how="diagonal_relaxed")
                context_frames[ctx] = feat_ctx_df.lazy()
            else:
                context_frames[ctx] = None
        
        feat_vs_any = context_frames.get("vs_any")
        feat_vs_team = context_frames.get("vs_team")
        feat_with_team = context_frames.get("with_team")
        
        # Join back (MLB pattern)
        out_base = ds.with_columns(
            pl.col("opponent").fill_null("").cast(pl.Utf8) if "opponent" in schema_names else pl.lit("").alias("opponent")
        )
        
        out_joined = out_base
        
        # Join vs_any
        if "vs_any" in contexts and feat_vs_any is not None:
            out_joined = out_joined.join_asof(
                feat_vs_any,
                left_on="__ts",
                right_on="__ts",
                by=["player_id"],
            )
        
        # Join vs_team
        if "vs_team" in contexts and feat_vs_team is not None:
            out_joined = (
                out_joined.join_asof(
                    feat_vs_team,
                    left_on="__ts",
                    right_on="__ts",
                    by=["player_id", "opponent"],
                )
                .select(pl.exclude(r"^.*_right$"))
            )

        # Join with_team
        if "with_team" in contexts and feat_with_team is not None:
            out_joined = (
                out_joined.join_asof(
                    feat_with_team,
                    left_on="__ts",
                    right_on="__ts",
                    by=["player_id", "team"],
                )
                .select(pl.exclude(r"^.*_right$"))
            )
        
        return out_joined.drop("__ts")


def add_rolling_features(df: pl.DataFrame, *, level: str = "game", stats: list[str], 
                         windows: list[str], contexts: list[str], 
                         date_col: str = "game_date", **kwargs) -> pl.DataFrame:
    """Add rolling features."""
    roller = RollingWindow(level=level)
    result = roller.compute(
        df.lazy(),
        keys=stats,
        windows=windows,
        contexts=contexts,
        date_col=date_col,
    ).collect()
    return result
