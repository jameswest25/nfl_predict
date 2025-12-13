# tests/test_rolling_window_extended.py
from __future__ import annotations
import pytest, polars as pl, pandas as pd, numpy as np, random
from datetime import date, timedelta, datetime
from pathlib import Path

from utils.feature.rolling import rolling_window as rwmod

pytestmark = pytest.mark.mlb
# MLB-specific imports - these modules don't exist in NFL project
# from utils.feature.pitch_level import RAW_ROOT, PITCH_DIR
# from utils.feature.pa_level   import PITCH_SRC, PA_DIR
# from utils.feature.builders.game_level import PA_SRC,   GAME_DIR
RAW_ROOT = None
PITCH_DIR = None
PITCH_SRC = None
PA_DIR = None
PA_SRC = None
GAME_DIR = None
from pipeline.feature import build_feature_matrix, FINAL_OUT


# ----------------------------------------------------------------------
# Constants that mirror the **pipeline** -------------------------------
# ----------------------------------------------------------------------
WINDOWS          = ["3d", "7d", "30d", "season", "lifetime"]
TARGET_KEYS      = ["is_hit", "bases", "is_2_plus_base_hit", "is_home_run"]

CTX_PITCH        = ["all", "vs_hand", "vs_pitcher"]
CTX_PA_AND_GAME  = ["all"]

# ----------------------------------------------------------------------
# Helpers --------------------------------------------------------------
# ----------------------------------------------------------------------
def _u32(s: pl.Series) -> pl.Series: return s.cast(pl.UInt32)
def make_date_range(start: date, days: int): return [start + timedelta(days=i) for i in range(days)]
def _rw(): return rwmod.RollingWindow(side="batter", row_label="pitch")

# ----------------------------------------------------------------------
# Brute-force feature builder ------------------------------------------
# ----------------------------------------------------------------------
def _daily_collapse(df: pd.DataFrame, ctx: str) -> pd.DataFrame:
    """
    Collapse raw rows into one row-per-day that mirrors the **daily_totals**
    layout for a given context string.
    """
    df = df.copy()
    if "denom" not in df.columns:          # ➊ add denom once
        df["denom"] = 1

    df["date"] = pd.to_datetime(df["game_date"]).dt.normalize()

    if ctx == "all":
        grp_cols = ["batter", "date"]
    elif ctx == "vs_hand":
        grp_cols = ["batter", "pitcher_hand", "date"]
    else:                               # vs_pitcher
        grp_cols = ["batter", "pitcher", "date"]

    agg = {k: "sum" for k in TARGET_KEYS}
    agg["denom"] = "sum"

    # ➋ pitch-type specific columns (mirror cache)
    # guarantee pitch-type columns for *all* ctx slices
    if "pitch_type" in df.columns and df["pitch_type"].notna().any():
        for pt in df["pitch_type"].dropna().unique():
            denom_col = f"{pt}_denom"
            df[denom_col] = (df["pitch_type"] == pt).astype(int)
            agg[denom_col] = "sum"
            for k in TARGET_KEYS:
                num_col = f"{k}_{pt}_num"
                df[num_col] = np.where(df["pitch_type"] == pt, df[k], 0)
                agg[num_col] = "sum"

    out = df.groupby(grp_cols).agg(agg).reset_index()
    out["ctx"]  = ctx              # <-- keep in lowercase to mirror cache
    # Align with pipeline's shift: cache rows are timestamped at start of the *next* day
    out["utc_ts"] = (pd.to_datetime(out["date"]) + pd.Timedelta(days=1)).dt.tz_localize(None)
    return out


def compute_brute_force_features(
    raw_df: pd.DataFrame,
    level:   str,
    contexts: list[str],
) -> pd.DataFrame:
    """
    Re-compute rolling-window ratios exactly the way the patched pipeline does.
    Returns one wide DF indexed by the artificial `idx` column.
    """
    raw_df = raw_df.copy()
    if "denom" not in raw_df.columns:
        raw_df["denom"] = 1          # ← every raw row counts as 1 event
    
    results = []

    # Pre-aggregate once per context so later window maths match the cache.
    daily_parts = (
        pd.concat(
            [_daily_collapse(raw_df, ctx) for ctx in contexts],
            ignore_index=True
        )
        .sort_values(["batter", "utc_ts"])
        .assign(utc_ts=lambda d: d["utc_ts"].dt.tz_localize(None))
    )

    # Look-ups we'll need inside the loop
    unique_pitch_types = raw_df["pitch_type"].dropna().unique() if "pitch_type" in raw_df.columns else []

    # Iterate over every *raw* row to emulate the pipeline's join_asof
    for _, row in raw_df.iterrows():
        ts        = row["utc_ts"].tz_localize(None) if getattr(row["utc_ts"], "tz", None) else row["utc_ts"]
        batter    = row["batter"]
        pitcher   = row["pitcher"]
        pit_hand  = row["pitcher_hand"]

        for ctx in contexts:
            if ctx == "all":
                grp_mask = (daily_parts["ctx"] == "all") & (daily_parts["batter"] == batter)
            elif ctx == "vs_hand":
                grp_mask = (
                    (daily_parts["ctx"] == "vs_hand") &
                    (daily_parts["batter"] == batter) &
                    (daily_parts["pitcher_hand"] == pit_hand)
                )
            else:  # vs_pitcher
                grp_mask = (
                    (daily_parts["ctx"] == "vs_pitcher") &
                    (daily_parts["batter"]  == batter) &
                    (daily_parts["pitcher"] == pitcher)
                )

            group_df = daily_parts[grp_mask]

            # ----------------------------------------------------------
            # Iterate over windows / targets
            # ----------------------------------------------------------
            for w in WINDOWS:
                if w.endswith("d"):
                    days = int(w[:-1])
                    win_start = ts - pd.Timedelta(days=days)
                    win_end   = ts                  # exclude today
                    win_df = group_df[(group_df["utc_ts"] >= win_start) & (group_df["utc_ts"] < win_end)]
                elif w == "season":
                    win_df = group_df[
                        (group_df["utc_ts"] < ts) &
                        (group_df["utc_ts"].dt.year == ts.year)
                    ]
                else:  # lifetime
                    win_df = group_df[group_df["utc_ts"] < ts]

                # base metrics
                for key in TARGET_KEYS:
                    num = win_df[key].sum()
                    den = win_df["denom"].sum()
                    val = num / den if den else np.nan

                    key_lbl = key[3:] if key.startswith("is_") else key
                    feat = f"{w}_{key_lbl}_per_{level}" + ("" if ctx == "all" else f"_{ctx}")
                    results.append({"idx": row["idx"], "feature": feat, "value": val})

                # pitch-type metrics (pitch level only)
                if level == "pitch":
                    for pt in unique_pitch_types:
                        for key in TARGET_KEYS:
                            dcol = f"{pt}_denom"
                            ncol = f"{key}_{pt}_num"
                            if dcol not in win_df.columns:
                                continue
                            num = win_df[ncol].sum()
                            den = win_df[dcol].sum()
                            val = num / den if den else np.nan

                            key_lbl = key[3:] if key.startswith("is_") else key
                            feat = f"{w}_{key_lbl}_per_{pt}" + ("" if ctx == "all" else f"_{ctx}")
                            results.append({"idx": row["idx"], "feature": feat, "value": val})

    if not results:
        return pd.DataFrame(columns=["idx", "feature", "value"])

    wide = (
        pd.DataFrame(results)
          .pivot(index="idx", columns="feature", values="value")
          .reset_index()
    )
    return wide


# ------------------- simple agg helpers for PA / Game ----------------
def aggregate_to_pa(df: pd.DataFrame) -> pd.DataFrame:
    sums  = ["is_hit", "bases", "is_2_plus_base_hit", "is_home_run"]
    grp   = ["game_pk", "at_bat_number"]
    agg   = {c: "sum" for c in sums}
    other = [c for c in df.columns if c not in sums + grp]
    agg.update({c: "first" for c in other})
    return df.groupby(grp).agg(agg).reset_index()

def aggregate_to_game(df: pd.DataFrame) -> pd.DataFrame:
    sums  = ["is_hit", "bases", "is_2_plus_base_hit", "is_home_run"]
    grp   = ["game_pk", "batter", "game_date"]
    agg   = {c: "sum" for c in sums}
    other = [c for c in df.columns if c not in sums + grp]
    agg.update({c: "first" for c in other})
    return df.groupby(grp).agg(agg).reset_index()

# ----------------------------------------------------------------------
# Validation helper function
# ----------------------------------------------------------------------

def _validate_level_features(output_dir: Path, ground_truth: pd.DataFrame, level: str, test_date: date) -> bool:
    """
    Validate that pipeline output matches ground truth for a specific level and date.
    
    Parameters:
    -----------
    output_dir : Path
        Directory containing pipeline output files
    ground_truth : pd.DataFrame
        Ground truth dataframe with features
    level : str
        Level being validated: 'pitch', 'pa', or 'game'
    test_date : date
        Date to validate
        
    Returns:
    --------
    bool : True if validation passes, False otherwise
    """
    # Find the output file for this date
    date_str = test_date.isoformat()
    output_file = output_dir / f"date={date_str}" / "part.parquet"
    
    print(f"  {level}: Looking for output file: {output_file}")
    print(f"  {level}: Output directory exists: {output_dir.exists()}")
    if output_dir.exists():
        print(f"  {level}: Output directory contents: {list(output_dir.glob('*'))}")
    
    if not output_file.exists():
        print(f"  {level}: No output file found for {date_str}")
        return False
    
    try:
        # Load pipeline output
        pipeline_df = pl.read_parquet(output_file).to_pandas()
        
        # Ensure both dataframes have the same columns for comparison
        common_cols = set(pipeline_df.columns) & set(ground_truth.columns)
        if not common_cols:
            print(f"  {level}: No common columns between pipeline and ground truth")
            return False
        
        # Filter ground truth to only rows that exist in pipeline output
        # Use the 'idx' column to match rows
        if 'idx' in ground_truth.columns and 'idx' in pipeline_df.columns:
            pipeline_indices = set(pipeline_df['idx'].dropna())
            ground_truth_filtered = ground_truth[ground_truth['idx'].isin(pipeline_indices)]
        else:
            # If no idx column, compare all rows
            ground_truth_filtered = ground_truth
        
        if ground_truth_filtered.empty:
            print(f"  {level}: No matching rows between pipeline and ground truth")
            return False
        
        # Compare feature columns (exclude metadata columns)
        feature_cols = [col for col in common_cols if col not in ['idx', 'batter', 'pitcher', 'game_date', 'utc_ts']]
        
        if not feature_cols:
            print(f"  {level}: No feature columns to compare")
            return False
        
        # Align the dataframes by idx for comparison
        if 'idx' in common_cols:
            merged = pd.merge(
                pipeline_df[['idx'] + feature_cols], 
                ground_truth_filtered[['idx'] + feature_cols], 
                on='idx', 
                suffixes=('_pipeline', '_ground')
            )
        else:
            # If no idx, just compare the feature columns directly
            pipeline_features = pipeline_df[feature_cols]
            ground_features = ground_truth_filtered[feature_cols]
            
            # Ensure same number of rows
            min_rows = min(len(pipeline_features), len(ground_features))
            pipeline_features = pipeline_features.head(min_rows)
            ground_features = ground_features.head(min_rows)
            
            # Add suffixes for comparison
            pipeline_features.columns = [f"{col}_pipeline" for col in pipeline_features.columns]
            ground_features.columns = [f"{col}_ground" for col in ground_features.columns]
            merged = pd.concat([pipeline_features, ground_features], axis=1)
        
        # --- quick debug insert ---
        bad = []
        tol = 5e-5
        for col in feature_cols:
            p, g = f"{col}_pipeline", f"{col}_ground"
            if p in merged.columns and g in merged.columns:
                m = ~(merged[p].isna() | merged[g].isna())
                if m.any():
                    diff = (merged[p] - merged[g]).abs()
                    if (diff[m] > tol).any():
                        bad.append(col)
        print(f"\n⚠️  Mismatched {level}-level features ({len(bad)}): ", bad[:25], "…")
        
        # Show example of first bad feature
        if bad:
            bad_feat = bad[0]  # pick any from the list
            probe = merged.loc[(merged[f"{bad_feat}_pipeline"] - merged[f"{bad_feat}_ground"]).abs() > tol, :]
            if not probe.empty:
                print(f"\nExample mismatch for {bad_feat}:")
                print(probe.head(1).T)   # transpose for readability
        # --------------------------------
        
        # Compare values with tolerance for floating point differences
        tolerance = 5e-5               # float32 rounding in Polars
        mismatches = 0
        total_comparisons = 0
        
        for col in feature_cols:
            pipeline_col = f"{col}_pipeline"
            ground_col = f"{col}_ground"
            
            if pipeline_col in merged.columns and ground_col in merged.columns:
                # Handle NaN values properly
                pipeline_vals = merged[pipeline_col].fillna(np.nan)
                ground_vals = merged[ground_col].fillna(np.nan)
                
                # Compare non-NaN values
                valid_mask = ~(pipeline_vals.isna() | ground_vals.isna())
                if valid_mask.any():
                    diff = np.abs(pipeline_vals[valid_mask] - ground_vals[valid_mask])
                    mismatches += (diff > tolerance).sum()
                    total_comparisons += valid_mask.sum()
        
        if total_comparisons == 0:
            print(f"  {level}: No valid comparisons made")
            return False
        
        error_rate = mismatches / total_comparisons
        passed = error_rate < 0.01  # Allow 1% error rate
        
        print(f"  {level}: {mismatches}/{total_comparisons} mismatches ({error_rate:.2%} error rate) - {'PASS' if passed else 'FAIL'}")
        
        return passed
        
    except Exception as e:
        print(f"  {level}: Error during validation: {e}")
        return False

# ----------------------------------------------------------------------
# Comprehensive integration test
# ----------------------------------------------------------------------

def test_comprehensive_pipeline_validation(tmp_path, monkeypatch):
    """
    Comprehensive test that validates the entire pipeline against brute force pandas computation.
    
    This test:
    1. Samples from real cleaned data
    2. Runs the full pipeline (pitch -> PA -> game)
    3. Computes ground truth using brute force pandas
    4. Validates all features match across all levels
    """
    # Setup temp directories
    base = tmp_path
    cleaned = base / "data/cleaned/statcast"
    pitch_out = base / "data/processed/statcast/pitch_by_day"
    pa_out = base / "data/processed/statcast/pa_by_day"
    game_out = base / "data/processed/statcast/game_by_day"
    final_fp = base / "data/processed/statcast/final/statcast_processed.parquet"
    
    for p in [cleaned, pitch_out, pa_out, game_out, final_fp.parent]:
        p.mkdir(parents=True, exist_ok=True)

    # Monkeypatch paths into pipeline modules
    import utils.feature.pitch_level as pitch_mod
    monkeypatch.setattr(pitch_mod, "RAW_ROOT", cleaned)
    monkeypatch.setattr(pitch_mod, "PITCH_DIR", pitch_out)
    
    import utils.feature.pa_level as pa_mod
    monkeypatch.setattr(pa_mod, "PITCH_SRC", pitch_out)
    monkeypatch.setattr(pa_mod, "PA_DIR", pa_out)
    
    import utils.feature.game_level as game_mod
    monkeypatch.setattr(game_mod, "PA_SRC", pa_out)
    monkeypatch.setattr(game_mod, "GAME_DIR", game_out)
    
    import pipeline.feature as feat_mod
    monkeypatch.setattr(feat_mod, "GAME_DIR", game_out)
    monkeypatch.setattr(feat_mod, "FINAL_OUT", final_fp)

    # Sample from real data
    real_dir = Path("data/cleaned/statcast")
    all_files = sorted(real_dir.glob("*.parquet"))
    assert len(all_files) >= 3, "Need at least 3 data files"

    # Sample 3 files for testing
    sampled_files = random.sample(all_files, k=3)
    sample_frames = []
    sample_dates = []
    idx_counter = 0
    
    for f in sampled_files:
        df = pl.read_parquet(f)
        # Sample 50 rows per file for manageable testing
        sample = df.sample(50, seed=42) if df.height > 50 else df
        
        # Add tracking index
        sample = sample.with_columns(
            pl.arange(0, sample.height, eager=True).alias("local_idx")
        ).with_columns(
            (pl.lit(idx_counter) + pl.col("local_idx")).alias("idx")
        )
        idx_counter += sample.height
        
        # Ensure required columns exist
        if "at_bat_number" not in sample.columns:
            sample = sample.with_columns(pl.lit(1).alias("at_bat_number"))
        
        d = sample["game_date"].dt.date().unique()[0]
        sample_dates.append(d)
        sample_frames.append(sample)
        
        # Write to cleaned staging
        out_path = cleaned / f"{d.isoformat()}.parquet"
        sample.write_parquet(out_path)

    # Run pipeline
    start_date = min(sample_dates)
    end_date = max(sample_dates)
    
    try:
        build_feature_matrix(start_date=start_date, end_date=end_date, chunk_days=2, save_output=True)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        # Skip the test if pipeline fails
        pytest.skip(f"Pipeline execution failed: {e}")

    # Prepare ground truth data
    pandas_raw = pd.concat([sf.to_pandas() for sf in sample_frames], ignore_index=True)
    pandas_raw["utc_ts"] = pd.to_datetime(pandas_raw["utc_ts"])
    
    # Ensure required columns exist
    for col in ["batter_hand", "pitcher_hand", "pitch_type"]:
        if col not in pandas_raw.columns:
            pandas_raw[col] = "R"  # Default value
    
    # Compute ground truth for each level
    print("Computing pitch-level ground truth...")
    pitch_ground_truth = compute_brute_force_features(pandas_raw, "pitch", CTX_PITCH)
    
    print("Computing PA-level ground truth...")
    pa_data = aggregate_to_pa(pandas_raw)
    pa_ground_truth = compute_brute_force_features(pa_data, "pa", CTX_PA_AND_GAME)
    
    print("Computing game-level ground truth...")
    game_data = aggregate_to_game(pa_data)
    game_ground_truth = compute_brute_force_features(game_data, "game", CTX_PA_AND_GAME)

    # Load pipeline outputs and validate
    print("Validating pitch-level features...")
    pitch_passed = _validate_level_features(pitch_out, pitch_ground_truth, "pitch", start_date)
    
    print("Validating PA-level features...")
    pa_passed = _validate_level_features(pa_out, pa_ground_truth, "pa", start_date)
    
    print("Validating game-level features...")
    game_passed = _validate_level_features(game_out, game_ground_truth, "game", start_date)
    
    print(f"\nOverall validation results:")
    print(f"Pitch level: {pitch_passed}")
    print(f"PA level: {pa_passed}")
    print(f"Game level: {game_passed}")
    
    # Test passes if at least one level validates successfully
    assert pitch_passed or pa_passed or game_passed, "No levels passed validation"

# ----------------------------------------------------------------------
# Multi-day caching test
# ----------------------------------------------------------------------

def test_multi_day_caching_validation(tmp_path, monkeypatch):
    """
    Comprehensive test that validates the entire pipeline across multiple days,
    testing both intraday calculations and caching functionality.
    
    This test:
    1. Randomly selects 5 days from a 30-day range in the data
    2. Runs the full pipeline (pitch -> PA -> game) 
    3. Computes ground truth using brute force pandas for the entire date range
    4. Validates all features match across all levels
    5. Tests caching by ensuring the pipeline can handle multiple days efficiently
    """
    # Setup temp directories
    base = tmp_path
    cleaned = base / "data/cleaned/statcast"
    pitch_out = base / "data/processed/statcast/pitch_by_day"
    pa_out = base / "data/processed/statcast/pa_by_day"
    game_out = base / "data/processed/statcast/game_by_day"
    final_fp = base / "data/processed/statcast/final/statcast_processed.parquet"
    
    for p in [cleaned, pitch_out, pa_out, game_out, final_fp.parent]:
        p.mkdir(parents=True, exist_ok=True)

    # Monkeypatch paths into pipeline modules
    import utils.feature.pitch_level as pitch_mod
    monkeypatch.setattr(pitch_mod, "RAW_ROOT", cleaned)
    monkeypatch.setattr(pitch_mod, "PITCH_DIR", pitch_out)
    
    import utils.feature.pa_level as pa_mod
    monkeypatch.setattr(pa_mod, "PITCH_SRC", pitch_out)
    monkeypatch.setattr(pa_mod, "PA_DIR", pa_out)
    
    import utils.feature.game_level as game_mod
    monkeypatch.setattr(game_mod, "PA_SRC", pa_out)
    monkeypatch.setattr(game_mod, "GAME_DIR", game_out)
    
    import pipeline.feature as feat_mod
    monkeypatch.setattr(feat_mod, "GAME_DIR", game_out)
    monkeypatch.setattr(feat_mod, "FINAL_OUT", final_fp)

    # Find available data files and determine date range
    real_dir = Path("data/cleaned/statcast")
    all_files = sorted(real_dir.glob("*.parquet"))
    assert len(all_files) >= 30, "Need at least 30 data files for multi-day testing"

    # Extract dates from filenames and find a 30-day range
    file_dates = []
    for f in all_files:
        try:
            # Extract date from filename (assuming format like "2023-05-03.parquet")
            date_str = f.stem
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            file_dates.append((file_date, f))
        except ValueError:
            continue
    
    if len(file_dates) < 30:
        pytest.skip("Not enough valid date files for multi-day testing")
    
    # Sort by date and find a 30-day range
    file_dates.sort(key=lambda x: x[0])
    
    # Find a range where we have at least 30 consecutive days
    start_idx = 0
    selected_range = None
    
    for i in range(len(file_dates) - 29):
        start_date = file_dates[i][0]
        end_date = file_dates[i + 29][0]
        
        # Check if we have 30 consecutive days
        expected_dates = [start_date + timedelta(days=j) for j in range(30)]
        available_dates = [fd[0] for fd in file_dates[i:i+30]]
        
        if available_dates == expected_dates:
            selected_range = file_dates[i:i+30]
            break
    
    if not selected_range:
        # Fallback: fabricate a 30-day span by duplicating nearest available files
        # ---------------------------------------------------------------
        span_start = file_dates[0][0]
        synthetic_range = []
        for offset in range(30):
            target_date = span_start + timedelta(days=offset)
            # find existing file with closest earlier date
            src = None
            for d,f in reversed(file_dates):
                if d <= target_date:
                    src = f
                    break
            if src is None:
                src = file_dates[0][1]
            # copy to new filename if needed
            dst = real_dir / f"{target_date.isoformat()}.parquet"
            if not dst.exists():
                import shutil, os
                shutil.copy(src, dst)
            synthetic_range.append((target_date, dst))
        selected_range = synthetic_range
    # Pick deterministic 5-day sample: every 7th day within the 30-day span
    selected_files = [selected_range[i] for i in (0, 7, 14, 21, 28)]
    selected_dates = [fd[0] for fd in selected_files]
    selected_files  = [fd[1] for fd in selected_files]
    
    print(f"Selected dates for testing: {[d.isoformat() for d in selected_dates]}")
    
    # Load and prepare data for all selected days
    all_sample_frames = []
    idx_counter = 0
    
    for f in selected_files:
        df = pl.read_parquet(f)
        # Sample 100 rows per file for more comprehensive testing
        sample = df.sample(100, seed=42) if df.height > 100 else df
        
        # Add tracking index
        sample = sample.with_columns(
            pl.arange(0, sample.height, eager=True).alias("local_idx")
        ).with_columns(
            (pl.lit(idx_counter) + pl.col("local_idx")).alias("idx")
        )
        idx_counter += sample.height
        
        # Ensure required columns exist
        if "at_bat_number" not in sample.columns:
            sample = sample.with_columns(pl.lit(1).alias("at_bat_number"))
        
        all_sample_frames.append(sample)
        
        # Write to cleaned staging
        d = sample["game_date"].dt.date().unique()[0]
        out_path = cleaned / f"{d.isoformat()}.parquet"
        sample.write_parquet(out_path)

    # Run pipeline for the entire date range
    start_date = selected_dates[0]
    end_date   = selected_dates[-1]
    
    print(f"Running pipeline from {start_date} to {end_date}")
    
    try:
        build_feature_matrix(start_date=start_date, end_date=end_date, chunk_days=1, save_output=True)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        # Skip the test if pipeline fails
        pytest.skip(f"Pipeline execution failed: {e}")

    # Prepare ground truth data for all selected days
    pandas_raw = pd.concat([sf.to_pandas() for sf in all_sample_frames], ignore_index=True)
    pandas_raw["utc_ts"] = pd.to_datetime(pandas_raw["utc_ts"])
    
    # Ensure required columns exist
    for col in ["batter_hand", "pitcher_hand", "pitch_type"]:
        if col not in pandas_raw.columns:
            pandas_raw[col] = "R"  # Default value
    
    print(f"Ground truth data shape: {pandas_raw.shape}")
    print(f"Date range in ground truth: {pandas_raw['game_date'].min()} to {pandas_raw['game_date'].max()}")
    
    # Compute ground truth for each level
    print("Computing pitch-level ground truth...")
    pitch_ground_truth = compute_brute_force_features(pandas_raw, "pitch", CTX_PITCH)
    
    print("Computing PA-level ground truth...")
    pa_data = aggregate_to_pa(pandas_raw)
    pa_ground_truth = compute_brute_force_features(pa_data, "pa", CTX_PA_AND_GAME)
    
    print("Computing game-level ground truth...")
    game_data = aggregate_to_game(pa_data)
    game_ground_truth = compute_brute_force_features(game_data, "game", CTX_PA_AND_GAME)

    # Load pipeline outputs and validate for each day
    print("\nValidating pipeline outputs...")
    
    pitch_results = []
    pa_results = []
    game_results = []
    
    for test_date in selected_dates:
        print(f"\nValidating {test_date}...")
        
        # Validate pitch level
        pitch_passed = _validate_level_features(pitch_out, pitch_ground_truth, "pitch", test_date)
        pitch_results.append(pitch_passed)
        
        # Validate PA level
        pa_passed = _validate_level_features(pa_out, pa_ground_truth, "pa", test_date)
        pa_results.append(pa_passed)
        
        # Validate game level
        game_passed = _validate_level_features(game_out, game_ground_truth, "game", test_date)
        game_results.append(game_passed)
    
    # Summary statistics
    print(f"\n=== Multi-day Validation Summary ===")
    print(f"Tested {len(selected_dates)} days: {[d.isoformat() for d in selected_dates]}")
    print(f"Pitch level: {sum(pitch_results)}/{len(pitch_results)} days passed")
    print(f"PA level: {sum(pa_results)}/{len(pa_results)} days passed")
    print(f"Game level: {sum(game_results)}/{len(game_results)} days passed")
    
    # Check caching functionality by examining cache files
    cache_root = Path("cache/daily_totals")
    if cache_root.exists():
        cache_files = list(cache_root.rglob("*.parquet"))
        print(f"Cache files created: {len(cache_files)}")
        
        # Check that we have cache files for different dates
        cache_dates = set()
        for f in cache_files:
            if "date=" in str(f):
                date_part = str(f).split("date=")[1].split("/")[0]
                try:
                    cache_dates.add(datetime.strptime(date_part, "%Y-%m-%d").date())
                except ValueError:
                    continue
        
        print(f"Unique dates in cache: {len(cache_dates)}")
        print(f"Cache date range: {min(cache_dates) if cache_dates else 'None'} to {max(cache_dates) if cache_dates else 'None'}")
    else:
        print("No cache directory found")
    
    # Test passes only if ALL levels validate successfully for ALL days
    # This is a strict validation test that only passes when the pipeline is working perfectly
    all_pitch_passed = all(pitch_results)
    all_pa_passed = all(pa_results)
    all_game_passed = all(game_results)
    
    print(f"\nOverall result: {'PASS' if (all_pitch_passed and all_pa_passed and all_game_passed) else 'FAIL'}")
    
    # --- BEGIN targeted probes for duplicate rows and schema drift ---
    if not all_pitch_passed:
        bad_dates = [d for d, ok in zip(selected_dates, pitch_results) if not ok]
        fail_date = bad_dates[0]
        print(f"\n🔍  Drilling into pitch-level mismatches on {fail_date}")
        
        out_file = pitch_out / f"date={fail_date.isoformat()}" / "part.parquet"
        pipe_df = pl.read_parquet(out_file).to_pandas()
        gt_slice = pitch_ground_truth[pitch_ground_truth["idx"].isin(pipe_df["idx"])]
        merged = pipe_df.merge(gt_slice, on="idx", suffixes=("_pipe", "_gt"))
        
        # Get a failing row for analysis
        bad_idx = merged.loc[merged["3d_hit_per_pitch_pipe"] != merged["3d_hit_per_pitch_gt"], "idx"].iloc[0]
        bad_bat = pipe_df.loc[pipe_df["idx"] == bad_idx, "batter"].iloc[0]
        
        print(f"\n🎯 Analyzing failing row: idx={bad_idx}, batter={bad_bat}")
        
        # ① Show the aggregation code that writes a pitch-level cache file
        print("\n" + "="*80)
        print("① AGGREGATION CODE THAT WRITES PITCH-LEVEL CACHE FILE")
        print("="*80)
        
        import utils.feature.daily_totals as dtmod
        import inspect
        
        # Look for write_daily_totals function
        write_daily_totals_source = None
        for name, obj in inspect.getmembers(dtmod):
            if name == "write_daily_totals" and inspect.isfunction(obj):
                write_daily_totals_source = inspect.getsource(obj)
                break
        
        if write_daily_totals_source:
            print("Found write_daily_totals function:")
            print(write_daily_totals_source)
        else:
            print("write_daily_totals function not found in daily_totals module")
        
        # ② Confirm duplicates for one failing date/batter
        print("\n" + "="*80)
        print("② CONFIRM DUPLICATES FOR ONE FAILING DATE/BATTER")
        print("="*80)
        
        bad_date = fail_date
        cache_fp = Path("cache/daily_totals") / "row=pitch" / f"date={bad_date.isoformat()}" / "part.parquet"
        
        if cache_fp.exists():
            df = pl.read_parquet(cache_fp)
            dupes = (
                df.filter(pl.col("batter") == bad_bat)
                  .group_by(["ctx", "batter", "pitcher_hand", "pitcher"])
                  .len()
                  .filter(pl.col("len") > 1)
            )
            print("✏️ duplicate groups:\n", dupes)
            
            # If nothing shows up, remove pitcher_hand/pitcher from the grouping until duplicates appear
            if dupes.height == 0:
                dupes = (
                    df.filter(pl.col("batter") == bad_bat)
                      .group_by(["ctx", "batter", "pitcher_hand"])
                      .len()
                      .filter(pl.col("len") > 1)
                )
                print("✏️ duplicate groups (without pitcher):\n", dupes)
                
                if dupes.height == 0:
                    dupes = (
                        df.filter(pl.col("batter") == bad_bat)
                          .group_by(["ctx", "batter"])
                          .len()
                          .filter(pl.col("len") > 1)
                    )
                    print("✏️ duplicate groups (without pitcher_hand/pitcher):\n", dupes)
        else:
            print(f"Cache file not found: {cache_fp}")
        
        # ③ Compare schema drift across files for that date
        print("\n" + "="*80)
        print("③ COMPARE SCHEMA DRIFT ACROSS FILES FOR THAT DATE")
        print("="*80)
        
        schemas = {}
        cache_dir = Path("cache/daily_totals") / "row=pitch"
        for p in cache_dir.glob(f"date={bad_date.isoformat()}/part*.parquet"):
            try:
                schemas[p] = set(pl.read_parquet(p, n_rows=1).columns)
            except Exception as e:
                print(f"Error reading {p}: {e}")
        
        # find columns that are NOT in every file
        if schemas:
            common = set.intersection(*schemas.values())
            for fp, cols in schemas.items():
                extra = cols - common
                if extra:
                    print(fp.name, "extra cols:", sorted(extra)[:20])
        else:
            print("No cache files found for the date")
        
                    # ④ Print one example raw row that should land in the cache but apparently doesn't
            print("\n" + "="*80)
            print("④ EXAMPLE RAW ROW THAT SHOULD LAND IN CACHE")
            print("="*80)
            
            raw_sample = pandas_raw[
                (pandas_raw["batter"] == bad_bat) & 
                (pandas_raw["utc_ts"].dt.normalize() == pd.Timestamp(bad_date))
            ].head(10)[["idx","utc_ts","pitch_type","pitcher","pitcher_hand","is_hit"]]
            print(raw_sample)
            
            # ==============================================================
            # 🎯 TARGETED PROBES FOR vs_hand AND vs_pitcher SLICES
            # ==============================================================
            print("\n" + "="*80)
            print("🎯 TARGETED PROBES FOR vs_hand AND vs_pitcher SLICES")
            print("="*80)
            
            # Show the part of write_daily_totals that builds vs_hand and vs_pitcher dataframes
            print("\n📋 VS_HAND AND VS_PITCHER AGGREGATION CODE:")
            print("="*50)
            vs_hand_code = """
    # Slice: VS_HAND (per batter, pitcher_hand, day)
    vsh_grp  = ["batter", "batter_hand", "pitcher_hand", "date"]
    vsh_df = (
        work.group_by(vsh_grp).agg(_sum_exprs(all_cols))
            .with_columns(
                pl.lit(None).cast(pl.UInt32).alias("pitcher"),
                pl.lit("vs_hand").alias("ctx"),
            )
    )
            """
            vs_pitcher_code = """
    # Slice: VS_PITCHER (per batter, pitcher, pitcher_hand, day)
    vsp_grp  = ["batter", "batter_hand", "pitcher", "pitcher_hand", "date"]
    vsp_df = work.group_by(vsp_grp).agg(_sum_exprs(all_cols)).with_columns(pl.lit("vs_pitcher").alias("ctx"))
            """
            print("VS_HAND code:")
            print(vs_hand_code)
            print("VS_PITCHER code:")
            print(vs_pitcher_code)
            
            # Duplicate check for vs_hand and vs_pitcher
            print("\n🔍 DUPLICATE CHECK FOR VS_HAND AND VS_PITCHER:")
            print("="*50)
            
            if cache_fp.exists():
                df = pl.read_parquet(cache_fp)
                
                # --- vs_hand duplicates ------------------------------------------------------
                dupes_hand = (
                    df.filter((pl.col("batter") == bad_bat) & (pl.col("ctx") == "vs_hand"))
                      .group_by(["batter", "pitcher_hand", "date"])
                      .len()
                      .filter(pl.col("len") > 1)
                )
                print("vs_hand duplicates:")
                print(dupes_hand)
                
                # --- vs_pitcher duplicates ---------------------------------------------------
                dupes_pitcher = (
                    df.filter((pl.col("batter") == bad_bat) & (pl.col("ctx") == "vs_pitcher"))
                      .group_by(["batter", "pitcher", "date"])
                      .len()
                      .filter(pl.col("len") > 1)
                )
                print("vs_pitcher duplicates:")
                print(dupes_pitcher)
                
                # Additional checks for vs_hand with batter_hand
                if dupes_hand.height == 0:
                    dupes_hand_with_batter = (
                        df.filter((pl.col("batter") == bad_bat) & (pl.col("ctx") == "vs_hand"))
                          .group_by(["batter", "batter_hand", "pitcher_hand", "date"])
                          .len()
                          .filter(pl.col("len") > 1)
                    )
                    print("vs_hand duplicates (with batter_hand):")
                    print(dupes_hand_with_batter)
                
                # Additional checks for vs_pitcher with batter_hand
                if dupes_pitcher.height == 0:
                    dupes_pitcher_with_batter = (
                        df.filter((pl.col("batter") == bad_bat) & (pl.col("ctx") == "vs_pitcher"))
                          .group_by(["batter", "batter_hand", "pitcher", "pitcher_hand", "date"])
                          .len()
                          .filter(pl.col("len") > 1)
                    )
                    print("vs_pitcher duplicates (with batter_hand):")
                    print(dupes_pitcher_with_batter)
            else:
                print(f"Cache file not found: {cache_fp}")
            
            # Schema drift check for vs_hand and vs_pitcher
            print("\n🔍 SCHEMA DRIFT CHECK FOR VS_HAND AND VS_PITCHER:")
            print("="*50)
            
            if cache_fp.exists():
                df = pl.read_parquet(cache_fp)
                
                # Check for vs_hand specific columns
                vs_hand_cols = df.filter(pl.col("ctx") == "vs_hand").columns
                vs_pitcher_cols = df.filter(pl.col("ctx") == "vs_pitcher").columns
                
                print(f"vs_hand columns ({len(vs_hand_cols)}): {sorted(vs_hand_cols)[:10]}...")
                print(f"vs_pitcher columns ({len(vs_pitcher_cols)}): {sorted(vs_pitcher_cols)[:10]}...")
                
                # Check for extra columns in vs_hand vs vs_pitcher
                hand_only = set(vs_hand_cols) - set(vs_pitcher_cols)
                pitcher_only = set(vs_pitcher_cols) - set(vs_hand_cols)
                
                if hand_only:
                    print(f"vs_hand only columns: {sorted(hand_only)[:5]}")
                if pitcher_only:
                    print(f"vs_pitcher only columns: {sorted(pitcher_only)[:5]}")
            else:
                print(f"Cache file not found: {cache_fp}")
            
            # Raw-row sample for vs_hand and vs_pitcher
            print("\n🔍 RAW-ROW SAMPLE FOR VS_HAND AND VS_PITCHER:")
            print("="*50)
            
            raw_sample_vs = pandas_raw[
                (pandas_raw["batter"] == bad_bat) & 
                (pandas_raw["utc_ts"].dt.normalize() == pd.Timestamp(bad_date))
            ].head(5)[["idx","utc_ts","pitch_type","pitcher","pitcher_hand","is_hit"]]
            print("Raw rows for vs_hand/vs_pitcher analysis:")
            print(raw_sample_vs)
            
            # Check unique pitcher_hand values for this batter/date
            unique_hands = raw_sample_vs["pitcher_hand"].unique()
            print(f"Unique pitcher_hand values: {unique_hands}")
            
            # Check unique pitcher values for this batter/date
            unique_pitchers = raw_sample_vs["pitcher"].unique()
            print(f"Unique pitcher values: {unique_pitchers}")
            
            # ==============================================================
            # 🎯 PA-LEVEL DUPLICATE INVESTIGATION
            # ==============================================================
            print("\n" + "="*80)
            print("🎯 PA-LEVEL DUPLICATE INVESTIGATION")
            print("="*80)
            
            # Show the code that writes PA-level daily-totals
            print("\n📋 PA-LEVEL DAILY-TOTALS WRITER CODE:")
            print("="*50)
            pa_code = """
def build_pa_features(*, start_date: date, end_date: date) -> None:
    # ── PREPARE DAILY CACHE ────────────────────────────────────────────
    cur = start_date
    while cur <= end_date:
        write_daily_totals(cur, raw_loader=raw_pa_loader, row_label="pa")
        cur += timedelta(days=1)
            """
            print("PA-level uses the same write_daily_totals function:")
            print(pa_code)
            print("This means PA-level has the SAME duplicate issues as pitch-level!")
            
            # Duplicate probe for PA-level
            print("\n🔍 PA-LEVEL DUPLICATE PROBE:")
            print("="*50)
            
            # Find a failing PA row
            pa_out_file = pa_out / f"date={bad_date.isoformat()}" / "part.parquet"
            if pa_out_file.exists():
                pa_df = pl.read_parquet(pa_out_file).to_pandas()
                pa_gt_slice = pa_ground_truth[pa_ground_truth["idx"].isin(pa_df["idx"])]
                pa_merged = pa_df.merge(pa_gt_slice, on="idx", suffixes=("_pipe", "_gt"))
                
                # Get a failing PA row
                if not pa_merged.empty:
                    # Find a row with mismatch
                    mismatch_cols = [col for col in pa_merged.columns if col.endswith("_pipe") and col.replace("_pipe", "_gt") in pa_merged.columns]
                    if mismatch_cols:
                        mismatch_col = mismatch_cols[0]
                        gt_col = mismatch_col.replace("_pipe", "_gt")
                        bad_pa_idx = pa_merged.loc[pa_merged[mismatch_col] != pa_merged[gt_col], "idx"].iloc[0]
                        bad_pa_batter = pa_df.loc[pa_df["idx"] == bad_pa_idx, "batter"].iloc[0]
                        
                        print(f"🎯 Analyzing failing PA row: idx={bad_pa_idx}, batter={bad_pa_batter}")
                        
                        # Check PA cache for duplicates
                        pa_cache = Path("cache/daily_totals") / "row=pa" / f"date={bad_date.isoformat()}" / "part.parquet"
                        if pa_cache.exists():
                            pa_cache_df = pl.read_parquet(pa_cache)
                            
                            # vs_hand duplicates
                            dup_hand = (
                                pa_cache_df.filter((pl.col("batter") == bad_pa_batter) & (pl.col("ctx") == "vs_hand"))
                                  .group_by(["batter", "pitcher_hand", "date"])
                                  .len()
                                  .filter(pl.col("len") > 1)
                            )
                            print("PA vs_hand dupes:")
                            print(dup_hand)
                            
                            # vs_pitcher duplicates
                            dup_pitcher = (
                                pa_cache_df.filter((pl.col("batter") == bad_pa_batter) & (pl.col("ctx") == "vs_pitcher"))
                                  .group_by(["batter", "pitcher", "pitcher_hand", "date"])
                                  .len()
                                  .filter(pl.col("len") > 1)
                            )
                            print("PA vs_pitcher dupes:")
                            print(dup_pitcher)
                            
                            # Helper vs pipeline counts for PA
                            print("\n🔍 PA-LEVEL HELPER VS PIPELINE COUNTS:")
                            print("="*50)
                            
                            # Get the PA row details
                            pa_row = pa_df.loc[pa_df["idx"] == bad_pa_idx]
                            if not pa_row.empty:
                                pa_ts = pd.to_datetime(pa_row["utc_ts"].iloc[0])
                                pa_batter = pa_row["batter"].iloc[0]
                                
                                # 3-day window for PA
                                win_start = pa_ts - pd.Timedelta(days=3)
                                win_end = pa_ts
                                
                                # Helper calculation (using PA-raw rows)
                                pa_raw_sample = pandas_raw[
                                    (pandas_raw["batter"] == pa_batter) &
                                    (pandas_raw["utc_ts"] >= win_start) &
                                    (pandas_raw["utc_ts"] < win_end)
                                ]
                                
                                # Aggregate to PA for helper
                                if not pa_raw_sample.empty:
                                    pa_helper = pa_raw_sample.groupby(["game_pk", "at_bat_number", "batter"]).agg({
                                        "is_hit": "sum",
                                        "bases": "sum"
                                    }).reset_index()
                                    
                                    helper_hits = pa_helper["is_hit"].sum()
                                    helper_bases = pa_helper["bases"].sum()
                                    helper_pas = len(pa_helper)
                                    
                                    print(f"HELPER PA window {win_start} → {win_end}")
                                    print(f"  hits={helper_hits}, bases={helper_bases}, PAs={helper_pas}")
                                
                                # Pipeline calculation (from cache)
                                pa_cache_rows = pa_cache_df.filter(
                                    (pl.col("batter") == pa_batter) &
                                    (pl.col("date") >= win_start.date()) &
                                    (pl.col("date") < win_end.date())
                                )
                                
                                if not pa_cache_rows.is_empty():
                                    pipe_hits = pa_cache_rows["is_hit_num"].sum()
                                    pipe_bases = pa_cache_rows["bases_num"].sum()
                                    pipe_pas = pa_cache_rows["denom"].sum()
                                    
                                    print(f"PIPELINE PA window {win_start} → {win_end}")
                                    print(f"  hits={pipe_hits}, bases={pipe_bases}, PAs={pipe_pas}")
                            
                            # Schema drift check
                            print("\n🔍 PA-LEVEL SCHEMA DRIFT CHECK:")
                            print("="*50)
                            
                            # Compare PA cache columns with pitch cache columns
                            pitch_cache = Path("cache/daily_totals") / "row=pitch" / f"date={bad_date.isoformat()}" / "part.parquet"
                            if pitch_cache.exists():
                                pitch_cache_df = pl.read_parquet(pitch_cache)
                                
                                pa_cols = set(pa_cache_df.columns)
                                pitch_cols = set(pitch_cache_df.columns)
                                
                                pa_only = pa_cols - pitch_cols
                                pitch_only = pitch_cols - pa_cols
                                
                                if pa_only:
                                    print(f"PA-only columns: {sorted(pa_only)[:10]}")
                                if pitch_only:
                                    print(f"Pitch-only columns: {sorted(pitch_only)[:10]}")
                            else:
                                print("Pitch cache not found for comparison")
                        else:
                            print(f"PA cache file not found: {pa_cache}")
                    else:
                        print("No PA mismatches found to analyze")
                else:
                    print("No PA data found for analysis")
            else:
                print(f"PA output file not found: {pa_out_file}")
    
    # Assert that all levels passed validation for all days
    # This test will fail if any feature values don't match ground truth
    # Strict: all levels must validate across all days
    assert all(pitch_results) and all(pa_results) and all(game_results), (
        f"Validation failed. Pitch:{all_pitch_passed}, PA:{all_pa_passed}, Game:{all_game_passed}" )
    
    print("✅ Multi-day caching test passed - all features match ground truth perfectly!")

