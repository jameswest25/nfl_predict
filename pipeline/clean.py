"""Cleans raw NFL play-by-play data to tidy Parquet optimized for feature engineering."""
from __future__ import annotations

import logging
import pandas as pd
from pathlib import Path
from utils.clean.audit import run_clean_audit
from utils.general.audit.tracer import DataTracer, clear_trace_log

# Import cleaning modules (NFL-only)
from utils.clean.io import save_clean
from utils.clean.clean_optimize import optimize_data
from utils.collect import attach_weather  # Use SQLite-optimized weather attachment


# Stub functions for backwards compatibility (not used in current NFL pipeline)


def clean_raw(df: pd.DataFrame = None, use_sample: bool = False, tracer: DataTracer = None, save_output: bool = True) -> pd.DataFrame:
    """
    Load and clean NFL play-by-play data.
    
    Args:
        df: DataFrame to clean (if None, loads from files)
        use_sample: Whether to use sample data for tracing
        tracer: DataTracer for debugging
        save_output: Whether to save cleaned data to files (default True)
    
    Returns:
        Cleaned DataFrame with NFL-specific features
    """
    
    def _trace(df, step_name):
        if tracer:
            tracer.trace(df, step_name)

    if df is None:
        # Load NFL data from pbp_by_day parquet files
        logging.info("Loading NFL data from pbp_by_day parquet files...")
        pbp_dir = Path("data/raw/pbp_by_day")
        
        if not pbp_dir.exists():
            raise FileNotFoundError(f"NFL data directory not found: {pbp_dir}")
        
        parquet_files = list(pbp_dir.glob("season=*/week=*/date=*/part.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {pbp_dir}")
        
        logging.info(f"Found {len(parquet_files)} parquet files to clean")
        
        # Load all parquet files
        dfs = []
        for pq_file in parquet_files:
            try:
                df_chunk = pd.read_parquet(pq_file)
                dfs.append(df_chunk)
            except Exception as e:
                logging.warning(f"Failed to load {pq_file}: {e}")
        
        if not dfs:
            raise ValueError("No data could be loaded from parquet files")
        
        df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Loaded {len(df)} rows from {len(dfs)} parquet files")
    
    _trace(df, "Initial Load")

    # NFL-only cleaning path (no imputation, minimal transforms)
    logging.info("Running NFL cleaning branch (no imputation, minimal transforms)")
    try:
        # Normalize datetime columns
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        if 'utc_ts' in df.columns:
            df['utc_ts'] = pd.to_datetime(df['utc_ts'], errors='coerce', utc=True)
        # Basic dtype coercions (no imputation)
        for c in ('season','week'):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
        # Attach weather using NFL stadium coords (with SQLite cache optimization)
        try:
            df = attach_weather(df)
        except Exception as e:
            logging.warning(f"Weather attachment skipped: {e}")

        # Per-play normalization for feature readiness (no drops, no impute)
        def _safe_col(name: str) -> bool:
            return name in df.columns

        # Ensure standard yardage columns exist; prefer existing nflfastR fields
        # rushing_yards / passing_yards / receiving_yards
        if not _safe_col('rushing_yards') and _safe_col('yards_gained') and _safe_col('rush'):
            df['rushing_yards'] = df['yards_gained'].where(df['rush'] == 1)
        if not _safe_col('passing_yards') and _safe_col('yards_gained') and _safe_col('pass'):
            df['passing_yards'] = df['yards_gained'].where(df['pass'] == 1)
        if not _safe_col('receiving_yards') and _safe_col('yards_gained') and _safe_col('pass'):
            df['receiving_yards'] = df['yards_gained'].where(df['pass'] == 1)

        # Ensure key player columns exist (keep originals if present)
        for pid, pname in (
            ('passer_player_id', 'passer_player_name'),
            ('rusher_player_id', 'rusher_player_name'),
            ('receiver_player_id', 'receiver_player_name'),
        ):
            if not _safe_col(pid):
                df[pid] = pd.NA
            if not _safe_col(pname):
                df[pname] = pd.NA

        # Outcome labeling with existing boolean flags; preserve originals
        def _mk_outcome(flags: dict) -> str:
            if flags.get('rush_touchdown', 0) == 1:
                return 'rush_td'
            if flags.get('pass_touchdown', 0) == 1:
                return 'pass_td'
            if flags.get('interception', 0) == 1:
                return 'interception'
            if flags.get('fumble', 0) == 1:
                return 'fumble'
            if flags.get('sack', 0) == 1:
                return 'sack'
            if flags.get('qb_spike', 0) == 1:
                return 'qb_spike'
            if flags.get('qb_kneel', 0) == 1:
                return 'qb_kneel'
            if flags.get('penalty', 0) == 1:
                return 'penalty'
            if flags.get('pass', 0) == 1:
                if flags.get('complete_pass', 0) == 1:
                    return 'pass_complete'
                if flags.get('incomplete_pass', 0) == 1:
                    return 'pass_incomplete'
                return 'pass'
            if flags.get('rush', 0) == 1:
                return 'rush'
            return 'other'

        cols_needed = ['rush_touchdown','pass_touchdown','interception','fumble','sack',
                       'qb_spike','qb_kneel','penalty','pass','complete_pass','incomplete_pass','rush']
        if any(_safe_col(c) for c in cols_needed):
            def _row_outcome(t):
                d = {k: getattr(t, k, 0) if hasattr(t, k) else 0 for k in cols_needed}
                return _mk_outcome(d)
            df['play_outcome'] = [
                _row_outcome(t) for t in df.itertuples(index=True)
            ]
            logging.info("play_outcome distribution: %s", df['play_outcome'].value_counts(dropna=False).to_dict())

        # Add derived stat columns for rolling window features
        logging.info("Adding derived stat columns for feature engineering...")

        # Pass attempts: prefer nflfastR flag, fall back to legacy heuristic
        if _safe_col('pass_attempt'):
            pass_attempt_series = pd.to_numeric(df['pass_attempt'], errors='coerce').fillna(0)
            df['pass_attempt'] = pass_attempt_series.astype('int8')
        else:
            pass_attempt_series = pd.Series(0, index=df.index)
            if _safe_col('passer_player_id') and _safe_col('pass'):
                pass_attempt_series = (
                    (df['passer_player_id'].notna()) & (df['pass'] == 1)
                ).astype('int8')
            df['pass_attempt'] = pass_attempt_series.astype('int8')
        pass_attempt_series = df['pass_attempt']

        # Completions: leverage official complete_pass indicator when available
        if _safe_col('complete_pass'):
            df['completion'] = pd.to_numeric(df['complete_pass'], errors='coerce').fillna(0).astype('int8')
        else:
            completion_mask = pd.Series(False, index=df.index)
            if _safe_col('pass') and _safe_col('incomplete_pass'):
                completion_mask = (df['pass'].fillna(0) == 1) & (df['incomplete_pass'].fillna(0) == 0)
            elif _safe_col('passing_yards') or _safe_col('pass_touchdown'):
                yards = df.get('passing_yards', 0).fillna(0)
                completion_mask = (df.get('pass_touchdown', 0).fillna(0) == 1) | (yards.astype(float).notna())
            df['completion'] = completion_mask.astype('int8')

        # Receptions: direct from nflfastR when present; otherwise infer from completion + receiver
        if _safe_col('reception'):
            df['reception'] = pd.to_numeric(df['reception'], errors='coerce').fillna(0).astype('int8')
        else:
            reception_mask = pd.Series(False, index=df.index)
            if _safe_col('receiver_player_id'):
                reception_mask = (df['receiver_player_id'].notna()) & (df['completion'] == 1)
            df['reception'] = reception_mask.astype('int8')

        # Targets: pass attempt to an identified receiver
        df['target'] = 0
        if _safe_col('receiver_player_id'):
            target_mask = (df['receiver_player_id'].notna()) & (pass_attempt_series > 0)
            df['target'] = target_mask.astype('int8')

        # Carries: prefer rush_attempt indicator, otherwise legacy rush flag heuristic
        if _safe_col('rush_attempt'):
            df['carry'] = pd.to_numeric(df['rush_attempt'], errors='coerce').fillna(0).astype('int8')
        else:
            carry_mask = pd.Series(False, index=df.index)
            if _safe_col('rusher_player_id') and _safe_col('rush'):
                carry_mask = (df['rusher_player_id'].notna()) & (df['rush'] == 1)
            df['carry'] = carry_mask.astype('int8')

        # Touchdowns: track scoring player separately from passing TD credit
        df['touchdown_player_id'] = pd.NA
        if _safe_col('td_player_id'):
            df['touchdown_player_id'] = df['td_player_id']

        def _assign_touchdown(flag_col: str, player_col: str):
            if not (_safe_col(flag_col) and _safe_col(player_col)):
                return
            mask = (
                df['touchdown_player_id'].isna()
                & (df[flag_col].fillna(0) == 1)
                & df[player_col].notna()
            )
            if mask.any():
                df.loc[mask, 'touchdown_player_id'] = df.loc[mask, player_col]

        _assign_touchdown('rush_touchdown', 'rusher_player_id')
        _assign_touchdown('pass_touchdown', 'receiver_player_id')
        _assign_touchdown('kickoff_return_touchdown', 'kickoff_returner_player_id')
        _assign_touchdown('punt_return_touchdown', 'punt_returner_player_id')
        _assign_touchdown('interception_return_touchdown', 'interception_player_id')
        _assign_touchdown('fumble_return_touchdown', 'fumble_recovery_1_player_id')
        _assign_touchdown('return_touchdown', 'returner_player_id')

        df['touchdown'] = df['touchdown_player_id'].notna().astype('int8')
        logging.info("Touchdown attribution complete: %d scoring plays identified", df['touchdown'].sum())

        # Situational derived stats
        if _safe_col('yardline_100'):
            yardline_vals = pd.to_numeric(df['yardline_100'], errors='coerce')
            red_zone_mask = yardline_vals.le(20).fillna(False)
        else:
            red_zone_mask = pd.Series(False, index=df.index)

        if _safe_col('goal_to_go'):
            goal_to_go_mask = df['goal_to_go'].fillna(False).astype(bool)
        else:
            goal_to_go_mask = pd.Series(False, index=df.index)

        df['red_zone_target'] = ((df['target'] == 1) & red_zone_mask).astype('int8')
        df['red_zone_carry'] = ((df['carry'] == 1) & red_zone_mask).astype('int8')
        df['goal_to_go_target'] = ((df['target'] == 1) & goal_to_go_mask).astype('int8')
        df['goal_to_go_carry'] = ((df['carry'] == 1) & goal_to_go_mask).astype('int8')

        logging.info("Derived stats added: target, reception, carry, pass_attempt, completion, touchdown, situational splits")

        # Compact human-readable players_involved string
        def _players_str(row):
            parts = []
            try:
                if pd.notna(row.get('passer_player_name')):
                    parts.append(f"PASSER:{row['passer_player_name']}")
                if pd.notna(row.get('rusher_player_name')):
                    parts.append(f"RUSHER:{row['rusher_player_name']}")
                if pd.notna(row.get('receiver_player_name')):
                    parts.append(f"RECEIVER:{row['receiver_player_name']}")
            except Exception:
                return pd.NA
            return " | ".join(parts) if parts else pd.NA
        try:
            df['players_involved'] = df.apply(_players_str, axis=1)
        except Exception:
            pass

        # Optimize dtypes
        df = optimize_data(df)
        _trace(df, "NFL Clean Complete")
        if save_output:
            save_clean(df)
        return df
    except Exception as e:
        logging.error(f"NFL cleaning branch failed: {e}")
        if save_output:
            try:
                save_clean(df)
            except Exception as save_err:
                logging.error("Failed to save cleaned data after error: %s", save_err)
        return df


def clean(use_sample: bool = False, trace: bool = False, season: int | None = None, week: int | None = None, start_date=None, end_date=None):
    """Main cleaning pipeline (NFL-only). Optionally scope to season/week weekly PBP input."""
    logging.info("Starting data cleaning pipeline with feature engineering")
    # Note: start_date and end_date are accepted for compatibility but not used in NFL path
    
    tracer = None
    if trace:
        clear_trace_log()
        tracer = DataTracer()

    # If season/week specified, load weekly PBP partition as input; else fall back to monolithic loader
    if season is not None and week is not None:
        from pathlib import Path
        import pandas as pd
        in_fp = Path("data/processed/pbp_by_week") / f"season={int(season)}" / f"week={int(week)}" / "part.parquet"
        if not in_fp.exists():
            raise FileNotFoundError(f"Weekly PBP not found: {in_fp}")
        logging.info(f"Loading weekly PBP for cleaning → {in_fp}")
        df_in = pd.read_parquet(in_fp)
        df_clean = clean_raw(df=df_in, use_sample=use_sample, tracer=tracer)
    else:
        # Clean raw data (will use load_data fallback)
        df_clean = clean_raw(use_sample=use_sample, tracer=tracer)
    
    # Run audit on the freshly written daily-partitioned folder
    run_clean_audit()
    
    logging.info("Data cleaning and feature engineering completed successfully!")
    return df_clean


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean NFL play-by-play data")
    parser.add_argument("--audit-only", action="store_true",
                       help="Run audit on existing cleaned data without cleaning")
    parser.add_argument("--use-sample", action="store_true",
                        help="Use sample data for a quick run")
    parser.add_argument("--trace", action="store_true", help="Enable detailed pipeline tracing.")
    parser.add_argument("--season", type=int, default=None, help="Season year to clean (uses weekly PBP)")
    parser.add_argument("--week", type=int, default=None, help="Week number to clean (uses weekly PBP)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    
    if args.audit_only:
        run_clean_audit()
    else:
        clean(use_sample=args.use_sample, trace=args.trace, season=args.season, week=args.week)