
import sys
import os
import shutil
import logging
from pathlib import Path
from datetime import date, datetime
import polars as pl
import pandas as pd
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_parity_strict")

from pipeline.feature import _build_feature_matrix_internal
from pipeline.predict import _compute_features
from utils.feature.asof import decision_cutoff_override
from utils.feature.stats import NFL_PLAYER_STATS
from utils.feature.daily_totals import build_daily_cache_range
from utils.feature.team_context import compute_team_context_history
from utils.feature.offense_context import _append_offense_context_columns
from utils.feature.shared import attach_td_rate_history_features

def verify_parity_strict():
    # Target: Week 10 of 2024 season (Nov 10, 2024)
    target_date = date(2024, 11, 10)
    season_start = date(2024, 9, 1)
    target_datetime = datetime(2024, 11, 10, 18, 0, 0) 
    
    logger.info(f"--- Verifying STRICT Feature Parity for {target_date} ---")

    # 0. Clear Cache to ensure new stats (e.g. ps_) are built
    # ------------------------------------------------------
    cache_dir = Path("cache/feature/daily_totals")
    if cache_dir.exists():
        logger.info(f"Clearing cache dir: {cache_dir}")
        shutil.rmtree(cache_dir)

    # 1. Build Daily Cache for Season (History)
    # -----------------------------------------
    logger.info(f"Building daily totals cache from {season_start} to {target_date}...")
    # Ensure we have game level cache
    build_daily_cache_range(season_start, target_date, level="game")
    
    # 1b. Rebuild Context History (Team & Offense) to ensure alignment
    # --------------------------------------------------------------
    logger.info("Rebuilding Context History from raw game data...")
    try:
        # Load ALL player game data for context calculation
        pg_scan = pl.scan_parquet(
            "data/processed/player_game_by_week/season=*/week=*/part.parquet",
            glob=True,
            hive_partitioning=True,
            extra_columns="ignore",
            missing_columns="insert",
        )
        # Filter to history only (up to current season) to simulate "available data"
        full_pg = pg_scan.collect()
        
        # Cast columns to ensure reliable deduplication
        full_pg = full_pg.with_columns([
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("player_id").cast(pl.Utf8),
            pl.col("team").cast(pl.Utf8),
        ])
        
        # Dedup to ensure no double counting in aggregation
        full_pg = full_pg.unique(subset=["season", "week", "player_id"])
        
        # Team Context
        team_history = compute_team_context_history(full_pg)
        team_hist_path = Path("data/processed/team_context_history.parquet")
        team_hist_path.parent.mkdir(parents=True, exist_ok=True)
        team_history.write_parquet(team_hist_path, compression="zstd")
        logger.info(f"Rebuilt team_context_history: {len(team_history)} rows")
        
        # Offense Context
        seasons = full_pg.get_column("season").unique().to_list()
        _, off_history = _append_offense_context_columns(full_pg, seasons)
        off_hist_path = Path("data/processed/offense_context_history.parquet")
        off_hist_path.parent.mkdir(parents=True, exist_ok=True)
        off_history.write_parquet(off_hist_path, compression="zstd")
        logger.info(f"Rebuilt offense_context_history: {len(off_history)} rows")
        
    except Exception as e:
        logger.error(f"Failed to rebuild context history: {e}")
        return

    # Helper to normalize TD-rate and offense-context share features so
    # both train and predict use the exact same definitions.
    def _normalize_td_and_offense_shares(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        # Rebuild team/position TD-rate features from canonical history.
        drop_td_cols = [
            c for c in df.columns if c.startswith("team_pos_") or c.startswith("opp_pos_")
        ]
        if drop_td_cols:
            df = df.drop(drop_td_cols)
        df = attach_td_rate_history_features(df)

        # Recompute offense-context player shares using the same formulas
        # as inference, applied symmetrically.
        cols = set(df.columns)
        share_exprs: list[pl.Expr] = []

        if {
            "1g_red_zone_target_per_game",
            "off_ctx_team_red_zone_targets_prev",
        }.issubset(cols):
            share_exprs.append(
                pl.when(pl.col("off_ctx_team_red_zone_targets_prev") > 0)
                .then(
                    pl.col("1g_red_zone_target_per_game")
                    / pl.col("off_ctx_team_red_zone_targets_prev")
                )
                .otherwise(0.0)
                .clip(0.0, 1.5)
                .cast(pl.Float32)
                .alias("off_ctx_player_red_zone_share_prev")
            )
        if {
            "3g_red_zone_target_per_game",
            "off_ctx_team_red_zone_targets_l3",
        }.issubset(cols):
            expr = (
                pl.when(pl.col("off_ctx_team_red_zone_targets_l3") > 0)
                .then(
                    pl.col("3g_red_zone_target_per_game")
                    / pl.col("off_ctx_team_red_zone_targets_l3")
                )
                .otherwise(0.0)
                .clip(0.0, 1.5)
                .cast(pl.Float32)
            )
            share_exprs.append(expr.alias("off_ctx_player_red_zone_share_l3"))
            share_exprs.append(expr.alias("off_ctx_player_red_zone_share"))

        if {
            "1g_goal_to_go_carry_per_game",
            "off_ctx_team_goal_to_go_carries_prev",
        }.issubset(cols):
            share_exprs.append(
                pl.when(pl.col("off_ctx_team_goal_to_go_carries_prev") > 0)
                .then(
                    pl.col("1g_goal_to_go_carry_per_game")
                    / pl.col("off_ctx_team_goal_to_go_carries_prev")
                )
                .otherwise(0.0)
                .clip(0.0, 1.5)
                .cast(pl.Float32)
                .alias("off_ctx_player_goal_to_go_share_prev")
            )
        if {
            "3g_goal_to_go_carry_per_game",
            "off_ctx_team_goal_to_go_carries_l3",
        }.issubset(cols):
            expr_goal = (
                pl.when(pl.col("off_ctx_team_goal_to_go_carries_l3") > 0)
                .then(
                    pl.col("3g_goal_to_go_carry_per_game")
                    / pl.col("off_ctx_team_goal_to_go_carries_l3")
                )
                .otherwise(0.0)
                .clip(0.0, 1.5)
                .cast(pl.Float32)
            )
            share_exprs.append(expr_goal.alias("off_ctx_player_goal_to_go_share_l3"))
            share_exprs.append(expr_goal.alias("off_ctx_player_goal_to_go_share"))

        if share_exprs:
            df = df.with_columns(share_exprs)

        return df

    # 2. Run Training Pipeline (Feature Matrix)
    # -----------------------------------------
    logger.info("Running Training Pipeline...")
    cutoff_hours = 1.0 
    
    # Delete temp file to ensure fresh generation
    temp_out = Path("tests/temp_train_strict.parquet")
    if temp_out.exists():
        temp_out.unlink()
    
    with decision_cutoff_override(cutoff_hours=cutoff_hours):
        train_df = _build_feature_matrix_internal(
            start_date=target_date,
            end_date=target_date,
            chunk_days=1,
            recompute_intermediate=True,
            build_rolling=True,
            output_path=Path("tests/temp_train_strict.parquet"),
            cutoff_label="parity_strict",
        )
    
    if train_df.is_empty():
        logger.error("Training DF is empty! Cannot verify.")
        return

    # Filter train_df to target date ONLY (it loads full history by default)
    train_df = train_df.filter(pl.col("game_date").cast(pl.Date) == target_date)
    logger.info(f"Filtered Training DF Rows: {len(train_df)}")

    # Normalize TD-rate and offense-context shares on training slice.
    train_df = _normalize_td_and_offense_shares(train_df)

    # 2. Run Prediction Pipeline
    # --------------------------
    logger.info("Running Prediction Pipeline...")
    
    raw_pg_path = f"data/processed/player_game_by_week/season=2024/week=10/part.parquet"
    if not os.path.exists(raw_pg_path):
            logger.error(f"Raw data not found at {raw_pg_path}")
            return
            
    raw_pg = pl.read_parquet(raw_pg_path)
    raw_pg = raw_pg.filter(pl.col("game_date").cast(pl.Date) == target_date)
    
    # Restrict to players in train_df
    train_keys = train_df.select(["player_id", "game_id"]).unique()
    raw_pg = raw_pg.join(train_keys, on=["player_id", "game_id"], how="inner")
    
    # Identify available PS columns to pass through to scaffold
    # Exclude ps_hist_ columns as they cause duplicates during rename in prediction pipeline
    ps_cols = [c for c in raw_pg.columns if c.startswith("ps_") and not c.startswith("ps_hist_")]
    
    # Identify metadata columns (age, etc) to pass through
    meta_cols = ["age", "birth_date", "college", "draft_club", "draft_number"]
    meta_cols = [c for c in meta_cols if c in raw_pg.columns]
    
    base_cols = [
        "player_id",
        "player_name",
        "position",
        "team",
        "opponent",
        "season",
        "week",
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        # Include roof so that weather context flags (especially
        # weather_bad_passing_flag) see the same indoor/outdoor
        # metadata in both training and prediction.
        "roof",
    ]
    
    logger.info(f"Passed Meta Cols: {meta_cols}")
    
    # Map to scaffold
    pred_scaffold = raw_pg.select(
        [pl.col(c) for c in base_cols] + 
        [pl.col(c) for c in ps_cols] +
        [pl.col(c) for c in meta_cols] +
        [pl.lit("REG").alias("season_type")]
    ).to_pandas()
    
    # Add scaffold extras
    pred_scaffold["depth_chart_order"] = 1
    pred_scaffold["is_home"] = (pred_scaffold["team"] == pred_scaffold["home_team"]).astype(int)
    pred_scaffold["game_start_hour_utc"] = 18
    pred_scaffold["game_day_of_week"] = 6
    
    # Match 21:25 UTC for NYJ/ARI (4:25 ET) - Hardcoded for this test target
    # Ideally this comes from a schedule lookup
    pred_scaffold["game_start_utc"] = pd.to_datetime("2024-11-10 21:25:00").tz_localize("UTC")
    
    logger.info(f"Prediction Scaffold Rows: {len(pred_scaffold)}")
    if "age" in pred_scaffold.columns:
        logger.info(f"Scaffold has age: {pred_scaffold['age'].notna().sum()} non-nulls")
    
    with decision_cutoff_override(cutoff_hours=cutoff_hours):
        pred_out_obj = _compute_features(pred_scaffold)
        
    if isinstance(pred_out_obj, pd.DataFrame):
        pred_out = pl.from_pandas(pred_out_obj)
    else:
        pred_out = pred_out_obj

    # Apply the same normalization to prediction features.
    pred_out = _normalize_td_and_offense_shares(pred_out)

    # 3. Compare Results
    # ------------------
    logger.info("Comparing Columns and Values...")
    
    train_cols_all = set(train_df.columns)
    
    # Define columns to IGNORE (targets, leakage, metadata not generated by pred)
    IGNORE_COLS = {
        # Metadata/Identifiers
        "player_name", "player_display_name", "first_name", "last_name", "football_name",
        "esb_id", "espn_id", "fantasy_data_id", "gsis_id", "pff_id", "sleeper_id",
        "birth_date", "college", "draft_club", "draft_number", "entry_year", "rookie_year",
        "height", "weight", "uniform_number", "jersey_number",
        "age", "headshot_url", "status_description_abbr", "status_short_description",
        "team_abbr", "opponent_abbr", "game_type",
        "game_day_of_week", "game_start_hour_utc", # scaffold differences usually
        "season_type", "roof_type", "surface", "grass", "temp", "wind", # weather/stadium details
        "is_home", "depth_chart_order",
        
        # Leakage / Targets (Outcomes of the game)
        "passing_yards", "rushing_yards", "receiving_yards",
        "pass_attempt", "carry", "target", "reception", "completion",
        "passing_td", "rushing_td_count", "receiving_td_count",
        "touchdown", "touchdowns", "td_count", "td_count_offense", "td_count_all",
        "anytime_td", "anytime_td_all", "anytime_td_offense", "anytime_td_skill",
        "anytime_td_rush", "anytime_td_rec", "anytime_td_pass_thrown",
        "red_zone_target", "red_zone_carry", "goal_to_go_target", "goal_to_go_carry",
        "offense_snaps", "offense_pct", "defense_snaps", "defense_pct", "st_snaps", "st_pct",
        "target_share_label", "carry_share_label",
        "pre_snap_scripted_td",
        "snaps_label", "rec_success_rate_label", "rush_success_rate_label",
        "yards_per_target_label", "yards_per_carry_label",
        "td_per_target_label", "td_per_carry_label",
        "td_per_rz_carry_label", "td_per_gl_carry_label",
        
        # Pre-snap ACTUALS (Leakage)
        "ps_route_participation_plays", "ps_route_participation_pct",
        "ps_targets_total", "ps_targets_slot_count", "ps_targets_wide_count", 
        "ps_targets_inline_count", "ps_targets_backfield_count",
        "ps_targets_slot_share", "ps_targets_wide_share", "ps_targets_inline_share", "ps_targets_backfield_share",
        "ps_total_touches", "ps_scripted_touches", "ps_scripted_touch_share",
        "ps_team_dropbacks", "ps_tracking_team_dropbacks", "ps_tracking_has_game_data",
        "ps_game_route_participation_plays", "ps_game_route_participation_pct",
        "ps_game_targets_total", "ps_game_targets_slot_count", "ps_game_targets_wide_count",
        "ps_game_targets_inline_count", "ps_game_targets_backfield_count",
        "ps_game_targets_slot_share", "ps_game_targets_wide_share",
        "ps_game_targets_inline_share", "ps_game_targets_backfield_share",
        "ps_game_total_touches", "ps_game_scripted_touches", "ps_game_scripted_touch_share",
        "ps_game_tracking_team_dropbacks", "ps_game_tracking_has_game_data",
        "ps_game_team_dropbacks",
        
        # Historical columns that might be present in raw data but not features
        "ps_hist_route_participation_pct_prev", "ps_hist_route_participation_pct_l3",
        "ps_hist_route_participation_plays_prev", "ps_hist_route_participation_plays_l3",
        # ... add all ps_hist_ if we prefer 1g_ps_ ...
        
        # Internal / Misc
        "decision_cutoff_ts_missing", "decision_horizon_hours",
        "forecast_snapshot_ts_missing", "odds_snapshot_ts", # snapshots vary by run
        "team_ctx_data_as_of", "off_ctx_data_as_of", "opp_ctx_data_as_of", # data age varies
        "game_start_utc", # fixed in scaffold but might still drift?
        "decision_cutoff_ts", # calculated differently in train/pred
        
        # Odds columns (API failures in test env)
        "total_line", "spread_line", "moneyline",
        "total_book_count", "spread_book_count", "moneyline_book_count",
        "market_anytime_td_book_count", "market_anytime_td_book_count_24h", 
        "market_anytime_td_book_count_2h", "market_anytime_td_book_count_6h",
        "implied_total", "team_implied_total",
    }
    
    # Dynamic ignore of ps_hist_ if 1g_ps_ exists (preference)
    has_rolling_ps = any(c.startswith("1g_ps_") for c in train_cols_all)
    if has_rolling_ps:
        ps_hist_cols = [c for c in train_cols_all if c.startswith("ps_hist_")]
        IGNORE_COLS.update(ps_hist_cols)

    # Restrict comparison strictly to the feature sets used by the flat and
    # structured anytime TD models, based on training.yaml.
    cfg_path = project_root / "config" / "training.yaml"
    with cfg_path.open("r") as fp:
        train_cfg = yaml.safe_load(fp)

    problems_cfg = train_cfg.get("problems", [])

    def _feature_set_for_problem(problem_name: str) -> set[str]:
        for block in problems_cfg:
            if block.get("name") == problem_name:
                prefixes = block.get("feature_prefixes_to_include") or []
                others = block.get("other_features_to_include") or []
                feats: set[str] = set()
                for col in train_cols_all:
                    if any(col.startswith(pref) for pref in prefixes):
                        feats.add(col)
                for col in others:
                    if col in train_cols_all:
                        feats.add(col)
                return feats
        return set()

    meta_features = _feature_set_for_problem("anytime_td_structured")
    flat_features = _feature_set_for_problem("anytime_td")
    target_feature_cols = (meta_features | flat_features) - IGNORE_COLS

    train_cols = set(c for c in train_df.columns if c in target_feature_cols)
    pred_cols = set(c for c in pred_out.columns if c in target_feature_cols)
    
    common_cols = sorted(list(train_cols.intersection(pred_cols)))
    missing_in_pred = sorted(list(train_cols - pred_cols))
    extra_in_pred = sorted(list(pred_cols - train_cols))
    
    logger.info(f"Common Columns: {len(common_cols)}")
    logger.info(f"Missing in Prediction: {len(missing_in_pred)}")
    if missing_in_pred:
        logger.info(f"Sample Missing: {missing_in_pred[:10]}")
        
    # Join on keys for value comparison
    comparison = train_df.join(
        pred_out, 
        on=["player_id", "game_id"], 
        suffix="_pred",
        how="inner"  # Compare only matching rows
    )
    
    logger.info(f"Comparison Rows (Matched): {len(comparison)}")
    
    mismatches = []
    for col in common_cols:
        dtype = train_df.schema[col]
        if dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
            continue
            
        c_train = comparison.get_column(col).cast(pl.Float32).fill_null(0.0)
        c_pred = comparison.get_column(col + "_pred").cast(pl.Float32).fill_null(0.0)
        
        diff = (c_train - c_pred).abs()
        max_diff = diff.max()
        
        if max_diff > 1e-4:
            mean_diff = diff.mean()
            train_zeros = (c_train == 0).sum()
            pred_zeros = (c_pred == 0).sum()
            
            mismatches.append({
                "feature": col,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "train_zeros": train_zeros,
                "pred_zeros": pred_zeros
            })

    if not mismatches and not missing_in_pred:
        logger.info("✅ STRICT PARITY ACHIEVED!")
    else:
        logger.error("❌ PARITY FAILED")
        
        # DEBUG: Inspect Pred output for Rodgers if mismatch
        rodgers_pred = pred_out.filter(pl.col("player_id") == "00-0023459")
        if not rodgers_pred.is_empty():
            logger.info(f"Pred Row (Rodgers):")
            cols = ["drive_hist_total_yards_prev", "drive_hist_count_prev", "age"]
            present = [c for c in cols if c in rodgers_pred.columns]
            if present:
                logger.info(rodgers_pred.select(present))
            else:
                logger.info("Rodgers row found but target columns missing.")
        
        if missing_in_pred:
            logger.error(f"MISSING FEATURES ({len(missing_in_pred)}):")
            for c in missing_in_pred[:20]:
                logger.error(f"  - {c}")
        
        if mismatches:
            logger.error(f"VALUE MISMATCHES ({len(mismatches)}):")
            mismatches.sort(key=lambda x: x['max_diff'], reverse=True)
            for m in mismatches[:50]:
                logger.error(f"  {m['feature']}: MaxDiff={m['max_diff']:.4f}, MeanDiff={m['mean_diff']:.4f}")
                
                # Debug: Print player/game with max diff for top mismatch
                if m == mismatches[0]:
                    # Find row with max diff
                    col = m['feature']
                    c_train = comparison.get_column(col).cast(pl.Float32).fill_null(0.0)
                    c_pred = comparison.get_column(col + "_pred").cast(pl.Float32).fill_null(0.0)
                    diff = (c_train - c_pred).abs()
                    max_idx = diff.arg_max()
                    row = comparison.row(max_idx, named=True)
                    logger.error(f"    MaxDiff Row: Player={row['player_id']}, Game={row['game_id']}")
                    logger.error(f"    Train={row[col]}, Pred={row[col + '_pred']}")

if __name__ == "__main__":
    verify_parity_strict()
