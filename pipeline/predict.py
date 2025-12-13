"""NFL prediction pipeline orchestrator.

This module generates anytime-touchdown probabilities for upcoming games by:

1. Loading the upcoming schedule and weekly rosters (via nfl_data_py)
2. Building a player-level scaffold for offensive skill positions
3. Enriching that scaffold with features using the SAME unified feature builder
   that powers the training pipeline - ensuring strict parity
4. Sequentially running the model chain:
   Availability -> Team Pace -> Usage -> Efficiency -> Anytime TD
5. Writing an actionable slate under `output/predictions/`

The actual feature building, model inference, and output formatting logic
is in utils/predict/ modules. This file is just the orchestrator.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.predict import (
    # Scaffold
    ensure_imports_ready,
    load_schedule,
    load_rosters,
    build_scaffold,
    # Features
    build_features,
    # Injuries
    ensure_injury_cache,
    attach_injury_features,
    # Inference
    load_artifacts,
    load_threshold,
    check_moe_available,
    predict_moe,
    predict_global,
    apply_guards_inline,
    apply_availability_floor,
    apply_snaps_ceiling_cap,
    apply_usage_targets_position_cap,
    apply_usage_carries_position_cap,
    apply_usage_target_yards_position_cap,
    apply_efficiency_rec_yards_air_cap,
    # Output
    inject_composed_features,
    format_output,
    save_predictions,
)
from utils.feature.enrichment.asof import (
    decision_cutoff_override,
    get_decision_cutoff_hours,
)
from utils.feature.enrichment.odds import collect_odds_snapshots
from utils.collect.weather_forecasts import collect_weather_forecasts
from utils.general.paths import PROJ_ROOT

logger = logging.getLogger(__name__)

CONFIG_PATH = PROJ_ROOT / "config/training.yaml"
PREDICTION_DIR = Path("output/predictions")
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def _load_config() -> dict:
    """Load training configuration."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NFL anytime touchdown prediction pipeline.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Prediction slate start date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of consecutive days (from --date) to include. Default: 1.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output filename. Defaults to predictions_<start>_<end>.csv",
    )
    parser.add_argument(
        "--decision-cutoff-hours",
        type=float,
        default=None,
        help="Override decision cutoff window (hours before kickoff).",
    )
    parser.add_argument(
        "--fallback-cutoff-hours",
        type=float,
        default=None,
        help="Override fallback cutoff when kickoff time is unavailable.",
    )
    return parser.parse_args()


def _resolve_dates(start_str: str | None, days: int) -> tuple[date, date]:
    """Resolve start and end dates from CLI args."""
    start = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else date.today()
    end = start + timedelta(days=max(days, 1) - 1)
    return start, end


def _ensure_weather_cache(seasons: list[int]) -> None:
    """Ensure weather forecast caches are fresh."""
    import datetime as dt
    
    today = dt.date.today()
    # NFL seasons span calendar years (e.g. 2025 season includes Jan/Feb 2026 games).
    nfl_season_year = today.year if today.month >= 3 else (today.year - 1)
    active_seasons = [int(s) for s in seasons if int(s) >= nfl_season_year]
    
    if not active_seasons:
        return
        
    try:
        logger.info("Ensuring weather forecasts for seasons %s", active_seasons)
        collect_weather_forecasts(active_seasons, force_refresh=False)
    except Exception as exc:
        logger.warning("Failed to ensure weather cache: %s", exc)


def _ensure_odds_cache(seasons: list[int], start_date=None, end_date=None) -> None:
    """Ensure odds snapshot caches are fresh."""
    import datetime as dt
    
    today = dt.date.today()
    nfl_season_year = today.year if today.month >= 3 else (today.year - 1)
    active_seasons = [int(s) for s in seasons if int(s) >= nfl_season_year]
    
    if not active_seasons:
        return

    try:
        logger.info("Ensuring odds snapshots for seasons %s (window: %s to %s)", 
                   active_seasons, start_date, end_date)
        collect_odds_snapshots(
            active_seasons,
            start_date=start_date,
            end_date=end_date,
            include_player_props=True,
        )
    except Exception as exc:
        logger.warning("Failed to ensure odds cache: %s", exc)


def _run_model_chain(features: pd.DataFrame, problems: list[dict]) -> pd.DataFrame:
    """Run the sequential model prediction chain.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature dataframe
    problems : list[dict]
        Problem configurations from training.yaml
    
    Returns
    -------
    pd.DataFrame
        Features with prediction columns added
    """
    for problem in problems:
        p_name = problem["name"]
        pred_col = f"pred_{p_name}"

        logger.info("Predicting %s -> %s", p_name, pred_col)

        try:
            artifacts = load_artifacts(p_name)
        except FileNotFoundError:
            logger.warning("Skipping %s (artifacts not found).", p_name)
            continue

        # IMPORTANT: For stacked parity, always materialize GLOBAL predictions as the
        # canonical `pred_{problem}` columns (these are what training artifacts use).
        #
        # If MoE artifacts exist, also compute `pred_{problem}_moe` for debugging /
        # comparison, but do not let it silently replace the canonical stack inputs.
        global_preds = predict_global(features, problem, artifacts)
        global_preds = _apply_problem_postprocessing(features, global_preds, p_name)
        features[f"pred_{p_name}_global"] = global_preds

        use_moe = check_moe_available(p_name)
        if use_moe:
            logger.info("Also running MoE models for %s", p_name)
            moe_preds = predict_moe(features, p_name)
            moe_preds = _apply_problem_postprocessing(features, moe_preds, p_name)
            features[f"pred_{p_name}_moe"] = moe_preds

        # Default stack input: GLOBAL preds
        features[pred_col] = global_preds

        # Debug: Print prediction stats
        print(f"--- {pred_col} stats ---")
        print(features[pred_col].describe().T[["mean", "min", "50%", "max"]])
        print("------------------------")

        # Inject composed features after efficiency_tds
        if p_name == "efficiency_tds":
            features = inject_composed_features(features)

    # Ensure composed features are present after the full chain
    features = inject_composed_features(features)
    
    return features


def _apply_problem_postprocessing(
    features: pd.DataFrame,
    preds: np.ndarray,
    p_name: str,
) -> np.ndarray:
    """Apply problem-specific post-processing guards."""
    if p_name == "availability_active":
        features["pred_availability_active_raw"] = preds.copy()
        preds = apply_guards_inline(features, preds)
        preds = apply_availability_floor(features, preds)
        preds = np.clip(preds, 0.0, 1.0)
    elif p_name == "availability_snapshare":
        preds = np.clip(preds, 0.0, 1.0)
    elif p_name == "pre_snap_routes":
        preds = np.clip(preds, 0.0, 1.0)
    elif p_name == "pre_snap_scripted_touches":
        preds = np.clip(preds, 0.0, None)
    elif p_name == "snaps":
        preds = apply_snaps_ceiling_cap(features, preds)
    elif p_name == "usage_targets":
        preds = apply_usage_targets_position_cap(features, preds)
    elif p_name == "usage_carries":
        preds = apply_usage_carries_position_cap(features, preds)
    elif p_name == "usage_target_yards":
        preds = apply_usage_target_yards_position_cap(features, preds)
    elif p_name == "efficiency_rec_yards_air":
        preds = apply_efficiency_rec_yards_air_cap(features, preds)

    return preds


def main() -> None:
    """Main entry point for prediction pipeline."""
    ensure_imports_ready()
    args = _parse_args()
    start_date, end_date = _resolve_dates(args.date, args.days)
    config = _load_config()
    problems = config.get("problems", [])

    with decision_cutoff_override(
        cutoff_hours=args.decision_cutoff_hours,
        fallback_hours=args.fallback_cutoff_hours,
    ):
        active_cutoff = get_decision_cutoff_hours()
        print(
            f"Generating predictions for {start_date} → {end_date} "
            f"(cutoff {active_cutoff:.2f}h before kickoff)"
        )
        
        # Step 1: Load schedule and roster
        logger.info("Loading schedule and roster...")
        games = load_schedule(start_date, end_date)
        seasons = list(games["season"].unique())
        roster = load_rosters(seasons)
        
        # Step 2: Build scaffold
        logger.info("Building player-game scaffold...")
        scaffold = build_scaffold(games, roster)
        
        # Step 3: Ensure caches are fresh
        logger.info("Ensuring data caches are fresh...")
        ensure_injury_cache(seasons)
        _ensure_weather_cache(seasons)
        _ensure_odds_cache(seasons, start_date=start_date, end_date=end_date)
        
        # Step 4: Build features using UNIFIED builder (same as training)
        logger.info("Building features (using unified builder for parity)...")
        pl_df = pl.from_pandas(scaffold)
        pl_df = build_features(pl_df, is_inference=True, seasons=seasons)
        
        # Step 5: Attach injury features
        logger.info("Attaching injury features...")
        pl_df = attach_injury_features(pl_df)
        
        # Convert to pandas for model inference
        features = pl_df.to_pandas()
        
        # Normalize types for model compatibility
        features["game_date"] = pd.to_datetime(features["game_date"])
        features["season"] = features["season"].astype(int)
        features["week"] = features["week"].astype(int)
        if "depth_chart_order" in features.columns:
            features["depth_chart_order"] = pd.to_numeric(
                features["depth_chart_order"], errors="coerce"
            )
        
        # Step 6: Run model chain
        logger.info("Running model prediction chain...")
        features = _run_model_chain(features, problems)
        
        # Step 7: Format and save output
        logger.info("Formatting output...")
        final_prob_col = "pred_anytime_td_structured"
        if final_prob_col not in features.columns:
            if "pred_anytime_td" in features.columns:
                logger.warning(
                    "pred_anytime_td_structured not found; using pred_anytime_td."
                )
                final_prob_col = "pred_anytime_td"
            else:
                logger.error("Final prediction column not found.")
                return

        # Use the threshold that matches the final probability column.
        # If we're outputting the structured calibrator, use its calibrated threshold.
        threshold_problem = (
            "anytime_td_structured"
            if final_prob_col == "pred_anytime_td_structured"
            else "anytime_td"
        )
        threshold = load_threshold(threshold_problem)
        output_df = format_output(features, final_prob_col, threshold)
        
        output_path = save_predictions(
            output_df, start_date, end_date, output_name=args.output
        )
        
        # Debug: Show top 5
        print("\nTop 5 Predictions:")
        print(
            output_df[
                [
                    "player_name",
                    "team",
                    "prob_anytime_td_flat",
                    "prob_anytime_td_global",
                    "prob_anytime_td_moe",
                    "implied_decimal_odds",
                ]
            ].head(5)
        )


if __name__ == "__main__":
    main()
