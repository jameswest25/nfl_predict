"""Mixture of Experts training script.

This script trains per-position models for specified tasks, replacing global models.
It can be run after the main training pipeline to convert tasks to per-position.

Usage:
    python pipeline/train_moe.py --tasks usage_targets
    python pipeline/train_moe.py --tasks usage_targets,usage_carries,snaps
    python pipeline/train_moe.py --all  # Train all MoE-enabled tasks
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import yaml
import joblib
from xgboost import XGBRegressor, XGBClassifier

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.train.data import load_feature_matrix, split_data_chronologically
from utils.train.feature_artifacts import (
    FeatureArtifacts,
    fit_feature_artifacts,
    apply_feature_artifacts,
)
from utils.train.moe import (
    POSITION_GROUPS,
    MIN_ROWS_PER_POSITION,
    DEFAULT_FALLBACK_POSITION,
    save_position_model,
    compare_global_vs_moe,
    PerPositionPredictor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/training.yaml") -> dict:
    """Load the training configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_moe_tasks(config: dict) -> List[str]:
    """Get list of tasks with per_position_training enabled."""
    moe_tasks = []
    for problem in config.get("problems", []):
        moe_cfg = problem.get("per_position_training", {})
        if moe_cfg.get("enabled", False):
            moe_tasks.append(problem["name"])
    return moe_tasks


def get_problem_config(config: dict, task_name: str) -> dict:
    """Get configuration for a specific problem."""
    for problem in config.get("problems", []):
        if problem["name"] == task_name:
            return problem
    raise ValueError(f"Problem {task_name} not found in config")


def build_feature_list(config: dict, problem_config: dict, df: pd.DataFrame) -> List[str]:
    """Build the feature list for a problem based on config prefixes and explicit features."""
    prefixes = problem_config.get("feature_prefixes_to_include", [])
    explicit = problem_config.get("other_features_to_include", [])
    discard = set(problem_config.get("columns_to_discard", []))
    discard.update(config.get("columns_to_discard", []))
    
    # Columns to always exclude (metadata, non-numeric)
    always_exclude = {
        "position_group", "player_id", "player_name", "game_id", "game_date",
        "home_team", "away_team", "team", "opponent",
        # Timestamp/datetime columns
        "injury_snapshot_ts", "off_ctx_data_as_of", "off_ctx_game_date",
        "opp_ctx_data_as_of", "team_ctx_data_as_of",
    }
    
    # Add position_group for routing (but not as a feature)
    available_cols = set(df.columns)
    
    features = []
    for col in df.columns:
        # Skip always-excluded columns
        if col in always_exclude:
            continue
        # Check if matches any prefix
        for prefix in prefixes:
            if col.startswith(prefix):
                features.append(col)
                break
    
    # Add explicit features
    for col in explicit:
        if col in available_cols and col not in features and col not in always_exclude:
            features.append(col)
    
    # Remove discarded columns and target
    target_col = problem_config["target_col"]
    features = [f for f in features if f not in discard and f != target_col]
    
    # Remove any datetime columns by checking dtype names
    datetime_cols = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if 'datetime' in dtype_str.lower():
            datetime_cols.append(col)
    features = [f for f in features if f not in datetime_cols]
    
    return sorted(set(features))


def prepare_features_for_xgb(X: pd.DataFrame) -> pd.DataFrame:
    """Prepare a feature DataFrame for XGBoost by converting non-numeric types."""
    X_numeric = X.copy()
    
    # Drop datetime columns by checking dtype names
    datetime_cols = [col for col in X_numeric.columns if 'datetime' in str(X_numeric[col].dtype).lower()]
    if datetime_cols:
        X_numeric = X_numeric.drop(columns=datetime_cols, errors='ignore')
    
    # Convert object/category columns to numeric
    for col in X_numeric.columns:
        if X_numeric[col].dtype == 'object':
            try:
                X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
            except:
                X_numeric[col] = pd.Categorical(X_numeric[col]).codes
        elif str(X_numeric[col].dtype) == 'category':
            X_numeric[col] = X_numeric[col].cat.codes
    
    # Fill NaN and ensure numeric
    X_numeric = X_numeric.fillna(0)
    for col in X_numeric.columns:
        if not pd.api.types.is_numeric_dtype(X_numeric[col]):
            X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce').fillna(0)
    
    return X_numeric


def train_xgb_model(X: pd.DataFrame, y: pd.Series, task_type: str = "regression", 
                    random_state: int = 42) -> Any:
    """Train an XGBoost model with default hyperparameters."""
    X_numeric = prepare_features_for_xgb(X)
    
    if task_type == "classification":
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    else:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
    
    model.fit(X_numeric, y)
    return model


def train_moe_for_task(
    task_name: str,
    config: dict,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    model_dir: Path,
    run_shadow_comparison: bool = True,
) -> Dict[str, Any]:
    """Train per-position models for a single task.
    
    Args:
        task_name: Name of the task to train
        config: Full training configuration
        df_train: Training DataFrame
        df_test: Test DataFrame
        model_dir: Base directory for saving models
        run_shadow_comparison: Whether to compare against global model
        
    Returns:
        Dictionary with trained models and metrics
    """
    problem_config = get_problem_config(config, task_name)
    moe_cfg = problem_config.get("per_position_training", {})
    
    target_col = problem_config["target_col"]
    task_type = problem_config.get("task_type", "regression")
    positions = moe_cfg.get("positions", POSITION_GROUPS)
    min_rows = moe_cfg.get("min_rows", MIN_ROWS_PER_POSITION)
    fallback_pos = moe_cfg.get("fallback_position", DEFAULT_FALLBACK_POSITION)
    
    # Build feature list
    feature_list = build_feature_list(config, problem_config, df_train)
    logger.info(f"Training {task_name} with {len(feature_list)} features")
    
    # Check for required columns
    if "position_group" not in df_train.columns:
        raise ValueError("df_train must have 'position_group' column. Run feature pipeline first.")
    
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in df_train")
    
    # Filter to rows with valid targets
    train_valid = df_train[df_train[target_col].notna()].copy()
    test_valid = df_test[df_test[target_col].notna()].copy()
    
    logger.info(f"Training data: {len(train_valid)} rows, Test data: {len(test_valid)} rows")
    
    # Train per-position models
    models = {}
    position_metrics = {}
    
    for pos in positions:
        df_pos = train_valid[train_valid["position_group"] == pos]
        
        if len(df_pos) < min_rows:
            logger.warning(f"Skipping {task_name}.{pos}: only {len(df_pos)} rows (min: {min_rows})")
            continue
        
        logger.info(f"Training {task_name}.{pos} with {len(df_pos)} rows...")
        
        # Get features and target
        X_pos = df_pos[feature_list].copy()
        y_pos = df_pos[target_col]
        
        # Train model
        model = train_xgb_model(X_pos, y_pos, task_type)
        models[pos] = model
        
        # Save model
        save_position_model(model, model_dir, task_name, pos, "xgboost")
        
        # Evaluate on position-specific test data
        df_test_pos = test_valid[test_valid["position_group"] == pos]
        if len(df_test_pos) > 0:
            X_test_pos = prepare_features_for_xgb(df_test_pos[feature_list].copy())
            
            y_test_pos = df_test_pos[target_col]
            preds_pos = model.predict(X_test_pos)
            
            mae = mean_absolute_error(y_test_pos, preds_pos)
            r2 = r2_score(y_test_pos, preds_pos) if len(np.unique(y_test_pos)) > 1 else 0.0
            
            position_metrics[pos] = {
                "n_train": len(df_pos),
                "n_test": len(df_test_pos),
                "mae": mae,
                "r2": r2,
            }
            logger.info(f"  {pos} test metrics: MAE={mae:.4f}, R²={r2:.4f}")
    
    if not models:
        raise ValueError(f"Failed to train any position models for {task_name}")
    
    # Save feature artifacts for MoE inference (before shadow comparison which may fail)
    artifacts_path = model_dir / f"inference_artifacts_{task_name}_moe.joblib"
    artifacts = {
        "feature_columns": feature_list,
        "positions": list(models.keys()),
        "fallback_position": fallback_pos,
        "task_type": task_type,
    }
    joblib.dump(artifacts, artifacts_path)
    logger.info(f"Saved MoE artifacts to {artifacts_path}")
    
    # Shadow comparison with global model if requested
    shadow_results = None
    if run_shadow_comparison:
        # Find latest global model (stored in timestamped subdirectories)
        model_base = model_dir / task_name / "xgboost"
        global_model_path = None
        
        # Look for timestamped subdirectories
        try:
            subdirs = sorted([d for d in model_base.iterdir() if d.is_dir()], reverse=True)
        except FileNotFoundError:
            subdirs = []
        for subdir in subdirs:
            candidate = subdir / "model.joblib"
            if candidate.exists():
                global_model_path = candidate
                break
        
        if global_model_path and global_model_path.exists():
            try:
                logger.info("Running shadow comparison with global model...")
                global_model = joblib.load(global_model_path)
                
                # Prepare test data
                X_test = prepare_features_for_xgb(test_valid[feature_list].copy())
                
                y_test = test_valid[target_col].values
                position_groups = test_valid["position_group"]
                
                # Global predictions
                global_preds = global_model.predict(X_test)
                
                # MoE predictions
                moe_predictor = PerPositionPredictor(task_name, models, fallback_pos)
                moe_preds = moe_predictor.predict(X_test, position_groups)
                
                # Compare
                shadow_results = compare_global_vs_moe(
                    global_preds, moe_preds, y_test, position_groups, task_name
                )
            except Exception as e:
                logger.warning(f"Shadow comparison failed: {e}")
                logger.info("MoE models trained successfully, but shadow comparison could not be completed.")
        else:
            logger.warning(f"Global model not found at {global_model_path}, skipping shadow comparison")
    
    return {
        "models": models,
        "position_metrics": position_metrics,
        "shadow_results": shadow_results,
        "artifacts": artifacts,
    }


def main():
    parser = argparse.ArgumentParser(description="Train per-position MoE models")
    parser.add_argument("--tasks", type=str, help="Comma-separated list of tasks to train")
    parser.add_argument("--all", action="store_true", help="Train all MoE-enabled tasks")
    parser.add_argument("--config", type=str, default="config/training.yaml", help="Config path")
    parser.add_argument("--no-shadow", action="store_true", help="Skip shadow comparison")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Determine which tasks to train
    if args.all:
        tasks = get_moe_tasks(config)
        if not tasks:
            logger.error("No tasks with per_position_training enabled in config")
            return 1
    elif args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        logger.error("Must specify --tasks or --all")
        return 1
    
    logger.info(f"Training MoE models for tasks: {tasks}")
    
    # Load data
    feature_matrix_path = Path(config["data"]["feature_matrix_path"])
    
    # Check for cutoff-labeled version
    from utils.feature.asof import get_decision_cutoff_hours
    from utils.general.constants import format_cutoff_label
    
    active_cutoff = float(get_decision_cutoff_hours())
    cutoff_label = format_cutoff_label(active_cutoff)
    labeled_path = feature_matrix_path.with_name(
        f"{feature_matrix_path.stem}_{cutoff_label}{feature_matrix_path.suffix}"
    )
    if labeled_path.exists():
        feature_matrix_path = labeled_path
        logger.info(f"Using cutoff-labeled feature matrix: {feature_matrix_path}")
    
    logger.info(f"Loading feature matrix from {feature_matrix_path}")
    df = load_feature_matrix(str(feature_matrix_path), time_col="game_date")
    
    # Split data
    split_cfg = config.get("data_split", {})
    train_df, val_df, test_df = split_data_chronologically(
        df, time_col="game_date", split_cfg=split_cfg, production_mode=False
    )
    
    # Combine train + val for final training
    df_train = pd.concat([train_df, val_df], ignore_index=True)
    df_test = test_df
    
    logger.info(f"Train: {len(df_train)} rows, Test: {len(df_test)} rows")
    
    # Model output directory
    model_dir = Path(config["data"]["model_output_dir"])
    
    # Train each task
    results = {}
    for task_name in tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training MoE for: {task_name}")
        logger.info(f"{'='*60}\n")
        
        try:
            result = train_moe_for_task(
                task_name=task_name,
                config=config,
                df_train=df_train,
                df_test=df_test,
                model_dir=model_dir,
                run_shadow_comparison=not args.no_shadow,
            )
            results[task_name] = result
            logger.info(f"✅ {task_name} MoE training complete")
        except Exception as e:
            logger.error(f"❌ Failed to train {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("MoE TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    
    for task_name, result in results.items():
        logger.info(f"\n{task_name}:")
        logger.info(f"  Positions trained: {list(result['models'].keys())}")
        for pos, metrics in result.get("position_metrics", {}).items():
            logger.info(f"    {pos}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
    
    logger.info(f"\n{'='*60}")
    logger.info("MoE training complete!")
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

