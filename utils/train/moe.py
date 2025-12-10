"""Mixture of Experts (MoE) utilities for per-position model training and routing.

This module provides utilities for training separate models per position group
(RB, WR, TE, QB) and routing predictions based on position.

Design Goals:
- Replace one global model with one model per position
- Keep the same input features and target per task
- Keep the same output shape/column names for downstream models
- Initially retain capping logic (as safety rail), easy to remove later
- Add shadow comparison to verify no performance regression
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

# Position groups for MoE routing
POSITION_GROUPS = ["RB", "WR", "TE", "QB"]

# Minimum rows required to train a position-specific model
MIN_ROWS_PER_POSITION = 500

# Default fallback position when a model is missing
DEFAULT_FALLBACK_POSITION = "WR"


def normalize_position_group(position: str) -> str:
    """Normalize a position to one of the four position groups.
    
    Args:
        position: Raw position string (e.g., "HB", "FB", "WR", "QB")
        
    Returns:
        Normalized position group: "RB", "WR", "TE", or "QB"
    """
    raw = (position or "").upper()
    if raw in {"RB", "HB", "FB"}:
        return "RB"
    if raw in {"WR"}:
        return "WR"
    if raw in {"TE"}:
        return "TE"
    if raw in {"QB"}:
        return "QB"
    # Default: treat unknown like WR
    return "WR"


def get_position_model_path(base_dir: Path, task_name: str, position: str, model_type: str = "xgboost") -> Path:
    """Get the path for a position-specific model.
    
    Args:
        base_dir: Base model directory
        task_name: Name of the task (e.g., "usage_targets")
        position: Position group (RB, WR, TE, QB)
        model_type: Model type (e.g., "xgboost")
        
    Returns:
        Path to the model file
    """
    return base_dir / task_name / model_type / f"model.{position}.joblib"


def save_position_model(model: Any, base_dir: Path, task_name: str, position: str, 
                        model_type: str = "xgboost") -> Path:
    """Save a position-specific model.
    
    Args:
        model: Trained model to save
        base_dir: Base model directory
        task_name: Name of the task
        position: Position group
        model_type: Model type
        
    Returns:
        Path where model was saved
    """
    model_path = get_position_model_path(base_dir, task_name, position, model_type)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Saved {task_name}.{position} model to {model_path}")
    return model_path


def load_position_models(base_dir: Path, task_name: str, model_type: str = "xgboost",
                         fallback_pos: str = DEFAULT_FALLBACK_POSITION) -> Dict[str, Any]:
    """Load all position-specific models for a task.
    
    Args:
        base_dir: Base model directory
        task_name: Name of the task
        model_type: Model type
        fallback_pos: Default position to use if a model is missing
        
    Returns:
        Dictionary mapping position -> model
    """
    models = {}
    for pos in POSITION_GROUPS:
        model_path = get_position_model_path(base_dir, task_name, pos, model_type)
        if model_path.exists():
            models[pos] = joblib.load(model_path)
            logger.info(f"Loaded {task_name}.{pos} model from {model_path}")
        else:
            logger.warning(f"No model found for {task_name}.{pos} at {model_path}")
    
    if not models:
        raise ValueError(f"No position models found for {task_name}")
    
    return models


def train_per_position_models(
    task_name: str,
    train_df: pd.DataFrame,
    train_fn: Callable[[pd.DataFrame, pd.Series], Any],
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    target_col: str,
    positions: List[str] = None,
    min_rows: int = MIN_ROWS_PER_POSITION,
    base_dir: Path = None,
    model_type: str = "xgboost",
) -> Dict[str, Any]:
    """Train separate models for each position group.
    
    Args:
        task_name: Name of the task
        train_df: Training DataFrame (must have 'position_group' column)
        train_fn: Function that takes (X, y) and returns a trained model
        feature_fn: Function that takes DataFrame and returns feature matrix X
        target_col: Name of the target column
        positions: List of positions to train (default: all)
        min_rows: Minimum rows required to train a model
        base_dir: Base directory to save models
        model_type: Model type string
        
    Returns:
        Dictionary mapping position -> trained model
    """
    positions = positions or POSITION_GROUPS
    models = {}
    
    if "position_group" not in train_df.columns:
        raise ValueError("train_df must have 'position_group' column")
    
    for pos in positions:
        df_pos = train_df[train_df["position_group"] == pos]
        
        if len(df_pos) < min_rows:
            logger.warning(f"Skipping {task_name}.{pos}: only {len(df_pos)} rows (min: {min_rows})")
            continue
        
        logger.info(f"Training {task_name}.{pos} with {len(df_pos)} rows...")
        
        X = feature_fn(df_pos)
        y = df_pos[target_col]
        
        model = train_fn(X, y)
        models[pos] = model
        
        if base_dir:
            save_position_model(model, base_dir, task_name, pos, model_type)
    
    if not models:
        raise ValueError(f"Failed to train any position models for {task_name}")
    
    return models


class PerPositionPredictor:
    """Wrapper for making predictions using position-specific models.
    
    This class routes each row to the appropriate position-specific model
    and produces predictions in the same format as a single global model.
    
    Attributes:
        task_name: Name of the prediction task
        models: Dictionary mapping position -> model
        fallback_pos: Position to use when model is missing
        cap_fn: Optional capping function to apply after predictions
    """
    
    def __init__(
        self,
        task_name: str,
        models: Dict[str, Any],
        fallback_pos: str = DEFAULT_FALLBACK_POSITION,
        cap_fn: Optional[Callable[[np.ndarray, pd.DataFrame], np.ndarray]] = None,
    ):
        """Initialize the predictor.
        
        Args:
            task_name: Name of the task
            models: Dictionary mapping position -> model
            fallback_pos: Default position for missing models
            cap_fn: Optional function(preds, df) -> capped_preds
        """
        self.task_name = task_name
        self.models = models
        self.fallback_pos = fallback_pos
        self.cap_fn = cap_fn
        
        if not models:
            raise ValueError("Must provide at least one model")
        if fallback_pos not in models:
            # Find first available position
            self.fallback_pos = next(iter(models.keys()))
            logger.warning(f"Fallback position {fallback_pos} not in models, using {self.fallback_pos}")
    
    def predict(self, X: pd.DataFrame, position_groups: pd.Series) -> np.ndarray:
        """Make predictions for all rows, routing by position.
        
        Args:
            X: Feature matrix
            position_groups: Series of position groups for each row
            
        Returns:
            Array of predictions
        """
        preds = np.zeros(len(X), dtype=np.float64)
        
        for pos in self.models.keys():
            mask = (position_groups == pos).values
            if mask.sum() == 0:
                continue
            
            X_pos = X.loc[mask]
            model = self.models[pos]
            
            # Handle different model types
            if hasattr(model, "predict_proba"):
                pos_preds = model.predict_proba(X_pos)
                if pos_preds.ndim > 1:
                    pos_preds = pos_preds[:, 1]
            else:
                pos_preds = model.predict(X_pos)
            
            preds[mask] = pos_preds
        
        # Handle rows with positions not in models (use fallback)
        known_positions = set(self.models.keys())
        fallback_mask = ~position_groups.isin(known_positions).values
        if fallback_mask.sum() > 0:
            X_fallback = X.loc[fallback_mask]
            fallback_model = self.models[self.fallback_pos]
            
            if hasattr(fallback_model, "predict_proba"):
                fallback_preds = fallback_model.predict_proba(X_fallback)
                if fallback_preds.ndim > 1:
                    fallback_preds = fallback_preds[:, 1]
            else:
                fallback_preds = fallback_model.predict(X_fallback)
            
            preds[fallback_mask] = fallback_preds
            logger.info(f"Used fallback model ({self.fallback_pos}) for {fallback_mask.sum()} rows")
        
        return preds
    
    def predict_with_cap(self, X: pd.DataFrame, df_full: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Make predictions with optional capping, returning both raw and capped.
        
        Args:
            X: Feature matrix
            df_full: Full DataFrame with position_group and other columns for capping
            
        Returns:
            Tuple of (raw_predictions, capped_predictions)
        """
        if "position_group" not in df_full.columns:
            raise ValueError("df_full must have 'position_group' column")
        
        position_groups = df_full["position_group"]
        raw_preds = self.predict(X, position_groups)
        
        if self.cap_fn is not None:
            capped_preds = self.cap_fn(raw_preds, df_full)
        else:
            capped_preds = raw_preds.copy()
        
        return raw_preds, capped_preds


def compare_global_vs_moe(
    global_preds: np.ndarray,
    moe_preds: np.ndarray,
    actual: np.ndarray,
    position_groups: pd.Series,
    task_name: str,
) -> Dict[str, Dict[str, float]]:
    """Compare global model predictions vs MoE predictions.
    
    Args:
        global_preds: Predictions from global model
        moe_preds: Predictions from MoE (per-position) models
        actual: Actual target values
        position_groups: Position group for each row
        task_name: Name of the task for logging
        
    Returns:
        Dictionary with metrics by position and overall
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    results = {}
    
    # Overall metrics
    results["overall"] = {
        "global_mae": mean_absolute_error(actual, global_preds),
        "moe_mae": mean_absolute_error(actual, moe_preds),
        "global_rmse": np.sqrt(mean_squared_error(actual, global_preds)),
        "moe_rmse": np.sqrt(mean_squared_error(actual, moe_preds)),
        "global_r2": r2_score(actual, global_preds) if len(np.unique(actual)) > 1 else 0.0,
        "moe_r2": r2_score(actual, moe_preds) if len(np.unique(actual)) > 1 else 0.0,
    }
    
    # Per-position metrics
    for pos in POSITION_GROUPS:
        mask = (position_groups == pos).values
        if mask.sum() == 0:
            continue
        
        y_pos = actual[mask]
        global_pos = global_preds[mask]
        moe_pos = moe_preds[mask]
        
        results[pos] = {
            "n_rows": mask.sum(),
            "global_mae": mean_absolute_error(y_pos, global_pos),
            "moe_mae": mean_absolute_error(y_pos, moe_pos),
            "global_r2": r2_score(y_pos, global_pos) if len(np.unique(y_pos)) > 1 else 0.0,
            "moe_r2": r2_score(y_pos, moe_pos) if len(np.unique(y_pos)) > 1 else 0.0,
        }
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Shadow Comparison: {task_name}")
    logger.info(f"{'='*60}")
    logger.info(f"{'Position':<10} {'N':<8} {'Global MAE':<12} {'MoE MAE':<12} {'Δ MAE':<10}")
    logger.info("-" * 52)
    
    for key in ["overall"] + POSITION_GROUPS:
        if key not in results:
            continue
        r = results[key]
        n = r.get("n_rows", len(actual) if key == "overall" else 0)
        delta = r["moe_mae"] - r["global_mae"]
        delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
        logger.info(f"{key:<10} {n:<8} {r['global_mae']:<12.4f} {r['moe_mae']:<12.4f} {delta_str:<10}")
    
    logger.info("-" * 52)
    overall = results["overall"]
    logger.info(f"Overall R² improvement: {overall['global_r2']:.4f} → {overall['moe_r2']:.4f}")
    logger.info(f"{'='*60}\n")
    
    return results

