"""Model inference for predictions.

This module handles loading trained models and generating predictions,
with proper post-processing and guard logic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

MODEL_DIR = Path("output/models")
METRICS_DIR = Path("output/metrics")

from utils.feature.enrichment.asof import get_decision_cutoff_hours
from utils.general.constants import format_cutoff_label


def _active_cutoff_label() -> str:
    """Resolve the active cutoff label for horizon-safe artifact loading."""
    try:
        hours = float(get_decision_cutoff_hours())
        return format_cutoff_label(hours)
    except Exception:
        return "default"


def _latest_inference_artifacts_path(problem_name: str) -> Path:
    """Find the most recent inference artifacts file for a problem.

    Training runs typically write artifacts under:
      output/models/<problem>/<run_id>/inference/inference_artifacts_<problem>.joblib

    Some workflows may also place artifacts directly at:
      output/models/inference_artifacts_<problem>.joblib
    """
    cutoff_label = _active_cutoff_label()

    # 1) Preferred: horizon-safe top-level artifacts (explicitly curated / copied)
    if cutoff_label != "default":
        top_level = MODEL_DIR / f"inference_artifacts_{problem_name}_{cutoff_label}.joblib"
        if top_level.exists():
            return top_level
    else:
        top_level = MODEL_DIR / f"inference_artifacts_{problem_name}.joblib"
        if top_level.exists():
            return top_level

    # 2) Search within the problem directory for the newest run artifacts
    problem_dir = MODEL_DIR / problem_name
    if problem_dir.exists():
        candidates = list(problem_dir.glob("*/inference/inference_artifacts_*.joblib"))
        if cutoff_label != "default":
            # Versioned artifacts in run dirs are horizon-safe via run_id naming.
            wanted = f"inference_artifacts_{problem_name}.joblib"
            candidates = [p for p in candidates if p.name == wanted and f"cutoff_{cutoff_label}" in p.parent.parent.name]
        else:
            wanted = f"inference_artifacts_{problem_name}.joblib"
            candidates = [p for p in candidates if p.name == wanted]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)

    return Path()


def load_artifacts(problem_name: str) -> dict:
    """Load inference artifacts for a problem.
    
    Parameters
    ----------
    problem_name : str
        Name of the problem (e.g., 'anytime_td_structured')
    
    Returns
    -------
    dict
        Inference artifacts including feature columns, category levels, etc.
    """
    path = _latest_inference_artifacts_path(problem_name)
    if not path.exists() or not path.is_file():
        if problem_name == "anytime_td":
            alt = MODEL_DIR / "inference_artifacts_anytime_td_meta.joblib"
            if alt.exists():
                logger.warning(
                    "Using legacy anytime_td_meta inference artifacts for anytime_td."
                )
                return joblib.load(alt)
        raise FileNotFoundError(
            f"Inference artifacts not found at {path}. "
            "Run the training pipeline first."
        )
    return joblib.load(path)


def _latest_model_path(problem: str) -> Path:
    """Find the latest model file for a problem."""
    cutoff_label = _active_cutoff_label()
    names_to_try = [problem]
    if problem == "anytime_td":
        names_to_try.append("anytime_td_meta")
    
    for name in names_to_try:
        problem_dir = MODEL_DIR / name / "xgboost"
        if not problem_dir.exists():
            continue
        candidates = [d for d in problem_dir.iterdir() if d.is_dir()]
        if cutoff_label != "default":
            candidates = [d for d in candidates if f"cutoff_{cutoff_label}" in d.name]
        if not candidates:
            continue
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        model_path = latest / "model.joblib"
        if model_path.exists():
            if name != problem:
                logger.warning(
                    "Using legacy model directory for %s located at %s",
                    problem, latest
                )
            return model_path
    
    raise FileNotFoundError(f"No model directory found for {problem}.")


def _metrics_path(problem: str) -> Path:
    """Find the metrics file for a problem."""
    cutoff_label = _active_cutoff_label()
    names_to_try = [problem]
    if problem == "anytime_td":
        names_to_try.append("anytime_td_meta")
    
    for name in names_to_try:
        problem_dir = METRICS_DIR / name / "xgboost"
        if not problem_dir.exists():
            continue
        runs = [d for d in problem_dir.iterdir() if d.is_dir()]
        if cutoff_label != "default":
            runs = [d for d in runs if f"cutoff_{cutoff_label}" in d.name]
        if not runs:
            continue
        latest = max(runs, key=lambda p: p.stat().st_mtime)
        path = latest / "metrics.yaml"
        if path.exists():
            if name != problem:
                logger.warning(
                    "Using legacy metrics directory for %s located at %s",
                    problem, latest
                )
            return path
    return Path()


def load_threshold(problem: str) -> float:
    """Load decision threshold for a classification problem."""
    metrics_path = _metrics_path(problem)
    if not metrics_path.exists():
        return 0.5
    with metrics_path.open("r") as fp:
        metrics = yaml.safe_load(fp)
    return float(metrics.get("decision_threshold", 0.5))


def prepare_feature_matrix(
    df: pd.DataFrame,
    artifacts: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare features for model inference.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataframe
    artifacts : dict
        Inference artifacts with feature_columns, category_levels, etc.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (X features, meta columns)
    """
    feature_cols = artifacts["feature_columns"]
    cat_levels = artifacts.get("category_levels", {})
    categorical_features = artifacts.get("categorical_features", [])

    # Meta columns for output
    META_COLUMNS = [
        "player_id", "player_name", "position", "team", "opponent",
        "season", "week", "game_id", "game_date",
    ]
    meta = df.reindex(columns=[c for c in META_COLUMNS if c in df.columns], fill_value=None).copy()
    
    X = df.reindex(columns=feature_cols, fill_value=np.nan)

    # Convert datetime columns to numeric epoch micros
    datetime_cols = X.select_dtypes(include=["datetime", "datetimetz"]).columns
    for col in datetime_cols:
        col_as_dt = pd.to_datetime(X[col], utc=True, errors="coerce")
        numeric = col_as_dt.view("int64").astype("float64", copy=False)
        numeric[col_as_dt.isna().to_numpy()] = np.nan
        X[col] = numeric / 1_000_000.0

    # Apply categorical levels
    for col, levels in cat_levels.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=levels)

    # Ensure categorical dtype for recorded categorical columns
    for col in categorical_features:
        if col in X.columns and not pd.api.types.is_categorical_dtype(X[col]):
            X[col] = pd.Categorical(X[col])

    # Normalize numeric types
    numeric_cols = X.select_dtypes(include=["int64", "int32", "float64"]).columns
    if len(numeric_cols):
        X[numeric_cols] = X[numeric_cols].astype("float32")

    return X, meta


def check_moe_available(problem_name: str) -> bool:
    """Check if MoE (per-position) models are available for a problem."""
    cutoff_label = _active_cutoff_label()
    if cutoff_label != "default":
        top = MODEL_DIR / f"inference_artifacts_{problem_name}_moe_{cutoff_label}.joblib"
        if top.exists():
            return True
        # Check for a versioned copy under the problem run directory for this cutoff
        problem_dir = MODEL_DIR / problem_name
        if problem_dir.exists():
            wanted = f"inference_artifacts_{problem_name}_moe_{cutoff_label}.joblib"
            candidates = list(problem_dir.glob(f"*/inference/{wanted}"))
            return bool(candidates)
        return False

    # Default (legacy) horizon
    return (MODEL_DIR / f"inference_artifacts_{problem_name}_moe.joblib").exists()


def predict_moe(features: pd.DataFrame, problem_name: str) -> np.ndarray:
    """Make predictions using per-position MoE models.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature dataframe
    problem_name : str
        Name of the problem
    
    Returns
    -------
    np.ndarray
        Predictions
    """
    cutoff_label = _active_cutoff_label()
    moe_artifacts_path = (
        MODEL_DIR / f"inference_artifacts_{problem_name}_moe_{cutoff_label}.joblib"
        if cutoff_label != "default"
        else MODEL_DIR / f"inference_artifacts_{problem_name}_moe.joblib"
    )
    if not moe_artifacts_path.exists():
        # Try to find a versioned copy under the problem run directory (same cutoff label)
        problem_dir = MODEL_DIR / problem_name
        if problem_dir.exists():
            wanted = moe_artifacts_path.name
            candidates = list(problem_dir.glob(f"*/inference/{wanted}"))
            if candidates:
                moe_artifacts_path = max(candidates, key=lambda p: p.stat().st_mtime)
    if not moe_artifacts_path.exists():
        raise FileNotFoundError(
            f"MoE inference artifacts not found for {problem_name} at cutoff {cutoff_label}. "
            "Train this horizon first."
        )
    artifacts = joblib.load(moe_artifacts_path)

    feature_columns = artifacts["feature_columns"]
    positions = artifacts["positions"]
    fallback_pos = artifacts.get("fallback_position", "WR")
    
    # Load per-position models
    models = {}
    model_base = MODEL_DIR / problem_name / "xgboost"
    for pos in positions:
        model_path = model_base / f"model.{pos}.joblib"
        if model_path.exists():
            models[pos] = joblib.load(model_path)
            logger.info(f"Loaded MoE model: {problem_name}.{pos}")
    
    if not models:
        raise ValueError(f"No MoE models found for {problem_name}")
    
    # Prepare features using the SAME categorical/date normalization as global inference.
    # This is critical for MoE parity: converting strings to numeric directly would coerce
    # them to NaN and destroy signal (team/opponent/position/etc).
    try:
        global_artifacts = load_artifacts(problem_name)
        X_full, _ = prepare_feature_matrix(features, global_artifacts)
        X = X_full.reindex(columns=feature_columns, fill_value=np.nan)
    except Exception:
        # Fallback to minimal column projection if global artifacts are unavailable
        X = features.reindex(columns=feature_columns, fill_value=np.nan).copy()

    # Ensure any categorical columns are numeric codes for XGBoost.
    for col in X.columns:
        if str(X[col].dtype) == "category":
            X[col] = X[col].cat.codes
        elif X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    # IMPORTANT: do NOT impute missing values with 0. XGBoost handles missing values.
    
    # Get position groups
    if "position_group" in features.columns:
        position_groups = features["position_group"]
    elif "position" in features.columns:
        def normalize_pos(p):
            p = str(p).upper()
            if p in {"RB", "HB", "FB"}:
                return "RB"
            if p == "WR":
                return "WR"
            if p == "TE":
                return "TE"
            if p == "QB":
                return "QB"
            return "WR"
        position_groups = features["position"].apply(normalize_pos)
    else:
        position_groups = pd.Series([fallback_pos] * len(features))
    
    # Make predictions by position
    preds = np.zeros(len(features), dtype=np.float64)
    
    # Determine whether these are classification-probability models.
    # For classifiers, prefer predict_proba; for regressors, use predict.
    any_has_proba = any(hasattr(m, "predict_proba") for m in models.values())

    for pos, model in models.items():
        mask = (position_groups == pos).values
        if mask.sum() == 0:
            continue
        
        X_pos = X.loc[mask]
        # NOTE: Many MoE models are saved as sklearn-style classifiers (e.g., XGBClassifier).
        # For classifiers, `.predict()` returns class labels (0/1) NOT probabilities.
        # We must use `predict_proba` when available.
        if any_has_proba and hasattr(model, "predict_proba"):
            pos_preds = model.predict_proba(X_pos)
            if getattr(pos_preds, "ndim", 1) > 1 and pos_preds.shape[1] > 1:
                pos_preds = pos_preds[:, 1]
        else:
            pos_preds = model.predict(X_pos)

        pos_preds = np.asarray(pos_preds, dtype=float)
        preds[mask] = pos_preds
        logger.info(
            f"MoE {problem_name}.{pos}: {mask.sum()} predictions, mean={pos_preds.mean():.4f}"
        )
    
    # Handle positions not in models (use fallback)
    known_positions = set(models.keys())
    fallback_mask = ~position_groups.isin(known_positions).values
    if fallback_mask.sum() > 0 and fallback_pos in models:
        X_fallback = X.loc[fallback_mask]
        fallback_model = models[fallback_pos]
        if any_has_proba and hasattr(fallback_model, "predict_proba"):
            fallback_preds = fallback_model.predict_proba(X_fallback)
            if getattr(fallback_preds, "ndim", 1) > 1 and fallback_preds.shape[1] > 1:
                fallback_preds = fallback_preds[:, 1]
        else:
            fallback_preds = fallback_model.predict(X_fallback)
        preds[fallback_mask] = np.asarray(fallback_preds, dtype=float)
        logger.info(f"MoE {problem_name}: {fallback_mask.sum()} using fallback ({fallback_pos})")
    
    # Only clip for probability outputs. Regression tasks should not be clipped.
    if any_has_proba:
        return np.clip(preds, 0.0, 1.0)
    return preds


def predict_global(
    features: pd.DataFrame,
    problem_config: dict,
    artifacts: dict,
) -> np.ndarray:
    """Make predictions using global model.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature dataframe
    problem_config : dict
        Problem configuration from training.yaml
    artifacts : dict
        Inference artifacts
    
    Returns
    -------
    np.ndarray
        Predictions
    """
    problem_name = problem_config.get("name", "<unknown>")
    task_type = str(
        problem_config.get("task_type") or artifacts.get("task_type") or ""
    ).lower()
    output_mode = artifacts.get("output_mode") or problem_config.get("output_mode")

    X, _ = prepare_feature_matrix(features, artifacts)
    model = joblib.load(_latest_model_path(problem_name))

    is_classification = task_type in {"classification", "binary", "multiclass"}
    if is_classification:
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)
            if preds.ndim > 1 and preds.shape[1] > 1:
                preds = preds[:, 1]
        else:
            preds = model.predict(X)
        preds = np.asarray(preds, dtype=float)
        preds = np.clip(preds, 0.0, 1.0)
    else:
        preds = np.asarray(model.predict(X), dtype=float)

    if output_mode and output_mode == "logit":
        preds = 1.0 / (1.0 + np.exp(-preds))

    return preds


# ---------------------------------------------------------------------------
# Post-processing guards
# ---------------------------------------------------------------------------

def apply_guards_inline(features_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Apply injury-based guards to predictions.
    
    Zeros out predictions for inactive players and reduces predictions
    for doubtful/questionable players.
    """
    adj = preds.copy()
    
    # Inactive designation
    if "injury_is_inactive_designation" in features_df.columns:
        mask = (
            pd.to_numeric(features_df["injury_is_inactive_designation"], errors="coerce")
            .fillna(0).astype(bool).to_numpy()
        )
        adj[mask] = 0.0

    if "injury_game_designation" in features_df.columns:
        des = features_df["injury_game_designation"].fillna("").astype(str).str.upper()
        out_mask = des.isin(["OUT", "INACTIVE"]).to_numpy()
        adj[out_mask] = 0.0

        doubtful_mask = (des == "DOUBTFUL").to_numpy()
        adj[doubtful_mask] *= 0.25

        q_mask = (des == "QUESTIONABLE").to_numpy()
        adj[q_mask] *= 0.85

    if "injury_practice_status" in features_df.columns:
        prac = features_df["injury_practice_status"].fillna("").astype(str).str.upper()
        dnp_mask = prac.str.contains("DID NOT PARTICIPATE", regex=False).to_numpy()
        adj[dnp_mask] *= 0.55

        lim_mask = prac.str.contains("LIMITED", regex=False).to_numpy()
        adj[lim_mask] *= 0.85

    return np.clip(adj, 0.0, 1.0)


def apply_availability_floor(features_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Apply minimum availability floor for active players with missing history."""
    adj = preds.copy()
    
    if "snap_offense_pct_l3" in features_df.columns:
        history = pd.to_numeric(features_df["snap_offense_pct_l3"], errors="coerce").fillna(0)
        no_history_mask = (history == 0).to_numpy()

        # Determine active status
        if "injury_is_inactive_designation" in features_df.columns:
            inactive = (
                pd.to_numeric(features_df["injury_is_inactive_designation"], errors="coerce")
                .fillna(0).astype(bool).to_numpy()
            )
            active_mask = ~inactive
        else:
            active_mask = np.ones(len(adj), dtype=bool)

        # Apply floor
        if "position" in features_df.columns:
            is_qb = (features_df["position"] == "QB").to_numpy()
            qb_floor_mask = no_history_mask & active_mask & is_qb
            other_floor_mask = no_history_mask & active_mask & (~is_qb)

            adj[qb_floor_mask] = np.maximum(adj[qb_floor_mask], 0.9)
            adj[other_floor_mask] = np.maximum(adj[other_floor_mask], 0.35)
        else:
            floor_mask = no_history_mask & active_mask
            adj[floor_mask] = np.maximum(adj[floor_mask], 0.35)

    return adj


def apply_snaps_ceiling_cap(features_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Apply post-processing cap for snaps predictions."""
    adjusted = preds.copy()
    
    ceiling = None
    max_snap_pct = None
    
    if "snap_ceiling_l5" in features_df.columns:
        ceiling = pd.to_numeric(features_df["snap_ceiling_l5"], errors="coerce").to_numpy()
    if "max_snap_pct_l5" in features_df.columns:
        max_snap_pct = pd.to_numeric(features_df["max_snap_pct_l5"], errors="coerce").to_numpy()
    
    if ceiling is None or max_snap_pct is None:
        return adjusted
    
    threshold = 0.25
    multiplier = 1.5
    
    should_cap = (~np.isnan(ceiling)) & (~np.isnan(max_snap_pct)) & (max_snap_pct < threshold)
    cap_values = ceiling * multiplier
    
    capped_mask = should_cap & (adjusted > cap_values)
    adjusted[capped_mask] = cap_values[capped_mask]
    
    n_capped = capped_mask.sum()
    if n_capped > 0:
        logger.info(f"Snaps ceiling cap applied to {n_capped} predictions")
    
    return adjusted


def apply_usage_targets_position_cap(features_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Apply post-processing for usage_targets predictions."""
    adjusted = preds.copy()
    n_zeroed = 0
    n_capped = 0
    
    if "position" in features_df.columns:
        is_qb = (features_df["position"] == "QB").to_numpy()
        n_zeroed = is_qb.sum()
        adjusted[is_qb] = 0.0
    
    if "is_fullback" in features_df.columns:
        is_fb = (features_df["is_fullback"] == 1).to_numpy()
        fb_cap = 0.10
        exceeds_cap = is_fb & (adjusted > fb_cap)
        n_capped = exceeds_cap.sum()
        adjusted[exceeds_cap] = fb_cap
    
    if "position" in features_df.columns and "hist_target_share_l3" in features_df.columns:
        is_rb = (features_df["position"] == "RB").to_numpy()
        hist_ts = pd.to_numeric(features_df["hist_target_share_l3"], errors="coerce").fillna(0).to_numpy()
        
        rb_cap = np.maximum(hist_ts * 1.5, 0.03)
        rb_exceeds = is_rb & (adjusted > rb_cap)
        n_rb_capped = rb_exceeds.sum()
        adjusted[rb_exceeds] = rb_cap[rb_exceeds]
        
        if n_rb_capped > 0:
            logger.info(f"Usage targets: capped {n_rb_capped} RB predictions at hist_target_share × 1.5")
    
    if n_zeroed > 0 or n_capped > 0:
        logger.info(f"Usage targets post-processing: zeroed {n_zeroed} QBs, capped {n_capped} FBs at 0.10")
    
    return adjusted


def apply_usage_carries_position_cap(features_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Apply post-processing for usage_carries predictions."""
    adjusted = preds.copy()
    n_capped = 0
    
    if "position" in features_df.columns:
        position = features_df["position"].to_numpy()
        
        is_wr = position == "WR"
        wr_exceeds = is_wr & (adjusted > 0.05)
        adjusted[wr_exceeds] = 0.05
        n_capped += wr_exceeds.sum()
        
        is_te = position == "TE"
        te_exceeds = is_te & (adjusted > 0.02)
        adjusted[te_exceeds] = 0.02
        n_capped += te_exceeds.sum()
        
        is_qb = position == "QB"
        qb_exceeds = is_qb & (adjusted > 0.30)
        adjusted[qb_exceeds] = 0.30
        n_capped += qb_exceeds.sum()
    
    if "is_fullback" in features_df.columns:
        is_fb = (features_df["is_fullback"] == 1).to_numpy()
        fb_exceeds = is_fb & (adjusted > 0.10)
        adjusted[fb_exceeds] = 0.10
        n_capped += fb_exceeds.sum()
    
    if n_capped > 0:
        logger.info(f"Usage carries post-processing: capped {n_capped} non-RB predictions")
    
    return adjusted


def apply_usage_target_yards_position_cap(features_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Apply post-processing for usage_target_yards predictions."""
    adjusted = preds.copy()
    n_zeroed = 0
    n_capped = 0
    
    if "position" in features_df.columns:
        position = features_df["position"].to_numpy()
        
        is_qb = position == "QB"
        n_zeroed = is_qb.sum()
        adjusted[is_qb] = 0.0
        
        is_rb = position == "RB"
        adjusted[is_rb] = np.clip(adjusted[is_rb], 0.0, 2.0)
        n_capped += (is_rb & (preds > 2.0)).sum()
    
    if "is_fullback" in features_df.columns:
        is_fb = (features_df["is_fullback"] == 1).to_numpy()
        adjusted[is_fb] = 0.0
    
    if n_zeroed > 0 or n_capped > 0:
        logger.info(f"Usage target yards post-processing: zeroed {n_zeroed} QBs, capped {n_capped} RBs")
    
    return adjusted


def apply_efficiency_rec_yards_air_cap(features_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Apply post-processing for efficiency_rec_yards_air predictions."""
    adjusted = preds.copy()
    n_capped = 0
    
    if "position" in features_df.columns:
        position = features_df["position"].to_numpy()
        
        is_rb = position == "RB"
        exceeds = is_rb & (adjusted > 0)
        adjusted[exceeds] = 0.0
        n_capped = exceeds.sum()
    
    if "is_fullback" in features_df.columns:
        is_fb = (features_df["is_fullback"] == 1).to_numpy()
        adjusted[is_fb] = 0.0
    
    if n_capped > 0:
        logger.info(f"Efficiency rec yards air post-processing: capped {n_capped} RBs at 0")
    
    return adjusted
