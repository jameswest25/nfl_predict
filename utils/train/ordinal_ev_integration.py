# utils/train/ordinal_ev_integration.py

from __future__ import annotations
import os, glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import logging

from .ordinal_ev_core import ordinal_expected_value_from_heads
from .xgb_utils import predict_proba_batched  # keep consistency with your stack

logger = logging.getLogger(__name__)

# We'll use only trainer accessors/paths to avoid tight-coupling with persist.py internals.

def _load_inference_artifacts(trainer, problem_name: str) -> dict:
    """Load per-problem frozen features & categorical levels."""
    art_path = trainer.model_dir / f"inference_artifacts_{problem_name}.joblib"  # Use legacy flat path for now
    if not os.path.exists(art_path):
        # Try versioned path if available
        from .persist import _vdir
        art_path = _vdir(trainer, problem_name, None, "artifacts") / "inference" / f"inference_artifacts_{problem_name}.joblib"
    try:
        import joblib
        return joblib.load(art_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load inference artifacts for {problem_name}: {e}")

def _prep_X(X_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Reindex columns, re-apply categorical levels, float32 numerics."""
    X = X_raw.reindex(columns=artifacts['feature_columns'])
    cats = artifacts.get('category_levels', {})
    for col, levels in cats.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=levels)
    # Downcast numerics
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        X[num_cols] = X[num_cols].astype(np.float32)
    return X

def _load_final_model(trainer, problem_name: str, model_name: str) -> object:
    """
    Return calibrated model if in-memory; otherwise load most recent saved pickle.
    Compatible with both run_id and legacy versioning modes via trainer.persist.
    """
    key = f"{problem_name}_{model_name}"
    m = trainer.models.get(key)
    if m is not None:
        return m
    # Use persist to find latest model for (problem, model)
    from .persist import _vdir
    model_path = _vdir(trainer, problem_name, model_name, "models") / "model.joblib"
    if not os.path.exists(model_path):
        # Try legacy path
        model_path = trainer.model_dir / f"{problem_name}_{model_name}.joblib"
    try:
        import joblib
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {key}: {e}")

def _select_inference_slice(trainer, df_all: pd.DataFrame) -> pd.DataFrame:
    """Honor config.ordinal_ev.inference_selector."""
    cfg = trainer.cfg.ordinal_ev
    time_col = trainer.cfg.data_split['time_col']

    selector_cfg = cfg.get('inference_selector') or {}
    mode = str(selector_cfg.get('mode', 'latest_date')).lower()
    if mode == 'exact_date':
        if 'exact_date' not in selector_cfg:
            raise ValueError("ordinal_ev.inference_selector.exact_date is required when mode='exact_date'.")
        target = pd.to_datetime(selector_cfg['exact_date']).date()
        return df_all[df_all[time_col].dt.date == target].copy()
    if mode == 'passthrough':
        return df_all.copy()
    # default: latest_date
    latest = df_all[time_col].max()
    return df_all[df_all[time_col] == latest].copy()

def score_heads_on_frame(
    trainer,
    X_raw: pd.DataFrame,
    heads: Dict[int, str],
    model_name: str
) -> Dict[int, np.ndarray]:
    """
    For each k -> problem_name, build the correct feature matrix and score calibrated probabilities.
    Returns {k: S>=k} aligned to X_raw.index.
    """
    out: Dict[int, np.ndarray] = {}
    # We score per head independently because each head can have its own frozen feature list.
    for k in sorted(heads.keys()):
        prob_name = heads[k]
        artifacts = _load_inference_artifacts(trainer, prob_name)
        Xk = _prep_X(X_raw, artifacts)
        model = _load_final_model(trainer, prob_name, model_name=model_name)
        proba = predict_proba_batched(model, Xk, artifacts['feature_columns'])
        out[k] = proba
    return out

def assemble_ev_dataframe(
    X_raw: pd.DataFrame,
    probas_by_k: Dict[int, np.ndarray],
    mu_tail_5plus: float,
    synth_tail_ratio: Optional[float] = None
) -> pd.DataFrame:
    ev, pmf, S = ordinal_expected_value_from_heads(
        probas_by_k=probas_by_k,
        mu_tail_5plus=mu_tail_5plus,
        synth_tail_ratio=synth_tail_ratio
    )
    df = pd.DataFrame({
        "p0": pmf[:, 0], "p1": pmf[:, 1], "p2": pmf[:, 2],
        "p3": pmf[:, 3], "p4": pmf[:, 4], "p5plus": pmf[:, 5],
        "S>=1": S[:, 0], "S>=2": S[:, 1], "S>=3": S[:, 2], "S>=4": S[:, 3], "S>=5": S[:, 4],
        "ev_tb": ev,
    }, index=X_raw.index)
    return df

def write_ev_output(trainer, df_join: pd.DataFrame, filename_prefix: str, fmt: str = "csv") -> str:
    """
    Writes EV output under the run's metrics directory (versioned if run_id mode).
    Returns the full path written.
    """
    from .persist import _vdir
    out_dir = _vdir(trainer, "", None, "metrics")  # Use empty problem_name for global outputs
    os.makedirs(out_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{filename_prefix}_{ts}.{('parquet' if fmt=='parquet' else 'csv')}")
    if fmt.lower() == "parquet":
        df_join.to_parquet(path, index=False)
    else:
        df_join.to_csv(path, index=False)
    logger.info(f"[ordinal-ev] wrote {path}")
    return path

def run_ordinal_ev_job(trainer) -> str:
    """
    Entrypoint used by train.py after (or independent of) training.
    1) Load the feature matrix (projected the same way your training stack does).
    2) Select inference slice per config.
    3) Score heads, assemble pmf+EV.
    4) Join common meta (e.g., ids) and write output.
    """
    cfg = trainer.cfg
    time_col = cfg.data_split['time_col']
    df_all = trainer.load_data()  # Use trainer's load_data method
    df_all[time_col] = pd.to_datetime(df_all[time_col])

    X_inf = _select_inference_slice(trainer, df_all)
    if X_inf.empty:
        raise RuntimeError("[ordinal-ev] inference slice is empty — check your selector config.")

    heads_cfg = cfg.ordinal_ev.get('heads')
    if not heads_cfg:
        logger.info("[ordinal-ev] No heads configured; skipping ordinal EV job.")
        return ""
    heads_map = {int(k): v for k, v in heads_cfg.items()}  # ensure int keys
    model_name = cfg.ordinal_ev.get("model_name", "xgboost")
    mu_tail = float(cfg.mu_tail_5plus)

    # If 5+ head missing, allow synth tail via ratio (you can estimate offline and put into config)
    synth_ratio = cfg.ordinal_ev.get("tail_ratio_if_no_S5", None)

    # Score
    probas_by_k = score_heads_on_frame(trainer, X_inf, heads=heads_map, model_name=model_name)

    # Assemble EV
    ev_df = assemble_ev_dataframe(X_inf, probas_by_k, mu_tail_5plus=mu_tail, synth_tail_ratio=synth_ratio)

    # Pick some ID/meta columns to keep next to predictions if present
    keep_meta_candidates = [
        "game_id",
        "player_id",
        "player_name",
        "team",
        "opponent",
        time_col,
    ]
    keep_meta = [c for c in keep_meta_candidates if c in X_inf.columns]
    out_df = X_inf[keep_meta].join(ev_df)

    # Write
    fmt = cfg.ordinal_ev.get("output", {}).get("format", "csv")
    prefix = cfg.ordinal_ev.get("output", {}).get("filename_prefix", "ordinal_ev_total_bases")
    out_path = write_ev_output(trainer, out_df.reset_index(drop=True), prefix, fmt)
    return out_path