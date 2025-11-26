from __future__ import annotations

import copy
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _merge_configs(base_cfg: Dict[str, Any] | None, override_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg) if isinstance(base_cfg, dict) else {}
    if isinstance(override_cfg, dict):
        cfg.update({k: v for k, v in override_cfg.items() if v is not None})
    return cfg


def compute_sample_weights(
    df: pd.DataFrame,
    base_cfg: Dict[str, Any] | None,
    problem_cfg: Dict[str, Any] | None,
    target_col: str | None = None,
) -> pd.Series:
    """
    Shared utility for computing time-decayed sample weights.
    """
    cfg = _merge_configs(base_cfg, (problem_cfg or {}).get("sample_weighting"))

    if not cfg.get("enabled", False) or df.empty:
        return pd.Series(np.ones(len(df), dtype=np.float32), index=df.index)

    if "season" not in df.columns:
        logger.warning("Sample weighting enabled but 'season' column missing; using uniform weights.")
        return pd.Series(np.ones(len(df), dtype=np.float32), index=df.index)

    season_series = pd.to_numeric(df["season"], errors="coerce")
    valid_seasons = season_series.dropna()
    if valid_seasons.empty:
        return pd.Series(np.ones(len(df), dtype=np.float32), index=df.index)

    try:
        current_season = cfg.get("reference_season")
        current_season = int(valid_seasons.max() if current_season is None else current_season)
    except Exception:
        current_season = int(valid_seasons.max())

    season_decay = float(cfg.get("season_decay", 0.6))
    season_gap = (current_season - season_series.fillna(current_season)).clip(lower=0)
    weights = np.power(season_decay, season_gap).astype(np.float64, copy=False)

    season_mask = season_series == current_season
    season_mask_array = season_mask.to_numpy(dtype=bool, copy=False)

    week_boost = float(cfg.get("current_season_week_boost", 0.0))
    if week_boost != 0.0 and "week" in df.columns:
        week_series = pd.to_numeric(df["week"], errors="coerce").fillna(0)
        if season_mask.any():
            max_week = week_series[season_mask].max()
            week_norm = (week_series / max_week).clip(lower=0, upper=1) if pd.notna(max_week) and max_week > 0 else pd.Series(0.0, index=df.index)
        else:
            week_norm = pd.Series(0.0, index=df.index)
        week_factor = 1.0 + week_boost * week_norm.where(season_mask, 0.0)
        weights = weights * week_factor.to_numpy(dtype=np.float64, copy=False)

    current_season_multiplier = float(cfg.get("current_season_multiplier", 1.0))
    if current_season_multiplier != 1.0:
        weights = weights * np.where(season_mask_array, current_season_multiplier, 1.0)

    current_season_positive_multiplier = float(cfg.get("current_season_positive_multiplier", 1.0))
    if current_season_positive_multiplier != 1.0 and target_col and target_col in df.columns:
        target_array = pd.to_numeric(df[target_col], errors="coerce").fillna(0).to_numpy()
        positive_mask = (target_array > 0) & season_mask_array
        weights = weights * np.where(positive_mask, current_season_positive_multiplier, 1.0)

    recent_week_boost_start = int(cfg.get("recent_week_boost_start", 0))
    recent_week_boost_multiplier = float(cfg.get("recent_week_boost_multiplier", 1.0))
    if recent_week_boost_multiplier != 1.0 and recent_week_boost_start > 0 and "week" in df.columns:
        week_series = pd.to_numeric(df["week"], errors="coerce").fillna(0)
        recency_mask = (week_series >= recent_week_boost_start) & season_mask_array
        weights = weights * np.where(recency_mask, recent_week_boost_multiplier, 1.0)

    min_weight = float(cfg.get("min_weight", 0.1))
    max_weight = float(cfg.get("max_weight", 3.0))
    weights = np.clip(weights, min_weight, max_weight)

    if cfg.get("normalize", True) and weights.sum() > 0:
        weights = weights / weights.mean()

    return pd.Series(weights.astype(np.float32, copy=False), index=df.index)

