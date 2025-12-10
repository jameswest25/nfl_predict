from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class TeamTotalConfig:
    """Configuration parameters required for team-level probability adjustment."""

    divisor: float
    global_scale: float
    mean_total: float
    group_cols: List[str]
    min_scale: float = 0.25
    max_scale: float = 3.0
    # Optional: columns used to compute intra-team weighting for TD allocation.
    # When empty or None, scaling is uniform within each group.
    weight_features: List[str] | None = None
    # Strength of role-aware adjustment within each group.
    # 0.0 => uniform; higher values put more mass on high-weight players.
    role_intensity: float = 0.5


def adjust_probabilities(X: pd.DataFrame, probs: np.ndarray, config: TeamTotalConfig) -> np.ndarray:
    """Scale per-player probabilities so team totals better align with betting expectations."""
    if probs.size == 0:
        return probs
    if not isinstance(X, pd.DataFrame):
        return probs
    if "team_implied_total" not in X.columns:
        return probs

    df = pd.DataFrame(index=X.index)
    df["_prob"] = probs
    df["team_implied_total"] = pd.to_numeric(
        X["team_implied_total"], errors="coerce"
    ).fillna(config.mean_total)

    for col in config.group_cols:
        if col in X.columns:
            df[col] = X[col]

    group_cols = [col for col in config.group_cols if col in df.columns]
    if not group_cols:
        return probs

    grouped = df.groupby(group_cols, dropna=False, sort=False)
    expected_td = grouped["team_implied_total"].mean() / config.divisor
    expected_td = expected_td.clip(lower=0.0)

    predicted_td = grouped["_prob"].sum()
    scale = expected_td / predicted_td.replace(0.0, np.nan)
    scale = scale.replace([np.inf, -np.inf], np.nan)
    
    # Sanity check: if predicted sum is extremely low (< 0.1), the model likely missed key features (e.g. odds).
    # Applying a 50x multiplier here causes blowout. Fallback to global scale or 1.0.
    safe_mask = predicted_td >= 0.1
    scale = scale.where(safe_mask, config.global_scale)

    scale = scale.fillna(config.global_scale)
    scale = scale.clip(lower=config.min_scale, upper=config.max_scale)

    df = df.join(scale.rename("scale"), on=group_cols)
    df["scale"] = df["scale"].fillna(config.global_scale)

    # Start with uniform team-total scaling
    adjusted = df["_prob"] * df["scale"]

    # Optional role-aware reweighting within each group
    weight_cols = (config.weight_features or []).copy()
    role_intensity = float(config.role_intensity or 0.0)
    if weight_cols and role_intensity > 0.0:
        # Build a non-negative aggregate weight per row
        w = np.zeros_like(adjusted, dtype=np.float64)
        for col in weight_cols:
            if col in X.columns:
                col_vals = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                col_vals[col_vals < 0.0] = 0.0
                w += col_vals
        # Fallback to uniform if all-zero within a group
        df["_weight_raw"] = w
        grouped_w = df.groupby(group_cols, dropna=False, sort=False)["_weight_raw"]
        sum_w = grouped_w.transform("sum")
        # Normalised weights within each group, defaulting to 1/N when sum_w == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            w_norm = np.where(sum_w.to_numpy(dtype=np.float64) > 0.0, w / sum_w.to_numpy(dtype=np.float64), 0.0)
        df["_weight_norm"] = w_norm
        # Compute mean weight per group to centre the adjustment
        mean_w = grouped_w.transform("mean").replace({0.0: 1.0})
        mean_w = pd.to_numeric(mean_w, errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        w_centered = np.where(mean_w > 0.0, w / mean_w, 1.0)
        # Apply intra-team adjustment
        role_factor = 1.0 + role_intensity * (w_centered - 1.0)
        role_factor = np.clip(role_factor, 0.25, 4.0)
        adjusted = adjusted.to_numpy(dtype=np.float64) * role_factor
        # Re-normalise within each group to preserve expected TD totals
        df["_adjusted_role"] = adjusted
        grouped_adj = df.groupby(group_cols, dropna=False, sort=False)["_adjusted_role"]
        sum_adj = grouped_adj.transform("sum").to_numpy(dtype=np.float64)
        group_index = pd.MultiIndex.from_frame(df[group_cols])
        fallback_expected = float(config.mean_total) / float(config.divisor)
        expected_vec = (
            expected_td.reindex(group_index, fill_value=fallback_expected)
            .to_numpy(dtype=np.float64)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            renorm = np.where(sum_adj > 0.0, expected_vec / sum_adj, 1.0)
        adjusted = adjusted * renorm

    adjusted = np.clip(adjusted, 0.0, 1.0)
    return adjusted.astype(np.float64)


class TeamTotalAdjustedClassifier:
    """Wrapper that enforces team-level probability scaling at prediction time."""

    def __init__(self, base_estimator, config: TeamTotalConfig):
        self.base_estimator = base_estimator
        self.config = config
        if hasattr(base_estimator, "classes_"):
            self.classes_ = base_estimator.classes_

    def predict_proba(self, X):
        probs = self.base_estimator.predict_proba(X)
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if probs.ndim == 1:
            adjusted = adjust_probabilities(frame, probs, self.config)
            return np.vstack([1.0 - adjusted, adjusted]).T

        if probs.shape[1] >= 2:
            adjusted = adjust_probabilities(frame, probs[:, 1], self.config)
            probs = probs.copy()
            probs[:, 1] = adjusted
            probs[:, 0] = np.clip(1.0 - adjusted, 0.0, 1.0)
        return probs

    def predict(self, X):
        return self.base_estimator.predict(X)

    def __getattr__(self, item):
        base = object.__getattribute__(self, "base_estimator")
        return getattr(base, item)


