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

    adjusted = np.clip(df["_prob"] * df["scale"], 0.0, 1.0)
    return adjusted.to_numpy(dtype=np.float64)


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


