# utils/train/selective/loss.py
from __future__ import annotations

import numpy as np
import torch

_EPS = 1e-9


def selective_objective(g: torch.Tensor, ce: torch.Tensor, cov_target: float, lam: float) -> tuple[torch.Tensor, float]:
    """
    g: (B,1) in (0,1)
    ce: (B,1) >= 0   per-sample cross-entropy of the base classifier
    Returns (loss, mean_g)
    """
    g = g.clamp(1e-6, 1 - 1e-6)
    ce = ce.clamp_min(0.0)
    sum_g = g.sum().clamp_min(1e-6)
    mean_g = g.mean()

    risk = (g * ce).sum() / sum_g
    cov_pen = lam * torch.relu(torch.as_tensor(cov_target, device=g.device) - mean_g)
    loss = risk + cov_pen
    return loss, float(mean_g.detach().cpu().item())


def binary_ce_from_proba(p_pos: np.ndarray, y_true: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """
    Compute per-sample binary cross-entropy: -[y log p + (1-y) log (1-p)]
    p_pos: shape [N] in (0,1)
    y_true: shape [N] in {0,1}
    """
    # Use a larger epsilon for numerical stability
    eps = max(eps, 1e-7)
    p = np.clip(p_pos, eps, 1 - eps)
    y = y_true.astype(np.float32)

    # Avoid log(0) by using stable computation
    log_p = np.log(p)
    log_1_minus_p = np.log(1.0 - p)

    # Replace any inf values with large negative numbers
    log_p = np.where(np.isfinite(log_p), log_p, -1e6)
    log_1_minus_p = np.where(np.isfinite(log_1_minus_p), log_1_minus_p, -1e6)

    return -(y * log_p + (1.0 - y) * log_1_minus_p)
