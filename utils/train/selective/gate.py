# utils/train/selective/gate.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:
    raise ImportError(
        "PyTorch is required for the Selective Gate. "
        "Please install: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from e


_EPS = 1e-6

def _sanitize_array(x: np.ndarray) -> np.ndarray:
    """
    Ensure float32, replace NaN/±Inf with finite values.
    NOTE: XGBoost tolerates NaNs; PyTorch MLP does not.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return x


class GateMLP(nn.Module):
    """
    Tiny MLP gate producing an abstain logit z_a(x).
    g(x) = sigmoid(z_a) serves as acceptance weight in training and a competing logit at inference.
    """
    def __init__(self, in_dim: int, hidden: Iterable[int] = (64,), dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout and dropout > 0:
                layers += [nn.Dropout(p=float(dropout))]
            prev = h
        layers += [nn.Linear(prev, 1)]  # abstain logit z_a
        self.net = nn.Sequential(*layers)

        # Kaiming init for ReLU layers; bias will be set by _init_last_bias_to_coverage
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Will be overridden by coverage-specific init

    def _init_last_bias_to_coverage(self, coverage_target: float):
        """Initialize the final layer bias to logit(coverage_target) for better starting point."""
        target_logit = float(np.log(coverage_target / (1.0 - coverage_target)))
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1:
                if m.bias is not None:
                    nn.init.constant_(m.bias, target_logit)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # returns z_a with shape [N, 1]
        return self.net(X)


@dataclass
class GateTrainingConfig:
    coverage_target: float
    lambda_val: float
    epochs: int = 3
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"


def _to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    x = _sanitize_array(x)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def fit_gate(
    model: GateMLP,
    X: np.ndarray,
    per_sample_ce: np.ndarray,   # non-negative, same length as X
    cfg: GateTrainingConfig,
) -> Tuple[GateMLP, float]:
    """
    Optimize gate parameters to minimize selective objective:
        L = (sum_i g_i * CE_i) / (sum_i g_i)  +  lambda * relu(cov_target - mean(g))
    where g_i = sigmoid(z_a(x_i)).

    Returns: (trained_model, mean_acceptance_g) on the final epoch.
    """
    assert X.shape[0] == per_sample_ce.shape[0], "X and per_sample_ce must align"
    device = cfg.device
    model = model.to(device)
    model.train()

    # Prepare tensors (sanitize first)
    X_t = _to_tensor(X, device)
    per_sample_ce = _sanitize_array(per_sample_ce).reshape(-1, 1)
    # guard: CE must be non-negative & finite
    per_sample_ce = np.clip(per_sample_ce, 0.0, 1e9)
    ce_t = torch.as_tensor(per_sample_ce, dtype=torch.float32, device=device)  # [N, 1]

    # Optimizer
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    N = X.shape[0]
    bs = min(N, int(cfg.batch_size))
    indices = np.arange(N)

    last_mean_g = 0.0
    # track epoch-level acceptance
    running_sum_g = 0.0
    running_count = 0

    for epoch in range(max(1, int(cfg.epochs))):
        epoch_loss = 0.0
        np.random.shuffle(indices)
        for start in range(0, N, bs):
            sl = indices[start:start + bs]
            xb = X_t[sl]
            ceb = ce_t[sl]

            z_a = model(xb)                   # [B,1]
            g = torch.sigmoid(z_a).clamp(_EPS, 1 - _EPS)
            sum_g = g.sum().clamp_min(_EPS)
            mean_g = g.mean()

            # Selective risk (normalized by accepted mass) + coverage penalty
            risk = (g * ceb).sum() / sum_g
            cov_pen = cfg.lambda_val * torch.relu(torch.as_tensor(cfg.coverage_target, device=device) - mean_g)
            loss = risk + cov_pen

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            epoch_loss += loss.item()
            last_mean_g = float(mean_g.detach().cpu().item())

            # Track for epoch-level mean
            running_sum_g += float(g.sum().detach().cpu().item())
            running_count += int(g.numel())

        # Log epoch progress only for final epoch
        if hasattr(cfg, 'logger') and cfg.logger and epoch == cfg.epochs - 1:
            cfg.logger.info(f"[Gate] Final epoch: loss={epoch_loss:.4f}, mean_g={last_mean_g:.4f}")

    epoch_mean_g = (running_sum_g / max(1, running_count)) if running_count else last_mean_g
    return model, float(epoch_mean_g)


@torch.no_grad()
def gate_forward_batched(model: GateMLP, X: np.ndarray, batch_size: int = 65536, device: str = "cpu") -> np.ndarray:
    """
    Returns abstain logits z_a for X in batches (shape [N, 1]).
    """
    model = model.to(device)
    model.eval()
    N = X.shape[0]
    out = np.empty((N, 1), dtype=np.float32)
    for start in range(0, N, batch_size):
        sl = slice(start, min(start + batch_size, N))
        xb = _to_tensor(_sanitize_array(X[sl]), device)
        z = model(xb).detach().cpu().numpy().astype(np.float32)
        out[sl] = z
    return out
