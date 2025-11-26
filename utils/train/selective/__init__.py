# utils/train/selective/__init__.py
"""Selective Risk Minimization components for MLB prediction pipeline."""

from .gate import GateMLP, fit_gate, gate_forward_batched
from .loss import selective_objective, binary_ce_from_proba
from .trainer import fit_selective, SelectiveConfig
from .wrapper import SelectiveClassifier

__all__ = [
    "GateMLP",
    "fit_gate",
    "gate_forward_batched",
    "selective_objective",
    "binary_ce_from_proba",
    "fit_selective",
    "SelectiveConfig",
    "SelectiveClassifier",
]
