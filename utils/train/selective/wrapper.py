# utils/train/selective/wrapper.py
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .gate import GateMLP, gate_forward_batched


_EPS = 1e-9


def _logit_from_proba(p: np.ndarray) -> np.ndarray:
    # Prevent RuntimeWarning: divide by zero encountered in log
    _CLIP = 1e-6
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    p = np.clip(p.astype(np.float64, copy=False), _CLIP, 1.0 - _CLIP)
    return np.log(p) - np.log(1 - p)


def _predict_pos_proba(estimator, X: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    N = X.shape[0]
    out = np.empty(N, dtype=np.float32)
    for start in range(0, N, batch_size):
        sl = slice(start, min(start + batch_size, N))
        xb = X[sl]
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(xb)
            if proba.ndim == 2 and proba.shape[1] == 2:
                out[sl] = proba[:, 1]
            elif proba.ndim == 1:
                out[sl] = proba
            else:
                out[sl] = proba[:, 1]
        elif hasattr(estimator, "decision_function"):
            logit = estimator.decision_function(xb).astype(np.float32)
            out[sl] = 1.0 / (1.0 + np.exp(-logit))
        else:
            pred = estimator.predict(xb)
            out[sl] = pred.astype(np.float32)
    return out


class SelectiveClassifier:
    """
    Composite model:
      - base_estimator: any sklearn-like binary classifier
      - gate_model: GateMLP that emits abstain logit z_a(x)
    Decision modes:
      - gate_first: accept iff z_gate < 0, then use base prediction
      - argmax: compete abstain logit vs pos/neg logits
    """
    def __init__(self, base_estimator: Any, gate_model: GateMLP, gate_scaler=None, gate_feature_cols=None, decision_mode: str = "gate_first", device: str = "cpu"):
        self.base_estimator = base_estimator
        self.gate_model = gate_model
        self.gate_scaler = gate_scaler  # StandardScaler for gate inputs
        self.gate_feature_cols = gate_feature_cols  # Feature column names for gate
        self.decision_mode = decision_mode  # "gate_first" or "argmax"
        self.device = device

    # Fit is coordinated by trainer.fit_selective, but we expose the API for symmetry.
    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Use utils/train/selective/trainer.fit_selective to train the SelectiveClassifier."
        )

    def _compute_logits(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (z_pos, z_neg, z_abstain) each of shape [N,1]
        """
        p_pos = _predict_pos_proba(self.base_estimator, X)  # [N]
        z_pos = _logit_from_proba(p_pos).reshape(-1, 1).astype(np.float32)
        z_neg = -z_pos  # logit(1-p) = -logit(p)

        # CRITICAL FIX: Apply gate preprocessing before forwarding through gate model
        if hasattr(self, "gate_feature_cols") and self.gate_feature_cols is not None:
            # Select gate features (assuming X is a DataFrame or has column names)
            if hasattr(X, 'loc') or hasattr(X, 'iloc'):
                try:
                    Z = X[self.gate_feature_cols].to_numpy(dtype=np.float32)
                except (KeyError, AttributeError):
                    # Fallback: assume X is already numpy array with correct features
                    Z = X.astype(np.float32) if X.dtype != np.float32 else X
            else:
                # X is numpy array - assume features are already in correct order
                Z = X.astype(np.float32) if X.dtype != np.float32 else X
        else:
            # No feature selection specified - use all features
            Z = X.astype(np.float32) if hasattr(X, 'astype') else np.array(X, dtype=np.float32)

        # Apply gate scaling if available
        if hasattr(self, "gate_scaler") and self.gate_scaler is not None:
            Z = self.gate_scaler.transform(Z)

        # Gate emits acceptance logit z_a; abstain logit must be the negative of it
        z_a = gate_forward_batched(self.gate_model, Z, device=self.device)  # [N,1] acceptance logit
        temp = getattr(self, "_abstain_temp", 1.0)

        # Apply temperature scaling: scale abstain logit more aggressively
        # This makes abstain more competitive at higher temperatures
        z_pos = (z_pos * float(temp)).astype(np.float32)
        z_neg = (z_neg * float(temp)).astype(np.float32)
        # Scale abstain by temp^2 to make it more competitive
        z_abstain = (-z_a * float(temp) * float(temp)).astype(np.float32)  # [N,1]
        return z_pos, z_neg, z_abstain

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns labels in {-1, 0, 1} where:
         -1 = ABSTAIN, 0 = NEG, 1 = POS

        Decision modes:
         - gate_first: accept iff z_gate > 0 (positive acceptance logit), then use base prediction
         - argmax: compete abstain logit vs pos/neg logits
        """
        if self.decision_mode == "gate_first":
            # Gate-first decision: accept iff gate_logit > 0 (positive acceptance logit)
            # Use the same preprocessing as _compute_logits
            Z = self._preprocess_gate_input(X)
            z_gate = gate_forward_batched(self.gate_model, Z, device=self.device)  # [N,1]
            accept = (z_gate > 0).flatten()  # [N] boolean mask

            out = np.full(X.shape[0], -1, dtype=np.int8)  # Default to abstain

            if np.any(accept):
                # For accepted samples, use base model prediction
                base_pred = self.base_estimator.predict(X[accept])
                out[accept] = base_pred.astype(np.int8)

            return out

        else:  # argmax mode (legacy)
            z_pos, z_neg, z_abs = self._compute_logits(X)
            logits = np.concatenate([z_pos, z_neg, z_abs], axis=1)  # [N,3]
            idx = np.argmax(logits, axis=1)
            # map: 0->pos(1), 1->neg(0), 2->abstain(-1)
            out = np.empty_like(idx, dtype=np.int8)
            out[idx == 0] = 1
            out[idx == 1] = 0
            out[idx == 2] = -1
            return out

    def _preprocess_gate_input(self, X) -> np.ndarray:
        """Apply the same preprocessing as _compute_logits for consistent gate inputs."""
        # select gate_feature_cols and apply gate_scaler (exactly as in _compute_logits)
        if hasattr(self, "gate_feature_cols") and self.gate_feature_cols is not None:
            if hasattr(X, 'loc') or hasattr(X, 'iloc'):
                try:
                    Z = X[self.gate_feature_cols].to_numpy(dtype=np.float32)
                except (KeyError, AttributeError):
                    Z = X.to_numpy(dtype=np.float32) if hasattr(X, 'to_numpy') else np.array(X, dtype=np.float32)
            else:
                Z = X.astype(np.float32) if hasattr(X, 'astype') else np.array(X, dtype=np.float32)
        else:
            Z = X.to_numpy(dtype=np.float32) if hasattr(X, 'to_numpy') else np.array(X, dtype=np.float32)
        if hasattr(self, "gate_scaler") and self.gate_scaler is not None:
            Z = self.gate_scaler.transform(Z)
        return Z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Softmax over [z_pos, z_neg, z_abstain] for auditing/analysis.
        Columns order: [p_pos, p_neg, p_abstain]
        """
        z_pos, z_neg, z_abs = self._compute_logits(X)
        logits = np.concatenate([z_pos, z_neg, z_abs], axis=1).astype(np.float32)  # [N,3]
        # stable softmax
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        p = e / e.sum(axis=1, keepdims=True)
        return p.astype(np.float32)

    def calibrate_abstain_temp(self, X_val, c_target, tol=5e-3):
        """
        Calibrate abstain temperature T so argmax-coverage ≈ target coverage c_target.
        Uses binary search on T to achieve desired coverage on validation split.

        Args:
            X_val: Validation features
            c_target: Target coverage rate (0-1)
            tol: Tolerance for coverage difference

        Returns:
            (T_star, achieved_coverage)
        """
        lo, hi = 0.25, 8.0
        best_T, best_cov = 1.0, 0.0

        for _ in range(20):
            T = 0.5 * (lo + hi)
            setattr(self, "_abstain_temp", float(T))
            y_hat = self.predict(X_val)
            cov = float((y_hat != -1).mean())

            if cov > c_target + tol:
                # Too many accepted → make abstain stronger → increase T
                lo = T
            elif cov < c_target - tol:
                # Too few accepted → make abstain weaker → decrease T
                hi = T
            else:
                # Within tolerance
                return float(T), cov

            # Track best attempt
            if abs(cov - c_target) < abs(best_cov - c_target):
                best_T, best_cov = T, cov

        # Return best attempt if we didn't converge
        return float(best_T), best_cov

    def calibrate_gate_threshold(self, X_val, c_target):
        """
        Calibrate gate threshold τ on validation set to hit target coverage c_target.
        For gate_first mode: accept iff z_a > τ
        """
        from .gate import gate_forward_batched
        import numpy as np

        # Use the same preprocessing as _compute_logits
        Z = self._preprocess_gate_input(X_val)
        z = gate_forward_batched(self.gate_model, Z, device=self.device).reshape(-1)
        tau = float(np.quantile(z, 1.0 - float(c_target)))  # accept iff z > tau
        cov = float((z > tau).mean())
        setattr(self, "_gate_threshold", tau)
        return tau, cov

    # Optional convenience for persistence (you can integrate with your persist module)
    def get_state(self) -> dict:
        return {
            "base_estimator": self.base_estimator,
            "gate_state_dict": self.gate_model.state_dict(),
            "gate_arch": {"in_dim": getattr(getattr(self.gate_model, "net", None)[0], "in_features", None)},
            "decision_mode": self.decision_mode,
            "device": self.device,
            "_abstain_temp": getattr(self, "_abstain_temp", 1.0),
        }
