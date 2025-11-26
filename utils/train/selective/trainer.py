# utils/train/selective/trainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .gate import GateMLP, GateTrainingConfig, fit_gate, gate_forward_batched
from .loss import binary_ce_from_proba


@dataclass
class SelectiveConfig:
    coverage_target: float = 0.12
    lambda_init: float = 1.0
    lambda_step: float = 0.5
    outer_rounds: int = 2

    # gate sub-config
    hidden_units: tuple[int, ...] = (64,)
    dropout: float = 0.0
    epochs: int = 3
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"

    # decision
    use_argmax_abstain: bool = True
    abstain_temp: float = 1.0


def _predict_pos_proba(estimator, X: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    """
    Generic positive-class probability extraction for XGBmodels.
    Falls back to decision_function + sigmoid if predict_proba missing.
    """
    N = X.shape[0]
    out = np.empty(N, dtype=np.float32)
    for start in range(0, N, batch_size):
        sl = slice(start, min(start + batch_size, N))
        xb = X[sl]
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(xb)
            if proba.ndim == 2 and proba.shape[1] == 2:
                out[sl] = proba[:, 1]
            elif proba.ndim == 1:  # some libs return 1-col for binary (pos prob)
                out[sl] = proba
            else:
                # If multiclass, assume column 1 is positive for binary-style labels
                out[sl] = proba[:, 1]
        elif hasattr(estimator, "decision_function"):
            logit = estimator.decision_function(xb).astype(np.float32)
            out[sl] = 1 / (1 + np.exp(-logit))
        else:
            # Last resort: predict labels; convert to pseudo-proba
            pred = estimator.predict(xb)
            out[sl] = pred.astype(np.float32)
    return out


def _safe_fit_base(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    X_es: np.ndarray | None,
    y_es: np.ndarray | None,
    sample_weight_es: np.ndarray | None = None,
    logger=None,
) -> Any:
    """
    Fits the base estimator. Tries to use eval_set if supported; otherwise falls back
    to a simple fit. Sample weights are passed when possible.
    """
    fit_kwargs: Dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    # Try to use eval_set for early stopping where available
    eval_set_supported = False
    if X_es is not None and y_es is not None:
        try:
            eval_set_kwargs = {"eval_set": [(X_es, y_es)]}
            if sample_weight_es is not None:
                eval_set_kwargs["sample_weight_eval_set"] = [sample_weight_es]

            estimator.fit(
                X_train, y_train,
                **eval_set_kwargs,
                **fit_kwargs
            )
            eval_set_supported = True
        except TypeError:
            # Some estimators don't accept eval_set
            pass
        except Exception as e:
            if logger:
                logger.warning(f"Base fit with eval_set failed ({e}); retrying without eval_set.")
            eval_set_supported = False

    if not eval_set_supported:
        estimator.fit(X_train, y_train, **fit_kwargs)
    return estimator


def fit_selective(
    base_estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_es: np.ndarray | None,
    y_es: np.ndarray | None,
    selective_cfg: Dict[str, Any],
    gate_scaler=None,  # Scaler for gate inputs (StandardScaler)
    gate_feature_cols=None,  # Feature column names for gate (for inference)
    logger=None,
) -> Tuple[Any, dict]:
    """
    Orchestrates the EM-like alternation:
      1) Fit base f with weights = g(x) (start with g=1.0 if untrained)
      2) Compute per-sample CE on train
      3) Fit gate g to minimize selective objective (risk on accepted + coverage penalty)
      4) Update lambda via dual ascent
    Repeat outer_rounds times.

    Args:
        gate_scaler: StandardScaler fitted on gate features (applied to gate inputs)
        gate_feature_cols: List of feature column names for gate (for inference metadata)

    Returns: (wrapped SelectiveClassifier instance, diagnostics dict)
    """
    from .wrapper import SelectiveClassifier  # local import to avoid cycles

    # Ensure required keys exist with defaults
    gate_cfg = selective_cfg.get("gate", {})
    decision_cfg = selective_cfg.get("decision", {})

    cfg = SelectiveConfig(
        coverage_target=selective_cfg.get("coverage_target", 0.15),
        lambda_init=selective_cfg.get("lambda_init", 0.01),
        lambda_step=selective_cfg.get("lambda_step", 0.1),
        outer_rounds=selective_cfg.get("outer_rounds", 3),
        hidden_units=tuple(gate_cfg.get("hidden_units", [64])),
        dropout=float(gate_cfg.get("dropout", 0.0)),
        epochs=int(gate_cfg.get("epochs", 4)),
        batch_size=int(gate_cfg.get("batch_size", 8192)),
        lr=float(gate_cfg.get("lr", 1e-3)),
        weight_decay=float(gate_cfg.get("l2", 0.0)),
        device=selective_cfg.get("device", "cpu"),
        use_argmax_abstain=bool(decision_cfg.get("use_argmax_abstain",
                                                  decision_cfg.get("mode") == "argmax")),
    )

    # Determine gate input matrix by selecting gate_feature_cols if provided
    # X_train may be a NumPy array or a pandas DataFrame
    N = X_train.shape[0]
    def _to_numpy_gate_view(X):
        try:
            # pandas path with column selection
            if gate_feature_cols is not None and hasattr(X, 'loc'):
                return X[gate_feature_cols].to_numpy(dtype=np.float32)
        except Exception:
            pass
        # fallback: assume X is already the correct gate feature subset
        return np.asarray(X, dtype=np.float32)

    X_gate_train = _to_numpy_gate_view(X_train)
    X_gate_es = _to_numpy_gate_view(X_es) if X_es is not None else None

    in_dim = X_gate_train.shape[1]
    gate = GateMLP(in_dim=in_dim, hidden=cfg.hidden_units, dropout=cfg.dropout)
    # Initialize gate bias to target coverage for better starting point
    gate._init_last_bias_to_coverage(cfg.coverage_target)
    lambda_val = float(cfg.lambda_init)

    mean_g_history: list[float] = []

    # CRITICAL FIX: Use the same preprocessing for gate during training on the filtered subset
    if gate_scaler is not None:
        Z_train = gate_scaler.transform(X_gate_train)
        Z_es = gate_scaler.transform(X_gate_es) if X_gate_es is not None else None
    else:
        Z_train, Z_es = X_gate_train, X_gate_es

    # Start with uniform acceptance weights
    weights = np.ones(N, dtype=np.float32)

    # Suppress verbose logging during hyperparameter tuning

    for r in range(max(1, cfg.outer_rounds)):
        # 1) Fit base f with current weights
        # Compute eval set weights using current gate
        weights_es = None
        if Z_es is not None:
            z_a_es = gate_forward_batched(gate, Z_es, device=cfg.device)
            g_es = 0.5 * (1.0 + np.tanh(z_a_es / 2.0))  # same stable sigmoid
            weights_es = g_es.astype(np.float32).reshape(-1)
            weights_es = np.nan_to_num(weights_es, nan=1e-6, posinf=1.0, neginf=1e-6)
            weights_es = np.clip(weights_es, 1e-2, 1.0)
            weights_es /= max(1e-8, weights_es.mean())

        _safe_fit_base(base_estimator, X_train, y_train, weights, X_es, y_es, sample_weight_es=weights_es, logger=None)  # Disable verbose logging

        # 2) Compute per-sample CE (honest, on train set with current base)
        p_pos = _predict_pos_proba(base_estimator, X_train)
        # guard against any stray NaNs from model outputs
        p_pos = np.nan_to_num(p_pos, nan=0.5, posinf=1.0, neginf=0.0)
        per_ce = binary_ce_from_proba(p_pos, y_train)  # shape [N]
        per_ce = np.nan_to_num(per_ce, nan=50.0, posinf=50.0, neginf=0.0)
        # Clip extreme CE values that might destabilize gate training
        per_ce = np.clip(per_ce, 0.0, 10.0)

        # 3) Fit gate on selective objective with inner loop for better coverage convergence
        gate_cfg = GateTrainingConfig(
            coverage_target=cfg.coverage_target,
            lambda_val=lambda_val,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            device=cfg.device,
        )
        # Disable verbose gate logging by not setting logger

        # Inner loop: Multiple gate updates per base model fit for better coverage convergence
        inner_updates = 3  # 2-3 quick inner updates
        prev_mean_g = None
        for inner_iter in range(inner_updates):
            gate, mean_g_epoch = fit_gate(gate, Z_train, per_ce, gate_cfg)
            # Optional early stop if gate coverage stabilizes
            if prev_mean_g is not None and abs(mean_g_epoch - prev_mean_g) < 1e-3:
                break
            prev_mean_g = mean_g_epoch
            # defer lambda update until we measure coverage on the FULL set

        # Measure full-dataset mean_g, then update lambda ONCE
        z_a_full = gate_forward_batched(gate, Z_train, device=cfg.device)  # [N,1]
        # numerically stable sigmoid
        g_full = 0.5 * (1.0 + np.tanh(z_a_full / 2.0))
        mean_g_full = float(np.mean(g_full))
        lambda_val = max(0.0, lambda_val + cfg.lambda_step * (cfg.coverage_target - mean_g_full))
        mean_g_history.append(mean_g_full)

        # Log gate stats for debugging (use preprocessed inputs)
        if logger:
            z = gate_forward_batched(gate, Z_train, device=cfg.device).reshape(-1)
            m = np.mean(z); s = np.std(z)
            p = np.percentile(z, [0, 5, 25, 50, 75, 95, 100])
            logger.info(f"[Selective] round={r}  z_a(mean={m:.2f}, std={s:.2f}, p5={p[1]:.2f}, p50={p[3]:.2f}, p95={p[5]:.2f})")
            logger.info(f"[Selective] round={r}  mean_g_full={mean_g_full:.3f}  lambda(before)={lambda_val:.3f}")

        # Prepare weights for the next round (detach g(x))
        g = g_full                                                    # reuse what we just computed
        weights = g.astype(np.float32).reshape(-1)

        # Ensure weights are positive and valid (XGBoost requirement)
        # Handle NaN, infinite, and zero/negative weights due to numerical precision
        weights = np.nan_to_num(weights, nan=1e-6, posinf=1.0, neginf=1e-6)
        weights = np.clip(weights, 1e-2, 1.0)       # raise the floor
        weights /= max(1e-8, weights.mean())        # normalize to mean ~1 so learning rate is sane

        # Final validation check
        if not np.all(weights > 0):
            weights = np.maximum(weights, 1e-6)

    # Wrap for inference (argmax over pos/neg/abstain logits)
    decision_mode = "argmax" if cfg.use_argmax_abstain else "gate_first"
    wrapper = SelectiveClassifier(
        base_estimator=base_estimator,
        gate_model=gate,
        gate_scaler=gate_scaler,  # CRITICAL: Pass gate preprocessing
        gate_feature_cols=gate_feature_cols,  # CRITICAL: Pass feature list
        decision_mode=decision_mode,
        device=cfg.device,
    )
    # set temperature for abstain logit if present
    setattr(wrapper, "_abstain_temp", float(getattr(cfg, "abstain_temp", 1.0)))
    diag = {
        "coverage_target": cfg.coverage_target,
        "lambda_final": lambda_val,
        "mean_g_per_round": mean_g_history,
        "outer_rounds": cfg.outer_rounds,
    }

    if logger:
        logger.info(f"[Selective] Training complete: final_coverage≈{mean_g_history[-1]:.3f}, lambda_final={lambda_val:.3f}")

    return wrapper, diag
