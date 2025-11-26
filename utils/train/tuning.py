# utils/train/tuning.py
from __future__ import annotations
import os, copy, inspect, logging
import numpy as np
import optuna
import xgboost as xgb
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, precision_score
from sklearn.model_selection import TimeSeriesSplit
from utils.train.purged_group_time_series_split import PurgedGroupTimeSeriesSplit

logger = logging.getLogger(__name__)

# ---- internal helpers (decoupled from ModelTrainer) ----
def _xgb_supports_callbacks() -> bool:
    try:
        return 'callbacks' in inspect.signature(xgb.XGBClassifier.fit).parameters
    except Exception:
        return False

def _compute_patience(n_estimators: int) -> int:
    return max(20, min(100, int(0.20 * int(n_estimators))))

def _auto_spw(y) -> float:
    try:
        p = float(np.mean(y))
    except Exception:
        p = 0.0
    if p <= 0.0 or p >= 1.0:
        return 1.0
    return float(np.clip((1.0 - p) / p, 0.25, 50.0))

def _prepare_splits(X_train, y_train, groups_train):
    # Try purged CV; fallback to time series
    try:
        cv = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=5)
        logger.info("Pre-computing CV splits with PurgedGroupTimeSeriesSplit...")
        return list(cv.split(X_train, y_train, groups=groups_train))
    except ValueError as e:
        logger.warning(f"PurgedGroupTimeSeriesSplit failed with error: {e}. Falling back to standard TimeSeriesSplit.")
        n_splits = max(2, min(5, len(X_train) // 10))
        if n_splits < 2:
            raise ValueError("Dataset is too small for cross-validation.")
        cv_fallback = TimeSeriesSplit(n_splits=n_splits)
        return list(cv_fallback.split(X_train, y_train, groups=groups_train))

def _suggest_xgb_params(trial: optuna.Trial, base_params: Dict, param_cfg: Dict, task_type: str, is_rare: bool):
    # Match original behavior: create clean_base_params first
    clean_base_params = copy.deepcopy(base_params)
    for k in ['gamma', 'grow_policy', 'max_leaves', 'scale_pos_weight', 'enable_categorical']:
        clean_base_params.pop(k, None)

    params = {
        **clean_base_params,  # Start with clean base params like original
        "objective": ("reg:squarederror" if task_type == "regression" else "binary:logistic"),
        "eval_metric": ("rmse" if task_type == "regression" else ("aucpr" if is_rare else "auc")),
        "enable_categorical": True,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
        "tree_method": "hist",
    }

    # Define parameter configurations with their types and centering factors
    param_specs = {
        'n_estimators': ('int', 0.5, 1.5),
        'learning_rate': ('float_log', 0.5, 2.0),
        'min_child_weight': ('int', 0.5, 1.5),
        'subsample': ('float', 0.5, 1.5),
        'colsample_bytree': ('float', 0.5, 1.5),
    }

    if not param_cfg:
        logger.warning("No hyperparameter distributions configured; using base XGBoost parameters.")
        return params

    for param_name, cfg in param_cfg.items():
        if "choices" in cfg:
            params[param_name] = trial.suggest_categorical(param_name, cfg["choices"])
            continue

        if "low" not in cfg or "high" not in cfg:
            raise ValueError(
                f"Hyperparameter '{param_name}' must define 'low' and 'high' in config/training.yaml."
            )

        cfg_low = cfg["low"]
        cfg_high = cfg["high"]
        if cfg_low >= cfg_high:
            raise ValueError(
                f"Hyperparameter '{param_name}' has invalid range [{cfg_low}, {cfg_high}]. "
                "Ensure 'low' < 'high' in config/training.yaml."
            )

        param_type, center_low_mult, center_high_mult = param_specs.get(
            param_name, ("float", 0.5, 1.5)
        )

        base_value = clean_base_params.get(param_name)
        # Infer type from base value when not explicitly known
        if param_type == "float" and isinstance(base_value, int) and not isinstance(base_value, bool):
            param_type = "int"

        if base_value is None or not (cfg_low <= base_value <= cfg_high):
            # Use full range when base value not provided or outside range
            if param_type == "int":
                params[param_name] = trial.suggest_int(param_name, int(cfg_low), int(cfg_high))
            elif param_type == "float_log" or cfg.get("log"):
                params[param_name] = trial.suggest_float(param_name, cfg_low, cfg_high, log=True)
            else:
                params[param_name] = trial.suggest_float(param_name, cfg_low, cfg_high)
            continue

        # Center search window around the base value
        window_low = max(cfg_low, base_value * center_low_mult)
        window_high = min(cfg_high, base_value * center_high_mult)
        if window_low >= window_high:
            window_low, window_high = cfg_low, cfg_high

        if param_type == "int":
            params[param_name] = trial.suggest_int(param_name, int(window_low), int(window_high))
        elif param_type == "float_log" or cfg.get("log"):
            params[param_name] = trial.suggest_float(param_name, window_low, window_high, log=True)
        else:
            params[param_name] = trial.suggest_float(param_name, window_low, window_high)

    # Handle grow_policy - prefer base value if it exists
    grow_policy_cfg = param_cfg.get("grow_policy", {}).get("choices", ["lossguide", "depthwise"])
    base_grow_policy = clean_base_params.get('grow_policy', grow_policy_cfg[0])
    if base_grow_policy in grow_policy_cfg:
        grow_policy = base_grow_policy
    else:
        grow_policy = trial.suggest_categorical("grow_policy", grow_policy_cfg)
    params["grow_policy"] = grow_policy

    # Handle max_leaves/max_depth based on grow_policy
    if grow_policy == "lossguide":
        _handle_max_leaves_or_depth(trial, params, param_cfg, clean_base_params, 'max_leaves')
        params.pop("max_depth", None)
    else:
        _handle_max_leaves_or_depth(trial, params, param_cfg, clean_base_params, 'max_depth')
        params.pop("max_leaves", None)

    # Ensure regularization params exist even if not tuned explicitly
    if "reg_alpha" not in params:
        params["reg_alpha"] = float(clean_base_params.get("reg_alpha", 0.0))
    if "reg_lambda" not in params:
        params["reg_lambda"] = float(clean_base_params.get("reg_lambda", 1.0))

    # Optional log scaling (kept compatible with your YAML) - CRITICAL FOR PERFORMANCE
    if param_cfg.get('reg_alpha', {}).get('log', True):
        params["reg_alpha"] = float(params["reg_alpha"])
    if param_cfg.get('reg_lambda', {}).get('log', True):
        params["reg_lambda"] = float(params["reg_lambda"])

    return params


def _handle_max_leaves_or_depth(trial: optuna.Trial, params: Dict, param_cfg: Dict, clean_base_params: Dict, param_name: str):
    """Helper function to handle max_leaves or max_depth with centering logic."""
    cfg = param_cfg.get(param_name, {'low': 128, 'high': 1024})
    base_value = clean_base_params.get(param_name, cfg['low'])

    if cfg['low'] <= base_value <= cfg['high']:
        # Base value is within range, center around it
        low = max(cfg['low'], int(base_value * 0.5))
        high = min(cfg['high'], int(base_value * 1.5))
        params[param_name] = trial.suggest_int(param_name, low, high)
    else:
        # Use full range
        params[param_name] = trial.suggest_int(param_name, cfg['low'], cfg['high'])

# ---- public API ----
def tune_hyperparameters(trainer, model_name: str, problem_config: Dict, X_train, y_train, groups_train, sample_weight=None):
    """Drop-in replacement for ModelTrainer.tune_hyperparameters; writes into trainer.best_params & meta."""
    task_type = problem_config.get("task_type", "classification").lower()
    direction = "minimize" if task_type == "regression" else "maximize"

    logger.info(f"Starting hyperparameter tuning for {model_name} with Optuna...")

    # splits (cached per (problem, features) like before)
    problem_name = problem_config['name']
    feature_set_key = tuple(X_train.columns)
    cache_key = (problem_name, feature_set_key)
    if cache_key in trainer._cached_splits:
        splits = trainer._cached_splits[cache_key]
        logger.info(f"Re-using cached CV splits for {problem_name}.")
    else:
        logger.info(f"Pre-computing CV splits for {problem_name}...")
        splits = _prepare_splits(X_train, y_train, groups_train)
        trainer._cached_splits[cache_key] = splits
        logger.info(f"CV splits cached ({len(splits)} splits).")

    # parallel config
    ht_cfg = trainer.cfg.hyperparameter_tuning
    cpu_count = os.cpu_count() or 4
    n_jobs = max(1, cpu_count // 2) if ht_cfg.get("parallel", True) else 1

    if n_jobs > 1:
        logger.info(f"Parallel tuning enabled. Using a config copy to force sub-model n_jobs=1.")

    # rarity (classification only)
    is_rare = False
    try:
        is_rare = (float(np.mean(y_train)) < 0.07) if task_type == 'classification' else False
    except Exception:
        is_rare = False

    # objective
    supports_callbacks = _xgb_supports_callbacks()

    if sample_weight is not None:
        sample_weight = sample_weight.reindex(X_train.index).astype(np.float32, copy=False)

    def objective(trial: optuna.Trial):
        if model_name not in ("xgboost", "xgb_selective"):
            raise ValueError("Only xgboost and xgb_selective supported in tuning module.")

        is_selective = (model_name == "xgb_selective")

        if is_selective and task_type == 'classification':
            # Following user's playbook: joint optimization of base + gate + selective knobs

            # --- Sample hyperparams per user's playbook ---
            xgb_params = {
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
                "n_estimators": 10000,  # use early stopping
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_jobs": 1,
                "random_state": 42,
                "verbosity": 0,
            }

            # Get selective config for this problem
            problem_selective = problem_config.get('selective', {})
            global_selective = trainer.cfg.selective

            # Convert dataclass to dict and merge with problem-specific config
            if hasattr(global_selective, '__dict__'):
                global_selective_dict = global_selective.__dict__
            else:
                global_selective_dict = dict(global_selective)

            # Start with merged config
            sel_cfg = {**global_selective_dict, **problem_selective}

            # Override with tuned gate and selective parameters
            tuned_lambda_init = trial.suggest_float("lambda_init", 0.5, 6.0)  # Stronger range
            tuned_lambda_step = trial.suggest_float("lambda_step", 0.3, 1.5)  # Stronger range

            sel_cfg.update({
                "lambda_init": tuned_lambda_init,
                "lambda_step": tuned_lambda_step,
                "outer_rounds": trial.suggest_int("outer_rounds", 3, 4),  # More rounds
                "gate": {
                    "hidden_units": [trial.suggest_categorical("gate_hidden", [32, 64, 128])],
                    "dropout": trial.suggest_float("gate_dropout", 0.0, 0.2),
                    "epochs": trial.suggest_int("gate_epochs", 3, 5),
                    "batch_size": trial.suggest_categorical("gate_bs", [4096, 8192, 16384]),
                    "lr": trial.suggest_float("gate_lr", 1e-4, 3e-3, log=True),
                    "l2": trial.suggest_float("gate_weight_decay", 1e-6, 1e-3, log=True),
                },
                "decision": {"use_argmax_abstain": True},
            })

            # Evaluate on all CV folds
            fold_scores = []
            fold_coverages = []
            fold_precisions = []

            for i, (tr_idx, va_idx) in enumerate(splits):
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
                y_tr_np = y_tr.to_numpy(); y_va_np = y_va.to_numpy()
                spw = _auto_spw(y_tr_np)

                try:
                    from utils.train.selective.trainer import fit_selective

                    # Create base model with trial params
                    base = xgb.XGBClassifier(**xgb_params, scale_pos_weight=spw)

                    # Convert to numpy for selective training
                    X_tr_np = X_tr.to_numpy() if hasattr(X_tr, 'to_numpy') else X_tr
                    X_va_np = X_va.to_numpy() if hasattr(X_va, 'to_numpy') else X_va

                    # Train selective model
                    model, diag = fit_selective(
                        base_estimator=base,
                        X_train=X_tr_np,
                        y_train=y_tr_np,
                        X_es=None,  # Skip early stopping for tuning speed
                        y_es=None,
                        selective_cfg=sel_cfg,
                        logger=None  # Disable verbose logging
                    )

                    # Calibrate abstain temperature for consistent coverage
                    target_coverage = sel_cfg["coverage_target"]
                    T_star, calibrated_coverage = model.calibrate_abstain_temp(X_va_np, target_coverage)
                    logger.info(f"Trial {trial.number} fold {i}: Calibrated T={T_star:.3f} to hit coverage≈{calibrated_coverage:.4f} (target={target_coverage:.4f})")

                    # Evaluate selective model performance
                    y_pred = model.predict(X_va_np)
                    accepted_mask = (y_pred != -1)
                    if np.any(accepted_mask):
                        y_test_acc = y_va_np[accepted_mask]
                        y_pred_acc = y_pred[accepted_mask]
                        if len(y_test_acc) > 0:
                            from sklearn.metrics import precision_score, recall_score
                            precision = precision_score(y_test_acc, y_pred_acc, zero_division=0)
                            recall = recall_score(y_test_acc, y_pred_acc, zero_division=0)
                            acceptance_rate = accepted_mask.mean()

                            logger.info(f"Trial {trial.number} fold {i}: Accepted {accepted_mask.sum()}/{len(y_pred)} ({acceptance_rate:.3f}); pred_pos {y_pred_acc.sum()} (rate {(y_pred_acc.sum()/len(y_pred_acc)):.3f}), TP={((y_test_acc == 1) & (y_pred_acc == 1)).sum()}, FP={((y_test_acc == 0) & (y_pred_acc == 1)).sum()}, precision_sel={precision:.4f}, recall_on_accepted={recall:.4f}")
                            logger.info(f"Trial {trial.number} fold {i}: penalty = {sel_cfg['lambda_step'] * abs(target_coverage - calibrated_coverage):.4f}, score = {precision:.4f}")
                        else:
                            logger.info(f"Trial {trial.number} fold {i}: No samples accepted")
                    else:
                        logger.info(f"Trial {trial.number} fold {i}: No samples accepted")

                    # Performance evaluation already done above

                    if accepted_mask.sum() == 0:
                        # Reject trials that accept nothing
                        fold_scores.append(-1e9)
                        fold_coverages.append(0.0)
                        fold_precisions.append(0.0)
                    else:
                        # Use the precision calculation from above
                        fold_scores.append(precision)
                        fold_coverages.append(acceptance_rate)
                        fold_precisions.append(precision)

                        # Business objective: selective precision with soft penalty for coverage deviation
                        target_coverage = sel_cfg["coverage_target"]
                        coverage_penalty = sel_cfg['lambda_step'] * abs(acceptance_rate - target_coverage)
                        business_score = precision - coverage_penalty

                        logger.info(f"Trial {trial.number} fold {i}: penalty = {coverage_penalty:.4f}, score = {business_score:.4f}")

                        # fold_scores, fold_coverages, fold_precisions already appended above

                except Exception as e:
                    logger.warning(f"Trial {trial.number} fold {i} failed: {e}")
                    fold_scores.append(-1e9)
                    fold_coverages.append(0.0)
                    fold_precisions.append(0.0)

            # Aggregate across folds
            if len(fold_scores) == 0 or all(s == -1e9 for s in fold_scores):
                final_score = -1e9
                mean_coverage = 0.0
                mean_precision = 0.0
            else:
                # Use mean business score across folds
                valid_scores = [s for s in fold_scores if s != -1e9]
                final_score = float(np.mean(valid_scores))

                # Also track coverage and precision for analysis
                valid_coverages = [c for c, s in zip(fold_coverages, fold_scores) if s != -1e9]
                valid_precisions = [p for p, s in zip(fold_precisions, fold_scores) if s != -1e9]
                mean_coverage = float(np.mean(valid_coverages)) if valid_coverages else 0.0
                mean_precision = float(np.mean(valid_precisions)) if valid_precisions else 0.0

            # Log attrs for analysis
            trial.set_user_attr("coverage", mean_coverage)
            trial.set_user_attr("precision_sel", mean_precision)
            trial.set_user_attr("target_coverage", sel_cfg["coverage_target"])

            logger.info(f"Trial {trial.number}: final_score = {final_score:.4f}, mean_coverage = {mean_coverage:.4f}, mean_precision = {mean_precision:.4f}")
            return final_score

        else:
            # Regular XGBoost (non-selective) - use original parameter structure
            base_yaml = copy.deepcopy(problem_config.get('xgboost_params', {}))
            param_cfg = ht_cfg['param_distributions']['xgboost']
            params = _suggest_xgb_params(trial, base_yaml, param_cfg, task_type, is_rare)
            patience = _compute_patience(params["n_estimators"])

            scores = []
            for i, (tr_idx, va_idx) in enumerate(splits):
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
                y_tr_np = y_tr.to_numpy(); y_va_np = y_va.to_numpy()
                spw = _auto_spw(y_tr_np) if task_type == 'classification' else 1.0
                if sample_weight is not None:
                    sw_tr = sample_weight.iloc[tr_idx].to_numpy(dtype=np.float32, copy=False)
                    sw_va = sample_weight.iloc[va_idx].to_numpy(dtype=np.float32, copy=False)
                else:
                    sw_tr = sw_va = None

                # Regular XGBoost training (original behavior)
                Model = xgb.XGBRegressor if task_type == 'regression' else xgb.XGBClassifier
                eval_sets = [(X_va, y_va_np)] if task_type == 'regression' else [(X_tr, y_tr_np), (X_va, y_va_np)]
                if supports_callbacks:
                    model = Model(**params, callbacks=[], )
                    try:
                        if task_type == 'classification':
                            model.set_params(scale_pos_weight=spw)
                    except Exception:
                        pass
                    fit_kwargs = {
                        "eval_set": eval_sets,
                        "callbacks": [xgb.callback.EarlyStopping(rounds=patience, save_best=False)],
                        "verbose": False,
                    }
                    if sw_tr is not None:
                        fit_kwargs["sample_weight"] = sw_tr
                    model.fit(X_tr, y_tr_np, **fit_kwargs)
                else:
                    model = Model(**params, early_stopping_rounds=patience)
                    try:
                        if task_type == 'classification':
                            model.set_params(scale_pos_weight=spw)
                    except Exception:
                        pass
                    fit_kwargs = {
                        "eval_set": eval_sets,
                        "verbose": False,
                    }
                    if sw_tr is not None:
                        fit_kwargs["sample_weight"] = sw_tr
                    model.fit(X_tr, y_tr_np, **fit_kwargs)

                # pick best iteration consistently
                n_best = getattr(model, 'best_iteration', None)
                if n_best is None:
                    try:
                        n_best = model.get_booster().best_iteration
                    except Exception:
                        n_best = model.get_booster().num_boosted_rounds() - 1

                if task_type == 'regression':
                    preds = model.predict(X_va, iteration_range=(0, n_best + 1))
                    score = float(np.sqrt(mean_squared_error(y_va_np, preds)))  # RMSE
                else:
                    proba = model.predict_proba(X_va, iteration_range=(0, n_best + 1))[:, 1]
                    score = float(average_precision_score(y_va_np, proba) if is_rare else roc_auc_score(y_va_np, proba))

                scores.append(score)
                trial.report(score, i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # record variability
            if task_type == 'regression':
                trial.set_user_attr('rmse_std', float(np.std(scores)))
                return float(np.mean(scores))
            else:
                trial.set_user_attr(('ap_std' if is_rare else 'auc_std'), float(np.std(scores)))
                return float(np.mean(scores))

    # run study
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=trainer.base_seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=ht_cfg.get('pruning_warmup_steps', 1))
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    logger.info(f"Optimizing with Optuna using {n_jobs} parallel jobs...")
    study.optimize(objective, n_trials=ht_cfg.get('n_trials', 10), n_jobs=n_jobs, show_progress_bar=True, gc_after_trial=True)

    metric_name = "RMSE" if task_type == "regression" else ("AP" if is_rare else "AUC")
    logger.info(f"Best trial for {model_name} completed with {metric_name}: {study.best_value:.4f}")
    logger.info(f"Best params for {model_name}: {study.best_params}")

    best_trial = study.best_trial
    params_key = f"{problem_name}_{model_name}"
    base_yaml_params = problem_config.get(f"{model_name}_params", {})
    trainer.best_params[params_key] = {**base_yaml_params, **best_trial.params}
    trainer.best_params_meta[params_key] = best_trial.user_attrs
    return trainer.best_params[params_key]
