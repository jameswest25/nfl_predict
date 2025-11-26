# utils/train/xgb_utils.py
import copy
import warnings
import numpy as np
import xgboost as xgb
from xgboost.callback import EarlyStopping
import threadpoolctl

def fit_model_es(
    model_name,
    model,
    X_train,
    y_train,
    X_es,
    y_es,
    patience,
    task_type,
    sample_weight=None,
    eval_sample_weight=None,
):
    """_fit_model_es from train.py (supports both XGBoost and selective models)."""
    if model_name == 'xgb_selective':
        # For selective models, extract the base XGBoost estimator
        if hasattr(model, 'base_estimator'):
            base_model = model.base_estimator
            model_name = 'xgboost'  # Treat as regular XGBoost for fitting
            model = base_model
        else:
            raise ValueError("Selective model does not have base_estimator")
    elif model_name != 'xgboost':
        raise ValueError(f"fit_model_es only supports XGBoost and xgb_selective, got {model_name}")
    # Prefer callbacks path if available
    has_callbacks = False
    try:
        has_callbacks = ('callbacks' in xgb.XGBClassifier.fit.__signature__.parameters)  # py3.11 ok
    except Exception:
        has_callbacks = False

    if has_callbacks:
        callbacks = [EarlyStopping(rounds=patience, save_best=False)]
        fit_kwargs = {
            "eval_set": [(X_es, y_es)],
            "callbacks": callbacks,
            "verbose": False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X_train, y_train, **fit_kwargs)
        return model
    else:
        Estimator = xgb.XGBRegressor if task_type == 'regression' else xgb.XGBClassifier
        cloned_params = model.get_params()
        cloned_params.pop('callbacks', None)
        cloned_params.pop('early_stopping_rounds', None)
        new_model = Estimator(**cloned_params, early_stopping_rounds=patience)
        fit_kwargs = {
            "eval_set": [(X_es, y_es)],
            "verbose": False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        new_model.fit(X_train, y_train, **fit_kwargs)
        return new_model

def get_best_iteration(model, model_name):
    """_get_best_iteration from train.py (unchanged)."""
    if model_name != 'xgboost':
        return -1
    n_best = getattr(model, 'best_iteration', None)
    if n_best is None:
        try:
            n_best = model.get_booster().best_iteration
        except Exception:
            try:
                n_best = model.get_booster().num_boosted_rounds() - 1
            except Exception:
                n_best = -1
    return n_best if n_best is not None else -1

def compute_patience(n_estimators: int) -> int:
    """_compute_patience from train.py."""
    return max(20, min(100, int(0.20 * int(n_estimators))))

def predict_proba_batched(model, X, features, batch_size: int = 200_000):
    """_predict_proba_batched from train.py (unchanged)."""
    n = len(X)
    y_pred_proba = np.empty(n, dtype=np.float32)
    write_pos = 0
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        X_chunk = X.iloc[start:stop]
        X_aligned = X_chunk[features]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")
            with threadpoolctl.threadpool_limits(limits=1, user_api='openmp'):
                with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
                    proba = model.predict_proba(X_aligned)[:, 1]
        y_pred_proba[write_pos:write_pos + len(X_chunk)] = proba.astype(np.float32, copy=False)
        write_pos += len(X_chunk)
    return y_pred_proba

def retrain_with_best_iter(
    model_name, problem_config, params, best_iter,
    X_train, y_train, X_es, y_es, task_type, get_model_instance, sample_weight=None
):
    """Extracted from train.py::_retrain_with_best_iter (supports selective models)."""
    adj_params = copy.deepcopy(params)
    if best_iter is not None and best_iter >= 0:
        adj_params['n_estimators'] = best_iter + 1 if model_name == 'xgboost' else best_iter
    if task_type == 'classification':
        # scale_pos_weight handled by caller if desired
        pass

    model_instance = get_model_instance(model_name, problem_config, adj_params)

    # Handle selective models - extract the base model from the tuple
    if model_name == 'xgb_selective' and isinstance(model_instance, tuple) and model_instance[0] == "selective":
        base_model = model_instance[1]  # The base XGBoost model
        fit_kwargs = {"verbose": False}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        base_model.fit(X_train, y_train, **fit_kwargs)
        return base_model
    else:
        # Regular model
        model_final = model_instance
        fit_kwargs = {"verbose": False}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model_final.fit(X_train, y_train, **fit_kwargs)
        return model_final
