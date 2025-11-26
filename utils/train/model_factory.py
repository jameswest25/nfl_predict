# utils/train/model_factory.py
import numpy as np
import xgboost as xgb

def get_task_type(problem_cfg: dict) -> str:
    return str(problem_cfg.get("task_type", "classification")).lower()

def is_classification(problem_cfg: dict) -> bool:
    return get_task_type(problem_cfg) in ("classification", "binary", "multiclass")

def is_regression(problem_cfg: dict) -> bool:
    return get_task_type(problem_cfg) == "regression"

def get_model_instance(model_name, problem_config, params=None):
    """Exact behavior preserved from train.py."""
    task_type = get_task_type(problem_config)
    model_params = params or problem_config.get(f'{model_name}_params', {})
    if model_name == 'xgboost':
        model_params['n_jobs'] = 1
        model_params.setdefault('enable_categorical', True)
        model_params.setdefault('tree_method', 'hist')
        if is_regression(problem_config):
            return xgb.XGBRegressor(**model_params)
        else:
            return xgb.XGBClassifier(**model_params)
    elif model_name == 'xgb_selective':
        # For selective models, we return a tuple indicating this is a selective model
        # The training loop will handle the special case

        # Filter out selective parameters before creating XGBoost model
        selective_keys = {'lambda_init','lambda_step','outer_rounds',
                          'gate_hidden','gate_dropout','gate_epochs',
                          'gate_bs','gate_lr','gate_weight_decay'}
        xgb_params = {k: v for k, v in model_params.items() if k not in selective_keys}

        xgb_params['n_jobs'] = 1
        xgb_params.setdefault('enable_categorical', True)
        xgb_params.setdefault('tree_method', 'hist')

        if is_regression(problem_config):
            # Note: Selective models are currently only supported for classification
            raise ValueError("xgb_selective is only supported for classification tasks")
        else:
            base_model = xgb.XGBClassifier(**xgb_params)
            return ("selective", base_model)  # Special marker for selective training
    raise ValueError(f"Unsupported model: {model_name}")

def auto_spw(y):
    """_auto_spw from train.py (unchanged)."""
    try:
        p = float(np.mean(y))
    except Exception:
        p = 0.0
    if p <= 0.0 or p >= 1.0:
        return 1.0
    spw = (1.0 - p) / p
    return float(np.clip(spw, 0.25, 50.0))
