# utils/train/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

def ece(y_true, y_prob, n_bins: int = 15) -> float:
    """_ece from train.py (unchanged)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece_val = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true.iloc[mask].mean() if hasattr(y_true, 'iloc') else y_true[mask].mean()
        w = mask.mean()
        ece_val += w * abs(conf - acc)
    return float(ece_val)

def confidence_bins_report(y_true, y_prob, decision_threshold: float = 0.5):
    """_calculate_confidence_metrics from train.py (unchanged)."""
    bins = (y_prob * 10).astype(int)
    bins = np.where(y_prob >= 1.0, 10, bins)
    bins = np.clip(bins, 0, 10)
    labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)] + ["1.0"]
    counts = np.bincount(bins, minlength=11)
    actual_positives = np.bincount(bins, weights=y_true, minlength=11)
    avg_confidence = np.bincount(bins, weights=y_prob, minlength=11)
    with np.errstate(divide="ignore", invalid="ignore"):
        actual_positive_rate = actual_positives / counts
        avg_confidence_rate = avg_confidence / counts
    predicted_labels = (y_prob >= decision_threshold).astype(int)
    correct_predictions = (predicted_labels == y_true).astype(int)
    accuracy_positives = np.bincount(bins, weights=correct_predictions, minlength=11)
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy_rate = accuracy_positives / counts
    report = {}
    for i, label in enumerate(labels):
        report[label] = {
            "count": int(counts[i]),
            "accuracy": float(np.nan_to_num(accuracy_rate[i])),
            "avg_confidence": float(np.nan_to_num(avg_confidence_rate[i])),
            "actual_positive_rate": float(np.nan_to_num(actual_positive_rate[i])),
        }
    return report

# Optional in Stage-1 (keeps regression block intact in train.py):
def _resolve_tail(y_test, y_pred, y_train_full, cfg):
    """Copied verbatim from train.py::_resolve_tail (unchanged)."""
    cfg = cfg or {}
    mode = str(cfg.get('mode', 'train_quantile')).lower()
    min_support = int(cfg.get('min_support', 20))
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)
    y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else np.asarray(y_pred)
    y_train_np = y_train_full.values if hasattr(y_train_full, 'values') else np.asarray(y_train_full)
    # import numpy as np  <-- Removed redundant import that caused UnboundLocalError
    if mode == 'train_quantile':
        q = float(cfg.get('value', 0.90))
        thresh = float(np.quantile(y_train_np, q)); mask = y_test_np >= thresh
    elif mode == 'quantile':
        q = float(cfg.get('value', 0.90))
        thresh = float(np.quantile(y_test_np, q)); mask = y_test_np >= thresh
    elif mode == 'pred_quantile':
        q = float(cfg.get('value', 0.90))
        thresh = float(np.quantile(y_pred_np, q)); mask = y_test_np >= thresh
    elif mode == 'iqr':
        k = float(cfg.get('k', 1.5)); q1, q3 = np.quantile(y_train_np, [0.25, 0.75])
        thresh = float(q3 + k * (q3 - q1)); mask = y_test_np >= thresh
    elif mode == 'mean_sigma':
        k = float(cfg.get('k', 2.0)); mu, sigma = float(np.mean(y_train_np)), float(np.std(y_train_np))
        thresh = mu + k * sigma; mask = y_test_np >= thresh
    else:
        thresh = float(cfg.get('value', 3)); mask = y_test_np >= thresh
    support = int(mask.sum())
    return mask, {'mode': mode, 'threshold': float(thresh), 'support': support, 'min_support': min_support}

def compute_regression_metrics(y_test, y_pred, y_train_full, tail_cfg=None):
    """Copied from train.py::_compute_regression_metrics (unchanged)."""
    import numpy as np
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    mdae = float(np.median(np.abs(y_pred - y_test)))
    bias = float((y_pred - y_test).mean())
    expl_var = float(explained_variance_score(y_test, y_pred))
    baseline_val = float(y_train_full.mean())
    baseline_pred = np.full(len(y_test), baseline_val, dtype=float)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    try:
        dec_bins = pd.qcut(y_pred, 10, labels=False, duplicates='drop')
        decile_rmse = {}; decile_mae = {}
        for d in np.unique(dec_bins):
            mask = dec_bins == d
            if mask.sum() == 0: continue
            decile_rmse[f'd{d}'] = float(np.sqrt(mean_squared_error(y_test[mask], y_pred[mask])))
            decile_mae[f'd{d}'] = float(mean_absolute_error(y_test[mask], y_pred[mask]))
    except Exception:
        decile_rmse, decile_mae = {}, {}
    tail_cfg = tail_cfg or {"mode": "train_quantile", "value": 0.90, "min_support": 20}
    tail_mask, tail_info = _resolve_tail(y_test, y_pred, y_train_full, tail_cfg)
    if tail_info['support'] >= tail_info['min_support']:
        tail_rmse = float(np.sqrt(mean_squared_error(y_test[tail_mask], y_pred[tail_mask])))
        tail_mae = float(mean_absolute_error(y_test[tail_mask], y_pred[tail_mask]))
    else:
        tail_rmse = tail_mae = float('nan')
    k = max(1, int(0.05 * len(y_test)))
    top_k_idx = np.argsort(y_pred)[-k:]
    recall_top5 = float((tail_mask[top_k_idx]).mean()) if k else float('nan')
    avg_bases_top5 = float(y_test.iloc[top_k_idx].mean()) if k else float('nan')
    lift_top5 = float(avg_bases_top5 / y_test.mean()) if k else float('nan')
    return {
        'rmse': rmse, 'mae': mae, 'mdae': mdae, 'bias': bias, 'r2': r2,
        'explained_variance': expl_var,
        'baseline_rmse': baseline_rmse, 'baseline_mae': baseline_mae,
        'decile_rmse': decile_rmse, 'decile_mae': decile_mae,
        'outlier_rmse': tail_rmse, 'outlier_mae': tail_mae,
        'outlier_mode': tail_info['mode'], 'outlier_threshold': tail_info['threshold'],
        'outlier_support': tail_info['support'],
        'recall_top5pct': recall_top5, 'lift_top5pct': lift_top5,
    }


def selective_classification_report(y_true, y_pred):
    """
    Compute selective classification metrics on accepted examples only.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predictions in {-1, 0, 1} where -1 = ABSTAIN

    Returns
    -------
    dict
        Dictionary with selective metrics:
        - coverage: acceptance rate (fraction of non-abstain predictions)
        - n_accepted: number of accepted examples
        - n_total: total number of examples
        - accuracy_sel: accuracy on accepted examples
        - precision_sel: precision on accepted examples
        - recall_sel: recall on accepted examples
        - f1_sel: F1 score on accepted examples
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Basic counts
    n_total = len(y_true)
    accepted_mask = (y_pred != -1)
    n_accepted = int(np.sum(accepted_mask))

    if n_accepted == 0:
        return {
            'coverage': 0.0,
            'n_accepted': 0,
            'n_total': n_total,
            'accuracy_sel': float('nan'),
            'precision_sel': float('nan'),
            'recall_sel': float('nan'),
            'f1_sel': float('nan'),
        }

    # Filter to accepted examples
    y_true_accepted = y_true[accepted_mask]
    y_pred_accepted = y_pred[accepted_mask]

    # Compute metrics on accepted examples
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy_sel = accuracy_score(y_true_accepted, y_pred_accepted)
    precision_sel = precision_score(y_true_accepted, y_pred_accepted, zero_division=0)
    recall_sel = recall_score(y_true_accepted, y_pred_accepted, zero_division=0)
    f1_sel = f1_score(y_true_accepted, y_pred_accepted, zero_division=0)

    return {
        'coverage': float(n_accepted / n_total),
        'n_accepted': n_accepted,
        'n_total': n_total,
        'accuracy_sel': float(accuracy_sel),
        'precision_sel': float(precision_sel),
        'recall_sel': float(recall_sel),
        'f1_sel': float(f1_sel),
    }


def risk_coverage_curve(y_true, pos_logit, neg_logit, abstain_logit, n_thresholds=20):
    """
    Compute risk-coverage curve by varying the abstain threshold.

    This simulates different coverage targets by sweeping the decision boundary
    on the abstain logit.

    Parameters
    ----------
    y_true : array-like
        True labels (0/1)
    pos_logit : array-like
        Positive class logits
    neg_logit : array-like
        Negative class logits
    abstain_logit : array-like
        Abstain logits
    n_thresholds : int
        Number of thresholds to test

    Returns
    -------
    dict
        Dictionary with 'risk' and 'coverage' arrays
    """
    y_true = np.asarray(y_true)
    pos_logit = np.asarray(pos_logit)
    neg_logit = np.asarray(neg_logit)
    abstain_logit = np.asarray(abstain_logit)

    # Get range of abstain logits
    min_abstain = np.min(abstain_logit)
    max_abstain = np.max(abstain_logit)
    thresholds = np.linspace(min_abstain, max_abstain, n_thresholds)

    risks = []
    coverages = []

    for thresh in thresholds:
        # Accept if abstain_logit > thresh (higher abstain logit means more likely to abstain)
        accepted_mask = abstain_logit <= thresh
        coverage = np.mean(accepted_mask)

        if coverage > 0:
            # Decision: argmax of [pos_logit, neg_logit] for accepted examples
            accepted_pos = pos_logit[accepted_mask]
            accepted_neg = neg_logit[accepted_mask]
            accepted_true = y_true[accepted_mask]

            # Predict positive if pos_logit > neg_logit
            predictions = (accepted_pos > accepted_neg).astype(int)

            # Risk: 1 - accuracy on accepted examples
            accuracy = np.mean(predictions == accepted_true)
            risk = 1.0 - accuracy
        else:
            risk = 1.0  # Maximum risk when nothing is accepted

        risks.append(risk)
        coverages.append(coverage)

    return {
        'risk': np.array(risks),
        'coverage': np.array(coverages),
        'thresholds': thresholds,
    }
