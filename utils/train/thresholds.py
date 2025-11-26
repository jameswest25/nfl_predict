# utils/train/thresholds.py
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

def select_threshold(y_true, y_proba, method='prc', target=None, max_fpr=0.1):
    """Moved from train.py::_select_threshold (unchanged)."""
    if y_true is None or y_proba is None or len(y_true) < 2:
        return 0.5
    method = (method or 'prc').lower()
    if method == 'roc':
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        valid = np.where(fpr <= float(max_fpr))[0]
        idx = valid[-1] if len(valid) else np.argmax(tpr)
        return float(thr[idx])
    if method in {'precision', 'recall', 'f1', 'prc'}:
        prec, rec, thr = precision_recall_curve(y_true, y_proba)
        p, r = prec[:-1], rec[:-1]
        metric = p if method == 'precision' else (r if method == 'recall' else 2 * (p * r) / (p + r + 1e-16))
        if target is not None:
            meets = np.where(metric >= float(target))[0]
            idx = meets[-1] if len(meets) else metric.argmax()
        else:
            idx = metric.argmax()
        return float(thr[idx]) if idx < len(thr) else 0.5
    if method == 'accuracy':
        order = np.argsort(y_proba)[::-1]
        scores_sorted = y_proba[order]
        y_sorted = y_true.iloc[order].values.astype(int) if hasattr(y_true, 'iloc') else y_true[order].astype(int)
        total_pos = y_sorted.sum()
        total_neg = len(y_sorted) - total_pos
        tp_cum = np.cumsum(y_sorted)
        fp_cum = np.cumsum(1 - y_sorted)
        tn_cum = total_neg - fp_cum
        acc_cum = (tp_cum + tn_cum) / len(y_sorted)
        change_mask = np.concatenate(([True], scores_sorted[1:] != scores_sorted[:-1]))
        acc_unique = acc_cum[change_mask]
        thr_unique = scores_sorted[change_mask]
        max_acc = acc_unique.max()
        return float(thr_unique[acc_unique == max_acc][-1])
    return 0.5
