# utils/train/calibration.py
import numpy as np
import warnings
import threadpoolctl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from .purged_group_time_series_split import PurgedGroupTimeSeriesSplit
from .thresholds import select_threshold
from .metrics import ece as _ece

# === Wrappers (moved verbatim) ===
class EnsembleCalibratedModel:
    def __init__(self, base_estimator, calibrators):
        self.base_estimator = base_estimator
        self.calibrators = calibrators or []
    def predict_proba(self, X):
        if not self.calibrators:
            return self.base_estimator.predict_proba(X)
        probs = None
        for cal in self.calibrators:
            p = cal.predict_proba(X)
            probs = p if probs is None else (probs + p)
        return probs / float(len(self.calibrators))
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype("int8")

class IsotonicCalibratedModel:
    def __init__(self, base_estimator, iso_model: IsotonicRegression):
        self.base_estimator = base_estimator; self.iso_model = iso_model
    def predict_proba(self, X):
        raw = np.clip(self.base_estimator.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        cal = np.clip(self.iso_model.transform(raw), 1e-6, 1 - 1e-6)
        return np.vstack([1 - cal, cal]).T

class BetaCalibratedModel:
    def __init__(self, base_estimator, lr_model: LogisticRegression):
        self.base_estimator = base_estimator; self.lr_model = lr_model
    def predict_proba(self, X):
        p = np.clip(self.base_estimator.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        feats = np.column_stack([np.log(p), np.log(1 - p)])
        cal = self.lr_model.predict_proba(feats)[:, 1]
        cal = np.clip(cal, 1e-6, 1 - 1e-6)
        return np.vstack([1 - cal, cal]).T

class HistogramBinningCalibratedModel:
    def __init__(self, base_estimator, bin_edges: np.ndarray, bin_rates: np.ndarray):
        self.base_estimator = base_estimator; self.bin_edges = bin_edges; self.bin_rates = bin_rates
    def predict_proba(self, X):
        p = np.clip(self.base_estimator.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        idx = np.clip(np.digitize(p, self.bin_edges, right=False) - 1, 0, len(self.bin_rates) - 1)
        cal = self.bin_rates[idx]
        cal = np.clip(cal, 1e-6, 1 - 1e-6)
        return np.vstack([1 - cal, cal]).T

# === Fitting utilities (moved) ===
def fit_full_calibrator(method: str, base_model, X, y):
    method = method.lower()
    if method == "raw":
        return base_model
    if method == "sigmoid":
        cal = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
        with threadpoolctl.threadpool_limits(limits=1, user_api="openmp"):
            cal.fit(X, y)
        return cal
    p = np.clip(base_model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return IsotonicCalibratedModel(base_model, iso)
    if method == "beta":
        feats = np.column_stack([np.log(p), np.log(1 - p)])
        lr = LogisticRegression(max_iter=1000).fit(feats, y)
        return BetaCalibratedModel(base_model, lr)
    if method in {"hist", "histogram_binning"}:
        n_bins = 20
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        rates = np.zeros(n_bins, dtype=float)
        for b in range(n_bins):
            mask = idx == b
            if np.any(mask):
                pos = y[mask].sum(); tot = mask.sum()
                rates[b] = (pos + 1.0) / (tot + 2.0)
            else:
                rates[b] = 0.5
        return HistogramBinningCalibratedModel(base_model, edges, rates)
    raise ValueError(f"Unknown calibration method: {method}")

def oof_calibrated_probs(method: str, base_model, X_cal, y_cal, groups_cal, cv_folds: int, group_gap: int = 5):
    method = method.lower()
    if method == "raw":
        return base_model.predict_proba(X_cal)[:, 1]
    splitter = PurgedGroupTimeSeriesSplit(n_splits=max(2, int(cv_folds)), group_gap=int(group_gap))
    import numpy as np
    oof = np.full(len(X_cal), np.nan, dtype=float)
    for tr_idx, va_idx in splitter.split(X_cal, y_cal, groups=groups_cal):
        y_tr = y_cal.iloc[tr_idx]
        if y_tr.nunique() < 2:
            continue
        if method == "sigmoid":
            cal = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
            with threadpoolctl.threadpool_limits(limits=1, user_api="openmp"):
                cal.fit(X_cal.iloc[tr_idx], y_tr)
        elif method == "isotonic":
            p_tr = np.clip(base_model.predict_proba(X_cal.iloc[tr_idx])[:, 1], 1e-6, 1 - 1e-6)
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_tr, y_tr)
            from .calibration import IsotonicCalibratedModel as _Iso  # avoid circular
            cal = _Iso(base_model, iso)
        elif method == "beta":
            p_tr = np.clip(base_model.predict_proba(X_cal.iloc[tr_idx])[:, 1], 1e-6, 1 - 1e-6)
            feats_tr = np.column_stack([np.log(p_tr), np.log(1 - p_tr)])
            lr = LogisticRegression(max_iter=1000).fit(feats_tr, y_tr)
            from .calibration import BetaCalibratedModel as _Beta
            cal = _Beta(base_model, lr)
        elif method in {"hist", "histogram_binning"}:
            n_bins = 20
            edges = np.linspace(0.0, 1.0, n_bins + 1)
            p_tr = np.clip(base_model.predict_proba(X_cal.iloc[tr_idx])[:, 1], 1e-6, 1 - 1e-6)
            idx_tr = np.clip(np.digitize(p_tr, edges) - 1, 0, n_bins - 1)
            rates = np.zeros(n_bins, dtype=float)
            for b in range(n_bins):
                mask = idx_tr == b
                if np.any(mask):
                    pos = y_cal.iloc[tr_idx][mask].sum()
                    tot = mask.sum()
                    rates[b] = (pos + 1.0) / (tot + 2.0)
                else:
                    rates[b] = 0.5
            from .calibration import HistogramBinningCalibratedModel as _Hist
            cal = _Hist(base_model, edges, rates)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        oof[va_idx] = cal.predict_proba(X_cal.iloc[va_idx])[:, 1]
    if np.isnan(oof).any():
        raw = base_model.predict_proba(X_cal)[:, 1]
        oof = np.where(np.isnan(oof), raw, oof)
    return oof

def calibrate_and_select_threshold(model, X_cal, y_cal, enable, cal_method, thresh_cfg, groups_cal=None, problem_config=None, calibration_cv_folds=3, group_gap=5):
    """Wrapper for the trio: OOF ECE selection → threshold on OOF → full fit of winner."""
    from .model_factory import is_classification
    if problem_config and not is_classification(problem_config):
        return model, None, None, None
    if (not enable) or (X_cal is None) or (y_cal is None) or (len(y_cal) < 100) or (y_cal.nunique() < 2):
        try:
            proba = model.predict_proba(X_cal)[:, 1]
            best_thresh = select_threshold(y_cal, proba,
                (thresh_cfg.get("method","prc")), thresh_cfg.get("target"), thresh_cfg.get("max_fpr", 0.1))
        except Exception:
            best_thresh = 0.5
        return model, best_thresh, "raw", float('nan')

    X_cal = X_cal.reset_index(drop=True); y_cal = y_cal.reset_index(drop=True)
    if groups_cal is not None:
        import pandas as pd
        groups_cal = pd.Series(groups_cal).reset_index(drop=True).values

    candidate_methods = ["sigmoid", "isotonic", "beta", "hist", "raw"]
    scores = []
    for m in candidate_methods:
        try:
            oof = oof_calibrated_probs(m, model, X_cal, y_cal, groups_cal,
                                       cv_folds=int(calibration_cv_folds), group_gap=int(group_gap))
            e = _ece(y_cal, oof)
            scores.append((m, e, oof))
        except Exception as e:
            warnings.warn(f"OOF calibration failed for method={m}: {e}")

    if not scores:
        try:
            proba = model.predict_proba(X_cal)[:, 1]
            best_thresh = select_threshold(y_cal, proba, (thresh_cfg.get("method","prc")),
                                           thresh_cfg.get("target"), thresh_cfg.get("max_fpr", 0.1))
        except Exception:
            best_thresh = 0.5
        return model, best_thresh, "raw", float('nan')

    scores.sort(key=lambda t: t[1])
    best_method, best_ece, best_oof = scores[0]
    method = str(thresh_cfg.get("method", "prc")).lower()
    target_val = thresh_cfg.get("target")
    max_fpr = thresh_cfg.get("max_fpr", 0.1)
    try:
        best_thresh = select_threshold(y_cal, best_oof, method, target_val, max_fpr)
    except Exception:
        best_thresh = 0.5
    try:
        final_model = fit_full_calibrator(best_method, model, X_cal, y_cal)
    except Exception:
        final_model = model
    return final_model, best_thresh, best_method, best_ece
