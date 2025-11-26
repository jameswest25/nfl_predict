# utils/train/conformal.py
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

class IsotonicMeanCalibratedRegressor:
    def __init__(self, base_estimator, iso: IsotonicRegression):
        self.base_estimator = base_estimator; self.iso = iso
    def predict(self, X): return self.base_estimator.predict(X)

class SplitConformalRegressor:
    def __init__(self, base_estimator, alpha=0.1, method='naive', q=0.0, scaler_model=None, clip_lower=None):
        self.base_estimator = base_estimator; self.alpha=float(alpha); self.method=method
        self.q=float(q); self.scaler_model=scaler_model; self.clip_lower=clip_lower
    def predict(self, X): return self.base_estimator.predict(X)
    def _scale(self, X):
        if self.method=='hc' and self.scaler_model is not None:
            s = np.clip(self.scaler_model.predict(X), 1e-6, np.inf); return s
        return np.ones(len(X), dtype=float)
    def predict_interval(self, X):
        y_hat = self.predict(X); s = self._scale(X)
        lo = y_hat - self.q * s; hi = y_hat + self.q * s
        if self.clip_lower is not None: lo = np.maximum(lo, float(self.clip_lower))
        return lo, hi

def regression_mean_calibrate(base_model, X_cal, y_cal):
    """_regression_mean_calibrate from train.py."""
    try:
        y_pred_cal = base_model.predict(X_cal)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_pred_cal, y_cal)
        return IsotonicMeanCalibratedRegressor(base_model, iso)
    except Exception:
        return base_model

def fit_split_conformal_regression(calibrated_model, X_cal, y_cal, cfg):
    """_fit_split_conformal_regression from train.py (unchanged)."""
    alpha = float(cfg.get("alpha", 0.1))
    method = str(cfg.get("method", "naive")).lower()
    clip_lower = cfg.get("clip_lower", None)
    y_hat_cal = calibrated_model.predict(X_cal)
    resid = np.abs(y_cal - y_hat_cal)
    if method == "hc":
        try:
            scaler = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=1, tree_method='hist', enable_categorical=True, verbosity=0
            )
            scaler.fit(X_cal, resid)
            s = np.clip(scaler.predict(X_cal), 1e-6, np.inf)
            norm_resid = resid / s
            n = len(norm_resid); k = int(np.ceil((n + 1) * (1 - alpha)))
            q = float(np.partition(norm_resid, k-1)[k-1])
            return SplitConformalRegressor(calibrated_model, alpha=alpha, method='hc', q=q, scaler_model=scaler, clip_lower=clip_lower)
        except Exception:
            method = "naive"
    n = len(resid); k = int(np.ceil((n + 1) * (1 - alpha)))
    q = float(np.partition(resid.values if hasattr(resid, 'values') else resid, k-1)[k-1])
    return SplitConformalRegressor(calibrated_model, alpha=alpha, method='naive', q=q, scaler_model=None, clip_lower=clip_lower)
