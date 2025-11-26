"""
Feature Evaluation Suite
========================
A one‑stop helper to answer three practical questions about any candidate
feature(s):

1. **Does it improve probability quality?**  
   *Metrics*: log‑loss ▾, Brier ▾, AUC ▴, ECE ▾.
2. **Is it redundant / collinear** with existing inputs?  
   *Metrics*: VIF, pairwise Spearman |ρ|, Cramér's V (cat‑cat).
3. **How would a whole bag of candidates perform if I add them *greedily*, one
   at a time?**  
   *Greedy forward search* ranked by Δ log‑loss (default) or any scorer.

The suite is intentionally **model‑lightweight** (single‑thread LightGBM) and
runs on the *post‑artefact* matrices you already build in `train.py`, so there
is no risk of data‑leak or schema drift.

---------------------------------------------------------------------------
Quick demo
---------------------------------------------------------------------------
```python
from utils.train.feature_tester import (
    evaluate_feature_impact,
    check_collinearity,
    forward_feature_search,
)

# X, y, groups are your training slice after artefact application
impact = evaluate_feature_impact(
    X=X_train,
    y=y_train,
    groups=groups_train,
    feature_names=["is_same_hand", "platoon_penalty"],
)
print(impact.to_df())

collin = check_collinearity(X_train, ["platoon_penalty"], base_features=X_train.columns)
print(collin)

search_report = forward_feature_search(
    X=X_train,
    y=y_train,
    groups=groups_train,
    candidate_features=my_generated_cols,
    base_features=X_train.columns.difference(my_generated_cols),
    scorer="logloss",
    max_steps=25,
)
print(search_report.head())
```
---------------------------------------------------------------------------
How to integrate
---------------------------------------------------------------------------
* **Standalone notebook / script** – run ad‑hoc when you prototype features.  
* **CI step** – fail the PR if no metric improves beyond a threshold.  
* **Pipeline hook** – call `forward_feature_search` after artefact fitting to
  auto‑curate a pruned feature list; feed that list into the rest of
  `train.py`.

If you embed it inside the full training pipeline, wrap calls in a
`if self.config['training'].get('feature_search', False)` guard so you can turn
it on/off separately from main training.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Callable

import warnings
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency
import pandas.api.types as patypes
import logging
from statsmodels.stats.proportion import proportion_confint
from sklearn.isotonic import IsotonicRegression
from scipy.special import logit, expit

__all__ = [
    "FeatureImpactReport",
    "evaluate_feature_impact",
    "check_collinearity",
    "forward_feature_search",
    "_fast_filter",
    "_backward_trim",
    "holdout_check",
    "_brier_decomposition",
    "_calibration_bins",
]

# module-level logger inherits root settings – bridge enables INFO under --verbose
logger = logging.getLogger(__name__)

# ╭──────────────────────────────────────────────────────────────────────╮
# │ 0 ▸  Calibration diagnostic helpers                                 │
# ╰──────────────────────────────────────────────────────────────────────╯

def _brier_decomposition(y, p, n_bins=10):
    """Return Murphy (1973) decomposition: reliability, resolution, uncertainty."""
    df = pd.DataFrame({'y': y, 'p': p})
    # qcut may drop bins if duplicates, that's fine.
    df['bin'] = pd.qcut(df['p'], n_bins, duplicates='drop')
    groups = df.groupby('bin', observed=True)
    n = len(df)
    base_rate = df['y'].mean()

    reliability = sum(
        len(g) * (g["p"].mean() - g["y"].mean()) ** 2 for _, g in groups
    ) / n
    resolution = sum(
        len(g) * (g["y"].mean() - base_rate) ** 2 for _, g in groups
    ) / n
    uncertainty = base_rate * (1 - base_rate)
    brier       = ((df['p'] - df['y'])**2).mean()
    return dict(brier=brier,
                reliability=reliability,
                resolution=resolution,
                uncertainty=uncertainty)


def _calibration_bins(y, p, n_bins=10, min_bin=50):
    """Return calibration bins with Wilson CIs and attach ECE/MCE attrs.

    Robust to degenerate / tiny folds: if no bin reaches the size threshold,
    we return an empty table but still set ECE/MCE = 0.0 (avoids NaNs upstream).
    """
    df = pd.DataFrame({'y': y, 'p': p}).sort_values('p')

    # Adapt min_bin to the sample size of this slice/fold
    eff_min_bin = max(10, int(len(df) / (n_bins * 4)))
    eff_min_bin = max(eff_min_bin, min_bin)

    df['bin'] = pd.qcut(df['p'], n_bins, duplicates='drop')
    rows = []
    for name, g in df.groupby('bin', observed=True):
        n = len(g)
        if n < eff_min_bin:
            continue
        emp = g['y'].mean()
        pred = g['p'].mean()
        low, high = proportion_confint(int(g['y'].sum()), n, method='wilson')
        rows.append((str(name), n, pred, emp, low, high))

    cols = ['bin', 'n', 'mean_pred', 'emp_rate', 'wilson_low', 'wilson_high', 'abs_gap']
    if not rows:
        out = pd.DataFrame(columns=cols)
        out.attrs['ECE'] = 0.0
        out.attrs['MCE'] = 0.0
        return out

    out = pd.DataFrame(rows, columns=cols[:-1])
    out['abs_gap'] = (out['mean_pred'] - out['emp_rate']).abs()
    N = out['n'].sum()
    ece = ((out['n'] / N) * out['abs_gap']).sum() if N else 0.0
    mce = out['abs_gap'].max() if len(out) else 0.0
    out.attrs['ECE'] = float(ece)
    out.attrs['MCE'] = float(mce)
    return out


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 1 ▸  Calibration helper                                             │
# ╰──────────────────────────────────────────────────────────────────────╯

def _ece(
    y_true: Sequence[int | float],
    y_prob: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Expected-Calibration-Error using equal-mass (quantile) bins."""

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Build quantile edges (unique & inclusive) ----------------------------
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_prob, quantiles)
    # Guarantee strictly-monotone edges to avoid empty slices
    bin_edges[0], bin_edges[-1] = 0.0, 1.0
    bin_edges = np.unique(bin_edges)

    bin_idx = np.digitize(y_prob, bin_edges[1:-1], right=True)

    ece = 0.0
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += abs(acc - conf) * mask.mean()
    return float(ece)


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 2 ▸  Core OO‑Report container                                       │
# ╰──────────────────────────────────────────────────────────────────────╯

class FeatureImpactReport:
    """Collects metric deltas and prints as tidy DataFrame."""

    def __init__(self) -> None:
        self.raw: dict[str, dict[str, float]] = {}
        self.calibration_data: dict[str, dict] = {}

    def add(self, metric: str, base: float, with_feat: float) -> None:
        self.raw[metric] = {
            "base": float(base),
            "with_feature": float(with_feat),
            "delta": float(with_feat - base),
        }

    def add_calibration(self, name: str, base_data: dict, with_feat_data: dict) -> None:
        """Add calibration diagnostics for base vs with-feature models."""
        self.calibration_data[name] = {
            "base": base_data,
            "with_feature": with_feat_data,
        }

    def to_df(self) -> pd.DataFrame:  # noqa: D401 (simple imperative)
        return (
            pd.DataFrame(self.raw)
            .T.reset_index()
            .rename(columns={"index": "metric"})
        )

    def save_calibration_data(self, path: str, which: str = "with_feature") -> None:
        """Save calibration table to CSV or Parquet file.
        
        Args:
            path: Output file path (.csv or .parquet)
            which: Which calibration data to save ("base" or "with_feature")
        """
        if not self.calibration_data:
            logger.warning("No calibration data available to save")
            return
            
        calib_data = self.calibration_data.get("calibration_bins", {})
        if not calib_data:
            logger.warning("No calibration_bins data found")
            return
            
        target_data = calib_data.get(which, {})
        if not target_data:
            logger.warning(f"No {which} calibration data found")
            return
            
        calib_tbl = target_data.get("table")
        if calib_tbl is None or len(calib_tbl) == 0:
            logger.warning(f"No calibration table found in {which} data")
            return
            
        if path.endswith('.parquet'):
            calib_tbl.to_parquet(path, index=False)
        else:
            calib_tbl.to_csv(path, index=False)
        logger.info(f"Saved {which} calibration table → {path}")

    __repr__ = lambda self: self.to_df().__repr__()


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 3 ▸  LightGBM OOF helper                                            │
# ╰──────────────────────────────────────────────────────────────────────╯

DEFAULT_LGB_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "n_jobs": max(1, os.cpu_count() // 4),
    "verbosity": -1,
    "seed": 42,
}


def _oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Sequence,
    use_cols: list[str],
    params: dict,
    n_splits: int,
    calibrator: str = "none",
) -> np.ndarray:
    cv = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype="float32")
    for tr_idx, va_idx in cv.split(X, y, groups):
        X_tr = X.iloc[tr_idx][use_cols]
        # ─── Guard: no usable variance in this fold -----------------------
        if (
            X_tr.shape[1] == 0
            or X_tr.nunique(dropna=False).max() <= 1
        ):
            oof[va_idx] = y.iloc[tr_idx].mean()
            continue

        # Create a copy of params to avoid modifying the original
        fold_params = params.copy()
        
        # Preprocess data to avoid circular reference issues
        X_tr_processed = X_tr.copy()
        X_va_processed = X.iloc[va_idx][use_cols].copy()
        
        # Convert categorical columns to numeric codes
        cat_cols = X_tr_processed.select_dtypes(include=['category', 'object']).columns
        for col in cat_cols:
            if col in X_tr_processed.columns:
                X_tr_processed[col] = pd.Categorical(X_tr_processed[col]).codes.astype('int32')
                X_va_processed[col] = pd.Categorical(X_va_processed[col]).codes.astype('int32')
        
        # Convert datetime columns to int64
        dt_cols = X_tr_processed.select_dtypes(include=['datetime64[ns]']).columns
        for col in dt_cols:
            if col in X_tr_processed.columns:
                X_tr_processed[col] = X_tr_processed[col].astype('int64')
                X_va_processed[col] = X_va_processed[col].astype('int64')
        
        # Fill NaNs
        X_tr_processed = X_tr_processed.fillna(0)
        X_va_processed = X_va_processed.fillna(0)
        
        # Remove problematic parameters that might cause serialization issues
        fold_params.pop('categorical_feature', None)  # Remove if present
        
        model = lgb.LGBMClassifier(**fold_params)
        
        # Try to fit without early stopping first
        try:
            model.fit(
                X_tr_processed,
                y.iloc[tr_idx],
                eval_set=[(X_va_processed, y.iloc[va_idx])],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            best_iter = model.best_iteration_ or params["n_estimators"]
        except Exception as e:
            # If early stopping fails, try without it
            logger.warning(f"Early stopping failed in fold, trying without: {e}")
            model.fit(
                X_tr_processed,
                y.iloc[tr_idx],
                eval_set=[(X_va_processed, y.iloc[va_idx])],
                eval_metric="binary_logloss",
            )
            best_iter = params["n_estimators"]
            
        oof[va_idx] = model.predict_proba(
            X_va_processed, num_iteration=best_iter
        )[:, 1]
        
        # -----------------------
        # Optional calibration
        # -----------------------
        if calibrator != 'none':
            # Get raw predictions on training fold for calibration
            y_train_raw = model.predict_proba(X_tr_processed, num_iteration=best_iter)[:, 1]
            
            if calibrator == 'platt':
                # Simple logistic recalibration
                lr = LogisticRegression(solver='lbfgs', max_iter=1000)
                # Avoid infs
                p_clip = np.clip(y_train_raw, 1e-6, 1-1e-6)
                lr.fit(logit(p_clip).reshape(-1,1), y.iloc[tr_idx])
                def cal_fn(p):
                    pc = np.clip(p, 1e-6, 1-1e-6)
                    return expit(lr.coef_[0,0]*logit(pc) + lr.intercept_[0])
            else:  # isotonic
                ir = IsotonicRegression(out_of_bounds='clip')
                ir.fit(y_train_raw, y.iloc[tr_idx])
                cal_fn = ir.predict
            
            # Apply calibration to validation predictions
            oof[va_idx] = cal_fn(oof[va_idx])
    
    return oof


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 4 ▸  Public helpers                                                 │
# ╰──────────────────────────────────────────────────────────────────────╯

def evaluate_feature_impact(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    groups: Sequence,
    feature_names: List[str],
    lgb_params: dict | None = None,
    n_splits: int = 5,
    calibration_bins: int = 10,
    min_cal_bin: int = 50,
    calibrator: str = "none",
) -> FeatureImpactReport:
    """Strict δ‑test with/without *feature_names* using GroupKFold OOFS."""
    if not set(feature_names) <= set(X.columns):
        raise ValueError("Some requested feature names are missing from X.")

    params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
    report = FeatureImpactReport()

    base_cols = [c for c in X.columns if c not in feature_names]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
        oof_base = _oof_predictions(X, y, groups, base_cols, params, n_splits, calibrator)
        oof_feat = _oof_predictions(X, y, groups, X.columns.tolist(), params, n_splits, calibrator)

    # Core metrics
    report.add("logloss", log_loss(y, oof_base), log_loss(y, oof_feat))
    report.add("brier", brier_score_loss(y, oof_base), brier_score_loss(y, oof_feat))
    report.add("auc", roc_auc_score(y, oof_base), roc_auc_score(y, oof_feat))
    report.add("ece", _ece(y, oof_base), _ece(y, oof_feat))
    
    # Calibration diagnostics
    calib_base = _calibration_bins(y, oof_base, n_bins=calibration_bins, min_bin=min_cal_bin)
    calib_feat = _calibration_bins(y, oof_feat, n_bins=calibration_bins, min_bin=min_cal_bin)
    
    bdec_base = _brier_decomposition(y, oof_base, n_bins=calibration_bins)
    bdec_feat = _brier_decomposition(y, oof_feat, n_bins=calibration_bins)
    
    # Add calibration metrics
    report.add("mce", calib_base.attrs.get('MCE', np.nan), calib_feat.attrs.get('MCE', np.nan))
    report.add("brier_reliability", bdec_base['reliability'], bdec_feat['reliability'])
    report.add("brier_resolution", bdec_base['resolution'], bdec_feat['resolution'])
    report.add("brier_uncertainty", bdec_base['uncertainty'], bdec_feat['uncertainty'])
    
    # Store detailed calibration data
    report.add_calibration("calibration_bins", 
                          {"table": calib_base, "brier_decomp": bdec_base},
                          {"table": calib_feat, "brier_decomp": bdec_feat})
    
    return report


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 5 ▸  Collinearity diagnostics                                       │
# ╰──────────────────────────────────────────────────────────────────────╯

def _spearman_abs(x: pd.Series, y: pd.Series) -> float:
    return float(abs(x.corr(y, method="spearman")))


def _cramers_v(cat1: pd.Series, cat2: pd.Series) -> float:
    """Bias-corrected Cramér’s V."""
    confusion = pd.crosstab(cat1, cat2)
    chi2, _, _, _ = chi2_contingency(confusion, correction=False)
    n = confusion.values.sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion.shape
    # --- Bias correction (Bergsma & Wicher, 2013)
    phi2_corr = max(phi2 - (k - 1) * (r - 1) / (n - 1), 0)
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = max((r_corr - 1) * (k_corr - 1), 1e-12)
    return float(np.sqrt(phi2_corr / denom))


def check_collinearity(
    X: pd.DataFrame,
    feature_names: List[str],
    *,
    base_features: Iterable[str] | None = None,
    y_target: pd.Series | None = None,
    vif_thresh: float = 10.0,
    corr_thresh: float = 0.85,
) -> pd.DataFrame:
    """Return collinearity diagnostics for *feature_names* vs *base_features*.

    Numeric-numeric pairs: Spearman |ρ| & VIF (if candidate numeric).
    Cat-cat pairs: Cramér's V.
    Mixed-type pairs currently skipped.
    """

    base_features = (
        list(base_features)
        if base_features is not None
        else list(X.columns.difference(feature_names))
    )

    rows: list[dict] = []
    for f in feature_names:
        s = X[f]
        is_num = patypes.is_numeric_dtype(s)

        # ---------- VIF (numeric only) --------------------------------
        vif_val = np.nan
        if is_num:
            try:
                vdf = pd.concat([X[base_features], s], axis=1)
                # NaNs → column mean to avoid artificial collinearity
                vdf = vdf.apply(lambda col: col.fillna(col.mean()), axis=0)
                vif_val = variance_inflation_factor(vdf.values, vdf.shape[1] - 1)
            except Exception:
                pass  # keep NaN

        # ---------- Max pairwise association -------------------------
        max_corr = 0.0
        for bf in base_features:
            t = X[bf]
            if is_num and patypes.is_numeric_dtype(t):
                corr = _spearman_abs(s, t)
            elif (not is_num) and (not patypes.is_numeric_dtype(t)):
                corr = _cramers_v(s, t)
            else:
                corr = np.nan
            if pd.notna(corr):
                max_corr = max(max_corr, corr)

        rows.append(
            {
                "feature": f,
                "vif": vif_val,
                "max_pairwise_corr": max_corr,
                "vif_flag": vif_val > vif_thresh if pd.notna(vif_val) else False,
                "corr_flag": max_corr > corr_thresh,
            }
        )

    return pd.DataFrame(rows)


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 6 ▸  Greedy forward search                                          │
# ╰──────────────────────────────────────────────────────────────────────╯

def forward_feature_search(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    groups: Sequence,
    candidate_features: list[str],
    base_features: list[str] | None = None,
    scorer: str = "logloss",  # "logloss" | "brier" | "auc" | "composite" | "logloss_brier"
    n_splits: int = 5,
    max_steps: int | None = None,
    top_frac: float = 1.0,
    top_k_step: int = 2000,
    progress_wrapper=lambda x, **k: x,
    log_level: int | None = None,
    calibration_bins: int = 10,
    min_cal_bin: int = 50,
    calibrator: str = "none",
    metric_weights: dict | None = None,
    calib_tol: float = 0.005,
    min_delta_abs: float = 0.0,
    min_delta_rel: float = 1e-3,
    c_se: float = 1.8,
    max_fails: int = 4,
    stop_window: int = 5,
) -> pd.DataFrame:
    """Greedy add‑one‑at‑a‑time search over *candidate_features*.

    Stops when no feature improves the *scorer* by > min_delta_abs OR > min_delta_rel 
    (relative to current score) OR > c_se * SE (noise-aware threshold), or after 
    max_fails consecutive rejections, or when rolling median improvement over 
    stop_window steps is too low.
    """
    if scorer not in {"logloss", "brier", "auc", "composite", "logloss_brier"}:
        raise ValueError(
            "scorer must be 'logloss', 'brier', 'auc', 'composite', or 'logloss_brier'"
        )

    params = {**DEFAULT_LGB_PARAMS}
    if log_level is not None:
        logger.setLevel(log_level)
    base_features = list(base_features) if base_features is not None else []
    remaining = [f for f in candidate_features if f not in base_features]

    # ── 0️⃣  Drop constant / single-value columns ---------------------------
    if remaining:
        nunq = X[remaining].nunique(dropna=False)
        remaining = nunq[nunq > 1].index.tolist()

    # ── 1️⃣  Univariate filter (top `top_frac`) ----------------------------
    if remaining and 0.0 < top_frac < 1.0:
        abs_score: dict[str, float] = {}
        for col in remaining:
            s = X[col]
            if patypes.is_numeric_dtype(s):
                # Spearman |rho| with NaN-safe fill
                base = s.fillna(0)
                rho = abs(base.corr(y, method="spearman"))
                # Polarity-flip rescue: include squared and cubed terms
                if rho < 0.01:
                    rho2 = abs((base ** 2).corr(y, method="spearman"))
                    rho3 = abs((base ** 3).corr(y, method="spearman"))
                    rho = max(rho, rho2, rho3)
            else:
                # Use shared helper for categorical association
                rho = _univariate_cat_score(s, y)
            abs_score[col] = float(rho)
        k_raw = int(len(remaining) * top_frac)
        k = max(1, min(len(remaining), max(k_raw, 1)))
        remaining = sorted(abs_score, key=abs_score.get, reverse=True)[:k]

        # Ensure we don't reconsider already-selected columns
        remaining = [c for c in remaining if c not in base_features]

    # ------------------------------------------------------------------
    # Helper functions for fast candidate evaluation using OOF cache
    # ------------------------------------------------------------------

    def _metric(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, weights: dict | None = None) -> float:  # noqa: D401
        """Return scalar score for the chosen *scorer* name."""
        if scorer == "logloss":
            return log_loss(y_true, y_pred)
        elif scorer == "brier":
            return brier_score_loss(y_true, y_pred)
        elif scorer == "auc":
            try:
                return roc_auc_score(y_true, y_pred)
            except ValueError:
                return 0.5  # Default for constant predictions
        elif scorer == "composite":
            # Negate so that *lower* is better (consistent with logloss/brier)
            return -_composite_score(y_true, y_pred, calibration_bins, min_cal_bin, weights, calib_tol)
        elif scorer == "logloss_brier":
            # ── Relative-gain mean (higher gain = better ⇒ negate for loss) ──
            y_base = np.full_like(y_pred, y_true.mean(), dtype=float)

            ll0 = log_loss(y_true, y_base)
            br0 = brier_score_loss(y_true, y_base)

            ll1 = log_loss(y_true, y_pred)
            br1 = brier_score_loss(y_true, y_pred)

            gain_ll = (ll0 - ll1) / (ll0 + 1e-12)          # ∈ (-∞ , 1]
            gain_br = (br0 - br1) / (br0 + 1e-12)          # ∈ (-∞ , 1]

            return -0.5 * (gain_ll + gain_br)              # lower still better
        else:
            raise ValueError("Unknown scorer")

    def _blend_with_feature(
        col: str,
        oof_base: np.ndarray,
        *,
        X_df: pd.DataFrame,
        y_ser: pd.Series,
        grp,
    ) -> np.ndarray:
        """Blend cached baseline OOF with quick 1-feature model predictions."""
        # 1-feature quick LightGBM
        oof_new = _oof_predictions(X_df, y_ser, grp, [col], params, n_splits, calibrator)

        # Fold-specific logistic blend: [oof_base, oof_new] → prob
        cv_inner = GroupKFold(n_splits=n_splits)
        oof_blend = np.zeros_like(oof_base)
        for tr_idx, va_idx in cv_inner.split(oof_base.reshape(-1, 1), y_ser, grp):
            lr = LogisticRegression(
                solver="lbfgs",
                max_iter=30,
                C=1.0,
                n_jobs=1,
                random_state=42,
            )
            lr.fit(
                np.c_[oof_base[tr_idx], oof_new[tr_idx]],
                y_ser.iloc[tr_idx],
            )
            oof_blend[va_idx] = lr.predict_proba(
                np.c_[oof_base[va_idx], oof_new[va_idx]]
            )[:, 1]
        return oof_blend

    history = []

    # ── 1. Fit baseline once per step ──────────────────────────────────
    logger.debug("▶️  starting search: %d candidates", len(remaining))
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    
    current_cols = base_features.copy()
    oof_current = _oof_predictions(X, y, groups, current_cols, params, n_splits, calibrator)
    # fold-wise scores for SE calc
    current_fold_scores = [ _metric(y.iloc[va], oof_current[va], metric_weights) for _, va in folds ]
    current_score = float(np.mean(current_fold_scores))

    step = 0
    fails = 0  # Track consecutive failures
    while remaining and (max_steps is None or step < max_steps):

        logger.debug("step %d  –  pool=%d", step + 1, len(remaining))
        # -------- Phase 1b : residual-based shortlist --------------------
        resid = y.values - oof_current
        shortlist = _fast_filter(X[remaining], y, residual=resid, top_frac=top_frac)
        shortlist = shortlist[:top_k_step]  # cap absolute number per step

        logger.debug("  shortlist → %d   top-5: %s", len(shortlist), shortlist[:5])
        # -------- Phase 2 : gain-based batch pruning ----------------------
        # Run a cheap gain-based LightGBM fit to discard the weakest 80 % of
        # the current pool whenever it grows beyond *top_k_step* – no hard
        # floors or ceilings.
        if len(remaining) > top_k_step:
            batch_model = lgb.LGBMClassifier(
                **{**params, "n_estimators": 80, "feature_fraction": 1.0}
            )
            batch_model.fit(X[remaining], y)
            gain = batch_model.booster_.feature_importance(importance_type="gain")
            ranked = sorted(zip(remaining, gain), key=lambda kv: kv[1], reverse=True)
            keep = max(1, int(len(remaining) * 0.2))
            remaining = [name for name, _ in ranked[:keep]]

        # -------- Phase 3 : analytic delta on shortlist -------------------
        def _delta_score(col):
            oof_blend = _blend_with_feature(col, oof_current, X_df=X, y_ser=y, grp=groups)
            fold_scores = [ _metric(y.iloc[va], oof_blend[va], metric_weights) for _, va in folds ]
            mean_score  = float(np.mean(fold_scores))
            # gain is "old - new" when lower is better (logloss/brier/composite)
            return mean_score, fold_scores

        results = Parallel(
            n_jobs=min(4, max(1, os.cpu_count() // 4)),
            backend="loky",
        )(delayed(_delta_score)(f) for f in progress_wrapper(shortlist))

        # unpack
        mean_scores, fold_scores_l = zip(*results)
        mean_scores   = list(mean_scores)
        fold_scores_l = list(fold_scores_l)

        if scorer in {"logloss", "brier", "composite", "logloss_brier"}:
            best_idx   = int(np.argmin(mean_scores))
            best_score = mean_scores[best_idx]
            gains_pf   = np.array(current_fold_scores) - np.array(fold_scores_l[best_idx])
            improvement = float(gains_pf.mean())
        else:  # auc higher better
            best_idx   = int(np.argmax(mean_scores))
            best_score = mean_scores[best_idx]
            gains_pf   = np.array(fold_scores_l[best_idx]) - np.array(current_fold_scores)
            improvement = float(gains_pf.mean())

        se = float(gains_pf.std(ddof=1) / np.sqrt(len(gains_pf)))
        best_feat = shortlist[best_idx]

        # Calculate relative gain for logging and history
        rel_gain = improvement / abs(current_score) if current_score else 0.0

        # diagnostics: show top-k gains
        k_dbg = min(5, len(shortlist))
        order = (np.argsort(mean_scores) if scorer in {"logloss", "brier", "composite", "logloss_brier"} else np.argsort(-np.array(mean_scores)))
        dbg = [(shortlist[i], float(current_score - mean_scores[i] if scorer in {"logloss", "brier", "composite", "logloss_brier"} else mean_scores[i] - current_score)) for i in order[:k_dbg]]
        logger.debug("  top-%d gains: %s", k_dbg, ", ".join(f"{n}:{g:+.6f}" for n, g in dbg))

        history.append(
            {
                "step": step + 1,
                "added_feature": best_feat,
                "prev_score": current_score,
                "new_score": best_score,
                "improvement": improvement,
                "rel_gain": rel_gain,
                "se": se,
                "accepted": False  # Will be set to True if accepted
            }
        )

        # New sophisticated stopping criteria
        accept = (
            (improvement > min_delta_abs) or
            (rel_gain     > min_delta_rel) or
            (improvement  > c_se * se)
        )
        
        if accept:
            fails = 0  # Reset failure counter
            history[-1]["accepted"] = True  # Mark as accepted
            # Accept feature and continue
            current_cols.append(best_feat)
            # Re-fit full LightGBM on *all* accepted features (no stacking)
            oof_current = _oof_predictions(
                X,
                y,
                groups,
                current_cols,
                params,
                n_splits,
                calibrator,
            )
            # fold-wise scores for SE calc
            current_fold_scores = [ _metric(y.iloc[va], oof_current[va], metric_weights) for _, va in folds ]
            current_score = float(np.mean(current_fold_scores))
            if best_feat in remaining:
                remaining.remove(best_feat)
            # Verbose progress log
            logger.info("[%d] + %s  (Δ=%.6f | rel=%.5g | se=%.6f)", step + 1, best_feat, improvement, rel_gain, se)
        else:
            fails += 1
            logger.debug("reject %s  (Δ=%.6f | rel=%.5g | se=%.6f)", best_feat, improvement, rel_gain, se)
            if fails >= max_fails:
                logger.info("🛑  Early stop: %d straight rejects", fails)
                break
        
        # Optional rolling median stop check
        if stop_window > 1 and len(history) >= stop_window:
            recent = history[-stop_window:]
            med = np.median([h["improvement"] for h in recent])
            # use a stable floor, e.g. recent SE median or rel threshold
            se_med = max(np.median([h["se"] for h in recent]), 1e-12)
            if med <= c_se * se_med:
                logger.info("🛑 Early stop: median Δ over last %d steps ≤ %.3g", stop_window, c_se * se_med)
                break
        
        step += 1

    return pd.DataFrame(history)


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 5½ ▸  Ultra-cheap univariate pre-filter                              │
# ╰──────────────────────────────────────────────────────────────────────╯

# Numeric: squared correlation with negative log-loss gradient

def _univariate_numeric_score(col: np.ndarray, grad: np.ndarray) -> float:
    col = col.astype("float32", copy=False)
    col -= col.mean()
    # Clip extremes to avoid overflow in (col ** 2).sum()
    col = np.clip(col, -1e6, 1e6)

    grad = grad.astype("float32", copy=False)
    grad -= grad.mean()
    num   = float(np.dot(col, grad))
    denom = float(np.sqrt((col ** 2).sum() * (grad ** 2).sum()) + 1e-12)
    rho   = num / denom
    return abs(rho)                         # keep sign info out of square


# Categorical: chi-square statistic scaled by sample size

def _univariate_cat_score(col: pd.Series, y: pd.Series) -> float:
    tbl = pd.crosstab(col, y)
    if tbl.size == 0:
        return 0.0
    chi2, _, _, _ = chi2_contingency(tbl, correction=False)
    dof = max((tbl.shape[0] - 1) * (tbl.shape[1] - 1), 1)
    return float(chi2 / dof)


def _fast_filter(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    residual: np.ndarray,
    top_frac: float = 1.0,
) -> List[str]:
    """Return top-M feature names based on cheap univariate proxy scores."""
    scores: dict[str, float] = {}
    for col in X.columns:
        s = X[col]
        if patypes.is_numeric_dtype(s):
            scores[col] = _univariate_numeric_score(s.values, residual.copy())
        else:
            scores[col] = _univariate_cat_score(s, y)
    k_raw = int(len(scores) * top_frac)
    k = max(1, min(len(scores), max(k_raw, 1)))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [name for name, _ in ranked]


# ──────────────────────────────────────────────────────────────────────
# Composite probability-quality scorer
# ---------------------------------------------------------------------
def _composite_score(y_true: Sequence[int | float],
                     y_pred: Sequence[float],
                     calibration_bins: int = 10,
                     min_cal_bin: int = 50,
                     weights: dict | None = None,
                     calib_tol: float = 0.005) -> float:
    """Mean %-improvement across logloss↓, Brier↓, AUC↑, ECE↓, MCE↓, Brier components (higher better).
       Enforces soft constraint: calibration metrics cannot worsen by more than calib_tol."""
    # Convert to numpy arrays up-front
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Core metrics for current model
    ll  = log_loss(y_true, y_pred)
    br  = brier_score_loss(y_true, y_pred)
    try:
        au = roc_auc_score(y_true, y_pred)
    except ValueError:
        au = 0.5
    ec  = _ece(y_true, y_pred)

    
    # Calibration diagnostics
    calib_tbl = _calibration_bins(y_true, y_pred, n_bins=calibration_bins, min_bin=min_cal_bin)
    bdec = _brier_decomposition(y_true, y_pred, n_bins=calibration_bins)
    mc = float(calib_tbl.attrs.get('MCE', 0.0))

    # Baseline = predict constant base-rate
    p0  = np.full_like(y_pred, y_true.mean(), dtype=float)
    ll0 = log_loss(y_true, p0)
    br0 = brier_score_loss(y_true, p0)
    try:
        au0 = roc_auc_score(y_true, p0)
    except ValueError:
        au0 = 0.5
    ec0 = _ece(y_true, p0)
    
    # Baseline calibration diagnostics
    calib_tbl0 = _calibration_bins(y_true, p0, n_bins=calibration_bins, min_bin=min_cal_bin)
    bdec0 = _brier_decomposition(y_true, p0, n_bins=calibration_bins)
    mc0 = float(calib_tbl0.attrs.get('MCE', 0.0))

    eps = 1e-8

    def _safe_ratio(num, denom):
        denom = max(abs(denom), eps)
        return num / denom

    # Core metric gains
    ll_gain = _safe_ratio(ll0 - ll, ll0)
    br_gain = _safe_ratio(br0 - br, br0)
    # For AUC we normalise to the maximum possible headroom (1 - baseline)
    auc_gain = _safe_ratio(au - au0, 1 - au0)

    # ECE: if baseline is 0, use absolute improvement scaled by initial error
    if ec0 > eps:
        ece_gain = _safe_ratio(ec0 - ec, ec0)
    else:
        ece_gain = _safe_ratio(-ec, ec + eps)
    
    # MCE: similar to ECE
    if mc0 > eps:
        mce_gain = _safe_ratio(mc0 - mc, mc0)
    else:
        mce_gain = _safe_ratio(-mc, mc + eps)
    
    # Brier decomposition gains (lower reliability is better, higher resolution is better)
    rel_gain = _safe_ratio(bdec0['reliability'] - bdec['reliability'], bdec0['reliability'])
    res_gain = _safe_ratio(bdec['resolution'] - bdec0['resolution'], bdec0['resolution'] + eps)

    # Soft constraint checks (relative worsening > calib_tol => reject)
    if ec0 > eps and (ec - ec0)/ec0 > calib_tol:
        return -np.inf
    if mc0 > eps and (mc - mc0)/mc0 > calib_tol:
        return -np.inf
    if bdec0['reliability'] > eps and (bdec['reliability'] - bdec0['reliability'])/bdec0['reliability'] > calib_tol:
        return -np.inf

    # Use weights if provided, otherwise equal weights
    if weights is not None:
        gains = np.array([
            weights.get('logloss', 1.0) * ll_gain,
            weights.get('brier', 1.0) * br_gain,
            weights.get('auc', 1.0) * auc_gain,
            weights.get('ece', 1.0) * ece_gain,
            weights.get('mce', 1.0) * mce_gain,
            weights.get('brier_reliability', 1.0) * rel_gain,
            weights.get('brier_resolution', 1.0) * res_gain
        ], dtype=float)
        return float(gains.mean())
    else:
        gains = np.array([ll_gain, br_gain, auc_gain, ece_gain, mce_gain, rel_gain, res_gain], dtype=float)
        return float(gains.mean())


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 7 ▸  Post-hoc backward elimination & hold-out sanity check          │
# ╰──────────────────────────────────────────────────────────────────────╯

def _backward_trim(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Sequence,
    *,
    accepted_feats: list[str],
    base_feats: list[str] | None = None,
    params: dict | None = None,
    n_splits: int = 5,
    eps: float = 1e-6,
    metric_weights: dict | None = None,
    calib_tol: float = 0.005,
) -> list[str]:
    """Greedy backward pass that removes redundant *accepted_feats*.

    A feature is dropped if its removal degrades composite score by ≤ *eps*
    (default 0.01 % under the negated-loss convention).  Returns the pruned
    list in the order they originally appeared.
    """

    if not accepted_feats:
        return []

    base_feats = list(base_feats) if base_feats is not None else []
    params = {**DEFAULT_LGB_PARAMS, **(params or {})}

    # Fit once with the full accepted bundle
    full_oof = _oof_predictions(
        X,
        y,
        groups,
        base_feats + accepted_feats,
        params,
        n_splits,
        calibrator="none",  # Use no calibration for backward trim
    )
    full_score = -_composite_score(y, full_oof, calibration_bins=10, min_cal_bin=50, weights=metric_weights, calib_tol=calib_tol)  # lower-is-better convention

    pruned = accepted_feats.copy()
    for feat in reversed(accepted_feats):  # try removing last-added first
        cols = base_feats + [f for f in pruned if f != feat]
        oof = _oof_predictions(X, y, groups, cols, params, n_splits, calibrator="none")
        score = -_composite_score(y, oof, calibration_bins=10, min_cal_bin=50, weights=metric_weights, calib_tol=calib_tol)
        if score - full_score <= eps:
            pruned.remove(feat)  # removal didn't hurt noticeably
    return pruned


def holdout_check(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups,
    X_hold: pd.DataFrame,
    y_hold: pd.Series,
    *,
    use_cols: list[str],
    params: dict | None = None,
    n_splits: int = 5,
):
    """Prints CV vs hold-out AUC to verify generalisation of *use_cols*."""

    # ─── Early exit if hold-out slice is empty ----------------------------
    if X_hold.empty or y_hold.empty:
        print("⚠️  Hold-out slice empty – skipping hold-out generalisation check.")
        return

    params = {**DEFAULT_LGB_PARAMS, **(params or {})}

    # CV score on training portion
    oof = _oof_predictions(X_train, y_train, groups, use_cols, params, n_splits, calibrator="none")
    auc_cv = roc_auc_score(y_train, oof)

    # Single fit on full train → evaluate on hold-out
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train[use_cols], y_train)
    preds_hold = model.predict_proba(X_hold[use_cols])[:, 1]
    auc_hold = roc_auc_score(y_hold, preds_hold)

    retained = 100.0 * auc_hold / auc_cv if auc_cv else float("nan")
    print(
        f"CV-AUC: {auc_cv:.4f} | Hold-out AUC: {auc_hold:.4f} (retained {retained:.1f} %)"
    )
