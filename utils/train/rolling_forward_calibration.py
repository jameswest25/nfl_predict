# utils/train/rolling_forward_calibration.py
"""
Rolling-forward calibration & performance harness
=================================================

Goal
----
Simulate "how the model would have performed if I had used it every MLB day", for any
configured problem (e.g. `gets_hit`). For each day *t*:

1. **Train** on all data strictly before *t* (chronological walk-forward).
2. **Predict** on day *t*.
3. **Store predictions** (probabilities, labels, metadata) and split into
   correct / incorrect Parquet files (with **full 2000+ feature set** re-attached).
4. **Update rolling calibration diagnostics**:
   - cumulative obs − exp line
   - 30-day rolling ECE / MCE (window configurable)
   - reliability (calibration) curve bins with Wilson CIs
5. (Optional) Generate plots per day or at the end.

Outputs (default locations)
---------------------------
```
output/rolling_eval/<problem>/
    predictions/
        preds_YYYY-MM-DD.parquet          # all preds for that day (full features + y, p)
        correct_YYYY-MM-DD.parquet        # subset correct
        incorrect_YYYY-MM-DD.parquet      # subset incorrect
    metrics/
        daily_metrics.csv                 # one row per day with obs-exp, ece, etc.
        bin_stats.parquet                 # exploded reliability bins across days
    plots/
        cumulative_obs_exp.png
        rolling_ece.png
        reliability_<last_day>.png
```

CLI usage
---------
```bash
python -m utils.train.rolling_forward_calibration \
    --problem gets_hit \
    --start-date 2024-04-01 \
    --end-date 2024-10-01 \
    --cv 5 \
    --feature-list output/feature_tester/greedy_gets_hit_20250724.csv \
    --calibrator isotonic \
    --window-days 30 \
    --save-plots
```

Integration assumptions
-----------------------
- We reuse the existing `ModelTrainer` (same as in `feature_tester_bridge.py`).
- `ModelTrainer` exposes:
  * `load_data()` → raw df
  * `split_data_chronologically(df)` → train/val/test slices (not used directly here)
  * `_fit_feature_artifacts(df_train, problem_cfg)`
  * `_apply_feature_artifacts(df_slice, problem_name)` → (X_slice, y_slice)
- `training.yaml` defines problems; we fetch the chosen one.
- Feature subset (the ~10 selected by greedy search) is optional. If omitted we use all columns from artifacts.

Implementation notes
--------------------
- To avoid refitting LightGBM hyperparams every day, we just use `DEFAULT_LGB_PARAMS` from `feature_tester.py` or
  the problem's `lightgbm_params` in YAML (you can switch via flag).
- We calibrate probabilities inside each day using the same scheme as your CV code (per-fold within train set), or we
  skip per-day calibration and just rely on the model's raw output. Flag `--calibrator` controls.
- For speed, we can cache artefacts across days because feature engineering may depend on entire history. Here, to 100% avoid leakage,
  we re-fit artefacts on the *training slice* of each day. (If this is too slow, implement a cumulative artefact fitter.)

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit

# Local imports (reusing your code base)
try:
    from utils.train.train import ModelTrainer
except ModuleNotFoundError:
    from pipeline.train import ModelTrainer

from utils.train.feature_tester import (
    _oof_predictions,
    DEFAULT_LGB_PARAMS,
    _ece,
    _calibration_bins,
    _brier_decomposition,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_reliability_bins(y: pd.Series, p: np.ndarray, n_bins: int = 10, min_bin: int = 50):
    """Wrapper returning the same structure as _calibration_bins but as plain DataFrame."""
    tbl = _calibration_bins(y, p, n_bins=n_bins, min_bin=min_bin)
    # unify dtype
    if not len(tbl):
        return pd.DataFrame(columns=["bin", "n", "emp_rate", "mean_pred", "wilson_low", "wilson_high", "abs_gap"])
    # Rename for consistency
    tbl = tbl.rename(columns={"mean_pred": "pred", "emp_rate": "emp"})
    return tbl


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    low, high = proportion_confint(k, n, method="wilson")
    return low, high


def fit_calibrator(train_p: np.ndarray, train_y: np.ndarray, method: str):
    if method == "none":
        return lambda x: x
    if method == "platt":
        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        p_clip = np.clip(train_p, 1e-6, 1 - 1e-6)
        lr.fit(logit(p_clip).reshape(-1, 1), train_y)
        return lambda p: expit(lr.coef_[0, 0] * logit(np.clip(p, 1e-6, 1 - 1e-6)) + lr.intercept_[0])
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(train_p, train_y)
        return ir.predict
    raise ValueError(f"Unknown calibrator {method}")


def _load_yaml_features(config_path: str, problem_name: str, all_cols: list[str]) -> list[str]:
    """
    Pull the feature list from training.yaml for the given problem.
    Uses `feature_prefixes_to_include` and `other_features_to_include`.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    problem = next(p for p in cfg["problems"] if p["name"] == problem_name)

    prefixes = problem.get("feature_prefixes_to_include", []) or []
    extras   = problem.get("other_features_to_include", []) or []

    # any column that starts with one of the prefixes
    prefixed_cols = [c for c in all_cols if any(c.startswith(pfx) for pfx in prefixes)]

    # keep order & dedupe
    feats = list(dict.fromkeys([*extras, *prefixed_cols]))
    return feats


@dataclass
class RollingConfig:
    problem: str
    start_date: str
    end_date: str | None
    cv: int
    window_days: int
    n_bins: int
    min_bin: int
    calibrator: str
    feature_list_path: str | None
    output_dir: Path
    save_plots: bool
    use_yaml_params: bool


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run_walkforward(cfg: RollingConfig, config_yaml: str | Path = "config/training.yaml"):
    output_dir = cfg.output_dir
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load full raw data once
    # ------------------------------------------------------------------
    tr = ModelTrainer(str(config_yaml))
    full_df = tr.load_data()
    time_col = tr.config["data_split"]["time_col"]

    # Filter date range
    full_df[time_col] = pd.to_datetime(full_df[time_col])
    start = pd.to_datetime(cfg.start_date)
    end = pd.to_datetime(cfg.end_date) if cfg.end_date else full_df[time_col].max()
    df_slice = full_df[(full_df[time_col] >= start) & (full_df[time_col] <= end)].copy()

    # For problem cfg
    prob_cfg = next(p for p in tr.config["problems"] if p["name"] == cfg.problem)
    tr.target_col = prob_cfg["target_col"]

    # Optional feature subset
    selected_feats = None
    if cfg.feature_list_path and Path(cfg.feature_list_path).exists():
        sel_df = pd.read_csv(cfg.feature_list_path)
        # Expect column 'added_feature' if it's the greedy CSV
        if "added_feature" in sel_df.columns:
            selected_feats = sel_df["added_feature"].tolist()
        else:
            # Fallback: assume single column of feature names
            selected_feats = sel_df.iloc[:, 0].tolist()
        logger.info("Using %d selected features from %s", len(selected_feats), cfg.feature_list_path)
    else:
        # fallback to YAML-defined features
        all_cols = df_slice.columns.tolist()
        selected_feats = _load_yaml_features(config_yaml, cfg.problem, all_cols)
        if not selected_feats:
            raise RuntimeError(
                "No features found in training.yaml for problem "
                f"{cfg.problem}. Provide --feature-list or update YAML."
            )
        logger.info("Using %d features from training.yaml (fallback).", len(selected_feats))
        # create a dummy sel_df just for downstream compatibility if needed
        sel_df = pd.DataFrame({"added_feature": selected_feats})

    # Choose model params
    if cfg.use_yaml_params:
        lgb_params = prob_cfg.get("lightgbm_params", {})
        params = {**DEFAULT_LGB_PARAMS, **lgb_params}
    else:
        params = DEFAULT_LGB_PARAMS.copy()

    # ------------------------------------------------------------------
    # 2. Iterate day by day (walk-forward)
    # ------------------------------------------------------------------
    unique_days = df_slice[time_col].drop_duplicates().sort_values()

    daily_rows = []           # summary metrics per day
    bin_rows = []             # exploded reliability-bin rows per day

    # cumulative trackers (expanding)
    cum_expected = 0.0
    cum_observed = 0.0

    for i, day in enumerate(unique_days):
        day = pd.to_datetime(day)
        logger.info("Processing day %s (%d/%d)", day.date(), i + 1, len(unique_days))

        # Train up to day-1 (strictly before 'day')
        df_train = full_df[full_df[time_col] < day]
        df_test_day = full_df[full_df[time_col] == day]

        if df_train.empty or df_test_day.empty:
            logger.warning("Skipping day %s (empty train or test)", day.date())
            continue

        # Fit artefacts on *this train slice*
        tr_day = ModelTrainer(str(config_yaml))
        tr_day.target_col = prob_cfg["target_col"]
        tr_day._fit_feature_artifacts(df_train, prob_cfg)

        # Apply features to train and day
        X_tr, y_tr = tr_day._apply_feature_artifacts(df_train, cfg.problem)
        X_te, y_te = tr_day._apply_feature_artifacts(df_test_day, cfg.problem)

        # Reattach FULL features to predictions for debug
        # Align on index, ensure no duplicates
        X_full_test = X_te.copy()
        if selected_feats is not None:
            use_cols = [c for c in selected_feats if c in X_tr.columns]
        else:
            use_cols = X_tr.columns.tolist()

        # CV OOF for calibrator fit (inside _oof_predictions we can calibrate, but we need the
        # calibrator to apply to final model if we train single model). We'll just train model on full train.
        # Option 1: simple fit -> predict -> calibrate using held-out fold preds. We'll reuse _oof_predictions.
        oof_tr = _oof_predictions(X_tr, y_tr, groups=pd.factorize(df_train["game_pk"])[0],
                                  use_cols=use_cols, params=params, n_splits=cfg.cv, calibrator="none")

        # Fit calibrator on train oof if required
        cal_fn = fit_calibrator(oof_tr, y_tr.values, cfg.calibrator)

        # Fit final model on full training set
        import lightgbm as lgb
        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(X_tr[use_cols], y_tr)
        p_raw = mdl.predict_proba(X_te[use_cols])[:, 1]
        p_cal = cal_fn(p_raw)

        # Store predictions with full features attached
        preds_day = X_full_test.copy()
        preds_day["p_pred"] = p_cal
        preds_day["y"] = y_te.values
        preds_day["game_date"] = day

        # Save Parquet for all, and split
        preds_path = output_dir / "predictions" / f"preds_{day.date()}.parquet"
        preds_day.to_parquet(preds_path, index=False)

        correct_mask = preds_day["y"] == 1
        correct_df = preds_day[correct_mask]
        incorrect_df = preds_day[~correct_mask]
        correct_df.to_parquet(output_dir / "predictions" / f"correct_{day.date()}.parquet", index=False)
        incorrect_df.to_parquet(output_dir / "predictions" / f"incorrect_{day.date()}.parquet", index=False)

        # --- Metrics for the day (windowed & expanding) -----------------
        expected_day = preds_day["p_pred"].sum()
        observed_day = preds_day["y"].sum()
        cum_expected += expected_day
        cum_observed += observed_day

        # Rolling window slice (last window_days of history including today)
        # We'll use previously stored preds to build window
        # Simplest: append to a CSV/DF; re-read last N days, compute ECE etc.
        # Here we just compute on today's preds (not great), AND at the end we recompute global tables.
        # So we also accumulate all preds in memory (if memory allows); else read from parquet.

        # Reliability bins on *today only* (for debug). For rolling windows you'll re-aggregate later.
        rel_tbl = compute_reliability_bins(preds_day["y"], preds_day["p_pred"], n_bins=cfg.n_bins, min_bin=cfg.min_bin)
        rel_tbl["as_of_date"] = day
        bin_rows.append(rel_tbl)

        # ECE on today slice
        ece_today = _ece(preds_day["y"], preds_day["p_pred"], n_bins=cfg.n_bins)

        daily_rows.append({
            "as_of_date": day,
            "N": len(preds_day),
            "expected_hits": expected_day,
            "observed_hits": observed_day,
            "obs_minus_exp": observed_day - expected_day,
            "cum_expected": cum_expected,
            "cum_observed": cum_observed,
            "cum_obs_minus_exp": cum_observed - cum_expected,
            "logloss": log_loss(preds_day["y"], preds_day["p_pred"]),
            "brier": brier_score_loss(preds_day["y"], preds_day["p_pred"]),
            "auc": roc_auc_score(preds_day["y"], preds_day["p_pred"]) if preds_day["y"].nunique() == 2 else np.nan,
            "ece": ece_today,
        })

    # ------------------------------------------------------------------
    # 3. Save aggregated metrics
    # ------------------------------------------------------------------
    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(output_dir / "metrics" / "daily_metrics.csv", index=False)

    if bin_rows:
        bins_df = pd.concat(bin_rows, ignore_index=True)
        bins_df.to_parquet(output_dir / "metrics" / "bin_stats.parquet", index=False)

    # ------------------------------------------------------------------
    # 4. Optionally generate plots
    # ------------------------------------------------------------------
    if cfg.save_plots and not daily_df.empty:
        # Cumulative obs-exp
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(daily_df["as_of_date"], daily_df["cum_obs_minus_exp"], label="Cumulative (Obs - Exp)")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"Cumulative Obs - Exp ({cfg.problem})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Obs - Exp")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / "cumulative_obs_exp.png", dpi=150)
        plt.close(fig)

        # Rolling ECE (window_days)
        wd = cfg.window_days
        if wd > 1:
            # read all preds back (memory ok? 200k rows is fine). We'll recompute ECE on sliding window
            all_preds = []
            for f in sorted((output_dir / "predictions").glob("preds_*.parquet")):
                tmp = pd.read_parquet(f)
                all_preds.append(tmp[["game_date", "p_pred", "y"]])
            all_preds = pd.concat(all_preds, ignore_index=True)
            all_preds["game_date"] = pd.to_datetime(all_preds["game_date"])
            all_preds = all_preds.sort_values("game_date")

            ece_dates = []
            ece_vals = []
            for d in daily_df["as_of_date"]:
                cutoff = d - pd.Timedelta(days=wd)
                win = all_preds[(all_preds["game_date"] <= d) & (all_preds["game_date"] > cutoff)]
                if len(win) < 100:
                    ece_vals.append(np.nan)
                    ece_dates.append(d)
                    continue
                ece_vals.append(_ece(win["y"], win["p_pred"], n_bins=cfg.n_bins))
                ece_dates.append(d)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(ece_dates, ece_vals, label=f"{wd}-day rolling ECE")
            ax.set_title(f"Rolling ECE ({cfg.problem})")
            ax.set_xlabel("Date")
            ax.set_ylabel("ECE")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "plots" / "rolling_ece.png", dpi=150)
            plt.close(fig)

        # Reliability curve on last day
        last_day = daily_df["as_of_date"].max()
        last_preds = pd.read_parquet(output_dir / "predictions" / f"preds_{last_day.date()}.parquet")
        rel_last = compute_reliability_bins(last_preds["y"], last_preds["p_pred"], n_bins=cfg.n_bins, min_bin=cfg.min_bin)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect")
        for _, r in rel_last.iterrows():
            ax.errorbar(r["pred"], r["emp"], yerr=[[r["emp"] - r["wilson_low"]], [r["wilson_high"] - r["emp"]]], fmt="o")
        ax.set_title(f"Reliability (last day {last_day.date()})")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical hit rate")
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / f"reliability_{last_day.date()}.png", dpi=150)
        plt.close(fig)

    logger.info("Walk-forward evaluation complete. Outputs in %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Rolling-forward calibration/backtest utility")
    p.add_argument("--problem", required=True, help="Problem name as in training.yaml (e.g. gets_hit)")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD (inclusive). Default: max date in data")
    p.add_argument("--cv", type=int, default=5, help="K folds for OOF calibrator fit")
    p.add_argument("--window-days", type=int, default=30, help="Rolling window for drift stats (ECE etc.)")
    p.add_argument("--n-bins", type=int, default=10, help="Calibration bins")
    p.add_argument("--min-bin", type=int, default=50, help="Min samples per bin to keep")
    p.add_argument("--calibrator", choices=["none", "platt", "isotonic"], default="isotonic")
    p.add_argument("--feature-list", dest="feature_list_path", default=None,
                   help="CSV file with selected features (greedy output). If missing, fall back to training.yaml.")
    p.add_argument("--output-dir", default="output/rolling_eval", help="Base dir for outputs")
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--use-yaml-params", action="store_true", help="Use lightgbm_params from YAML instead of DEFAULT_LGB_PARAMS")
    p.add_argument("--config", default="config/training.yaml", help="Path to training.yaml")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = RollingConfig(
        problem=args.problem,
        start_date=args.start_date,
        end_date=args.end_date,
        cv=args.cv,
        window_days=args.window_days,
        n_bins=args.n_bins,
        min_bin=args.min_bin,
        calibrator=args.calibrator,
        feature_list_path=args.feature_list_path,
        output_dir=Path(args.output_dir) / args.problem,
        save_plots=args.save_plots,
        use_yaml_params=args.use_yaml_params,
    )

    run_walkforward(cfg, config_yaml=args.config)


if __name__ == "__main__":
    main()
