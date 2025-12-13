# utils/train/feature_tester_bridge.py
"""Feature‑evaluation CLI / helper (v2)

Changes vs v1
-------------
1. **Composite scorer** – greedy search now optimises *all* four metrics
   (log‑loss ↓, Brier ↓, AUC ↑, ECE ↓) by turning each into a relative
   %‑improvement and taking the mean.
2. **Sophisticated stopping criteria** – stops when improvement > min_delta_abs OR 
   > min_delta_rel (relative) OR > c_se * SE (noise-aware), or after max_fails 
   consecutive rejections, or when rolling median improvement over stop_window steps 
   is too low.
3. **Candidate default** – if `--candidates` is omitted, the search will
   automatically consider **every column not already in `base_features`**.
4. **Output location** – defaults to `output/feature_tester/<auto‑name>.csv`
   when `--out` is not supplied.

CLI quick start
---------------
```bash
# greedy forward search over *all remaining* columns
python -m utils.train.feature_tester_bridge greedy --problem gets_hit

# impact test two specific columns
python -m utils.train.feature_tester_bridge impact \
       --problem gets_hit \
       --features is_same_hand platoon_penalty
```
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence
import argparse
import sys
import logging
from datetime import datetime
import os

import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import json

try:
    from utils.train.train import ModelTrainer  # original location (older versions)
except ModuleNotFoundError:
    # Fallback to the updated location if the legacy path is missing
    from pipeline.train import ModelTrainer

from utils.train.feature_tester import (
    evaluate_feature_impact,
    check_collinearity,
    forward_feature_search,
    _backward_trim,
    holdout_check,
    _brier_decomposition,
    _calibration_bins,
    _ece,
)

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_CFG = Path("config/training.yaml")
_DEFAULT_OUTDIR = Path("output/feature_tester")
_DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Calibration helpers (mirrored from feature_tester.py for CLI access)         
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_METRICS = {
    'logloss': {'lower_better': True},
    'brier'  : {'lower_better': True},
    'auc'    : {'lower_better': False},
    'ece'    : {'lower_better': True},
    'mce'    : {'lower_better': True},
    'brier_reliability': {'lower_better': True},
    'brier_resolution' : {'lower_better': False},  # we *like* sharpness
}

def load_metric_weights(args, base_fold_metrics=None):
    """
    If user supplies --metric-weights, use them.
    Else auto-scale: weight = 1 / SE(metric), where SE computed from baseline fold metrics.
    If SE missing/zero, fall back to 1.0.
    """
    if args.metric_weights:
        return json.loads(args.metric_weights)
    if not base_fold_metrics:
        # Fallback – equal weights
        return {m: 1.0 for m in DEFAULT_METRICS}

    weights = {}
    for m in DEFAULT_METRICS:
        vals = base_fold_metrics.get(m, None)
        if vals is None or len(vals) < 2:
            weights[m] = 1.0
        else:
            se = np.std(vals, ddof=1) / np.sqrt(len(vals))
            weights[m] = 1.0 / se if se > 1e-9 else 1.0
    return weights

# ─────────────────────────────────────────────────────────────────────────────
# internal helpers                                                             
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_design_matrices(
    *,
    config_path: str | Path,
    problem_name: str,
    force_features: List[str] | None = None,
    extra_cols: list[str] | None = None,
):
    """Return (X, y, groups) using *exactly* the artefact logic in train.py."""
    tr = ModelTrainer(str(config_path))
    prob = next((p for p in tr.config["problems"] if p["name"] == problem_name), None)
    if prob is None:
        raise ValueError(f"Problem '{problem_name}' not found in training.yaml")

    if force_features:
        # Stash list for later injection just before artefact fitting
        prob["__force_features__"] = force_features

    df = tr.load_data()
    df_train, _, _ = tr.split_data_chronologically(df)

    tr.target_col = prob["target_col"]

    # ── NEW: force-include every requested feature into artefacts ───────────
    if "__force_features__" in prob:
        extra_force = prob.pop("__force_features__")
        prob.setdefault("other_features_to_include", []).extend(extra_force)
        # deduplicate while preserving order
        prob["other_features_to_include"] = list(dict.fromkeys(prob["other_features_to_include"]))

    # Fit artefacts now that the problem config has been finalised
    tr._fit_feature_artifacts(df_train, prob)

    # ─── Inject any extra columns requested by caller (legacy path) ─────────
    if extra_cols:
        missing = [c for c in extra_cols if c not in tr.feature_columns]
        if missing:
            tr.feature_columns.extend(missing)

    X_train, y_train = tr._apply_feature_artifacts(df_train, problem_name)

    # Columns that are still absent (brand-new raw features) → pull directly
    # from the original DataFrame and concat as-is.  LightGBM can handle NaNs.
    if extra_cols:
        truly_missing = [c for c in extra_cols if c not in X_train.columns]
        if truly_missing:
            X_train = pd.concat(
                [X_train, df_train[truly_missing].reset_index(drop=True)],
                axis=1,
            )

    # ─── Dtype hygiene for LightGBM ------------------------------------------------
    # 1) Datetime columns → numeric int64 (epoch ns) to avoid dtype promotion errors.
    dt_cols = X_train.select_dtypes(include=["datetime64[ns]"]).columns
    if dt_cols.any():
        X_train[dt_cols] = X_train[dt_cols].astype("int64")

    # 2) Raw object columns → pandas 'category' so LightGBM treats them correctly.
    obj_cols = X_train.select_dtypes(include=["object"]).columns
    if obj_cols.any():
        for _c in obj_cols:
            X_train[_c] = X_train[_c].astype("category")

    # NFL uses `game_id`; MLB legacy uses `game_pk`.
    group_col = "game_id" if "game_id" in df_train.columns else "game_pk"
    groups = pd.factorize(df_train[group_col])[0]
    return X_train, y_train, groups

# ─────────────────────────────────────────────────────────────────────────────
# public API                                                                  
# ─────────────────────────────────────────────────────────────────────────────

def impact_report(*, problem_name: str, feature_names: List[str], config_path: str | Path = _DEFAULT_CFG, n_splits: int = 5, calibration_bins: int = 10, min_cal_bin: int = 50, calibrator: str = "none", metric_weights: dict | None = None) -> pd.DataFrame:
    X, y, groups = _prepare_design_matrices(config_path=config_path, problem_name=problem_name)
    rep = evaluate_feature_impact(X=X, y=y, groups=groups, feature_names=feature_names, n_splits=n_splits, calibration_bins=calibration_bins, min_cal_bin=min_cal_bin)
    return rep.to_df()


def collinearity_report(*, problem_name: str, feature_names: List[str], base_features: Iterable[str] | None = None, config_path: str | Path = _DEFAULT_CFG):
    from utils.train.feature_tester import check_collinearity  # delayed import to avoid heavy deps if not needed
    X, _, _ = _prepare_design_matrices(config_path=config_path, problem_name=problem_name)
    return check_collinearity(X, feature_names, base_features=base_features or X.columns)


def greedy_search(
    *,
    problem_name: str,
    candidate_features: List[str] | None = None,
    base_features: Iterable[str] | None = None,
    scorer: str = "composite",
    max_steps: int | None = None,
    config_path: str | Path = _DEFAULT_CFG,
    n_splits: int = 3,
    progress_wrapper=lambda x, **k: x,
    force_features: List[str] | None = None,
    log_level: int = logging.INFO,
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
    """Forward add‑one search using composite metric.

    Stops when mean %%‑gain across (logloss, brier, auc, ece) < ``min_delta``.
    """
    X, y, groups = _prepare_design_matrices(
        config_path=config_path,
        problem_name=problem_name,
        force_features=force_features,
        extra_cols=candidate_features,
    )

    base_feats = list(base_features) if base_features is not None else []
    # ------------------------------------------------------------------
    # Build exclusion set: true target, *any* problem targets in YAML, plus
    # a small safety list of obvious label derivatives (user-supplied).
    # ------------------------------------------------------------------
    target_col = y.name  # canonical label for this problem

    import yaml as _yaml
    try:
        with open(config_path, "r") as _f:
            _cfg = _yaml.safe_load(_f)
            yaml_targets = {
                p.get("target_col") for p in _cfg.get("problems", []) if p.get("target_col")
            }
    except Exception:
        yaml_targets = set()

    extra_exclude = {  # hand-curated once; extend as needed
        "hit_count",
        "bases_sum",
        "is_hit",  # prevent single-hit label from leaking into multi-hit tasks
        "actual_plate_appearances"
    }

    exclude_cols = {target_col, *yaml_targets, *extra_exclude}

    if candidate_features is None:
        candidate_pool = [c for c in X.columns if c not in base_feats and c not in exclude_cols]
    else:
        candidate_pool = [c for c in candidate_features if c not in base_feats and c not in exclude_cols]

    if not candidate_pool:
        raise ValueError("No candidate columns found in the training matrix.")

    # ---- Compute baseline fold metrics for auto-weighting (if needed) ----
    # We'll reuse _oof_predictions to get preds and then split per fold to get metric arrays.
    from utils.train.feature_tester import _oof_predictions, DEFAULT_LGB_PARAMS, _ece, _brier_decomposition, _calibration_bins
    params = DEFAULT_LGB_PARAMS.copy()
    oof_base = _oof_predictions(X, y, groups, base_feats, params, n_splits, calibrator=calibrator)

    # Build per-fold metrics
    fold_metrics = {m: [] for m in DEFAULT_METRICS}
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, va_idx in gkf.split(X, y, groups):
        y_fold = y.iloc[va_idx]
        p_fold = oof_base[va_idx]
        fold_metrics['logloss'].append(float(log_loss(y_fold, p_fold)))
        fold_metrics['brier'].append(float(brier_score_loss(y_fold, p_fold)))
        try:
            fold_metrics['auc'].append(float(roc_auc_score(y_fold, p_fold)))
        except ValueError:
            fold_metrics['auc'].append(np.nan)
        fold_metrics['ece'].append(float(_ece(y_fold, p_fold)))
        calib_tbl = _calibration_bins(y_fold, p_fold, n_bins=calibration_bins, min_bin=min_cal_bin)
        bdec = _brier_decomposition(y_fold, p_fold, n_bins=calibration_bins)
        fold_metrics['mce'].append(float(calib_tbl.attrs.get('MCE', 0.0)))
        fold_metrics['brier_reliability'].append(float(bdec['reliability']))
        fold_metrics['brier_resolution'].append(float(bdec['resolution']))

    # Auto weights if not provided
    if metric_weights is None:
        metric_weights = load_metric_weights(argparse.Namespace(metric_weights=None), fold_metrics)

    # Delegate to the optimised routine in utils.train.feature_tester
    from utils.train.feature_tester import forward_feature_search  # local import to avoid heavy deps when not needed

    return forward_feature_search(
        X=X,
        y=y,
        groups=groups,
        candidate_features=candidate_pool,
        base_features=base_feats,
        scorer=scorer,
        n_splits=n_splits,
        max_steps=max_steps,
        progress_wrapper=progress_wrapper,
        log_level=log_level,
        calibration_bins=calibration_bins,
        min_cal_bin=min_cal_bin,
        calibrator=calibrator,
        metric_weights=metric_weights,
        calib_tol=calib_tol,
        min_delta_abs=min_delta_abs,
        min_delta_rel=min_delta_rel,
        c_se=c_se,
        max_fails=max_fails,
        stop_window=stop_window,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CLI                                                                         
# ─────────────────────────────────────────────────────────────────────────────

def _save_or_print(df: pd.DataFrame, out: str | Path | None, prefix: str):
    if out is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = _DEFAULT_OUTDIR / f"{prefix}_{timestamp}.csv"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved report → {out}")


def _main(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Feature evaluation CLI (composite scorer)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # impact
    spi = sub.add_parser("impact")
    spi.add_argument("--problem", required=True)
    spi.add_argument("--features", nargs="+", required=True)
    spi.add_argument("--config", default=_DEFAULT_CFG)
    spi.add_argument("--out")
    spi.add_argument("--calibration-bins", type=int, default=10, help="Number of bins for calibration curves")
    spi.add_argument("--min-cal-bin", type=int, default=50, help="Minimum samples per calibration bin")
    spi.add_argument("--save-calibration", type=str, default=None,
                    help="Path (CSV/Parquet) to dump calibration bin table for the final model")
    spi.add_argument("--calibrator", choices=['none','platt','isotonic'], default='none',
                    help="Fit a per-fold probability calibrator on train and apply to valid")
    spi.add_argument("--metric-weights", type=str, default=None,
                    help="JSON dict (metric->weight). Overrides defaults for composite score.")

    # collinear
    spc = sub.add_parser("collinear")
    spc.add_argument("--problem", required=True)
    spc.add_argument("--features", nargs="+", required=True)
    spc.add_argument("--config", default=_DEFAULT_CFG)
    spc.add_argument("--out")

    # greedy
    spg = sub.add_parser("greedy")
    spg.add_argument("--problem", required=True)
    spg.add_argument("--candidates", nargs="*")  # explicit list
    spg.add_argument("--all-cols", action="store_true", help="use every column present in the Parquet schema as candidates")
    spg.add_argument("--config", default=_DEFAULT_CFG)
    spg.add_argument("-v", "--verbose", action="store_true", help="chatty INFO logs for every accepted feature")
    spg.add_argument("--progress", action="store_true", help="show tqdm progress bar per search step")
    spg.add_argument("--min-delta-abs", type=float, default=0.0,
                    help='Absolute composite gain floor.')
    spg.add_argument("--min-delta-rel", type=float, default=1e-3,
                    help='Relative gain floor (abs_gain / prev_score).')
    spg.add_argument("--c-se", type=float, default=1.8,
                    help='Multiplier on SE of fold gains for noise-aware stop.')
    spg.add_argument("--max-fails", type=int, default=4,
                    help='Consecutive rejected features before stopping.')
    spg.add_argument("--stop-window", type=int, default=5,
                    help='Window size for optional rolling-median stop check.')
    spg.add_argument("--max-steps", type=int)
    spg.add_argument("--cv", type=int, default=3, metavar="K", help="K-fold GroupCV (default: 3)")
    spg.add_argument("--out")
    spg.add_argument("--post-trim", action="store_true", help="run backward trim + hold-out check after search")
    spg.add_argument("--log-level", default="INFO", help="python logging level (default INFO)")
    spg.add_argument("--calibration-bins", type=int, default=10, help="Number of bins for calibration curves")
    spg.add_argument("--min-cal-bin", type=int, default=50, help="Minimum samples per calibration bin")
    spg.add_argument("--save-calibration", type=str, default=None,
                    help="Path (CSV/Parquet) to dump calibration bin table for the final model")
    spg.add_argument("--calibrator", choices=['none','platt','isotonic'], default='none',
                    help="Fit a per-fold probability calibrator on train and apply to valid")
    spg.add_argument("--metric-weights", type=str, default=None,
                    help="JSON dict (metric->weight). Overrides defaults for composite score.")
    spg.add_argument("--calib-tol", type=float, default=0.005,
                    help="Max allowed relative worsening for calibration metrics (ece/mce/reliability). Default 0.005 (0.5%%).")

    spg.add_argument(
        "--scorer",
        default="composite",
        choices=["logloss", "brier", "auc", "composite", "logloss_brier"],
        help="metric to optimise (default: composite)",
    )

    spg.add_argument(
        "--base-features",
        nargs="*",
        help="freeze these features as the starting baseline for greedy search",
    )

    args = p.parse_args(argv)

    if args.cmd == "impact":
        X, y, groups = _prepare_design_matrices(config_path=args.config, problem_name=args.problem)
        rep = evaluate_feature_impact(
            X=X, y=y, groups=groups, 
            feature_names=args.features, 
            n_splits=5,
            calibration_bins=args.calibration_bins,
            min_cal_bin=args.min_cal_bin
        )
        df = rep.to_df()
        _save_or_print(df, args.out, f"impact_{args.problem}")
        
        # Save calibration data if requested
        if args.save_calibration and rep.calibration_data:
            rep.save_calibration_data(args.save_calibration, which="with_feature")

    elif args.cmd == "collinear":
        df = collinearity_report(problem_name=args.problem, feature_names=args.features, config_path=args.config)
        _save_or_print(df, args.out, f"collinear_{args.problem}")

    elif args.cmd == "greedy":
        # Suppress multiprocessing resource_tracker warnings (joblib)
        os.environ.setdefault("JOBLIB_NO_RESOURCE_TRACKER", "1")

        if args.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # Optional progress bar using tqdm -----------------------------------
        try:
            from tqdm import tqdm  # type: ignore
            _tqdm = tqdm if args.progress else (lambda x, **k: x)
        except ModuleNotFoundError:
            if args.progress:
                print(
                    "⚠️  tqdm not installed → run `pip install tqdm` or omit --progress",
                    file=sys.stderr,
                )
            _tqdm = lambda x, **k: x

        # Build candidate list ------------------------------------------------
        cand: List[str] | None = None
        if args.all_cols:
            try:
                import pyarrow.parquet as pq  # type: ignore
                cand = pq.read_schema("data/processed/statcast/statcast_processed.parquet").names
            except Exception as e:
                print(f"⚠️  Could not load Parquet schema for --all-cols: {e}", file=sys.stderr)
                sys.exit(1)
        elif args.candidates:
            cand = args.candidates

        # Remove problem target column from candidates (e.g. 'is_hit') ---------
        if cand:
            # Will filter again later using y.name to be extra safe.
            problem_target = args.problem.replace("gets_", "is_") if args.problem.startswith("gets_") else None
            if problem_target:
                cand = [c for c in cand if c != problem_target]

        base_feats: list[str] = []  # extend when baseline columns are supported via CLI

        import logging as _lg
        log_lvl = getattr(_lg, args.log_level.upper(), _lg.INFO)

        df = greedy_search(
            problem_name=args.problem,
            candidate_features=cand,
            force_features=cand,
            base_features=args.base_features,
            scorer=args.scorer,
            n_splits=args.cv,
            max_steps=args.max_steps,
            config_path=args.config,
            progress_wrapper=_tqdm,
            log_level=log_lvl,
            calibration_bins=args.calibration_bins,
            min_cal_bin=args.min_cal_bin,
            calibrator=args.calibrator,
            metric_weights=json.loads(args.metric_weights) if args.metric_weights else None,
            calib_tol=args.calib_tol,
            min_delta_abs=args.min_delta_abs,
            min_delta_rel=args.min_delta_rel,
            c_se=args.c_se,
            max_fails=args.max_fails,
            stop_window=args.stop_window,
        )

        # ─── Optional backward-trim + hold-out verification ─────────────
        if args.post_trim and not df.empty:
            accepted = df["added_feature"].tolist()
            
            # Get metric weights for backward trim
            mw = json.loads(args.metric_weights) if args.metric_weights else None

            # Re-create train matrix to run trim on the same folds
            X_tr, y_tr, groups_tr = _prepare_design_matrices(
                config_path=args.config,
                problem_name=args.problem,
                force_features=None,
                extra_cols=accepted,
            )

            pruned = _backward_trim(
                X_tr,
                y_tr,
                groups_tr,
                accepted_feats=accepted,
                base_feats=base_feats,
                n_splits=args.cv,
                metric_weights=mw,
                calib_tol=args.calib_tol,
            )
            if not pruned:
                pruned = accepted  # ensure at least baseline feature list
            print(f"🪓  backward-trimmed {len(accepted)} → {len(pruned)} features")

            # Prepare hold-out slice processed through the SAME artifact pipeline
            tr_hold = ModelTrainer(str(args.config))
            # identify problem config to fit artifacts
            prob_cfg = next(p for p in tr_hold.config["problems"] if p["name"] == args.problem)
            tr_hold.target_col = prob_cfg["target_col"]

            df_full = tr_hold.load_data()
            df_train, _, df_test = tr_hold.split_data_chronologically(df_full)

            # Fit artifacts on training slice using identical logic
            tr_hold._fit_feature_artifacts(df_train, prob_cfg)
            X_hold, _ = tr_hold._apply_feature_artifacts(df_test, args.problem)
            # Align columns: add missing as NaN, preserve order
            X_hold = X_hold.reindex(columns=X_tr.columns)
            y_hold = df_test[tr_hold.target_col]

            holdout_check(
                X_tr,
                y_tr,
                groups_tr,
                X_hold,
                y_hold,
                use_cols=pruned,
                n_splits=args.cv,
            )

        _save_or_print(df, args.out, f"greedy_{args.problem}")

        # Save calibration data if requested and search found features
        if args.save_calibration and not df.empty:
            # Get the final feature set and compute calibration
            final_features = df["added_feature"].tolist()
            X_final, y_final, groups_final = _prepare_design_matrices(
                config_path=args.config,
                problem_name=args.problem,
                force_features=final_features,
            )
            
            # Compute final model predictions
            from utils.train.feature_tester import _oof_predictions, DEFAULT_LGB_PARAMS
            params = DEFAULT_LGB_PARAMS.copy()
            oof_final = _oof_predictions(X_final, y_final, groups_final, X_final.columns.tolist(), params, args.cv, calibrator=args.calibrator)
            
            # Generate calibration table
            calib_tbl = _calibration_bins(y_final, oof_final, n_bins=args.calibration_bins, min_bin=args.min_cal_bin)
            
            if len(calib_tbl):
                if args.save_calibration.endswith('.parquet'):
                    calib_tbl.to_parquet(args.save_calibration, index=False)
                else:
                    calib_tbl.to_csv(args.save_calibration, index=False)
                print(f"Saved calibration table → {args.save_calibration}")

if __name__ == "__main__":
    _main(sys.argv[1:])
