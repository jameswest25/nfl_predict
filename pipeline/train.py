import os
import random
import inspect
import sys
import logging
import math
import re


# Move logging to top and remove global thread pinning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path to allow module imports when running as script
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import yaml
import joblib
import optuna
import xgboost as xgb
import polars as pl
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from pathlib import Path
from datetime import datetime, date
import subprocess
import json
import argparse
import copy
from sklearn.calibration import CalibratedClassifierCV
import warnings
import gc

# New utility imports
from pandas.api.types import is_datetime64_any_dtype
from utils.train.model_factory import (
    get_task_type, is_classification, is_regression, get_model_instance, auto_spw
)
from utils.train.xgb_utils import (
    fit_model_es, get_best_iteration, compute_patience, predict_proba_batched, retrain_with_best_iter
)
from utils.train.data import load_feature_matrix
from utils.train.purged_group_time_series_split import PurgedGroupTimeSeriesSplit
from utils.train.purged_cv import compute_purged_group_splits
from utils.train.sample_weights import compute_sample_weights as compute_sample_weights_util
from utils.train.feature_artifacts import (
    FeatureArtifacts,
    fit_feature_artifacts as build_feature_artifacts,
    apply_feature_artifacts as transform_with_artifacts,
)
from utils.train.metrics import confidence_bins_report, compute_regression_metrics
from utils.train.calibration import (
    calibrate_and_select_threshold, EnsembleCalibratedModel,
    IsotonicCalibratedModel, BetaCalibratedModel, HistogramBinningCalibratedModel
)
from utils.train.conformal import regression_mean_calibrate, fit_split_conformal_regression, \
    IsotonicMeanCalibratedRegressor, SplitConformalRegressor

from utils.train.config_types import Paths, TrainingCfg
from utils.train.tuning import (
    tune_hyperparameters as tune_hyperparameters_ext,
    normalize_param_distributions,
    TuningConfigError,
)
from utils.train.seed import deterministic_seed, set_global_seed
from utils.train.run_manifest import make_run_id, write_manifest
from utils.train.team_total import TeamTotalAdjustedClassifier, TeamTotalConfig
from utils.feature.enrichment.asof import decision_cutoff_override, fallback_cutoff_hours, get_decision_cutoff_hours
from utils.collect.espn_injuries import collect_espn_injuries
from utils.general.constants import format_cutoff_label

PLAYER_GAME_DIR = Path("data/processed/player_game_by_week")

# Stage-2 modules
from utils.train.persist import (
    save_inference_artifacts,
    save_model_and_metrics,
    save_feature_importance,
    save_prediction_analysis,
    clean_output_dirs,
    save_pr_curves,
    _vdir,
)
from utils.train.data import load_feature_matrix, split_data_chronologically as split_chrono

from utils.train.conformal_composite import (
    fit_composite_conformal_sums,
    fit_team_conformal_sums,
    composite_sum_interval,
)
from utils.feature.core.leak_guard import DEFAULT_LEAK_POLICY, enforce_leak_guard
from utils.feature.core.targets import require_target_column

# Ignore specific, noisy warnings from dependencies
warnings.filterwarnings(
    "ignore",
    message=".*swapaxes.*",
    category=FutureWarning
)


def _infer_seasons_from_parquet(path: Path) -> list[int]:
    if not path.exists():
        return []
    try:
        scan = pl.scan_parquet(str(path))
        seasons_df = (
            scan.select(pl.col("season").drop_nulls().cast(pl.Int32).unique())
            .collect(streaming=True)
        )
        if seasons_df.is_empty():
            return []
        seasons = seasons_df.to_series().drop_nulls().to_list()
        return [int(s) for s in seasons if s is not None]
    except Exception as exc:
        logger.warning("Unable to infer seasons from %s: %s", path, exc)
        return []

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    pq = None
    pa = None

# Setup logging
logger = logging.getLogger(__name__)

# ========= end helpers =========

class ModelTrainer:
    def __init__(self, config_path='config/training.yaml', overrides: dict | None = None):
        # Set seeds for full reproducibility
        # (Stage-4) Global seed will be finalized per-problem; keep a temp default here.
        np.random.seed(42)
        random.seed(42)
        
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        overrides = overrides or {}
        
        # --- Stage-3: build convenience dataclasses ---
        self.paths = Paths(
            feature_matrix_path=Path(self.config['data']['feature_matrix_path']),
            model_dir=Path(self.config['data']['model_output_dir']),
            metric_dir=Path(self.config['data']['metrics_output_dir']),
            analysis_dir=Path(self.config['data']['metrics_output_dir']).parent / 'prediction_analysis',
        )
        active_cutoff_hours = float(get_decision_cutoff_hours())
        cutoff_label = format_cutoff_label(active_cutoff_hours)
        self.cutoff_label = "default"
        default_feature_path = self.paths.feature_matrix_path
        labeled_feature_path = default_feature_path.with_name(
            f"{default_feature_path.stem}_{cutoff_label}{default_feature_path.suffix}"
        )
        if labeled_feature_path.exists():
            self.paths = Paths(
                feature_matrix_path=labeled_feature_path,
                model_dir=self.paths.model_dir,
                metric_dir=self.paths.metric_dir,
                analysis_dir=self.paths.analysis_dir,
            )
            self.config['data']['feature_matrix_path'] = str(labeled_feature_path)
            self.cutoff_label = cutoff_label
            logger.info(
                "Using feature matrix for cutoff %.2f hours (%s) → %s",
                active_cutoff_hours,
                cutoff_label,
                labeled_feature_path,
            )
        else:
            logger.warning(
                "Feature matrix for cutoff %.2f hours (%s) not found at %s; defaulting to %s",
                active_cutoff_hours,
                cutoff_label,
                labeled_feature_path,
                default_feature_path,
            )
        self.cfg = TrainingCfg(
            production_mode=bool(self.config.get('training', {}).get('production_mode', False)),
            data_split=dict(self.config['data_split']),
            models_to_train=list(self.config['training']['models_to_train']),
            hyperparameter_tuning=dict(self.config['training']['hyperparameter_tuning']),
            calibrate=bool(self.config['training'].get('calibrate', True)),
            calibration_method=str(self.config['training'].get('calibration_method', 'isotonic')),
            threshold_selection=dict(self.config['training'].get('threshold_selection', {})),
            regression_mean_calibration=dict(self.config['training'].get('regression_mean_calibration', {'enabled': False})),
            regression_intervals=dict(self.config['training'].get('regression_intervals', {'enabled': True, 'alpha': 0.1})),
            mu_tail_5plus=float(self.config['training'].get('mu_tail_5plus', 5.6)),
            ordinal_ev=dict(self.config.get('ordinal_ev', {'enabled': False})),
            selective=dict(self.config['training'].get('selective', {'enabled': False, 'coverage_target': 0.12})),
            base_seed=int(self.config['training'].get('base_seed', 42)),
            write_run_manifest=bool(self.config['training'].get('write_run_manifest', True)),
            versioning_mode=str(self.config['training'].get('versioning_mode', 'run_id')),
            run_tag=str(self.config['training'].get('run_tag', '')),
        )
        try:
            self._normalized_param_distributions = normalize_param_distributions(
                self.cfg.hyperparameter_tuning
            )
        except TuningConfigError as exc:
            raise ValueError(f"Invalid hyperparameter tuning configuration: {exc}") from exc
        # Apply CLI overrides (if provided) by creating new config instances
        cfg_overrides = {}
        if "production_mode" in overrides:
            cfg_overrides["production_mode"] = bool(overrides["production_mode"])
        if "models_to_train" in overrides and overrides["models_to_train"]:
            cfg_overrides["models_to_train"] = list(overrides["models_to_train"])
        if "run_tag" in overrides:
            cfg_overrides["run_tag"] = str(overrides["run_tag"] or "")
        if "run_tuning" in overrides:
            cfg_overrides["hyperparameter_tuning"] = {**self.cfg.hyperparameter_tuning, "run_tuning": bool(overrides["run_tuning"])}
        if "versioning_mode" in overrides:
            cfg_overrides["versioning_mode"] = str(overrides["versioning_mode"])

        # Rebuild cfg with overrides if any exist
        if cfg_overrides:
            self.cfg = TrainingCfg(**{**self.cfg.__dict__, **cfg_overrides})
            try:
                self._normalized_param_distributions = normalize_param_distributions(
                    self.cfg.hyperparameter_tuning
                )
            except TuningConfigError as exc:
                raise ValueError(f"Invalid hyperparameter tuning configuration: {exc}") from exc

        if "problems" in overrides and overrides["problems"]:
            self.config["problems"] = [p for p in self.config["problems"] if p["name"] in set(overrides["problems"])]
        
        # keep originals for backward compat
        self.model_dir = self.paths.model_dir
        self.metric_dir = self.paths.metric_dir
        self.production_mode = self.cfg.production_mode
        self.base_seed = int(self.cfg.base_seed)
        self.run_tag = str(self.cfg.run_tag or "")
        self.versioning_mode = str(self.cfg.versioning_mode)
        if self.cutoff_label != "default":
            suffix = f"cutoff_{self.cutoff_label}"
            self.run_tag = f"{self.run_tag}_{suffix}" if self.run_tag else suffix
        self.run_id = make_run_id(self.run_tag)

        # Sample weighting configuration
        self.sample_weight_cfg = copy.deepcopy(self.config['training'].get('sample_weighting', {}))

        self.time_col = self.config['data_split']['time_col']
        self.group_col = self.config['data_split']['group_col']  # game_id for NFL

        # Ensure output directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)

        # Per-problem attributes initialised later
        self.target_col = None
        self.label_version = None
        self.feature_columns = None

        self.models = {}
        self.best_params = {}
        self.best_params_meta = {} # Store trial metadata separately
        self.categorical_features = {} # Per-problem
        self.imputation_values = {} # Per-problem
        self.category_levels = {} # Per-problem
        self._cached_splits = {} # Cache for CV splits per problem
        self._groups_index_map = {} # Map problem -> Series(index=df_train_full.index, value=group label)
        self._composite_cal_store = {}  # problem -> {"index": X_cal.index, "y_cal": np.array, "y_hat_cal": np.array}
        self._composite_test_store = {}  # problem -> {"index": X_test.index, "y_test": np.array, "y_hat_test": np.array}
        self._train_full_frames: dict[str, pd.DataFrame] = {}
        self._sample_weights: dict[str, dict[str, pd.Series]] = {}
        self._team_total_config: dict[str, TeamTotalConfig] = {}
        self.datetime_features: dict[str, list[str]] = {}
        self._oof_preds_store = {}       # Store GLOBAL OOF predictions for stacking leakage prevention
        self._oof_preds_store_moe = {}   # Store MoE OOF predictions (per-position) for stacking
        self._oof_keys_store = {}        # Store key frames aligned to OOF arrays (for merge-based injection)
        self._best_thresholds: dict[str, float] = {}  # problem_name -> best decision threshold (GLOBAL)
        self._player_game_actuals: pd.DataFrame | None = None


        # ------------------------------------------------------------------
        # Version-compat check: older xgboost (<1.6) sklearn wrapper does not
        # accept the `callbacks` kwarg.  Record the capability once so we can
        # branch everywhere we call .fit()/constructor with callbacks.
        # ------------------------------------------------------------------
        try:
            self._xgb_supports_callbacks = (
                'callbacks' in inspect.signature(xgb.XGBClassifier.fit).parameters
            )
        except Exception:
            self._xgb_supports_callbacks = False

        # Problems list from config
        self.problems = self.config['problems']
        
        # Store predictions + merge keys for downstream problems
        # Format: {problem_name: {"data": DataFrame, "keys": [cols...]}}
        self.problem_predictions: dict[str, dict[str, object]] = {}

        # Which upstream problems are used as stack inputs anywhere downstream.
        self._stack_input_problems: set[str] = set()
        try:
            for p in (self.problems or []):
                for ip in (p.get("input_predictions") or []):
                    self._stack_input_problems.add(str(ip))
        except Exception:
            self._stack_input_problems = set()

        # ==========================================================================
        # MoE (Mixture of Experts) Configuration
        # ==========================================================================
        # model_architecture: "global", "moe", or "both"
        self.model_architecture = self.config.get('training', {}).get('model_architecture', 'global')
        if "model_architecture" in overrides and overrides["model_architecture"]:
            self.model_architecture = str(overrides["model_architecture"])
            try:
                self.config.setdefault("training", {})["model_architecture"] = self.model_architecture
            except Exception:
                pass
        self.moe_position_groups = self.config.get('training', {}).get('moe_position_groups', ['RB', 'WR', 'TE', 'QB'])
        self.moe_models: dict[str, dict[str, object]] = {}  # {problem_name: {position: model}}
        self.moe_comparison_results: dict[str, dict] = {}  # Store comparison metrics
        
        logger.info(f"Model architecture mode: {self.model_architecture}")
        if self.model_architecture in ("moe", "both"):
            logger.info(f"MoE position groups: {self.moe_position_groups}")

        self._ensure_injury_cache()

    def load_data(self, columns=None):
        """Wrapper for load_feature_matrix from utils.train.data."""
        return load_feature_matrix(
            path=self.config['data']['feature_matrix_path'],
            time_col=self.time_col,
            columns=columns
        )

    def split_data_chronologically(self, df):
        """Wrapper for split_data_chronologically from utils.train.data."""
        return split_chrono(
            df=df,
            time_col=self.time_col,
            split_cfg=self.config['data_split'],
            production_mode=self.production_mode
        )

    def _clean_output_dirs(self):
        """Wrapper for clean_output_dirs from utils.train.persist."""
        clean_output_dirs(self)

    def _log_top_feature_importance(
        self,
        model,
        problem_name: str,
        model_name: str,
        top_n: int = 12,
    ) -> None:
        """Emit a concise importance summary for quick regression checks."""
        base_model = self._unwrap_base_model(model)
        if base_model is None:
            return

        importance: dict[str, float] = {}
        source = ""

        if hasattr(base_model, "get_booster"):
            booster = base_model.get_booster()
            for imp_type in ("gain", "weight"):
                try:
                    raw_scores = booster.get_score(importance_type=imp_type)
                except Exception:
                    raw_scores = {}
                if raw_scores:
                    importance = {str(k): float(v) for k, v in raw_scores.items()}
                    source = f"xgb_{imp_type}"
                    break
        elif hasattr(base_model, "feature_importances_"):
            scores = getattr(base_model, "feature_importances_", None)
            if scores is not None and self.feature_columns:
                importance = {
                    str(feature): float(weight)
                    for feature, weight in zip(self.feature_columns, scores)
                }
                source = "feature_importances_"

        if not importance:
            return

        feature_columns = list(self.feature_columns or [])
        if feature_columns:
            remapped: dict[str, float] = {}
            used_remap = False
            for key, value in importance.items():
                mapped = False
                if key.startswith("f") and key[1:].isdigit():
                    idx = int(key[1:])
                    if 0 <= idx < len(feature_columns):
                        remapped[feature_columns[idx]] = value
                        mapped = True
                elif key.isdigit():
                    idx = int(key)
                    if 0 <= idx < len(feature_columns):
                        remapped[feature_columns[idx]] = value
                        mapped = True
                if not mapped:
                    remapped[key] = value
                used_remap = used_remap or mapped
            if used_remap:
                importance = remapped

        sorted_items = sorted(
            importance.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:top_n]
        if not sorted_items:
            return

        summary = ", ".join(f"{feat}: {weight:.3f}" for feat, weight in sorted_items)
        logger.info(
            "Top %d features for %s/%s [%s]: %s",
            len(sorted_items),
            problem_name,
            model_name,
            source or "default",
            summary,
        )
        self._persist_importance_highlight(problem_name, model_name, sorted_items, importance)

    def _persist_importance_highlight(
        self,
        problem_name: str,
        model_name: str,
        top_items: list[tuple[str, float]],
        importance_dict: dict[str, float],
    ) -> None:
        """Write a lightweight JSON summary emphasizing key feature families."""
        try:
            out_dir = self.metric_dir / "importance_highlights"
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix_targets = ["drive_hist_", "role_", "weather_", "travel_"]
            total_weight = sum(abs(val) for val in importance_dict.values()) or 1.0
            prefix_summary = {
                prefix: {
                    "weight": float(
                        sum(abs(w) for feat, w in importance_dict.items() if feat.startswith(prefix))
                    ),
                }
                for prefix in prefix_targets
            }
            for prefix, payload in prefix_summary.items():
                payload["share"] = float(payload["weight"] / total_weight)

            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "problem": problem_name,
                "model": model_name,
                "top_features": [
                    {"feature": feat, "importance": float(weight)}
                    for feat, weight in top_items
                ],
                "prefix_summary": prefix_summary,
            }
            out_path = out_dir / f"{problem_name}_{model_name}_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
            out_path.write_text(json.dumps(data, indent=2))
        except Exception as exc:  # pragma: no cover - diagnostics
            logger.warning("Unable to persist importance highlight for %s/%s: %s", problem_name, model_name, exc)




    def _ensure_injury_cache(self) -> None:
        """
        Ensure ESPN injury caches exist for all seasons referenced by the feature matrix.
        """
        seasons = _infer_seasons_from_parquet(self.paths.feature_matrix_path)
        if not seasons:
            logger.warning(
                "Could not determine seasons from %s; skipping injury cache automation.",
                self.paths.feature_matrix_path,
            )
            return

        cache_dir = Path("cache") / "feature" / "injuries"
        cache_dir.mkdir(parents=True, exist_ok=True)
        missing = [
            season
            for season in seasons
            if not (cache_dir / f"injury_{int(season)}.parquet").exists()
        ]
        if not missing:
            return
        try:
            logger.info("Refreshing ESPN injury caches for seasons %s", missing)
            collect_espn_injuries(missing, overwrite=False)
        except Exception as exc:
            logger.warning("Failed to refresh ESPN injury cache: %s", exc)

    # ==========================================================================
    # MoE (Mixture of Experts) Helper Methods
    # ==========================================================================
    
    @staticmethod
    def _normalize_position_group(position: str) -> str:
        """Normalize a position string to a position group (RB, WR, TE, QB)."""
        pos = str(position).upper() if position else ""
        if pos in {"RB", "HB", "FB"}:
            return "RB"
        if pos == "WR":
            return "WR"
        if pos == "TE":
            return "TE"
        if pos == "QB":
            return "QB"
        # Default fallback - treat unknown as WR (most common skill position)
        return "WR"
    
    def _add_position_group(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position_group column to DataFrame if not present."""
        if "position_group" not in df.columns:
            if "position" in df.columns:
                df = df.copy()
                df["position_group"] = df["position"].apply(self._normalize_position_group)
            else:
                logger.warning("Cannot add position_group: 'position' column not found")
        return df
    
    def _train_moe_models(
        self,
        problem_config: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        sample_weight: pd.Series | None,
        feature_columns_override: list[str] | None = None,
    ) -> dict[str, object]:
        """
        Train per-position MoE models for a single problem.
        
        Returns dict mapping position -> trained model.
        """
        problem_name = problem_config["name"]
        task_type = problem_config.get("task_type", "classification")
        
        logger.info(f"[MoE] Training per-position models for {problem_name}")
        
        # Ensure position_group is available
        df_train = self._add_position_group(df_train)
        df_test = self._add_position_group(df_test)
        
        position_models = {}
        position_metrics = {}
        
        per_pos_cfg = problem_config.get("per_position_training", {}) or {}
        allowed_positions = per_pos_cfg.get("positions", self.moe_position_groups)
        min_rows = int(per_pos_cfg.get("min_rows", 50) or 50)
        fallback_position = str(per_pos_cfg.get("fallback_position", "WR") or "WR").upper()

        for pos in self.moe_position_groups:
            if allowed_positions and pos not in allowed_positions:
                continue
            # Create masks for this position
            train_mask = (df_train["position_group"] == pos).values
            test_mask = (df_test["position_group"] == pos).values
            
            n_train = train_mask.sum()
            n_test = test_mask.sum()
            
            if n_train < min_rows:
                logger.warning(
                    f"[MoE] Skipping {pos} for {problem_name}: only {n_train} training samples "
                    f"(min_rows={min_rows})"
                )
                continue
            
            logger.info(f"[MoE] Training {problem_name}.{pos}: {n_train} train, {n_test} test samples")
            
            # Subset data
            X_train_pos = X_train.loc[train_mask].copy()
            if X_train_pos.columns.duplicated().any():
                dupes = X_train_pos.columns[X_train_pos.columns.duplicated()].tolist()
                logger.warning("[MoE] Duplicate feature columns detected for %s.%s (deduping): %s", problem_name, pos, dupes)
                X_train_pos = X_train_pos.loc[:, ~X_train_pos.columns.duplicated()].copy()
            y_train_pos = y_train.loc[train_mask].copy()
            sw_pos = sample_weight.loc[train_mask] if sample_weight is not None else None
            
            # Convert categorical columns to numeric for XGBoost
            for col in X_train_pos.columns:
                if X_train_pos[col].dtype.name == 'category':
                    X_train_pos[col] = X_train_pos[col].cat.codes
                elif is_datetime64_any_dtype(X_train_pos[col]):
                    col_as_dt = pd.to_datetime(X_train_pos[col], utc=True, errors="coerce")
                    numeric = col_as_dt.astype("int64", copy=False).astype("float64")
                    numeric[col_as_dt.isna().to_numpy()] = np.nan
                    X_train_pos[col] = numeric / 1_000_000.0
                elif X_train_pos[col].dtype == 'object':
                    X_train_pos[col] = pd.to_numeric(X_train_pos[col], errors='coerce')
            # IMPORTANT: do NOT fill missing values with 0. XGBoost handles missing values.
            
            # Get model parameters for MoE
            # Use config defaults, then apply light regularization for smaller datasets.
            # Analysis found that RB/TE prediction is dominated by general features 
            # (snap_offense_pct_prev, is_inactive), so position-specific models 
            # struggle to outperform Global without MUCH more data.
            xgb_params = self.config.get("xgboost", {}).copy()
            
            # Light regularization to prevent overfitting on smaller per-position data
            xgb_params["n_estimators"] = 250
            xgb_params["max_depth"] = 5
            xgb_params["learning_rate"] = 0.05
            xgb_params["min_child_weight"] = 5
            xgb_params["subsample"] = 0.8
            xgb_params["colsample_bytree"] = 0.8

            # For hard-to-generalize usage/efficiency regressions, apply extra regularization
            hard_regression = {
                "usage_target_yards",
                "efficiency_rec_yards_air",
                "efficiency_rec_yards_yac",
                "efficiency_rush_yards",
                "usage_carries",
            }
            if task_type == "regression" and problem_name in hard_regression:
                xgb_params["n_estimators"] = 300
                xgb_params["max_depth"] = 3
                xgb_params["learning_rate"] = 0.05
                xgb_params["min_child_weight"] = 8
                xgb_params["subsample"] = 0.7
                xgb_params["colsample_bytree"] = 0.7
                xgb_params["reg_alpha"] = xgb_params.get("reg_alpha", 0) + 0.2
                xgb_params["reg_lambda"] = xgb_params.get("reg_lambda", 1) * 2
            
            # Adjust objective based on task type
            if task_type == "regression":
                xgb_params["objective"] = "reg:squarederror"
                xgb_params.pop("eval_metric", None)
            else:
                xgb_params["objective"] = "binary:logistic"
                xgb_params["eval_metric"] = "auc"
            
            # Create and train model
            if task_type == "regression":
                model = xgb.XGBRegressor(
                    **{k: v for k, v in xgb_params.items() if k not in ("objective", "eval_metric")},
                    objective=xgb_params.get("objective", "reg:squarederror"),
                    random_state=self.base_seed,
                    n_jobs=-1,
                )
            else:
                model = xgb.XGBClassifier(
                    **{k: v for k, v in xgb_params.items() if k not in ("objective", "eval_metric")},
                    objective=xgb_params.get("objective", "binary:logistic"),
                    eval_metric=xgb_params.get("eval_metric", "auc"),
                    random_state=self.base_seed,
                    n_jobs=-1,
                    use_label_encoder=False,
                )
            
            # Fit model
            fit_kwargs = {"sample_weight": sw_pos} if sw_pos is not None else {}
            model.fit(X_train_pos, y_train_pos, **fit_kwargs)
            
            position_models[pos] = model
            
            # Compute test metrics for this position
            if n_test >= 10:
                X_test_pos = X_test.loc[test_mask].copy()
                if X_test_pos.columns.duplicated().any():
                    dupes = X_test_pos.columns[X_test_pos.columns.duplicated()].tolist()
                    logger.warning("[MoE] Duplicate feature columns detected for %s.%s test slice (deduping): %s", problem_name, pos, dupes)
                    X_test_pos = X_test_pos.loc[:, ~X_test_pos.columns.duplicated()].copy()
                y_test_pos = y_test.loc[test_mask].copy()
                
                # Convert categorical columns to numeric for XGBoost
                for col in X_test_pos.columns:
                    if X_test_pos[col].dtype.name == 'category':
                        X_test_pos[col] = X_test_pos[col].cat.codes
                    elif is_datetime64_any_dtype(X_test_pos[col]):
                        col_as_dt = pd.to_datetime(X_test_pos[col], utc=True, errors="coerce")
                        numeric = col_as_dt.astype("int64", copy=False).astype("float64")
                        numeric[col_as_dt.isna().to_numpy()] = np.nan
                        X_test_pos[col] = numeric / 1_000_000.0
                    elif X_test_pos[col].dtype == 'object':
                        X_test_pos[col] = pd.to_numeric(X_test_pos[col], errors='coerce')
                # IMPORTANT: do NOT fill missing values with 0. XGBoost handles missing values.
                
                if task_type == "regression":
                    preds_pos = model.predict(X_test_pos)
                    from sklearn.metrics import mean_absolute_error, r2_score
                    mae = mean_absolute_error(y_test_pos, preds_pos)
                    r2 = r2_score(y_test_pos, preds_pos) if len(np.unique(y_test_pos)) > 1 else 0
                    position_metrics[pos] = {"mae": mae, "r2": r2, "n_train": n_train, "n_test": n_test}
                    logger.info(f"[MoE] {problem_name}.{pos}: MAE={mae:.4f}, R²={r2:.4f}")
                else:
                    preds_pos = model.predict_proba(X_test_pos)[:, 1]
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    try:
                        auc = roc_auc_score(y_test_pos, preds_pos)
                        pr_auc = average_precision_score(y_test_pos, preds_pos)
                    except Exception:
                        auc = 0.5
                        pr_auc = y_test_pos.mean()
                    position_metrics[pos] = {"auc": auc, "pr_auc": pr_auc, "n_train": n_train, "n_test": n_test}
                    logger.info(f"[MoE] {problem_name}.{pos}: AUC={auc:.4f}, PR-AUC={pr_auc:.4f}")
        
        # Store MoE models
        self.moe_models[problem_name] = position_models
        
        # Save MoE models to disk
        self._save_moe_models(
            problem_name,
            position_models,
            fallback_position=fallback_position,
            feature_columns_override=feature_columns_override,
        )
        
        # Store comparison metrics
        self.moe_comparison_results[problem_name] = {
            "position_metrics": position_metrics,
            "positions_trained": list(position_models.keys()),
        }
        
        return position_models
    
    def _save_moe_models(
        self,
        problem_name: str,
        position_models: dict[str, object],
        fallback_position: str = "WR",
        feature_columns_override: list[str] | None = None,
    ) -> None:
        """Save MoE models to disk with position-specific filenames."""
        problem_dir = self.model_dir / problem_name / "xgboost"
        problem_dir.mkdir(parents=True, exist_ok=True)
        
        for pos, model in position_models.items():
            model_path = problem_dir / f"model.{pos}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"[MoE] Saved {problem_name}.{pos} model to {model_path}")
        
        # Also save MoE inference artifacts
        moe_artifacts = {
            "feature_columns": (
                list(feature_columns_override)
                if feature_columns_override is not None
                else (list(self.feature_columns) if self.feature_columns else [])
            ),
            "positions": list(position_models.keys()),
            "fallback_position": fallback_position,  # From config
        }
        # Horizon-safe legacy artifact name (avoid mixing cutoffs).
        cutoff_label = getattr(self, "cutoff_label", "default")
        legacy_name = (
            f"inference_artifacts_{problem_name}_moe_{cutoff_label}.joblib"
            if cutoff_label != "default"
            else f"inference_artifacts_{problem_name}_moe.joblib"
        )
        artifacts_path = self.model_dir / legacy_name
        joblib.dump(moe_artifacts, artifacts_path)
        logger.info(f"[MoE] Saved inference artifacts to {artifacts_path}")

        # Also save a versioned copy under the run directory for robust discovery.
        try:
            v_dir = _vdir(self, problem_name, None, "artifacts") / "inference"
            v_dir.mkdir(parents=True, exist_ok=True)
            v_path = v_dir / legacy_name
            joblib.dump(moe_artifacts, v_path)
        except Exception:
            pass
    
    def _predict_with_moe(
        self,
        X: pd.DataFrame,
        df: pd.DataFrame,
        problem_name: str,
    ) -> np.ndarray:
        """
        Generate predictions using MoE models, routing by position group.
        
        Args:
            X: Feature matrix (already transformed)
            df: Original DataFrame with position column
            problem_name: Name of the problem
            
        Returns:
            Array of predictions
        """
        if problem_name not in self.moe_models:
            raise ValueError(f"No MoE models found for problem {problem_name}")
        
        position_models = self.moe_models[problem_name]
        df = self._add_position_group(df)
        
        preds = np.zeros(len(X), dtype=np.float64)
        position_groups = df["position_group"]
        
        # Route predictions by position
        for pos, model in position_models.items():
            mask = (position_groups == pos).values
            if mask.sum() == 0:
                continue
            
            X_pos = X.loc[mask].copy()
            # Convert categorical columns to numeric for XGBoost
            for col in X_pos.columns:
                if X_pos[col].dtype.name == 'category':
                    X_pos[col] = X_pos[col].cat.codes
                elif is_datetime64_any_dtype(X_pos[col]):
                    col_as_dt = pd.to_datetime(X_pos[col], utc=True, errors="coerce")
                    numeric = col_as_dt.astype("int64", copy=False).astype("float64")
                    numeric[col_as_dt.isna().to_numpy()] = np.nan
                    X_pos[col] = numeric / 1_000_000.0
                elif X_pos[col].dtype == 'object':
                    X_pos[col] = pd.to_numeric(X_pos[col], errors='coerce')
            # IMPORTANT: do NOT fill missing values with 0. XGBoost handles missing values.
            
            if hasattr(model, "predict_proba"):
                pos_preds = model.predict_proba(X_pos)
                if pos_preds.ndim > 1:
                    pos_preds = pos_preds[:, 1]
            else:
                pos_preds = model.predict(X_pos)
            
            preds[mask] = pos_preds
        
        # Handle positions not covered by trained models (use fallback)
        known_positions = set(position_models.keys())
        fallback_mask = ~position_groups.isin(known_positions).values
        if fallback_mask.sum() > 0:
            fallback_pos = "WR" if "WR" in position_models else list(position_models.keys())[0]
            fallback_model = position_models[fallback_pos]
            X_fallback = X.loc[fallback_mask].copy()
            
            # Convert categorical columns to numeric for XGBoost
            for col in X_fallback.columns:
                if X_fallback[col].dtype.name == 'category':
                    X_fallback[col] = X_fallback[col].cat.codes
                elif is_datetime64_any_dtype(X_fallback[col]):
                    col_as_dt = pd.to_datetime(X_fallback[col], utc=True, errors="coerce")
                    numeric = col_as_dt.astype("int64", copy=False).astype("float64")
                    numeric[col_as_dt.isna().to_numpy()] = np.nan
                    X_fallback[col] = numeric / 1_000_000.0
                elif X_fallback[col].dtype == 'object':
                    X_fallback[col] = pd.to_numeric(X_fallback[col], errors='coerce')
            # IMPORTANT: do NOT fill missing values with 0. XGBoost handles missing values.
            
            if hasattr(fallback_model, "predict_proba"):
                fallback_preds = fallback_model.predict_proba(X_fallback)
                if fallback_preds.ndim > 1:
                    fallback_preds = fallback_preds[:, 1]
            else:
                fallback_preds = fallback_model.predict(X_fallback)
            
            preds[fallback_mask] = fallback_preds
            logger.info(f"[MoE] Used {fallback_pos} model as fallback for {fallback_mask.sum()} rows")
        
        return preds
    
    def _compare_global_vs_moe(
        self,
        problem_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        df_test: pd.DataFrame,
        global_preds: np.ndarray,
        task_type: str,
    ) -> dict:
        """
        Compare global model vs MoE predictions and return comparison metrics.
        """
        if problem_name not in self.moe_models:
            return {}
        
        # Get MoE predictions
        moe_preds = self._predict_with_moe(X_test, df_test, problem_name)
        df_test = self._add_position_group(df_test)
        position_groups = df_test["position_group"]
        
        comparison = {"global": {}, "moe": {}, "by_position": {}}
        
        if task_type == "regression":
            from sklearn.metrics import mean_absolute_error, r2_score
            
            # Overall metrics
            comparison["global"]["mae"] = mean_absolute_error(y_test, global_preds)
            comparison["global"]["r2"] = r2_score(y_test, global_preds) if len(np.unique(y_test)) > 1 else 0
            comparison["moe"]["mae"] = mean_absolute_error(y_test, moe_preds)
            comparison["moe"]["r2"] = r2_score(y_test, moe_preds) if len(np.unique(y_test)) > 1 else 0
            
            # By position
            for pos in self.moe_position_groups:
                mask = (position_groups == pos).values
                if mask.sum() < 10:
                    continue
                y_pos = y_test.loc[mask]
                g_pos = global_preds[mask]
                m_pos = moe_preds[mask]
                
                comparison["by_position"][pos] = {
                    "n": int(mask.sum()),
                    "global_mae": mean_absolute_error(y_pos, g_pos),
                    "moe_mae": mean_absolute_error(y_pos, m_pos),
                    "global_r2": r2_score(y_pos, g_pos) if len(np.unique(y_pos)) > 1 else 0,
                    "moe_r2": r2_score(y_pos, m_pos) if len(np.unique(y_pos)) > 1 else 0,
                }
        else:
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            # Overall metrics
            try:
                comparison["global"]["auc"] = roc_auc_score(y_test, global_preds)
                comparison["global"]["pr_auc"] = average_precision_score(y_test, global_preds)
                comparison["moe"]["auc"] = roc_auc_score(y_test, moe_preds)
                comparison["moe"]["pr_auc"] = average_precision_score(y_test, moe_preds)
            except Exception:
                pass
        
        # Log comparison
        logger.info(f"[MoE] Comparison for {problem_name}:")
        if task_type == "regression":
            g_mae = comparison["global"].get("mae", 0)
            m_mae = comparison["moe"].get("mae", 0)
            delta = m_mae - g_mae
            logger.info(f"  Overall: Global MAE={g_mae:.4f}, MoE MAE={m_mae:.4f}, Δ={delta:+.4f}")
            for pos, metrics in comparison["by_position"].items():
                delta_pos = metrics["moe_mae"] - metrics["global_mae"]
                better = "✓" if delta_pos < 0 else ""
                logger.info(f"  {pos}: Global={metrics['global_mae']:.4f}, MoE={metrics['moe_mae']:.4f}, Δ={delta_pos:+.4f} {better}")
        
        return comparison

    


    def _fit_feature_artifacts(self, df_train, problem_config):
        """
        Delegate to shared artifact fitter to freeze feature list + encoders.
        """
        problem_name = problem_config['name']
        self.current_problem_name = problem_name
        logger.info(f"Fitting feature artifacts for problem '{problem_name}' from training data...")

        artifacts = build_feature_artifacts(df_train, problem_config)
        self.feature_columns = artifacts.feature_columns
        self.datetime_features[problem_name] = artifacts.datetime_features
        self.categorical_features[problem_name] = artifacts.categorical_features
        self.category_levels[problem_name] = artifacts.category_levels
    def _normalize_datetime_like_columns(self, df: pd.DataFrame, problem_name: str) -> None:
        """Force datetime/date-like columns into pandas datetime dtype so downstream encoding works."""
        dt_cols: list[str] = []
        for col in df.columns:
            series = df[col]
            if is_datetime64_any_dtype(series):
                dt_cols.append(col)
                continue
            if series.dtype == object:
                sample = series.dropna().head(5)
                if sample.empty:
                    continue
                if all(isinstance(val, (datetime, date)) for val in sample):
                    dt_cols.append(col)
        if not dt_cols:
            return
        dt_cols = list(dict.fromkeys(dt_cols))
        for col in dt_cols:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        tracked = self.datetime_features.setdefault(problem_name, [])
        for col in dt_cols:
            if col not in tracked:
                tracked.append(col)

    def _inject_composed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive reality-aligned features (expected targets/carries/opportunities)
        from specialist predictions so downstream models learn the causal chain
        instead of raw, disjoint signals.
        """
        if df is None or df.empty:
            return df

        def _ensure_availability_composites(frame: pd.DataFrame) -> None:
            if (
                "pred_availability" in frame.columns
                and "pred_availability_raw" in frame.columns
            ):
                return
            has_active = "pred_availability_active" in frame.columns
            has_share = "pred_availability_snapshare" in frame.columns
            if not (has_active and has_share):
                return
            active = pd.to_numeric(frame["pred_availability_active"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            share = pd.to_numeric(frame["pred_availability_snapshare"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            combined = (active * share).astype(np.float32)
            if "pred_availability" not in frame.columns:
                frame["pred_availability"] = combined
            if "pred_availability_raw" not in frame.columns:
                if "pred_availability_active_raw" in frame.columns:
                    raw_active = (
                        pd.to_numeric(frame["pred_availability_active_raw"], errors="coerce")
                        .fillna(0.0)
                        .clip(0.0, 1.0)
                    )
                    frame["pred_availability_raw"] = (raw_active * share).astype(np.float32)
                else:
                    frame["pred_availability_raw"] = combined

        _ensure_availability_composites(df)

        def _safe_mul(a: str, b: str, out_col: str):
            if a in df.columns and b in df.columns and out_col not in df.columns:
                df[out_col] = df[a] * df[b]

        def _select_column(candidates: list[str]) -> str | None:
            for name in candidates:
                if name in df.columns:
                    return name
            return None

        # Team pass/rush rate proxies reused across expectations
        pass_rate_candidates = [
                    "team_ctx_pass_rate_prev",
                    "team_ctx_pass_rate_l3",
                    "team_ctx_pass_rate_l5",
            "team_pass_rate_prev",
            "team_pass_rate_l3",
                ]
        pass_rate_col = _select_column(pass_rate_candidates)
        default_pass_rate = 0.55
        if len(df):
            if pass_rate_col:
                pass_rate_series = (
                    pd.to_numeric(df[pass_rate_col], errors="coerce")
                    .fillna(default_pass_rate)
                    .clip(0.0, 1.0)
                )
            else:
                pass_rate_series = pd.Series(default_pass_rate, index=df.index, dtype=np.float32)
        else:
            pass_rate_series = pd.Series([], dtype=np.float32)
        rush_rate_series = (1.0 - pass_rate_series).clip(0.0, 1.0)

        # Team-level expected plays from pace model
        if "pred_team_pace" in df.columns:
            team_pace = pd.to_numeric(df["pred_team_pace"], errors="coerce").clip(lower=0.0)
            df["expected_team_plays"] = team_pace
            if len(team_pace) and len(pass_rate_series) == len(team_pace):
                df["expected_team_pass_plays"] = (team_pace * pass_rate_series).astype(np.float32)
                df["expected_team_rush_plays"] = (team_pace * rush_rate_series).astype(np.float32)
            else:
                df["expected_team_pass_plays"] = (
                    team_pace * default_pass_rate
                ).astype(np.float32)
                df["expected_team_rush_plays"] = (
                    team_pace * (1.0 - default_pass_rate)
                ).astype(np.float32)

        # Explicit snaps expectation from stage-1 model or availability fallback
        expected_snaps_series: pd.Series | None = None
        if "pred_snaps" in df.columns:
            expected_snaps_series = (
                pd.to_numeric(df["pred_snaps"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
            df["expected_snaps"] = expected_snaps_series.astype(np.float32)
        elif (
            "pred_availability" in df.columns
            and "pred_availability_snapshare" in df.columns
            and "expected_team_plays" in df.columns
        ):
            avail = pd.to_numeric(df["pred_availability"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            snapshare = pd.to_numeric(df["pred_availability_snapshare"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            team_plays = pd.to_numeric(df["expected_team_plays"], errors="coerce").fillna(0.0).clip(lower=0.0)
            expected_snaps_series = (avail * snapshare * team_plays).astype(np.float32)
            df["expected_snaps"] = expected_snaps_series

        # Raw snaps proxy (used for diagnostics/guard rails)
        expected_snaps_raw_series: pd.Series | None = None
        if (
            "pred_availability_raw" in df.columns
            and "pred_availability_snapshare" in df.columns
            and "expected_team_plays" in df.columns
        ):
            avail_raw = pd.to_numeric(df["pred_availability_raw"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            snapshare = pd.to_numeric(df["pred_availability_snapshare"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            team_plays = pd.to_numeric(df["expected_team_plays"], errors="coerce").fillna(0.0).clip(lower=0.0)
            expected_snaps_raw_series = (avail_raw * snapshare * team_plays).astype(np.float32)
            df["expected_snaps_raw"] = expected_snaps_raw_series

        if expected_snaps_series is None and "expected_snaps" in df.columns:
            expected_snaps_series = (
                pd.to_numeric(df["expected_snaps"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
        if expected_snaps_series is not None:
            expected_snaps_series = expected_snaps_series.astype(np.float32)

        # Uncertainty-weighted snaps and opportunities for fragile volume handling
        if expected_snaps_series is not None and "availability_uncertainty" in df.columns:
            unc = pd.to_numeric(df["availability_uncertainty"], errors="coerce").fillna(0.0).clip(0.0, 0.25)
            df["expected_snaps_stable"] = (expected_snaps_series * (1.0 - unc)).astype(np.float32)

        # Usage share × expected snaps × team rates = expected counts
        if expected_snaps_series is not None and "pred_usage_targets" in df.columns:
            share_tgt = pd.to_numeric(df["pred_usage_targets"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            df["expected_targets"] = (
                share_tgt.astype(np.float32) * expected_snaps_series * pass_rate_series.reindex(df.index, fill_value=default_pass_rate)
            ).astype(np.float32)

        if expected_snaps_raw_series is not None and "pred_usage_targets" in df.columns:
            share_tgt = pd.to_numeric(df["pred_usage_targets"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            df["expected_targets_raw"] = (
                share_tgt.astype(np.float32) * expected_snaps_raw_series * pass_rate_series.reindex(df.index, fill_value=default_pass_rate)
            ).astype(np.float32)

        if expected_snaps_series is not None and "pred_usage_carries" in df.columns:
            share_car = pd.to_numeric(df["pred_usage_carries"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            df["expected_carries"] = (
                share_car.astype(np.float32) * expected_snaps_series * rush_rate_series.reindex(df.index, fill_value=1.0 - default_pass_rate)
            ).astype(np.float32)

        if expected_snaps_raw_series is not None and "pred_usage_carries" in df.columns:
            share_car = pd.to_numeric(df["pred_usage_carries"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            df["expected_carries_raw"] = (
                share_car.astype(np.float32) * expected_snaps_raw_series * rush_rate_series.reindex(df.index, fill_value=1.0 - default_pass_rate)
            ).astype(np.float32)

        if "expected_targets" in df.columns and "expected_targets_raw" not in df.columns:
            df["expected_targets_raw"] = df["expected_targets"]
        if "expected_carries" in df.columns and "expected_carries_raw" not in df.columns:
            df["expected_carries_raw"] = df["expected_carries"]

        if "expected_targets" in df.columns and "expected_carries" in df.columns:
            # Optional: enforce per-team consistency between expected targets/carries
            # and team-level play expectations by rescaling within (season, week, team, game_id).
            group_keys = [k for k in ("season", "week", "team", "game_id") if k in df.columns]
            if group_keys and {"expected_team_pass_plays", "expected_team_rush_plays"} <= set(df.columns):
                grouped = df.groupby(group_keys, dropna=False, sort=False)
                sum_targets = grouped["expected_targets"].transform("sum")
                sum_carries = grouped["expected_carries"].transform("sum")
                team_pass = df["expected_team_pass_plays"]
                team_rush = df["expected_team_rush_plays"]
                eps = 1e-6
                # Only rescale where team expectations are positive to avoid
                # blowing up very small totals.
                tgt_mask = (team_pass > 0.0) & (sum_targets > 0.0)
                car_mask = (team_rush > 0.0) & (sum_carries > 0.0)
                scale_tgt = np.ones_like(team_pass, dtype=np.float32)
                scale_car = np.ones_like(team_rush, dtype=np.float32)
                scale_tgt[tgt_mask] = (team_pass[tgt_mask] / (sum_targets[tgt_mask] + eps)).clip(0.25, 4.0)
                scale_car[car_mask] = (team_rush[car_mask] / (sum_carries[car_mask] + eps)).clip(0.25, 4.0)
                df["expected_targets"] = df["expected_targets"] * scale_tgt
                df["expected_carries"] = df["expected_carries"] * scale_car
            df["expected_opportunities"] = df["expected_targets"] + df["expected_carries"]
            if "availability_uncertainty" in df.columns:
                unc = pd.to_numeric(df["availability_uncertainty"], errors="coerce").fillna(0.0).clip(0.0, 0.25)
                df["expected_opportunities_stable"] = (
                    df["expected_opportunities"] * (1.0 - unc)
                ).astype(np.float32)
        if "expected_targets_raw" in df.columns and "expected_carries_raw" in df.columns:
            df["expected_opportunities_raw"] = df["expected_targets_raw"] + df["expected_carries_raw"]

        if "pred_efficiency_tds" in df.columns and "expected_opportunities" in df.columns:
            df["expected_td_signal"] = df["pred_efficiency_tds"] * df["expected_opportunities"]
        if "pred_efficiency_tds" in df.columns and "expected_opportunities_raw" in df.columns:
            df["expected_td_signal_raw"] = df["pred_efficiency_tds"] * df["expected_opportunities_raw"]
        if (
            "pred_efficiency_rec_success" in df.columns
            and "expected_targets" in df.columns
        ):
            rec_success = pd.to_numeric(df["pred_efficiency_rec_success"], errors="coerce").clip(0.0, 1.0)
            targets = pd.to_numeric(df["expected_targets"], errors="coerce").fillna(0.0).clip(lower=0.0)
            df["expected_rec_success"] = (rec_success * targets).astype(np.float32)
        if (
            "pred_efficiency_rush_success" in df.columns
            and "expected_carries" in df.columns
        ):
            rush_success = pd.to_numeric(df["pred_efficiency_rush_success"], errors="coerce").clip(0.0, 1.0)
            carries = pd.to_numeric(df["expected_carries"], errors="coerce").fillna(0.0).clip(lower=0.0)
            df["expected_rush_success"] = (rush_success * carries).astype(np.float32)
        td_rate_rec = None
        if "pred_td_conv_rec" in df.columns:
            td_rate_rec = pd.to_numeric(df["pred_td_conv_rec"], errors="coerce").clip(0.0, 1.0)
            df["expected_td_rate_per_target"] = td_rate_rec.astype(np.float32)
        td_rate_rush = None
        if "pred_td_conv_rush" in df.columns:
            td_rate_rush = pd.to_numeric(df["pred_td_conv_rush"], errors="coerce").clip(0.0, 1.0)
            df["expected_td_rate_per_carry"] = td_rate_rush.astype(np.float32)
        if td_rate_rec is not None and "expected_targets" in df.columns:
            expected_targets = pd.to_numeric(df["expected_targets"], errors="coerce").fillna(0.0).clip(lower=0.0)
            df["expected_td_from_targets"] = (td_rate_rec * expected_targets).astype(np.float32)
        if td_rate_rush is not None and "expected_carries" in df.columns:
            expected_carries = pd.to_numeric(df["expected_carries"], errors="coerce").fillna(0.0).clip(lower=0.0)
            df["expected_td_from_carries"] = (td_rate_rush * expected_carries).astype(np.float32)
        if td_rate_rec is not None and "expected_rz_targets" in df.columns:
            rz_targets = pd.to_numeric(df["expected_rz_targets"], errors="coerce").fillna(0.0).clip(lower=0.0)
            df["expected_rz_td_from_targets"] = (td_rate_rec * rz_targets).astype(np.float32)
        if td_rate_rush is not None and "expected_rz_carries" in df.columns:
            rz_carries = pd.to_numeric(df["expected_rz_carries"], errors="coerce").fillna(0.0).clip(lower=0.0)
            df["expected_rz_td_from_carries"] = (td_rate_rush * rz_carries).astype(np.float32)
        td_components: list[pd.Series] = []
        for col in ("expected_td_from_targets", "expected_td_from_carries"):
            if col in df.columns:
                td_components.append(pd.to_numeric(df[col], errors="coerce").fillna(0.0))
        if td_components:
            total_td = td_components[0]
            for comp in td_components[1:]:
                total_td = total_td.add(comp, fill_value=0.0)
            total_td = total_td.clip(lower=0.0)
            df["expected_td_count_structural"] = total_td.astype(np.float32)
            df["expected_td_prob_structural"] = (1.0 - np.exp(-total_td)).astype(np.float32)
            df["anytime_td_structural"] = df["expected_td_prob_structural"]
            if "expected_opportunities" in df.columns:
                opp = pd.to_numeric(df["expected_opportunities"], errors="coerce").fillna(0.0)
                opp_safe = opp.replace(0, np.nan)
                df["expected_td_rate_per_opportunity"] = (total_td / opp_safe).astype(np.float32)

        def _expected_red_zone(base_col: str, share_col: str | None, out_col: str) -> None:
            if base_col in df.columns and share_col:
                share = pd.to_numeric(df[share_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
                df[out_col] = df[base_col] * share

        rz_target_share = _select_column(
            [
                "hist_red_zone_target_share_prev",
                "hist_red_zone_target_share_l3",
                "red_zone_target_share_prev",
                "red_zone_target_share_l3",
            ]
        )
        rz_carry_share = _select_column(
            [
                "hist_red_zone_carry_share_prev",
                "hist_red_zone_carry_share_l3",
                "red_zone_carry_share_prev",
                "red_zone_carry_share_l3",
            ]
        )
        team_rz_rate = _select_column(
            [
                "team_red_zone_play_rate_prev",
                "team_red_zone_play_rate_l3",
            ]
        )
        team_gtg_rate = _select_column(
            [
                "team_goal_to_go_play_rate_prev",
                "team_goal_to_go_play_rate_l3",
            ]
        )

        _expected_red_zone("expected_targets", rz_target_share, "expected_rz_targets")
        _expected_red_zone("expected_targets_raw", rz_target_share, "expected_rz_targets_raw")
        _expected_red_zone("expected_carries", rz_carry_share, "expected_rz_carries")
        _expected_red_zone("expected_carries_raw", rz_carry_share, "expected_rz_carries_raw")

        def _team_factor(rate_col: str | None, out_col: str) -> None:
            if rate_col and rate_col in df.columns:
                df[out_col] = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)

        _team_factor(team_rz_rate, "team_red_zone_rate")
        _team_factor(team_gtg_rate, "team_goal_to_go_rate")

        if "expected_opportunities" in df.columns and "team_red_zone_rate" in df.columns:
            df["team_based_rz_opportunities"] = df["expected_opportunities"] * df["team_red_zone_rate"]
        if "expected_opportunities_raw" in df.columns and "team_red_zone_rate" in df.columns:
            df["team_based_rz_opportunities_raw"] = df["expected_opportunities_raw"] * df["team_red_zone_rate"]

        def _combine_rz(base: str, team_col: str, out_col: str):
            if base in df.columns and team_col in df.columns:
                df[out_col] = df[base].fillna(0.0) + df[team_col].fillna(0.0)

        _combine_rz("expected_rz_opportunities", "team_based_rz_opportunities", "expected_rz_opportunities")
        _combine_rz("expected_rz_opportunities_raw", "team_based_rz_opportunities_raw", "expected_rz_opportunities_raw")

        if "expected_rz_targets" in df.columns and "expected_rz_carries" in df.columns:
            df["expected_rz_opportunities"] = df["expected_rz_targets"] + df["expected_rz_carries"]
        if "expected_rz_targets_raw" in df.columns and "expected_rz_carries_raw" in df.columns:
            df["expected_rz_opportunities_raw"] = df["expected_rz_targets_raw"] + df["expected_rz_carries_raw"]

        if "pred_efficiency_tds" in df.columns and "expected_rz_opportunities" in df.columns:
            df["expected_rz_td_signal"] = df["pred_efficiency_tds"] * df["expected_rz_opportunities"]
        if "pred_efficiency_tds" in df.columns and "expected_rz_opportunities_raw" in df.columns:
            df["expected_rz_td_signal_raw"] = df["pred_efficiency_tds"] * df["expected_rz_opportunities_raw"]

        if "pred_efficiency_rec_yards" in df.columns:
            rec_rate = pd.to_numeric(df["pred_efficiency_rec_yards"], errors="coerce").clip(lower=0.0)
            if "expected_targets" in df.columns:
                targets = pd.to_numeric(df["expected_targets"], errors="coerce").fillna(0.0).clip(lower=0.0)
                df["expected_receiving_yards"] = (rec_rate * targets).astype(np.float32)
            else:
                df["expected_receiving_yards"] = rec_rate.astype(np.float32)
        if "pred_efficiency_rush_yards" in df.columns:
            rush_rate = pd.to_numeric(df["pred_efficiency_rush_yards"], errors="coerce").clip(lower=0.0)
            if "expected_carries" in df.columns:
                carries = pd.to_numeric(df["expected_carries"], errors="coerce").fillna(0.0).clip(lower=0.0)
                df["expected_rushing_yards"] = (rush_rate * carries).astype(np.float32)
            else:
                df["expected_rushing_yards"] = rush_rate.astype(np.float32)
        if "expected_receiving_yards" in df.columns and "expected_rushing_yards" in df.columns:
            total_yards = df["expected_receiving_yards"] + df["expected_rushing_yards"]
            df["expected_total_yards"] = total_yards
            if "expected_opportunities" in df.columns:
                opp = pd.to_numeric(df["expected_opportunities"], errors="coerce").fillna(0.0)
                opp_safe = opp.replace(0, np.nan)
                df["expected_yards_per_opportunity"] = (total_yards / opp_safe).astype(np.float32)

        if "expected_td_signal" in df.columns and "expected_td_prob_poisson" not in df.columns:
            df["expected_td_prob_poisson"] = 1.0 - np.exp(-df["expected_td_signal"].clip(lower=0.0))
        if "expected_td_signal_raw" in df.columns and "expected_td_prob_poisson_raw" not in df.columns:
            df["expected_td_prob_poisson_raw"] = 1.0 - np.exp(-df["expected_td_signal_raw"].clip(lower=0.0))
        if "expected_rz_td_signal" in df.columns and "expected_rz_td_prob_poisson" not in df.columns:
            df["expected_rz_td_prob_poisson"] = 1.0 - np.exp(-df["expected_rz_td_signal"].clip(lower=0.0))
        if "expected_rz_td_signal_raw" in df.columns and "expected_rz_td_prob_poisson_raw" not in df.columns:
            df["expected_rz_td_prob_poisson_raw"] = 1.0 - np.exp(-df["expected_rz_td_signal_raw"].clip(lower=0.0))

        return df

    def _persist_usage_expected_metrics(self, problem_name: str, df_problem: pd.DataFrame, preds: np.ndarray) -> None:
        usage_meta = {
            "usage_targets": {
                "share_col": "target_share_label",
                "total_col": "team_targets_total",
                "actual_col": "target",
                "expected_col": "expected_targets",
            },
            "usage_carries": {
                "share_col": "carry_share_label",
                "total_col": "team_carries_total",
                "actual_col": "carry",
                "expected_col": "expected_carries",
            },
        }
        meta = usage_meta.get(problem_name)
        if not meta or df_problem is None or preds is None:
            return

        try:
            work = df_problem.copy()
            pred_col = f"pred_{problem_name}"
            work[pred_col] = preds
            work = self._inject_composed_features(work)
            if meta["expected_col"] not in work.columns:
                logger.warning("Usage diagnostics skipped for %s (missing %s).", problem_name, meta["expected_col"])
                return
            expected_series = pd.to_numeric(work[meta["expected_col"]], errors="coerce")
            actual_series = None

            if meta["share_col"] in work.columns and meta["total_col"] in work.columns:
                share = pd.to_numeric(work[meta["share_col"]], errors="coerce")
                totals = pd.to_numeric(work[meta["total_col"]], errors="coerce")
                actual_series = share * totals
            if actual_series is None or actual_series.isna().all():
                if meta["actual_col"] in work.columns:
                    actual_series = pd.to_numeric(work[meta["actual_col"]], errors="coerce")

            if actual_series is None:
                logger.warning("Usage diagnostics skipped for %s (missing actual counts).", problem_name)
                return

            mask = expected_series.notna() & actual_series.notna()
            if not mask.any():
                logger.warning("Usage diagnostics skipped for %s (no overlapping rows).", problem_name)
                return

            expected = expected_series[mask]
            actual = actual_series[mask]
            diff = expected - actual

            try:
                r2 = r2_score(actual, expected)
            except Exception:
                r2 = float("nan")
            std_expected = float(expected.std(ddof=0))
            std_actual = float(actual.std(ddof=0))
            corr = float(actual.corr(expected)) if std_expected > 0 and std_actual > 0 else float("nan")

            metrics = {
                "samples": int(mask.sum()),
                "actual_mean": float(actual.mean()),
                "expected_mean": float(expected.mean()),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(np.square(diff)))),
                "r2": float(r2) if not math.isnan(r2) else None,
                "corr": corr if not math.isnan(corr) else None,
            }

            if "position" in work.columns:
                pos_df = pd.DataFrame(
                    {
                        "position": work.loc[mask, "position"],
                        "actual": actual,
                        "expected": expected,
                    }
                )
                pos_stats: dict[str, dict[str, float]] = {}
                for pos, grp in pos_df.groupby("position"):
                    if grp.empty:
                        continue
                    delta = grp["expected"] - grp["actual"]
                    pos_stats[str(pos)] = {
                        "samples": int(len(grp)),
                        "actual_mean": float(grp["actual"].mean()),
                        "expected_mean": float(grp["expected"].mean()),
                        "mae": float(np.mean(np.abs(delta))),
                    }
                metrics["position_breakdown"] = pos_stats

            out_dir = _vdir(self, problem_name, "xgboost", "metrics")
            diag_path = out_dir / "expected_counts.yaml"
            with diag_path.open("w") as fh:
                yaml.safe_dump(metrics, fh, sort_keys=False)
            logger.info(
                "Wrote usage expected-count diagnostics for %s → %s (samples=%d)",
                problem_name,
                diag_path,
                metrics["samples"],
            )
        except Exception as exc:
            logger.warning("Failed to persist usage diagnostics for %s: %s", problem_name, exc)

    def _persist_efficiency_metrics(self, problem_name: str, df_problem: pd.DataFrame, preds: np.ndarray) -> None:
        target_map = {
            "efficiency_rec_success": "rec_success_rate_label",
            "efficiency_rush_success": "rush_success_rate_label",
            "efficiency_rec_yards": "yards_per_target_label",
            "efficiency_rush_yards": "yards_per_carry_label",
        }
        target_col = target_map.get(problem_name)
        if not target_col or target_col not in df_problem.columns or preds is None:
            return

        try:
            actual = pd.to_numeric(df_problem[target_col], errors="coerce")
            predicted = pd.Series(preds, index=df_problem.index)
            mask = actual.notna() & predicted.notna()
            if not mask.any():
                logger.warning("Efficiency diagnostics skipped for %s (no overlapping rows).", problem_name)
                return

            actual = actual[mask]
            predicted = predicted[mask]
            diff = predicted - actual

            try:
                r2 = r2_score(actual, predicted)
            except Exception:
                r2 = float("nan")
            std_pred = float(predicted.std(ddof=0))
            std_actual = float(actual.std(ddof=0))
            corr = float(actual.corr(predicted)) if std_pred > 0 and std_actual > 0 else float("nan")

            metrics = {
                "samples": int(mask.sum()),
                "actual_mean": float(actual.mean()),
                "pred_mean": float(predicted.mean()),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(np.square(diff)))),
                "r2": float(r2) if not math.isnan(r2) else None,
                "corr": corr if not math.isnan(corr) else None,
            }

            if "position" in df_problem.columns:
                pos_stats: dict[str, dict[str, float]] = {}
                pos_series = df_problem.loc[mask, "position"]
                for pos, grp in pd.DataFrame(
                    {
                        "position": pos_series,
                        "actual": actual,
                        "pred": predicted,
                    }
                ).groupby("position"):
                    if grp.empty:
                        continue
                    delta = grp["pred"] - grp["actual"]
                    pos_stats[str(pos)] = {
                        "samples": int(len(grp)),
                        "actual_mean": float(grp["actual"].mean()),
                        "pred_mean": float(grp["pred"].mean()),
                        "mae": float(np.mean(np.abs(delta))),
                    }
                metrics["position_breakdown"] = pos_stats

            out_dir = _vdir(self, problem_name, "xgboost", "metrics")
            diag_path = out_dir / "efficiency_metrics.yaml"
            with diag_path.open("w") as fh:
                yaml.safe_dump(metrics, fh, sort_keys=False)
            logger.info(
                "Wrote efficiency diagnostics for %s → %s (samples=%d)",
                problem_name,
                diag_path,
                metrics["samples"],
            )
        except Exception as exc:
            logger.warning("Failed to persist efficiency diagnostics for %s: %s", problem_name, exc)

    def _persist_td_conversion_metrics(self, problem_name: str, df_problem: pd.DataFrame, preds: np.ndarray) -> None:
        target_map = {
            "td_conv_rec": "td_per_target_label",
            "td_conv_rush": "td_per_carry_label",
        }
        target_col = target_map.get(problem_name)
        if not target_col or target_col not in df_problem.columns or preds is None:
            return

        try:
            actual = pd.to_numeric(df_problem[target_col], errors="coerce")
            predicted = pd.Series(preds, index=df_problem.index)
            mask = actual.notna() & predicted.notna()
            if not mask.any():
                logger.warning("TD conversion diagnostics skipped for %s (no overlapping rows).", problem_name)
                return

            actual = actual[mask]
            predicted = predicted[mask]
            diff = predicted - actual

            try:
                r2 = r2_score(actual, predicted)
            except Exception:
                r2 = float("nan")
            std_pred = float(predicted.std(ddof=0))
            std_actual = float(actual.std(ddof=0))
            corr = float(actual.corr(predicted)) if std_pred > 0 and std_actual > 0 else float("nan")

            metrics = {
                "samples": int(mask.sum()),
                "actual_mean": float(actual.mean()),
                "pred_mean": float(predicted.mean()),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(np.square(diff)))),
                "r2": float(r2) if not math.isnan(r2) else None,
                "corr": corr if not math.isnan(corr) else None,
            }

            if "position" in df_problem.columns:
                pos_stats: dict[str, dict[str, float]] = {}
                pos_series = df_problem.loc[mask, "position"]
                pos_df = pd.DataFrame(
                    {
                        "position": pos_series,
                        "actual": actual,
                        "pred": predicted,
                    }
                )
                for pos, grp in pos_df.groupby("position"):
                    if grp.empty:
                        continue
                    delta = grp["pred"] - grp["actual"]
                    pos_stats[str(pos)] = {
                        "samples": int(len(grp)),
                        "actual_mean": float(grp["actual"].mean()),
                        "pred_mean": float(grp["pred"].mean()),
                        "mae": float(np.mean(np.abs(delta))),
                    }
                metrics["position_breakdown"] = pos_stats

            out_dir = _vdir(self, problem_name, "xgboost", "metrics")
            diag_path = out_dir / "td_conversion_metrics.yaml"
            with diag_path.open("w") as fh:
                yaml.safe_dump(metrics, fh, sort_keys=False)
            logger.info(
                "Wrote TD conversion diagnostics for %s → %s (samples=%d)",
                problem_name,
                diag_path,
                metrics["samples"],
            )
        except Exception as exc:
            logger.warning("Failed to persist TD conversion diagnostics for %s: %s", problem_name, exc)

    def _persist_snaps_metrics(self, df_problem: pd.DataFrame, preds: np.ndarray) -> None:
        target_col = "snaps_label"
        if target_col not in df_problem.columns or preds is None:
            return
        try:
            actual = pd.to_numeric(df_problem[target_col], errors="coerce")
            predicted = pd.Series(preds, index=df_problem.index)
            mask = actual.notna() & predicted.notna()
            if not mask.any():
                logger.warning("Snaps diagnostics skipped (no overlapping rows).")
                return
            actual = actual[mask]
            predicted = predicted[mask]
            diff = predicted - actual
            try:
                r2 = r2_score(actual, predicted)
            except Exception:
                r2 = float("nan")
            metrics = {
                "samples": int(mask.sum()),
                "actual_mean": float(actual.mean()),
                "pred_mean": float(predicted.mean()),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(np.square(diff)))),
                "r2": float(r2) if not math.isnan(r2) else None,
            }
            if "position" in df_problem.columns:
                pos_stats: dict[str, dict[str, float]] = {}
                pos_series = df_problem.loc[mask, "position"]
                pos_df = pd.DataFrame({"position": pos_series, "actual": actual, "pred": predicted})
                for pos, grp in pos_df.groupby("position"):
                    if grp.empty:
                        continue
                    delta = grp["pred"] - grp["actual"]
                    pos_stats[str(pos)] = {
                        "samples": int(len(grp)),
                        "actual_mean": float(grp["actual"].mean()),
                        "pred_mean": float(grp["pred"].mean()),
                        "mae": float(np.mean(np.abs(delta))),
                    }
                metrics["position_breakdown"] = pos_stats
            out_dir = _vdir(self, "snaps", "xgboost", "metrics")
            diag_path = out_dir / "snaps_metrics.yaml"
            with diag_path.open("w") as fh:
                yaml.safe_dump(metrics, fh, sort_keys=False)
            logger.info("Wrote snaps diagnostics → %s (samples=%d)", diag_path, metrics["samples"])
        except Exception as exc:
            logger.warning("Failed to persist snaps diagnostics: %s", exc)

    def _persist_anytime_td_structured_metrics(self, df_problem: pd.DataFrame, preds: np.ndarray) -> None:
        if "anytime_td_structural" not in df_problem.columns or preds is None:
            logger.warning("Structured diagnostics skipped (missing structural column or preds).")
            return
        try:
            actual = pd.to_numeric(df_problem.get("anytime_td_skill", df_problem.get("anytime_td")), errors="coerce")
            structural = pd.to_numeric(df_problem["anytime_td_structural"], errors="coerce")
            calibrated = pd.Series(preds, index=df_problem.index)
            mask_struct = actual.notna() & structural.notna()
            mask_cal = actual.notna() & calibrated.notna()
            metrics = {}

            def _class_metrics(y_true, y_pred) -> dict[str, float | None]:
                out: dict[str, float | None] = {}
                try:
                    out["auc"] = roc_auc_score(y_true, y_pred)
                except Exception:
                    out["auc"] = None
                try:
                    out["pr_auc"] = average_precision_score(y_true, y_pred)
                except Exception:
                    out["pr_auc"] = None
                try:
                    eps = np.clip(y_pred, 1e-6, 1 - 1e-6)
                    out["log_loss"] = log_loss(y_true, eps)
                except Exception:
                    out["log_loss"] = None
                try:
                    out["brier"] = brier_score_loss(y_true, y_pred)
                except Exception:
                    out["brier"] = None
                return out

            def _calibration_table(y_true, y_pred, bins: int = 10) -> list[dict[str, float]]:
                df = pd.DataFrame({"y": y_true, "p": y_pred})
                df = df.sort_values("p").reset_index(drop=True)
                df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
                table: list[dict[str, float]] = []
                for interval, grp in df.groupby("bin"):
                    if grp.empty:
                        continue
                    start = float(interval.left) if interval is not None else float(grp["p"].min())
                    end = float(interval.right) if interval is not None else float(grp["p"].max())
                    table.append(
                        {
                            "bin_start": start,
                            "bin_end": end,
                            "avg_pred": float(grp["p"].mean()),
                            "actual_rate": float(grp["y"].mean()),
                            "count": int(len(grp)),
                        }
                    )
                return table

            diagnostics: dict[str, dict[str, object]] = {}
            if mask_struct.any():
                diagnostics["structural"] = {
                    **_class_metrics(actual[mask_struct], structural[mask_struct]),
                    "calibration": _calibration_table(actual[mask_struct], structural[mask_struct]),
                    "samples": int(mask_struct.sum()),
                }
            if mask_cal.any():
                diagnostics["structured_model"] = {
                    **_class_metrics(actual[mask_cal], calibrated[mask_cal]),
                    "calibration": _calibration_table(actual[mask_cal], calibrated[mask_cal]),
                    "samples": int(mask_cal.sum()),
                }

            out_dir = _vdir(self, "anytime_td_structured", "xgboost", "metrics")
            diag_path = out_dir / "structured_vs_structural.yaml"
            with diag_path.open("w") as fh:
                yaml.safe_dump(diagnostics, fh, sort_keys=False)
            logger.info(
                "Wrote structured vs structural diagnostics → %s (struct samples=%s, model samples=%s)",
                diag_path,
                diagnostics.get("structural", {}).get("samples"),
                diagnostics.get("structured_model", {}).get("samples"),
            )
        except Exception as exc:
            logger.warning("Failed to persist structured diagnostics: %s", exc)

    def _apply_problem_level_overrides(self, df: pd.DataFrame, problem_config: dict) -> pd.DataFrame:
        derived_cfg = problem_config.get("derived_target")
        if derived_cfg:
            df = self._derive_target_column(
                df=df,
                target_col=problem_config["target_col"],
                derived_cfg=derived_cfg,
                problem_name=problem_config["name"],
            )
        return df

    def _ensure_team_game_actuals(self) -> pd.DataFrame:
        if hasattr(self, "_team_game_actuals") and self._team_game_actuals is not None:
            return self._team_game_actuals

        path_pattern = Path("data/processed/game_by_week/season=*/week=*/part.parquet")
        if not Path("data/processed/game_by_week").exists():
             # fallback or error
             logger.warning("game_by_week not found, cannot load team actuals.")
             return pd.DataFrame()
        
        scan = pl.scan_parquet(str(path_pattern), glob=True, hive_partitioning=True)
        schema = scan.collect_schema().names()
        cols = ["game_id", "season", "week", "posteam", "n_plays"]
        cols = [c for c in cols if c in schema]
        df = scan.select(cols).collect().to_pandas()
        
        # Standardize columns to match feature matrix
        if "posteam" in df.columns:
            df.rename(columns={"posteam": "team"}, inplace=True)
        if "game_id" in df.columns:
            df["game_id"] = df["game_id"].astype(str)
            
        self._team_game_actuals = df
        logger.info("Loaded team-game actuals with %d rows.", len(df))
        return self._team_game_actuals

    def _derive_target_column(
        self,
        df: pd.DataFrame,
        target_col: str,
        derived_cfg: dict,
        problem_name: str,
    ) -> pd.DataFrame:
        source_col = derived_cfg.get("source_col")
        if not source_col:
            raise ValueError(f"Problem '{problem_name}' missing source_col for derived target.")
        
        external_source = derived_cfg.get("external_source")
        if external_source == "game_by_week":
             actuals = self._ensure_team_game_actuals()
             if actuals.empty or source_col not in actuals.columns:
                  raise ValueError(f"Cannot derive {target_col} from game_by_week (missing source or data).")
             
             # Join on game_id + team
             merge_keys = ["game_id", "team"]
             if not all(k in df.columns for k in merge_keys):
                  raise ValueError(f"Feature matrix missing merge keys {merge_keys} for team target derivation.")
             
             merged = df.merge(actuals[merge_keys + [source_col]], on=merge_keys, how="left")
             merged.rename(columns={source_col: target_col}, inplace=True)
             
             before = len(merged)
             merged = merged[merged[target_col].notna()].reset_index(drop=True)
             dropped = before - len(merged)
             if dropped:
                 logger.info("Dropped %d rows missing team actuals for %s", dropped, problem_name)
             return merged

        if source_col not in df.columns:
            if external_source == "player_game":
                df = self._attach_player_game_stat(df, source_col)
            else:
                raise ValueError(
                    f"Problem '{problem_name}' cannot derive target; '{source_col}' not in frame."
                )
        
        # ... existing shift logic for standard derivation ...
        group_cols = derived_cfg.get("group_cols") or ["team"]
        missing_groups = [col for col in group_cols if col not in df.columns]
        if missing_groups:
            raise ValueError(
                f"Problem '{problem_name}' missing group columns for derived target: {missing_groups}"
            )
        order_cols = [col for col in (derived_cfg.get("order_cols") or []) if col in df.columns]
        if not order_cols:
            order_cols = [self.time_col] if self.time_col in df.columns else []
        sort_cols = group_cols + order_cols
        df = df.sort_values(sort_cols).reset_index(drop=True)
        shift = int(derived_cfg.get("shift", -1))
        df[target_col] = (
            df.groupby(group_cols, sort=False)[source_col]
            .shift(shift)
            .astype(np.float32)
        )
        before = len(df)
        df = df[df[target_col].notna()].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.info(
                "Dropped %d rows with missing derived target for problem '%s'.",
                dropped,
                problem_name,
            )
        return df

    @staticmethod
    def _filter_odds_snapshot_columns(df: pd.DataFrame, horizon_key: str | None) -> pd.DataFrame:
        """Keep a single odds snapshot horizon (drop other suffixes)."""
        if df is None or df.empty or not horizon_key:
            return df

        normalized = str(horizon_key).strip().lower()
        horizon_map = {
            "cutoff": "",
            "default": "",
            "": "",
            "2h": "_2h",
            "h120m": "_2h",
            "6h": "_6h",
            "h360m": "_6h",
            "24h": "_24h",
            "h1440m": "_24h",
            "open": "_open",
        }
        allowed_suffix = horizon_map.get(normalized, normalized)
        if allowed_suffix and not allowed_suffix.startswith("_"):
            allowed_suffix = f"_{allowed_suffix.lstrip('_')}"

        horizon_re = re.compile(r"^(?P<base>.+)(?P<suffix>_2h|_6h|_24h|_open)$")
        suffixes_seen: dict[str, set[str]] = {}
        for col in df.columns:
            match = horizon_re.match(col)
            if match:
                suffixes_seen.setdefault(match.group("base"), set()).add(match.group("suffix"))

        drop_cols: list[str] = []
        for col in df.columns:
            match = horizon_re.match(col)
            if match:
                suffix = match.group("suffix")
                if suffix != allowed_suffix:
                    drop_cols.append(col)
            else:
                base = col
                if base in suffixes_seen and allowed_suffix:
                    drop_cols.append(col)

        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
            logger.info(
                "Removed %d odds snapshot columns outside horizon '%s'.",
                len(drop_cols),
                horizon_key,
            )
        return df

    def _ensure_player_game_actuals(self) -> pd.DataFrame:
        if self._player_game_actuals is not None:
            return self._player_game_actuals

        path_pattern = PLAYER_GAME_DIR / "season=*/week=*/part.parquet"
        if not PLAYER_GAME_DIR.exists():
            raise FileNotFoundError(
                f"Player-game directory missing at {PLAYER_GAME_DIR}; cannot derive yardage targets."
            )

        scan = pl.scan_parquet(
            str(path_pattern),
            glob=True,
            hive_partitioning=True,
            missing_columns="insert",
            extra_columns="ignore",
        )
        schema_names = scan.collect_schema().names()
        required = {"game_id", "player_id"}
        if not required.issubset(schema_names):
            raise ValueError("Player-game parquet missing required keys for target derivation.")
        stat_cols = [col for col in ("receiving_yards", "rushing_yards") if col in schema_names]
        selection = ["game_id", "player_id", *stat_cols]
        df = scan.select([pl.col(col) for col in selection]).collect(streaming=True)
        actuals = df.to_pandas()
        actuals["game_id"] = actuals["game_id"].astype(str)
        actuals["player_id"] = actuals["player_id"].astype(str)
        self._player_game_actuals = actuals
        logger.info(
            "Loaded player-game actuals for derived targets with %d rows.",
            len(actuals),
        )
        return self._player_game_actuals

    def _attach_player_game_stat(self, df: pd.DataFrame, stat_col: str) -> pd.DataFrame:
        actuals = self._ensure_player_game_actuals()
        if stat_col not in actuals.columns:
            raise ValueError(f"Player-game actuals do not contain column '{stat_col}'.")
        if "game_id" not in df.columns or "player_id" not in df.columns:
            raise ValueError("Cannot attach player-game stat without 'game_id' and 'player_id'.")
        merged = df.merge(
            actuals[["game_id", "player_id", stat_col]],
            on=["game_id", "player_id"],
            how="left",
        )
        return merged

    def _apply_availability_guards(
        self,
        df_slice: pd.DataFrame | None,
        preds: np.ndarray,
    ) -> np.ndarray:
        """
        Apply deterministic post-processing rules to availability predictions.
        These guards zero-out players who are ruled inactive and dampen snaps
        when practice participation signals concern.
        """
        if df_slice is None or preds.size == 0:
            return preds

        adjusted = preds.astype(np.float32, copy=True)

        def _series(name: str) -> pd.Series | None:
            if name in df_slice.columns:
                return df_slice[name]
            return None

        inactive_flag = _series("injury_is_inactive_designation")
        if inactive_flag is not None:
            mask = pd.to_numeric(inactive_flag, errors="coerce").fillna(0).astype(bool).to_numpy()
            adjusted[mask] = 0.0

        designation = _series("injury_game_designation")
        if designation is not None:
            designation = designation.fillna("").astype(str).str.upper()
            out_mask = designation.isin(["OUT", "INACTIVE"])
            doubtful_mask = designation == "DOUBTFUL"
            questionable_mask = designation == "QUESTIONABLE"
            adjusted[out_mask.to_numpy()] = 0.0
            adjusted[doubtful_mask.to_numpy()] *= 0.25
            adjusted[questionable_mask.to_numpy()] *= 0.85

        practice_status = _series("injury_practice_status")
        if practice_status is not None:
            practice_status = practice_status.fillna("").astype(str).str.upper()
            dnp_mask = practice_status.str.contains("DID NOT PARTICIPATE", regex=False)
            limited_mask = practice_status.str.contains("LIMITED", regex=False)
            adjusted[dnp_mask.to_numpy()] *= 0.55
            adjusted[limited_mask.to_numpy()] *= 0.85

        prob_model = _series("injury_inactive_probability_model")
        prob_heuristic = _series("injury_inactive_probability")
        prob = None
        if prob_model is not None:
            prob = pd.to_numeric(prob_model, errors="coerce")
        elif prob_heuristic is not None:
            prob = pd.to_numeric(prob_heuristic, errors="coerce")
        if prob is not None:
            prob = prob.fillna(0.0).clip(lower=0.0, upper=0.95)
            high_prob_mask = prob >= 0.5
            if high_prob_mask.any():
                adj = (1.0 - prob[high_prob_mask]).to_numpy(dtype=np.float32, copy=False)
                adjusted[high_prob_mask.to_numpy()] *= np.clip(adj, 0.05, 1.0)

        np.clip(adjusted, 0.0, 1.0, out=adjusted)
        return adjusted

    def _apply_snaps_ceiling_cap(
        self,
        df_slice: pd.DataFrame | None,
        preds: np.ndarray,
    ) -> np.ndarray:
        """
        Apply post-processing cap for snaps predictions.
        
        For low-usage players (max_snap_pct_l5 < 0.25), cap predictions at
        snap_ceiling_l5 * 1.5 to prevent absurd overpredictions for specialists.
        
        Analysis showed:
        - Low snaps (1-15): MAE improved from 15.06 to 11.85 (-3.21)
        - Zero snaps: MAE improved from 6.01 to 4.01 (-2.00)
        - Fullbacks: MAE improved from 9.88 to 8.73 (-1.15)
        - High usage (>25% max): Completely unaffected (0.00 change)
        - Starters (>40): Minimal impact (+0.22 MAE)
        """
        if df_slice is None or preds.size == 0:
            return preds
        
        adjusted = preds.copy()
        
        # Get ceiling and max snap % columns
        ceiling = None
        max_snap_pct = None
        
        if "snap_ceiling_l5" in df_slice.columns:
            ceiling = pd.to_numeric(df_slice["snap_ceiling_l5"], errors="coerce").to_numpy()
        if "max_snap_pct_l5" in df_slice.columns:
            max_snap_pct = pd.to_numeric(df_slice["max_snap_pct_l5"], errors="coerce").to_numpy()
        
        if ceiling is None or max_snap_pct is None:
            logger.debug("Snaps ceiling cap skipped: missing snap_ceiling_l5 or max_snap_pct_l5")
            return adjusted
        
        # Only cap low-usage players (max_snap_pct < 0.25)
        threshold = 0.25
        multiplier = 1.5
        
        should_cap = (~np.isnan(ceiling)) & (~np.isnan(max_snap_pct)) & (max_snap_pct < threshold)
        cap_values = ceiling * multiplier
        
        # Apply cap where conditions are met and prediction exceeds cap
        capped_mask = should_cap & (adjusted > cap_values)
        adjusted[capped_mask] = cap_values[capped_mask]
        
        n_capped = capped_mask.sum()
        if n_capped > 0:
            logger.info(f"Snaps ceiling cap applied to {n_capped} predictions (max_snap_pct < {threshold}, cap = ceiling × {multiplier})")
        
        return adjusted

    def _apply_usage_targets_position_cap(
        self,
        df_slice: pd.DataFrame | None,
        preds: np.ndarray,
    ) -> np.ndarray:
        """
        Apply post-processing for usage_targets predictions.
        
        - QBs: Set target_share to 0 (99.2% have 0 targets in reality)
        - FBs: Cap at 0.10 (max realistic is ~15%, most have 0)
        
        Analysis showed:
        - QB predictions were ~11% when actual is ~0.03%
        - FB predictions were overpredicted but some FBs (Juszczyk) do get targets
        """
        if df_slice is None or preds.size == 0:
            return preds
        
        adjusted = preds.copy()
        n_zeroed = 0
        n_capped = 0
        
        # Zero out QBs
        if "position" in df_slice.columns:
            is_qb = (df_slice["position"] == "QB").to_numpy()
            n_zeroed = is_qb.sum()
            adjusted[is_qb] = 0.0
        
        # Cap FBs at 0.10
        if "is_fullback" in df_slice.columns:
            is_fb = (df_slice["is_fullback"] == 1).to_numpy()
            fb_cap = 0.10
            exceeds_cap = is_fb & (adjusted > fb_cap)
            n_capped = exceeds_cap.sum()
            adjusted[exceeds_cap] = fb_cap
        
        # Cap RBs based on historical target tier
        # Analysis shows RBs are overpredicted by 73-386% depending on tier
        if "position" in df_slice.columns and "hist_target_share_l3" in df_slice.columns:
            is_rb = (df_slice["position"] == "RB").to_numpy()
            hist_ts = pd.to_numeric(df_slice["hist_target_share_l3"], errors="coerce").fillna(0).to_numpy()
            
            # Cap RB predictions at historical target share + 50% buffer, minimum 0.03
            rb_cap = np.maximum(hist_ts * 1.5, 0.03)
            rb_exceeds = is_rb & (adjusted > rb_cap)
            n_rb_capped = rb_exceeds.sum()
            adjusted[rb_exceeds] = rb_cap[rb_exceeds]
            
            if n_rb_capped > 0:
                logger.info(f"Usage targets: capped {n_rb_capped} RB predictions at hist_target_share × 1.5")
        
        if n_zeroed > 0 or n_capped > 0:
            logger.info(f"Usage targets post-processing: zeroed {n_zeroed} QBs, capped {n_capped} FBs at 0.10")
        
        return adjusted

    def _apply_usage_carries_position_cap(
        self,
        df_slice: pd.DataFrame | None,
        preds: np.ndarray,
    ) -> np.ndarray:
        """
        Apply post-processing for usage_carries predictions.
        
        Data analysis shows:
        - WRs: 90.1% have 0 carries, mean 0.005 → Cap at 0.05
        - TEs: 98.0% have 0 carries, mean 0.001 → Cap at 0.02
        - QBs: 58.8% have 0 carries, mean 0.06, but rushing QBs can have 25%+ → Cap at 0.30
        - FBs: 85.2% have 0 carries, mean 0.007 → Cap at 0.10
        - RBs: Keep as-is (primary ball carriers)
        """
        if df_slice is None or preds.size == 0:
            return preds
        
        adjusted = preds.copy()
        n_capped = 0
        
        if "position" in df_slice.columns:
            position = df_slice["position"].to_numpy()
            
            # Cap WRs at 0.05
            is_wr = position == "WR"
            wr_exceeds = is_wr & (adjusted > 0.05)
            adjusted[wr_exceeds] = 0.05
            n_capped += wr_exceeds.sum()
            
            # Cap TEs at 0.02
            is_te = position == "TE"
            te_exceeds = is_te & (adjusted > 0.02)
            adjusted[te_exceeds] = 0.02
            n_capped += te_exceeds.sum()
            
            # Cap QBs at 0.30 (rushing QBs like Lamar, Hurts can have ~25%)
            is_qb = position == "QB"
            qb_exceeds = is_qb & (adjusted > 0.30)
            adjusted[qb_exceeds] = 0.30
            n_capped += qb_exceeds.sum()
        
        # Cap FBs at 0.10
        if "is_fullback" in df_slice.columns:
            is_fb = (df_slice["is_fullback"] == 1).to_numpy()
            fb_exceeds = is_fb & (adjusted > 0.10)
            adjusted[fb_exceeds] = 0.10
            n_capped += fb_exceeds.sum()
        
        if n_capped > 0:
            logger.info(f"Usage carries post-processing: capped {n_capped} non-RB predictions")
        
        return adjusted

    def _apply_usage_target_yards_position_cap(
        self,
        df_slice: pd.DataFrame | None,
        preds: np.ndarray,
    ) -> np.ndarray:
        """
        Apply post-processing for usage_target_yards predictions.
        
        Based on data analysis:
        - RBs: Mean 0.0, median 0.0 - cap at 2.0 yards (tightened from 6.0)
        - FBs: Rarely targeted, always short - set to 0
        - QBs: Only 37 games with 43 total targets in entire dataset - set to 0
        - TEs: No cap (range from -1 to 16)
        - WRs: No cap (natural deep targets)
        """
        if df_slice is None or preds.size == 0:
            return preds
        
        adjusted = preds.copy()
        n_zeroed = 0
        n_capped = 0
        
        if "position" in df_slice.columns:
            position = df_slice["position"].to_numpy()
            
            # QBs: Set to 0 (extremely rare, trick plays only)
            is_qb = position == "QB"
            n_zeroed = is_qb.sum()
            adjusted[is_qb] = 0.0
            
            # RBs: Cap between 0 and 2 yards (median is 0, most are screens/checkdowns)
            is_rb = position == "RB"
            adjusted[is_rb] = np.clip(adjusted[is_rb], 0.0, 2.0)
            n_capped += (is_rb & (preds > 2.0)).sum()
        
        # FBs: Set to 0 (almost never targeted)
        if "is_fullback" in df_slice.columns:
            is_fb = (df_slice["is_fullback"] == 1).to_numpy()
            adjusted[is_fb] = 0.0
        
        if n_zeroed > 0 or n_capped > 0:
            logger.info(f"Usage target yards post-processing: zeroed {n_zeroed} QBs, capped {n_capped} RBs")
        
        return adjusted

    def _apply_efficiency_rec_yards_air_cap(
        self,
        df_slice: pd.DataFrame | None,
        preds: np.ndarray,
    ) -> np.ndarray:
        """
        Apply post-processing for efficiency_rec_yards_air predictions.
        
        Based on data analysis:
        - RBs: Mean -0.6 (checkdowns behind line) - cap at 0
        - TEs: Mean 5.5 - no cap
        - WRs: Mean 8.9 - no cap
        """
        if df_slice is None or preds.size == 0:
            return preds
        
        adjusted = preds.copy()
        n_capped = 0
        
        if "position" in df_slice.columns:
            position = df_slice["position"].to_numpy()
            
            # RBs: Cap at 0 (they get checkdowns, often behind line of scrimmage)
            is_rb = position == "RB"
            exceeds = is_rb & (adjusted > 0)
            adjusted[exceeds] = 0.0
            n_capped = exceeds.sum()
        
        # FBs: Also cap at 0
        if "is_fullback" in df_slice.columns:
            is_fb = (df_slice["is_fullback"] == 1).to_numpy()
            adjusted[is_fb] = 0.0
        
        if n_capped > 0:
            logger.info(f"Efficiency rec yards air post-processing: capped {n_capped} RBs at 0")
        
        return adjusted

    def _apply_feature_artifacts(self, df, problem_name):
        """
        Applies pre-fitted feature engineering artifacts to a new DataFrame.
        """
        logger.info(f"_apply_feature_artifacts input shape: {df.shape}")
        artifacts = FeatureArtifacts(
            feature_columns=list(self.feature_columns or []),
            datetime_features=self.datetime_features.get(problem_name, []) or [],
            categorical_features=self.categorical_features.get(problem_name, []) or [],
            category_levels=self.category_levels.get(problem_name, {}) or {},
        )
        X, y = transform_with_artifacts(df, artifacts, self.target_col)
        logger.info(f"_apply_feature_artifacts reindex shape: {X.shape}")
        return X, y


    def _build_group_labels(self, df: pd.DataFrame) -> pd.Series:
        """Construct group labels prioritising the configured group_col."""
        if df is None or df.empty:
            return pd.Series(dtype="object")
        if self.group_col and self.group_col in df.columns:
            labels = df[self.group_col].fillna("__missing__").astype(str)
        elif {"season", "week"}.issubset(df.columns):
            season = df["season"].fillna(-1).astype(int).astype(str)
            week = df["week"].fillna(-1).astype(int).astype(str)
            labels = season + "-" + week
        elif self.time_col in df.columns:
            dates = pd.to_datetime(df[self.time_col], errors="coerce").dt.strftime("%Y-%m-%d")
            labels = dates.fillna("__missing__")
        else:
            labels = pd.Series(np.arange(len(df)), dtype="object").astype(str)
        labels.index = df.index
        return labels


    def _convert_embargo_days_to_group_gap(
        self,
        df: pd.DataFrame | None,
        group_labels: pd.Series | None,
        embargo_days: int,
    ) -> int:
        """Approximate PurgedGroupTimeSeriesSplit group_gap from day-based embargo config."""
        if embargo_days <= 0:
            return 0
        if df is None or group_labels is None or df.empty:
            return max(int(embargo_days), 0)
        if self.time_col not in df.columns:
            return max(int(embargo_days), 0)
        dates = pd.to_datetime(df[self.time_col], errors="coerce").dt.normalize()
        info = (
            pd.DataFrame({"group": group_labels.values, "date": dates})
            .dropna(subset=["group", "date"])
            .drop_duplicates(subset=["group"])
        )
        if info.empty:
            return max(int(embargo_days), 0)
        day_counts = info.groupby("date")["group"].count()
        groups_per_day = float(day_counts.mean())
        if not math.isfinite(groups_per_day) or groups_per_day <= 0:
            groups_per_day = 1.0
        gap = int(math.ceil(groups_per_day * embargo_days))
        return max(gap, 1)


    def _compute_sample_weights(self, df: pd.DataFrame, problem_config: dict) -> pd.Series:
        """
        Compute per-row sample weights via shared helper.
        """
        return compute_sample_weights_util(
            df=df,
            base_cfg=self.sample_weight_cfg,
            problem_cfg=problem_config,
            target_col=self.target_col,
        )


    def _initialize_team_total_config(self, problem_name: str, df_train: pd.DataFrame) -> None:
        """Prepare team total adjustment configuration for classification problems."""
        if problem_name in self._team_total_config:
            return

        if self.target_col not in df_train.columns:
            return
        if "team_implied_total" not in df_train.columns or "team" not in df_train.columns:
            return

        totals = pd.to_numeric(df_train["team_implied_total"], errors="coerce")
        mean_total = float(totals.mean())
        if not np.isfinite(mean_total) or mean_total <= 0:
            mean_total = 21.0
        totals = totals.fillna(mean_total)

        targets = pd.to_numeric(df_train[self.target_col], errors="coerce").fillna(0.0)

        df_meta = pd.DataFrame(
            {
                "team_implied_total": totals,
                self.target_col: targets,
            },
            index=df_train.index,
        )

        group_cols = [col for col in ("season", "week", "team") if col in df_train.columns]
        if len(group_cols) < 2:
            group_cols = [col for col in ("team",) if col in df_train.columns]
        if not group_cols:
            return

        for col in group_cols:
            df_meta[col] = df_train[col]

        grouped = df_meta.groupby(group_cols, dropna=False, sort=False)
        implied_means = grouped["team_implied_total"].mean()
        actual = grouped[self.target_col].sum()
        total_implied_points = float(implied_means.sum())
        actual_total = float(actual.sum())
        if actual_total <= 1e-6 or not np.isfinite(actual_total):
            divisor = 6.8
        else:
            divisor = float(np.clip(total_implied_points / actual_total, 3.0, 12.0))
        expected = (implied_means / divisor).clip(lower=0.0)

        expected_total = float(expected.sum())
        if expected_total <= 1e-6:
            global_scale = 1.0
        else:
            global_scale = float(actual_total / expected_total)
        global_scale = float(np.clip(global_scale, 0.25, 3.0))

        # Columns that express role and usage concentration; used to weight intra-team scaling.
        weight_features = [
            # Historical share features
            "hist_target_share_prev",
            "hist_target_share_l3",
            "hist_carry_share_prev",
            "hist_carry_share_l3",
            # Offense context role shares (red zone / goal line)
            "off_ctx_player_red_zone_share",
            "off_ctx_player_goal_to_go_share",
            # Expected opportunity signal
            "expected_opportunities",
        ]
        # Filter to columns actually present in the training frame
        weight_features = [c for c in weight_features if c in df_train.columns]

        config = TeamTotalConfig(
            divisor=divisor,
            global_scale=global_scale,
            mean_total=mean_total,
            group_cols=group_cols,
            min_scale=0.25,
            max_scale=3.0,
            weight_features=weight_features,
            role_intensity=float(self.config.get("team_total_role_intensity", 0.5)),
        )
        self._team_total_config[problem_name] = config
        logger.info(
            "Team total scaling initialised for %s: groups=%s, mean_total=%.2f, global_scale=%.3f",
            problem_name,
            group_cols,
            mean_total,
            global_scale,
        )


    def _maybe_wrap_team_total_adjuster(self, problem_name: str, model):
        """Wrap model with team-total probability adjuster if configured."""
        config = self._team_total_config.get(problem_name)
        if not config:
            return model
        if isinstance(model, TeamTotalAdjustedClassifier):
            return model
        if hasattr(model, "gate_model"):
            # Selective models manage coverage internally; skip adjuster
            return model
        return TeamTotalAdjustedClassifier(model, config)





    def tune_hyperparameters(
        self,
        model_name,
        problem_config,
        X_train,
        y_train,
        groups_train,
        sample_weight=None,
        time_values=None,
    ):
        """Hyperparameter optimisation via Optuna (currently classification-only). Skips regression tasks."""
        return tune_hyperparameters_ext(
            self,
            model_name,
            problem_config,
            X_train,
            y_train,
            groups_train,
            sample_weight=sample_weight,
            time_values=time_values,
            purge_days=float(self.config['data_split'].get('purge_td', 0) or self.config['data_split'].get('purge_window_days', 0) or 0),
            purge_groups=int(self.config['data_split'].get('purge_games', self.config['data_split'].get('group_gap', 0) or 0)),
            embargo_days=float(self.config['data_split'].get('embargo_td', 0) or 0),
        )



    def _generate_oof_predictions(self, model_name, problem_config, X, y, groups_series, sample_weight=None):
        """
        Generate Out-of-Fold predictions using the shared PurgedGroupTimeSeriesSplit logic.
        """
        if groups_series is None:
            raise ValueError("groups_series is required for purged OOF generation.")
        n_splits = int(self.config['data_split'].get('n_splits', 5))
        embargo_days = int(self.config['data_split'].get('group_gap', self.config['data_split'].get('embargo_td', 0) or 0))
        df_full = self._train_full_frames.get(problem_config['name'])
        group_gap = self._convert_embargo_days_to_group_gap(df_full, groups_series, embargo_days)
        max_train = self.config['data_split'].get('max_train_group_size', np.inf)
        max_test = self.config['data_split'].get('max_test_group_size', np.inf)

        splitter = PurgedGroupTimeSeriesSplit(
            n_splits=n_splits,
            group_gap=group_gap,
            max_train_group_size=max_train if max_train is not None else np.inf,
            max_test_group_size=max_test if max_test is not None else np.inf,
        )
        groups_aligned = groups_series.loc[X.index].to_numpy()

        logger.info(
            "Generating purged OOF predictions for %s on %s (n_splits=%d, embargo_days=%d, group_gap=%d)...",
            model_name,
            problem_config['name'],
            n_splits,
            embargo_days,
            group_gap,
        )

        oof_preds = np.full(len(X), np.nan, dtype=np.float32)

        params_key = f"{problem_config['name']}_{model_name}"
        base_params = copy.deepcopy(self.best_params.get(params_key, problem_config.get(f'{model_name}_params', {})))

        dummy_X = np.zeros((len(X), 1), dtype=np.int8)

        for fold, (train_idx, val_idx) in enumerate(splitter.split(dummy_X, groups=groups_aligned)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            sw_tr = sample_weight.iloc[train_idx] if sample_weight is not None else None
            sw_val = sample_weight.iloc[val_idx] if sample_weight is not None else None
            
            model = get_model_instance(model_name, problem_config, base_params)
            
            # Quick ES setup for XGBoost
            if model_name == 'xgboost':
                patience = 25
                model = fit_model_es(
                    model_name,
                    model,
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    patience,
                    get_task_type(problem_config),
                    sample_weight=sw_tr,
                    eval_sample_weight=sw_val,
                )
            else:
                # For others, standard fit (assuming they handle or ignore sample_weight)
                if sample_weight is not None:
                    try:
                        model.fit(X_tr, y_tr, sample_weight=sw_tr)
                    except TypeError:
                        model.fit(X_tr, y_tr)
                else:
                    model.fit(X_tr, y_tr)
            
            # Predict
            if is_classification(problem_config):
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(X_val)[:, 1]
                else:
                    p = model.predict(X_val)
            else:
                p = model.predict(X_val)
                
            oof_preds[val_idx] = p
            logger.info(f"  Fold {fold+1}/{n_splits} complete. Val size: {len(val_idx)}")
            
        return oof_preds

    def _generate_oof_predictions_moe(
        self,
        problem_config: dict,
        X: pd.DataFrame,
        y: pd.Series,
        df_raw: pd.DataFrame,
        groups_series: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> np.ndarray:
        """
        Generate leak-safe Out-of-Fold predictions for MoE (Mixture-of-Experts) models.

        This mirrors inference-time MoE routing:
        - Train one model per position_group (RB/WR/TE/QB) on each fold's training slice.
        - Predict on the corresponding fold validation slice.
        - For any rows whose position_group is not trained, use the configured fallback_position.

        Returns an array aligned to X (train+val frame), containing OOF predictions.
        """
        if groups_series is None:
            raise ValueError("groups_series is required for MoE purged OOF generation.")
        if df_raw is None or df_raw.empty:
            raise ValueError("df_raw is required for MoE purged OOF generation.")

        problem_name = problem_config["name"]
        per_pos_cfg = problem_config.get("per_position_training", {}) or {}
        if not per_pos_cfg.get("enabled"):
            raise ValueError(f"Requested MoE OOF for {problem_name}, but per_position_training is not enabled.")

        task_type = str(problem_config.get("task_type", "")).lower()
        is_cls = task_type in {"classification", "binary", "multiclass"}

        allowed_positions = [str(p).upper() for p in (per_pos_cfg.get("positions") or self.moe_position_groups)]
        fallback_pos = str(per_pos_cfg.get("fallback_position", "WR") or "WR").upper()
        min_rows = int(per_pos_cfg.get("min_rows", 50) or 50)

        # Ensure position_group exists/aligned
        df_raw = self._add_position_group(df_raw)
        if "position_group" not in df_raw.columns:
            raise ValueError(f"Could not derive position_group for MoE OOF on {problem_name}.")
        position_groups = df_raw["position_group"].astype(str)

        n_splits = int(self.config["data_split"].get("n_splits", 5))
        embargo_days = int(self.config["data_split"].get("group_gap", self.config["data_split"].get("embargo_td", 0) or 0))
        df_full = self._train_full_frames.get(problem_name)
        group_gap = self._convert_embargo_days_to_group_gap(df_full, groups_series, embargo_days)
        max_train = self.config["data_split"].get("max_train_group_size", np.inf)
        max_test = self.config["data_split"].get("max_test_group_size", np.inf)

        splitter = PurgedGroupTimeSeriesSplit(
            n_splits=n_splits,
            group_gap=group_gap,
            max_train_group_size=max_train if max_train is not None else np.inf,
            max_test_group_size=max_test if max_test is not None else np.inf,
        )
        groups_aligned = groups_series.loc[X.index].to_numpy()
        dummy_X = np.zeros((len(X), 1), dtype=np.int8)

        logger.info(
            "Generating purged MoE OOF predictions for %s (n_splits=%d, embargo_days=%d, group_gap=%d, min_rows=%d, fallback=%s)...",
            problem_name,
            n_splits,
            embargo_days,
            group_gap,
            min_rows,
            fallback_pos,
        )

        oof_preds = np.full(len(X), np.nan, dtype=np.float32)

        def _prep_matrix(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            for col in out.columns:
                if out[col].dtype.name == "category":
                    out[col] = out[col].cat.codes
                elif out[col].dtype == "object":
                    out[col] = pd.to_numeric(out[col], errors="coerce")
            # IMPORTANT: do NOT fillna(0) – XGBoost handles missing values.
            return out

        def _make_xgb_model() -> object:
            # Mirror _train_moe_models parameterization
            xgb_params = self.config.get("xgboost", {}).copy()
            # Light regularization defaults for MoE
            xgb_params["n_estimators"] = 250
            xgb_params["max_depth"] = 5
            xgb_params["learning_rate"] = 0.05
            xgb_params["min_child_weight"] = 5
            xgb_params["subsample"] = 0.8
            xgb_params["colsample_bytree"] = 0.8

            hard_regression = {
                "usage_target_yards",
                "efficiency_rec_yards_air",
                "efficiency_rec_yards_yac",
                "efficiency_rush_yards",
                "usage_carries",
            }
            if (not is_cls) and problem_name in hard_regression:
                xgb_params["n_estimators"] = 300
                xgb_params["max_depth"] = 3
                xgb_params["learning_rate"] = 0.05
                xgb_params["min_child_weight"] = 8
                xgb_params["subsample"] = 0.7
                xgb_params["colsample_bytree"] = 0.7
                xgb_params["reg_alpha"] = xgb_params.get("reg_alpha", 0) + 0.2
                xgb_params["reg_lambda"] = xgb_params.get("reg_lambda", 1) * 2

            if is_cls:
                xgb_params["objective"] = "binary:logistic"
                xgb_params["eval_metric"] = "auc"
                return xgb.XGBClassifier(
                    **{k: v for k, v in xgb_params.items() if k not in ("objective", "eval_metric")},
                    objective=xgb_params.get("objective", "binary:logistic"),
                    eval_metric=xgb_params.get("eval_metric", "auc"),
                    random_state=self.base_seed,
                    n_jobs=-1,
                    use_label_encoder=False,
                )
            xgb_params["objective"] = "reg:squarederror"
            xgb_params.pop("eval_metric", None)
            return xgb.XGBRegressor(
                **{k: v for k, v in xgb_params.items() if k not in ("objective", "eval_metric")},
                objective=xgb_params.get("objective", "reg:squarederror"),
                random_state=self.base_seed,
                n_jobs=-1,
            )

        for fold, (train_idx, val_idx) in enumerate(splitter.split(dummy_X, groups=groups_aligned)):
            # train/val masks in full index space
            train_mask_all = np.zeros(len(X), dtype=bool)
            val_mask_all = np.zeros(len(X), dtype=bool)
            train_mask_all[train_idx] = True
            val_mask_all[val_idx] = True

            # Fit per-position models and predict their own val slices.
            trained_models: dict[str, object] = {}

            for pos in allowed_positions:
                pos_train_mask = train_mask_all & (position_groups == pos).to_numpy()
                pos_val_mask = val_mask_all & (position_groups == pos).to_numpy()
                if pos_val_mask.sum() == 0:
                    continue
                if int(pos_train_mask.sum()) < min_rows:
                    logger.warning(
                        "MoE OOF cannot train %s.%s for fold %d: train_rows=%d < min_rows=%d (val_rows=%d). "
                        "Leaving these OOF predictions as NaN; downstream stacked training will drop them.",
                        problem_name,
                        pos,
                        fold + 1,
                        int(pos_train_mask.sum()),
                        min_rows,
                        int(pos_val_mask.sum()),
                    )
                    continue

                model = _make_xgb_model()
                X_tr = _prep_matrix(X.loc[pos_train_mask])
                y_tr = y.loc[pos_train_mask]
                sw_tr = sample_weight.loc[pos_train_mask] if sample_weight is not None else None
                fit_kwargs = {"sample_weight": sw_tr} if sw_tr is not None else {}
                model.fit(X_tr, y_tr, **fit_kwargs)
                trained_models[pos] = model

                X_val = _prep_matrix(X.loc[pos_val_mask])
                if is_cls and hasattr(model, "predict_proba"):
                    p = model.predict_proba(X_val)[:, 1]
                else:
                    p = model.predict(X_val)
                oof_preds[pos_val_mask] = np.asarray(p, dtype=np.float32)

            # Predict fallback for any val rows not covered by trained positions
            unknown_val_mask = val_mask_all & (~position_groups.isin(allowed_positions)).to_numpy()
            if int(unknown_val_mask.sum()) > 0:
                if fallback_pos not in trained_models:
                    # Train fallback model on its own training slice (fold train only)
                    fb_train_mask = train_mask_all & (position_groups == fallback_pos).to_numpy()
                    if int(fb_train_mask.sum()) < min_rows:
                        logger.warning(
                            "MoE OOF cannot train fallback %s.%s for fold %d: train_rows=%d < min_rows=%d (unknown_val_rows=%d). "
                            "Leaving these OOF predictions as NaN; downstream stacked training will drop them.",
                            problem_name,
                            fallback_pos,
                            fold + 1,
                            int(fb_train_mask.sum()),
                            min_rows,
                            int(unknown_val_mask.sum()),
                        )
                        continue
                    fb_model = _make_xgb_model()
                    X_tr = _prep_matrix(X.loc[fb_train_mask])
                    y_tr = y.loc[fb_train_mask]
                    sw_tr = sample_weight.loc[fb_train_mask] if sample_weight is not None else None
                    fit_kwargs = {"sample_weight": sw_tr} if sw_tr is not None else {}
                    fb_model.fit(X_tr, y_tr, **fit_kwargs)
                    trained_models[fallback_pos] = fb_model

                fb_model = trained_models[fallback_pos]
                X_val = _prep_matrix(X.loc[unknown_val_mask])
                if is_cls and hasattr(fb_model, "predict_proba"):
                    p = fb_model.predict_proba(X_val)[:, 1]
                else:
                    p = fb_model.predict(X_val)
                oof_preds[unknown_val_mask] = np.asarray(p, dtype=np.float32)

            logger.info("  MoE OOF fold %d/%d complete. Val size: %d", fold + 1, n_splits, int(val_mask_all.sum()))

        return oof_preds

    def train_and_evaluate_models(
        self,
        problem_config,
        X_train_full,
        y_train_full,
        X_test,
        y_test,
        df_test_full,
        sample_weight_train_full=None,
    ):
        """Train, evaluate, and store each model for the given problem.

        Parameters
        ----------
        problem_config : dict
        X_train_full, y_train_full : pd.DataFrame / pd.Series
            Training data.
        X_test, y_test : pd.DataFrame / pd.Series
            Feature-restricted test set (post-artifact application) and labels.
        df_test_full : pd.DataFrame
            The *raw* test slice straight from the feature parquet; used to
            enrich the correct / incorrect prediction dumps with every column
            available in the pipeline output.
        sample_weight_train_full : pd.Series or None
            Per-row weights aligned with X_train_full (train+validation data).
        """
        model_names = [m for m in self.config['training']['models_to_train']]
        problem_name = problem_config['name']
        self.current_problem_name = problem_name
        # Sample weighting removed
        
        if sample_weight_train_full is not None:
            sample_weight_train_full = sample_weight_train_full.reindex(X_train_full.index).astype(np.float32, copy=False)

        for name in model_names:
            # All configured models are supported in this pipeline.

            logger.info(f"--- Processing model: {name.upper()} for problem: {problem_name.upper()} ---")

            # Generate OOF predictions for stacking if this is the primary model
            # Must be done BEFORE any 'continue' statements
            if name == 'xgboost':
                groups_full_series = self._groups_index_map.get(problem_name)
                # Always store key frame aligned to X_train_full for merge-based OOF injection.
                df_train_full = self._train_full_frames.get(problem_name)
                if df_train_full is None or df_train_full.empty:
                    raise ValueError(f"Missing train_full frame required for OOF generation ({problem_name}).")
                key_candidates = ["game_id", "player_id", "season", "week", "team", "position"]
                key_cols = [c for c in key_candidates if c in df_train_full.columns]
                if not key_cols:
                    raise ValueError(f"No key columns available for OOF injection ({problem_name}).")
                self._oof_keys_store[problem_name] = df_train_full[key_cols].copy()

                # GLOBAL OOF (required for leak-safe stacking inputs)
                try:
                    oof_preds = self._generate_oof_predictions(
                        name,
                        problem_config,
                        X_train_full,
                        y_train_full,
                        groups_series=groups_full_series,
                        sample_weight=sample_weight_train_full,
                    )
                    self._oof_preds_store[problem_name] = oof_preds
                except Exception as e:
                    if problem_name in self._stack_input_problems:
                        raise
                    logger.warning("Failed to generate GLOBAL OOF predictions for %s: %s", problem_name, e)

                # MoE OOF (required to build a real MoE stacked branch)
                per_pos_cfg = problem_config.get("per_position_training", {}) or {}
                if self.model_architecture in ("moe", "both") and per_pos_cfg.get("enabled"):
                    try:
                        oof_moe = self._generate_oof_predictions_moe(
                            problem_config=problem_config,
                            X=X_train_full,
                            y=y_train_full,
                            df_raw=df_train_full,
                            groups_series=groups_full_series,
                            sample_weight=sample_weight_train_full,
                        )
                        self._oof_preds_store_moe[problem_name] = oof_moe
                    except Exception as e:
                        if problem_name in self._stack_input_problems:
                            raise
                        logger.warning("Failed to generate MoE OOF predictions for %s: %s", problem_name, e)


            # In production mode df_test is empty; skip evaluation to avoid errors.
            if X_test.empty:
                logger.info("Production mode active – training with ES, calibrating on a recent slice, saving threshold.")

                # Prepare params as usual
                params_key = f"{problem_name}_{name}"
                params = copy.deepcopy(self.best_params.get(params_key, problem_config.get(f'{name}_params', {})))

                # Split: oldest 80% train, newest 20% holdout
                holdout_fraction = 0.2
                split_index = int(len(X_train_full) * (1 - holdout_fraction))
                X_train_initial = X_train_full.iloc[:split_index]
                y_train_initial = y_train_full.iloc[:split_index]
                X_holdout_full = X_train_full.iloc[split_index:]
                y_holdout_full = y_train_full.iloc[split_index:]

                sw_train_initial = sw_es = sw_cal = None
                if sample_weight_train_full is not None:
                    sw_train_initial = sample_weight_train_full.iloc[:split_index]
                    sw_holdout = sample_weight_train_full.iloc[split_index:]
                else:
                    sw_holdout = None

                # Further split holdout into ES (older half) and Calibration (newer half)
                es_split = int(len(X_holdout_full) * 0.5)
                X_es = X_holdout_full.iloc[:es_split]
                y_es = y_holdout_full.iloc[:es_split]
                X_cal = X_holdout_full.iloc[es_split:]
                y_cal = y_holdout_full.iloc[es_split:]
                if sw_holdout is not None:
                    sw_es = sw_holdout.iloc[:es_split]
                    sw_cal = sw_holdout.iloc[es_split:]

                # Downcast labels for classification
                if is_classification(problem_config):
                    y_train_initial = y_train_initial.astype('int8')
                    y_es = y_es.astype('int8') if y_es is not None else y_es
                    y_cal = y_cal.astype('int8') if y_cal is not None else y_cal

                # Memory tweak for numerics
                try:
                    numeric_cols = X_train_initial.select_dtypes(include=[np.number]).columns
                    for i, _df in enumerate((X_train_initial, X_es, X_cal)):
                        if _df is not None and not _df.empty:
                            common_cols = [c for c in numeric_cols if c in _df.columns]
                            if common_cols:
                                # Use .loc to avoid SettingWithCopyWarning
                                _df.loc[:, common_cols] = _df[common_cols].astype(np.float32)
                except Exception:
                    pass

                # Set generous n_estimators and compute patience as in evaluation branch
                if name == 'xgboost':
                    params['n_estimators'] = 5000
                    try:
                        params['max_bin'] = min(int(params.get('max_bin', 256)), 128)
                    except Exception:
                        params['max_bin'] = 128
                    patience = compute_patience(int(params.get("n_estimators", 1000)))
                else:
                    # Safe default for ES on non-XGB models
                    patience = 50

                # Instantiate and apply class weighting if classification
                model_instance = get_model_instance(name, problem_config, params)

                # Check if this is a selective model
                is_selective = isinstance(model_instance, tuple) and model_instance[0] == "selective"
                if is_selective:
                    model = model_instance[1]  # The base estimator
                else:
                    model = model_instance

                if is_classification(problem_config) and not is_selective:
                    try:
                        model.set_params(scale_pos_weight=auto_spw(y_train_initial))
                    except Exception:
                        pass

                # Stage 1: Fit with early stopping on ES slice
                logger.info(f"Fitting base {name} with early stopping (ES size={len(X_es)})…")

                # For selective models, we're training the base estimator, so use 'xgboost' as model name
                model_name_for_fit = 'xgboost' if is_selective else name

                model = fit_model_es(
                    model_name_for_fit,
                    model,
                    X_train_initial,
                    y_train_initial,
                    X_es,
                    y_es,
                    patience,
                    get_task_type(problem_config),
                    sample_weight=sw_train_initial,
                    eval_sample_weight=sw_es,
                )

                # Stage 2: Retrain with best iteration for stability
                if not is_selective:
                    best_iter = get_best_iteration(model, name)
                    if best_iter is not None and best_iter >= 0:
                        logger.info(f"Best iteration found: {best_iter}. Retraining final base model…")
                        model = retrain_with_best_iter(
                            name,
                            problem_config,
                            params,
                            best_iter,
                            X_train_initial,
                            y_train_initial,
                            X_es,
                            y_es,
                            get_task_type(problem_config),
                            get_model_instance,
                            sample_weight=sw_train_initial,
                        )
                else:
                    # For selective models, retrain the base model with best iteration
                    best_iter = get_best_iteration(model, 'xgboost')  # Check best iteration on the base model
                    if best_iter is not None and best_iter >= 0:
                        logger.info(f"Best iteration found for selective base model: {best_iter}. Retraining…")
                        # ⚠️ Strip selective-only keys when instantiating the base XGB
                        selective_keys = {'lambda_init','lambda_step','outer_rounds',
                                          'gate_hidden','gate_dropout','gate_epochs',
                                          'gate_bs','gate_lr','gate_weight_decay'}
                        base_params = {k: v for k, v in params.items() if k not in selective_keys}
                        model = retrain_with_best_iter(
                            'xgboost',  # Use xgboost name for the base model
                            problem_config,
                            base_params,
                            best_iter,
                            X_train_initial,
                            y_train_initial,
                            X_es,
                            y_es,
                            get_task_type(problem_config),
                            get_model_instance,
                            sample_weight=sw_train_initial,
                        )

                # Handle selective training if this is a selective model
                if is_selective and is_classification(problem_config):
                    # Check if selective training is enabled for this problem
                    selective_cfg = problem_config.get('selective', self.config['training'].get('selective', {}))
                    if not selective_cfg.get('enabled', False):
                        logger.warning(f"Selective model requested but selective.enabled=False for {problem_name}")
                        # Fall back to regular classification path
                        is_selective = False
                    else:
                        logger.info(f"Training selective model for {problem_name} with coverage target {selective_cfg.get('coverage_target', 0.12)}")

                        # Start from YAML then overlay Optuna-tuned values if present in `params`
                        selective_cfg = copy.deepcopy(problem_config.get('selective', self.config['training'].get('selective', {})))
                        def _apply_if(k):
                            if k in params: selective_cfg[k] = params[k]
                        for k in [
                            'lambda_init','lambda_step','outer_rounds',
                            'gate_hidden','gate_dropout','gate_epochs','gate_bs','gate_lr','gate_weight_decay'
                        ]: _apply_if(k)
                        # Optional: keep decision mode from YAML unless you also tuned it
                        if 'decision' not in selective_cfg:
                            selective_cfg['decision'] = {'mode': 'argmax'}

                        from utils.train.selective.trainer import fit_selective
                        from sklearn.preprocessing import StandardScaler

                        # Create gate preprocessing components (NFL-specific)
                        # Use numeric feature columns to ensure scaler compatibility
                        gate_feature_cols = X_train_initial.select_dtypes(include=[np.number]).columns.tolist()
                        if not gate_feature_cols:
                            raise ValueError("No numeric feature columns available for selective gate training.")
                        gate_scaler = StandardScaler()
                        gate_scaler.fit(X_train_initial[gate_feature_cols].values)  # Fit on filtered columns

                        logger.info(f"Gate will use {len(gate_feature_cols)} numeric features with StandardScaler")

                        # Use the full training data for selective training
                        selective_model, selective_diag = fit_selective(
                            base_estimator=model,
                            # Pass DataFrames so trainer can slice gate_feature_cols before scaling
                            X_train=X_train_initial,
                            y_train=y_train_initial.values,
                            X_es=X_es if X_es is not None else None,
                            y_es=y_es.values if y_es is not None else None,
                            selective_cfg=selective_cfg,
                            gate_scaler=gate_scaler,  # Pass gate preprocessing
                            gate_feature_cols=gate_feature_cols,
                            logger=logger,
                        )
                        model = selective_model  # Replace with selective wrapper

                        # Calibrate decision to the requested coverage on the calibration slice
                        try:
                            # Pass DataFrame so wrapper can slice gate_feature_cols and scale correctly
                            X_cal_df = X_cal
                            cov_target = selective_cfg.get('coverage_target', 0.15)
                            if getattr(model, 'decision_mode', 'gate_first') == 'argmax':
                                T_star, cov = model.calibrate_abstain_temp(X_cal_df, cov_target)
                                logger.info(f"[Selective] Raw calibrated abstain temperature T={T_star:.3f} → coverage≈{cov:.3f}")

                                # 1. Load previous T for EMA smoothing
                                T_prev = None
                                prev_diag_path = None
                                try:
                                    # Look for previous selective diagnostics
                                    problem_dir = self.model_dir / problem_name / "xgb_selective"
                                    if problem_dir.exists():
                                        prev_diags = [d for d in problem_dir.iterdir() if d.is_dir()]
                                        if prev_diags:
                                            latest_prev = max(prev_diags, key=lambda d: d.stat().st_mtime)
                                            prev_diag_path = latest_prev / "selective_diagnostics.json"
                                            if prev_diag_path.exists():
                                                with open(prev_diag_path, "r") as f:
                                                    prev = json.load(f)
                                                    T_prev = float(prev.get("abstain_temperature", None))
                                                    logger.info(f"[EMA] Loaded previous T={T_prev:.3f} from {prev_diag_path}")
                                except Exception as e:
                                    logger.debug(f"[EMA] Could not load previous T: {e}")

                                # 2. Apply EMA smoothing to avoid day-to-day swings
                                alpha = 0.6  # Weight on today's calibration (0.5-0.8)
                                if T_prev is not None:
                                    T_eff = alpha * T_star + (1 - alpha) * T_prev
                                    logger.info(f"[EMA] Smoothed T from {T_star:.3f} → {T_eff:.3f} (prev: {T_prev:.3f}, alpha: {alpha})")
                                else:
                                    T_eff = T_star
                                    logger.info(f"[EMA] No previous T found, using raw T={T_eff:.3f}")

                                # 3. Clamp to safe range (T^2 amplifies abstain penalty)
                                T_eff = float(np.clip(T_eff, 0.5, 6.0))
                                logger.info(f"[CLAMP] Final T={T_eff:.3f} (clamped to [0.5, 6.0])")

                                # 4. Save gate distribution stats for observability
                                z_pos, z_neg, z_abs = model._compute_logits(X_cal_df)
                                T_temp = float(getattr(model, "_abstain_temp", 1.0))
                                z_a_cal = -(z_abs.squeeze()) / (T_temp * T_temp)  # z_abs = -z_a * T^2

                                q_cal = np.percentile(z_a_cal, [5, 25, 50, 75, 95])
                                logger.info(f"[CALIBRATION] gate z_a stats — mean={z_a_cal.mean():.3f} sd={z_a_cal.std():.3f} p5={q_cal[0]:.3f} p50={q_cal[2]:.3f} p95={q_cal[4]:.3f}")
                                logger.info(f"[CALIBRATION] frac z_a>0 (gate wants accept): {(z_a_cal > 0).mean():.3f}")

                                # 5. Save all calibration data for inference and observability
                                selective_diag["abstain_temperature"] = float(T_eff)
                                selective_diag["abstain_temperature_calibrated_raw"] = float(T_star)
                                selective_diag["abstain_temperature_prev"] = T_prev
                                selective_diag["calibrated_coverage"] = float(cov)
                                selective_diag["gate_logit_percentiles_cal"] = {
                                    "p5": float(q_cal[0]), "p25": float(q_cal[1]),
                                    "p50": float(q_cal[2]), "p75": float(q_cal[3]),
                                    "p95": float(q_cal[4])
                                }
                                selective_diag["calibration_frac_z_a_positive"] = float((z_a_cal > 0).mean())

                                # Also set on model for immediate use
                                setattr(model, "abstain_temperature", float(T_eff))

                                logger.info(f"[Selective] Saved smoothed temperature {T_eff:.3f} to diagnostics for inference")

                            else:
                                # gate_first ⇒ calibrate a gate threshold τ on z_a
                                try:
                                    tau, cov = model.calibrate_gate_threshold(X_cal_df, cov_target)
                                    logger.info(f"[Selective] Calibrated gate threshold τ={tau:.3f} → coverage≈{cov:.3f}")

                                    # Save calibrated threshold to diagnostics
                                    selective_diag["gate_threshold"] = float(tau)
                                    selective_diag["calibrated_coverage"] = float(cov)

                                    # Also set on model for immediate use
                                    setattr(model, "gate_threshold", float(tau))

                                except AttributeError:
                                    logger.warning("[Selective] Missing calibrate_gate_threshold; consider switching to argmax or add the method (see patch).")
                        except Exception as e:
                            logger.warning(f"[Selective] Decision calibration skipped due to error: {e}")

                # --- NEW: stash cal-slice preds for composite conformal sums (regression only) ---
                if is_regression(problem_config):
                    try:
                        if X_cal is not None and len(X_cal) >= 100:
                            y_hat_cal_for_store = model.predict(X_cal)
                            
                            # Attach team meta for these rows (if available)
                            # NFL-first meta (MLB columns are not required and should not be assumed).
                            meta_cols = [self.group_col, "home_team", "away_team", "team", "opponent", self.time_col]
                            meta_df = None
                            try:
                                base_df = self._train_full_frames.get(problem_name)
                                if base_df is not None:
                                    keep = [c for c in meta_cols if c in base_df.columns]
                                    if keep:
                                        meta_df = base_df.loc[X_cal.index, keep].copy()
                            except Exception:
                                meta_df = None

                            self._composite_cal_store[problem_name] = {
                                "index": X_cal.index,
                                "y_cal": np.asarray(y_cal),
                                "y_hat_cal": np.asarray(y_hat_cal_for_store),
                                "meta": (meta_df if meta_df is None else {c: meta_df[c].to_numpy() for c in meta_df.columns}),
                            }
                    except Exception as e:
                        logger.warning(f"[composite] Failed to stash cal preds for {problem_name} (prod path): {e}")
                # --- END NEW ---

                # Classification vs Regression production handling
                if is_classification(problem_config):
                    if is_selective:
                        # For selective models in production, no calibration needed
                        final_model = model
                        best_thresh = None
                        best_method = 'selective'
                        best_ece = None
                    else:
                        # Calibration (use groups when available), respecting config toggle
                        cal_method = self.cfg.calibration_method
                        thresh_cfg = self.cfg.threshold_selection
                        groups_series = self._groups_index_map.get(problem_name)
                        groups_cal = None
                        if groups_series is not None:
                            try:
                                groups_cal = groups_series.loc[X_cal.index].values
                            except Exception:
                                groups_cal = None
                        final_model, best_thresh, best_method, best_ece = calibrate_and_select_threshold(
                            model, X_cal, y_cal, self.cfg.calibrate, cal_method, thresh_cfg,
                            groups_cal=groups_cal, problem_config=problem_config
                        )
                    self.models[params_key] = final_model
                    if is_selective:
                        metrics = {
                            "note": "Production run – no evaluation set. Selective model trained with ES.",
                            "calibration_method": best_method,
                            "calibration_ece_oof": None,
                            "calibration_cv_folds": None,
                            "decision_threshold": None,
                            # Add selective diagnostics if available
                        }
                        if 'selective_diag' in locals():
                            metrics.update({
                                'selective_coverage_target': selective_diag.get('coverage_target'),
                                'selective_lambda_final': selective_diag.get('lambda_final'),
                            })
                    else:
                        metrics = {
                            "note": "Production run – no evaluation set. Model trained with ES and calibrated on recent slice.",
                            "calibration_method": best_method,
                            "calibration_ece_oof": round(best_ece, 6) if best_ece is not None else None,
                            "calibration_cv_folds": int(self.config["training"].get("calibration_cv_folds", 3)),
                            "decision_threshold": float(best_thresh) if best_thresh is not None else 0.5,
                        }
                    save_model_and_metrics(self, problem_name, name, metrics)
                    # Feature importance in prod
                    try:
                        save_feature_importance(self, self._unwrap_base_model(final_model), problem_name, name, self.feature_columns)
                        self._log_top_feature_importance(final_model, problem_name, name)
                    except Exception:
                        pass
                    # Save selective diagnostics if available
                    if 'selective_diag' in locals():
                        from utils.train.persist import save_selective_diagnostics
                        save_selective_diagnostics(self, final_model, problem_name, name, selective_diag)
                else:
                    # Regression: skip classification calibrator; optionally do mean calibration / conformal if configured
                    reg_cal_cfg = self.cfg.regression_mean_calibration
                    if reg_cal_cfg.get("enabled", False) and len(X_cal) >= 200:
                        logger.info("Applying isotonic mean calibration for regression (prod)…")
                        model = regression_mean_calibrate(model, X_cal, y_cal)
                    pi_cfg = self.cfg.regression_intervals
                    if pi_cfg.get("enabled", True) and len(X_cal) >= 100:
                        logger.info(f"Fitting split conformal intervals (prod; method={pi_cfg.get('method','naive')}, alpha={pi_cfg.get('alpha',0.1)})…")
                        model = fit_split_conformal_regression(model, X_cal, y_cal, pi_cfg)
                    self.models[params_key] = model
                    metrics = {
                        "note": "Production run – no evaluation set. Model trained with ES on recent slice.",
                        "task_type": "regression",
                    }
                    save_model_and_metrics(self, problem_name, name, metrics)
                    # Feature importance in prod
                    try:
                        save_feature_importance(self, self._unwrap_base_model(model), problem_name, name, self.feature_columns)
                        self._log_top_feature_importance(model, problem_name, name)
                    except Exception:
                        pass
                continue
            
            params_key = f"{problem_name}_{name}"
            # Use deepcopy to avoid modifying the original best_params dict
            params = copy.deepcopy(self.best_params.get(params_key, problem_config.get(f'{name}_params', {})))
            
            # Reserve the *newest* 20 % strictly for hold-out purposes (ES + calibration).
            holdout_fraction = 0.2
            split_index = int(len(X_train_full) * (1 - holdout_fraction))

            # Oldest data → fitting; newest data → hold-out (no leakage back).
            X_train_final = X_train_full.iloc[:split_index]
            y_train_final = y_train_full.iloc[:split_index]

            sw_train_final = sw_holdout_full = sw_es = sw_cal = None
            if sample_weight_train_full is not None:
                sw_train_final = sample_weight_train_full.iloc[:split_index]

            X_holdout_full = X_train_full.iloc[split_index:]
            y_holdout_full = y_train_full.iloc[split_index:]
            if sample_weight_train_full is not None:
                sw_holdout_full = sample_weight_train_full.iloc[split_index:]

            # Sequential split again – older half for ES, newer half for calibration
            es_split = int(len(X_holdout_full) * 0.5)
            X_es = X_holdout_full.iloc[:es_split]
            y_es = y_holdout_full.iloc[:es_split]
            X_cal = X_holdout_full.iloc[es_split:]
            y_cal = y_holdout_full.iloc[es_split:]
            if sw_holdout_full is not None:
                sw_es = sw_holdout_full.iloc[:es_split]
                sw_cal = sw_holdout_full.iloc[es_split:]
            logger.info(f"Sequential holdout created successfully. ES size: {len(X_es)}, Cal size: {len(X_cal)}")

            if is_classification(problem_config) and y_train_final.nunique() < 2:
                logger.warning(f"Training data for {name} on {problem_name} has only one class after splitting. Skipping model.")
                continue

            # Memory optimization: downcast numeric features to float32 and labels to int8 for classification
            try:
                numeric_cols = X_train_final.select_dtypes(include=[np.number]).columns
                for i, _df in enumerate((X_train_final, X_es, X_cal, X_test)):
                    if _df is not None and not _df.empty:
                        common_cols = [c for c in numeric_cols if c in _df.columns]
                        if common_cols:
                            # Use .loc to avoid SettingWithCopyWarning
                            _df.loc[:, common_cols] = _df[common_cols].astype(np.float32)
                if is_classification(problem_config):
                    y_train_final = y_train_final.astype('int8')
                    y_es = y_es.astype('int8') if y_es is not None else y_es
                    y_cal = y_cal.astype('int8') if y_cal is not None else y_cal
            except Exception:
                pass


            if name == 'xgboost':
                params['n_estimators'] = 5000
                # Reduce histogram memory footprint
                try:
                    params['max_bin'] = min(int(params.get('max_bin', 256)), 128)
                except Exception:
                    params['max_bin'] = 128
                patience = compute_patience(int(params.get("n_estimators", 1000)))
            else:
                # Safe default for ES on non-XGB models
                patience = 50

            model = get_model_instance(name, problem_config, params)
            # Apply auto scale_pos_weight for classification tasks prior to fitting
            if is_classification(problem_config):
                try:
                    model.set_params(scale_pos_weight=auto_spw(y_train_final))
                except Exception:
                    pass
            logger.info(f"Training final {name} model with early stopping on a holdout set of size {len(X_es) if X_es is not None else 0}...")
            try:
                gc.collect()
            except Exception:
                pass
            
            # Stage 1: Fit with early stopping to find the best number of trees
            model = fit_model_es(
                name,
                model,
                X_train_final,
                y_train_final,
                X_es,
                y_es,
                patience,
                get_task_type(problem_config),
                sample_weight=sw_train_final,
                eval_sample_weight=sw_es,
            )

            # Stage 2: Re-train a new model with the optimal number of estimators.
            # This ensures the model is not overfit and correctly sized.
            best_iter = get_best_iteration(model, name)
            if best_iter is None or best_iter <= 0:
                logger.warning("Could not find best iteration from early stopping. Using the full model.")
            else:
                logger.info(f"Found best iteration: {best_iter}. Re-training final model with this parameter.")
                model = retrain_with_best_iter(
                    name,
                    problem_config,
                    params,
                    best_iter,
                    X_train_final,
                    y_train_final,
                    X_es,
                    y_es,
                    get_task_type(problem_config),
                    get_model_instance,
                    sample_weight=sw_train_final,
                )

            # --------------------------------------------------
            # Regression branch with optional mean calibration + conformal intervals
            # --------------------------------------------------
            if is_regression(problem_config):
                # 1) Optional mean calibration on the calibration slice
                reg_cal_cfg = self.cfg.regression_mean_calibration
                if reg_cal_cfg.get("enabled", False) and len(X_cal) >= 200:
                    logger.info("Applying isotonic mean calibration for regression…")
                    model = regression_mean_calibrate(model, X_cal, y_cal)
                else:
                    logger.info("Skipping regression mean calibration.")

                # --- NEW: stash cal-slice preds for composite conformal sums ---
                try:
                    if len(X_cal) >= 100:
                        y_hat_cal_for_store = model.predict(X_cal)  # uses mean-calibrated model if enabled
                        
                        # Attach team meta for these rows (if available)
                        # NFL-first meta (MLB columns are not required and should not be assumed).
                        meta_cols = [self.group_col, "home_team", "away_team", "team", "opponent", self.time_col]
                        meta_df = None
                        try:
                            base_df = self._train_full_frames.get(problem_name)
                            if base_df is not None:
                                keep = [c for c in meta_cols if c in base_df.columns]
                                if keep:
                                    meta_df = base_df.loc[X_cal.index, keep].copy()
                        except Exception:
                            meta_df = None

                        self._composite_cal_store[problem_name] = {
                            "index": X_cal.index,
                            "y_cal": np.asarray(y_cal),
                            "y_hat_cal": np.asarray(y_hat_cal_for_store),
                            "meta": (meta_df if meta_df is None else {c: meta_df[c].to_numpy() for c in meta_df.columns}),
                        }
                except Exception as e:
                    logger.warning(f"[composite] Failed to stash cal preds for {problem_name}: {e}")
                # --- END NEW ---

                # 2) Optional split conformal prediction intervals
                pi_cfg = self.cfg.regression_intervals
                if pi_cfg.get("enabled", True) and len(X_cal) >= 100:
                    logger.info(f"Fitting split conformal intervals (method={pi_cfg.get('method','naive')}, alpha={pi_cfg.get('alpha',0.1)})…")
                    model = fit_split_conformal_regression(model, X_cal, y_cal, pi_cfg)
                else:
                    logger.info("Skipping conformal intervals for regression.")
                    # model remains as-is

                # 3) Evaluate on test
                logger.info("Evaluating regression metrics on hold-out test set…")
                y_pred = model.predict(X_test)
                if problem_name == "availability":
                    try:
                        aligned_test = df_test_full.loc[X_test.index] if not df_test_full.empty else df_test_full
                    except Exception:
                        aligned_test = df_test_full
                    y_pred = self._apply_availability_guards(aligned_test, y_pred)
                
                # Stash test-slice predictions for composite diagnostics (non-production mode)
                try:
                    self._composite_test_store[problem_name] = {
                        "index": X_test.index,
                        "y_test": np.asarray(y_test),
                        "y_hat_test": np.asarray(y_pred),
                    }
                except Exception as e:
                    logger.warning(f"[composite] Failed to stash test preds for {problem_name}: {e}")
                
                tail_cfg = problem_config.get(
                    "regression_tail",
                    self.config["training"].get("regression_tail", {"mode": "train_quantile", "value": 0.90, "min_support": 20})
                )
                metrics = compute_regression_metrics(y_test, y_pred, y_train_full, tail_cfg=tail_cfg)

                # If conformal wrapper present, compute coverage/width
                pi_coverage = pi_width_mean = pi_width_med = float('nan')
                try:
                    if hasattr(model, "predict_interval"):
                        lo, hi = model.predict_interval(X_test)
                        hit = ((y_test.values if hasattr(y_test, 'values') else y_test) >= lo) & ((y_test.values if hasattr(y_test, 'values') else y_test) <= hi)
                        pi_coverage = float(np.mean(hit))
                        widths = (hi - lo)
                        pi_width_mean = float(np.mean(widths))
                        pi_width_med = float(np.median(widths))
                except Exception as e:
                    logger.warning(f"Failed PI coverage computation: {e}")

                # Attach PI diagnostics if available
                metrics.update({
                    "pi_enabled": bool(pi_cfg.get("enabled", True)),
                    "pi_method": str(pi_cfg.get("method", "naive")),
                    "pi_alpha": float(pi_cfg.get("alpha", 0.1)),
                    "pi_coverage": pi_coverage,
                    "pi_width_mean": pi_width_mean,
                    "pi_width_median": pi_width_med,
                    "reg_mean_calibration_enabled": bool(reg_cal_cfg.get("enabled", False)),
                })

                self.models[params_key] = model
                save_model_and_metrics(self, problem_name, name, metrics)
                save_feature_importance(self, self._unwrap_base_model(model), problem_name, name, self.feature_columns)
                self._log_top_feature_importance(model, problem_name, name)

                # Save selective diagnostics if available
                if 'selective_diag' in locals():
                    from utils.train.persist import save_selective_diagnostics
                    save_selective_diagnostics(self, model, problem_name, name, selective_diag)

                continue

            # --------------------------------------------------
            # 1) Optional probability calibration on hold-out set (classification only)
            # --------------------------------------------------
            calibration_enabled = self.cfg.calibrate
            calibration_method_cfg = self.cfg.calibration_method

            # Calibration and threshold selection
            thresh_cfg = self.cfg.threshold_selection
            groups_series = self._groups_index_map.get(problem_name)
            groups_cal = None
            if groups_series is not None:
                try:
                    groups_cal = groups_series.loc[X_cal.index].values
                except Exception:
                    groups_cal = None
            final_model, best_thresh, best_method, best_ece = calibrate_and_select_threshold(
                model, X_cal, y_cal, calibration_enabled, calibration_method_cfg, thresh_cfg, groups_cal=groups_cal, problem_config=problem_config
            )
            self.models[params_key] = final_model

            # Keep the best decision threshold around for MoE holdout evaluation parity.
            try:
                if best_thresh is not None and math.isfinite(float(best_thresh)):
                    self._best_thresholds[problem_name] = float(best_thresh)
            except Exception:
                pass

            if is_classification(problem_config):
                final_model = self._maybe_wrap_team_total_adjuster(problem_name, final_model)
            self.models[params_key] = final_model

            logger.info(f"Evaluating {name} on the hold-out test set...")
                
            # Use the canonical, frozen feature list for all predictions. This is guaranteed to be in sync.
            model_features = self.feature_columns
                
            # Branch predictions based on task type
            if is_classification(problem_config):
                # Check if this is a selective model (different prediction path)
                is_selective_model = hasattr(final_model, 'gate_model')  # SelectiveClassifier has gate_model attribute

                if is_selective_model:
                    # Selective model: use argmax decision rule (no thresholds)
                    logger.info("Generating predictions from selective model...")
                    y_pred = final_model.predict(X_test)  # Returns {-1, 0, 1}
                    y_pred_proba = final_model.predict_proba(X_test)  # For metrics only
                    coverage = np.mean(y_pred != -1)  # Abstain rate
                else:
                    # Regular classification: generate probabilities and thresholded predictions
                    y_pred_proba = predict_proba_batched(final_model, X_test, model_features)
                    y_pred = (y_pred_proba >= best_thresh).astype(np.int8) if best_thresh is not None else (y_pred_proba >= 0.5).astype(np.int8)

                # Classification metrics
                if is_selective_model:
                    # For selective models, compute metrics on accepted examples only
                    accepted_mask = (y_pred != -1)
                    y_test_accepted = y_test[accepted_mask]
                    y_pred_accepted = y_pred[accepted_mask]
                    # Wrapper returns [p_pos, p_neg, p_abstain] – use column 0 for positive class
                    y_pred_proba_accepted = (y_pred_proba[accepted_mask]
                                             if y_pred_proba.ndim == 1
                                             else y_pred_proba[accepted_mask, 0])

                    if len(y_test_accepted) > 0:
                        auc = roc_auc_score(y_test_accepted, y_pred_proba_accepted)
                        try:
                            pr_auc = average_precision_score(y_test_accepted, y_pred_proba_accepted)
                        except Exception:
                            pr_auc = float('nan')
                        prec_at_thresh = precision_score(y_test_accepted, y_pred_accepted, zero_division=0)
                        rec_at_thresh = recall_score(y_test_accepted, y_pred_accepted, zero_division=0)
                    else:
                        auc = pr_auc = prec_at_thresh = rec_at_thresh = float('nan')
                else:
                    auc = roc_auc_score(y_test, y_pred_proba)
                    # Also compute PR-AUC for rare events
                    try:
                        pr_auc = average_precision_score(y_test, y_pred_proba)
                    except Exception:
                        pr_auc = float('nan')
                    prec_at_thresh = precision_score(y_test, y_pred, zero_division=0)
                    rec_at_thresh = recall_score(y_test, y_pred, zero_division=0)
                # Pre/post calibration diagnostics where possible
                try:
                    logloss = log_loss(y_test, y_pred_proba)
                except Exception:
                    logloss = float('nan')
                
                report, cm = None, None
                if self.config['training'].get('save_clf_report', False):
                    logger.info("Generating full classification report and confusion matrix...")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    cm = confusion_matrix(y_test, y_pred).tolist()

                confidence_metrics = confidence_bins_report(
                    y_test, y_pred_proba, best_thresh if best_thresh is not None else 0.5
                )

                # Optional: save PR curves from calibration slice (pre- and post-calibration) for auditability
                try:
                    save_pr_curves(self, model, self.models[params_key], X_cal, y_cal, name, problem_name)
                except Exception:
                    pass

                # Add selective-specific metrics if applicable
                if is_selective_model:
                    accepted_mask = (y_pred != -1)
                    n_accepted = int(np.sum(accepted_mask))
                    n_total = len(y_test)

                    metrics = {
                        'auc': auc,
                        'pr_auc': pr_auc,
                        'precision_at_thresh': prec_at_thresh,
                        'recall_at_thresh': rec_at_thresh,
                        'decision_threshold': None,  # No threshold for selective models
                        'brier_score': float('nan'),  # Not applicable for selective
                        'log_loss': float('nan'),  # Not applicable for selective
                        'classification_report': None,  # Will be computed separately for accepted
                        'confusion_matrix': None,  # Will be computed separately for accepted
                        'confidence_analysis': None,  # Not applicable for selective
                        'calibration_method': 'selective',  # Mark as selective
                        'calibration_ece_oof': None,
                        # Selective-specific metrics
                        'selective_coverage': float(coverage),
                        'selective_n_accepted': n_accepted,
                        'selective_n_total': n_total,
                        'selective_acceptance_rate': float(n_accepted / n_total),
                    }

                    # Add selective diagnostics if available
                    if 'selective_diag' in locals():
                        metrics.update({
                            'selective_coverage_target': selective_diag.get('coverage_target'),
                            'selective_lambda_final': selective_diag.get('lambda_final'),
                            'selective_mean_g_per_round': selective_diag.get('mean_g_per_round'),
                            'selective_outer_rounds': selective_diag.get('outer_rounds'),
                        })
                else:
                    metrics = {
                        'auc': auc,
                        'pr_auc': pr_auc,
                        'precision_at_thresh': prec_at_thresh,
                        'recall_at_thresh': rec_at_thresh,
                        'decision_threshold': best_thresh,
                        'brier_score': brier_score_loss(y_test, y_pred_proba),
                        'log_loss': logloss,
                        'classification_report': report,
                        'confusion_matrix': cm,
                        'confidence_analysis': confidence_metrics,
                        'calibration_method': best_method,
                        'calibration_ece_oof': round(best_ece, 6) if best_ece is not None else None,
                    }

                    # Segmented PR-AUC and Brier score diagnostics for anytime TD.
                    if problem_name == "anytime_td":
                        try:
                            eval_df = df_test_full.loc[X_test.index].copy()
                            eval_df["_y_true"] = y_test
                            eval_df["_y_prob"] = y_pred_proba

                            segment_metrics: dict[str, dict] = {}

                            # By horizon
                            if "decision_horizon_hours" in eval_df.columns:
                                by_horizon: dict[str, dict] = {}
                                for h_val, g in eval_df.groupby("decision_horizon_hours"):
                                    g = g.dropna(subset=["_y_true", "_y_prob"])
                                    if g.empty:
                                        continue
                                    y_seg = g["_y_true"].to_numpy()
                                    p_seg = g["_y_prob"].to_numpy()
                                    # Require both classes for stable PR-AUC / Brier
                                    if np.unique(y_seg).size < 2:
                                        continue
                                    try:
                                        pr_seg = average_precision_score(y_seg, p_seg)
                                    except Exception:
                                        pr_seg = float("nan")
                                    try:
                                        brier_seg = brier_score_loss(y_seg, p_seg)
                                    except Exception:
                                        brier_seg = float("nan")
                                    by_horizon[str(h_val)] = {
                                        "pr_auc": float(pr_seg),
                                        "brier_score": float(brier_seg),
                                        "n": int(len(g)),
                                    }
                                if by_horizon:
                                    segment_metrics["by_horizon"] = by_horizon

                            # By position group (fallback to position if needed)
                            pos_col = None
                            for candidate in ("position_group", "position"):
                                if candidate in eval_df.columns:
                                    pos_col = candidate
                                    break
                            if pos_col is not None:
                                by_pos: dict[str, dict] = {}
                                for pos_val, g in eval_df.groupby(pos_col):
                                    g = g.dropna(subset=["_y_true", "_y_prob"])
                                    if g.empty:
                                        continue
                                    y_seg = g["_y_true"].to_numpy()
                                    p_seg = g["_y_prob"].to_numpy()
                                    if np.unique(y_seg).size < 2:
                                        continue
                                    try:
                                        pr_seg = average_precision_score(y_seg, p_seg)
                                    except Exception:
                                        pr_seg = float("nan")
                                    try:
                                        brier_seg = brier_score_loss(y_seg, p_seg)
                                    except Exception:
                                        brier_seg = float("nan")
                                    by_pos[str(pos_val)] = {
                                        "pr_auc": float(pr_seg),
                                        "brier_score": float(brier_seg),
                                        "n": int(len(g)),
                                    }
                                if by_pos:
                                    segment_metrics["by_position"] = by_pos

                            # Cross: position × horizon
                            if (
                                "decision_horizon_hours" in eval_df.columns
                                and pos_col is not None
                            ):
                                by_pos_h: dict[str, dict] = {}
                                grouped = eval_df.groupby([pos_col, "decision_horizon_hours"])
                                for (pos_val, h_val), g in grouped:
                                    g = g.dropna(subset=["_y_true", "_y_prob"])
                                    if g.empty:
                                        continue
                                    y_seg = g["_y_true"].to_numpy()
                                    p_seg = g["_y_prob"].to_numpy()
                                    if np.unique(y_seg).size < 2:
                                        continue
                                    try:
                                        pr_seg = average_precision_score(y_seg, p_seg)
                                    except Exception:
                                        pr_seg = float("nan")
                                    try:
                                        brier_seg = brier_score_loss(y_seg, p_seg)
                                    except Exception:
                                        brier_seg = float("nan")
                                    key = f"{pos_val}|{h_val}"
                                    by_pos_h[key] = {
                                        "pr_auc": float(pr_seg),
                                        "brier_score": float(brier_seg),
                                        "n": int(len(g)),
                                    }
                                if by_pos_h:
                                    segment_metrics["by_position_horizon"] = by_pos_h

                            if segment_metrics:
                                metrics["segment_pr_auc_brier"] = segment_metrics
                        except Exception as seg_exc:
                            logger.warning("Failed to compute segmented PR-AUC/Brier metrics: %s", seg_exc)
            else:
                # Regression: generate direct predictions
                y_pred = final_model.predict(X_test)
                
                # Regression metrics
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                
                logger.info(f"Regression metrics — RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
                
                metrics = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                }

            save_model_and_metrics(self, problem_name, name, metrics)
            save_feature_importance(self, self._unwrap_base_model(final_model), problem_name, name, self.feature_columns)
            self._log_top_feature_importance(final_model, problem_name, name)
            
            # --- Save Prediction Analysis (disabled by default; enable via config) ---
            if self.config['training'].get('save_prediction_analysis', False):
                logger.info(f"Saving prediction analysis for {name} on problem {problem_name}...")

                if not hasattr(self, "_feature_matrix_full_df") or self._feature_matrix_full_df is None:
                    feature_matrix_path = str(self.paths.feature_matrix_path)
                    logger.info(f"Loading full feature matrix from {feature_matrix_path} for prediction analysis merge…")
                    try:
                        try:
                            full_df = pd.read_parquet(feature_matrix_path)
                        except Exception as e_primary:
                            logger.warning(
                                "pyarrow.read_parquet failed for analysis merge (%s: %s). Retrying with fastparquet…",
                                type(e_primary).__name__, e_primary,
                            )
                            full_df = pd.read_parquet(feature_matrix_path, engine="fastparquet")
                        full_df[self.time_col] = pd.to_datetime(full_df[self.time_col])
                        full_df = full_df.sort_values(self.time_col).reset_index(drop=True)
                        self._feature_matrix_full_df = full_df
                        logger.info("Full feature matrix loaded into cache (shape: %s).", full_df.shape)
                    except Exception as e:
                        logger.error(f"Failed to load full feature matrix: {e}")
                        self._feature_matrix_full_df = None

                if self._feature_matrix_full_df is not None:
                    analysis_df = self._feature_matrix_full_df.loc[X_test.index].copy()
                else:
                    logger.warning("Falling back to df_test_full for analysis merge due to earlier loading failure.")
                    analysis_df = df_test_full.loc[X_test.index].copy()

                if is_classification(problem_config):
                    # Classification: use existing logic
                    analysis_df['true_label'] = y_test
                    analysis_df['predicted_label'] = y_pred
                    analysis_df['predicted_probability'] = y_pred_proba
                    analysis_df["is_error"] = (analysis_df["predicted_label"] != analysis_df["true_label"]).astype("int8")

                    correct_df = analysis_df[analysis_df['true_label'] == analysis_df['predicted_label']]
                    incorrect_df = analysis_df[analysis_df['true_label'] != analysis_df['predicted_label']]
                    save_prediction_analysis(self, correct_df, incorrect_df, problem_name, name)

                    slim_cols = [
                        "true_label",
                        "predicted_label",
                        "predicted_probability",
                        "is_error",
                        *self.feature_columns,
                    ]
                else:
                    # Regression: use regression-specific logic
                    analysis_df['true_value'] = y_test
                    analysis_df['predicted_value'] = y_pred
                    analysis_df['absolute_error'] = np.abs(y_test - y_pred)
                    analysis_df['squared_error'] = (y_test - y_pred) ** 2
                    analysis_df['relative_error'] = np.abs(y_test - y_pred) / (np.abs(y_test) + 1e-8)

                    slim_cols = [
                        "true_value",
                        "predicted_value",
                        "absolute_error",
                        "squared_error",
                        "relative_error",
                        *self.feature_columns,
                    ]

                slim_df = analysis_df.reindex(columns=slim_cols, copy=False)
                # Write slim file into versioned analysis folder for consistency
                v_anal_dir = _vdir(self, problem_name, name, "analysis")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                slim_file = v_anal_dir / f"preds_slim_{timestamp}.parquet"
                try:
                    slim_df.to_parquet(slim_file, index=False)
                    logger.info("Saved slim prediction Parquet for deep-dive to %s", slim_file)
                except Exception as e:
                    logger.warning("Failed to write slim Parquet: %s", e)
            else:
                logger.info("Skipping prediction analysis (save_prediction_analysis=false).")



        # --- Save inference artifacts AFTER all models for the problem are trained ---
        # This ensures the state (imputation, categories) is from the pre-tuning/pre-holdout phase.
        save_inference_artifacts(self, problem_name, problem_config)

    def _unwrap_base_model(self, model):
        """Return the true fitted estimator, no matter how it's wrapped."""
        # 0) Our ensemble calibrated wrapper
        try:
            if isinstance(model, EnsembleCalibratedModel):
                return model.base_estimator
        except Exception:
            pass
        # 1) CalibratedClassifierCV (possibly cv='prefit')
        if isinstance(model, CalibratedClassifierCV):
            if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
                inner = model.calibrated_classifiers_[0]
                return getattr(inner, "base_estimator", getattr(inner, "estimator", None))
            # cv='prefit' path
            return getattr(model, "base_estimator", getattr(model, "estimator", None))

        # 2) sklearn Pipeline – return the *final* step (usually the estimator)
        try:
            from sklearn.pipeline import Pipeline as _SkPipeline
            if isinstance(model, _SkPipeline):
                if 'clf' in model.named_steps:
                    return model.named_steps['clf']
                else:
                    return model.steps[-1][1]
        except Exception:
            pass

        # 3) Our custom calibration wrappers
        try:
            if isinstance(model, (IsotonicCalibratedModel, BetaCalibratedModel, HistogramBinningCalibratedModel)):
                return model.base_estimator
        except Exception:
            pass

        try:
            if isinstance(model, (IsotonicMeanCalibratedRegressor, SplitConformalRegressor)):
                return model.base_estimator
        except Exception:
            pass

        # 4) SelectiveClassifier – return the base estimator
        try:
            if hasattr(model, 'base_estimator'):  # SelectiveClassifier has base_estimator attribute
                return model.base_estimator
        except Exception:
            pass

        return model

    def run(self):
        """Execute the full training pipeline for all defined problems."""
        self._clean_output_dirs()
        # Manifest (start)
        if self.cfg.write_run_manifest:
            write_manifest(self, stage="start")
        
        # Lightweight schema read once for column projection downstream
        feature_matrix_path = str(self.paths.feature_matrix_path)
        all_parquet_cols = None
        if pq:
            try:
                all_parquet_cols = pq.ParquetFile(feature_matrix_path).schema.names
            except Exception as e:
                logger.warning(f"Could not read Parquet schema with pyarrow: {e}")
                try:
                    # Fallback to fastparquet for schema reading to match data loading behavior
                    df_fallback = pd.read_parquet(feature_matrix_path, engine="fastparquet")
                    all_parquet_cols = list(df_fallback.columns)
                    logger.info(f"Using fastparquet for schema reading: {len(all_parquet_cols)} columns")
                except Exception as e2:
                    logger.warning(f"Could not read Parquet schema with fastparquet either: {e2}")

        for problem in self.problems:
            problem_name = problem['name']
            logger.info(f"========== Starting Problem: {problem_name.upper()} ==========")

            # (Stage-4) Deterministic per-problem seed
            # Temporarily disable per-problem seeding to match original behavior
            # pseed = deterministic_seed(self.base_seed, problem_name)
            # set_global_seed(pseed)
            # logger.info(f"[seed] Using deterministic seed {pseed} for problem '{problem_name}'.")

            # Use original global seeding for consistency
            set_global_seed(self.base_seed)
            logger.info(f"[seed] Using global seed {self.base_seed} for problem '{problem_name}' (matching original behavior).")
            
            problem_copy = copy.deepcopy(problem)
            self.target_col = problem_copy['target_col']
            self.label_version = problem_copy.get('label_version') or problem_copy.get("labels", {}).get("version")
            derived_cfg = problem_copy.get('derived_target')
            target_load_cols: list[str] = []
            if derived_cfg:
                src_col = derived_cfg.get('source_col')
                if not src_col:
                    raise ValueError(
                        f"Problem '{problem_copy['name']}' missing source_col for derived target."
                    )
                if not derived_cfg.get('external_source'):
                    target_load_cols.append(src_col)
            else:
                target_load_cols.append(self.target_col)
            
            # --- Artifact Caching & "Golden" Feature List Creation ---
            use_cached_for_problem = self.config['training'].get('use_cached_artifacts', True)
            # Horizon-safe artifact caching:
            # Prefer cutoff-suffixed artifacts (prevents mixing horizons in legacy paths).
            artifact_path = self.model_dir / f"inference_artifacts_{problem_name}.joblib"
            try:
                if getattr(self, "cutoff_label", "default") != "default":
                    suffixed = self.model_dir / f"inference_artifacts_{problem_name}_{self.cutoff_label}.joblib"
                    if suffixed.exists():
                        artifact_path = suffixed
            except Exception:
                pass
            
            if artifact_path.exists() and use_cached_for_problem:
                logger.info(f"Loading cached inference artifacts for {problem_name} from {artifact_path}...")
                artifacts = joblib.load(artifact_path)
                
                # --- VALIDATION 1: Structural Integrity ---
                required_keys = {'feature_columns', 'imputation_values', 'category_levels', 'categorical_features'}
                is_structurally_valid = required_keys.issubset(artifacts.keys())

                if not is_structurally_valid:
                    logger.warning(f"STALE CACHE DETECTED for problem '{problem_name}'. "
                                   "Artifact file has an outdated structure. "
                                   "Discarding stale artifacts and regenerating from scratch.")
                    use_cached_for_problem = False
                    artifact_path.unlink(missing_ok=True)
                
                # --- VALIDATION 2: Data Consistency ---
                if is_structurally_valid:
                    cached_features = artifacts.get('feature_columns', [])
                    if all_parquet_cols is not None:
                        missing_features = set(cached_features) - set(all_parquet_cols)
                        if missing_features:
                            logger.warning(
                                f"STALE CACHE DETECTED for problem '{problem_name}'. "
                                f"Cached artifacts require features that are missing from the input data: {sorted(list(missing_features))}. "
                                "Discarding stale artifacts and regenerating from scratch."
                            )
                            # Invalidate cache for this problem run
                            use_cached_for_problem = False
                            # Clean the specific stale artifact file
                            artifact_path.unlink(missing_ok=True)
                
                if use_cached_for_problem:
                    self.feature_columns = artifacts['feature_columns']
                    self.imputation_values[problem_name] = artifacts['imputation_values']
                    self.category_levels[problem_name] = artifacts['category_levels']
                    self.categorical_features[problem_name] = artifacts.get('categorical_features', []) # Use .get for safety
            else:
                 use_cached_for_problem = False # Cache doesn't exist or we're not using it

            # Determine which columns to load for this problem
            if use_cached_for_problem:
                needed_cols = list({self.time_col, self.group_col, *target_load_cols, *self.feature_columns})
            else:
                include_prefixes = tuple(problem_copy.get('feature_prefixes_to_include') or [])
                other_features = problem_copy.get('other_features_to_include') or []
                feature_blacklist = set(problem_copy.get('feature_blacklist') or [])
                if all_parquet_cols is None:
                    logger.warning("Parquet schema unavailable; falling back to loading full dataset for this problem.")
                    needed_cols = None
                else:
                    prefix_features = [col for col in all_parquet_cols if col.startswith(include_prefixes)]
                    # Never auto-include diagnostic MoE columns from the parquet.
                    # These are intended for inspection and must not silently become training features.
                    prefix_features = [c for c in prefix_features if not str(c).endswith("_moe")]
                    prelim_features = sorted(list(set(prefix_features + [c for c in other_features if c in all_parquet_cols])))
                    if feature_blacklist:
                        prelim_features = [c for c in prelim_features if c not in feature_blacklist]
                    # Use group_col (game_id) from config
                    needed_cols = list({self.time_col, self.group_col, *target_load_cols, *prelim_features})
            
            # Always ensure player_id is present for prediction tracking/merging
            if "player_id" not in needed_cols:
                needed_cols.append("player_id")

            # Ensure `position` is available for MoE routing even if it is not a model feature.
            # (Per-position training uses position_group derived from `position`.)
            per_pos_cfg = problem_copy.get("per_position_training") or {}
            if per_pos_cfg.get("enabled") and needed_cols is not None:
                if "position" not in needed_cols and all_parquet_cols is not None and "position" in all_parquet_cols:
                    needed_cols.append("position")

            # Check if this problem requires input predictions from previous problems
            input_preds = problem_copy.get('input_predictions', [])
            if input_preds and needed_cols is not None:
                # Ensure we have join keys to merge predictions
                # We need player_id (and game_id which is group_col)
                if "player_id" not in needed_cols:
                    needed_cols.append('player_id')

            # NOTE: Do not auto-include MLB-only meta columns (away_team_abbr/home_team_abbr/inning_topbot).
            # NFL meta needed for modeling should be explicitly included via config.

            # Load problem-scoped frame with only the required columns
            logger.info(f"Loading problem-scoped columns from Parquet (cols={len(needed_cols) if needed_cols else 'ALL'}) …")
            if needed_cols:
                needed_cols = sorted(list(set(needed_cols)))
                logger.info(f"Unique needed_cols count: {len(needed_cols)}")

            df_problem = load_feature_matrix(
                path=feature_matrix_path,
                time_col=self.time_col,
                columns=needed_cols,
            )
            feature_blacklist = set(problem_copy.get('feature_blacklist') or [])
            if feature_blacklist:
                drop_now = list(feature_blacklist & set(df_problem.columns))
                if drop_now:
                    logger.info("Dropping %d blacklisted features for %s: %s", len(drop_now), problem_name, drop_now)
                    df_problem = df_problem.drop(columns=drop_now, errors="ignore")
            if df_problem.columns.duplicated().any():
                dup_cols = df_problem.columns[df_problem.columns.duplicated()].tolist()
                logger.warning(f"Duplicate columns detected after load: {dup_cols}")
                df_problem = df_problem.loc[:, ~df_problem.columns.duplicated()]

            df_problem = self._apply_problem_level_overrides(df_problem, problem_copy)
            df_problem = require_target_column(
                df_problem,
                self.target_col,
                label_version=self.label_version,
                min_non_null=1,
            )
            
            # Merge input predictions if required
            if input_preds:
                logger.info(f"Merging input predictions for {problem_name}: {input_preds}")
                for pred_name in input_preds:
                    payload = self.problem_predictions.get(pred_name)
                    if not payload:
                        logger.warning(f"Input prediction '{pred_name}' not found in memory! Skipping.")
                        continue
                    pred_df = payload.get("data")
                    pred_keys = payload.get("keys", [])
                    pred_col = f"pred_{pred_name}"
                    if pred_df is None or pred_col not in pred_df.columns:
                        logger.warning(f"Prediction column {pred_col} missing for '{pred_name}'.")
                        continue
                    merge_keys = [col for col in pred_keys if col in df_problem.columns]
                    if not merge_keys:
                        logger.warning(f"No overlapping merge keys for '{pred_name}'. Available keys: {pred_keys}")
                        continue

                    extra_pred_cols = [c for c in pred_df.columns if c.startswith(pred_col)]
                    logger.info(
                        f"Merging {pred_name} (shape {pred_df.shape}, keys={merge_keys}, cols={extra_pred_cols}) into current frame (shape {df_problem.shape})"
                    )
                    df_problem = df_problem.merge(
                        pred_df[merge_keys + extra_pred_cols],
                        on=merge_keys,
                        how='left',
                    )

                df_problem = self._inject_composed_features(df_problem)

                # STRICT STACKING: upstream OOF predictions are not defined for the earliest
                # time blocks (no prior data to train on without peeking). Those rows must be
                # excluded from downstream stacked training rather than imputed.
                required_pred_cols: list[str] = []
                for pred_name in input_preds:
                    base = f"pred_{pred_name}"
                    if base in df_problem.columns:
                        required_pred_cols.append(base)
                    moe_col = f"{base}_moe"
                    if moe_col in df_problem.columns:
                        required_pred_cols.append(moe_col)
                required_pred_cols = list(dict.fromkeys(required_pred_cols))
                if required_pred_cols:
                    before = len(df_problem)
                    df_problem = df_problem.dropna(subset=required_pred_cols).reset_index(drop=True)
                    dropped = before - len(df_problem)
                    if dropped > 0:
                        logger.info(
                            "Dropped %d rows for %s due to missing OOF upstream preds (cols=%s).",
                            dropped,
                            problem_name,
                            required_pred_cols,
                        )

            odds_horizon = problem_copy.get("odds_horizon")
            if odds_horizon:
                df_problem = self._filter_odds_snapshot_columns(df_problem, odds_horizon)

            # Leak guard on the problem-scoped frame
            allow_prefixes = tuple(problem_copy.get('feature_prefixes_to_include') or [])
            allow_exact = set(problem_copy.get('other_features_to_include') or [])
            allow_exact.update(input_preds or [])
            if self.target_col:
                allow_exact.add(self.target_col)
            df_problem, leak_info = enforce_leak_guard(
                df_problem,
                policy=DEFAULT_LEAK_POLICY,
                allow_prefixes=allow_prefixes,
                allow_exact=allow_exact,
                drop_banned=True,
                drop_non_allowlisted=False,
                raise_on_banned=True,
            )
            if leak_info.dropped:
                logger.info(
                    "Dropped %d leak-prone columns for %s: %s",
                    len(leak_info.dropped),
                    problem_name,
                    sorted(leak_info.dropped),
                )

            self._normalize_datetime_like_columns(df_problem, problem_name)
            df_problem[self.time_col] = pd.to_datetime(df_problem[self.time_col])
            sort_cols = [self.time_col]
            if self.group_col and self.group_col in df_problem.columns:
                sort_cols.append(self.group_col)
            df_problem = df_problem.sort_values(sort_cols).reset_index(drop=True)

            if self.target_col and self.target_col in df_problem.columns:
                before_rows = len(df_problem)
                df_problem = df_problem[df_problem[self.target_col].notna()].reset_index(drop=True)
                removed = before_rows - len(df_problem)
                if removed > 0:
                    logger.info(
                        "Dropped %d rows with null target for problem %s.",
                        removed,
                        problem_name,
                    )

            # Split and fit/apply artifacts
            df_train, df_val, df_test = self.split_data_chronologically(df_problem)
            if is_classification(problem_copy):
                try:
                    self._initialize_team_total_config(problem_name, df_train)
                except Exception as exc:
                    logger.warning("Failed to initialise team total scaling for %s: %s", problem_name, exc)
            if not use_cached_for_problem:
                self._fit_feature_artifacts(df_train, problem_copy)
                save_inference_artifacts(self, problem_name, problem_copy)

            # Log final feature list used
            logger.info(f"--- Using {len(self.feature_columns)} features for {problem_name} ---")
            for f in sorted(self.feature_columns):
                logger.info(f"  - {f}")

            logger.info(f"Applying feature artifacts to create datasets ({len(self.feature_columns)} features)...")
            X_train, y_train = self._apply_feature_artifacts(df_train, problem_name)
            X_val, y_val = self._apply_feature_artifacts(df_val, problem_name)
            X_test, y_test = self._apply_feature_artifacts(df_test, problem_name)

            sample_weight_train = self._compute_sample_weights(df_train, problem_copy)
            sample_weight_val = self._compute_sample_weights(df_val, problem_copy) if not df_val.empty else pd.Series([], dtype=np.float32)
            
            group_labels_train = self._build_group_labels(df_train)
            groups_train = group_labels_train.to_numpy()

            # --- RE-ENABLE HYPERPARAMETER TUNING ---
            if self.cfg.hyperparameter_tuning.get('run_tuning', True):
                for model_name in [m for m in self.cfg.models_to_train]:
                    logger.info(f"--- Tuning hyperparameters for {model_name.upper()} on {problem_name.upper()} ---")
                    self.tune_hyperparameters(
                        model_name,
                        problem_copy,
                        X_train,
                        y_train,
                        groups_train,
                        sample_weight=sample_weight_train,
                        time_values=df_train[self.time_col] if self.time_col in df_train.columns else None,
                    )
            else:
                logger.info("Skipping hyperparameter tuning as per config `run_tuning: false`.")
            # -----------------------------------------
            
            # Combine train and validation for final model training
            # Robust concat – if validation slice is empty (production mode) skip concat to
            # avoid dtype upcasting and future pandas warnings.
            if df_val.empty:
                df_train_full = df_train.copy()
            else:
                df_train_full = (
                    pd.concat([df_train, df_val], ignore_index=True)
                    .sort_values(
                        [self.time_col, self.group_col]
                        if (self.group_col and self.group_col in df_train.columns)
                        else [self.time_col]
                    )
                    .reset_index(drop=True)
                )
            # Ensure contiguity of group labels for purged CV / OOF generation.
            if self.group_col and self.group_col in df_train_full.columns:
                df_train_full = df_train_full.sort_values([self.time_col, self.group_col]).reset_index(drop=True)
            X_train_full, y_train_full = self._apply_feature_artifacts(df_train_full, problem_name)
            sample_weight_train_full = self._compute_sample_weights(df_train_full, problem_copy)

            self._sample_weights[problem_name] = {
                "train": sample_weight_train,
                "val": sample_weight_val,
                "train_full": sample_weight_train_full,
            }

            # 👇 NEW: keep the raw train+val frame for this problem (for meta lookup)
            self._train_full_frames[problem_name] = df_train_full

            # Build groups map aligned to df_train_full for downstream calibration CV
            try:
                group_labels_full = self._build_group_labels(df_train_full)
                self._groups_index_map[problem_name] = group_labels_full
            except Exception:
                self._groups_index_map[problem_name] = None
            
            # ==========================================================================
            # MODEL ARCHITECTURE: Train Global and/or MoE models based on config
            # ==========================================================================
            use_moe = problem_copy.get("per_position_training", False)
            train_global = self.model_architecture in ("global", "both")
            train_moe = self.model_architecture in ("moe", "both") and use_moe
            
            # Train global model (traditional single model)
            if train_global:
                logger.info(f"[Global] Training global model for {problem_name}")
            self.train_and_evaluate_models(
                problem_copy,
                X_train_full,
                y_train_full,
                X_test,
                y_test,
                df_test,
                sample_weight_train_full,
            )
            
            # Train MoE models (per-position models)
            if train_moe:
                logger.info(f"[MoE] Training per-position models for {problem_name}")
                # For stacked problems, train MoE models on MoE upstream predictions.
                # We do this by selecting `pred_<upstream>_moe` columns (OOF-injected) as inputs
                # when available, instead of the GLOBAL `pred_<upstream>` columns.
                moe_feature_cols_override: list[str] | None = None
                X_train_full_moe = X_train_full
                X_test_moe = X_test

                input_preds = problem_copy.get("input_predictions") or []
                if input_preds:
                    moe_feature_cols_override = []
                    missing: list[str] = []
                    for base in list(self.feature_columns or []):
                        if base.startswith("pred_"):
                            # map `pred_snaps` -> `pred_snaps_moe` if snaps is an upstream input
                            base_name = base.replace("pred_", "", 1)
                            if base_name in input_preds:
                                cand = f"pred_{base_name}_moe"
                                if cand in df_train_full.columns and cand in df_test.columns:
                                    moe_feature_cols_override.append(cand)
                                    continue
                                missing.append(cand)
                        moe_feature_cols_override.append(base)

                    if missing:
                        raise ValueError(
                            f"MoE structured training requested for {problem_name}, but missing MoE upstream inputs: {sorted(set(missing))}"
                        )

                    X_train_full_moe = df_train_full.reindex(columns=moe_feature_cols_override)
                    X_test_moe = df_test.reindex(columns=moe_feature_cols_override)

                self._train_moe_models(
                    problem_copy,
                    X_train_full_moe,
                    y_train_full,
                    X_test_moe,
                    y_test,
                    df_train_full,
                    df_test,
                    sample_weight_train_full,
                    feature_columns_override=moe_feature_cols_override,
                )

                # --- NEW: evaluate MoE models on the same hold-out slice (classification only) ---
                if is_classification(problem_copy) and problem_name in self.moe_models and not X_test_moe.empty:
                    try:
                        moe_probs = self._predict_with_moe(X_test_moe, df_test, problem_name)
                        y_true = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)

                        thr = float(self._best_thresholds.get(problem_name, 0.5))
                        moe_pred = (moe_probs >= thr).astype(np.int8)

                        # Some small test windows can have a single class; handle gracefully.
                        if np.unique(y_true).size >= 2:
                            moe_auc = float(roc_auc_score(y_true, moe_probs))
                            try:
                                moe_pr_auc = float(average_precision_score(y_true, moe_probs))
                            except Exception:
                                moe_pr_auc = float("nan")
                            try:
                                moe_brier = float(brier_score_loss(y_true, moe_probs))
                            except Exception:
                                moe_brier = float("nan")
                            try:
                                moe_logloss = float(log_loss(y_true, moe_probs))
                            except Exception:
                                moe_logloss = float("nan")
                        else:
                            moe_auc = moe_pr_auc = moe_brier = moe_logloss = float("nan")

                        moe_metrics = {
                            "auc": moe_auc,
                            "pr_auc": moe_pr_auc,
                            "brier_score": moe_brier,
                            "log_loss": moe_logloss,
                            "precision_at_thresh": float(precision_score(y_true, moe_pred, zero_division=0)),
                            "recall_at_thresh": float(recall_score(y_true, moe_pred, zero_division=0)),
                            "decision_threshold": thr,
                            "n_test": int(len(y_true)),
                            "note": "MoE hold-out evaluation on same slice as GLOBAL; threshold uses GLOBAL best_thresh when available.",
                        }

                        out_dir = _vdir(self, problem_name, "xgboost", "metrics")
                        moe_path = out_dir / "moe_metrics.yaml"
                        with moe_path.open("w") as fh:
                            yaml.safe_dump(moe_metrics, fh, sort_keys=False)
                        logger.info("Saved MoE evaluation report for %s to %s", problem_name, moe_path)
                    except Exception as e:
                        logger.warning("Failed to evaluate MoE holdout metrics for %s: %s", problem_name, e)
                
                # If both trained, run comparison
                if train_global and problem_name in self.moe_models:
                    task_type = problem_copy.get("task_type", "classification")
                    model_key = f"{problem_name}_xgboost"
                    if model_key in self.models:
                        global_model = self.models[model_key]
                        if hasattr(global_model, "predict_proba"):
                            global_preds = global_model.predict_proba(X_test)
                            if global_preds.ndim > 1:
                                global_preds = global_preds[:, 1]
                        else:
                            global_preds = global_model.predict(X_test)
                        
                        # IMPORTANT: If MoE models were trained with a MoE-specific feature schema
                        # (e.g., using `pred_<upstream>_moe` inputs), we must evaluate MoE using
                        # the matching X_test_moe matrix. Otherwise XGBoost will throw a
                        # feature_names mismatch and the comparison is invalid.
                        X_test_for_moe = X_test_moe if "X_test_moe" in locals() else X_test

                        comparison = self._compare_global_vs_moe(
                            problem_name, X_test_for_moe, y_test, df_test, global_preds, task_type
                        )
                        self.moe_comparison_results[problem_name]["comparison"] = comparison
        
        # --- NEW: build composite & team conformal sum artifacts after all problems trained ---
            # Generate and store predictions for downstream models.
            #
            # CRITICAL (stacking integrity):
            # We only generate leak-safe OOF predictions for the GLOBAL model. Therefore the
            # canonical stacked inputs `pred_<problem>` must be sourced from the GLOBAL model
            # (with OOF injection), even if MoE models exist. If MoE predictions are needed for
            # diagnostics, we persist them separately as `pred_<problem>_moe`.
            primary_model_name = 'xgboost'
            model_key = f"{problem_name}_{primary_model_name}"
            
            # Decide whether to use MoE or global model for predictions
            use_moe_for_preds = (
                self.model_architecture in ("moe", "both") 
                and problem_name in self.moe_models
                and len(self.moe_models.get(problem_name, {})) > 0
            )
            has_global_model = model_key in self.models
            
            if use_moe_for_preds or has_global_model:
                logger.info(
                    "Generating full predictions for %s using GLOBAL for stacking%s",
                    problem_name,
                    " (also computing MoE diagnostic preds)" if use_moe_for_preds else "",
                )
                
                # We need to apply artifacts to the FULL df_problem to get X_full
                # Note: df_problem was modified in place (merged with inputs), so it has all cols.
                # But we need to transform it to X via _apply_feature_artifacts
                
                # CAUTION: _apply_feature_artifacts might rely on columns being present. 
                # We assume df_problem still has them.
                try:
                    logger.info(f"Generating predictions for {problem_name}. df_problem shape: {df_problem.shape}")
                    X_full, _ = self._apply_feature_artifacts(df_problem, problem_name)
                    logger.info(f"X_full shape after artifacts: {X_full.shape}")
                    
                    preds = None
                    moe_preds = None

                    # Always compute GLOBAL preds for canonical pred_<problem> when available.
                    if has_global_model:
                        model = self.models[model_key]
                        base_model = self._unwrap_base_model(model)
                        target_model = base_model if base_model is not None else model

                        # If it's a pipeline/calibrated classifier, it usually has predict_proba
                        if hasattr(target_model, "predict_proba"):
                            # Passing self.feature_columns to ensure correct column alignment/selection in batching
                            preds = predict_proba_batched(target_model, X_full, self.feature_columns, batch_size=10000)
                            if preds.ndim > 1:
                                preds = preds[:, 1]
                        elif hasattr(target_model, "predict"):
                            preds = target_model.predict(X_full)

                    # Optionally compute MoE preds for diagnostics (NOT used for stacking inputs).
                    if use_moe_for_preds:
                        # IMPORTANT: Some MoE models are trained with a MoE-specific feature schema
                        # (e.g., consuming `pred_<upstream>_moe` inputs). When that's the case, we
                        # must generate MoE predictions using the matching feature matrix.
                        if "moe_feature_cols_override" in locals() and moe_feature_cols_override is not None:
                            X_full_moe = df_problem.reindex(columns=moe_feature_cols_override)
                            moe_preds = self._predict_with_moe(X_full_moe, df_problem, problem_name)
                        else:
                            moe_preds = self._predict_with_moe(X_full, df_problem, problem_name)
                        logger.info(f"[MoE] Generated {len(moe_preds)} predictions for {problem_name} (diagnostic only)")
                    
                    if preds is not None:
                        # STACKING FIX: overwrite training rows with OOF predictions using keys (not index length).
                        # NOTE: GLOBAL OOF is generated for the GLOBAL model only.
                        if has_global_model and problem_name in self._oof_preds_store and problem_name in self._oof_keys_store:
                            try:
                                oof_arr = self._oof_preds_store[problem_name]
                                oof_keys = self._oof_keys_store[problem_name]
                                if len(oof_arr) != len(oof_keys):
                                    raise ValueError("OOF keys/preds length mismatch.")
                                pred_col = f"pred_{problem_name}"
                                oof_df = oof_keys.copy()
                                oof_df["__oof_row__"] = 1
                                oof_df[pred_col] = oof_arr

                                # merge into df_problem key space, overwrite where present
                                key_cols = list(oof_keys.columns)
                                df_keys = df_problem[key_cols].copy()
                                merged = df_keys.merge(oof_df, on=key_cols, how="left")
                                mask = (merged["__oof_row__"] == 1).to_numpy()
                                n_mask = int(mask.sum())
                                if n_mask > 0:
                                    logger.info(
                                        "Injecting GLOBAL OOF predictions for %s (%d/%d rows) via keys to prevent stacking leakage.",
                                        problem_name, n_mask, len(preds)
                                    )
                                    preds[mask] = merged.loc[mask, pred_col].to_numpy(dtype=float)
                            except Exception as e:
                                logger.warning("Failed GLOBAL OOF injection via keys for %s: %s", problem_name, e)

                        raw_preds = preds.copy()
                        if problem_name == "availability":
                            preds = self._apply_availability_guards(df_problem, preds)
                        # Apply post-processing cap for snaps predictions
                        # Cap low-usage players (max_snap_pct < 0.25) to ceiling * 1.5
                        if problem_name == "snaps":
                            preds = self._apply_snaps_ceiling_cap(df_problem, preds)
                        # Apply position-based zeroing/capping for usage_targets
                        # Zero QBs (99.2% have 0 targets), cap FBs at 0.10
                        if problem_name == "usage_targets":
                            preds = self._apply_usage_targets_position_cap(df_problem, preds)
                        # Apply position-based capping for usage_carries
                        # Cap WRs at 0.05, TEs at 0.02, QBs at 0.30, FBs at 0.10
                        if problem_name == "usage_carries":
                            preds = self._apply_usage_carries_position_cap(df_problem, preds)
                        # Apply position-based capping for usage_target_yards
                        # Cap RBs/FBs/QBs at their realistic target depth (short targets)
                        if problem_name == "usage_target_yards":
                            preds = self._apply_usage_target_yards_position_cap(df_problem, preds)
                        # Apply position-based capping for efficiency_rec_yards_air
                        # RBs get checkdowns (often negative air yards), cap at 0
                        if problem_name == "efficiency_rec_yards_air":
                            preds = self._apply_efficiency_rec_yards_air_cap(df_problem, preds)

                        # Apply the same post-processing to MoE diagnostic preds (if present),
                        # and inject MoE OOF via keys (when available).
                        if moe_preds is not None:
                            # MoE OOF injection (aligned by keys)
                            if problem_name in self._oof_preds_store_moe and problem_name in self._oof_keys_store:
                                try:
                                    oof_arr = self._oof_preds_store_moe[problem_name]
                                    oof_keys = self._oof_keys_store[problem_name]
                                    if len(oof_arr) != len(oof_keys):
                                        raise ValueError("MoE OOF keys/preds length mismatch.")
                                    pred_col_moe = f"pred_{problem_name}_moe"
                                    oof_df = oof_keys.copy()
                                    oof_df["__oof_row__"] = 1
                                    oof_df[pred_col_moe] = oof_arr
                                    key_cols = list(oof_keys.columns)
                                    df_keys = df_problem[key_cols].copy()
                                    merged = df_keys.merge(oof_df, on=key_cols, how="left")
                                    mask = (merged["__oof_row__"] == 1).to_numpy()
                                    n_mask = int(mask.sum())
                                    if n_mask > 0:
                                        logger.info(
                                            "Injecting MoE OOF predictions for %s (%d/%d rows) via keys.",
                                            problem_name, n_mask, len(moe_preds)
                                        )
                                        moe_preds[mask] = merged.loc[mask, pred_col_moe].to_numpy(dtype=float)
                                except Exception as e:
                                    logger.warning("Failed MoE OOF injection via keys for %s: %s", problem_name, e)

                            if problem_name == "availability":
                                moe_preds = self._apply_availability_guards(df_problem, moe_preds)
                            if problem_name == "snaps":
                                moe_preds = self._apply_snaps_ceiling_cap(df_problem, moe_preds)
                            if problem_name == "usage_targets":
                                moe_preds = self._apply_usage_targets_position_cap(df_problem, moe_preds)
                            if problem_name == "usage_carries":
                                moe_preds = self._apply_usage_carries_position_cap(df_problem, moe_preds)
                            if problem_name == "usage_target_yards":
                                moe_preds = self._apply_usage_target_yards_position_cap(df_problem, moe_preds)
                            if problem_name == "efficiency_rec_yards_air":
                                moe_preds = self._apply_efficiency_rec_yards_air_cap(df_problem, moe_preds)
                        # Store
                        pred_col = f"pred_{problem_name}"
                        # Ensure we have the keys
                        key_candidates = ['game_id', 'player_id', 'season', 'week', 'team', 'position']
                        key_cols = [col for col in key_candidates if col in df_problem.columns]
                        if not key_cols:
                            logger.warning(f"Skipping storage for {problem_name} predictions – no shared key columns available.")
                        else:
                            out_df = df_problem[key_cols].copy()
                            out_df[pred_col] = preds
                            if problem_name == "availability":
                                out_df[f"{pred_col}_raw"] = raw_preds
                            if moe_preds is not None:
                                out_df[f"{pred_col}_moe"] = moe_preds
                            before = len(out_df)
                            out_df = out_df.drop_duplicates(subset=key_cols, keep='last')
                            if len(out_df) != before:
                                logger.info(
                                    "Dropped %d duplicate prediction rows for %s based on keys %s",
                                    before - len(out_df),
                                    problem_name,
                                    key_cols,
                                )
                            self.problem_predictions[problem_name] = {
                                "data": out_df,
                                "keys": key_cols,
                            }
                            logger.info(f"Stored {len(out_df)} predictions for {problem_name} with keys {key_cols}")

                            if problem_name == "snaps":
                                self._persist_snaps_metrics(df_problem, preds)
                            if problem_name in {"usage_targets", "usage_carries"}:
                                self._persist_usage_expected_metrics(problem_name, df_problem, preds)
                            if problem_name in {
                                "efficiency_rec_success",
                                "efficiency_rush_success",
                                "efficiency_rec_yards",
                                "efficiency_rush_yards",
                            }:
                                self._persist_efficiency_metrics(problem_name, df_problem, preds)
                            if problem_name in {"td_conv_rec", "td_conv_rush"}:
                                self._persist_td_conversion_metrics(problem_name, df_problem, preds)
                            if problem_name == "anytime_td_structured":
                                self._persist_anytime_td_structured_metrics(df_problem, preds)
                    else:
                        logger.warning(f"Could not generate predictions for {problem_name}: Model has no predict method")
                except Exception as e:
                    logger.error(f"Failed to generate downstream predictions for {problem_name}: {e}")

        fit_composite_conformal_sums(self)
        fit_team_conformal_sums(self)
        # --- END NEW ---

        # ==========================================================================
        # Print MoE Comparison Summary (if "both" mode was used)
        # ==========================================================================
        if self.model_architecture == "both" and self.moe_comparison_results:
            logger.info("\n" + "="*80)
            logger.info("MoE vs GLOBAL MODEL COMPARISON SUMMARY")
            logger.info("="*80)
            
            for problem_name, results in self.moe_comparison_results.items():
                comparison = results.get("comparison", {})
                if not comparison:
                    continue
                
                logger.info(f"\n{problem_name}:")
                
                # Overall metrics
                global_metrics = comparison.get("global", {})
                moe_metrics = comparison.get("moe", {})
                
                if "mae" in global_metrics:
                    g_mae = global_metrics.get("mae", 0)
                    m_mae = moe_metrics.get("mae", 0)
                    g_r2 = global_metrics.get("r2", 0)
                    m_r2 = moe_metrics.get("r2", 0)
                    delta_mae = m_mae - g_mae
                    delta_r2 = m_r2 - g_r2
                    winner = "MoE ✓" if delta_mae < 0 else "Global"
                    logger.info(f"  Overall: Global MAE={g_mae:.4f}, MoE MAE={m_mae:.4f}, Δ={delta_mae:+.4f} ({winner})")
                    logger.info(f"           Global R²={g_r2:.4f}, MoE R²={m_r2:.4f}, Δ={delta_r2:+.4f}")
                elif "auc" in global_metrics:
                    g_auc = global_metrics.get("auc", 0.5)
                    m_auc = moe_metrics.get("auc", 0.5)
                    delta_auc = m_auc - g_auc
                    winner = "MoE ✓" if delta_auc > 0 else "Global"
                    logger.info(f"  Overall: Global AUC={g_auc:.4f}, MoE AUC={m_auc:.4f}, Δ={delta_auc:+.4f} ({winner})")
                
                # By position
                by_position = comparison.get("by_position", {})
                if by_position:
                    logger.info("  By Position:")
                    for pos, metrics in by_position.items():
                        if "global_mae" in metrics:
                            g_mae = metrics["global_mae"]
                            m_mae = metrics["moe_mae"]
                            delta = m_mae - g_mae
                            better = "✓" if delta < 0 else ""
                            logger.info(f"    {pos}: n={metrics['n']}, Global MAE={g_mae:.4f}, MoE MAE={m_mae:.4f}, Δ={delta:+.4f} {better}")
            
            logger.info("\n" + "="*80)
            
            # Save comparison to file
            try:
                import json
                comparison_path = self.metric_dir / "moe_comparison.json"
                with open(comparison_path, "w") as f:
                    json.dump(self.moe_comparison_results, f, indent=2, default=str)
                logger.info(f"Saved MoE comparison results to {comparison_path}")
            except Exception as e:
                logger.warning(f"Failed to save MoE comparison: {e}")

        # Manifest (end)
        if self.cfg.write_run_manifest:
            try:
                # Provide a small roll-up: which models produced artifacts for each problem.
                results = {}
                for p in self.problems:
                    pname = p["name"]
                    results[pname] = {}
                    for m in self.cfg.models_to_train:
                        k = f"{pname}_{m}"
                        results[pname][m] = bool(self.models.get(k) is not None)
                    # Add MoE model info
                    if pname in self.moe_models:
                        results[pname]["moe_positions"] = list(self.moe_models[pname].keys())
                write_manifest(self, stage="end", extra=results)
            except Exception:
                pass

        logger.info("========== ENTIRE TRAINING PIPELINE COMPLETED SUCCESSFULLY ==========")

def tune_features(problem_name: str, n_trials: int):
    """
    Uses Optuna to find the optimal half-life for recency-weighted stats by
    re-running the feature pipeline and a lightweight training for each trial.
    """
    logger.info(f"--- Starting Optuna search for optimal half-life for problem: {problem_name} ---")

    # A temporary trainer instance to access config and utility methods
    base_trainer = ModelTrainer()
    problem_config = next((p for p in base_trainer.config['problems'] if p['name'] == problem_name), None)
    if not problem_config:
        raise ValueError(f"Problem '{problem_name}' not found in training config.")
        
    original_feature_config = {}
    try:
        # Load the main config to find feature params
        from utils.general.config import load_config
        main_config = load_config()
        original_feature_config['half_life'] = main_config.get('feature_params', {}).get('recency_stats', {}).get('half_life')
    except Exception as e:
        logger.warning(f"Could not back up original half-life from config: {e}")

    # Get base_seed from the trainer instance
    base_seed = base_trainer.base_seed

    def objective(trial: optuna.Trial):
        # Suggest an integer half-life in days.
        half_life = trial.suggest_int('half_life', 250, 250)
        # Suggest a float for the shrink_k parameter.
        shrink_k = trial.suggest_float('shrink_k', 45, 45)
        # Suggest an integer for the crude stats rolling window in days.
        crude_window = trial.suggest_int('crude_window', 1, 10)
        logger.info(f"--- NEW TRIAL #{trial.number}: Testing half_life={half_life}, shrink_k={shrink_k:.2f}, crude_window={crude_window} ---")
        
        # 1. Re-run feature engineering with the new half-life
        logger.info(f"Step 1/2: Running feature engineering pipeline for trial #{trial.number}...")
        try:
            # Use sys.executable to ensure we're using the python from the correct venv
            subprocess.run(
                [
                    sys.executable, "pipeline/feature.py", 
                    "--half-life", str(half_life),
                    "--shrink-k", str(shrink_k),
                    "--crude-window", str(crude_window)
                ],
                check=True, capture_output=True, text=True, timeout=600  # 10-minute timeout
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Feature engineering pipeline failed for trial #{trial.number}.")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise optuna.TrialPruned() # Prune trial if features fail
        except subprocess.TimeoutExpired as e:
            logger.error(f"Feature engineering pipeline timed out for trial #{trial.number}.")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise optuna.TrialPruned()

        # 2. Run a lightweight training and evaluation cycle using the updated logic
        logger.info(f"Step 2/2: Running lightweight training & validation for trial #{trial.number}...")
        try:
            # Use a fresh ModelTrainer instance for each trial to ensure no state leaks
            trainer = ModelTrainer()
            trainer.target_col = problem_config['target_col']
            
            # Load the newly generated feature data
            df = trainer.load_data()
            df_train, df_val, _ = trainer.split_data_chronologically(df)

            # --- This is the corrected, modern feature processing logic ---
            trainer._fit_feature_artifacts(df_train, problem_config)
            X_train, y_train = trainer._apply_feature_artifacts(df_train, problem_name)
            X_val, y_val = trainer._apply_feature_artifacts(df_val, problem_name)
            # --- End of corrected logic ---

            # Use a simple, fast, single-threaded model for tuning features
            xgb_params = {
                'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_estimators': 250, 
                'learning_rate': 0.05, 'max_depth': 6, 'n_jobs': 1, 'verbosity': 0,
                'tree_method': 'hist', 'enable_categorical': True
            }
            model = xgb.XGBClassifier(**xgb_params)
            
            model.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      early_stopping_rounds=25, verbose=False)

            preds = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, preds)
            
            logger.info(f"--- TRIAL #{trial.number} COMPLETE: half_life={half_life}, shrink_k={shrink_k:.2f}, crude_window={crude_window}, Validation AUC={auc_score:.4f} ---")
            return auc_score
        except Exception as e:
            logger.error(f"Training/validation failed for trial #{trial.number} with params: half_life={half_life}, shrink_k={shrink_k:.2f}, crude_window={crude_window}: {e}", exc_info=True)
            return 0.0 # Return a poor score to penalize failing trials

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=base_seed))
    # We run trials serially (n_jobs=1) because each trial modifies a shared resource (the feature file)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    logger.info("--- Optuna feature tuning complete ---")
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best validation AUC: {study.best_value:.4f}")
    
    if original_feature_config.get('half_life') is not None:
        logger.info(f"NOTE: The optimal parameters were found to be: {study.best_params}.")
        logger.info("To make this permanent, update the `half_life`, `shrink_k`, and `crude_window_days` parameters in `config/config.yaml`")
        logger.info(f"and then re-run the main feature pipeline (`python pipeline/feature.py`) before full model training.")
        logger.info(f"The original value was: {original_feature_config['half_life']}")



def train(config_path='config/training.yaml', ordinal_only=False, ordinal_ev=None, **overrides):
    """Main function to run the training pipeline."""
    cutoff_hours = overrides.pop("decision_cutoff_hours", None)
    fallback_override = fallback_cutoff_hours() if cutoff_hours is not None else None

    with decision_cutoff_override(cutoff_hours=cutoff_hours, fallback_hours=fallback_override):
        trainer = ModelTrainer(config_path, overrides=overrides)

        ran_training = False
        if not ordinal_only:
            trainer.run()          # your normal training loop
            ran_training = True

        # Run ordinal EV by default after training, unless explicitly disabled
        if ordinal_ev is None:  # Default behavior - run if training was successful
            should_run_ordinal = ran_training and not ordinal_only  # Don't run if only doing ordinal
        else:
            should_run_ordinal = ordinal_ev

        if should_run_ordinal:
            from utils.train.ordinal_ev_integration import run_ordinal_ev_job
            try:
                out_path = run_ordinal_ev_job(trainer)
                logger.info(f"Ordinal EV job complete → {out_path}")
            except Exception as e:
                logger.exception(f"Ordinal EV job failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Pipeline")
    parser.add_argument('--tune-features', action='store_true', help="Run Optuna to find the best feature engineering parameters.")
    parser.add_argument('--problem', type=str, default='anytime_td', help="The problem to optimize for during tuning.")
    parser.add_argument('--n-trials', type=int, default=25, help="Number of Optuna trials to run.")
    # Stage-4 CLI polish
    parser.add_argument('--config', type=str, default='config/training.yaml', help="Path to training config.")
    parser.add_argument('--problems', type=str, default='', help="Comma-separated subset of problems to run.")
    parser.add_argument('--models', type=str, default='', help="Comma-separated subset of models to train.")
    parser.add_argument('--prod', action='store_true', help="Override to production mode for this run.")
    parser.add_argument('--no-tuning', action='store_true', help="Disable hyperparameter tuning for this run.")
    parser.add_argument('--run-tag', type=str, default='', help="Optional tag to include in run_id and manifest.")
    parser.add_argument('--versioning', type=str, default='', choices=['', 'run_id', 'legacy'], help="Override artifact versioning mode.")
    parser.add_argument('--ordinal-ev', action='store_true',
                        help="Force ordinal EV job to run (default behavior after training).")
    parser.add_argument('--no-ordinal-ev', action='store_true',
                        help="Disable ordinal EV job (overrides default behavior).")
    parser.add_argument('--ordinal-only', action='store_true',
                        help="Skip training entirely; just run the ordinal EV job using the latest saved models.")
    parser.add_argument('--decision-cutoff-hours', type=float, default=None,
                        help="Override decision cutoff horizon (hours before kickoff) for feature selection.")
    args = parser.parse_args()

    try:
        if args.tune_features:
            tune_features(problem_name=args.problem, n_trials=args.n_trials)
        else:
            overrides = {
                "problems": [p.strip() for p in args.problems.split(",") if p.strip()] if args.problems else None,
                "models_to_train": [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None,
                "production_mode": True if args.prod else None,
                "run_tuning": False if args.no_tuning else None,
                "run_tag": args.run_tag,
                "versioning_mode": (args.versioning or None),
                "decision_cutoff_hours": args.decision_cutoff_hours,
            }
            # strip None values
            overrides = {k: v for k, v in overrides.items() if v is not None}
            # Handle ordinal EV flags: --ordinal-ev forces on, --no-ordinal-ev forces off, None = default
            if hasattr(args, 'no_ordinal_ev') and args.no_ordinal_ev:
                ordinal_ev_flag = False  # Explicitly disabled
            elif hasattr(args, 'ordinal_ev') and args.ordinal_ev:
                ordinal_ev_flag = True   # Explicitly enabled
            else:
                ordinal_ev_flag = None   # Default behavior (run after training)

            train(config_path=args.config, ordinal_only=args.ordinal_only, ordinal_ev=ordinal_ev_flag, **overrides)
    except Exception as e:
        logger.critical(f"The pipeline has failed with a critical error: {e}", exc_info=True)
        sys.exit(1)
