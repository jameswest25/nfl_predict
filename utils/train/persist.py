# utils/train/persist.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import yaml
import joblib
import json
import logging
import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)

def _vdir(trainer, problem_name: str, model_name: str|None, kind: str) -> Path:
    """
    kind in {'models','metrics','analysis','artifacts'}
    Returns a versioned directory when trainer.run_id exists; otherwise legacy dirs.
    """
    base = {
        "models": trainer.paths.model_dir,
        "metrics": trainer.paths.metric_dir,
        "analysis": trainer.paths.analysis_dir,
        "artifacts": trainer.paths.model_dir,  # keep artifacts near model_dir
    }[kind]
    if getattr(trainer, "versioning_mode", "run_id") == "run_id" and getattr(trainer, "run_id", None):
        parts = [base, problem_name]
        if model_name:
            parts.append(model_name)
        parts.append(trainer.run_id)
        out = Path(*parts)
    else:
        out = base
    out.mkdir(parents=True, exist_ok=True)
    return out

def format_metrics_for_yaml(data: dict) -> dict:
    """Moved from _format_metrics_for_yaml."""
    formatted = {}
    for k, v in data.items():
        import numpy as _np
        if isinstance(v, (int, float, _np.floating, _np.integer)):
            formatted[k] = round(v, 4)
        else:
            formatted[k] = v
    return formatted

def save_inference_artifacts(trainer, problem_name: str, problem_config: dict | None = None) -> None:
    # Versioned artifact copy
    v_dir = _vdir(trainer, problem_name, None, "artifacts") / "inference"
    v_dir.mkdir(parents=True, exist_ok=True)
    v_path = v_dir / f"inference_artifacts_{problem_name}.joblib"
    # Legacy artifact path (kept for BC, but MUST be horizon-safe).
    cutoff_label = getattr(trainer, "cutoff_label", "default")
    legacy = (
        trainer.paths.model_dir / f"inference_artifacts_{problem_name}_{cutoff_label}.joblib"
        if cutoff_label != "default"
        else trainer.paths.model_dir / f"inference_artifacts_{problem_name}.joblib"
    )
    task_type = None
    output_mode = None
    if problem_config:
        task_type = problem_config.get("task_type")
        output_mode = problem_config.get("output_mode")
    if output_mode is None and task_type:
        if str(task_type).lower() in ("classification", "binary", "multiclass"):
            output_mode = "probability"
        else:
            output_mode = "value"

    payload = {
        "feature_columns": trainer.feature_columns,
        "imputation_values": trainer.imputation_values.get(problem_name, {}),
        "category_levels": trainer.category_levels.get(problem_name, {}),
        "categorical_features": trainer.categorical_features.get(problem_name, []),
        "task_type": task_type,
        "output_mode": output_mode,
    }
    joblib.dump(payload, v_path)
    # keep legacy in sync
    try:
        joblib.dump(payload, legacy)
    except Exception:
        pass

def save_model_and_metrics(trainer, problem_name: str, model_name: str, metrics: dict) -> None:
    # Versioned paths
    v_models = _vdir(trainer, problem_name, model_name, "models")
    v_metrics = _vdir(trainer, problem_name, model_name, "metrics")
    model_path_v = v_models / f"model.joblib"
    metrics_path_v = v_metrics / f"metrics.yaml"

    # Legacy paths (kept for BC, but MUST be horizon-safe to avoid mixing cutoffs).
    cutoff_label = getattr(trainer, "cutoff_label", "default")
    legacy_model = (
        trainer.paths.model_dir / f"{problem_name}_{model_name}_{cutoff_label}.joblib"
        if cutoff_label != "default"
        else trainer.paths.model_dir / f"{problem_name}_{model_name}.joblib"
    )
    legacy_metrics = (
        trainer.paths.metric_dir / f"{problem_name}_{model_name}_{cutoff_label}.yaml"
        if cutoff_label != "default"
        else trainer.paths.metric_dir / f"{problem_name}_{model_name}.yaml"
    )

    try:
        formatted_metrics = format_metrics_for_yaml(metrics)
        with open(metrics_path_v, 'w') as f:
            yaml.dump(formatted_metrics, f, indent=4, sort_keys=False)
        logger.info(f"Saved evaluation report for {problem_name} to {metrics_path_v}")
    except Exception as e:
        logger.error(f"Failed to save metrics report to {metrics_path_v}: {e}")
    
    model_saved = False
    try:
        joblib.dump(trainer.models[f"{problem_name}_{model_name}"], model_path_v, compress=3)
        model_saved = True
    except OSError as e:
        logger.warning(f"joblib.dump failed with standard compression (error: {e}). Retrying with gzip...")
        try:
            joblib.dump(trainer.models[f"{problem_name}_{model_name}"], model_path_v, compress=('gzip', 3))
            model_saved = True
            logger.info("Successfully saved model with fallback joblib protocol.")
        except Exception as e2:
            logger.error(f"Fallback joblib.dump also failed: {e2}")
    
    if not model_saved:
        logger.error(f"CRITICAL: Could not save model for {problem_name} and {model_name}. Downstream tasks will fail.")
    
    # Also write/refresh legacy copies
    try:
        joblib.dump(trainer.models[f"{problem_name}_{model_name}"], legacy_model)
        with open(legacy_metrics, 'w') as f:
            yaml.dump(formatted_metrics, f, indent=4, sort_keys=False)
    except Exception:
        pass

def save_selective_diagnostics(trainer, model, problem_name: str, model_name: str, selective_diag: dict) -> None:
    """Save selective model training diagnostics and gate model state."""
    try:
        v_dir = _vdir(trainer, problem_name, model_name, "models")

        # Add gate preprocessing info to diagnostics before saving
        if hasattr(model, 'gate_scaler') and model.gate_scaler is not None:
            # Save gate scaler
            scaler_path = v_dir / "gate_scaler.joblib"
            joblib.dump(model.gate_scaler, scaler_path)
            selective_diag["gate_scaler_path"] = str(scaler_path)
            logger.info(f"Saved gate scaler for {problem_name} to {scaler_path}")

        if hasattr(model, 'gate_feature_cols') and model.gate_feature_cols is not None:
            selective_diag["gate_feature_cols"] = model.gate_feature_cols
            logger.info(f"Saved gate feature columns ({len(model.gate_feature_cols)} features) for {problem_name}")

        # Save class indices for predict_proba column mapping
        selective_diag["pos_index"] = getattr(model, "pos_index", 0)  # predict_proba: [p_pos, p_neg, p_abstain]
        selective_diag["abstain_index"] = getattr(model, "abstain_index", 2)
        selective_diag["class_order"] = ["pos", "neg", "abstain"]  # For reference

        # Save decision mode
        selective_diag["decision_mode"] = getattr(model, "decision_mode", "argmax")
        logger.info(f"Saved class indices and decision mode for {problem_name}")

        # Save selective configuration for reference
        if hasattr(model, '_selective_cfg'):
            selective_diag["selective_config"] = model._selective_cfg

        # Save selective diagnostics as JSON
        diag_path = v_dir / "selective_diagnostics.json"
        with open(diag_path, 'w') as f:
            json.dump(selective_diag, f, indent=2, default=str)
        logger.info(f"Saved selective diagnostics for {problem_name} to {diag_path}")

        # Save gate model state if available
        if hasattr(model, 'gate_model'):
            import torch
            gate_path = v_dir / "gate_model.pt"
            torch.save(model.gate_model.state_dict(), gate_path)
            logger.info(f"Saved gate model state for {problem_name} to {gate_path}")

            # Also save gate architecture metadata
            arch_path = v_dir / "gate_architecture.json"
            gate_model = model.gate_model
            import torch.nn as nn
            first_linear = next((m for m in gate_model.modules() if isinstance(m, nn.Linear)), None)
            hidden_units = [m.out_features for m in gate_model.modules()
                            if isinstance(m, nn.Linear) and m.out_features != 1]  # exclude final 1
            arch_info = {
                "in_dim": getattr(first_linear, "in_features", None),
                "hidden_units": hidden_units,
                "device": str(getattr(model, 'device', 'cpu')),
            }
            with open(arch_path, 'w') as f:
                json.dump(arch_info, f, indent=2)
            logger.info(f"Saved gate architecture for {problem_name} to {arch_path}")

        # Copy inference artifacts to the selective model directory so prediction can find them
        cutoff_label = getattr(trainer, "cutoff_label", "default")
        artifacts_src = (
            trainer.paths.model_dir / f"inference_artifacts_{problem_name}_{cutoff_label}.joblib"
            if cutoff_label != "default"
            else trainer.paths.model_dir / f"inference_artifacts_{problem_name}.joblib"
        )
        artifacts_dst = v_dir / f"inference_artifacts_{problem_name}.joblib"
        if artifacts_src.exists():
            import shutil
            shutil.copy2(artifacts_src, artifacts_dst)
            logger.info(f"Copied inference artifacts for {problem_name} to {artifacts_dst}")

    except Exception as e:
        logger.error(f"Failed to save selective diagnostics for {problem_name}: {e}")

def save_feature_importance(trainer, final_model, problem_name: str, model_name: str, feature_columns: list[str]) -> None:
    base_model = trainer._unwrap_base_model(final_model)
    if base_model is None:
        logger.warning("Could not unwrap a base estimator from %s for problem %s – skipping importance export.",
                       model_name, problem_name)
        return

    def _save(df, type_name: str):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = trainer.metric_dir / f"feature_importance_{model_name}_{problem_name}_{type_name}_{timestamp}.csv"
        try:
            df.to_csv(out, index=False)
            logger.info(f"Saved {type_name} feature importance report to {out}")
        except Exception as e:
            logger.error(f"Failed to save {type_name} feature importance report: {e}")
    
    # Also save to versioned directory
    try:
        v_dir = _vdir(trainer, problem_name, model_name, "metrics")
        v_path = v_dir / "feature_importance.json"
        if hasattr(base_model, "get_booster"):
            booster = base_model.get_booster()
            scores = booster.get_score(importance_type="gain")
        elif hasattr(base_model, "feature_importances_"):
            scores = {f: float(w) for f, w in zip(feature_columns, getattr(base_model, "feature_importances_", []))}
        else:
            scores = {}
        payload = {"features": feature_columns, "importance": scores}
        with open(v_path, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass

    if isinstance(base_model, xgb.XGBModel):
        booster = base_model.get_booster()
        for imp_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
            try:
                scores = booster.get_score(importance_type=imp_type)
                if scores:
                    import pandas as pd
                    df = (pd.DataFrame(scores.items(), columns=['feature', 'importance'])
                            .sort_values(by='importance', ascending=False).reset_index(drop=True))
                    _save(df, f"xgb_{imp_type}")
            except Exception as e:
                logger.warning(f"Could not calculate XGBoost importance type '{imp_type}': {e}")
        return

    import pandas as pd
    handled = False
    if hasattr(base_model, 'feature_importances_'):
        imp = base_model.feature_importances_
        if len(imp) == len(feature_columns):
            df = pd.DataFrame({'feature': feature_columns, 'importance': imp}).sort_values(
                by='importance', ascending=False).reset_index(drop=True)
            _save(df, 'default'); handled = True
    elif hasattr(base_model, 'coef_'):
        coef = base_model.coef_.ravel()
        if len(coef) == len(feature_columns):
            df = pd.DataFrame({'feature': feature_columns, 'importance': coef}).sort_values(
                'importance', key=np.abs, ascending=False).reset_index(drop=True)
            _save(df, 'coef'); handled = True
    if not handled:
        logger.warning("Base model type %s has no standard importance API; skipping.", type(base_model).__name__)

def save_prediction_analysis(trainer, correct_df, incorrect_df, problem_name: str, model_name: str) -> None:
    out_dir = _vdir(trainer, problem_name, model_name, "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    correct_path = out_dir / f"correct_{ts}.parquet"
    wrong_path = out_dir / f"incorrect_{ts}.parquet"
    try:
        correct_df.to_parquet(correct_path, index=False)
        incorrect_df.to_parquet(wrong_path, index=False)
        logger.info(f"Saved prediction analysis files to {out_dir}")
    except Exception as e:
        logger.error(f"Failed to save prediction analysis files: {e}")
    # consolidated error_frame
    try:
        import pandas as pd
        ef = (pd.concat([correct_df.assign(is_error=0), incorrect_df.assign(is_error=1)], axis=0)
                .reset_index(drop=True))
        ef["confusion"] = (
            ef["true_label"].astype(str) + ef["predicted_label"].astype(str)
        ).map({"00": "TN", "11": "TP", "01": "FN", "10": "FP"}).astype("category")
        error_path = out_dir / f"error_frame_{ts}.parquet"
        ef.to_parquet(error_path, index=False)
        logger.info("Saved consolidated error frame to %s", error_path)
    except Exception as e:
        logger.error("Failed to save consolidated error frame: %s", e)

def save_pr_curves(trainer, model_before, model_after, X_cal, y_cal, model_name: str, problem_name: str) -> None:
    """Generate & save pre/post-calibration PR curves on the calibration slice."""
    import pandas as pd
    from sklearn.metrics import precision_recall_curve
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    pr_dir = Path(trainer.metric_dir)
    pr_dir.mkdir(parents=True, exist_ok=True)

    try:
        p1, r1, t1 = precision_recall_curve(y_cal, model_before.predict_proba(X_cal)[:, 1])
        df1 = pd.DataFrame({'precision': p1[:-1], 'recall': r1[:-1], 'threshold': t1})
        df1.to_csv(pr_dir / f"pr_curve_precal_{model_name}_{problem_name}_{ts}.csv", index=False)
    except Exception:
        pass

    try:
        p2, r2, t2 = precision_recall_curve(y_cal, model_after.predict_proba(X_cal)[:, 1])
        df2 = pd.DataFrame({'precision': p2[:-1], 'recall': r2[:-1], 'threshold': t2})
        df2.to_csv(pr_dir / f"pr_curve_postcal_{model_name}_{problem_name}_{ts}.csv", index=False)
    except Exception:
        pass

def clean_output_dirs(trainer) -> None:
    """Delete old artifacts based on age while protecting the newest model per problem.

    Policy:
    - Only delete files older than 7 days (by filename timestamp: _YYYYMMDD_HHMMSS).
    - In the models directory, NEVER delete the most recent .pkl per problem, even if older than 7 days.
    - Metrics and analysis files are removed if older than 7 days.
    """
    import re
    from datetime import datetime, timedelta

    logger.info("Cleaning output directories – removing files older than 7 days and protecting newest model per problem…")

    analysis_dir = trainer.paths.metric_dir.parent / 'prediction_analysis'
    dirs_to_clean = {
        trainer.paths.model_dir: ['*.pkl'],
        trainer.paths.metric_dir: ['*.yaml', '*.csv'],
        analysis_dir: ['*.csv']
    }

    # Regex to extract timestamp from filenames like ..._20250623_005355.ext
    timestamp_regex = re.compile(r'_(\d{8}_\d{6})\.')
    cutoff_dt = datetime.now() - timedelta(days=7)

    total_cleaned_count = 0

    # Precompute newest model per problem (by timestamp) so we can always keep it
    newest_model_for_problem = {}
    # Precompute newest evaluation report (YAML) per (model, problem) so we keep at least one metrics file
    newest_metric_for_pair = {}
    # Precompute newest feature-importance CSV per (model, problem, type)
    newest_importance_for_triplet = {}
    try:
        # Build mapping only from model_dir .pkl files that contain timestamp
        model_files = []
        for pattern in dirs_to_clean.get(trainer.paths.model_dir, []):
            model_files.extend(trainer.paths.model_dir.glob(pattern))
        for f in model_files:
            m = timestamp_regex.search(f.name)
            if not m:
                continue
            ts_str = m.group(1)
            try:
                ts_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            # Infer problem name from known problems list
            prefix = f.name.split(f"_{ts_str}")[0]  # strip _YYYYMMDD_HHMMSS suffix
            problem_name_detected = None
            for p in trainer.problems:
                pname = p['name']
                if prefix.endswith(f"_{pname}") or prefix == pname:
                    problem_name_detected = pname
                    break
            if problem_name_detected is None:
                # fallback: skip protection if we cannot infer the problem name
                continue
            prev = newest_model_for_problem.get(problem_name_detected)
            if prev is None or ts_dt > prev[1]:
                newest_model_for_problem[problem_name_detected] = (f, ts_dt)

        # Build mapping from metric_dir .yaml files
        metric_files = []
        for pattern in dirs_to_clean.get(trainer.paths.metric_dir, []):
            metric_files.extend(trainer.paths.metric_dir.glob(pattern))
        for f in metric_files:
            m = timestamp_regex.search(f.name)
            if not m:
                continue
            ts_str = m.group(1)
            try:
                ts_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            # Infer model and problem from filename like backtest_model_problem_20250623_005355.yaml
            parts = f.name.split(f"_{ts_str}")[0].split('_')
            if len(parts) >= 2:
                model_name = parts[-2]
                problem_name = parts[-1]
                key = (model_name, problem_name)
                prev = newest_metric_for_pair.get(key)
                if prev is None or ts_dt > prev[1]:
                    newest_metric_for_pair[key] = (f, ts_dt)

        # Build mapping from metric_dir feature importance CSVs
        importance_files = []
        for pattern in dirs_to_clean.get(trainer.paths.metric_dir, []):
            importance_files.extend(trainer.paths.metric_dir.glob(pattern))
        for f in importance_files:
            if 'feature_importance' not in f.name:
                continue
            m = timestamp_regex.search(f.name)
            if not m:
                continue
            ts_str = m.group(1)
            try:
                ts_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            # Infer model, problem, and type from filename
            parts = f.name.split(f"_{ts_str}")[0].split('_')
            if len(parts) >= 3:
                model_name = parts[-3]
                problem_name = parts[-2]
                importance_type = parts[-1]
                key = (model_name, problem_name, importance_type)
                prev = newest_importance_for_triplet.get(key)
                if prev is None or ts_dt > prev[1]:
                    newest_importance_for_triplet[key] = (f, ts_dt)

    except Exception as e:
        logger.warning(f"Failed to pre-compute newest models per problem: {e}")

    # Now clean each directory
    for dir_path, patterns in dirs_to_clean.items():
        dir_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        for pattern in patterns:
            for f in dir_path.glob(pattern):
                if not f.is_file():
                    continue

                # Extract timestamp
                m = timestamp_regex.search(f.name)
                if not m:
                    continue

                ts_str = m.group(1)
                try:
                    ts_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                except ValueError:
                    continue

                # Check if file is newer than cutoff
                if ts_dt >= cutoff_dt:
                    continue

                # Protection logic: never delete the newest model per problem
                should_protect = False
                if dir_path == trainer.paths.model_dir and f.suffix == '.pkl':
                    # Check if this is the newest model for any problem
                    for problem_name, (newest_f, newest_ts) in newest_model_for_problem.items():
                        if f == newest_f:
                            should_protect = True
                            break
                elif dir_path == trainer.paths.metric_dir and f.suffix == '.yaml':
                    # Check if this is the newest metric for any (model, problem) pair
                    for (model_name, problem_name), (newest_f, newest_ts) in newest_metric_for_pair.items():
                        if f == newest_f:
                            should_protect = True
                            break
                elif dir_path == trainer.paths.metric_dir and f.suffix == '.csv' and 'feature_importance' in f.name:
                    # Check if this is the newest feature importance for any (model, problem, type) triplet
                    for (model_name, problem_name, importance_type), (newest_f, newest_ts) in newest_importance_for_triplet.items():
                        if f == newest_f:
                            should_protect = True
                            break

                if should_protect:
                    logger.info(f"Protecting newest file: {f.name}")
                    continue

                # Delete the file
                try:
                    f.unlink()
                    total_cleaned_count += 1
                    logger.info(f"Removed old file: {f.name}")
                except Exception as e:
                    logger.error(f"Error removing file {f}: {e}")

    # Also clean inference artifacts if not using cached artifacts
    if not trainer.config.get('training', {}).get('use_cached_artifacts', True):
        logger.info("`use_cached_artifacts` is false, cleaning old inference artifacts as well...")
        artifact_pattern = "inference_artifacts_*.joblib"
        for f in trainer.paths.model_dir.glob(artifact_pattern):
            if f.is_file():
                try:
                    f.unlink()
                    total_cleaned_count += 1
                    logger.info(f"Removed old inference artifact: {f.name}")
                except Exception as e:
                    logger.error(f"Error removing inference artifact {f}: {e}")

    logger.info(f"Finished cleaning. Removed a total of {total_cleaned_count} old file(s).")
