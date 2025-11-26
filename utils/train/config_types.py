# utils/train/config_types.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

@dataclass(frozen=True)
class Paths:
    feature_matrix_path: Path
    model_dir: Path
    metric_dir: Path
    analysis_dir: Path  # metric_dir.parent / 'prediction_analysis'

@dataclass(frozen=True)
class TrainingCfg:
    # high-use flags/sections with safe defaults
    production_mode: bool
    data_split: Dict[str, Any]
    models_to_train: List[str]
    hyperparameter_tuning: Dict[str, Any]
    calibrate: bool
    calibration_method: str
    threshold_selection: Dict[str, Any]
    regression_mean_calibration: Dict[str, Any]
    regression_intervals: Dict[str, Any]
    mu_tail_5plus: float
    ordinal_ev: Dict[str, Any]  # ordinal expected value configuration
    selective: Dict[str, Any]  # selective model configuration
    # Stage-4 additions (all optional; set by code defaults if not present in YAML)
    base_seed: int = 42
    write_run_manifest: bool = True
    versioning_mode: str = "run_id"  # ['run_id', 'legacy']
    run_tag: str = ""
