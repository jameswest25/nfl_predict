from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from utils.general.paths import PROJ_ROOT

logger = logging.getLogger(__name__)


DATA_DIR = PROJ_ROOT / "data" / "processed"
MODEL_DIR = PROJ_ROOT / "output" / "models" / "injury_availability"
METRIC_DIR = PROJ_ROOT / "output" / "metrics" / "injury_availability"

LABEL_COL = "injury_is_inactive_designation"
PROB_HEURISTIC_COL = "injury_inactive_probability"
MODEL_PROB_COL = "injury_inactive_probability_model"
PROB_LOW_COL = "injury_inactive_probability_p10"
PROB_HIGH_COL = "injury_inactive_probability_p90"

DEFAULT_ALPHA = 0.1  # 90% interval


@dataclass
class InjuryModelArtifact:
    model: CalibratedClassifierCV
    feature_columns: list[str]
    categorical_features: list[str]
    datetime_features: list[str]
    category_levels: dict[str, list[str]]
    conformal_margin: float
    alpha: float
    metrics: dict[str, float]
    train_seasons: list[int]
    calibration_seasons: list[int]
    created_at: str


def _load_feature_schema(feature_matrix_path: Path) -> Sequence[str]:
    scan = pl.scan_parquet(str(feature_matrix_path))
    schema = scan.collect_schema()
    return schema.names()


def _select_feature_columns(columns: Sequence[str]) -> list[str]:
    selected: list[str] = []
    prefixes = (
        "injury_",
        "practice_",
        "status_",
        "report_",
        "team_ctx_",
        "opp_ctx_",
        "ps_",
        "drive_hist_",
        "snap_",
        "off_ctx_",
        "moneyline_",
        "spread_",
        "total_",
        "implied_prob_",
        "team_implied_total",
        "opp_implied_total",
    )
    blacklist = {
        "injury_game_designation",
        "injury_report_status",
        "injury_report_status_sequence",
        "injury_practice_status",
        "injury_practice_status_day1",
        "injury_practice_status_day2",
        "injury_practice_status_day3",
        "injury_practice_status_sequence",
        "injury_practice_primary",
        "status",
        "status_description_abbr",
        PROB_HEURISTIC_COL,
        MODEL_PROB_COL,
        PROB_LOW_COL,
        PROB_HIGH_COL,
        "injury_inactive_probability_interval_width",
        "injury_inactive_probability_source",
    }
    whitelist = {
        "team",
        "opponent",
        "season",
        "week",
        "game_day_of_week",
        "is_home",
        "season_type",
        "depth_chart_position",
        "depth_chart_order",
        "position",
        "position_group",
        "primary_qb_id",
    }
    for col in columns:
        if col == LABEL_COL:
            continue
        if col in blacklist:
            continue
        if col.startswith(prefixes) or col in whitelist:
            selected.append(col)
    return selected


def _load_feature_matrix(feature_matrix_path: Path, feature_columns: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(feature_columns + [LABEL_COL]))
    scan = pl.scan_parquet(str(feature_matrix_path))
    available = set(scan.collect_schema().names())
    projection = [pl.col(col) for col in columns if col in available]
    df = scan.select(projection).collect(streaming=True).to_pandas()
    missing = [col for col in columns if col not in df.columns]
    for col in missing:
        df[col] = pd.NA
    return df[columns]


def _prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df[df[LABEL_COL].notna()].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df.drop(columns=[LABEL_COL]), df[LABEL_COL]


def _train_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    calibration_seasons: Iterable[int],
    test_seasons: Iterable[int],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    seasons = X.get("season")
    if seasons is None:
        raise ValueError("Feature matrix must contain 'season' column for time-based split.")
    seasons = seasons.astype(int)
    calibration_seasons = {int(s) for s in calibration_seasons}
    test_seasons = {int(s) for s in test_seasons}
    train_mask = ~seasons.isin(calibration_seasons | test_seasons)
    cal_mask = seasons.isin(calibration_seasons)
    test_mask = seasons.isin(test_seasons)
    if train_mask.any() and cal_mask.any() and test_mask.any():
        return (
            X.loc[train_mask],
            y.loc[train_mask],
            X.loc[cal_mask],
            y.loc[cal_mask],
            X.loc[test_mask],
            y.loc[test_mask],
        )

    logger.warning(
        "Falling back to chronological split for injury model "
        "(train=%d, cal=%d, test=%d).",
        int(train_mask.sum()),
        int(cal_mask.sum()),
        int(test_mask.sum()),
    )

    def _sorted_index(frame: pd.DataFrame) -> pd.Index:
        sort_cols: list[str] = []
        for col in ("season", "week", "game_day_of_week", "game_start_utc"):
            if col in frame.columns:
                sort_cols.append(col)
        if not sort_cols:
            return frame.index
        return frame.sort_values(sort_cols).index

    order = _sorted_index(X)
    total_rows = len(order)
    if total_rows < 10:
        raise ValueError("Not enough rows to construct fallback injury training split.")

    train_end = max(int(total_rows * 0.6), 1)
    cal_end = max(train_end + int(total_rows * 0.2), train_end + 1)
    cal_end = min(cal_end, total_rows - 1)

    idx_train = order[:train_end]
    idx_cal = order[train_end:cal_end]
    idx_test = order[cal_end:]

    return (
        X.loc[idx_train],
        y.loc[idx_train],
        X.loc[idx_cal],
        y.loc[idx_cal],
        X.loc[idx_test],
        y.loc[idx_test],
    )


def _prepare_categories(
    X_train: pd.DataFrame,
    X_cal: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, list[str]]]:
    datetime_cols = X_train.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    for col in datetime_cols:
        for frame in (X_train, X_cal, X_test):
            dt_values = pd.to_datetime(frame[col], utc=True, errors="coerce")
            mask = dt_values.isna().to_numpy()
            int_values = dt_values.to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
            numeric = int_values.astype(float)
            numeric[mask] = np.nan
            frame[col] = numeric / 1e9
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    category_levels: dict[str, list[str]] = {}
    for col in categorical_cols:
        train_values = X_train[col].fillna("__missing__").astype(str)
        categories = sorted(set(train_values))
        category_levels[col] = categories
        X_train[col] = pd.Categorical(train_values, categories=categories)
        X_cal[col] = pd.Categorical(X_cal[col].fillna("__missing__").astype(str), categories=categories)
        X_test[col] = pd.Categorical(X_test[col].fillna("__missing__").astype(str), categories=categories)
    numeric_cols = X_train.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    X_train[numeric_cols] = X_train[numeric_cols].astype(np.float32)
    X_cal[numeric_cols] = X_cal[numeric_cols].astype(np.float32)
    X_test[numeric_cols] = X_test[numeric_cols].astype(np.float32)
    return X_train, X_cal, X_test, categorical_cols, datetime_cols, category_levels


def _fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_cols: list[str],
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=600,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        num_leaves=63,
        reg_alpha=0.1,
        reg_lambda=0.3,
        min_child_samples=25,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_cols)
    return model


def _calibrate_model(
    base_model: lgb.LGBMClassifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
) -> CalibratedClassifierCV:
    calibrator = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    calibrator.fit(X_cal, y_cal)
    return calibrator


def _compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics["auc"] = roc_auc_score(y_true, proba)
    metrics["pr_auc"] = average_precision_score(y_true, proba)
    metrics["brier_score"] = brier_score_loss(y_true, proba)
    metrics["log_loss"] = log_loss(y_true, proba, labels=[0, 1])
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    metrics["pr_curve"] = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }
    return metrics


def _conformal_margin(y_true: np.ndarray, proba: np.ndarray, alpha: float) -> float:
    residuals = np.abs(proba - y_true)
    return float(np.quantile(residuals, 1 - alpha))


def train_injury_model(
    feature_matrix_path: Path | None = None,
    alpha: float = DEFAULT_ALPHA,
    train_seasons: Iterable[int] | None = None,
    calibration_seasons: Iterable[int] | None = None,
    test_seasons: Iterable[int] | None = None,
) -> InjuryModelArtifact:
    feature_matrix_path = feature_matrix_path or DATA_DIR / "final" / "processed.parquet"
    columns = _load_feature_schema(feature_matrix_path)
    feature_cols = _select_feature_columns(columns)
    df = _load_feature_matrix(feature_matrix_path, feature_cols)
    X, y = _prepare_dataframe(df)
    all_seasons = sorted({int(s) for s in X["season"].unique()})
    calibration_seasons = (
        list(calibration_seasons)
        if calibration_seasons is not None
        else [max(all_seasons) - 1]
    )
    test_seasons = (
        list(test_seasons)
        if test_seasons is not None
        else [max(all_seasons)]
    )
    train_seasons = (
        list(train_seasons)
        if train_seasons is not None
        else [s for s in all_seasons if s not in calibration_seasons + test_seasons]
    )
    X_train, y_train, X_cal, y_cal, X_test, y_test = _train_validation_split(
        X, y, calibration_seasons, test_seasons
    )
    X_train, X_cal, X_test, categorical_cols, datetime_cols, category_levels = _prepare_categories(
        X_train.copy(), X_cal.copy(), X_test.copy()
    )
    base_model = _fit_model(X_train, y_train, categorical_cols)
    calibrated_model = _calibrate_model(base_model, X_cal, y_cal)
    proba_cal = calibrated_model.predict_proba(X_cal)[:, 1]
    proba_test = calibrated_model.predict_proba(X_test)[:, 1]
    cal_metrics = _compute_metrics(y_cal.to_numpy(), proba_cal)
    test_metrics = _compute_metrics(y_test.to_numpy(), proba_test)
    metrics = {
        "calibration": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in cal_metrics.items()},
        "test": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in test_metrics.items()},
    }
    margin = _conformal_margin(y_cal.to_numpy(), proba_cal, alpha)
    artifact = InjuryModelArtifact(
        model=calibrated_model,
        feature_columns=feature_cols,
        categorical_features=categorical_cols,
        datetime_features=datetime_cols,
        category_levels=category_levels,
        conformal_margin=margin,
        alpha=float(alpha),
        metrics={
            "calibration_auc": metrics["calibration"]["auc"],
            "calibration_pr_auc": metrics["calibration"]["pr_auc"],
            "calibration_brier": metrics["calibration"]["brier_score"],
            "calibration_log_loss": metrics["calibration"]["log_loss"],
            "test_auc": metrics["test"]["auc"],
            "test_pr_auc": metrics["test"]["pr_auc"],
            "test_brier": metrics["test"]["brier_score"],
            "test_log_loss": metrics["test"]["log_loss"],
        },
        train_seasons=[int(s) for s in train_seasons],
        calibration_seasons=[int(s) for s in calibration_seasons],
        created_at=datetime.utcnow().isoformat(),
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = MODEL_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "injury_availability_model.joblib"
    joblib.dump(artifact, artifact_path)
    latest_path = MODEL_DIR / "injury_availability_model.joblib"
    joblib.dump(artifact, latest_path)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRIC_DIR / timestamp / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as fp:
        json.dump(
            {
                "train_seasons": [int(s) for s in train_seasons],
                "calibration_seasons": [int(s) for s in calibration_seasons],
                "test_seasons": [int(s) for s in test_seasons],
                "alpha": alpha,
                "margin": margin,
                "calibration_metrics": metrics["calibration"],
                "test_metrics": metrics["test"],
                "feature_count": len(feature_cols),
                "categorical_features": categorical_cols,
                "created_at": artifact.created_at,
            },
            fp,
            indent=2,
        )
    logger.info(
        "Injury availability model trained. Calibration AUC %.4f, Test AUC %.4f, conformal margin %.4f",
        metrics["calibration"]["auc"],
        metrics["test"]["auc"],
        margin,
    )
    return artifact


def load_latest_artifact() -> InjuryModelArtifact | None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    direct_path = MODEL_DIR / "injury_availability_model.joblib"
    if direct_path.exists():
        try:
            return joblib.load(direct_path)
        except (ModuleNotFoundError, ImportError, Exception) as e:
            logger.warning("Failed to load injury model artifact from %s: %s", direct_path, e)
    candidates = sorted(MODEL_DIR.glob("*/injury_availability_model.joblib"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        return joblib.load(latest)
    except (ModuleNotFoundError, ImportError, Exception) as e:
        logger.warning("Failed to load injury model artifact from %s: %s", latest, e)
        return None


def predict_probabilities(
    df: pl.DataFrame,
    artifact: InjuryModelArtifact,
) -> pl.DataFrame:
    missing_cols = [col for col in artifact.feature_columns if col not in df.columns]
    if missing_cols:
        additions = []
        for col in missing_cols:
            if col in artifact.categorical_features:
                additions.append(pl.lit("__missing__").alias(col))
            elif col in artifact.datetime_features:
                additions.append(pl.lit(None).cast(pl.Datetime("ms")).alias(col))
            else:
                additions.append(pl.lit(None).cast(pl.Float32).alias(col))
        df = df.with_columns(additions)
    pandas_df = df.select([pl.col(col) for col in artifact.feature_columns]).to_pandas()
    for col in artifact.categorical_features:
        if col not in pandas_df.columns:
            pandas_df[col] = "__missing__"
        pandas_df[col] = (
            pandas_df[col]
            .fillna("__missing__")
            .astype(str)
        )
        categories = artifact.category_levels.get(col, None)
        if categories:
            known = set(categories)
            pandas_df.loc[~pandas_df[col].isin(known), col] = "__missing__"
            pandas_df[col] = pd.Categorical(pandas_df[col], categories=categories)
    if artifact.datetime_features:
        for col in artifact.datetime_features:
            if col in pandas_df.columns:
                dt_values = pd.to_datetime(pandas_df[col], utc=True, errors="coerce")
                numeric = dt_values.to_numpy(dtype="datetime64[ns]").astype("int64", copy=False).astype(float)
                mask = dt_values.isna().to_numpy()
                numeric[mask] = np.nan
                pandas_df[col] = numeric / 1e9
    numeric_cols = pandas_df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    pandas_df[numeric_cols] = pandas_df[numeric_cols].astype(np.float32)
    object_cols = [
        col
        for col in pandas_df.select_dtypes(include=["object"]).columns
        if col not in artifact.categorical_features
    ]
    for col in object_cols:
        pandas_df[col] = pd.to_numeric(pandas_df[col], errors="coerce")
    numeric_cols = pandas_df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    pandas_df[numeric_cols] = pandas_df[numeric_cols].astype(np.float32)
    proba = artifact.model.predict_proba(pandas_df)[:, 1]
    margin = artifact.conformal_margin
    lower = np.clip(proba - margin, 0.0, 1.0)
    upper = np.clip(proba + margin, 0.0, 1.0)
    reliability = upper - lower
    return pl.DataFrame(
        {
            MODEL_PROB_COL: proba.astype(np.float32),
            PROB_LOW_COL: lower.astype(np.float32),
            PROB_HIGH_COL: upper.astype(np.float32),
            "injury_inactive_probability_interval_width": reliability.astype(np.float32),
        }
    )


