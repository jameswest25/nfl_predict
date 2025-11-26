from pathlib import Path
import glob
import json
from typing import List

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

# Optional – SHAP can be heavy; import lazily in case user only wants metrics
try:
    import shap  # noqa: F401
    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False

__all__ = [
    "load_error_frames",
    "run_error_meta_model",
]

def load_error_frames(pattern: str | List[str] = "error_frame_*.parquet", base_dir: str | Path = "output/prediction_analysis") -> pd.DataFrame:
    """Load one or multiple consolidated error_frame parquet files.

    Parameters
    ----------
    pattern : str | list[str]
        Glob pattern(s) relative to *base_dir* selecting the files to load.
    base_dir : str | Path
        Directory where error_frame files are stored.
    """
    base_dir = Path(base_dir)
    patterns = pattern if isinstance(pattern, list) else [pattern]
    files: list[Path] = []
    for p in patterns:
        files.extend(sorted(base_dir.glob(p)))
    if not files:
        raise FileNotFoundError(f"No error_frame files matching {patterns} inside {base_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df

def _fit_meta_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> tuple[LGBMClassifier, float, float]:
    """Fit a small LightGBM classifier and return model + mean AUC + mean Brier."""
    meta = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs: list[float] = []
    briers: list[float] = []
    for tr, te in cv.split(X, y):
        meta.fit(X.iloc[tr], y.iloc[tr])
        prob_te = meta.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], prob_te))
        briers.append(brier_score_loss(y.iloc[te], prob_te))
    return meta.fit(X, y), float(np.mean(aucs)), float(np.mean(briers))

def run_error_meta_model(df: pd.DataFrame, out_dir: str | Path = "output/error_analysis") -> dict:
    """Train meta-model & dump artefacts. Returns metrics dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split features / target
    drop_cols = [
        "is_error",
        "confusion",
        "true_label",
        "predicted_label",
        "predicted_probability",
    ]

    deny_targets = [
        "bases_sum",  # regression target from expected_bases problem
        "gets_hit",
        "is_hit",
        "hit_count",
        "gets_2plus_bases",
        "run_or_rbi",
    ]

    X = df.drop(columns=[c for c in (drop_cols + deny_targets) if c in df.columns], errors="ignore")
    y = df["is_error"].astype(int)

    # --- NEW: Preprocess X to satisfy LightGBM dtype requirements ---
    # 1) Convert datetime columns to integer timestamp (ns since epoch)
    datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for col in datetime_cols:
        # Cast datetime64[ns] to integer nanoseconds to keep temporal ordering
        X[col] = X[col].astype("int64")

    # 2) Drop non-numeric columns that are still object or category; optional: encode later
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("After preprocessing, no numeric columns remain for meta-model training.")
    X = X[numeric_cols].copy()

    model, auc, brier = _fit_meta_model(X, y)

    # Persist model & metrics
    joblib.dump(model, out_dir / "meta_model.joblib")
    metrics = {"auc": auc, "brier": brier, "n_rows": int(len(df))}
    (out_dir / "meta_metrics.json").write_text(json.dumps(metrics, indent=2))

    # SHAP summary
    if _SHAP_AVAILABLE:
        import shap  # local import for speed
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(model)

        # SHAP API changed around v0.40; handle both list/array and Explanation
        try:
            shap_values = explainer.shap_values(X, check_additivity=False)
        except TypeError:
            # Newer API prefers calling the explainer directly
            shap_values = explainer(X, check_additivity=False)

        # If list → binary classifier returns [neg_class, pos_class]. Take pos.
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        shap.summary_plot(shap_values, X, show=False, max_display=40)
        plt.tight_layout()
        plt.savefig(out_dir / "meta_shap_summary.png", dpi=150)
        plt.close()
    else:
        print("[error_meta] SHAP not installed — skipping summary plot.")

    return metrics 