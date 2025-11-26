from pathlib import Path
from typing import Tuple, List

import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
import numpy as np

__all__ = [
    "ks_screen",
    "pa_history_bias",
]

# Columns that should never be considered when searching for separators –
# they are either direct leakages or not genuine explanatory features.
_DENY_COLS = {
    "confusion",
    "true_label",
    "predicted_label",
    "predicted_probability",
    "bases_sum",
    "gets_hit",
    "gets_2plus_bases",
    "run_or_rbi",
    "hit_count",
    "is_hit",
}

# ----------------------------------------------------------------------------------
# KS / χ² separation scan
# ----------------------------------------------------------------------------------
# NOTE: The signature keeps default hyper-strict thresholds but accepts
# additional tuning parameters (min_n_samples) and smarter type handling.

def ks_screen(
    df: pd.DataFrame,
    alpha: float = 0.001,
    d_min: float = 0.1,
    min_cat_rows: int = 30,
    min_n_samples: int = 50,
) -> Tuple[List[tuple], List[tuple]]:
    """Identify continuous & categorical columns that separate error vs non-error.

    Returns
    -------
    cont_separators : list[tuple[str, float, float]]
        Each tuple is (column, KS distance, p-value).
    cat_separators : list[tuple[str, float, float]]
        Each tuple is (column, chi² statistic, p-value).
    """
    # Ensure 'is_error' present and drop deny columns
    if "is_error" not in df.columns:
        raise ValueError("DataFrame must contain 'is_error' column")

    df = df.drop(columns=[c for c in _DENY_COLS if c in df.columns], errors="ignore")

    cont, cat = [], []
    error_mask = df["is_error"] == 1
    for col in df.columns:
        if col not in df.columns or col == "is_error":
            continue
        series = df[col]
        # Decide if this should be treated as numeric or categorical.
        # Low-cardinality numeric columns (e.g. 0/1 flags or 1-9 buckets) behave
        # more like categoricals – chi² is more appropriate than KS.
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 30:
            a = series[error_mask].dropna()
            b = series[~error_mask].dropna()
            if len(a) >= min_n_samples and len(b) >= min_n_samples:
                d, p = ks_2samp(a, b)
                if p < alpha and abs(d) > d_min:
                    cont.append((col, float(d), float(p)))
        else:
            tbl_full = pd.crosstab(series, df["is_error"])
            # Drop levels with too few rows to avoid spurious significance
            tbl = tbl_full[tbl_full.sum(axis=1) >= min_cat_rows]
            if tbl.shape[0] > 1:
                chi2, p, *_ = chi2_contingency(tbl)
                if p < alpha:
                    cat.append((col, float(chi2), float(p)))

    cont_sorted = sorted(cont, key=lambda x: -abs(x[1]))
    cat_sorted = sorted(cat, key=lambda x: -abs(x[1]))
    return cont_sorted, cat_sorted

# ----------------------------------------------------------------------------------
# Player-history bias diagnostic
# ----------------------------------------------------------------------------------
def pa_history_bias(
    df: pd.DataFrame,
    pa_col: str = "hist_lifetime_pas",
    bins: tuple = (0, 50, 100, 250, 500, 1000, 5_000, 1e9),
    quantile: bool = False,
    log10_buckets: bool = False,
    per_player: bool = True,
    player_col: str = "batter_name",
) -> pd.DataFrame:
    """Return error-rate vs. lifetime-PA buckets.

    If *per_player* is True (default) we first aggregate to one row per player so
    that frequent hitters do not dominate the average error rate. Otherwise the
    original row-level behaviour is preserved.
    """

    if pa_col not in df.columns:
        raise ValueError(f"{pa_col} column not found in dataframe")

    if per_player and player_col in df.columns:
        # Aggregate to player level first
        per_p = (
            df.groupby(player_col)
            .agg(
                pa_val=(pa_col, "max"),  # maximum lifetime PAs seen for that player
                error_rate=("is_error", "mean"),
            )
            .reset_index()
        )
        tmp = per_p.rename(columns={"pa_val": pa_col})[[pa_col, "error_rate"]]
        tmp[pa_col] = tmp[pa_col].fillna(0)
    else:
        tmp = df[[pa_col, "is_error"]].copy()
        tmp = tmp.rename(columns={"is_error": "error_rate"})
        tmp[pa_col] = tmp[pa_col].fillna(0)

    # Bin the lifetime PA values
    if log10_buckets:
        tmp["_bin"] = (
            np.floor(np.log10(tmp[pa_col].replace(0, 1)))
            .clip(0)
            .astype(int)
        )
    elif quantile:
        try:
            tmp["_bin"] = pd.qcut(tmp[pa_col], q=10, duplicates="drop")
        except ValueError:
            tmp["_bin"] = pd.cut(tmp[pa_col], bins=bins, right=False)
    else:
        tmp["_bin"] = pd.cut(tmp[pa_col], bins=bins, right=False)

    # Final aggregation – mean error rate in each bucket
    out = (
        tmp.groupby("_bin")["error_rate"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "error_rate"})
    )

    return out.reset_index() 