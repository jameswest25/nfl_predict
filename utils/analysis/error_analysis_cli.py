"""
Post-hoc error analysis – trains a meta-model on model misses and scans feature separations.
"""
from pathlib import Path
import argparse
import json
from itertools import combinations

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from utils.analysis.error_meta import load_error_frames, run_error_meta_model
from utils.analysis.error_stats import ks_screen, pa_history_bias

try:
    import polars as pl  # type: ignore

    _POLARS_OK = True
except ImportError:
    _POLARS_OK = False


def _sigmoid(z: pd.Series, k: float = 4.0, center: float = 0.25) -> pd.Series:
    """Smoothly compress extremes while preserving ordering."""
    return 1.0 / (1.0 + np.exp(-k * (z - center)))


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc error analysis – trains a meta-model on model misses and scans feature separations."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="error_frame_*.parquet",
        help="Glob pattern (inside output/prediction_analysis) selecting error_frame files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/error_analysis",
        help="Destination directory for meta-model and reports.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip meta-model training and only produce KS/chi² separator scan.",
    )
    parser.add_argument(
        "--expr",
        action="append",
        default=[],
        help=(
            "Custom feature expression(s) in the form 'new_col = python_expr'. "
            "Variables can reference existing dataframe columns as df['col'] or simply col. "
            "NumPy is available as np. Repeat --expr for multiple features."
        ),
    )
    args = parser.parse_args()

    df = load_error_frames(args.pattern)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss

        base_auc = roc_auc_score(df["true_label"], df["predicted_probability"])
        base_brier = brier_score_loss(df["true_label"], df["predicted_probability"])
    except Exception as exc:
        base_auc = base_brier = float("nan")
        print("[error_analysis] Failed to compute main-model metrics:", exc)

    for expr in args.expr:
        try:
            lhs, rhs = expr.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
        except ValueError:
            print(f"[error_analysis] Could not parse expression: {expr} – use 'new_col = expr'.")
            continue
        try:
            local_ns = {**{c: df[c] for c in df.columns}, "np": np, "df": df}
            df[lhs] = eval(rhs, {}, local_ns)
            print(f"[error_analysis] Added column '{lhs}' via expression.")
        except Exception as e:
            print(f"[error_analysis] Failed to evaluate expression '{expr}': {e}")

    bucket_cols = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        series = df[col]
        if series.count() < 200 or series.nunique(dropna=True) < 10:
            continue
        try:
            q = pd.qcut(series, q=5, duplicates="drop")
            bcol = f"{col}_q5"
            df[bcol] = q.astype(str)
            bucket_cols.append(bcol)
        except ValueError:
            continue

    cat_cols_all = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols_all:
        if col in {"batter_name", "pitcher_name"}:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 30:
            if col not in bucket_cols:
                bucket_cols.append(col)
        else:
            top_levels = df[col].value_counts().nlargest(20).index
            df[f"{col}_cat"] = np.where(df[col].isin(top_levels), df[col].astype(str), "OTHER")
            new_col_name = f"{col}_cat"
            if new_col_name not in bucket_cols:
                bucket_cols.append(new_col_name)
    bucket_cols = list(dict.fromkeys(bucket_cols))
    print(f"[error_analysis] Created {len(bucket_cols)} bucket columns for feature analysis.")

    def _strip_suffix(col_name: str) -> str:
        for suf in ("_q5", "_cat"):
            if col_name.endswith(suf):
                return col_name[: -len(suf)]
        return col_name

    uni_rows = []
    for col in bucket_cols:
        diag = (
            df.groupby(col)[["is_error", "predicted_probability", "true_label"]]
            .agg(
                error_rate=("is_error", "mean"),
                count=("is_error", "size"),
                avg_prob=("predicted_probability", "mean"),
                actual_rate=("true_label", "mean"),
            )
            .reset_index()
        )
        diag["bias"] = diag["avg_prob"] - diag["actual_rate"]
        diag.insert(0, "feature", _strip_suffix(col))
        uni_rows.append(diag)
    feature_buckets_df = pd.concat(uni_rows, ignore_index=True)
    fb_sorted = feature_buckets_df.sort_values("error_rate", ascending=False).head(200)
    fb_sorted.to_csv(out_dir / "feature_buckets.csv", index=False)
    print(f"[error_analysis] feature_buckets.csv written (top 200 of {len(feature_buckets_df)} rows).")

    ALLOWED_PREFIXES = ("hist_", "rw_", "crude_", "mix_", "wind", "air", "temp", "pitch_", "opp_", "log_")
    selected_cols = [
        col for col in bucket_cols if any(_strip_suffix(col).startswith(pref) for pref in ALLOWED_PREFIXES)
    ]

    MIN_PAIR_COUNT = 50

    if _POLARS_OK:
        nn_counts = df[selected_cols].notna().sum().to_dict()
        pl_cols = selected_cols + ["is_error"]
        pl_df = pl.from_pandas(df[pl_cols])

        def _compute_pair(a: str, b: str):
            if min(nn_counts.get(a, 0), nn_counts.get(b, 0)) < MIN_PAIR_COUNT:
                return None
            res = (
                pl_df.group_by([a, b])
                .agg([
                    pl.col("is_error").mean().alias("error_rate"),
                    pl.len().alias("count"),
                ])
                .filter(pl.col("count") >= MIN_PAIR_COUNT)
            )
            if res.height == 0:
                return None
            sel = res.to_pandas()
            sel = sel.rename(columns={a: "bucket_a", b: "bucket_b"})
            sel.insert(0, "feature_a", _strip_suffix(a))
            sel.insert(1, "feature_b", _strip_suffix(b))
            sel["pair"] = sel["feature_a"] + "+" + sel["feature_b"]
            return sel

    else:

        def _compute_pair(a: str, b: str):
            grp = (
                df.groupby([a, b], as_index=False, observed=True)
                .agg(error_rate=("is_error", "mean"), count=("is_error", "size"))
            )
            sel = grp[grp["count"] >= MIN_PAIR_COUNT].copy()
            if sel.empty:
                return None
            sel = sel.rename(columns={a: "bucket_a", b: "bucket_b"})
            sel.insert(0, "feature_a", _strip_suffix(a))
            sel.insert(1, "feature_b", _strip_suffix(b))
            sel["pair"] = sel["feature_a"] + "+" + sel["feature_b"]
            return sel

    top_pair_features: list[str] = []
    pair_rows_final = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_compute_pair)(a, b) for a, b in combinations(selected_cols, 2)
    )
    pair_rows_final = [p for p in pair_rows_final if p is not None]
    if pair_rows_final:
        pair_df = pd.concat(pair_rows_final, ignore_index=True)
        pair_df_sorted = pair_df.sort_values("error_rate", ascending=False).head(200)
        pair_df_sorted.to_csv(out_dir / "pairwise_buckets.csv", index=False)
        print(f"[error_analysis] pairwise_buckets.csv written (top 200 of {len(pair_df)} rows).")
        top_pair_features = pd.unique(pair_df_sorted[["feature_a", "feature_b"]].values.ravel()).tolist()[:200]
    else:
        pd.DataFrame().to_csv(out_dir / "pairwise_buckets.csv", index=False)
        print("[error_analysis] pairwise_buckets.csv written (rows=0).")

    def _multiway(level: int, max_cols: int = 1275, min_count: int = 50, top_k: int = 200):
        import heapq

        cols = [c for c in bucket_cols if any(_strip_suffix(c).startswith(pref) for pref in ALLOWED_PREFIXES)]
        if top_pair_features:
            cols = [c for c in cols if _strip_suffix(c) in top_pair_features]
        cols = cols[:max_cols]

        if _POLARS_OK:
            pl_cols = cols + ["is_error"]
            pl_df_sub = pl.from_pandas(df[pl_cols])

            def _compute_combo(combo):
                res = (
                    pl_df_sub.group_by(list(combo))
                    .agg([
                        pl.col("is_error").mean().alias("error_rate"),
                        pl.len().alias("count"),
                    ])
                    .filter(pl.col("count") >= min_count)
                )
                if res.height == 0:
                    return None
                sel = res.to_pandas()
                for idx, col in enumerate(combo):
                    sel = sel.rename(columns={col: f"bucket_{idx+1}"})
                    sel.insert(idx, f"feature_{idx+1}", _strip_suffix(col))
                sel["combo"] = "+".join(_strip_suffix(c) for c in combo)
                return sel

        else:

            def _compute_combo(combo):
                grp = (
                    df.groupby(list(combo), as_index=False, observed=True)
                    .agg(error_rate=("is_error", "mean"), count=("is_error", "size"))
                )
                sel = grp[grp["count"] >= min_count].copy()
                if sel.empty:
                    return None
                for idx, col in enumerate(combo):
                    sel = sel.rename(columns={col: f"bucket_{idx+1}"})
                    sel.insert(idx, f"feature_{idx+1}", _strip_suffix(col))
                sel["combo"] = "+".join(_strip_suffix(c) for c in combo)
                return sel

        heap: list[tuple[float, int, dict]] = []
        seq = 0

        for combo in combinations(cols, level):
            sel = _compute_combo(combo)
            if sel is None or sel.empty:
                continue
            threshold = heap[0][0] if len(heap) == top_k else float("-inf")
            for _, row in sel.iterrows():
                er = float(row["error_rate"])
                if er <= threshold:
                    continue
                seq += 1
                item = (er, seq, row.to_dict())
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                else:
                    heapq.heapreplace(heap, item)
                    threshold = heap[0][0]

        if not heap:
            return pd.DataFrame()

        best_rows = [i[2] for i in heapq.nlargest(top_k, heap, key=lambda x: x[0])]
        out_df = pd.DataFrame(best_rows)
        return out_df.sort_values("error_rate", ascending=False).head(top_k)

    three_df = _multiway(3, max_cols=200)
    if not three_df.empty:
        three_df.to_csv(out_dir / "threeway_buckets.csv", index=False)
        print(f"[error_analysis] threeway_buckets.csv written (rows={len(three_df)})")
    else:
        pd.DataFrame().to_csv(out_dir / "threeway_buckets.csv", index=False)
        print("[error_analysis] threeway_buckets.csv written (rows=0)")

    df_base = df.drop(columns=bucket_cols, errors="ignore")
    cont, cat = ks_screen(df_base, alpha=0.01, d_min=0.05, min_cat_rows=15, min_n_samples=25)
    pd.DataFrame(cont, columns=["feature", "ks_distance", "p_value"]).to_csv(
        out_dir / "continuous_separators.csv", index=False
    )
    pd.DataFrame(cat, columns=["feature", "chi2", "p_value"]).to_csv(
        out_dir / "categorical_separators.csv", index=False
    )

    try:
        model_key_parts = args.pattern.strip("*").split("_")
        model_name = next((p for p in model_key_parts if p in {"lightgbm", "xgboost"}), "lightgbm")
        problem_name = next(
            (p for p in ["gets_hit", "gets_2plus_bases", "gets_run_rbi", "expected_bases"] if p in args.pattern),
            "gets_hit",
        )

        model_files = sorted(Path("output/models").glob(f"{model_name}_{problem_name}_*.pkl"))
        if model_files:
            model_path = model_files[-1]
            raw_model = joblib.load(model_path)
            if hasattr(raw_model, "base_estimator"):
                model = raw_model.base_estimator
            elif hasattr(raw_model, "estimator"):
                model = raw_model.estimator
            else:
                model = raw_model

            if hasattr(model, "booster_"):
                feat_names = model.booster_.feature_name()
            elif hasattr(model, "feature_name_"):
                feat_names = model.feature_name_
            else:
                feat_names = [
                    c
                    for c in df.columns
                    if c not in {"is_error", "true_label", "predicted_label", "predicted_probability", "confusion"}
                ]

            missing = [f for f in feat_names if f not in df.columns]
            if missing:
                print(f"[error_analysis] WARNING – {len(missing)} model features missing from error frame.")
                feat_names = [f for f in feat_names if f in df.columns]

            X_full = df[feat_names].copy()
            for col in X_full.select_dtypes(include=["object", "category"]).columns:
                X_full[col] = pd.Categorical(X_full[col]).codes.astype("int32")
            X_full = X_full.fillna(0)

            explainer = shap.TreeExplainer(model)
            try:
                shap_vals = explainer.shap_values(X_full, check_additivity=False)
            except TypeError:
                shap_vals = explainer(X_full, check_additivity=False)

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

            shap.summary_plot(shap_vals, X_full, show=False, max_display=40)
            plt.tight_layout()
            plt.savefig(out_dir / "main_shap_summary.png", dpi=150)
            plt.close()

            mean_abs = np.abs(shap_vals).mean(axis=0)
            top_idx = np.argsort(mean_abs)[-40:][::-1]
            top_features = [X_full.columns[i] for i in top_idx]
            shap_df = pd.DataFrame(shap_vals[:, top_idx], columns=top_features)
            shap_df.to_parquet(out_dir / "shap_values.parquet", index=False)

            if "batter_name" in df.columns:
                abs_shap = np.abs(shap_vals[:, top_idx])
                batter_abs = (
                    pd.DataFrame(abs_shap, columns=top_features)
                    .assign(batter=df["batter_name"].values)
                    .groupby("batter")
                    .mean()
                )
                counts = df.groupby("batter_name").size()
                eligible = counts[counts >= 30].index
                batter_abs.loc[eligible].to_csv(out_dir / "shap_by_batter.csv")
            print("[error_analysis] Main-model SHAP files written.")
        else:
            print("[error_analysis] No model file found for SHAP analysis; skipping.")
    except Exception as exc:
        print("[error_analysis] SHAP analysis failed:", exc)

    if not args.stats_only:
        metrics = run_error_meta_model(df, out_dir)
        print("[error_analysis] Meta-model metrics:", metrics)
        comp = {
            "main_model_auc": base_auc,
            "main_model_brier": base_brier,
            "meta_model_auc": metrics.get("auc"),
            "meta_model_brier": metrics.get("brier"),
            "n_rows": metrics.get("n_rows"),
        }
        (out_dir / "main_vs_meta_metrics.json").write_text(json.dumps(comp, indent=2))
        print("[error_analysis] main_vs_meta_metrics.json written.")

    try:
        bat = (
            df.groupby("batter_name")
            .agg(
                error_rate=("is_error", "mean"),
                count=("is_error", "size"),
                avg_prob=("predicted_probability", "mean"),
                actual_rate=("true_label", "mean"),
            )
            .query("count >= 10")
        )
        bat["bias"] = bat["avg_prob"] - bat["actual_rate"]
        bat.sort_values("error_rate", ascending=False).to_csv(out_dir / "batter_diagnostics.csv")
        print(f"[error_analysis] Wrote batter_diagnostics.csv (n={len(bat)}) to {out_dir}")
        pa_bias = pa_history_bias(df, quantile=True)
        pa_bias.to_csv(out_dir / "pa_history_bias.csv", index=False)
        if "pitcher_name" in df.columns:
            pit = (
                df.groupby("pitcher_name")
                .agg(
                    error_rate=("is_error", "mean"),
                    count=("is_error", "size"),
                    avg_prob=("predicted_probability", "mean"),
                    actual_rate=("true_label", "mean"),
                )
                .query("count >= 10")
            )
            pit["bias"] = pit["avg_prob"] - pit["actual_rate"]
            pit.sort_values("error_rate", ascending=False).to_csv(out_dir / "pitcher_diagnostics.csv")
            print(f"[error_analysis] Wrote pitcher_diagnostics.csv (n={len(pit)}) to {out_dir}")
        elif "starter_pitcher_id" in df.columns:
            pit = (
                df.groupby("starter_pitcher_id")
                .agg(
                    error_rate=("is_error", "mean"),
                    count=("is_error", "size"),
                    avg_prob=("predicted_probability", "mean"),
                    actual_rate=("true_label", "mean"),
                )
                .query("count >= 10")
            )
            pit["bias"] = pit["avg_prob"] - pit["actual_rate"]
            if "pitcher_y" in df.columns:
                mapping = df.drop_duplicates("starter_pitcher_id")[["starter_pitcher_id", "pitcher_y"]].set_index(
                    "starter_pitcher_id"
                )
                pit = pit.merge(mapping, left_index=True, right_index=True, how="left")
                pit.index.name = "pitcher_id"
            pit.sort_values("error_rate", ascending=False).to_csv(out_dir / "pitcher_diagnostics.csv")
            print(f"[error_analysis] Wrote pitcher_diagnostics.csv (n={len(pit)}) to {out_dir}")
    except Exception as exc:
        print("[error_analysis] Could not compute PA history bias:", exc)

    if "batter_name" in df.columns:
        bat_err = (
            df.groupby("batter_name")["is_error"].agg(["mean", "count"])
            .rename(columns={"mean": "error_rate"})
            .query("count >= 10")
            .sort_values("error_rate", ascending=False)
        )
        bat_err.to_csv(out_dir / "batter_error_rates.csv")
        print(f"[error_analysis] Wrote batter_error_rates.csv (n={len(bat_err)}) to {out_dir}")
    print(
        f"[error_analysis] Wrote {len(cont)} continuous and {len(cat)} categorical separator candidates to {out_dir}"
    )


if __name__ == "__main__":
    main()


