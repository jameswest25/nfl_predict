# utils/train/conformal_composite.py
from __future__ import annotations
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def fit_composite_conformal_sums(trainer) -> None:
    """Post-training: builds conformal intervals for sums of multiple regression problems.
    Controlled by training.composite_conformal_sums in YAML, e.g.:

      training:
        composite_conformal_sums:
          - name: bases_total
            components: ["tb_reg", "bb_reg", "hbp_reg"]  # <-- problem names
            alpha: 0.1
            method: "naive"   # or "hc"

    Saves artifacts: composite_conformal_sum_<name>_<ts>.joblib
    """
    specs = trainer.config.get("training", {}).get("composite_conformal_sums", [])
    if not specs:
        return

    from functools import reduce

    for spec in specs:
        try:
            name = spec["name"]
            comps = list(spec["components"])
            alpha = float(spec.get("alpha", 0.1))
            method = str(spec.get("method", "naive")).lower()

            blobs = [trainer._composite_cal_store.get(p) for p in comps]
            if any(b is None for b in blobs):
                missing = [c for c, b in zip(comps, blobs) if b is None]
                logger.warning(f"[composite:{name}] Missing cal store for {missing}; skipping.")
                continue

            # Align on common calibration indices across components
            idx_sets = [set(b["index"]) for b in blobs]
            common = reduce(lambda a, b: a & b, idx_sets)
            if not common:
                logger.warning(f"[composite:{name}] No common calibration indices; skipping.")
                continue
            common = sorted(common)
            def align(b):
                pos = pd.Index(b["index"]).get_indexer(common)
                return b["y_cal"][pos], b["y_hat_cal"][pos]

            ys, yhs = zip(*(align(b) for b in blobs))
            y_sum = np.sum(np.column_stack(ys), axis=1)
            yhat_sum = np.sum(np.column_stack(yhs), axis=1)
            resid = np.abs(y_sum - yhat_sum)

            artifact = {"name": name, "components": comps, "alpha": alpha, "method": method, "clip_lower": 0.0}

            if method == "hc":
                try:
                    # Lightweight heteroscedastic scaler on per-component preds
                    Xs = np.column_stack(yhs)  # features = predicted components
                    import xgboost as xgb
                    scaler = xgb.XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=300, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8,
                        n_jobs=1, tree_method="hist", enable_categorical=True, verbosity=0
                    )
                    scaler.fit(Xs, resid)
                    s = np.clip(scaler.predict(Xs), 1e-6, np.inf)
                    norm = resid / s
                    k = int(np.ceil((len(norm) + 1) * (1 - alpha)))
                    q = float(np.partition(norm, k - 1)[k - 1])
                    artifact.update({"q": q, "scaler": scaler})
                except Exception as e:
                    logger.warning(f"[composite:{name}] HC scaler failed ({e}); falling back to naive.")
                    method = "naive"

            if method == "naive":
                k = int(np.ceil((len(resid) + 1) * (1 - alpha)))
                q = float(np.partition(resid, k - 1)[k - 1])
                artifact.update({"q": q})

            # --- Diagnostics (always on cal slice; test slice if available) ---
            def _point_metrics(y_true, y_hat):
                rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
                mae  = float(np.mean(np.abs(y_true - y_hat)))
                bias = float(np.mean(y_hat - y_true))
                return rmse, mae, bias

            def _interval_metrics(y_true, y_hat, q_val, method, s_vec=None):
                if method == "hc" and s_vec is not None:
                    lo = y_hat - q_val * s_vec
                    hi = y_hat + q_val * s_vec
                else:
                    lo = y_hat - q_val
                    hi = y_hat + q_val
                hit   = (y_true >= lo) & (y_true <= hi)
                cov   = float(np.mean(hit))
                width = hi - lo
                return cov, float(np.mean(width)), float(np.median(width))

            # --- Calibration-slice metrics ---
            # yhat_sum, y_sum already computed above for cal slice
            cal_rmse, cal_mae, cal_bias = _point_metrics(y_sum, yhat_sum)
            if method == "hc":
                # s(x) on cal features (Xs defined above for 'hc': np.column_stack(yhs))
                s_cal = np.clip(artifact.get("scaler").predict(np.column_stack(yhs)), 1e-6, np.inf)
                cal_cov, cal_w_mean, cal_w_med = _interval_metrics(y_sum, yhat_sum, artifact["q"], method, s_cal)
            else:
                cal_cov, cal_w_mean, cal_w_med = _interval_metrics(y_sum, yhat_sum, artifact["q"], method)

            # --- Optional test-slice metrics (only when not in production mode) ---
            test_metrics = None
            try:
                blobs_test = [trainer._composite_test_store.get(p) for p in comps]
                if all(b is not None for b in blobs_test):
                    # align by common index across components
                    from functools import reduce
                    idx_sets = [set(b["index"]) for b in blobs_test]
                    common = reduce(lambda a, b: a & b, idx_sets)
                    if common:
                        common = sorted(common)
                        def align(b):
                            pos = pd.Index(b["index"]).get_indexer(common)
                            return b["y_test"][pos], b["y_hat_test"][pos]

                        ys_t, yhs_t = zip(*(align(b) for b in blobs_test))
                        y_sum_t   = np.sum(np.column_stack(ys_t), axis=1)
                        yhat_sum_t = np.sum(np.column_stack(yhs_t), axis=1)

                        t_rmse, t_mae, t_bias = _point_metrics(y_sum_t, yhat_sum_t)
                        if method == "hc":
                            Xs_t = np.column_stack(yhs_t)  # features for scaler at test time
                            s_t  = np.clip(artifact.get("scaler").predict(Xs_t), 1e-6, np.inf)
                            t_cov, t_w_mean, t_w_med = _interval_metrics(y_sum_t, yhat_sum_t, artifact["q"], method, s_t)
                        else:
                            t_cov, t_w_mean, t_w_med = _interval_metrics(y_sum_t, yhat_sum_t, artifact["q"], method)

                        test_metrics = {
                            "rmse": round(t_rmse, 6),
                            "mae": round(t_mae, 6),
                            "bias": round(t_bias, 6),
                            "coverage": round(t_cov, 6),
                            "width_mean": round(t_w_mean, 6),
                            "width_median": round(t_w_med, 6),
                            "n": int(len(y_sum_t)),
                        }
            except Exception as e:
                logger.warning(f"[composite:{name}] test diagnostics failed: {e}")

            # Emit to logs
            logger.info(f"[composite:{name}] cal: RMSE={cal_rmse:.4f} MAE={cal_mae:.4f} "
                        f"bias={cal_bias:.4f} coverage={cal_cov:.3f} w_mean={cal_w_mean:.3f} n={len(y_sum)}")
            if test_metrics is not None:
                logger.info(f"[composite:{name}] test: RMSE={test_metrics['rmse']:.4f} "
                            f"MAE={test_metrics['mae']:.4f} bias={test_metrics['bias']:.4f} "
                            f"coverage={test_metrics['coverage']:.3f} w_mean={test_metrics['width_mean']:.3f} "
                            f"n={test_metrics['n']}")

            # Persist a YAML report alongside the artifact
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                report = {
                    "name": name,
                    "components": comps,
                    "alpha": alpha,
                    "method": method,
                    "timestamp": ts,
                    "calibration_slice": {
                        "rmse": round(cal_rmse, 6),
                        "mae": round(cal_mae, 6),
                        "bias": round(cal_bias, 6),
                        "coverage": round(cal_cov, 6),
                        "width_mean": round(cal_w_mean, 6),
                        "width_median": round(cal_w_med, 6),
                        "n": int(len(y_sum)),
                    },
                    "test_slice": test_metrics,  # may be None in production mode
                }
                report_file = trainer.metric_dir / f"composite_report_{name}_{ts}.yaml"
                with open(report_file, "w") as f:
                    yaml.dump(report, f, sort_keys=False)
                logger.info(f"[composite:{name}] Wrote composite report → {report_file}")
            except Exception as e:
                logger.warning(f"[composite:{name}] Failed to write YAML report: {e}")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = trainer.model_dir / f"composite_conformal_sum_{name}_{ts}.joblib"
            joblib.dump(artifact, out, compress=1)
            logger.info(f"[composite:{name}] Saved artifact → {out}")

        except Exception as e:
            logger.error(f"[composite:{spec.get('name','?')}] Failed to build composite artifact: {e}")

def fit_team_conformal_sums(trainer) -> None:
    """Build team/game-level conformal artifacts using the per-row calibration
    predictions we stashed during problem training.

    Expects config:
      training:
        team_conformal_sums:
          - name: bases_total
            components: ["expected_bases", "expected_walks", "expected_hbps"]
            alpha: 0.10
            method: "heteroscedastic_conformal"   # or "naive"
    """
    specs = trainer.config.get("training", {}).get("team_conformal_sums", [])
    if not specs:
        logger.info("[team-conformal] No team_conformal_sums specs found — skipping.")
        return

    from functools import reduce

    for spec in specs:
        try:
            name = spec["name"]
            comps = list(spec["components"])
            alpha = float(spec.get("alpha", 0.10))
            method = str(spec.get("method", "naive")).lower()

            blobs = [trainer._composite_cal_store.get(p) for p in comps]
            if any(b is None for b in blobs):
                missing = [c for c, b in zip(comps, blobs) if b is None]
                logger.warning(f"[team:{name}] Missing cal store for {missing}; skipping.")
                continue

            # Align on common row indices across all components
            idx_objs = [pd.Index(b["index"]) for b in blobs]
            common = reduce(lambda a, b: a.intersection(b), idx_objs)
            if common.empty:
                logger.warning(f"[team:{name}] No common calibration indices across components; skipping.")
                continue
            common = common.sort_values()

            # Pull aligned arrays
            ys, yhs = [], []
            for b, idx in zip(blobs, idx_objs):
                pos = idx.get_indexer(common)
                ys.append(np.asarray(b["y_cal"])[pos])
                yhs.append(np.asarray(b["y_hat_cal"])[pos])

            Ytrue_row = np.sum(np.column_stack(ys), axis=1)      # per-row actual sum
            Yhat_row  = np.sum(np.column_stack(yhs), axis=1)     # per-row predicted sum
            resid_row = np.abs(Ytrue_row - Yhat_row)

            # Pull meta from the first component's stash
            base_blob = blobs[0]
            base_idx  = idx_objs[0]
            base_pos  = base_idx.get_indexer(common)
            meta_dict = base_blob.get("meta") or {}
            # NFL/MLB-compatible meta handling:
            # - Prefer explicit `team` per row when available (NFL).
            # - Fall back to MLB's inning_topbot mapping when available (MLB).
            group_col = getattr(trainer, "group_col", "game_id")
            alt_group_col = "game_pk"
            need = [group_col, alt_group_col, "team", "home_team", "away_team", "away_team_abbr", "home_team_abbr", "inning_topbot", trainer.time_col]
            meta = {k: np.asarray(meta_dict[k])[base_pos] for k in need if k in meta_dict}

            dt = pd.to_datetime(meta[trainer.time_col]) if trainer.time_col in meta else None

            # Determine group id column
            if group_col in meta:
                gid = meta[group_col]
                gid_col_name = group_col
            elif alt_group_col in meta:
                gid = meta[alt_group_col]
                gid_col_name = alt_group_col
            else:
                logger.warning(f"[team:{name}] Missing group id meta columns; skipping.")
                continue

            # Determine team_for_row
            if "team" in meta:
                team_for_row = meta["team"].astype(str)
            elif "inning_topbot" in meta and (("away_team_abbr" in meta) and ("home_team_abbr" in meta)):
                itb = meta["inning_topbot"].astype(str)
                away = meta["away_team_abbr"].astype(str)
                home = meta["home_team_abbr"].astype(str)
                team_for_row = np.where(np.char.startswith(itb, "Top"), away, home)
            else:
                logger.warning(f"[team:{name}] Missing team meta columns; skipping.")
                continue

            df_row = pd.DataFrame({
                gid_col_name: gid,
                "team_for_row": team_for_row,
                "y_true_row": Ytrue_row,
                "y_hat_row": Yhat_row,
                "resid_row": resid_row,
                "date": (dt.dt.date if dt is not None else pd.NaT),
            })

            # --- Fit "sigma model" on per-row component preds if HC
            sigma_model = None
            if method.startswith("hetero"):
                try:
                    Xs = np.column_stack(yhs)  # features = per-component y_hat at row level
                    import xgboost as xgb
                    sigma_model = xgb.XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=300, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8,
                        n_jobs=1, tree_method="hist", enable_categorical=True, verbosity=0
                    )
                    sigma_model.fit(Xs, df_row["resid_row"].values)
                    sigma_i = np.clip(sigma_model.predict(Xs), 1e-6, np.inf)
                except Exception as e:
                    logger.warning(f"[team:{name}] HC sigma_model failed ({e}); using naive method.")
                    method = "naive"
            # Aggregate to team/game
            if method == "naive":
                agg = (df_row.groupby([gid_col_name, "team_for_row", "date"], as_index=False)
                            .agg(y_true=("y_true_row","sum"),
                                 y_hat=("y_hat_row","sum")))
                res_team = np.abs(agg["y_true"].values - agg["y_hat"].values)
                # finite-sample split-conformal quantile k = ceil((n+1)*(1-alpha))
                n = len(res_team)
                k = int(np.ceil((n + 1) * (1 - alpha)))
                q_team = float(np.partition(res_team, k - 1)[k - 1])
                artifact = {
                    "name": name,
                    "team_method": "naive",
                    "q": q_team,
                }
            else:
                # HC: RMS aggregate of per-row sigma_i within each team/game
                df_row["_sig2"] = sigma_i ** 2
                agg = (df_row.groupby([gid_col_name, "team_for_row", "date"], as_index=False)
                            .agg(y_true=("y_true_row","sum"),
                                 y_hat=("y_hat_row","sum"),
                                 sigsum=("_sig2","sum")))
                res_team = np.abs(agg["y_true"].values - agg["y_hat"].values)
                sig_team = np.sqrt(np.clip(agg["sigsum"].values, 1e-12, np.inf))
                norm = res_team / sig_team
                n = len(norm)
                k = int(np.ceil((n + 1) * (1 - alpha)))
                q_team = float(np.partition(norm, k - 1)[k - 1])
                artifact = {
                    "name": name,
                    "team_method": "heteroscedastic_conformal",
                    "q": q_team,
                    "sigma_model": sigma_model,
                    "aggregate": "rms",
                }

            # Add common fields
            artifact.update({
                "components": comps,
                "alpha": alpha,
            })

            # Persist artifact
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = trainer.model_dir / f"team_conformal_sum_{name}_{ts}.joblib"
            joblib.dump(artifact, out, compress=1)
            logger.info(f"[team:{name}] Saved team conformal artifact → {out}")
        except Exception as e:
            logger.error(f"[team:{spec.get('name','?')}] Failed to build team artifact: {e}")

def composite_sum_interval(artifact_path: str, preds_by_component: dict):
    """Load a CompositeConformalSum artifact and return (yhat_sum, lo, hi).
    preds_by_component: {"tb_reg": np.array, "bb_reg": np.array, "hbp_reg": np.array}
    """
    art = joblib.load(artifact_path)
    comps = art["components"]
    yhat_sum = np.sum(np.column_stack([preds_by_component[c] for c in comps]), axis=1)
    if art["method"] == "hc":
        Xs = np.column_stack([preds_by_component[c] for c in comps])
        s = np.clip(art["scaler"].predict(Xs), 1e-6, np.inf)
        q = float(art["q"])
        lo, hi = yhat_sum - q * s, yhat_sum + q * s
    else:
        q = float(art["q"])
        lo, hi = yhat_sum - q, yhat_sum + q
    return yhat_sum, lo, hi
