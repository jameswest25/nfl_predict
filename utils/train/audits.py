from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TdConvAuditResult:
    label_col: str
    opp_col: str
    td_col: str
    n_rows: int
    n_with_opp: int
    mae: float
    p95: float
    p99: float
    max_abs_err: float
    frac_err_gt_quarter_td: float
    frac_err_gt_half_td: float


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def audit_td_conv_labels(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    raise_on_fail: bool = True,
    key_cols: Iterable[str] = ("season", "week", "game_id", "player_id", "team", "position"),
) -> dict[str, Any]:
    """Sanity-check td_conv labels match Poisson math assumptions.

    Required semantics:
      td_per_target_label ~= receiving_td_count / target
      td_per_carry_label  ~= rushing_td_count / carry

    This audit checks that label * opp approximately reconstructs the raw TD counts.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    audits: list[TdConvAuditResult] = []
    payload: dict[str, Any] = {
        "ok": True,
        "audits": [],
        "top_outliers": {},
    }

    specs = [
        ("td_per_target_label", "target", "receiving_td_count"),
        ("td_per_carry_label", "carry", "rushing_td_count"),
    ]

    for label_col, opp_col, td_col in specs:
        if not {label_col, opp_col, td_col}.issubset(df.columns):
            payload["audits"].append(
                {
                    "label_col": label_col,
                    "opp_col": opp_col,
                    "td_col": td_col,
                    "skipped": True,
                    "reason": "missing required columns",
                }
            )
            continue

        frame = df[[label_col, opp_col, td_col, *[c for c in key_cols if c in df.columns]]].copy()
        opp = pd.to_numeric(frame[opp_col], errors="coerce").fillna(0.0)
        label = pd.to_numeric(frame[label_col], errors="coerce")
        td = pd.to_numeric(frame[td_col], errors="coerce").fillna(0.0)

        with_opp = opp > 0
        sub = frame.loc[with_opp].copy()
        sub_opp = opp.loc[with_opp]
        sub_label = label.loc[with_opp]
        sub_td = td.loc[with_opp]

        # If label is null for rows with opportunities, that's incompatible with the math.
        n_bad_null = int(sub_label.isna().sum())
        if n_bad_null > 0:
            payload["ok"] = False
            payload["audits"].append(
                {
                    "label_col": label_col,
                    "opp_col": opp_col,
                    "td_col": td_col,
                    "skipped": False,
                    "error": f"{n_bad_null} rows have opp>0 but null label",
                }
            )
            continue

        implied = (sub_label.astype(float) * sub_opp.astype(float)).astype(float)
        abs_err = (implied - sub_td.astype(float)).abs()

        # Summary stats
        n_rows = int(len(frame))
        n_with_opp = int(with_opp.sum())
        mae = float(abs_err.mean()) if n_with_opp else float("nan")
        p95 = float(abs_err.quantile(0.95)) if n_with_opp else float("nan")
        p99 = float(abs_err.quantile(0.99)) if n_with_opp else float("nan")
        max_abs_err = float(abs_err.max()) if n_with_opp else float("nan")
        frac_q = float((abs_err > 0.25).mean()) if n_with_opp else float("nan")
        frac_h = float((abs_err > 0.50).mean()) if n_with_opp else float("nan")

        audits.append(
            TdConvAuditResult(
                label_col=label_col,
                opp_col=opp_col,
                td_col=td_col,
                n_rows=n_rows,
                n_with_opp=n_with_opp,
                mae=mae,
                p95=p95,
                p99=p99,
                max_abs_err=max_abs_err,
                frac_err_gt_quarter_td=frac_q,
                frac_err_gt_half_td=frac_h,
            )
        )

        # Outliers for debugging
        sub = sub.assign(
            _opp=sub_opp.to_numpy(),
            _label=sub_label.to_numpy(),
            _td=sub_td.to_numpy(),
            _implied=implied.to_numpy(),
            _abs_err=abs_err.to_numpy(),
        )
        top = sub.sort_values("_abs_err", ascending=False).head(25)
        payload["top_outliers"][label_col] = top.to_dict(orient="records")

        payload["audits"].append(
            {
                "label_col": label_col,
                "opp_col": opp_col,
                "td_col": td_col,
                "skipped": False,
                "n_rows": n_rows,
                "n_with_opp": n_with_opp,
                "mae": mae,
                "p95": p95,
                "p99": p99,
                "max_abs_err": max_abs_err,
                "frac_err_gt_quarter_td": frac_q,
                "frac_err_gt_half_td": frac_h,
            }
        )

    # Fail policy: allow rare outliers (e.g., float roundtrip or exotic label generation),
    # but fail if we see a material mismatch rate.
    for a in audits:
        if not math.isfinite(a.mae):
            continue
        if a.frac_err_gt_quarter_td > 0.01 or a.frac_err_gt_half_td > 0.001:
            payload["ok"] = False

    path = out_dir / "td_conv_label_audit.json"
    path.write_text(json.dumps(payload, indent=2, default=_safe_float))
    if raise_on_fail and not payload["ok"]:
        raise ValueError(f"td_conv label sanity audit failed (see {path}).")
    return payload


def summarize_chain_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Create a small chain-level diagnostics summary (safe to log/write)."""
    cols = [
        # Stage-gating flags (0/1) to explain which rows were skipped
        "gate_availability_active_global",
        "gate_snaps_global",
        "gate_usage_targets_global",
        "gate_usage_carries_global",
        "gate_usage_target_yards_global",
        "gate_efficiency_rec_yards_air_global",
        "gate_efficiency_rec_yards_yac_global",
        "gate_efficiency_rush_yards_global",
        "gate_td_conv_rec_global",
        "gate_td_conv_rush_global",
        "gate_anytime_td_structured_global",
        "gate_availability_active_moe",
        "gate_snaps_moe",
        "gate_usage_targets_moe",
        "gate_usage_carries_moe",
        "gate_usage_target_yards_moe",
        "gate_efficiency_rec_yards_air_moe",
        "gate_efficiency_rec_yards_yac_moe",
        "gate_efficiency_rush_yards_moe",
        "gate_td_conv_rec_moe",
        "gate_td_conv_rush_moe",
        "gate_anytime_td_structured_moe",
        "expected_targets",
        "expected_carries",
        "expected_opportunities",
        "expected_td_count_structural",
        "expected_td_prob_structural",
        "pred_td_conv_rec",
        "pred_td_conv_rush",
        "pred_anytime_td_structured",
        "pred_anytime_td_structured_moe",
        "pred_anytime_td",
        "prob_anytime_td_global",
        "prob_anytime_td_moe",
        "prob_anytime_td_flat",
    ]
    out: dict[str, Any] = {"columns": {}, "row_count": int(len(df))}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        out["columns"][c] = {
            "non_null": int(s.notna().sum()),
            "mean": float(s.mean()) if s.notna().any() else None,
            "p10": float(s.quantile(0.10)) if s.notna().any() else None,
            "p50": float(s.quantile(0.50)) if s.notna().any() else None,
            "p90": float(s.quantile(0.90)) if s.notna().any() else None,
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
        }

    # Explicit gate summary for quick human inspection
    gate_cols = [c for c in df.columns if str(c).startswith("gate_")]
    gate_summary: dict[str, Any] = {}
    for c in sorted(gate_cols):
        s = pd.to_numeric(df[c], errors="coerce")
        total = int(len(df))
        passed = int((s.fillna(0.0) > 0.0).sum())
        gate_summary[c] = {
            "passed": passed,
            "total": total,
            "pass_rate": float(passed / total) if total else None,
        }
    out["gate_summary"] = gate_summary
    return out


def audit_horizon_knownness(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    raise_on_fail: bool = False,
) -> dict[str, Any]:
    """Audit 'known at cutoff' semantics using snapshot timestamps.

    This does not mutate the dataset; it writes a coverage report so you can see
    whether the current horizon is starving key feeds (injury/roster/odds/weather).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    required = "decision_cutoff_ts"
    if required not in df.columns:
        payload = {"ok": False, "error": "decision_cutoff_ts missing"}
        (out_dir / "horizon_knownness_audit.json").write_text(json.dumps(payload, indent=2))
        if raise_on_fail:
            raise ValueError("horizon knownness audit failed: decision_cutoff_ts missing")
        return payload

    cutoff = pd.to_datetime(df["decision_cutoff_ts"], errors="coerce", utc=True)
    payload: dict[str, Any] = {"ok": True, "row_count": int(len(df)), "feeds": {}}

    feeds = [
        ("injury", "injury_snapshot_ts"),
        ("roster", "roster_snapshot_ts"),
        ("odds", "odds_snapshot_ts"),
        ("forecast", "forecast_snapshot_ts"),
        ("roster_status", "roster_status_snapshot_ts"),
    ]
    for name, col in feeds:
        if col not in df.columns:
            payload["feeds"][name] = {"present": False}
            continue
        ts = pd.to_datetime(df[col], errors="coerce", utc=True)
        present = ts.notna().sum()
        known = (ts.notna()) & (cutoff.notna()) & (ts <= cutoff)
        payload["feeds"][name] = {
            "present": True,
            "non_null": int(present),
            "known_at_cutoff": int(known.sum()),
            "known_rate": float(known.mean()) if len(df) else None,
            "violations_after_cutoff": int(((ts.notna()) & (cutoff.notna()) & (ts > cutoff)).sum()),
        }

    path = out_dir / "horizon_knownness_audit.json"
    path.write_text(json.dumps(payload, indent=2))
    return payload

