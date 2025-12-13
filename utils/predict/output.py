"""Prediction output formatting and composed feature injection.

This module handles computing composed prediction features and
formatting the final prediction output.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PREDICTION_DIR = Path("output/predictions")
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

# Meta columns to include in output
META_COLUMNS = [
    "player_id", "player_name", "position", "team", "opponent",
    "season", "week", "game_id", "game_date",
    "depth_chart_position", "depth_chart_order",
    "injury_game_designation", "injury_report_status", "injury_practice_status",
    "injury_is_listed",
    "is_home", "game_start_hour_utc", "game_day_of_week", "season_type",
    "stadium_key", "stadium_name", "stadium_tz", "roof", "surface",
    "home_team", "away_team",
    "spread_line", "total_line", "team_implied_total", "opp_implied_total",
]

# Prefixes for columns to include in output
OUTPUT_COLUMN_PREFIXES = (
    "pred_", "expected_", "hist_", "ps_", "snap_",
    "team_ctx_", "opp_ctx_", "opp_def_", "market_", "travel_",
    "injury_", "drive_hist_",
)

OUTPUT_COLUMN_WHITELIST = [
    "spread_line", "team_implied_total",
    "season", "week", "game_id", "position",
]


def inject_composed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composed prediction features.
    
    These are derived from the sequential model predictions and
    create expected value features for the final prediction.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with model predictions
    
    Returns
    -------
    pd.DataFrame
        DataFrame with composed features added
    """
    if df is None or df.empty:
        return df

    # Ensure availability composites
    _ensure_availability_composites(df)

    # Helper for safe multiplication
    def _safe_mul(a: str, b: str, out_col: str):
        if a in df.columns and b in df.columns:
            df[out_col] = df[a] * df[b]

    def _select_column(candidates: list[str]) -> str | None:
        for name in candidates:
            if name in df.columns:
                return name
        return None

    # Availability x usage → expected opportunities
    _safe_mul("pred_availability", "pred_usage_targets", "expected_targets")
    _safe_mul("pred_availability_raw", "pred_usage_targets", "expected_targets_raw")
    _safe_mul("pred_availability", "pred_usage_carries", "expected_carries")
    _safe_mul("pred_availability_raw", "pred_usage_carries", "expected_carries_raw")

    # Team pace based expectations
    if "pred_team_pace" in df.columns:
        team_pace = pd.to_numeric(df["pred_team_pace"], errors="coerce").clip(lower=0.0)
        df["expected_team_plays"] = team_pace
        
        pass_rate_col = _select_column([
            "team_ctx_pass_rate_prev", "team_ctx_pass_rate_l3", "team_ctx_pass_rate_l5",
        ])
        if pass_rate_col:
            pass_rate = (
                pd.to_numeric(df[pass_rate_col], errors="coerce")
                .fillna(0.5).clip(lower=0.0, upper=1.0)
            )
            df["expected_team_pass_plays"] = team_pace * pass_rate
            df["expected_team_rush_plays"] = team_pace * (1.0 - pass_rate)

    # Combined opportunities
    if "expected_targets" in df.columns and "expected_carries" in df.columns:
        df["expected_opportunities"] = df["expected_targets"] + df["expected_carries"]
    if "expected_targets_raw" in df.columns and "expected_carries_raw" in df.columns:
        df["expected_opportunities_raw"] = df["expected_targets_raw"] + df["expected_carries_raw"]

    # TD signal from efficiency
    if "pred_efficiency_tds" in df.columns and "expected_opportunities" in df.columns:
        df["expected_td_signal"] = df["pred_efficiency_tds"] * df["expected_opportunities"]
    if "pred_efficiency_tds" in df.columns and "expected_opportunities_raw" in df.columns:
        df["expected_td_signal_raw"] = df["pred_efficiency_tds"] * df["expected_opportunities_raw"]

    # Red zone expectations
    _compute_red_zone_expectations(df, _select_column)

    # Yard expectations
    if "pred_efficiency_rec_yards" in df.columns:
        df["expected_receiving_yards"] = (
            pd.to_numeric(df["pred_efficiency_rec_yards"], errors="coerce").clip(lower=0.0)
        )
    if "pred_efficiency_rush_yards" in df.columns:
        df["expected_rushing_yards"] = (
            pd.to_numeric(df["pred_efficiency_rush_yards"], errors="coerce").clip(lower=0.0)
        )
    if "expected_receiving_yards" in df.columns and "expected_rushing_yards" in df.columns:
        df["expected_total_yards"] = df["expected_receiving_yards"] + df["expected_rushing_yards"]

    # Poisson TD probabilities
    if "expected_td_signal" in df.columns:
        df["expected_td_prob_poisson"] = 1.0 - np.exp(-df["expected_td_signal"].clip(lower=0.0))
    if "expected_td_signal_raw" in df.columns:
        df["expected_td_prob_poisson_raw"] = 1.0 - np.exp(-df["expected_td_signal_raw"].clip(lower=0.0))
    if "expected_rz_td_signal" in df.columns:
        df["expected_rz_td_prob_poisson"] = 1.0 - np.exp(-df["expected_rz_td_signal"].clip(lower=0.0))
    if "expected_rz_td_signal_raw" in df.columns:
        df["expected_rz_td_prob_poisson_raw"] = 1.0 - np.exp(-df["expected_rz_td_signal_raw"].clip(lower=0.0))

    # Structural TD expectations
    _compute_structural_td_expectations(df)

    return df


def _ensure_availability_composites(frame: pd.DataFrame) -> None:
    """Ensure availability composite features are computed."""
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
                .fillna(0.0).clip(0.0, 1.0)
            )
            frame["pred_availability_raw"] = (raw_active * share).astype(np.float32)
        else:
            frame["pred_availability_raw"] = combined


def _compute_red_zone_expectations(df: pd.DataFrame, _select_column) -> None:
    """Compute red zone and goal-to-go expectation features."""
    rz_target_share = _select_column([
        "hist_red_zone_target_share_prev", "hist_red_zone_target_share_l3",
        "red_zone_target_share_prev", "red_zone_target_share_l3",
    ])
    rz_carry_share = _select_column([
        "hist_red_zone_carry_share_prev", "hist_red_zone_carry_share_l3",
        "red_zone_carry_share_prev", "red_zone_carry_share_l3",
    ])
    team_rz_rate = _select_column([
        "team_red_zone_play_rate_prev", "team_red_zone_play_rate_l3",
    ])
    team_gtg_rate = _select_column([
        "team_goal_to_go_play_rate_prev", "team_goal_to_go_play_rate_l3",
    ])

    def _team_factor(rate_col: str | None, out_col: str) -> None:
        if rate_col and rate_col in df.columns:
            df[out_col] = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    _team_factor(team_rz_rate, "team_red_zone_rate")
    _team_factor(team_gtg_rate, "team_goal_to_go_rate")

    def _expected_red_zone(base_col: str, share_col: str | None, out_col: str):
        if base_col in df.columns and share_col:
            share = pd.to_numeric(df[share_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            df[out_col] = df[base_col] * share

    _expected_red_zone("expected_targets", rz_target_share, "expected_rz_targets")
    _expected_red_zone("expected_targets_raw", rz_target_share, "expected_rz_targets_raw")
    _expected_red_zone("expected_carries", rz_carry_share, "expected_rz_carries")
    _expected_red_zone("expected_carries_raw", rz_carry_share, "expected_rz_carries_raw")

    if "expected_opportunities" in df.columns and "team_red_zone_rate" in df.columns:
        df["team_based_rz_opportunities"] = df["expected_opportunities"] * df["team_red_zone_rate"]
    if "expected_opportunities_raw" in df.columns and "team_red_zone_rate" in df.columns:
        df["team_based_rz_opportunities_raw"] = df["expected_opportunities_raw"] * df["team_red_zone_rate"]

    if "expected_rz_targets" in df.columns and "expected_rz_carries" in df.columns:
        df["expected_rz_opportunities"] = df["expected_rz_targets"] + df["expected_rz_carries"]
    if "expected_rz_targets_raw" in df.columns and "expected_rz_carries_raw" in df.columns:
        df["expected_rz_opportunities_raw"] = df["expected_rz_targets_raw"] + df["expected_rz_carries_raw"]

    if "pred_efficiency_tds" in df.columns and "expected_rz_opportunities" in df.columns:
        df["expected_rz_td_signal"] = df["pred_efficiency_tds"] * df["expected_rz_opportunities"]
    if "pred_efficiency_tds" in df.columns and "expected_rz_opportunities_raw" in df.columns:
        df["expected_rz_td_signal_raw"] = df["pred_efficiency_tds"] * df["expected_rz_opportunities_raw"]


def _compute_structural_td_expectations(df: pd.DataFrame) -> None:
    """Compute structural TD expectations for anytime_td_structured model."""
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


def format_output(
    features: pd.DataFrame,
    final_prob_col: str,
    threshold: float,
    include_debug_prediction_columns: bool = False,
) -> pd.DataFrame:
    """Format the final prediction output.
    
    Parameters
    ----------
    features : pd.DataFrame
        Full feature dataframe with predictions
    final_prob_col : str
        Column name containing final probability predictions
    threshold : float
        Decision threshold for binary predictions
    
    Returns
    -------
    pd.DataFrame
        Formatted output dataframe
    """
    from utils.predict.inference import apply_guards_inline

    def _guarded_prob(col_name: str) -> np.ndarray:
        if col_name not in features.columns:
            return np.full(len(features), np.nan, dtype=np.float64)
        vals = pd.to_numeric(features[col_name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        vals = apply_guards_inline(features, vals)
        return np.clip(vals, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # OUTPUT POLICY (default = "consumer-friendly")
    #
    # The user wants ONE probability column per "model flavor":
    # - flat:          pred_anytime_td
    # - structured:    pred_anytime_td_structured_global
    # - structured-moe pred_anytime_td_structured_moe
    #
    # We keep these near player/team and avoid dumping all pred_* columns by
    # default. Debug columns are still available behind a flag.
    # -------------------------------------------------------------------------
    prob_flat = _guarded_prob("pred_anytime_td")

    structured_global_col = (
        "pred_anytime_td_structured_global"
        if "pred_anytime_td_structured_global" in features.columns
        else "pred_anytime_td_structured"
    )
    prob_structured_global = _guarded_prob(structured_global_col)
    prob_structured_moe = _guarded_prob("pred_anytime_td_structured_moe")

    # Decision is based on the configured final_prob_col (typically structured-global).
    proba_for_decision = _guarded_prob(final_prob_col)
    picks = proba_for_decision >= float(threshold)

    meta_cols = [c for c in META_COLUMNS if c in features.columns]
    out = features[meta_cols].copy()

    out["prob_anytime_td_flat"] = prob_flat
    out["prob_anytime_td_global"] = prob_structured_global
    out["prob_anytime_td_moe"] = prob_structured_moe

    out["prediction"] = picks.astype(int)
    out["model_threshold"] = float(threshold)
    # implied odds for the primary published probability (structured global)
    out["implied_decimal_odds"] = np.where(
        out["prob_anytime_td_global"] > 0, 1.0 / out["prob_anytime_td_global"], np.nan
    )

    if include_debug_prediction_columns:
        extra_cols = [
            col for col in features.columns
            if col not in out.columns
            and (
                col in OUTPUT_COLUMN_WHITELIST
                or any(col.startswith(prefix) for prefix in OUTPUT_COLUMN_PREFIXES)
            )
        ]
        if extra_cols:
            out = pd.concat([out, features[extra_cols]], axis=1)

    # Re-order: keep the three prob columns close to player/team
    preferred_order = [
        "player_name",
        "team",
        "position",
        "opponent",
        "game_date",
        "week",
        "season",
        "game_id",
        "prob_anytime_td_flat",
        "prob_anytime_td_global",
        "prob_anytime_td_moe",
        "implied_decimal_odds",
        "prediction",
        "model_threshold",
    ]
    ordered = [c for c in preferred_order if c in out.columns]
    remaining = [c for c in out.columns if c not in ordered]
    out = out[ordered + remaining]

    # Sort by the structured-global probability (the main published signal)
    out.sort_values(
        ["prob_anytime_td_global", "game_date", "team", "player_name"],
        ascending=[False, True, True, True],
        inplace=True,
    )

    return out


def save_predictions(
    df: pd.DataFrame,
    start_date,
    end_date,
    output_name: str | None = None,
) -> Path:
    """Save predictions to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prediction output dataframe
    start_date : date
        Start date for predictions
    end_date : date
        End date for predictions
    output_name : str | None
        Optional output filename
    
    Returns
    -------
    Path
        Path to saved file
    """
    if output_name is None:
        output_name = f"anytime_td_predictions_{start_date.isoformat()}_{end_date.isoformat()}.csv"
    
    output_path = PREDICTION_DIR / output_name
    df.to_csv(output_path, index=False)
    
    positive_rate = df["prediction"].mean() * 100.0
    logger.info(f"Wrote {len(df)} rows → {output_path}")
    logger.info(f"Positive predictions: {df['prediction'].sum()} ({positive_rate:.2f}% of slate)")
    
    return output_path
