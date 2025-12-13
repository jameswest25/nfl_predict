"""Injury enrichment functions for player-game level aggregation.

These functions handle injury-related feature computation including
historical injury rates and injury availability model predictions.
"""

from __future__ import annotations

import logging

import polars as pl

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None

from utils.train.injury_availability import (
    load_latest_artifact,
    predict_probabilities as predict_injury_probabilities,
    MODEL_PROB_COL,
    PROB_LOW_COL,
    PROB_HIGH_COL,
)

logger = logging.getLogger(__name__)

INJURY_MODEL_FALLBACK_INTERVAL_THRESHOLD = 0.65


def compute_injury_history_rates(df: pl.DataFrame) -> pl.DataFrame:
    """Compute historical inactive rates for players, depth chart slots, and practice patterns."""
    if df.is_empty():
        return df.with_columns(
            [
                pl.lit(None).cast(pl.Float32).alias("injury_player_inactive_rate_prior"),
                pl.lit(None).cast(pl.Float32).alias("injury_depth_slot_inactive_rate_prior"),
                pl.lit(None).cast(pl.Float32).alias("injury_practice_pattern_inactive_rate_prior"),
            ]
        )

    if pd is None or np is None:
        logger.debug("Pandas/numpy unavailable; skipping injury history rate computation.")
        return df.with_columns(
            [
                pl.lit(None).cast(pl.Float32).alias("injury_player_inactive_rate_prior"),
                pl.lit(None).cast(pl.Float32).alias("injury_depth_slot_inactive_rate_prior"),
                pl.lit(None).cast(pl.Float32).alias("injury_practice_pattern_inactive_rate_prior"),
            ]
        )

    feature_cols = [
        "season",
        "week",
        "player_id",
        "team",
        "depth_chart_position",
        "injury_practice_status_sequence",
        "injury_is_inactive_designation",
    ]
    available_cols = [col for col in feature_cols if col in df.columns]
    pdf = df.select(available_cols).to_pandas().copy()
    pdf["_row_id"] = np.arange(len(pdf))
    pdf["season"] = pd.to_numeric(pdf.get("season"), errors="coerce").fillna(0).astype(int)
    pdf["week"] = pd.to_numeric(pdf.get("week"), errors="coerce").fillna(0).astype(int)
    pdf["injury_is_inactive_designation"] = (
        pd.to_numeric(pdf.get("injury_is_inactive_designation"), errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    pdf = pdf.sort_values(["season", "week", "_row_id"]).reset_index(drop=True)

    if "player_id" in pdf.columns:
        pdf["_player_count"] = pdf.groupby("player_id", dropna=False).cumcount()
        pdf["_player_cum"] = (
            pdf.groupby("player_id", dropna=False)["injury_is_inactive_designation"].cumsum()
            - pdf["injury_is_inactive_designation"]
        )
        pdf["injury_player_inactive_rate_prior"] = np.where(
            pdf["_player_count"] > 0,
            pdf["_player_cum"] / pdf["_player_count"].replace(0, np.nan),
            np.nan,
        )
    else:
        pdf["injury_player_inactive_rate_prior"] = np.nan

    if {"team", "depth_chart_position"} <= set(pdf.columns):
        pdf["_depth_key"] = (
            pdf["team"].fillna("UNKNOWN").astype(str)
            + "|"
            + pdf["depth_chart_position"].fillna("UNKNOWN").astype(str)
        )
        pdf["_depth_count"] = pdf.groupby("_depth_key", dropna=False).cumcount()
        pdf["_depth_cum"] = (
            pdf.groupby("_depth_key", dropna=False)["injury_is_inactive_designation"].cumsum()
            - pdf["injury_is_inactive_designation"]
        )
        pdf["injury_depth_slot_inactive_rate_prior"] = np.where(
            pdf["_depth_count"] > 0,
            pdf["_depth_cum"] / pdf["_depth_count"].replace(0, np.nan),
            np.nan,
        )
    else:
        pdf["injury_depth_slot_inactive_rate_prior"] = np.nan

    if "injury_practice_status_sequence" in pdf.columns:
        pdf["_pattern_key"] = pdf["injury_practice_status_sequence"].fillna("UNKNOWN").astype(str)
        pdf["_pattern_count"] = pdf.groupby("_pattern_key", dropna=False).cumcount()
        pdf["_pattern_cum"] = (
            pdf.groupby("_pattern_key", dropna=False)["injury_is_inactive_designation"].cumsum()
            - pdf["injury_is_inactive_designation"]
        )
        pdf["injury_practice_pattern_inactive_rate_prior"] = np.where(
            pdf["_pattern_count"] > 0,
            pdf["_pattern_cum"] / pdf["_pattern_count"].replace(0, np.nan),
            np.nan,
        )
    else:
        pdf["injury_practice_pattern_inactive_rate_prior"] = np.nan

    pdf = pdf.sort_values("_row_id")
    result = pl.DataFrame(
        {
            "injury_player_inactive_rate_prior": pdf["injury_player_inactive_rate_prior"].astype("float32"),
            "injury_depth_slot_inactive_rate_prior": pdf["injury_depth_slot_inactive_rate_prior"].astype("float32"),
            "injury_practice_pattern_inactive_rate_prior": pdf["injury_practice_pattern_inactive_rate_prior"].astype("float32"),
        }
    )
    df = df.with_columns(
        [
            result["injury_player_inactive_rate_prior"],
            result["injury_depth_slot_inactive_rate_prior"],
            result["injury_practice_pattern_inactive_rate_prior"],
        ]
    )
    return df


def apply_injury_availability_model(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the pre-trained injury availability model and merge predictions."""
    artifact = load_latest_artifact()
    if artifact is None:
        logger.warning("Injury availability model artifact not found; retaining heuristic probabilities.")
        return df

    missing_cols = [col for col in artifact.feature_columns if col not in df.columns]
    if missing_cols:
        logger.warning(
            "Injury model missing %d required columns (e.g. %s); predictions may rely on null defaults.",
            len(missing_cols),
            ", ".join(missing_cols[:5]),
        )

    try:
        preds = predict_injury_probabilities(df, artifact)
    except Exception as exc:  # pragma: no cover - inference failure
        logger.warning("Failed to apply injury availability model: %s", exc)
        return df

    width_col = "injury_inactive_probability_interval_width"
    fallback_count = preds.filter(pl.col(width_col) > INJURY_MODEL_FALLBACK_INTERVAL_THRESHOLD).height
    total_rows = preds.height or 1
    if fallback_count:
        logger.info(
            "Injury model interval width exceeded %.2f for %d players (%.2f%%); using heuristic fallback.",
            INJURY_MODEL_FALLBACK_INTERVAL_THRESHOLD,
            fallback_count,
            fallback_count / total_rows * 100.0,
        )

    for col in preds.columns:
        if col in df.columns:
            df = df.drop(col)
    df = df.hstack(preds)

    fallback_expr = pl.col(width_col) > INJURY_MODEL_FALLBACK_INTERVAL_THRESHOLD
    heuristic_prob = pl.col("injury_inactive_probability")

    df = df.with_columns(
        pl.when(fallback_expr)
        .then(pl.lit("heuristic"))
        .otherwise(pl.lit("model"))
        .alias("injury_inactive_probability_source")
    )

    df = df.with_columns(
        pl.when(fallback_expr)
        .then(heuristic_prob)
        .otherwise(pl.col(MODEL_PROB_COL))
        .alias("injury_inactive_probability")
    )

    df = df.with_columns(
        [
            pl.when(fallback_expr)
            .then(pl.col("injury_inactive_probability"))
            .otherwise(pl.col(PROB_LOW_COL))
            .alias(PROB_LOW_COL),
            pl.when(fallback_expr)
            .then(pl.col("injury_inactive_probability"))
            .otherwise(pl.col(PROB_HIGH_COL))
            .alias(PROB_HIGH_COL),
        ]
    )

    return df


# Default injury column values
INJURY_DEFAULTS: dict[str, object] = {
    "injury_report_status": "UNKNOWN",
    "injury_practice_status": "UNKNOWN",
    "injury_report_primary": "UNKNOWN",
    "injury_practice_primary": "UNKNOWN",
    "injury_practice_status_day1": "UNKNOWN",
    "injury_practice_status_day2": "UNKNOWN",
    "injury_practice_status_day3": "UNKNOWN",
    "injury_practice_status_sequence": "UNKNOWN",
    "injury_report_status_sequence": "UNKNOWN",
    "injury_game_designation": "UNKNOWN",
    "injury_is_inactive_designation": 0.0,
    "injury_inactive_probability": 0.1,
    "injury_inactive_probability_model": None,
    "injury_practice_report_count": 0.0,
    "injury_report_count": 0.0,
    "injury_practice_dnp_last3": 0.0,
    "injury_practice_limited_last3": 0.0,
    "injury_practice_full_last3": 0.0,
    "injury_hours_since_last_report": None,
    "injury_hours_until_game_at_last_report": None,
    "injury_hours_between_last_reports": None,
    "roster_hours_since_last_game": None,
    "rest_days_since_last_game": None,
    "roster_depth_chart_order_delta": None,
    "roster_status_changed": 0.0,
    "injury_player_inactive_rate_prior": None,
    "injury_depth_slot_inactive_rate_prior": None,
    "injury_practice_pattern_inactive_rate_prior": None,
    "injury_snapshot_valid": 0,
    "injury_transaction_days_since": None,
    "injury_last_transaction_note": "UNKNOWN",
}

# Numeric injury columns that should be cast to Float32
INJURY_NUMERIC_COLS = [
    "injury_practice_dnp_count",
    "injury_practice_limited_count",
    "injury_practice_full_count",
    "injury_is_listed",
    "injury_is_inactive_designation",
    "injury_inactive_probability",
    "injury_practice_report_count",
    "injury_report_count",
    "injury_practice_dnp_last3",
    "injury_practice_limited_last3",
    "injury_practice_full_last3",
    "injury_hours_since_last_report",
    "injury_hours_until_game_at_last_report",
    "injury_hours_between_last_reports",
    "roster_hours_since_last_game",
    "rest_days_since_last_game",
    "roster_depth_chart_order_delta",
    "roster_status_changed",
    "injury_player_inactive_rate_prior",
    "injury_depth_slot_inactive_rate_prior",
    "injury_practice_pattern_inactive_rate_prior",
    "injury_inactive_probability_model",
    "injury_snapshot_valid",
    "injury_transaction_days_since",
]

# String injury columns that should be filled with "UNKNOWN"
INJURY_STRING_COLS = [
    "injury_report_status",
    "injury_practice_status",
    "injury_report_primary",
    "injury_practice_primary",
    "injury_practice_status_day1",
    "injury_practice_status_day2",
    "injury_practice_status_day3",
    "injury_practice_status_sequence",
    "injury_report_status_sequence",
    "injury_game_designation",
    "injury_last_transaction_note",
]


def fill_missing_injury_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Fill missing injury columns with default values."""
    missing_cols = [col for col in INJURY_DEFAULTS if col not in df.columns]
    if missing_cols:
        df = df.with_columns([
            pl.lit(INJURY_DEFAULTS[col]).alias(col) for col in missing_cols
        ])

    # Cast numeric columns
    for col in INJURY_NUMERIC_COLS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float32).alias(col))
        else:
            df = df.with_columns(pl.col(col).cast(pl.Float32))

    if "injury_snapshot_valid" in df.columns:
        df = df.with_columns(pl.col("injury_snapshot_valid").cast(pl.Int8))

    return df


def fill_injury_string_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Fill null injury string columns with 'UNKNOWN'."""
    df = df.with_columns(
        [
            pl.col(col).fill_null("UNKNOWN").cast(pl.Utf8).alias(col)
            for col in INJURY_STRING_COLS
            if col in df.columns
        ]
    )

    if "injury_game_designation" in df.columns:
        df = df.with_columns(
            pl.col("injury_game_designation").cast(pl.Utf8).alias("injury_game_designation")
        )

    return df


def compute_roster_deltas(df: pl.DataFrame) -> pl.DataFrame:
    """Compute roster-related delta features (rest days, depth chart changes)."""
    sort_keys = [k for k in ["player_id", "season", "week"] if k in df.columns]
    if sort_keys:
        df = df.sort(sort_keys)

        if "game_start_utc" in df.columns:
            df = df.with_columns(
                (pl.col("game_start_utc") - pl.col("game_start_utc").shift(1).over("player_id"))
                .dt.total_minutes()
                .alias("roster_minutes_since_last_game")
            )
            df = df.with_columns(
                (pl.col("roster_minutes_since_last_game") / 60.0).cast(pl.Float32).alias("roster_hours_since_last_game")
            )
            df = df.with_columns(
                (pl.col("roster_hours_since_last_game") / 24.0)
                .cast(pl.Float32)
                .alias("rest_days_since_last_game")
            )
        else:
            df = df.with_columns(
                [
                    pl.lit(None).cast(pl.Float32).alias("roster_hours_since_last_game"),
                    pl.lit(None).cast(pl.Float32).alias("rest_days_since_last_game"),
                ]
            )

        if "depth_chart_order" in df.columns:
            df = df.with_columns(
                (
                    pl.col("depth_chart_order").cast(pl.Float32)
                    - pl.col("depth_chart_order").cast(pl.Float32).shift(1).over("player_id")
                ).alias("roster_depth_chart_order_delta")
            )
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Float32).alias("roster_depth_chart_order_delta"))

        df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("roster_status_changed"))

    else:
        df = df.with_columns(
            [
                pl.lit(None).cast(pl.Float32).alias("roster_hours_since_last_game"),
                pl.lit(None).cast(pl.Float32).alias("rest_days_since_last_game"),
                pl.lit(None).cast(pl.Float32).alias("roster_depth_chart_order_delta"),
                pl.lit(0).cast(pl.Int8).alias("roster_status_changed"),
            ]
        )

    return df

