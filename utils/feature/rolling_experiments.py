from __future__ import annotations

from typing import Iterable

import polars as pl


def add_experimental_rolling_features(
    df: pl.DataFrame,
    stats: Iterable[str],
    *,
    epsilon: float = 1e-3,
) -> pl.DataFrame:
    """
    Derive experimental rolling-window features to capture burst performance,
    exponential-style recency weighting, and directional trends.

    Parameters
    ----------
    df : pl.DataFrame
        Player-level frame that already contains rolling aggregates such as
        `1g_target_per_game` and `3g_target_per_game`.
    stats : Iterable[str]
        Base stat names that correspond to the rolling outputs from
        `add_rolling_features`.
    epsilon : float, default 1e-3
        Small constant to stabilise ratios.
    """

    exprs: list[pl.Expr] = []
    for stat in stats:
        col_1g = f"1g_{stat}_per_game"
        col_2g = f"2g_{stat}_per_game"
        col_3g = f"3g_{stat}_per_game"
        col_4g = f"4g_{stat}_per_game"

        if col_1g not in df.columns or col_3g not in df.columns:
            continue

        # Short-burst vs medium-term delta
        exprs.append(
            (pl.col(col_1g) - pl.col(col_3g))
            .alias(f"recent_delta_{stat}_per_game")
            .cast(pl.Float32)
        )

        # Ratio of short-burst to medium-term average
        exprs.append(
            (
                pl.col(col_1g)
                / (pl.col(col_3g).abs() + epsilon)
            )
            .alias(f"recent_ratio_{stat}_per_game")
            .cast(pl.Float32)
        )

        # Exponential-style decay blend favouring the most recent contest
        weight_exprs = [pl.col(col_1g) * 0.6]
        if col_2g in df.columns:
            weight_exprs.append(pl.col(col_2g) * 0.3)
        if col_3g in df.columns:
            weight_exprs.append(pl.col(col_3g) * 0.1)
        exprs.append(
            sum(weight_exprs)
            .alias(f"exp_decay_{stat}_per_game")
            .cast(pl.Float32)
        )

        # Trend proxy: comparing medium vs slightly longer rolling window
        if col_4g in df.columns:
            exprs.append(
                (pl.col(col_3g) - pl.col(col_4g))
                .alias(f"trend_{stat}_per_game")
                .cast(pl.Float32)
            )

    if not exprs:
        return df

    return df.with_columns(exprs)

