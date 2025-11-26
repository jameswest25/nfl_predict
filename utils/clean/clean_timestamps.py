"""Utility functions for cleaning and imputing timestamp columns.

This module contains helpers used by the main cleaning pipeline to
repair missing `utc_ts` values that occasionally appear in Statcast
pitch-by-pitch data.  The strategy is entirely *in-memory* – it does not
attempt to reach out to the MLB API because we already have more than
enough temporal granularity within each individual game.

The algorithm is simple:
1.  Work **within each game** (identified by `game_pk`).
2.  Sort the pitches by their natural order: (`at_bat_number`,
    `pitch_number`).
3.  For every pitch that is missing `utc_ts`, look at the **nearest**
   previous and next pitches *in the same game* that have non-null
   timestamps.
4.  If timestamps exist on **both** sides, take the arithmetic mean of
   the two neighbouring timestamps.
5.  If only one neighbour exists (e.g. the first or last pitch of a
   game), fall back to that single value.
6.  If neither neighbour exists (entire game missing timestamps) leave
   the value as NA – this case is extremely rare and will be handled by
   later auditing steps.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

__all__ = ["fill_missing_utc_ts"]


def _impute_timestamp(prev: pd.Timestamp | None, nxt: pd.Timestamp | None) -> pd.Timestamp | None:
    """Return an imputed timestamp using neighbouring values.

    * If **both** ``prev`` and ``nxt`` are valid → midpoint between the two.
    * If only ``prev`` is valid         → ``prev`` + 10 ms.
    * If only ``nxt`` is valid          → ``nxt``  − 10 ms.
    * If neither are valid              → ``None``.
    """
    if pd.isna(prev) and pd.isna(nxt):
        return None
    if pd.isna(nxt):  # only prev exists
        return prev + pd.Timedelta(milliseconds=10) if not pd.isna(prev) else None
    if pd.isna(prev):  # only next exists
        return nxt - pd.Timedelta(milliseconds=10)

    # Both neighbours present – take midpoint in nanoseconds for accuracy
    ns1 = prev.value  # type: ignore[attr-defined]
    ns2 = nxt.value   # type: ignore[attr-defined]
    mid_ns = (ns1 + ns2) // 2
    return pd.to_datetime(mid_ns, utc=True)


def fill_missing_utc_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``utc_ts`` values using neighbouring pitches.

    Parameters
    ----------
    df:
        The full Statcast dataframe.  The caller must ensure that the
        columns ``game_pk``, ``at_bat_number``, ``pitch_number`` and
        ``utc_ts`` exist.  Missing columns will cause the original frame
        to be returned unmodified.

    Returns
    -------
    pd.DataFrame
        A dataframe where as many missing ``utc_ts`` values as possible
        have been imputed.
    """
    required_cols = {"game_pk", "at_bat_number", "pitch_number", "utc_ts"}
    if not required_cols.issubset(df.columns):
        logging.warning(
            "Cannot fill utc_ts – required columns %s are missing.",
            required_cols - set(df.columns),
        )
        return df

    # Fast exit if nothing to do
    if df["utc_ts"].notna().all():
        return df

    # Work on a *copy* to avoid mutating caller's frame in-place.
    df = df.copy()
    df["utc_ts"] = pd.to_datetime(df["utc_ts"], errors="coerce", utc=True)

    filled_count = 0

    def _fill_group(group: pd.DataFrame) -> pd.DataFrame:  # noqa: D401 – simple description
        """Impute missing timestamps within a *single* game."""
        nonlocal filled_count

        # Ensure natural pitch ordering within the game
        group = group.sort_values(
            ["at_bat_number", "pitch_number"], ascending=[True, True], na_position="last"
        )

        # Pre-compute forward/backward fills so we don't repeatedly scan
        prev_ts = group["utc_ts"].ffill()
        next_ts = group["utc_ts"].bfill()

        na_mask = group["utc_ts"].isna()
        if not na_mask.any():
            return group

        # Compute new timestamps for every row; originals retained for non-na rows
        new_ts = [
            _impute_timestamp(prev_ts.iloc[i], next_ts.iloc[i]) if na else orig
            for i, (na, orig) in enumerate(zip(na_mask, group["utc_ts"]))
        ]

        # Count how many previously missing values are now filled (i.e., result is not NA)
        imputed_series = pd.Series(new_ts, index=group.index)
        filled_in_group = (na_mask & imputed_series.notna()).sum()
        filled_count += int(filled_in_group)

        # Assign only to rows that were NA so that dtypes are preserved
        group = group.copy()
        group.loc[na_mask, "utc_ts"] = imputed_series.loc[na_mask]
        return group

    df = df.groupby("game_pk", group_keys=False).apply(_fill_group)

    logging.info("Filled %d missing utc_ts values via neighbour averaging", filled_count)
    return df
