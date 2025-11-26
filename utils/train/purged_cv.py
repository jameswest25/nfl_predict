"""Time-aware, purged group CV helpers to prevent temporal leakage."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _determine_test_group_size(
    n_groups: int,
    n_splits: int,
    max_test_group_size: int | float,
    explicit: int | None,
) -> int:
    """Choose a reasonable test group size given constraints."""
    if explicit is not None:
        size = int(explicit)
        if size <= 0:
            raise ValueError("test_group_size must be positive when provided.")
        if size > n_groups:
            raise ValueError("test_group_size cannot exceed number of groups.")
        return min(size, int(max_test_group_size) if np.isfinite(max_test_group_size) else size)

    denom = max(n_splits + 1, 2)
    size = max(1, n_groups // denom)
    if np.isfinite(max_test_group_size):
        size = min(size, int(max_test_group_size))
    return size


def compute_purged_group_splits(
    df: pd.DataFrame,
    *,
    time_col: str,
    group_col: str,
    n_splits: int,
    purge_days: float = 0.0,
    purge_groups: int = 0,
    embargo_days: float = 0.0,
    max_train_group_size: int | float = np.inf,
    max_test_group_size: int | float = np.inf,
    test_group_size: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build tail-heavy, time-ordered splits with purge/embargo windows.

    Groups are ordered by their earliest timestamp. Train groups come strictly
    before the validation block, then are purged by both a group-gap and a
    time-based window to respect rolling feature lookbacks.
    """

    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' missing from frame.")
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' missing from frame.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for CV.")

    meta = df[[time_col, group_col]].copy()
    meta[time_col] = pd.to_datetime(meta[time_col], errors="coerce")
    meta = meta.dropna(subset=[time_col, group_col])

    if meta.empty:
        return []

    meta = meta.sort_values([time_col, group_col]).reset_index()
    meta.rename(columns={"index": "_orig_idx"}, inplace=True)

    group_stats = (
        meta.groupby(group_col)[time_col]
        .agg(["min", "max"])
        .rename(columns={"min": "t_min", "max": "t_max"})
        .sort_values("t_min")
    )
    group_order = group_stats.index.to_list()
    n_groups = len(group_order)
    if n_groups < n_splits + 1:
        raise ValueError(f"Insufficient distinct groups ({n_groups}) for n_splits={n_splits}.")

    group_to_indices = {
        grp: meta.loc[meta[group_col] == grp, "_orig_idx"].to_numpy()
        for grp in group_order
    }

    t_min = group_stats["t_min"]
    t_max = group_stats["t_max"]

    test_size = _determine_test_group_size(
        n_groups=n_groups,
        n_splits=n_splits,
        max_test_group_size=max_test_group_size,
        explicit=test_group_size,
    )
    test_starts = list(range(n_groups - n_splits * test_size, n_groups, test_size))

    purge_delta = pd.Timedelta(days=float(purge_days)) if purge_days else pd.Timedelta(0)
    embargo_delta = pd.Timedelta(days=float(embargo_days)) if embargo_days else pd.Timedelta(0)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for start in test_starts:
        test_groups = group_order[start : start + test_size]
        if not test_groups:
            continue

        earliest_test = t_min.loc[test_groups].min()
        latest_test = t_max.loc[test_groups].max()

        candidate_train = group_order[:start]
        if purge_groups > 0:
            candidate_train = candidate_train[:-int(purge_groups)] or []

        # Enforce time-based purge and embargo (train strictly before test).
        keep_train: list[str] = []
        purge_cutoff = earliest_test - purge_delta
        embargo_cutoff = latest_test + embargo_delta
        for grp in candidate_train:
            if t_max.loc[grp] >= purge_cutoff:
                continue
            if t_max.loc[grp] >= earliest_test:
                continue
            if embargo_delta and t_min.loc[grp] <= embargo_cutoff:
                # This branch mainly protects against oddly-ordered frames; with tail splits
                # train groups should already precede the test window.
                continue
            keep_train.append(grp)

        if np.isfinite(max_train_group_size):
            keep_train = keep_train[-int(max_train_group_size) :]

        if not keep_train:
            continue

        train_idx_parts = [group_to_indices[g] for g in keep_train if g in group_to_indices]
        test_idx_parts = [group_to_indices[g] for g in test_groups if g in group_to_indices]
        if not train_idx_parts or not test_idx_parts:
            continue

        train_idx = np.sort(np.concatenate(train_idx_parts))
        test_idx = np.sort(np.concatenate(test_idx_parts))
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))

    return splits

