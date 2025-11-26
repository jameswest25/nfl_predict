"""Leak guardrails for feature matrices.

This module centralizes pattern-based leak detection so both feature assembly
and training can fail fast when post-outcome columns sneak into the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import Iterable, Mapping, Sequence

import polars as pl

from utils.general.constants import IDENTIFIER_COLUMNS, LEAK_PRONE_COLUMNS, NFL_TARGET_COLUMNS

try:  # Optional dependency when imported from training (pandas path)
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None


@dataclass(frozen=True)
class LeakPolicy:
    version: str
    banned_exact: set[str]
    banned_prefixes: tuple[str, ...]
    banned_suffixes: tuple[str, ...]
    banned_regex: tuple[tuple[re.Pattern[str], str], ...]
    description: str = ""


@dataclass
class LeakCheckResult:
    policy_version: str
    banned: dict[str, str]
    kept: list[str]
    dropped: list[str]
    allowlist_kept: list[str]


# Core policy tuned for anytime-TD style problems (can be extended per-problem).
_BASE_BANNED_EXACT = set(LEAK_PRONE_COLUMNS) | {
    "final_score",
    "final_total",
    "points_scored",
    "actual_td_count",
    "actual_anytime_td",
    "postgame_anytime_td",
    "postgame_td_count",
}
_BASE_BANNED_PREFIXES = (
    "postgame_",
    "post_game_",
    "post_",
    "live_",
    "actual_",
    "realized_",
    "final_",
    "postmatch_",
)
_BASE_BANNED_SUFFIXES = (
    "_post",
    "_postgame",
    "_final",
    "_fullgame",
    "_actual",
)
_BASE_BANNED_REGEX: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r".*post.?game.*", flags=re.IGNORECASE), "postgame marker"),
    (re.compile(r".*final_score.*", flags=re.IGNORECASE), "final score"),
    (re.compile(r".*actual.*td.*", flags=re.IGNORECASE), "actual TD total"),
    (re.compile(r".*anytime_td.*(post|final|actual).*", flags=re.IGNORECASE), "post-decision anytime TD"),
    (re.compile(r".*fullgame.*", flags=re.IGNORECASE), "full game cumulative value"),
)

DEFAULT_LEAK_POLICY = LeakPolicy(
    version="v1",
    description="Core post-outcome leak guard for NFL anytime TD features.",
    banned_exact=_BASE_BANNED_EXACT,
    banned_prefixes=_BASE_BANNED_PREFIXES,
    banned_suffixes=_BASE_BANNED_SUFFIXES,
    banned_regex=_BASE_BANNED_REGEX,
)


def _ensure_sequence(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def evaluate_leak_prone_columns(
    columns: Iterable[str],
    *,
    policy: LeakPolicy = DEFAULT_LEAK_POLICY,
) -> LeakCheckResult:
    """Identify leak-prone columns using the configured policy."""

    banned: dict[str, str] = {}
    columns_list = list(columns)
    for col in columns_list:
        if col in policy.banned_exact:
            banned[col] = "exact"
            continue
        for prefix in policy.banned_prefixes:
            if col.startswith(prefix):
                banned[col] = f"prefix:{prefix}"
                break
        if col in banned:
            continue
        for suffix in policy.banned_suffixes:
            if col.endswith(suffix):
                banned[col] = f"suffix:{suffix}"
                break
        if col in banned:
            continue
        for pattern, reason in policy.banned_regex:
            if pattern.match(col):
                banned[col] = reason
                break

    kept = [c for c in columns_list if c not in banned]
    return LeakCheckResult(
        policy_version=policy.version,
        banned=banned,
        kept=kept,
        dropped=[],
        allowlist_kept=[],
    )


def enforce_leak_guard(
    df: pl.DataFrame | "pd.DataFrame",
    *,
    policy: LeakPolicy = DEFAULT_LEAK_POLICY,
    allow_prefixes: Sequence[str] | str | None = None,
    allow_exact: Sequence[str] | str | None = None,
    drop_banned: bool = True,
    drop_non_allowlisted: bool = False,
    raise_on_banned: bool = True,
) -> tuple[pl.DataFrame | "pd.DataFrame", LeakCheckResult]:
    """Apply leak guardrails to a feature frame.

    Parameters
    ----------
    df : DataFrame
        Polars or pandas DataFrame.
    policy : LeakPolicy
        Leak policy to apply.
    allow_prefixes : Sequence[str] | str | None
        Optional prefixes that define the allowed feature surface for a problem.
    allow_exact : Sequence[str] | str | None
        Explicitly allowed columns (e.g., identifiers).
    drop_banned : bool
        If True, drop any banned columns automatically.
    drop_non_allowlisted : bool
        If True, drop columns that are not covered by allow_prefixes/allow_exact/identifiers/targets.
    raise_on_banned : bool
        If True, raise when banned columns are detected (after optional drop).
    """

    result = evaluate_leak_prone_columns(df.columns, policy=policy)
    dropped: list[str] = []

    allow_prefixes = _ensure_sequence(allow_prefixes)
    allow_exact = set(_ensure_sequence(allow_exact))
    allow_exact.update({col for col in IDENTIFIER_COLUMNS if col in df.columns})
    allow_exact.update({col for col in NFL_TARGET_COLUMNS if col in df.columns})

    # Permit allowlisted columns even if they match a banned rule (e.g., labels)
    banned = {col: reason for col, reason in result.banned.items() if col not in allow_exact}

    if banned and drop_banned:
        df = df.drop(list(banned.keys()))
        dropped.extend(banned.keys())

    if banned and raise_on_banned:
        raise ValueError(f"Leak-prone columns detected: {sorted(banned.items())}")

    allowlist_cols: set[str] = set()
    if allow_prefixes:
        allowlist_cols.update({c for c in df.columns if c.startswith(allow_prefixes)})
    allowlist_cols.update({c for c in allow_exact if c in df.columns})

    if drop_non_allowlisted and allowlist_cols:
        drop_candidates = [c for c in df.columns if c not in allowlist_cols]
        if drop_candidates:
            df = df.drop(drop_candidates)
            dropped.extend(drop_candidates)

    if drop_non_allowlisted and not allowlist_cols:
        raise ValueError("No allowlisted columns remain after applying leak guard allowlist.")

    # Store adjusted banned info on the result
    result.banned = banned
    result.dropped = dropped
    result.kept = [c for c in df.columns]
    result.allowlist_kept = sorted(list(allowlist_cols)) if allowlist_cols else []
    return df, result


def build_schema_snapshot(
    df: pl.DataFrame | "pd.DataFrame",
    *,
    policy: LeakPolicy = DEFAULT_LEAK_POLICY,
    metadata: Mapping[str, object] | None = None,
    banned: Mapping[str, str] | None = None,
) -> dict:
    """Render a schema snapshot describing each column and whether it was flagged."""

    banned = banned or {}
    meta = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "policy_version": policy.version,
    }
    meta.update(metadata or {})

    # Extract columns / dtypes
    if isinstance(df, pl.DataFrame):
        cols_and_dtypes = [(col, str(dtype)) for col, dtype in zip(df.columns, df.dtypes)]
    else:  # pandas or pandas-like
        cols_and_dtypes = [(col, str(dtype)) for col, dtype in zip(df.columns, df.dtypes)]

    schema = {
        "metadata": meta,
        "columns": [],
    }

    for col, dtype in cols_and_dtypes:
        tags: list[str] = []
        if col in IDENTIFIER_COLUMNS:
            tags.append("id")
        if col in NFL_TARGET_COLUMNS:
            tags.append("label")
        if col.startswith("market_"):
            tags.append("odds")
        if col.startswith("injury_") or col.startswith("practice_"):
            tags.append("injury")
        if col.startswith("weather_"):
            tags.append("weather")
        if col.startswith("ps_"):
            tags.append("pre_snap")
        schema["columns"].append(
            {
                "name": col,
                "dtype": dtype,
                "banned": col in banned,
                "tags": tags,
            }
        )
    return schema


def write_schema_snapshot(path, schema: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(schema, fp, indent=2, sort_keys=True)


__all__ = [
    "DEFAULT_LEAK_POLICY",
    "LeakPolicy",
    "LeakCheckResult",
    "evaluate_leak_prone_columns",
    "enforce_leak_guard",
    "build_schema_snapshot",
    "write_schema_snapshot",
]
