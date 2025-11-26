"""Label semantics and versioning for NFL player-level targets.

This module centralizes anytime-TD label definitions so that:
- Semantics are explicit and versioned.
- Training can request a specific label column per problem.
- Feature generation can emit multiple label variants for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping

import polars as pl


@dataclass(frozen=True)
class LabelRules:
    """Rules controlling which TD components count toward a label."""

    include_rush: bool = True
    include_rec: bool = True
    include_pass_thrown: bool = True  # credit passers for thrown TDs
    include_other: bool = False  # fumble returns, defensive/special teams if present


@dataclass(frozen=True)
class LabelSpec:
    """Container describing a label version."""

    name: str
    primary: str
    labels: Mapping[str, LabelRules]
    aliases: Mapping[str, str] = field(default_factory=dict)

    @property
    def all_columns(self) -> set[str]:
        cols = set(self.labels.keys())
        cols.update(self.aliases.keys())
        return cols


DEFAULT_LABEL_VERSION = "v1_any_offense"


LABEL_VERSIONS: Dict[str, LabelSpec] = {
    "v1_any_offense": LabelSpec(
        name="v1_any_offense",
        primary="anytime_td_offense",
        labels={
            "anytime_td_offense": LabelRules(include_other=False),
            "anytime_td_rush": LabelRules(include_rush=True, include_rec=False, include_pass_thrown=False, include_other=False),
            "anytime_td_rec": LabelRules(include_rush=False, include_rec=True, include_pass_thrown=False, include_other=False),
            "anytime_td_pass_thrown": LabelRules(include_rush=False, include_rec=False, include_pass_thrown=True, include_other=False),
            "anytime_td_all": LabelRules(include_other=True),
        },
        aliases={
            # Maintain legacy column name as an alias to the offense-only label.
            "anytime_td": "anytime_td_offense",
            # Keep td_count pointing to the offense definition for backward compatibility.
            "td_count": "td_count_offense",
        },
    ),
    "v1_any_all": LabelSpec(
        name="v1_any_all",
        primary="anytime_td_all",
        labels={
            "anytime_td_all": LabelRules(include_other=True),
            "anytime_td_offense": LabelRules(include_other=False),
            "anytime_td_rush": LabelRules(include_rush=True, include_rec=False, include_pass_thrown=False, include_other=False),
            "anytime_td_rec": LabelRules(include_rush=False, include_rec=True, include_pass_thrown=False, include_other=False),
            "anytime_td_pass_thrown": LabelRules(include_rush=False, include_rec=False, include_pass_thrown=True, include_other=False),
        },
        aliases={
            "anytime_td": "anytime_td_all",
            "td_count": "td_count_all",
        },
    ),
}


def get_label_spec(version: str | None) -> LabelSpec:
    """Return a concrete LabelSpec for the requested version (or default)."""

    version = version or DEFAULT_LABEL_VERSION
    if version not in LABEL_VERSIONS:
        raise ValueError(
            f"Unknown label_version '{version}'. "
            f"Available: {sorted(LABEL_VERSIONS.keys())}"
        )
    return LABEL_VERSIONS[version]


def _sum_expr(exprs: Iterable[pl.Expr]) -> pl.Expr:
    exprs = list(exprs)
    if not exprs:
        return pl.lit(0).cast(pl.Int64)
    acc = exprs[0]
    for expr in exprs[1:]:
        acc = acc + expr
    return acc


def compute_td_labels(df: pl.DataFrame, *, version: str | None = None) -> pl.DataFrame:
    """Compute TD label columns according to a label version.

    Parameters
    ----------
    df : pl.DataFrame
        Player-game frame containing touchdown components.
    version : str | None
        Label version identifier. Defaults to DEFAULT_LABEL_VERSION.
    """

    spec = get_label_spec(version)
    required_cols = ["rushing_td_count", "receiving_td_count", "passing_td", "touchdowns"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(0).cast(pl.Int64).alias(col) for col in missing])
    rush = pl.col("rushing_td_count").fill_null(0).cast(pl.Int64)
    rec = pl.col("receiving_td_count").fill_null(0).cast(pl.Int64)
    passing = pl.col("passing_td").fill_null(0).cast(pl.Int64)
    reported_total = pl.col("touchdowns").fill_null(0).cast(pl.Int64)

    offense_total = _sum_expr([rush, rec, passing]).alias("_td_offense_total")
    other_total = (reported_total - offense_total).clip_min(0).alias("_td_other_total")
    all_total = (offense_total + other_total).alias("_td_all_total")

    totals_exprs = [
        offense_total,
        other_total,
        all_total,
    ]

    label_exprs: list[pl.Expr] = [
        pl.col("_td_offense_total").alias("td_count_offense"),
        pl.col("_td_all_total").alias("td_count_all"),
    ]

    for label_name, rules in spec.labels.items():
        parts: list[pl.Expr] = []
        if rules.include_rush:
            parts.append(rush)
        if rules.include_rec:
            parts.append(rec)
        if rules.include_pass_thrown:
            parts.append(passing)
        if rules.include_other:
            parts.append(pl.col("_td_other_total"))
        total_expr = _sum_expr(parts)
        label_exprs.append((total_expr > 0).cast(pl.Int8).alias(label_name))

    # Canonical counts for compatibility
    if "td_count" not in df.columns:
        # Fill td_count using the alias mapping; default to offense count.
        target_alias = spec.aliases.get("td_count") or "td_count_offense"
        label_exprs.append(pl.col(target_alias).alias("td_count"))

    df = df.with_columns(totals_exprs).with_columns(label_exprs)

    # Resolve aliases (e.g., anytime_td -> anytime_td_offense)
    if spec.aliases:
        alias_exprs = []
        for alias_name, source_name in spec.aliases.items():
            if source_name in df.columns:
                alias_exprs.append(pl.col(source_name).alias(alias_name))
        if alias_exprs:
            df = df.with_columns(alias_exprs)

    drop_cols = [col for col in ("_td_offense_total", "_td_other_total", "_td_all_total") if col in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)
    return df
