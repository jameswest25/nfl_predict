from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import polars as pl

from utils.feature.pace import compute_pace_metrics, add_pace_history

logger = logging.getLogger(__name__)


TEAM_REQUIRED_COLUMNS: set[str] = {
    "season",
    "week",
    "team",
    "opponent",
    "player_id",
    "pass_attempt",
    "completion",
    "carry",
    "target",
    "red_zone_target",
    "red_zone_carry",
    "goal_to_go_target",
    "goal_to_go_carry",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "passing_td",
    "rushing_td_count",
    "receiving_td_count",
    "touchdowns",
    "game_date",
}


def _safe_rate(numerator: str, denominator: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(0.0)
        .alias(alias)
    )


def _aggregate_context(
    base_df: pl.DataFrame,
    *,
    group_col: str,
    prefix: str,
    data_as_of_col: str,
) -> pl.DataFrame:
    """
    Aggregate offensive or defensive totals for a given grouping column.
    """
    df = base_df.filter(pl.col(group_col).is_not_null())
    
    totals = (
        df.group_by(["season", "week", group_col])
        .agg(
            [
                pl.col("game_date").max().alias("game_date"),
                pl.col("pass_attempt").sum().alias("pass_attempts"),
                pl.col("completion").sum().alias("completions"),
                pl.col("carry").sum().alias("carries"),
                pl.col("target").sum().alias("targets"),
                pl.col("red_zone_target").sum().alias("red_zone_targets"),
                pl.col("red_zone_carry").sum().alias("red_zone_carries"),
                pl.col("goal_to_go_target").sum().alias("goal_to_go_targets"),
                pl.col("goal_to_go_carry").sum().alias("goal_to_go_carries"),
                pl.col("passing_yards").sum().alias("passing_yards"),
                pl.col("rushing_yards").sum().alias("rushing_yards"),
                pl.col("receiving_yards").sum().alias("receiving_yards"),
                pl.col("passing_td").sum().alias("passing_tds"),
                pl.col("rushing_td_count").sum().alias("rushing_tds"),
                pl.col("receiving_td_count").sum().alias("receiving_tds"),
                pl.col("touchdowns").sum().alias("total_tds"),
                pl.col("player_id").n_unique().alias("skill_players_active"),
            ]
        )
        .rename({group_col: "team"})
        .with_columns(
            [
                (pl.col("pass_attempts") + pl.col("carries")).alias("offensive_plays"),
                (pl.col("passing_yards") + pl.col("rushing_yards")).alias("total_yards"),
                (
                    pl.col("red_zone_targets") + pl.col("red_zone_carries")
                ).alias("red_zone_plays"),
                (
                    pl.col("goal_to_go_targets") + pl.col("goal_to_go_carries")
                ).alias("goal_to_go_plays"),
            ]
        )
    )
    
    rate_metrics = {
        "pass_rate": ("pass_attempts", "offensive_plays"),
        "rush_rate": ("carries", "offensive_plays"),
        "target_per_play": ("targets", "offensive_plays"),
        "td_per_play": ("total_tds", "offensive_plays"),
        "yards_per_play": ("total_yards", "offensive_plays"),
        "red_zone_play_rate": ("red_zone_plays", "offensive_plays"),
        "goal_to_go_play_rate": ("goal_to_go_plays", "offensive_plays"),
        "pass_completion_rate": ("completions", "pass_attempts"),
        "pass_yards_per_attempt": ("passing_yards", "pass_attempts"),
        "rush_yards_per_carry": ("rushing_yards", "carries"),
    }
    
    for name, (num, denom) in rate_metrics.items():
        totals = totals.with_columns(
            _safe_rate(num, denom, name)
        )
    
    totals = totals.sort(["season", "team", "game_date"])
    if "game_date" in totals.columns:
        totals = totals.with_columns(
            pl.col("game_date")
            .shift(1)
            .over(["season", "team"])
            .alias(data_as_of_col)
        )
    
    feature_names: list[str] = []
    
    for metric in rate_metrics.keys():
        feature_names.extend(
            [
                f"{prefix}{metric}_prev",
                f"{prefix}{metric}_l3",
                f"{prefix}{metric}_l5",
            ]
        )
        totals = totals.with_columns(
            [
                pl.col(metric)
                .shift(1)
                .over(["season", "team"])
                .alias(f"{prefix}{metric}_prev"),
                pl.col(metric)
                .rolling_mean(window_size=3, min_periods=1)
                .shift(1)
                .over(["season", "team"])
                .alias(f"{prefix}{metric}_l3"),
                pl.col(metric)
                .rolling_mean(window_size=5, min_periods=1)
                .shift(1)
                .over(["season", "team"])
                .alias(f"{prefix}{metric}_l5"),
            ]
        )
    
    volume_metrics = [
        "pass_attempts",
        "carries",
        "targets",
        "red_zone_targets",
        "red_zone_carries",
        "goal_to_go_targets",
        "goal_to_go_carries",
        "offensive_plays",
        "total_yards",
        "total_tds",
        "skill_players_active",
    ]
    
    for metric in volume_metrics:
        feature_names.extend(
            [
                f"{prefix}{metric}_prev",
                f"{prefix}{metric}_l3",
            ]
        )
        totals = totals.with_columns(
            [
                pl.col(metric)
                .cast(pl.Float32)
                .shift(1)
                .over(["season", "team"])
                .alias(f"{prefix}{metric}_prev"),
                pl.col(metric)
                .cast(pl.Float32)
                .rolling_mean(window_size=3, min_periods=1)
                .shift(1)
                .over(["season", "team"])
                .alias(f"{prefix}{metric}_l3"),
            ]
        )
    
    keep_cols = ["season", "week", "team", "game_date", data_as_of_col] + feature_names
    totals = totals.select(keep_cols)
    totals = totals.with_columns(
        [pl.col(col).cast(pl.Float32).fill_null(0.0) for col in feature_names]
    )
    if "game_date" in totals.columns:
        totals = totals.with_columns(pl.col("game_date").cast(pl.Datetime("ms")))
    return totals


def compute_team_context_history(base_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute team-level offensive and defensive context metrics (prior games only).
    
    Returns a DataFrame keyed by (season, week, team, game_date) with engineered
    features prefixed `team_ctx_` (offense) and `opp_ctx_` (defense allowed).
    """
    missing = TEAM_REQUIRED_COLUMNS - set(base_df.columns)
    if missing:
        logger.warning(
            "Team context history skipped; missing columns: %s",
            ", ".join(sorted(missing)),
        )
        return pl.DataFrame(
            {
                "season": [],
                "week": [],
                "team": [],
                "game_date": [],
            }
        )

    usage_cols = [
        "pass_attempt",
        "completion",
        "carry",
        "target",
        "red_zone_target",
        "red_zone_carry",
        "goal_to_go_target",
        "goal_to_go_carry",
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "passing_td",
        "rushing_td_count",
        "receiving_td_count",
        "touchdowns",
        "offense_snaps",
    ]
    available_usage_cols = [col for col in usage_cols if col in base_df.columns]
    if available_usage_cols:
        usage_signal = pl.sum_horizontal(
            [pl.col(col).fill_null(0).abs() for col in available_usage_cols]
        )
        base_df = base_df.filter(usage_signal > 0)

    if base_df.is_empty():
        logger.warning("Team context history skipped; no non-zero usage rows available.")
        return pl.DataFrame(
            {
                "season": [],
                "week": [],
                "team": [],
                "game_date": [],
            }
        )

    offense_history = _aggregate_context(
        base_df,
        group_col="team",
        prefix="team_ctx_",
        data_as_of_col="data_as_of",
    )
    defense_history = _aggregate_context(
        base_df,
        group_col="opponent",
        prefix="opp_ctx_",
        data_as_of_col="opp_data_as_of",
    )
    
    join_keys = ["season", "week", "team", "game_date"]
    history = offense_history.join(
        defense_history,
        on=join_keys,
        how="outer",
    )
    if "data_as_of" in history.columns:
        history = history.with_columns(pl.col("data_as_of").cast(pl.Datetime("ms")))
    if "opp_data_as_of" in history.columns:
        history = history.with_columns(pl.col("opp_data_as_of").cast(pl.Datetime("ms")))
    if {"data_as_of", "opp_data_as_of"} <= set(history.columns):
        history = history.with_columns(
            pl.when(pl.col("data_as_of").is_null() & pl.col("opp_data_as_of").is_not_null())
            .then(pl.col("opp_data_as_of"))
            .otherwise(
                pl.when(
                    pl.col("data_as_of").is_not_null() & pl.col("opp_data_as_of").is_not_null()
                )
                .then(pl.min_horizontal([pl.col("data_as_of"), pl.col("opp_data_as_of")]))
                .otherwise(pl.col("data_as_of"))
            )
            .alias("data_as_of")
        ).drop("opp_data_as_of")
    elif "opp_data_as_of" in history.columns:
        history = history.rename({"opp_data_as_of": "data_as_of"})
    
    team_cols = [col for col in history.columns if col.startswith("team_ctx_")]
    opp_cols = [col for col in history.columns if col.startswith("opp_ctx_")]
    if team_cols:
        history = history.with_columns([pl.col(col).fill_null(0.0) for col in team_cols])
    if opp_cols:
        history = history.with_columns([pl.col(col).fill_null(0.0) for col in opp_cols])
    
    # ------------------------------------------------------------------
    # Add pace metrics (no-huddle rate, PROE, neutral pass rate)
    # ------------------------------------------------------------------
    pace_history = pl.DataFrame()
    seasons_available: list[int] = []
    try:
        if "season" in base_df.columns:
            seasons_available = (
                base_df.get_column("season")
                .drop_nulls()
                .unique()
                .to_list()
            )
    except Exception:
        seasons_available = []

    if seasons_available:
        try:
            pace_metrics = compute_pace_metrics(seasons_available)
            pace_history = add_pace_history(pace_metrics)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to compute pace history: %s", exc)
            pace_history = pl.DataFrame()

    if not pace_history.is_empty():
        pace_history = pace_history.drop("game_date", strict=False)
        pace_history = pace_history.with_columns(
            [
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                pl.col("team").cast(pl.Utf8),
            ]
        )
        history = history.with_columns(
            [
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                pl.col("team").cast(pl.Utf8),
            ]
        )
        history = history.join(
            pace_history,
            on=["season", "week", "team"],
            how="left",
        )
        pace_cols = [
            col
            for col in pace_history.columns
            if col not in {"season", "week", "team"}
        ]
        if pace_cols:
            history = history.with_columns(
                [pl.col(col).cast(pl.Float32).fill_null(0.0) for col in pace_cols]
            )
    else:
        logger.info("Pace history unavailable; pace features skipped.")
    
    if "game_date" in history.columns:
        history = history.with_columns(pl.col("game_date").cast(pl.Datetime("ms")))
    
    history = history.sort(["season", "team", "game_date"])
    return history


def attach_team_context(
    df: pl.DataFrame,
    team_history: pl.DataFrame,
    *,
    join_on_date: bool,
    cutoff_column: str | None = None,
) -> pl.DataFrame:
    if team_history.is_empty():
        return df

    # 1. Identify feature groups
    team_ctx_cols = [col for col in team_history.columns if col.startswith("team_ctx_")]
    opp_ctx_cols = [col for col in team_history.columns if col.startswith("opp_ctx_")]
    pace_cols = [col for col in team_history.columns if col.startswith("team_pace_")]

    target_ts_col = cutoff_column if cutoff_column and cutoff_column in df.columns else "game_date"

    # 2. Attach TEAM-side features (Team's Offense + Team's Defense + Team's Pace)
    # We want:
    # - team_ctx_* (Team Offense)
    # - opp_ctx_*  (Team Defense) -> This is usually what we want for "Team Context"
    # - team_pace_* (Team Pace)
    
    # Note: Existing logic only selected team_ctx_ for the first join?
    # Let's check the original code.
    # Original: team_feature_names = [starts with team_ctx_]
    #           opp_feature_names = [starts with opp_ctx_]
    #           team_join selected team_feature_names.
    #           opp_join selected opp_feature_names.
    
    # WAIT. "opp_join" in the original code renames {team: opponent}.
    # So it joins `team_history` rows matching the *opponent*.
    # If we pick `opp_ctx_` from the Opponent's row, that is "Opponent's Defense".
    # Usually "opp_ctx_" features in the final matrix mean "My Opponent's Defense".
    
    # Let's verify the convention.
    # If I am BAL. My opponent is KC.
    # I want features describing KC's defense.
    # KC's row in team_history has `opp_ctx_` (KC allowed stats).
    # So yes, we want `opp_ctx_` from the Opponent's row.
    
    # But what about `team_ctx_`?
    # `team_join` joins on `team` (BAL). It selects `team_ctx_` (BAL Offense).
    
    # So:
    # - team_join (on Team): gets `team_ctx_` (My Offense), `team_pace_` (My Pace)
    # - opp_join (on Opponent): gets `opp_ctx_` (Opponent Defense), `team_pace_` -> renamed `opp_pace_` (Opponent Pace)
    
    # 3. Build lists
    team_features_to_pull = team_ctx_cols + pace_cols
    opp_features_to_pull = opp_ctx_cols
    
    # Additional logic for Pace on opponent side
    # We want to pull `team_pace_` from opponent row, but rename to `opp_pace_`
    
    # --- TEAM JOIN ---
    if join_on_date:
        team_join = (
            team_history
            .drop_nulls("data_as_of")
            .select(["team", "data_as_of", *team_features_to_pull])
            .rename({"data_as_of": "__team_ctx_data_as_of"})
            .sort(["team", "__team_ctx_data_as_of"])
        )
        df_out = df.with_columns(
            pl.col(target_ts_col).cast(pl.Datetime("ms")).alias("__team_ctx_target_ts")
            if target_ts_col in df.columns
            else pl.col("game_date").cast(pl.Datetime("ms")).alias("__team_ctx_target_ts")
        ).join_asof(
            team_join,
            left_on="__team_ctx_target_ts",
            right_on="__team_ctx_data_as_of",
            by="team",
            strategy="backward",
        ).rename(
            {
                "__team_ctx_data_as_of": "team_ctx_data_as_of",
            }
        ).drop("__team_ctx_target_ts")
    else:
        df_out = df.join(
            team_history.select(["season", "week", "team", "data_as_of", *team_features_to_pull]),
            on=["season", "week", "team"],
            how="left",
        ).rename({"data_as_of": "team_ctx_data_as_of"})

    # --- OPPONENT JOIN ---
    # We fetch `opp_ctx_` (their defense) AND `team_pace_` (their pace -> aliased to `opp_pace_`)
    
    if opp_features_to_pull or pace_cols:
        # Prepare selection and renaming map
        sel_cols = ["team", "data_as_of"] + opp_features_to_pull + pace_cols
        
        # Rename map:
        # team -> opponent
        # data_as_of -> ...
        # opp_ctx_* -> opp_ctx_* (keep as is, it already means "defense stats")
        # team_pace_* -> opp_pace_*
        
        rename_map = {"team": "opponent"}
        if join_on_date:
            rename_map["data_as_of"] = "__opp_ctx_data_as_of"
        else:
            rename_map["data_as_of"] = "opp_ctx_data_as_of"
            
        for p in pace_cols:
            rename_map[p] = p.replace("team_pace_", "opp_pace_")
            
        if join_on_date:
            opp_join = (
                team_history
                .drop_nulls("data_as_of")
                .select(sel_cols)
                .rename(rename_map)
                .sort(["opponent", "__opp_ctx_data_as_of"])
            )
            df_out = df_out.with_columns(
                pl.col(target_ts_col).cast(pl.Datetime("ms")).alias("__opp_ctx_target_ts")
                if target_ts_col in df_out.columns
                else pl.col("game_date").cast(pl.Datetime("ms")).alias("__opp_ctx_target_ts")
            ).join_asof(
                opp_join,
                left_on="__opp_ctx_target_ts",
                right_on="__opp_ctx_data_as_of",
                by="opponent",
                strategy="backward",
            ).rename(
                {"__opp_ctx_data_as_of": "opp_ctx_data_as_of"}
            ).drop("__opp_ctx_target_ts")
        else:
            df_out = df_out.join(
                team_history
                .select(["season", "week", *sel_cols])
                .rename(rename_map),
                on=["season", "week", "opponent"],
                how="left",
            )
        
        # Fill nulls
        # The join produces opp_ctx_* and opp_pace_* columns
        new_cols = opp_features_to_pull + [rename_map[p] for p in pace_cols]
        df_out = df_out.with_columns(
            [pl.col(col).cast(pl.Float32).fill_null(0.0) for col in new_cols if col in df_out.columns]
        )

    cast_asof_cols = []
    if "team_ctx_data_as_of" in df_out.columns:
        cast_asof_cols.append(pl.col("team_ctx_data_as_of").cast(pl.Datetime("ms")))
    if "opp_ctx_data_as_of" in df_out.columns:
        cast_asof_cols.append(pl.col("opp_ctx_data_as_of").cast(pl.Datetime("ms")))
    if cast_asof_cols:
        df_out = df_out.with_columns(cast_asof_cols)

    if join_on_date and target_ts_col in df_out.columns:
        if "team_ctx_data_as_of" in df_out.columns:
            potential = (
                df_out
                .filter(
                    pl.col("team_ctx_data_as_of").is_not_null()
                    & pl.col(target_ts_col).is_not_null()
                    & (pl.col("team_ctx_data_as_of") >= pl.col(target_ts_col))
                )
                .height
            )
            if potential > 0:
                logger.debug(
                    "team_ctx_data_as_of >= target_ts detected for %d rows; filtering already enforced.",
                    potential,
                )
        if "opp_ctx_data_as_of" in df_out.columns:
            potential = (
                df_out
                .filter(
                    pl.col("opp_ctx_data_as_of").is_not_null()
                    & pl.col(target_ts_col).is_not_null()
                    & (pl.col("opp_ctx_data_as_of") >= pl.col(target_ts_col))
                )
                .height
            )
            if potential > 0:
                logger.debug(
                    "opp_ctx_data_as_of >= target_ts detected for %d rows; filtering already enforced.",
                    potential,
                )

    return df_out


def add_team_context_features(
    df: pl.DataFrame,
    *,
    join_on_date: bool = False,
    history: pl.DataFrame | None = None,
    return_history: bool = False,
    cutoff_column: str | None = None,
) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]:
    """
    Convenience wrapper that computes (if needed) and attaches team context features.

    Parameters
    ----------
    df : pl.DataFrame
        Player-game level data to enrich.
    join_on_date : bool, default False
        Whether to use an as-of join on game_date (for predictions).
    history : pl.DataFrame | None
        Optional pre-computed team history. When provided the costly aggregation is skipped.
    return_history : bool, default False
        If True, also return the team history DataFrame for caching.
    """
    team_history = history if history is not None else compute_team_context_history(df)
    df_out = attach_team_context(
        df,
        team_history,
        join_on_date=join_on_date,
        cutoff_column=cutoff_column,
    )

    if return_history:
        return df_out, team_history
    return df_out

