import polars as pl

def compute_drive_level_aggregates(drive_df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate raw drive data to game-level summaries for each player-game.
    Mirrors logic from pipeline/feature.py but exposed for prediction pipeline.
    """
    drive_schema_names = set(drive_df.columns)
    
    agg_exprs: list[pl.Expr] = [
        pl.count().alias("drive_count"),
        pl.col("game_date").max().alias("game_date"),
        pl.when(
            (
                pl.col("carry").fill_null(0)
                + pl.col("target").fill_null(0)
                + pl.col("pass_attempt").fill_null(0)
                + pl.col("reception").fill_null(0)
            )
            > 0
        )
        .then(1)
        .otherwise(0)
        .sum()
        .cast(pl.Float32)
        .alias("drive_touch_drives"),
        pl.when(pl.col("touchdown").fill_null(0) > 0)
        .then(1)
        .otherwise(0)
        .sum()
        .cast(pl.Float32)
        .alias("drive_td_drives"),
        (
            pl.col("passing_yards").fill_null(0)
            + pl.col("rushing_yards").fill_null(0)
            + pl.col("receiving_yards").fill_null(0)
        )
        .sum()
        .cast(pl.Float32)
        .alias("drive_total_yards"),
    ]

    if {"red_zone_carry", "red_zone_target"} <= drive_schema_names:
        agg_exprs.append(
            pl.when(
                (
                    pl.col("red_zone_carry").fill_null(0)
                    + pl.col("red_zone_target").fill_null(0)
                )
                > 0
            )
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Float32)
            .alias("drive_red_zone_drives")
        )

    if {"goal_to_go_carry", "goal_to_go_target"} <= drive_schema_names:
        agg_exprs.append(
            pl.when(
                (
                    pl.col("goal_to_go_carry").fill_null(0)
                    + pl.col("goal_to_go_target").fill_null(0)
                )
                > 0
            )
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.Float32)
            .alias("drive_goal_to_go_drives")
        )

    # Group by player-game keys
    # Note: We aggregate by game_id, but need to keep season/week for downstream joins if present
    group_cols = ["game_id", "team", "player_id"]
    for c in ["season", "week"]:
        if c in drive_schema_names:
            group_cols.append(c)
            
    return (
        drive_df
        .group_by(group_cols)
        .agg(agg_exprs)
    )

def finalize_drive_history_features(drive_features: pl.DataFrame) -> pl.DataFrame:
    """Apply shared smoothing/lag logic to drive-level aggregates."""
    if drive_features.is_empty():
        return drive_features

    drive_features = drive_features.with_columns(
        [
            pl.col("game_date").cast(pl.Datetime("ms")),
            pl.col("drive_count").cast(pl.Float32),
        ]
    )
    drive_features = drive_features.sort(["player_id", "team", "game_date"])

    group_cols = ["player_id", "team"]
    numeric_cols = ["drive_count", "drive_touch_drives", "drive_td_drives", "drive_total_yards"]
    if "drive_red_zone_drives" in drive_features.columns:
        numeric_cols.append("drive_red_zone_drives")
    if "drive_goal_to_go_drives" in drive_features.columns:
        numeric_cols.append("drive_goal_to_go_drives")

    drive_features = drive_features.with_columns(
        [
            pl.col(col).cast(pl.Float32)
            for col in numeric_cols
            if col != "drive_count"
        ]
    )

    drive_features = drive_features.with_columns(
        pl.col("game_date").shift(1).over(group_cols).alias("data_as_of")
    )
    cumulative_cols = [f"{col}_cumulative" for col in numeric_cols]
    drive_features = drive_features.with_columns(
        [
            pl.col(col).cum_sum().over(group_cols).alias(f"{col}_cumulative")
            for col in numeric_cols
        ]
    )

    def _prev_name(col: str) -> str:
        return f"drive_hist_{col.removeprefix('drive_')}_prev"

    def _l3_name(col: str) -> str:
        return f"drive_hist_{col.removeprefix('drive_')}_l3"

    drive_features = drive_features.with_columns(
        [
            pl.col(f"{col}_cumulative")
            .shift(1)
            .over(group_cols)
            .fill_null(0.0)
            .alias(_prev_name(col))
            for col in numeric_cols
        ]
    )
    drive_features = drive_features.with_columns(
        [
            pl.col(col)
            .rolling_sum(window_size=3, min_periods=1)
            .shift(1)
            .over(group_cols)
            .fill_null(0.0)
            .alias(_l3_name(col))
            for col in numeric_cols
        ]
    )
    drive_features = drive_features.drop(cumulative_cols)

    rate_specs = [
        ("drive_touch_drives", "drive_count", "touch_rate"),
        ("drive_td_drives", "drive_count", "td_rate"),
        ("drive_total_yards", "drive_count", "yards_per_drive"),
    ]
    if "drive_red_zone_drives" in numeric_cols:
        rate_specs.append(("drive_red_zone_drives", "drive_count", "red_zone_rate"))
    if "drive_goal_to_go_drives" in numeric_cols:
        rate_specs.append(("drive_goal_to_go_drives", "drive_count", "goal_to_go_rate"))

    rate_exprs: list[pl.Expr] = []
    rate_feature_names: list[str] = []
    for numerator, denominator, label in rate_specs:
        prev_num = _prev_name(numerator)
        prev_den = _prev_name(denominator)
        l3_num = _l3_name(numerator)
        l3_den = _l3_name(denominator)
        prev_label = f"drive_hist_{label}_prev"
        l3_label = f"drive_hist_{label}_l3"
        rate_exprs.append(
            pl.when(pl.col(prev_den) > 0)
            .then(pl.col(prev_num) / pl.col(prev_den))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias(prev_label)
        )
        rate_exprs.append(
            pl.when(pl.col(l3_den) > 0)
            .then(pl.col(l3_num) / pl.col(l3_den))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias(l3_label)
        )
        rate_feature_names.extend([prev_label, l3_label])

    if rate_exprs:
        drive_features = drive_features.with_columns(rate_exprs)

    feature_cols = (
        [_prev_name(col) for col in numeric_cols]
        + [_l3_name(col) for col in numeric_cols]
    )
    feature_cols.extend(rate_feature_names)
    feature_cols = list(dict.fromkeys(feature_cols))

    # Keep minimal columns for join
    keep_cols = [
        "season",
        "week",
        "game_id",
        "team",
        "player_id",
        "game_date",
        "data_as_of",
        *feature_cols,
    ]
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in drive_features.columns]

    drive_features = drive_features.select(keep_cols)

    return drive_features.with_columns(
        [pl.col(col).fill_null(0.0).cast(pl.Float32) for col in feature_cols if col in drive_features.columns]
    )


def compute_td_rate_history(
    df: pl.DataFrame,
    *,
    group_cols: list[str],
    flag_col: str,
    prefix: str,
) -> pl.DataFrame:
    """
    Generic helper to compute rolling TD frequencies for binary flags.

    For each group (defined by `group_cols`) and game sequence, it returns:
      - {prefix}_rate_1g        : flag in previous game (1-game lookback)
      - {prefix}_rate_2g        : mean flag over last 2 games before current
      - {prefix}_rate_season    : season-to-date mean flag before current game
      - {prefix}_rate_lifetime  : career-to-date mean flag before current game
    """
    required = {"season", "week", "game_id", "game_date", flag_col, *group_cols}
    if not required.issubset(set(df.columns)):
        return pl.DataFrame()
    if df.is_empty():
        return pl.DataFrame()

    # Stable sort within groups
    sort_cols = group_cols + ["season", "week", "game_date"]
    work = df.sort(sort_cols)

    work = work.with_columns(
        [
            pl.col("game_date").cast(pl.Datetime("ms")).alias("_td_game_ts"),
            pl.col(flag_col).cast(pl.Float32).alias("_td_flag"),
            pl.lit(1.0).cast(pl.Float32).alias("_td_one"),
        ]
    )

    lifetime_group = group_cols
    season_group = [*group_cols, "season"]

    work = work.with_columns(
        [
            pl.col("_td_flag")
            .cum_sum()
            .over(lifetime_group)
            .alias("_td_cum_flag_lifetime"),
            pl.col("_td_one")
            .cum_sum()
            .over(lifetime_group)
            .alias("_td_cum_n_lifetime"),
            pl.col("_td_flag")
            .cum_sum()
            .over(season_group)
            .alias("_td_cum_flag_season"),
            pl.col("_td_one")
            .cum_sum()
            .over(season_group)
            .alias("_td_cum_n_season"),
        ]
    )

    rate_1g = (
        pl.col("_td_flag")
        .shift(1)
        .over(lifetime_group)
        .alias(f"{prefix}_rate_1g")
    )
    rate_2g = (
        pl.col("_td_flag")
        .rolling_mean(window_size=2, min_periods=1)
        .over(lifetime_group)
        .shift(1)
        .alias(f"{prefix}_rate_2g")
    )
    rate_season = (
        (pl.col("_td_cum_flag_season").shift(1) / pl.col("_td_cum_n_season").shift(1))
        .alias(f"{prefix}_rate_season")
    )
    rate_lifetime = (
        (
            pl.col("_td_cum_flag_lifetime").shift(1)
            / pl.col("_td_cum_n_lifetime").shift(1)
        ).alias(f"{prefix}_rate_lifetime")
    )

    work = work.with_columns([rate_1g, rate_2g, rate_season, rate_lifetime])

    keep_cols = [
        "season",
        "week",
        "game_id",
        "game_date",
        *group_cols,
        f"{prefix}_rate_1g",
        f"{prefix}_rate_2g",
        f"{prefix}_rate_season",
        f"{prefix}_rate_lifetime",
    ]
    # Deduplicate while preserving order
    keep_cols = list(dict.fromkeys(keep_cols))
    return work.select(keep_cols)


def add_team_and_opp_td_rate_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add simple rolling TD frequency features for teams and opponents.

    For each team (offense) and opponent (defense), compute how often they:
      - score rushing / receiving / passing TDs
      - allow rushing / receiving / passing TDs
    plus "any TD" versions, over 1g, 2g, season, and lifetime windows.
    """
    required = {
        "season",
        "week",
        "team",
        "opponent",
        "game_id",
        "game_date",
        "rushing_td_count",
        "receiving_td_count",
        "passing_td",
        "td_count",
    }
    if not required.issubset(set(df.columns)):
        return df
    if df.is_empty():
        return df

    base = df.select(
        [
            "season",
            "week",
            "team",
            "opponent",
            "game_id",
            "game_date",
            "rushing_td_count",
            "receiving_td_count",
            "passing_td",
            "td_count",
        ]
    )

    # Offense: TDs scored by team in each game
    off_games = (
        base.group_by(["season", "week", "team", "game_id"])
        .agg(
            [
                pl.col("game_date").max().alias("game_date"),
                (pl.col("rushing_td_count").fill_null(0).sum() > 0)
                .cast(pl.Int8)
                .alias("has_rush_td"),
                (pl.col("receiving_td_count").fill_null(0).sum() > 0)
                .cast(pl.Int8)
                .alias("has_rec_td"),
                (pl.col("passing_td").fill_null(0).sum() > 0)
                .cast(pl.Int8)
                .alias("has_pass_td"),
                (pl.col("td_count").fill_null(0).sum() > 0)
                .cast(pl.Int8)
                .alias("has_any_td"),
            ]
        )
        .sort(["season", "team", "game_date"])
    )

    # Defense: TDs allowed by opponent in each game
    def_games = (
        base.group_by(["season", "week", "opponent", "game_id"])
        .agg(
            [
                pl.col("game_date").max().alias("game_date"),
                (pl.col("rushing_td_count").fill_null(0).sum() > 0)
                .cast(pl.Int8)
                .alias("has_rush_td"),
                (pl.col("receiving_td_count").fill_null(0).sum() > 0)
                .cast(pl.Int8)
                .alias("has_rec_td"),
                (pl.col("passing_td").fill_null(0).sum() > 0)
                .cast(pl.Int8)
                .alias("has_pass_td"),
                (pl.col("td_count").fill_null(0).sum() > 0)
                .cast(pl.Int8)
                .alias("has_any_td"),
            ]
        )
        .rename({"opponent": "team"})
        .sort(["season", "team", "game_date"])
    )

    # Helper to accumulate multiple flag histories for a given games frame
    def _build_histories(games: pl.DataFrame, team_label: str) -> pl.DataFrame:
        if games.is_empty():
            return pl.DataFrame()

        histories: pl.DataFrame | None = None
        flag_specs = [
            ("has_rush_td", f"{team_label}_rush_td"),
            ("has_rec_td", f"{team_label}_rec_td"),
            ("has_pass_td", f"{team_label}_pass_td"),
            ("has_any_td", f"{team_label}_any_td"),
        ]
        for flag_col, prefix in flag_specs:
            hist = compute_td_rate_history(
                games, group_cols=["team"], flag_col=flag_col, prefix=prefix
            )
            if hist.is_empty():
                continue
            if histories is None:
                histories = hist
            else:
                histories = histories.join(
                    hist,
                    on=["season", "week", "game_id", "game_date", "team"],
                    how="left",
                )
        if histories is None:
            return pl.DataFrame()
        return histories

    off_hist = _build_histories(off_games, "team")
    def_hist = _build_histories(def_games, "opp")
    if not def_hist.is_empty():
        def_hist = def_hist.rename({"team": "opponent"})

    out = df
    if not off_hist.is_empty():
        # Cast and fill nulls for stability
        rate_cols = [c for c in off_hist.columns if c.startswith("team_") and "_rate_" in c]
        rate_cols = list(dict.fromkeys(rate_cols))
        if rate_cols:
            off_hist = off_hist.with_columns(
                [pl.col(c).cast(pl.Float32).alias(c) for c in rate_cols]
            )
        out = out.join(
            off_hist.drop("game_date"),
            on=["season", "week", "team", "game_id"],
            how="left",
        )

    if not def_hist.is_empty():
        rate_cols = [
            c for c in def_hist.columns if c.startswith("opp_") and "_rate_" in c
        ]
        if rate_cols:
            def_hist = def_hist.with_columns(
                [pl.col(c).cast(pl.Float32).alias(c) for c in rate_cols]
            )
        out = out.join(
            def_hist.drop("game_date"),
            on=["season", "week", "opponent", "game_id"],
            how="left",
        )

    return out


def add_position_td_rate_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add rolling TD frequencies by position bucket for team offense and opponent defense.

    For each row we compute, for the player's generic and specific position buckets:
      - how often that bucket scores a TD for this team
      - how often that bucket scores a TD *against* this opponent
    over 1g, 2g, season, and lifetime windows.
    """
    required = {
        "season",
        "week",
        "team",
        "opponent",
        "game_id",
        "game_date",
        "td_count",
        "position",
    }
    if not required.issubset(set(df.columns)):
        return df
    if df.is_empty():
        return df

    # Build generic and specific buckets using only columns that are guaranteed to exist.
    # Generic: coarse position family (just use roster position).
    # Specific: depth-chart slot when available, else fallback to position.
    df = df.with_columns(
        [
            pl.col("position").cast(pl.Utf8).alias("pos_bucket_generic"),
            pl.coalesce([pl.col("depth_chart_position"), pl.col("position")])
            .cast(pl.Utf8)
            .alias("pos_bucket_specific"),
        ]
    )

    base = df.select(
        [
            "season",
            "week",
            "team",
            "opponent",
            "game_id",
            "game_date",
            "td_count",
            "pos_bucket_generic",
            "pos_bucket_specific",
        ]
    )

    def _build_pos_histories(bucket_col: str, team_label: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        # Offense (team)
        off = (
            base.filter(pl.col(bucket_col).is_not_null())
            .group_by(["season", "week", "team", "game_id", bucket_col])
            .agg(
                [
                    pl.col("game_date").max().alias("game_date"),
                    (pl.col("td_count").fill_null(0).sum() > 0)
                    .cast(pl.Int8)
                    .alias("has_td"),
                ]
            )
            .sort(["season", "team", "game_date"])
        )

        # Defense (opponent)
        opp = (
            base.filter(pl.col(bucket_col).is_not_null())
            .group_by(["season", "week", "opponent", "game_id", bucket_col])
            .agg(
                [
                    pl.col("game_date").max().alias("game_date"),
                    (pl.col("td_count").fill_null(0).sum() > 0)
                    .cast(pl.Int8)
                    .alias("has_td"),
                ]
            )
            .rename({"opponent": "team"})
            .sort(["season", "team", "game_date"])
        )

        off_hist = compute_td_rate_history(
            off,
            group_cols=["team", bucket_col],
            flag_col="has_td",
            prefix=f"team_pos_{team_label}_td",
        )
        opp_hist = compute_td_rate_history(
            opp,
            group_cols=["team", bucket_col],
            flag_col="has_td",
            prefix=f"opp_pos_{team_label}_td_allowed",
        )
        if not opp_hist.is_empty():
            opp_hist = opp_hist.rename({"team": "opponent"})
        return off_hist, opp_hist

    off_gen, opp_gen = _build_pos_histories("pos_bucket_generic", "generic")
    off_spec, opp_spec = _build_pos_histories("pos_bucket_specific", "specific")

    out = df

    def _join_hist(
        current: pl.DataFrame,
        hist: pl.DataFrame,
        *,
        side_team_col: str,
        bucket_col: str,
        prefix_match: str,
    ) -> pl.DataFrame:
        if hist.is_empty():
            return current
        rate_cols = [c for c in hist.columns if c.startswith(prefix_match) and "_rate_" in c]
        if rate_cols:
            hist = hist.with_columns(
                [pl.col(c).cast(pl.Float32).alias(c) for c in rate_cols]
            )
        join_keys = ["season", "week", side_team_col, "game_id", bucket_col]
        drop_cols = [c for c in ["game_date"] if c in hist.columns]
        return current.join(
            hist.drop(drop_cols),
            on=join_keys,
            how="left",
        )

    # Offense: team position buckets
    if not off_gen.is_empty():
        out = _join_hist(
            out,
            off_gen,
            side_team_col="team",
            bucket_col="pos_bucket_generic",
            prefix_match="team_pos_generic_td",
        )
    if not off_spec.is_empty():
        out = _join_hist(
            out,
            off_spec,
            side_team_col="team",
            bucket_col="pos_bucket_specific",
            prefix_match="team_pos_specific_td",
        )

    # Defense: opponent position buckets
    if not opp_gen.is_empty():
        out = _join_hist(
            out,
            opp_gen,
            side_team_col="opponent",
            bucket_col="pos_bucket_generic",
            prefix_match="opp_pos_generic_td_allowed",
        )
    if not opp_spec.is_empty():
        out = _join_hist(
            out,
            opp_spec,
            side_team_col="opponent",
            bucket_col="pos_bucket_specific",
            prefix_match="opp_pos_specific_td_allowed",
        )

    return out


def attach_td_rate_history_features(enriched: pl.DataFrame) -> pl.DataFrame:
    """
    Attach team / opponent and position-bucket TD rate history to an
    existing frame using the canonical player-game history as the source.

    This is the SINGLE entry point that both training and prediction
    pipelines should use for these features to guarantee parity.

    The function:
      1. Loads `player_game_by_week` history lazily.
      2. Restricts history to games on or before the max `game_date`
         present in `enriched` (to avoid any future leakage).
      3. Applies `add_team_and_opp_td_rate_features` and
         `add_position_td_rate_features` to that history table to compute
         per-game, per-team, per-position TD rates.
      4. Projects those rates down to player-game keys and joins them
         back onto `enriched`.

    Only TD-rate columns are added; no additional null-filling or
    inference-only defaults are introduced here.
    """
    if enriched.is_empty():
        return enriched

    from utils.general.paths import PLAYER_GAME_BY_WEEK_DIR as PLAYER_GAME_DIR  # local import to avoid cycles

    # We only need the minimal set of history columns required by the TD-rate helpers.
    req_cols = [
        "season",
        "week",
        "player_id",
        "team",
        "opponent",
        "game_id",
        "game_date",
        "position",
        "depth_chart_position",
        "rushing_td_count",
        "receiving_td_count",
        "passing_td",
        "td_count",
    ]

    try:
        hist_scan = pl.scan_parquet(
            str(PLAYER_GAME_DIR / "season=*/week=*/part.parquet"),
            glob=True,
            hive_partitioning=True,
            missing_columns="insert",
            extra_columns="ignore",
        )
        # Normalise partition columns early
        hist_scan = hist_scan.with_columns(
            [
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
            ]
        )

        # Restrict history to games on or before the max game_date present in
        # the enriched frame to avoid any future leakage while still allowing
        # lifetime statistics to use all prior seasons.
        max_game_date = None
        if "game_date" in enriched.columns:
            try:
                max_game_date = enriched.get_column("game_date").cast(pl.Date).max()
            except Exception:
                max_game_date = None
        if max_game_date is not None and "game_date" in hist_scan.columns:
            hist_scan = hist_scan.filter(pl.col("game_date") <= max_game_date)

        hist_df = (
            hist_scan.select([c for c in req_cols if c in hist_scan.columns]).collect()
        )
        if hist_df.is_empty():
            return enriched

        # Normalise dtypes needed for stable joins / grouping.
        if "player_id" in hist_df.columns:
            hist_df = hist_df.with_columns(pl.col("player_id").cast(pl.Utf8))
        if "game_id" in hist_df.columns:
            hist_df = hist_df.with_columns(pl.col("game_id").cast(pl.Utf8))
        if "team" in hist_df.columns:
            hist_df = hist_df.with_columns(pl.col("team").cast(pl.Utf8))
        if "opponent" in hist_df.columns:
            hist_df = hist_df.with_columns(pl.col("opponent").cast(pl.Utf8))
        if "season" in hist_df.columns:
            hist_df = hist_df.with_columns(pl.col("season").cast(pl.Int32))
        if "week" in hist_df.columns:
            hist_df = hist_df.with_columns(pl.col("week").cast(pl.Int32))
        if "position" in hist_df.columns:
            hist_df = hist_df.with_columns(pl.col("position").cast(pl.Utf8))
        if "depth_chart_position" in hist_df.columns:
            hist_df = hist_df.with_columns(pl.col("depth_chart_position").cast(pl.Utf8))

        # Compute TD rates on the historical table only. This mirrors the
        # original training-time behaviour, where rates are defined purely
        # from past games and then joined back to player rows.
        hist_enriched = add_team_and_opp_td_rate_features(hist_df)
        hist_enriched = add_position_td_rate_features(hist_enriched)

        rate_cols = [
            c
            for c in hist_enriched.columns
            if ("td_rate" in c) or ("td_allowed_rate" in c)
        ]
        if not rate_cols:
            return enriched

        join_cols = ["season", "week", "game_id", "player_id"]
        # Some historical rows (e.g. zero-usage roster rows) may not have player_id.
        # We only join where the key set is complete.
        available_join = [c for c in join_cols if c in hist_enriched.columns]
        if len(available_join) < 3:
            return enriched

        to_join = hist_enriched.select(available_join + rate_cols)
        # Align id types
        cast_exprs: list[pl.Expr] = []
        for key in ("game_id", "player_id"):
            if key in to_join.columns:
                cast_exprs.append(pl.col(key).cast(pl.Utf8))
        if cast_exprs:
            to_join = to_join.with_columns(cast_exprs)
        if "game_id" in enriched.columns:
            enriched = enriched.with_columns(pl.col("game_id").cast(pl.Utf8))
        if "player_id" in enriched.columns:
            enriched = enriched.with_columns(pl.col("player_id").cast(pl.Utf8))

        enriched = enriched.join(to_join, on=available_join, how="left")
        return enriched

    except Exception as exc:  # pragma: no cover - defensive
        import logging

        logging.getLogger(__name__).warning(
            "Failed to attach TD rate history features: %s", exc
        )
        return enriched
