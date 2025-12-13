# utils/train/data.py
import pandas as pd
import numpy as np
import logging
import math
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def load_feature_matrix(path: str, time_col: str, columns=None):
    """Was ModelTrainer.load_data; identical behavior sans 'self'."""
    import pandas as pd
    logger.info(f"Loading feature matrix from {path}...")
    def _read_parquet(**kwargs):
        try:
            return pd.read_parquet(path, **kwargs)
        except Exception as e:
            logger.warning(f"pyarrow.read_parquet failed ({type(e).__name__}: {e}). Retrying with fastparquet…")
            return pd.read_parquet(path, engine="fastparquet", **kwargs)
    if columns:
        logger.info(f"Reading only {len(columns)} pre-determined columns.")
        df = _read_parquet(columns=columns)
        df = df[[c for c in columns if c in df.columns]]
    else:
        df = _read_parquet()
    if columns:
        missing = set(columns) - set(df.columns)
        if missing:
            raise AssertionError(f"Requested columns missing from DataFrame: {missing}")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    # NOTE: Do NOT create MLB-era aliases like `game_pk` in the NFL pipeline.
    # Grouping is handled by the caller via `group_col`/`groups_series`.
    if 'game_id' in df.columns:
        logger.info("Loaded feature matrix with NFL group column game_id")
    elif 'game_pk' in df.columns:
        logger.info("Loaded feature matrix with MLB group column game_pk")
    else:
        logger.warning("No game_id/game_pk column present in loaded feature matrix.")
    
    logger.info(f"Loaded {len(df)} records.")
    df = _augment_injury_signals(df)
    df = _augment_implied_totals(df)
    return df


def _augment_injury_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Derive leak-safe injury availability features directly on the feature matrix."""
    if df.empty:
        return df

    prob_col = "injury_inactive_probability"
    model_col = "injury_inactive_probability_model"
    low_col = "injury_inactive_probability_p10"
    high_col = "injury_inactive_probability_p90"
    width_col = "injury_inactive_probability_interval_width"
    source_col = "injury_inactive_probability_source"
    if prob_col in df.columns:
        df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce").clip(0.0, 1.0)
        df.loc[df[prob_col].isna(), prob_col] = 0.1
        df[prob_col] = df[prob_col].astype("float32")
        if model_col in df.columns:
            df[model_col] = pd.to_numeric(df[model_col], errors="coerce").clip(0.0, 1.0).astype("float32")
        if low_col in df.columns:
            df[low_col] = pd.to_numeric(df[low_col], errors="coerce").clip(0.0, 1.0).astype("float32")
        if high_col in df.columns:
            df[high_col] = pd.to_numeric(df[high_col], errors="coerce").clip(0.0, 1.0).astype("float32")
        if width_col in df.columns:
            df[width_col] = pd.to_numeric(df[width_col], errors="coerce").clip(lower=0.0).astype("float32")
        if source_col in df.columns:
            df[source_col] = df[source_col].fillna("heuristic").astype("category")
        return df

    report_col = "injury_report_status"
    practice_col = "injury_practice_status"
    dnp_col = "injury_practice_dnp_count"

    if report_col not in df.columns and practice_col not in df.columns:
        return df

    report_series = (
        df.get(report_col, pd.Series(index=df.index, dtype="object"))
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )
    practice_series = (
        df.get(practice_col, pd.Series(index=df.index, dtype="object"))
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    report_prob_map = {
        "OUT": 0.98,
        "DOUBTFUL": 0.85,
        "QUESTIONABLE": 0.55,
        "SUSPENDED": 0.90,
        "INACTIVE": 0.98,
    }
    practice_prob_map = {
        "DID NOT PARTICIPATE": 0.80,
        "LIMITED": 0.50,
        "FULL": 0.10,
    }

    report_prob = report_series.map(report_prob_map).fillna(0.10)
    practice_prob = practice_series.map(practice_prob_map).fillna(0.15)

    inactive_prob = np.maximum(report_prob, practice_prob)

    if dnp_col in df.columns:
        dnp_counts = pd.to_numeric(df[dnp_col], errors="coerce").fillna(0)
        boost_mask = dnp_counts >= 2
        inactive_prob = inactive_prob.where(~boost_mask, np.clip(inactive_prob + 0.10, 0.0, 1.0))

    df["injury_inactive_probability"] = inactive_prob.astype("float32")
    return df


def _augment_implied_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Derive team/opp implied totals from spread/total when missing.

    This mirrors the inference-time computation in pipeline/predict.py so that
    models see consistent features during training and prediction.
    """
    if df.empty:
        return df

    required = {"total_line", "spread_line", "team", "home_team", "away_team"}
    if not required.issubset(df.columns):
        return df

    # If team_implied_total already has non-null values, respect them.
    if "team_implied_total" in df.columns:
        col = pd.to_numeric(df["team_implied_total"], errors="coerce")
        if col.notna().any():
            return df

    total = pd.to_numeric(df["total_line"], errors="coerce")
    spread = pd.to_numeric(df["spread_line"], errors="coerce")

    home_total = (total - spread) / 2.0
    away_total = total - home_total

    team = df["team"].astype(str)
    home = df["home_team"].astype(str)
    away = df["away_team"].astype(str)

    team_it = pd.Series(index=df.index, dtype="float32")
    opp_it = pd.Series(index=df.index, dtype="float32")

    home_mask = team == home
    away_mask = team == away

    team_it[home_mask] = home_total[home_mask]
    opp_it[home_mask] = away_total[home_mask]
    team_it[away_mask] = away_total[away_mask]
    opp_it[away_mask] = home_total[away_mask]

    df["team_implied_total"] = team_it.astype("float32")
    df["opp_implied_total"] = opp_it.astype("float32")
    return df

def split_data_chronologically(df: pd.DataFrame, time_col: str, split_cfg: dict, production_mode: bool):
    """Was ModelTrainer.split_data_chronologically; identical behavior."""
    from datetime import datetime, timedelta
    import pandas as pd
    if production_mode:
        yesterday = (datetime.now() - timedelta(days=1)).date()
        logger.info("Production mode ON – using data up to %s as training; no dedicated validation/test slices.", yesterday)
        df_train = df[df[time_col].dt.date <= yesterday].copy()
        df_val   = pd.DataFrame(columns=df.columns)
        df_test  = pd.DataFrame(columns=df.columns)
        logger.info("Train set size: %d | Validation: 0 | Test: 0", len(df_train))
        return df_train, df_val, df_test

    train_end_cfg = split_cfg.get('train_end_date')
    val_end_cfg = split_cfg.get('validation_end_date')
    test_end_cfg = split_cfg.get('test_end_date')
    rolling_cfg = split_cfg.get('rolling_windows') or []
    rolling_enabled = str(split_cfg.get('split_strategy', '')).lower() == 'rolling'
    df_sorted = df.sort_values(time_col)

    def _apply_rolling_windows():
        if not rolling_enabled:
            return None
        required_cols = {'season', 'week'}
        if not required_cols.issubset(df.columns):
            logger.info("Rolling split requested but columns %s missing; falling back to fraction-based split.", required_cols)
            return None

        latest_season = df_sorted['season'].max()
        latest_rows = df_sorted[df_sorted['season'] == latest_season]
        if latest_rows.empty:
            return None

        weeks_available = sorted({int(w) for w in latest_rows['week'].dropna().unique()})
        if not weeks_available:
            logger.info("Rolling split requested but no week values present for season %s; falling back.", latest_season)
            return None

        def _windows_for_phase(phase_name: str) -> list[dict]:
            return [w for w in rolling_cfg if str(w.get('phase')).lower() == phase_name]

        def _weeks_from_windows(windows: list[dict]) -> set[int]:
            weeks: set[int] = set()
            for w in windows:
                try:
                    start_wk = int(w.get('start_week'))
                    end_wk = int(w.get('end_week'))
                except (TypeError, ValueError):
                    continue
                for wk in weeks_available:
                    if start_wk <= wk <= end_wk:
                        weeks.add(wk)
            return weeks

        def _window_span(windows: list[dict]) -> int:
            span = 0
            for w in windows:
                try:
                    start_wk = int(w.get('start_week'))
                    end_wk = int(w.get('end_week'))
                except (TypeError, ValueError):
                    continue
                span += max(0, end_wk - start_wk + 1)
            return span

        def _allocate_weeks(windows: list[dict], available_pool: list[int], default_span: int) -> list[int]:
            preset = _weeks_from_windows(windows)
            if preset:
                return sorted(preset)
            if not windows:
                return []
            span = _window_span(windows) or default_span
            span = max(0, span)
            span = min(span, len(available_pool))
            if span == 0:
                return []
            return sorted(available_pool)[-span:]

        train_windows = _windows_for_phase('train')
        val_windows = _windows_for_phase('val')
        test_windows = _windows_for_phase('test')
        post_windows = _windows_for_phase('postseason')

        test_weeks = _allocate_weeks(test_windows, weeks_available, default_span=2)
        if test_windows and not _weeks_from_windows(test_windows):
            logger.info("Rolling split test window fallback → assigned latest weeks %s", test_weeks)
        remaining_after_test = [wk for wk in weeks_available if wk not in test_weeks]

        val_weeks = _allocate_weeks(val_windows, remaining_after_test, default_span=4)
        if val_windows and not _weeks_from_windows(val_windows):
            logger.info("Rolling split validation window fallback → assigned latest weeks %s", val_weeks)
        remaining_after_val = [wk for wk in remaining_after_test if wk not in val_weeks]

        train_weeks = sorted(set(_weeks_from_windows(train_windows)) & set(remaining_after_val))
        if not train_weeks:
            train_weeks = remaining_after_val.copy()
            if train_windows:
                logger.info("Rolling split training window fallback → assigned remaining weeks %s", train_weeks)

        df_prior_seasons = df_sorted[df_sorted['season'] < latest_season].copy()
        df_latest = df_sorted[df_sorted['season'] == latest_season]

        df_test_local = df_latest[df_latest['week'].isin(test_weeks)].copy() if test_windows else pd.DataFrame(columns=df.columns)
        df_val_local = df_latest[df_latest['week'].isin(val_weeks)].copy() if val_windows else pd.DataFrame(columns=df.columns)
        df_train_latest = df_latest[df_latest['week'].isin(train_weeks)].copy()

        if df_test_local.empty and test_windows:
            logger.info("Rolling split test window fallback failed (no weeks available); reverting to fraction-based split.")
            return None
        if df_val_local.empty and val_windows:
            logger.info("Rolling split validation window fallback failed (no weeks available); reverting to fraction-based split.")
            return None
        if df_train_latest.empty and df_prior_seasons.empty:
            logger.info("Rolling split training window fallback failed (no weeks available); reverting to fraction-based split.")
            return None

        df_train_local = (
            pd.concat([df_prior_seasons, df_train_latest], ignore_index=True)
            if not df_prior_seasons.empty
            else df_train_latest
        )

        df_post_local = pd.DataFrame(columns=df.columns)
        if post_windows:
            post_weeks = _weeks_from_windows(post_windows)
            if post_weeks:
                df_post_local = df_latest[df_latest['week'].isin(sorted(post_weeks))].copy()

        # Capture any remaining in-season weeks that were not assigned by the configured windows.
        assigned_weeks = set(train_weeks) | set(val_weeks) | set(test_weeks)
        unused_weeks = [wk for wk in weeks_available if wk not in assigned_weeks]
        df_unused = pd.DataFrame(columns=df.columns)
        if unused_weeks:
            df_unused = df_latest[df_latest['week'].isin(unused_weeks)].copy()
            logger.info("Rolling split: assigning remaining regular-season weeks %s to test set.", sorted(unused_weeks))

        if not df_post_local.empty:
            logger.info(
                "Postseason slice captured %d rows (weeks %s).",
                len(df_post_local),
                sorted(df_post_local['week'].unique()),
            )

        if not df_unused.empty or not df_post_local.empty:
            df_test_local = pd.concat([df_test_local, df_unused, df_post_local], ignore_index=True)
            df_test_local = df_test_local.sort_values(time_col).reset_index(drop=True)

        logger.info(
            "Rolling split active → train weeks %s (plus prior seasons=%s), validation weeks %s, test weeks %s (season %s).",
            sorted(df_train_latest['week'].unique()),
            bool(len(df_prior_seasons)),
            sorted(df_val_local['week'].unique()),
            sorted(df_test_local['week'].unique()),
            latest_season,
        )

        return df_train_local, df_val_local, df_test_local

    rolling_result = _apply_rolling_windows()
    if rolling_result:
        df_train, df_val, df_test = rolling_result
        logger.info(
            "Train set size: %d | Validation: %d | Test: %d",
            len(df_train),
            len(df_val),
            len(df_test),
        )
        return df_train, df_val, df_test

    if train_end_cfg and val_end_cfg:
        train_end = pd.to_datetime(train_end_cfg)
        val_end = pd.to_datetime(val_end_cfg)
        if test_end_cfg:
            test_end_date = pd.to_datetime(test_end_cfg)
        else:
            test_period_days = split_cfg.get('test_period_days')
            if test_period_days is not None:
                test_start_date = val_end + timedelta(days=1)
                test_end_date = test_start_date + timedelta(days=test_period_days - 1)
            else:
                test_end_date = df_sorted[time_col].max()
        logger.info("Splitting data chronologically using configured cutoffs...")
        logger.info(f"Train set: up to {train_end.date()}")
        logger.info(f"Validation set: {train_end.date()} to {val_end.date()}")
        logger.info(f"Test set: {val_end.date()} to {test_end_date.date()}")
        df_train = df_sorted[df_sorted[time_col] <= train_end].copy()
        df_val = df_sorted[(df_sorted[time_col] > train_end) & (df_sorted[time_col] <= val_end)].copy()
        df_test = df_sorted[(df_sorted[time_col] > val_end) & (df_sorted[time_col] <= test_end_date)].copy()
    else:
        logger.info("Splitting data chronologically using dynamic season boundaries...")
        train_frac = float(split_cfg.get('train_size', 0.7))
        val_frac = float(split_cfg.get('val_size', 0.15))
        test_frac = float(split_cfg.get('test_size', max(0.0, 1.0 - train_frac - val_frac)))
        if train_frac < 0 or val_frac < 0 or test_frac < 0:
            raise ValueError("Split fractions must be non-negative")
        total = train_frac + val_frac + test_frac
        if not math.isclose(total, 1.0):
            logger.info("Normalising split fractions (current sum %.3f)", total)
            if total <= 0:
                train_frac, val_frac, test_frac = 0.7, 0.15, 0.15
            else:
                train_frac /= total
                val_frac /= total
                test_frac /= total

        unique_dates = df_sorted[time_col].drop_duplicates().reset_index(drop=True)
        n_dates = len(unique_dates)
        if n_dates == 0:
            return df_sorted.iloc[0:0], df_sorted.iloc[0:0], df_sorted.iloc[0:0]

        def _cut_index(fraction: float) -> int:
            if fraction <= 0:
                return 0
            idx = int(fraction * n_dates) - 1
            return max(0, min(n_dates - 1, idx))

        train_idx = _cut_index(train_frac)
        val_idx = max(train_idx, _cut_index(train_frac + val_frac))
        test_idx = max(val_idx, _cut_index(train_frac + val_frac + test_frac))

        train_end = unique_dates.iloc[train_idx]
        val_end = unique_dates.iloc[val_idx]
        test_end_date = unique_dates.iloc[test_idx]

        logger.info(f"Derived boundaries → train_end={train_end.date()}, "
                    f"val_end={val_end.date()}, test_end={test_end_date.date()}")

        df_train = df_sorted[df_sorted[time_col] <= train_end].copy()
        df_val = df_sorted[(df_sorted[time_col] > train_end) & (df_sorted[time_col] <= val_end)].copy()
        df_test = df_sorted[(df_sorted[time_col] > val_end) & (df_sorted[time_col] <= test_end_date)].copy()
    df_post_test = df[df[time_col] > test_end_date].copy()
    logger.info(f"Train set size: {len(df_train)}")
    logger.info(f"Validation set size: {len(df_val)}")
    logger.info(f"Test set size: {len(df_test)}")
    if 'test_end_date' in locals():
        test_end_display = test_end_date.date() if hasattr(test_end_date, 'date') else test_end_date
    else:
        test_end_display = df_test[time_col].max().date() if not df_test.empty else None
    if not df_post_test.empty:
        logger.info(f"Post-test (unused) data size: {len(df_post_test)}")
    return df_train, df_val, df_test
