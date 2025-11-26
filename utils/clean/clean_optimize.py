import logging
import pandas as pd

__all__ = ["optimize_data"]


def optimize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Light-weight, centralised dtype optimisation used by the cleaning pipeline."""

    logging.info("Optimising dtypes…")

    identifier_cols = {
        "game_id",
        "old_game_id",
        "nflverse_game_id",
        "nfl_api_id",
        "play_clock",
    }
    for col in identifier_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # ▸ Binary flags – only roof and runner flags survive the refactor
    flag_cols = [c for c in df.columns if c in ("is_roof_closed",) or c.startswith("runner_on_")]
    for col in flag_cols:
        df[col] = df[col].fillna(0).astype("uint8")

    # ▸ Wind features → float32
    for col in (
        "wind_sin", "wind_cos", "wind_x", "wind_y",
        "headwind_component", "crosswind_component",
        "wind_resistance", "wind_assistance",
    ):
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # ▸ Handedness → category
    for col in ("batter_hand", "pitcher_hand"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    # ▸ Generic numeric down-casting
    try:
        df = _generic_downcast(df)
    except Exception as exc:
        logging.warning("Generic downcast skipped due to error: %s", exc)

    # ▸ Object columns with low cardinality → category  (heuristic: ≤1 000 unique or <50% of rows)
    row_cnt = len(df)
    for col in df.select_dtypes("object").columns:
        if col in identifier_cols:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1000 or nunique / max(row_cnt, 1) < 0.5:
            df[col] = df[col].astype("category")

    # ▸ Further down-cast signed ints that are non-negative → unsigned
    skip_unsigned = {
        "game_id",
        "old_game_id",
        "nflverse_game_id",
        "nfl_api_id",
    }
    for col in df.select_dtypes(include=["int8", "int16", "int32", "int64"]).columns:
        if col in skip_unsigned:
            continue
        try:
            if df[col].min(skipna=True) >= 0:
                df[col] = pd.to_numeric(df[col], downcast="unsigned")
        except (ValueError, TypeError):
            logging.debug("Skipping unsigned downcast for column %s", col)
    return df


def _generic_downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Utility: down-cast int/float64 → smallest possible subtype."""

    skip_cols = {
        "game_id",
        "old_game_id",
        "nflverse_game_id",
        "nfl_api_id",
    }

    for col in df.select_dtypes(include=["int64"]).columns:
        if col in skip_cols:
            continue
        try:
            df[col] = pd.to_numeric(df[col], downcast="integer", errors="ignore")
        except (ValueError, TypeError):
            logging.debug("Skipping integer downcast for column %s", col)
    for col in df.select_dtypes(include=["float64"]).columns:
        if col in skip_cols:
            continue
        try:
            df[col] = pd.to_numeric(df[col], downcast="float", errors="ignore")
        except (ValueError, TypeError):
            logging.debug("Skipping float downcast for column %s", col)
    return df 