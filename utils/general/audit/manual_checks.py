from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import polars as pl


def _load_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def check_integrity(feature_path: Path | str = Path("data/processed/final/processed.parquet")) -> None:
    path = Path(feature_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {path}")
    df = _load_parquet(path)

    leak_candidates = ["team_score", "opponent_score", "home_score", "away_score"]
    found_leaks = [c for c in leak_candidates if c in df.columns]
    print(f"[integrity] Score columns present: {found_leaks}")

    if "vacated_targets_position" in df.columns:
        max_vacated = df["vacated_targets_position"].max()
        print(f"[integrity] Max vacated targets: {max_vacated}")
    else:
        print("[integrity] Column 'vacated_targets_position' missing")

    if {"passing_td", "anytime_td", "rushing_td_count", "receiving_td_count"}.issubset(df.columns):
        qbs = df.filter(
            (pl.col("passing_td") > 0)
            & (pl.col("rushing_td_count") == 0)
            & (pl.col("receiving_td_count") == 0)
        )
        if qbs.is_empty():
            print("[integrity] No pure passing-TD games found to verify anytime_td logic.")
        else:
            bad = qbs.filter(pl.col("anytime_td") == 1)
            if bad.is_empty():
                print("[integrity] ✅ Passing-only TD games do not trigger anytime_td.")
            else:
                print(f"[integrity] 🚨 Found {len(bad)} rows with anytime_td=1 on pure passing TDs:")
                print(bad.select(["player_name", "season", "week", "passing_td", "anytime_td"]).head())
    else:
        print("[integrity] Skipping anytime_td check (columns missing).")


def check_rolling_ina(feature_path: Path | str = Path("data/processed/final/processed.parquet")) -> None:
    path = Path(feature_path)
    df = _load_parquet(path)

    if "status" not in df.columns or "3g_target_per_game" not in df.columns:
        print("[rolling] Required columns missing (status, 3g_target_per_game).")
        return

    ina_df = df.filter(pl.col("status") == "INA")
    print(f"[rolling] INA rows: {len(ina_df)}")

    valid = ina_df.filter(pl.col("3g_target_per_game") > 0)
    print(f"[rolling] INA rows with rolling targets > 0: {len(valid)}")
    if not valid.is_empty():
        print(valid.select(["player_name", "season", "week", "3g_target_per_game", "vacated_targets_position"]).head())
    else:
        null_rolling = ina_df.filter(pl.col("3g_target_per_game").is_null())
        print(f"[rolling] INA rows with NULL rolling targets: {len(null_rolling)}")


def check_travel_leakage(
    travel_path: Path | str = Path("data/processed/travel_calendar/season=2024/week=1/part.parquet"),
) -> None:
    path = Path(travel_path)
    df = _load_parquet(path)

    leak_candidates = ["team_score", "opponent_score"]
    found = [c for c in leak_candidates if c in df.columns]
    if found:
        print(f"[travel] Score columns present: {found}")
        print(df.select(found).head())
    else:
        print("[travel] No obvious score columns detected.")


def debug_vacated(feature_path: Path | str = Path("data/processed/final/processed.parquet")) -> None:
    df = _load_parquet(Path(feature_path))
    required_cols = {"status", "injury_game_designation", "3g_target_per_game", "3g_carry_per_game"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"[vacated] Missing columns: {missing}")
        return

    df = df.with_columns(
        (
            (pl.col("status").str.to_uppercase() == "INA")
            | (pl.col("injury_game_designation").str.to_uppercase() == "OUT")
        ).alias("is_unavailable")
    )
    unavailable = df.filter(pl.col("is_unavailable"))
    print(f"[vacated] Unavailable rows: {len(unavailable)}")

    vacated_usage = (
        unavailable.group_by(["season", "week", "team", "position"])
        .agg(
            [
                pl.col("3g_target_per_game").sum().alias("vacated_targets_position"),
                pl.col("3g_carry_per_game").sum().alias("vacated_carries_position"),
            ]
        )
        .sort("vacated_targets_position", descending=True)
    )
    print(f"[vacated] Aggregated rows: {len(vacated_usage)}")
    if vacated_usage.is_empty():
        print("[vacated] No aggregation output.")
        return

    example = vacated_usage.filter(pl.col("vacated_targets_position") > 5).head(1)
    if example.is_empty():
        print("[vacated] No high vacated usage example found.")
        return

    print("[vacated] Example high vacated usage row:")
    print(example)
    s, w, t, p = example.row(0)[:4]
    target_rows = df.filter(
        (pl.col("season") == s) & (pl.col("week") == w) & (pl.col("team") == t) & (pl.col("position") == p)
    )
    print(f"[vacated] Rows in main DF for that group: {len(target_rows)}")
    if not target_rows.is_empty():
        joined = target_rows.join(
            vacated_usage,
            on=["season", "week", "team", "position"],
            how="left",
            suffix="_new",
        )
        print(joined.select(["player_name", "vacated_targets_position", "vacated_targets_position_new"]).head())


def verify_opponent_split_lag(
    feature_path: Path | str = Path("data/processed/player_game_by_week/season=2024/week=1/part.parquet"),
    column: str = "opp_def_pass_epa_allowed_prev",
    team: str = "KC",
    reference_value: Optional[float] = 0.36948,
) -> None:
    path = Path(feature_path)
    df = _load_parquet(path)
    if column not in df.columns:
        print(f"[verify] Column {column} not found in {path}")
        return

    print(f"[verify] Column {column} present.")
    sample = df.filter(pl.col("team") == team).select(["player_name", column]).head(1)
    if sample.is_empty():
        print(f"[verify] No rows found for team {team}.")
        return
    value = sample[column][0]
    print(f"[verify] Example value for {team}: {value}")

    if reference_value is None:
        return
    if value is None:
        print("[verify] Value is None (history missing).")
    elif abs(value - reference_value) > 1e-3:
        print("[verify] ✅ Lagged value differs from actual game outcome (no leakage).")
    else:
        print("[verify] 🚨 Value matches actual game outcome! Investigate potential leakage.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual audit utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("integrity")
    subparsers.add_parser("rolling")
    subparsers.add_parser("travel")
    subparsers.add_parser("vacated")
    subparsers.add_parser("verify")

    parser.add_argument("--feature-path", type=str, default="data/processed/final/processed.parquet")
    parser.add_argument("--travel-path", type=str, default="data/processed/travel_calendar/season=2024/week=1/part.parquet")
    parser.add_argument("--verify-path", type=str, default="data/processed/player_game_by_week/season=2024/week=1/part.parquet")
    parser.add_argument("--verify-column", type=str, default="opp_def_pass_epa_allowed_prev")
    parser.add_argument("--verify-team", type=str, default="KC")
    parser.add_argument("--reference-value", type=float, default=0.36948)

    args = parser.parse_args()
    cmd = args.command

    if cmd == "integrity":
        check_integrity(args.feature_path)
    elif cmd == "rolling":
        check_rolling_ina(args.feature_path)
    elif cmd == "travel":
        check_travel_leakage(args.travel_path)
    elif cmd == "vacated":
        debug_vacated(args.feature_path)
    elif cmd == "verify":
        verify_opponent_split_lag(
            feature_path=args.verify_path,
            column=args.verify_column,
            team=args.verify_team,
            reference_value=args.reference_value,
        )


if __name__ == "__main__":
    main()


