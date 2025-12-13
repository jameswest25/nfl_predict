from __future__ import annotations

import yaml
from pathlib import Path

import polars as pl

from utils.feature.core.leak_guard import (
    DEFAULT_LEAK_POLICY,
    enforce_leak_guard,
    evaluate_leak_prone_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = PROJECT_ROOT / "config" / "training.yaml"
BANNED_FEATURES = {
    "status",
    "injury_game_status",
    "injury_is_inactive_designation",
    "injury_is_inactive_prob",
    "touchdown_player_id",
}


def _load_training_problem(name: str) -> dict:
    with TRAINING_CONFIG.open("r") as fp:
        config = yaml.safe_load(fp)
    problems = config.get("problems", [])
    for problem in problems:
        if problem.get("name") == name:
            return problem
    raise KeyError(f"Problem '{name}' not found in training config.")


def test_anytime_td_feature_list_has_no_banned_columns():
    anytime_td = _load_training_problem("anytime_td")
    other_features = set(anytime_td.get("other_features_to_include", []))
    assert not (other_features & BANNED_FEATURES), (
        "Leak-prone columns found in anytime_td feature list: "
        f"{other_features & BANNED_FEATURES}"
    )


def test_anytime_td_prefixes_do_not_cover_banned_columns():
    anytime_td = _load_training_problem("anytime_td")
    prefixes = [str(p) for p in anytime_td.get("feature_prefixes_to_include", []) or []]
    disallowed = []
    for banned in BANNED_FEATURES:
        for prefix in prefixes:
            if prefix and banned.startswith(prefix):
                disallowed.append((prefix, banned))
    assert not disallowed, (
        "Some feature prefixes would include leak-prone columns: "
        f"{disallowed}"
    )


def test_leak_guard_flags_pattern_columns():
    cols = [
        "market_anytime_td_prob",
        "final_score",
        "postgame_anytime_td",
        "actual_td_count",
        "safe_feature",
    ]
    result = evaluate_leak_prone_columns(cols, policy=DEFAULT_LEAK_POLICY)
    assert set(result.banned.keys()) == {"final_score", "postgame_anytime_td", "actual_td_count"}
    assert "safe_feature" in result.kept
    assert "market_anytime_td_prob" in result.kept


def test_enforce_leak_guard_drops_banned_and_keeps_allowlisted():
    df = pl.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "safe_feature": [1.0, 2.0],
            "postgame_anytime_td": [1, 0],
            "final_score": [30, 24],
        }
    )
    cleaned, info = enforce_leak_guard(
        df,
        policy=DEFAULT_LEAK_POLICY,
        allow_prefixes=("safe_",),
        allow_exact=("game_id",),
        drop_banned=True,
        drop_non_allowlisted=False,
        raise_on_banned=False,
    )
    assert "postgame_anytime_td" not in cleaned.columns
    assert "final_score" not in cleaned.columns
    assert "safe_feature" in cleaned.columns
    assert "game_id" in cleaned.columns
    assert set(info.dropped) == {"postgame_anytime_td", "final_score"}

