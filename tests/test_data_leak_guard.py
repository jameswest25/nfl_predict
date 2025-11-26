from __future__ import annotations

import yaml
from pathlib import Path


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


