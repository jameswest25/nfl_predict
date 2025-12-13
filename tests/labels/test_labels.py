import polars as pl

from utils.feature.core.labels import compute_td_labels, get_label_spec


def _lookup(frame: pl.DataFrame, player_id: str) -> dict:
    return frame.filter(pl.col("player_id") == player_id).to_dicts()[0]


def test_compute_td_labels_offense_and_all_variants():
    df = pl.DataFrame(
        {
            "player_id": ["rush", "rec", "pass", "def"],
            "game_id": ["g1"] * 4,
            "rushing_td_count": [1, 0, 0, 0],
            "receiving_td_count": [0, 1, 0, 0],
            "passing_td": [0, 0, 1, 0],
            # Generic touchdown attribution (e.g., defensive return)
            "touchdowns": [1, 1, 0, 1],
        }
    )

    result = compute_td_labels(df, version="v1_any_offense")

    rush = _lookup(result, "rush")
    assert rush["anytime_td_offense"] == 1
    assert rush["anytime_td_rush"] == 1
    assert rush["anytime_td_rec"] == 0
    assert rush["anytime_td_pass_thrown"] == 0
    # Skill-only labels treat rush-only TD as a hit
    assert rush["anytime_td_skill"] == 1
    assert rush["td_count_skill"] == 1
    assert rush["td_count_offense"] == 1
    assert rush["td_count_all"] == 1
    assert rush["anytime_td"] == rush["anytime_td_offense"]

    rec = _lookup(result, "rec")
    assert rec["anytime_td_offense"] == 1
    assert rec["anytime_td_rec"] == 1
    assert rec["anytime_td_rush"] == 0
    # Receiving TD also triggers skill-only label
    assert rec["anytime_td_skill"] == 1

    passer = _lookup(result, "pass")
    assert passer["anytime_td_pass_thrown"] == 1
    assert passer["anytime_td_offense"] == 1
    assert passer["anytime_td_all"] == 1  # includes offensive TDs
    # Passing TDs alone should NOT count as a skill TD
    assert passer["anytime_td_skill"] == 0
    assert passer["td_count_skill"] == 0

    defensive = _lookup(result, "def")
    assert defensive["anytime_td_offense"] == 0  # no offensive scoring events
    assert defensive["anytime_td_all"] == 1  # credited via generic touchdown
    assert defensive["anytime_td"] == defensive["anytime_td_offense"]
    # Defensive-only TD also excluded from skill-only label
    assert defensive["anytime_td_skill"] == 0


def test_label_version_any_all_aliases_anytime_td():
    spec = get_label_spec("v1_any_all")
    assert spec.primary == "anytime_td_all"

    df = pl.DataFrame(
        {
            "player_id": ["all"],
            "game_id": ["g1"],
            "rushing_td_count": [0],
            "receiving_td_count": [0],
            "passing_td": [0],
            "touchdowns": [1],
        }
    )
    result = compute_td_labels(df, version="v1_any_all")
    row = _lookup(result, "all")
    # Alias ensures legacy anytime_td follows the version's primary definition.
    assert row["anytime_td_all"] == 1
    assert row["anytime_td"] == row["anytime_td_all"]
