import polars as pl

from utils.feature.player_game_level import _append_role_flags


def test_role_flags_use_quantiles_per_team_position():
    df = pl.DataFrame(
        {
            "team": ["A"] * 5 + ["B"] * 5,
            "position": ["WR"] * 5 + ["RB"] * 5,
            "hist_red_zone_target_share_l3": [0.05, 0.1, 0.2, 0.4, 0.6] + [None] * 5,
            "hist_goal_to_go_carry_share_l3": [None] * 5 + [0.1, 0.15, 0.25, 0.35, 0.6],
        }
    )

    result = _append_role_flags(df)

    assert "role_primary_red_zone_target" in result.columns
    assert "role_goal_line_back" in result.columns

    flagged_red_zone = (
        result.filter((pl.col("team") == "A") & (pl.col("role_primary_red_zone_target") == 1)).height
    )
    assert flagged_red_zone == 2  # Top ~30% of WR red-zone share

    flagged_goal_line = (
        result.filter((pl.col("team") == "B") & (pl.col("role_goal_line_back") == 1)).height
    )
    assert flagged_goal_line == 2  # Top ~30% of RB goal-to-go share


def test_role_flags_gracefully_handle_missing_data():
    df = pl.DataFrame(
        {
            "team": ["C", "C"],
            "position": ["WR", "WR"],
            "hist_red_zone_target_share_l3": [None, None],
            "hist_goal_to_go_carry_share_l3": [None, None],
        }
    )

    result = _append_role_flags(df)
    assert result["role_primary_red_zone_target"].sum() == 0
    assert result["role_goal_line_back"].sum() == 0

