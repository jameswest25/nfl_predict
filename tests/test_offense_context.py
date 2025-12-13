from __future__ import annotations

import polars as pl

from utils.feature.enrichment.offense_context import _build_is_unavailable_expr


def test_is_unavailable_uses_injury_designations_overrides():
    df = pl.DataFrame(
        {
            "injury_game_designation": ["OUT", "QUESTIONABLE", None],
            "injury_report_status": ["QUESTIONABLE", "OUT", "DOUBTFUL"],
        }
    )

    result = df.with_columns(_build_is_unavailable_expr(df).alias("is_unavailable"))

    assert result["is_unavailable"].to_list() == [True, True, True]


def test_is_unavailable_does_not_use_roster_status_only():
    df = pl.DataFrame({"status": ["INA", "ACT"]})
    result = df.with_columns(_build_is_unavailable_expr(df).alias("is_unavailable"))
    assert result["is_unavailable"].to_list() == [False, False]

