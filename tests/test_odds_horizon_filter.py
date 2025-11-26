from __future__ import annotations

import pandas as pd
import pytest

train_module = pytest.importorskip("pipeline.train")
ModelTrainer = train_module.ModelTrainer


def test_filter_odds_snapshot_columns_cutoff_keeps_base_only():
    df = pd.DataFrame(
        {
            "market_anytime_td_prob": [0.1, 0.2],
            "market_anytime_td_prob_2h": [0.15, 0.25],
            "market_anytime_td_prob_6h": [0.2, 0.3],
            "other_feature": [1, 2],
        }
    )
    filtered = ModelTrainer._filter_odds_snapshot_columns(df.copy(), "cutoff")
    assert "market_anytime_td_prob" in filtered.columns
    assert "market_anytime_td_prob_2h" not in filtered.columns
    assert "market_anytime_td_prob_6h" not in filtered.columns
    assert "other_feature" in filtered.columns


def test_filter_odds_snapshot_columns_specific_suffix():
    df = pd.DataFrame(
        {
            "market_anytime_td_prob": [0.1, 0.2],
            "market_anytime_td_prob_2h": [0.15, 0.25],
            "market_anytime_td_prob_6h": [0.2, 0.3],
        }
    )
    filtered = ModelTrainer._filter_odds_snapshot_columns(df.copy(), "6h")
    assert list(filtered.columns) == ["market_anytime_td_prob_6h"]
