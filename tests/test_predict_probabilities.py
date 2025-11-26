from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import pipeline.predict as predict


def test_predict_for_problem_uses_predict_proba(monkeypatch):
    calls = {"proba": 0, "predict": 0}

    class DummyModel:
        def predict_proba(self, X):
            calls["proba"] += 1
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.7)])

        def predict(self, X):
            calls["predict"] += 1
            return np.full(len(X), 0.2)

    monkeypatch.setattr(predict.joblib, "load", lambda path: DummyModel())
    monkeypatch.setattr(predict, "_latest_model_path", lambda name: Path("dummy"))

    artifacts = {"feature_columns": ["a"], "task_type": "classification"}
    df = pd.DataFrame({"a": [0, 1]})

    preds = predict._predict_for_problem(df, {"name": "any", "task_type": "classification"}, artifacts)

    assert calls["proba"] == 1
    assert calls["predict"] == 0
    assert np.allclose(preds, 0.7)
    assert preds.dtype.kind == "f"


def test_apply_ps_fallback_sets_coverage(monkeypatch):
    baseline = pl.DataFrame(
        {
            "player_id": ["b"],
            "ps_team_dropbacks": [8.0],
            "ps_route_participation_plays": [6.0],
            "ps_hist_route_participation_pct_prev": [0.55],
            "ps_hist_tracking_has_game_data_prev": [1.0],
        }
    )
    monkeypatch.setattr(predict, "_load_ps_baselines", lambda season: baseline)

    enriched = pl.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "ps_team_dropbacks": [5.0, None, None],
            "ps_route_participation_plays": [4.0, None, None],
            "ps_hist_route_participation_pct_prev": [0.4, None, None],
            "ps_hist_tracking_has_game_data_prev": [1.0, None, None],
        }
    )

    result = predict._apply_ps_fallback(enriched, season_hint=2024).to_pandas()

    assert result.loc[result["player_id"] == "a", "ps_baseline_source"].item() == "actual"
    assert result.loc[result["player_id"] == "a", "ps_tracking_has_actual"].item() == 1
    assert result.loc[result["player_id"] == "a", "ps_tracking_used_baseline"].item() == 0

    assert result.loc[result["player_id"] == "b", "ps_baseline_source"].item() == "baseline"
    assert result.loc[result["player_id"] == "b", "ps_tracking_used_baseline"].item() == 1
    assert result.loc[result["player_id"] == "b", "ps_team_dropbacks"].item() == 8.0

    assert result.loc[result["player_id"] == "c", "ps_baseline_source"].item() == "none"
    assert result.loc[result["player_id"] == "c", "ps_tracking_has_actual"].item() == 0
    assert result.loc[result["player_id"] == "c", "ps_tracking_used_baseline"].item() == 0
