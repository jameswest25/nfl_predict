import pytest

from utils.train.tuning import (
    normalize_param_distributions,
    TuningConfigError,
)


def test_normalize_param_distributions_valid():
    cfg = {
        "center_width_pct": 0.4,
        "param_distributions": {
            "xgboost": {
                "n_estimators": {"low": 200, "high": 800, "type": "int"},
                "learning_rate": {"low": 0.01, "high": 0.1, "log": True},
                "grow_policy": {"choices": ["depthwise", "lossguide"]},
            }
        },
    }

    normalized = normalize_param_distributions(cfg)
    assert "xgboost" in normalized
    estimators = normalized["xgboost"]["n_estimators"]
    assert estimators["kind"] == "numeric"
    assert estimators["type"] == "int"
    assert estimators["low"] == 200
    assert estimators["high"] == 800
    assert pytest.approx(estimators["center_width_pct"], rel=1e-6) == 0.4

    lr = normalized["xgboost"]["learning_rate"]
    assert lr["kind"] == "numeric"
    assert lr["log"] is True

    grow_policy = normalized["xgboost"]["grow_policy"]
    assert grow_policy["kind"] == "categorical"
    assert grow_policy["choices"] == ["depthwise", "lossguide"]


def test_normalize_param_distributions_infers_int_type():
    cfg = {
        "param_distributions": {
            "xgboost": {
                "n_estimators": {"low": 100, "high": 500},
            }
        }
    }
    normalized = normalize_param_distributions(cfg)
    spec = normalized["xgboost"]["n_estimators"]
    assert spec["type"] == "int"
    assert spec["kind"] == "numeric"


def test_normalize_param_distributions_invalid_range():
    cfg = {
        "param_distributions": {
            "xgboost": {
                "n_estimators": {"low": 300, "high": 300},
            }
        }
    }
    with pytest.raises(TuningConfigError):
        normalize_param_distributions(cfg)

