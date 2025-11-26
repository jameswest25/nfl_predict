"""
Centralized configuration loading for NFL Prediction Pipeline.

Provides cached access to config/config.yaml and config/training.yaml
to avoid repeated file reads across modules.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

# Project root (relative to this file's location: utils/general/config.py)
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


@lru_cache(maxsize=1)
def load_config() -> Mapping[str, Any]:
    """Load the main configuration from config/config.yaml.
    
    Returns a cached, immutable view of the configuration.
    Subsequent calls return the same cached result.
    """
    config_path = _CONFIG_DIR / "config.yaml"
    try:
        with config_path.open("r") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data


@lru_cache(maxsize=1)
def load_training_config() -> Mapping[str, Any]:
    """Load the training configuration from config/training.yaml.
    
    Returns a cached, immutable view of the configuration.
    """
    config_path = _CONFIG_DIR / "training.yaml"
    try:
        with config_path.open("r") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data


def get_odds_api_key() -> str | None:
    """Get the Odds API key from configuration."""
    config = load_config()
    return config.get("odds_api_key")


def get_weather_api_key() -> str | None:
    """Get the weather API key from configuration."""
    config = load_config()
    weather_cfg = config.get("weather", {})
    return weather_cfg.get("api_key")


def get_pipeline_config() -> Mapping[str, Any]:
    """Get the pipeline section of the main config."""
    config = load_config()
    return config.get("pipeline", {})


def get_collect_params() -> Mapping[str, Any]:
    """Get the collect_params section of the main config."""
    config = load_config()
    return config.get("collect_params", {})


def get_decision_cutoff_config() -> Mapping[str, Any]:
    """Get the decision_cutoff section of the main config."""
    config = load_config()
    return config.get("decision_cutoff", {})


def get_odds_snapshot_config() -> Mapping[str, Any]:
    """Get odds snapshot settings embedded in the decision_cutoff section."""
    cutoff_cfg = get_decision_cutoff_config() or {}
    return {
        "mode": cutoff_cfg.get("odds_snapshot_mode", "cutoff"),
        "fixed_hour_utc": cutoff_cfg.get("odds_fixed_hour_utc", 12),
        "fallback_hours_before_kickoff": cutoff_cfg.get(
            "odds_fallback_hours_before_kickoff",
            cutoff_cfg.get("fallback_hours_before_kickoff", 36),
        ),
    }


def clear_config_cache() -> None:
    """Clear all cached configuration (useful for testing)."""
    load_config.cache_clear()
    load_training_config.cache_clear()


__all__ = [
    "load_config",
    "load_training_config",
    "get_odds_api_key",
    "get_weather_api_key",
    "get_pipeline_config",
    "get_collect_params",
    "get_decision_cutoff_config",
    "get_odds_snapshot_config",
    "clear_config_cache",
]

