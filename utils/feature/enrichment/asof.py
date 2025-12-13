from __future__ import annotations

"""Helpers for aligning feature generation to a consistent decision-time snapshot."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Mapping

import yaml

from utils.general.paths import PROJ_ROOT


@lru_cache(maxsize=1)
def _load_cutoff_config() -> Mapping[str, Any]:
    """Load decision cutoff configuration from config/config.yaml."""
    config_path = PROJ_ROOT / "config" / "config.yaml"
    try:
        with config_path.open("r") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data.get("decision_cutoff", {}) or {}


def decision_cutoff_hours_default() -> float:
    """Default hours before kickoff to freeze features."""
    cfg = _load_cutoff_config()
    value = cfg.get("hours_before_kickoff", 6)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 6.0


def decision_cutoff_hours_for_season_type(season_type: str | None) -> float:
    """Hours before kickoff honoring optional per-season-type overrides."""
    cfg = _load_cutoff_config()
    overrides = cfg.get("season_type_overrides", {}) or {}
    if season_type and season_type in overrides:
        try:
            return float(overrides[season_type])
        except (TypeError, ValueError):
            pass
    return decision_cutoff_hours_default()


def fallback_cutoff_hours() -> float:
    """Fallback hours when kickoff timestamp is unavailable."""
    cfg = _load_cutoff_config()
    value = cfg.get("fallback_hours_before_kickoff", 36)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 36.0


_cutoff_hours_ctx: ContextVar[float | None] = ContextVar(
    "decision_cutoff_hours_override", default=None
)
_fallback_hours_ctx: ContextVar[float | None] = ContextVar(
    "decision_cutoff_fallback_override", default=None
)


def decision_cutoff_horizons() -> list[float]:
    """Configured horizons (hours before kickoff) for multi-view feature builds."""
    cfg = _load_cutoff_config()
    raw = cfg.get("horizons_hours")
    if not raw:
        return [decision_cutoff_hours_default()]
    horizons: list[float] = []
    for value in raw:
        try:
            horizons.append(float(value))
        except (TypeError, ValueError):
            continue
    if not horizons:
        horizons = [decision_cutoff_hours_default()]
    horizons = sorted({round(h, 4) for h in horizons})
    return horizons


def get_decision_cutoff_hours() -> float:
    """Return the active decision cutoff hours (override-aware)."""
    override = _cutoff_hours_ctx.get()
    if override is not None:
        return float(override)
    return decision_cutoff_hours_default()


def get_fallback_cutoff_hours() -> float:
    """Return the active fallback cutoff hours (override-aware)."""
    override = _fallback_hours_ctx.get()
    if override is not None:
        return float(override)
    return fallback_cutoff_hours()


@contextmanager
def decision_cutoff_override(
    *,
    cutoff_hours: float | None = None,
    fallback_hours: float | None = None,
):
    """Context manager to temporarily override decision cutoff hours."""
    tokens: list[tuple[ContextVar[float | None], object]] = []
    try:
        if cutoff_hours is not None:
            tokens.append((_cutoff_hours_ctx, _cutoff_hours_ctx.set(float(cutoff_hours))))
        if fallback_hours is not None:
            tokens.append((_fallback_hours_ctx, _fallback_hours_ctx.set(float(fallback_hours))))
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def drop_missing_snapshots_enabled() -> bool:
    """Return whether rows missing pre-cutoff snapshots should be dropped."""
    cfg = _load_cutoff_config()
    value = cfg.get("drop_missing_snapshots", True)
    # Accept truthy strings or ints in config
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


__all__ = [
    "decision_cutoff_hours_default",
    "decision_cutoff_hours_for_season_type",
    "decision_cutoff_horizons",
    "fallback_cutoff_hours",
    "get_decision_cutoff_hours",
    "get_fallback_cutoff_hours",
    "decision_cutoff_override",
    "drop_missing_snapshots_enabled",
]

