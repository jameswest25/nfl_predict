"""Derived feature engineering utilities.

This subpackage contains modules for computing derived features:
- usage_features: Usage model helpers and position baselines
- position_features: MoE and position-specific features
- catch_rate_features: Catch rate efficiency features
- target_depth_features: Air yards and route features
- snap_features: Rolling snaps, expected snaps, role stability
- historical_share_features: Historical usage shares
- market_features: Market/odds-derived interaction features
"""

from utils.feature.derived.usage_features import add_usage_helper_features
from utils.feature.derived.position_features import (
    add_position_group,
    add_specialist_role_flags,
    add_moe_position_features,
    add_context_gaps_features,
)
from utils.feature.derived.catch_rate_features import add_catch_rate_features
from utils.feature.derived.target_depth_features import add_target_depth_features
from utils.feature.derived.snap_features import add_snap_features
from utils.feature.derived.historical_share_features import (
    add_historical_share_features,
    add_combined_usage_features,
    add_role_share_flags,
    drop_leakage_columns,
)
from utils.feature.derived.market_features import add_market_features

__all__ = [
    # Usage features
    "add_usage_helper_features",
    # Position features
    "add_position_group",
    "add_specialist_role_flags",
    "add_moe_position_features",
    "add_context_gaps_features",
    # Catch rate features
    "add_catch_rate_features",
    # Target depth features
    "add_target_depth_features",
    # Snap features
    "add_snap_features",
    # Historical share features
    "add_historical_share_features",
    "add_combined_usage_features",
    "add_role_share_flags",
    "drop_leakage_columns",
    # Market features
    "add_market_features",
]

