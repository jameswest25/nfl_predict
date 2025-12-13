"""NFL feature engineering utilities.

This package contains all the feature engineering logic for NFL player predictions.
The modules are organized into subpackages by category:

Subpackages
-----------
builders/
    Construct intermediate data levels from raw data:
    - play_level, drive_level, game_level
    - player_drive_level, player_game_level
    - opponent_splits

enrichment/
    Enrich data with external/context sources:
    - odds, weather_features
    - team_context, offense_context, qb_profiles
    - travel_calendar
    - asof, asof_metadata, asof_enrichment

rolling/
    Rolling window feature computation:
    - rolling_window, daily_totals
    - stats, pace

derived/
    Derived feature engineering:
    - usage_features, position_features
    - catch_rate_features, target_depth_features
    - snap_features, historical_share_features
    - market_features

core/
    Core utilities and shared helpers:
    - labels, targets
    - leak_guard
    - shared
"""

# Re-export commonly used functions for convenience
# Builders
from utils.feature.builders import (
    build_play_level,
    build_drive_level,
    build_game_level,
    build_player_drive_level,
    build_player_game_level,
    build_opponent_splits,
)

# Enrichment
from utils.feature.enrichment import (
    add_nfl_odds_features_to_df,
    add_weather_forecast_features_training,
    append_weather_context_flags,
    add_team_context_features,
    compute_team_context_history,
    add_offense_context_features_training,
    attach_asof_metadata,
    apply_snapshot_guards,
    load_and_build_asof_metadata,
    decision_cutoff_horizons,
    decision_cutoff_override,
    get_decision_cutoff_hours,
    fallback_cutoff_hours,
    drop_missing_snapshots_enabled,
    build_asof_metadata,
    load_asof_metadata,
)

# Rolling
from utils.feature.rolling import (
    add_rolling_features,
    build_daily_cache_range,
    ROLLING_FEATURE_STATS,
    ROLLING_WINDOWS,
    ROLLING_CONTEXTS,
)

# Derived features
from utils.feature.derived import (
    add_usage_helper_features,
    add_position_group,
    add_specialist_role_flags,
    add_moe_position_features,
    add_catch_rate_features,
    add_target_depth_features,
    add_snap_features,
    add_historical_share_features,
    add_combined_usage_features,
    add_role_share_flags,
    drop_leakage_columns,
    add_market_features,
)

# Core utilities
from utils.feature.core import (
    DEFAULT_LABEL_VERSION,
    get_label_spec,
    validate_target_columns,
    DEFAULT_LEAK_POLICY,
    build_schema_snapshot,
    enforce_leak_guard,
    write_schema_snapshot,
    finalize_drive_history_features,
    attach_td_rate_history_features,
)

__all__ = [
    # Builders
    "build_play_level",
    "build_drive_level",
    "build_game_level",
    "build_player_drive_level",
    "build_player_game_level",
    "build_opponent_splits",
    # Enrichment - odds
    "add_nfl_odds_features_to_df",
    # Enrichment - weather
    "add_weather_forecast_features_training",
    "append_weather_context_flags",
    # Enrichment - context
    "add_team_context_features",
    "compute_team_context_history",
    "add_offense_context_features_training",
    # Enrichment - asof
    "attach_asof_metadata",
    "apply_snapshot_guards",
    "load_and_build_asof_metadata",
    "decision_cutoff_horizons",
    "decision_cutoff_override",
    "get_decision_cutoff_hours",
    "fallback_cutoff_hours",
    "drop_missing_snapshots_enabled",
    "build_asof_metadata",
    "load_asof_metadata",
    # Rolling
    "add_rolling_features",
    "build_daily_cache_range",
    "ROLLING_FEATURE_STATS",
    "ROLLING_WINDOWS",
    "ROLLING_CONTEXTS",
    # Derived features
    "add_usage_helper_features",
    "add_position_group",
    "add_specialist_role_flags",
    "add_moe_position_features",
    "add_catch_rate_features",
    "add_target_depth_features",
    "add_snap_features",
    "add_historical_share_features",
    "add_combined_usage_features",
    "add_role_share_flags",
    "drop_leakage_columns",
    "add_market_features",
    # Core utilities
    "DEFAULT_LABEL_VERSION",
    "get_label_spec",
    "validate_target_columns",
    "DEFAULT_LEAK_POLICY",
    "build_schema_snapshot",
    "enforce_leak_guard",
    "write_schema_snapshot",
    "finalize_drive_history_features",
    "attach_td_rate_history_features",
]
