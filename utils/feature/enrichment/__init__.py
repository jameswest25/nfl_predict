"""Feature enrichment - add external/context data to feature matrices.

This subpackage contains modules that enrich data with:
- odds: NFL betting odds features
- weather_features: Weather forecast enrichment
- team_context: Team-level context features
- offense_context: Offensive coordinator and QB context
- qb_profiles: QB profile features
- travel_calendar: Travel and rest features
- asof: As-of time utilities for point-in-time correctness
- asof_metadata: As-of metadata management
- asof_enrichment: As-of metadata enrichment utilities
"""

from utils.feature.enrichment.odds import add_nfl_odds_features_to_df
from utils.feature.enrichment.weather_features import (
    add_weather_forecast_features_training,
    append_weather_context_flags,
)
from utils.feature.enrichment.team_context import (
    add_team_context_features,
    compute_team_context_history,
)
from utils.feature.enrichment.offense_context import (
    add_offense_context_features_training,
    _append_offense_context_columns,
)
from utils.feature.enrichment.qb_profiles import build_qb_profiles
from utils.feature.enrichment.asof import (
    decision_cutoff_horizons,
    decision_cutoff_override,
    get_decision_cutoff_hours,
    fallback_cutoff_hours,
    drop_missing_snapshots_enabled,
)
from utils.feature.enrichment.asof_metadata import build_asof_metadata, load_asof_metadata
from utils.feature.enrichment.asof_enrichment import (
    attach_asof_metadata,
    apply_snapshot_guards,
    load_and_build_asof_metadata,
)

__all__ = [
    # Odds
    "add_nfl_odds_features_to_df",
    # Weather
    "add_weather_forecast_features_training",
    "append_weather_context_flags",
    # Team context
    "add_team_context_features",
    "compute_team_context_history",
    # Offense context
    "add_offense_context_features_training",
    "_append_offense_context_columns",
    # QB profiles
    "build_qb_profiles",
    # As-of utilities
    "decision_cutoff_horizons",
    "decision_cutoff_override",
    "get_decision_cutoff_hours",
    "fallback_cutoff_hours",
    "drop_missing_snapshots_enabled",
    # As-of metadata
    "build_asof_metadata",
    "load_asof_metadata",
    # As-of enrichment
    "attach_asof_metadata",
    "apply_snapshot_guards",
    "load_and_build_asof_metadata",
]

