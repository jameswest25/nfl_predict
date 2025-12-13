"""NFL prediction utilities.

This package contains all the prediction pipeline utilities for NFL player predictions.
The modules are organized by functionality:

Modules
-------
scaffold
    Build player-game scaffolds from schedule and roster data
    - load_schedule, load_rosters, build_scaffold

features
    Unified feature builder for training and inference
    - build_features (main entry point)

loaders
    Data loaders for context features
    - load_qb_profile_features, load_travel_calendar_features
    - load_injury_history_features, load_ps_baselines

injuries
    Injury feature enrichment
    - ensure_injury_cache, attach_injury_features

inference
    Model inference and post-processing
    - load_artifacts, predict_moe, predict_global
    - apply_guards_inline, apply_availability_floor
    - various position-specific caps

output
    Output formatting and composed features
    - inject_composed_features, format_output, save_predictions
"""

from utils.predict.scaffold import (
    ensure_imports_ready,
    load_schedule,
    load_rosters,
    build_scaffold,
)

from utils.predict.features import (
    build_features,
)

from utils.predict.loaders import (
    load_qb_profile_features,
    load_travel_calendar_features,
    load_injury_history_features,
    load_ps_baselines,
)

from utils.predict.injuries import (
    ensure_injury_cache,
    attach_injury_features,
)

from utils.predict.inference import (
    load_artifacts,
    load_threshold,
    prepare_feature_matrix,
    check_moe_available,
    predict_moe,
    predict_global,
    apply_guards_inline,
    apply_availability_floor,
    apply_snaps_ceiling_cap,
    apply_usage_targets_position_cap,
    apply_usage_carries_position_cap,
    apply_usage_target_yards_position_cap,
    apply_efficiency_rec_yards_air_cap,
)

from utils.predict.output import (
    inject_composed_features,
    format_output,
    save_predictions,
    META_COLUMNS,
    OUTPUT_COLUMN_PREFIXES,
    OUTPUT_COLUMN_WHITELIST,
)

__all__ = [
    # Scaffold
    "ensure_imports_ready",
    "load_schedule",
    "load_rosters",
    "build_scaffold",
    # Features
    "build_features",
    # Loaders
    "load_qb_profile_features",
    "load_travel_calendar_features",
    "load_injury_history_features",
    "load_ps_baselines",
    # Injuries
    "ensure_injury_cache",
    "attach_injury_features",
    # Inference
    "load_artifacts",
    "load_threshold",
    "prepare_feature_matrix",
    "check_moe_available",
    "predict_moe",
    "predict_global",
    "apply_guards_inline",
    "apply_availability_floor",
    "apply_snaps_ceiling_cap",
    "apply_usage_targets_position_cap",
    "apply_usage_carries_position_cap",
    "apply_usage_target_yards_position_cap",
    "apply_efficiency_rec_yards_air_cap",
    # Output
    "inject_composed_features",
    "format_output",
    "save_predictions",
    "META_COLUMNS",
    "OUTPUT_COLUMN_PREFIXES",
    "OUTPUT_COLUMN_WHITELIST",
]
