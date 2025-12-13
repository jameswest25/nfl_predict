"""Core feature utilities.

This subpackage contains core utilities and shared helpers:
- labels: Label definitions and specs
- targets: Target column validation
- leak_guard: Leakage prevention utilities
- shared: Shared helper functions (drive aggregates, TD rate history)
"""

from utils.feature.core.labels import DEFAULT_LABEL_VERSION, get_label_spec, compute_td_labels
from utils.feature.core.targets import validate_target_columns
from utils.feature.core.leak_guard import (
    DEFAULT_LEAK_POLICY,
    build_schema_snapshot,
    enforce_leak_guard,
    write_schema_snapshot,
)
from utils.feature.core.shared import (
    finalize_drive_history_features,
    attach_td_rate_history_features,
    compute_td_rate_history,
    add_team_and_opp_td_rate_features,
    add_position_td_rate_features,
    compute_drive_level_aggregates,
)

__all__ = [
    # Labels
    "DEFAULT_LABEL_VERSION",
    "get_label_spec",
    "compute_td_labels",
    # Targets
    "validate_target_columns",
    # Leak guard
    "DEFAULT_LEAK_POLICY",
    "build_schema_snapshot",
    "enforce_leak_guard",
    "write_schema_snapshot",
    # Shared
    "finalize_drive_history_features",
    "attach_td_rate_history_features",
    "compute_td_rate_history",
    "add_team_and_opp_td_rate_features",
    "add_position_td_rate_features",
    "compute_drive_level_aggregates",
]

