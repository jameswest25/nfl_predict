"""NFL feature engineering utilities.

This package contains all the feature engineering logic for NFL player predictions.
The modules are organized by feature category:

Core builders (called from pipeline/feature.py):
- play_level.py          - Build play-level slices
- drive_level.py         - Build drive-level aggregates (team)
- player_drive_level.py  - Build player-drive-level aggregates
- player_game_level.py   - Build player-game-level aggregates
- game_level.py          - Build game-level aggregates (team)

Feature enrichment utilities:
- usage_features.py      - Usage model helpers and position baselines
- position_features.py   - MoE and position-specific features
- catch_rate_features.py - Catch rate efficiency features
- target_depth_features.py - Air yards and route features
- snap_features.py       - Rolling snaps, expected snaps, role stability
- historical_share_features.py - Historical usage shares
- market_features.py     - Market/odds-derived features
- asof_enrichment.py     - As-of metadata enrichment

Context features:
- team_context.py        - Team context features
- offense_context.py     - Offense context features (OC, QB)
- weather_features.py    - Weather forecast features
- odds.py                - NFL odds features
- opponent_splits.py     - Opponent defensive splits

Rolling window utilities:
- rolling_window.py      - Rolling window feature computation
- daily_totals.py        - Daily totals cache for rolling windows
- stats.py               - Rolling feature stat definitions

Other utilities:
- shared.py              - Shared helper functions
- labels.py              - Label definitions and specs
- targets.py             - Target column validation
- leak_guard.py          - Leakage prevention
- asof.py                - As-of time utilities
- asof_metadata.py       - As-of metadata management
- keygen.py              - Key generation utilities
- io.py                  - I/O utilities
"""

# Re-export commonly used functions for convenience
from utils.feature.usage_features import add_usage_helper_features
from utils.feature.position_features import (
    add_position_group,
    add_specialist_role_flags,
    add_moe_position_features,
)
from utils.feature.catch_rate_features import add_catch_rate_features
from utils.feature.target_depth_features import add_target_depth_features
from utils.feature.snap_features import add_snap_features
from utils.feature.historical_share_features import (
    add_historical_share_features,
    add_combined_usage_features,
    add_role_share_flags,
)
from utils.feature.market_features import add_market_features
from utils.feature.asof_enrichment import (
    attach_asof_metadata,
    apply_snapshot_guards,
)
from utils.feature.shared import (
    finalize_drive_history_features,
    attach_td_rate_history_features,
)

__all__ = [
    # Usage features
    "add_usage_helper_features",
    # Position features
    "add_position_group",
    "add_specialist_role_flags",
    "add_moe_position_features",
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
    # Market features
    "add_market_features",
    # Asof enrichment
    "attach_asof_metadata",
    "apply_snapshot_guards",
    # Shared utilities
    "finalize_drive_history_features",
    "attach_td_rate_history_features",
]
