"""NFL player statistics configuration for rolling window features.

Defines the standard set of stats tracked at all aggregation levels
(play, drive, game) for rolling window computations.

This module is the SINGLE SOURCE OF TRUTH for stat definitions used
throughout the pipeline. Import from here rather than defining locally.
"""

from typing import List

# =============================================================================
# Core Stats (used for drive and game level aggregation)
# =============================================================================

# Basic stats aggregated at player-drive level
PLAYER_DRIVE_STATS: List[str] = [
    # Yardage
    "receiving_yards",
    "rushing_yards",
    "passing_yards",
    # Counts/Attempts
    "target",
    "reception",
    "carry",
    "pass_attempt",
    "completion",
    # Touchdowns
    "touchdowns",
    "passing_td",
    "rushing_td_count",
    "receiving_td_count",
]

# =============================================================================
# Full Player Stats (includes all drive stats plus game-level only stats)
# =============================================================================

NFL_PLAYER_STATS: List[str] = [
    # --- Core stats (also in PLAYER_DRIVE_STATS) ---
    "receiving_yards",
    "rushing_yards",
    "passing_yards",
    "target",
    "reception",
    "carry",
    "pass_attempt",
    "completion",
    "touchdown",
    "passing_td",
    "rushing_td_count",
    "receiving_td_count",
    
    # --- Situational usage (game-level) ---
    "red_zone_target",
    "red_zone_carry",
    "goal_to_go_target",
    "goal_to_go_carry",
    # Situational rushing (carry context)
    "short_yard_carry",       # carries with <=2 yards to go
    "third_down_carry",       # carries on third down

    # --- Snap counts / participation (game-level) ---
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",

    # --- Pre-snap route participation (game-level) ---
    "ps_route_participation_pct",
    "ps_route_participation_plays",
    "ps_targets_total",
    "ps_targets_slot_count",
    "ps_targets_wide_count",
    "ps_targets_inline_count",
    "ps_targets_backfield_count",
    "ps_targets_slot_share",
    "ps_targets_wide_share",
    "ps_targets_inline_share",
    "ps_targets_backfield_share",
]

# Windows for rolling computations (at game level)
# Integers = N most recent games, "season" = current season, "lifetime" = all history
ROLLING_WINDOWS = [1, 3, 5, "season"]

# Contexts for rolling computations
# "vs_any"     = all opponents
# "with_team"  = career with current team only (uses ctx='with_team' in daily_totals cache)
# "vs_team"    = games vs specific opponent; features are gated on support in RollingWindow
ROLLING_CONTEXTS = ["vs_any", "with_team", "vs_team"]

# Aggregation levels for rolling windows
# Only per-game rolling is currently supported; drive-level cache not wired
ROLLING_LEVELS = ["game"]

# Curated rolling feature surface (trimmed to well-populated, high-signal stats)
ROLLING_FEATURE_STATS: List[str] = [
    "touchdowns",
    "target",
    "carry",
    "pass_attempt",
    "red_zone_target",
    "red_zone_carry",
    "goal_to_go_target",
    "goal_to_go_carry",
    "short_yard_carry",
    "third_down_carry",
    "receiving_yards",
    "rushing_yards",
    "passing_yards",
    # Pre-snap realized stats (for historical projections only)
    "ps_route_participation_pct",
    "ps_route_participation_plays",
    "ps_targets_total",
    "ps_targets_slot_count",
    "ps_targets_wide_count",
    "ps_targets_inline_count",
    "ps_targets_backfield_count",
    "ps_targets_slot_share",
    "ps_targets_wide_share",
    "ps_targets_inline_share",
    "ps_targets_backfield_share",
    "ps_total_touches",
    "ps_scripted_touches",
    "ps_scripted_touch_share",
]

# Columns required for player identification and time-series operations
PLAYER_ID_COLS = ["player_id", "player_name"]
GAME_ID_COLS = ["game_id", "game_date", "season", "week"]
TEAM_ID_COLS = ["team", "opponent"]

# Feature naming convention:
# {window}g_{stat}_per_{level}[_vs_team]
#
# Examples:
#   - 1g_receiving_yards_per_game          (last 1 game, all opponents)
#   - 3g_rushing_yards_per_game_vs_team    (last 3 games vs this opponent)
#   - seasong_touchdowns_per_game          (season-to-date)
#   - lifetimeg_passing_yards_per_drive    (career avg per drive)
