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
    "touchdown",
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
ROLLING_WINDOWS = [1, 2, 3, 4, "season", "lifetime"]

# Contexts for rolling computations
# "vs_any" = all opponents, "vs_team" = specific opponent only
ROLLING_CONTEXTS = ["vs_any", "vs_team", "with_team"]

# Aggregation levels for rolling windows
# "game" = per-game rates, "drive" = per-drive rates
ROLLING_LEVELS = ["game", "drive"]

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
#   - seasong_touchdown_per_game           (season-to-date)
#   - lifetimeg_passing_yards_per_drive    (career avg per drive)
