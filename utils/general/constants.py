"""
Shared constants for NFL Prediction Pipeline.

Centralized location for team mappings, column definitions, and utility functions
used across multiple modules to avoid duplication.

This module is the SINGLE SOURCE OF TRUTH for column definitions.
config/training.yaml should mirror LEAK_PRONE_COLUMNS for YAML-based configuration,
validated by tests/test_training_config_features.py.
"""

from __future__ import annotations

from typing import Dict, Final, Tuple

# =============================================================================
# NFL Team Abbreviation Mappings
# =============================================================================

TEAM_ABBR_TO_NAME: Final[Dict[str, str]] = {
    # NFC
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "DAL": "Dallas Cowboys",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "LA": "Los Angeles Rams",
    "LAR": "Los Angeles Rams",
    "MIN": "Minnesota Vikings",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "PHI": "Philadelphia Eagles",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "WAS": "Washington Commanders",
    # AFC
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DEN": "Denver Broncos",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "LVR": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "NE": "New England Patriots",
    "NYJ": "New York Jets",
    "PIT": "Pittsburgh Steelers",
    "TEN": "Tennessee Titans",
    # Historical / alternate abbreviations seen in archives
    "OAK": "Las Vegas Raiders",
    "SD": "Los Angeles Chargers",
    "STL": "Los Angeles Rams",
}

# Reverse mapping: full name -> abbreviation (uses canonical abbreviations)
TEAM_NAME_TO_ABBR: Final[Dict[str, str]] = {
    name.upper(): abbr for abbr, name in TEAM_ABBR_TO_NAME.items()
}


def normalize_team(team_val: str | None) -> str:
    """Return the OddsAPI-recognised full team name for various inputs.

    Accepts team abbreviations (KC) or full names. Falls back to original.
    """
    if not team_val:
        return team_val or ""
    team_val = team_val.strip()
    if team_val.upper() in TEAM_ABBR_TO_NAME:
        return TEAM_ABBR_TO_NAME[team_val.upper()]
    # Handle certain common variations
    if team_val.lower().startswith("kansas city"):
        return "Kansas City Chiefs"
    return team_val


# =============================================================================
# NFL Teams with Retractable Roofs or Domes
# =============================================================================

ROOF_CLOSED_TEAMS: Final[Dict[str, bool]] = {
    "ARI": True,  # State Farm Stadium (retractable)
    "ATL": True,  # Mercedes-Benz Stadium (fixed)
    "DAL": True,  # AT&T Stadium (retractable)
    "DET": True,  # Ford Field (fixed)
    "HOU": True,  # NRG Stadium (retractable)
    "IND": True,  # Lucas Oil Stadium (retractable)
    "LAR": True,  # SoFi Stadium (fixed)
    "LVR": True,  # Allegiant Stadium (fixed)
    "MIN": True,  # U.S. Bank Stadium (fixed)
    "NO": True,   # Caesars Superdome (fixed)
}


# =============================================================================
# Leak-Prone Columns (Same-Game Outcomes)
# =============================================================================
# Stats that leak same-game outcomes and must be excluded from training features.
# These represent what happened in the game we're trying to predict.
#
# This is the AUTHORITATIVE definition. config/training.yaml mirrors this list
# using YAML anchors for per-problem configuration. Keep them in sync!
# Validated by: tests/test_training_config_features.py

LEAK_PRONE_COLUMNS: Tuple[str, ...] = (
    # Yardage (same-game outcomes)
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    # Usage counts (same-game outcomes)
    "pass_attempt",
    "completion",
    "carry",
    "target",
    "reception",
    # Situational usage (same-game outcomes)
    "red_zone_target",
    "red_zone_carry",
    "goal_to_go_target",
    "goal_to_go_carry",
    # Touchdowns (same-game outcomes)
    "passing_td",
    "rushing_td_count",
    "receiving_td_count",
    "touchdowns",
    "td_count",
    "touchdown",
    "touchdown_player_id",
    # Snap counts (same-game outcomes)
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
)


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_cutoff_label(hours: float) -> str:
    """Format cutoff horizon hours into a compact label.
    
    Examples:
        1.5 hours -> "h090m"
        2.0 hours -> "h02h"
        6.0 hours -> "h06h"
    """
    minutes = int(round(hours * 60))
    if minutes % 60 == 0:
        return f"h{int(round(hours)):02d}h"
    return f"h{minutes:03d}m"


# =============================================================================
# Data Quality Tokens
# =============================================================================

MISSING_VALUE_TOKENS: frozenset[str] = frozenset({
    "none", "error", "null", "unknown", "n/a", ""
})

# =============================================================================
# Shared model/feature constants
# =============================================================================

NFL_TARGET_COLUMNS: Tuple[str, ...] = (
    'anytime_td',
    'td_count',
    'receiving_yards',
    'rushing_yards',
    'passing_yards',
    'target',
    'carry',
    'reception',
    'offense_pct',
)

IDENTIFIER_COLUMNS: Tuple[str, ...] = (
    'game_date',
    'game_id',
    'player_id',
    'player_name',
    'team',
    'opponent',
    'season',
    'week',
)


__all__ = [
    # Team mappings
    "TEAM_ABBR_TO_NAME",
    "TEAM_NAME_TO_ABBR",
    "normalize_team",
    "ROOF_CLOSED_TEAMS",
    # Column definitions
    "LEAK_PRONE_COLUMNS",
    # Formatting
    "format_cutoff_label",
    # Data quality
    "MISSING_VALUE_TOKENS",
    # Shared features
    "IDENTIFIER_COLUMNS",
    "NFL_TARGET_COLUMNS",
]

