"""Rolling window feature utilities.

This subpackage contains modules for computing rolling window features:
- rolling_window: Main rolling window feature computation
- daily_totals: Daily totals cache for efficient rolling calculations
- stats: Rolling feature stat definitions
- pace: Pace-related metrics computation
"""

from utils.feature.rolling.rolling_window import add_rolling_features
from utils.feature.rolling.daily_totals import build_daily_cache_range
from utils.feature.rolling.stats import ROLLING_FEATURE_STATS, ROLLING_WINDOWS, ROLLING_CONTEXTS
from utils.feature.rolling.pace import compute_pace_metrics, add_pace_history

__all__ = [
    "add_rolling_features",
    "build_daily_cache_range",
    "ROLLING_FEATURE_STATS",
    "ROLLING_WINDOWS",
    "ROLLING_CONTEXTS",
    "compute_pace_metrics",
    "add_pace_history",
]

