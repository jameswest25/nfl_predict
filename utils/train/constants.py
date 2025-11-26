# utils/train/constants.py
"""
Constants for NFL Model Training

NFL-specific constants for model training and evaluation.
All paths should be imported from utils.general.paths.
"""

from utils.general.constants import IDENTIFIER_COLUMNS, NFL_TARGET_COLUMNS
from utils.general.paths import (
    FINAL_FEATURES_DIR as PROCESSED_DIR,
    FINAL_FEATURES_PARQUET as FEATURES_PARQUET,
)

# Feature categories for documentation and analysis
FEATURE_CATEGORIES = {
    'recency': [
        # Rolling stats columns (populated dynamically)
    ],
    'usage': [
        # Target share, carry share, etc.
    ],
    'temporal': [
        'days_rest',
        'travel_distance',
    ]
}

# Data quality thresholds
MIN_PROBABILITY_SUM = 0.999  # Strict tolerance for probability validation
MAX_PROBABILITY_SUM = 1.001

# Memory optimization settings
FLOAT_PRECISION = 'float32'  # Use 32-bit floats to save memory
INT_PRECISION = 'int16'      # Use 16-bit ints for small integers

# Multicollinearity threshold
CORRELATION_THRESHOLD = 0.95  # Remove features with correlation > 95%
