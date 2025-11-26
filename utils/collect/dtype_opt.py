#!/usr/bin/env python3
"""
DataFrame Dtype Optimization

Memory-efficient dtype conversion for DataFrames, specifically optimized for Statcast data.
Converts appropriate columns to categorical and optimized numeric types.

Exports:
    - OptimizationConfig: Configuration dataclass for optimization settings
    - optimize_dtypes(df: pd.DataFrame, config: OptimizationConfig = None) -> pd.DataFrame
    - get_memory_usage(df: pd.DataFrame) -> float
    - maybe_optimize(df: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame
"""

import logging
from dataclasses import dataclass
from typing import Dict, Final, List

import pandas as pd

logger = logging.getLogger(__name__)

# Constants for dtype optimization
CATEGORICAL_COLUMNS: Final[List[str]] = [
    "home_team", "away_team", "batter_name", "pitcher_name", 
    "batter_side", "pitcher_handedness", "pitch_type", "description",
    "events", "bb_type", "if_fielding_alignment", "of_fielding_alignment"
]

NUMERIC_OPTIMIZATIONS: Final[Dict[str, str]] = {
    'game_pk': 'int32',
    'inning': 'int8',
    'at_bat_number': 'int16',
    'pitch_number': 'int8',
    'balls': 'int8',
    'strikes': 'int8',
    'outs_when_up': 'int8',
    'batter': 'int32',
    'pitcher': 'int32'
}

FLOAT_COLUMNS: Final[List[str]] = [
    'release_speed', 'effective_speed', 'release_spin_rate', 
    'launch_speed', 'launch_angle', 'hit_distance_sc',
    'temp', 'humidity', 'windspeed', 'winddir', 'pressure'
]

@dataclass
class OptimizationConfig:
    """Configuration for dtype optimization"""
    enabled: bool = True
    categorical_enabled: bool = True
    categorical_columns: List[str] = None
    numeric_optimizations: Dict[str, str] = None
    float_columns: List[str] = None
    
    def __post_init__(self):
        if self.categorical_columns is None:
            self.categorical_columns = CATEGORICAL_COLUMNS.copy()
        if self.numeric_optimizations is None:
            self.numeric_optimizations = NUMERIC_OPTIMIZATIONS.copy()
        if self.float_columns is None:
            self.float_columns = FLOAT_COLUMNS.copy()

def get_memory_usage(df: pd.DataFrame) -> float:
    """Calculate memory usage of DataFrame in megabytes"""
    return df.memory_usage(deep=True).sum() / 1024**2

def optimize_dtypes(df: pd.DataFrame, config: OptimizationConfig = None) -> pd.DataFrame:
    """Optimize DataFrame dtypes for memory efficiency with configurable optimization settings"""
    if df.empty:
        return df
    
    if config is None:
        config = OptimizationConfig()
    
    if not config.enabled:
        return df
    
    original_memory = get_memory_usage(df)
    
    # Convert categorical columns for memory efficiency
    if config.categorical_enabled:
        for col in config.categorical_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype("category")
                logger.debug(f"Converted {col} to categorical")
    
    # Optimize numeric columns
    for col, dtype in config.numeric_optimizations.items():
        if col in df.columns and df[col].dtype != dtype:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                logger.debug(f"Optimized {col} to {dtype}")
            except (ValueError, OverflowError):
                logger.warning(f"Could not optimize {col} to {dtype}")
    
    # Optimize float columns to float32 where appropriate
    for col in config.float_columns:
        if col in df.columns and df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
            logger.debug(f"Converted {col} to float32")
    
    optimized_memory = get_memory_usage(df)
    memory_savings = original_memory - optimized_memory
    
    if memory_savings > 0:
        logger.info(f"Memory optimization: {original_memory:.1f}MB -> {optimized_memory:.1f}MB "
                   f"(saved {memory_savings:.1f}MB, {memory_savings/original_memory*100:.1f}%)")
    
    return df

def maybe_optimize(df: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame:
    """Conditionally optimize dtypes based on enabled flag"""
    if config.enabled:
        return optimize_dtypes(df, config)
    return df 