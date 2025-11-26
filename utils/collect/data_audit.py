#!/usr/bin/env python3
"""
Comprehensive Data Audit Module

Provides detailed analysis of every field in the NFL play-by-play dataset including:
- Coverage percentages and missing value patterns
- Uniqueness and cardinality analysis
- Value distributions and spreads
- Outlier detection for numeric fields
- Expected vs actual values based on nflfastR specifications
- Data quality recommendations

Exports:
    - audit_dataset() -> Dict: Complete dataset audit
    - audit_field(df, field_name) -> Dict: Individual field analysis
    - generate_audit_report() -> str: Human-readable report
"""

import logging
import warnings
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, date

from utils.general.paths import PROJ_ROOT
from utils.collect.parquet_io import load_data

logger = logging.getLogger(__name__)

# Expected value ranges and constraints for NFL play-by-play data (nflfastR)
# Expected value ranges and constraints for NFL play-by-play data (nflfastR)
NFL_FIELD_SPECS = {
    # Game identifiers
    'game_id': {'type': 'string', 'description': 'Unique game identifier (format: YYYY_WW_AWAY_HOME)'},
    'old_game_id': {'type': 'string', 'description': 'Legacy game identifier'},
    'season': {'type': 'int', 'min': 1999, 'max': 2030, 'description': 'NFL season year'},
    'season_type': {'type': 'categorical', 'values': ['REG', 'POST', 'PRE'], 'description': 'Season type'},
    'week': {'type': 'int', 'min': 1, 'max': 22, 'description': 'Week number (18 reg + 4 post)'},
    'game_date': {'type': 'datetime', 'min_year': 1999, 'max_year': 2030, 'description': 'Game date'},
    
    # Teams
    'home_team': {'type': 'categorical', 'expected_count': 32, 'description': 'Home team abbreviation'},
    'away_team': {'type': 'categorical', 'expected_count': 32, 'description': 'Away team abbreviation'},
    'posteam': {'type': 'categorical', 'expected_count': 32, 'description': 'Possession team'},
    'defteam': {'type': 'categorical', 'expected_count': 32, 'description': 'Defensive team'},
    
    # Game state
    'qtr': {'type': 'int', 'min': 1, 'max': 5, 'description': 'Quarter (5 = OT)'},
    'down': {'type': 'int', 'min': 1, 'max': 4, 'description': 'Down'},
    'ydstogo': {'type': 'int', 'min': 0, 'max': 99, 'description': 'Yards to go for first down'},
    'yardline_100': {'type': 'int', 'min': 0, 'max': 100, 'description': 'Yards from opponent endzone'},
    'game_seconds_remaining': {'type': 'int', 'min': 0, 'max': 3600, 'description': 'Seconds remaining in game'},
    'half_seconds_remaining': {'type': 'int', 'min': 0, 'max': 1800, 'description': 'Seconds remaining in half'},
    'game_half': {'type': 'categorical', 'values': ['Half1', 'Half2', 'Overtime'], 'description': 'Half of game'},
    
    # Score
    'home_score': {'type': 'int', 'min': 0, 'max': 100, 'description': 'Home team score'},
    'away_score': {'type': 'int', 'min': 0, 'max': 100, 'description': 'Away team score'},
    'posteam_score': {'type': 'int', 'min': 0, 'max': 100, 'description': 'Possession team score'},
    'defteam_score': {'type': 'int', 'min': 0, 'max': 100, 'description': 'Defensive team score'},
    'score_differential': {'type': 'int', 'min': -50, 'max': 50, 'description': 'Score differential'},
    
    # Play identifiers
    'play_id': {'type': 'int', 'min': 1, 'max': 10000, 'description': 'Play identifier within game'},
    'drive': {'type': 'int', 'min': 1, 'max': 50, 'description': 'Drive number'},
    'series': {'type': 'int', 'min': 1, 'max': 100, 'description': 'Series number'},
    
    # Play type
    'play_type': {'type': 'categorical', 'values': ['pass', 'run', 'punt', 'field_goal', 'kickoff', 'extra_point', 'qb_kneel', 'qb_spike', 'no_play'], 'description': 'Type of play'},
    'pass': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Pass attempt indicator'},
    'rush': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Rush attempt indicator'},
    
    # Outcomes
    'yards_gained': {'type': 'float', 'min': -50, 'max': 110, 'description': 'Yards gained on play'},
    'touchdown': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Touchdown indicator'},
    'interception': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Interception indicator'},
    'fumble': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Fumble indicator'},
    'fumble_lost': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Fumble lost indicator'},
    'safety': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Safety indicator'},
    'penalty': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Penalty indicator'},
    
    # Pass-specific
    'complete_pass': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Complete pass indicator'},
    'incomplete_pass': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Incomplete pass indicator'},
    'air_yards': {'type': 'float', 'min': -20, 'max': 80, 'description': 'Air yards on pass'},
    'yards_after_catch': {'type': 'float', 'min': -20, 'max': 100, 'description': 'Yards after catch'},
    'sack': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Sack indicator'},
    
    # Players
    'passer_id': {'type': 'string', 'description': 'Passer gsis ID'},
    'passer_player_name': {'type': 'string', 'description': 'Passer name'},
    'rusher_id': {'type': 'string', 'description': 'Rusher gsis ID'},
    'rusher_player_name': {'type': 'string', 'description': 'Rusher name'},
    'receiver_id': {'type': 'string', 'description': 'Receiver gsis ID'},
    'receiver_player_name': {'type': 'string', 'description': 'Receiver name'},
    
    # Advanced metrics
    'epa': {'type': 'float', 'min': -10, 'max': 10, 'description': 'Expected points added'},
    'wp': {'type': 'float', 'min': 0, 'max': 1, 'description': 'Win probability'},
    'wpa': {'type': 'float', 'min': -1, 'max': 1, 'description': 'Win probability added'},
    'success': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Successful play indicator'},
    
    # Situation
    'shotgun': {'type': 'int', 'min': 0, 'max': 1, 'description': 'Shotgun formation indicator'},
    'no_huddle': {'type': 'int', 'min': 0, 'max': 1, 'description': 'No huddle indicator'},
    'qb_dropback': {'type': 'int', 'min': 0, 'max': 1, 'description': 'QB dropback indicator'},
    'qb_scramble': {'type': 'int', 'min': 0, 'max': 1, 'description': 'QB scramble indicator'},
    
    # Venue
    'roof': {'type': 'categorical', 'values': ['dome', 'outdoors', 'closed', 'open'], 'description': 'Stadium roof type'},
    'surface': {'type': 'categorical', 'values': ['grass', 'turf', 'fieldturf', 'astroturf', 'matrixturf'], 'description': 'Playing surface'},
    'temp': {'type': 'float', 'min': -10, 'max': 120, 'description': 'Temperature (F)'},
    'wind': {'type': 'float', 'min': 0, 'max': 50, 'description': 'Wind speed (mph)'},
    
    # Timestamps
    'utc_ts': {'type': 'datetime', 'description': 'UTC timestamp of play (game start)'},
}

# Consume everything via a neutral alias to avoid MLB terminology
FIELD_SPECS = NFL_FIELD_SPECS

def analyze_field_coverage(df: pd.DataFrame, field: str) -> Dict[str, Any]:
    """Analyze coverage and missing value patterns for a field"""
    total_rows = len(df)
    
    if field not in df.columns:
        return {
            'exists': False,
            'coverage_pct': 0.0,
            'missing_count': total_rows,
            'non_null_count': 0,
            'notes': f'Field {field} does not exist in dataset'
        }
    
    series = df[field]
    non_null_count = series.notna().sum()
    missing_count = series.isna().sum()
    coverage_pct = (non_null_count / total_rows) * 100 if total_rows > 0 else 0
    
    # Analyze missing patterns
    missing_patterns = {}
    if missing_count > 0:
        # Missing by game type
        if 'game_type' in df.columns:
            missing_by_game_type = df[series.isna()]['game_type'].value_counts()
            if len(missing_by_game_type) > 0:
                missing_patterns['by_game_type'] = missing_by_game_type.to_dict()
        
        # Missing by date ranges
        if 'game_date' in df.columns:
            missing_df = df[series.isna()]
            if len(missing_df) > 0 and 'game_date' in missing_df.columns:
                date_range = {
                    'earliest': str(missing_df['game_date'].min()),
                    'latest': str(missing_df['game_date'].max())
                }
                missing_patterns['date_range'] = date_range
    
    return {
        'exists': True,
        'coverage_pct': round(coverage_pct, 2),
        'missing_count': missing_count,
        'non_null_count': non_null_count,
        'missing_patterns': missing_patterns
    }

def analyze_field_uniqueness(df: pd.DataFrame, field: str) -> Dict[str, Any]:
    """Analyze uniqueness and cardinality for a field"""
    if field not in df.columns:
        return {'exists': False}
    
    series = df[field].dropna()
    if len(series) == 0:
        return {'exists': True, 'no_data': True}
    
    unique_count = series.nunique()
    total_count = len(series)
    uniqueness_pct = (unique_count / total_count) * 100 if total_count > 0 else 0
    
    # Value counts for categorical analysis
    value_counts = series.value_counts()
    top_values = value_counts.head(10).to_dict()
    
    # Identify if likely categorical vs continuous
    is_likely_categorical = uniqueness_pct < 5 or unique_count < 50
    
    return {
        'exists': True,
        'unique_count': unique_count,
        'total_count': total_count,
        'uniqueness_pct': round(uniqueness_pct, 2),
        'is_likely_categorical': is_likely_categorical,
        'top_values': top_values,
        'has_duplicates': unique_count < total_count
    }

def analyze_numeric_field(df: pd.DataFrame, field: str) -> Dict[str, Any]:
    """Analyze numeric field for distribution, outliers, and range validation"""
    if field not in df.columns:
        return {'exists': False}
    
    series = df[field].dropna()
    if len(series) == 0:
        return {'exists': True, 'no_data': True}
    
    # Convert to numeric if possible
    if not pd.api.types.is_numeric_dtype(series):
        try:
            series = pd.to_numeric(series, errors='coerce')
            series = series.dropna()
        except:
            return {'exists': True, 'not_numeric': True}
    
    if len(series) == 0:
        return {'exists': True, 'no_numeric_data': True}
    
    # Basic statistics
    stats = {
        'min': float(series.min()),
        'max': float(series.max()),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
        'q25': float(series.quantile(0.25)),
        'q75': float(series.quantile(0.75))
    }
    
    # Outlier detection using IQR method
    iqr = stats['q75'] - stats['q25']
    lower_bound = stats['q25'] - 1.5 * iqr
    upper_bound = stats['q75'] + 1.5 * iqr
    
    outliers_low = series[series < lower_bound]
    outliers_high = series[series > upper_bound]
    
    outlier_info = {
        'count_low': len(outliers_low),
        'count_high': len(outliers_high),
        'total_outliers': len(outliers_low) + len(outliers_high),
        'outlier_pct': ((len(outliers_low) + len(outliers_high)) / len(series)) * 100
    }
    
    if len(outliers_low) > 0:
        outlier_info['extreme_low'] = float(outliers_low.min())
    if len(outliers_high) > 0:
        outlier_info['extreme_high'] = float(outliers_high.max())
    
    # Range validation against expected values
    range_validation = {}
    if field in FIELD_SPECS:
        spec = FIELD_SPECS[field]
        if 'min' in spec:
            below_min = series[series < spec['min']]
            range_validation['below_min'] = {
                'count': len(below_min),
                'expected_min': spec['min'],
                'actual_min': stats['min']
            }
        if 'max' in spec:
            above_max = series[series > spec['max']]
            range_validation['above_max'] = {
                'count': len(above_max),
                'expected_max': spec['max'],
                'actual_max': stats['max']
            }
    
    return {
        'exists': True,
        'is_numeric': True,
        'stats': stats,
        'outliers': outlier_info,
        'range_validation': range_validation
    }

def analyze_categorical_field(df: pd.DataFrame, field: str) -> Dict[str, Any]:
    """Analyze categorical field for expected values and data quality"""
    if field not in df.columns:
        return {'exists': False}
    
    series = df[field].dropna().astype(str)
    if len(series) == 0:
        return {'exists': True, 'no_data': True}
    
    value_counts = series.value_counts()
    unique_values = set(value_counts.index)
    
    # Validation against expected values
    validation = {}
    if field in FIELD_SPECS:
        spec = FIELD_SPECS[field]
        
        if 'values' in spec:
            expected_values = set(spec['values'])
            missing_expected = expected_values - unique_values
            unexpected_values = unique_values - expected_values
            
            validation = {
                'expected_values': list(expected_values),
                'actual_values': list(unique_values),
                'missing_expected': list(missing_expected),
                'unexpected_values': list(unexpected_values),
                'has_unexpected': len(unexpected_values) > 0,
                'missing_expected_count': len(missing_expected)
            }
        
        if 'expected_count' in spec:
            expected_count = spec['expected_count']
            actual_count = len(unique_values)
            validation['expected_unique_count'] = expected_count
            validation['actual_unique_count'] = actual_count
            validation['count_matches'] = expected_count == actual_count
    
    # Look for data quality issues
    quality_issues = []
    
    # Check for empty strings
    empty_strings = (series == '').sum()
    if empty_strings > 0:
        quality_issues.append(f"{empty_strings} empty strings")
    
    # Check for suspicious values
    suspicious_patterns = ['unknown', 'null', 'none', 'n/a', 'error', r'\?', 'undefined']
    for pattern in suspicious_patterns:
        try:
            pattern_count = series.str.lower().str.contains(pattern, na=False, regex=True).sum()
            if pattern_count > 0:
                quality_issues.append(f"{pattern_count} values contain '{pattern}'")
        except Exception:
            # If regex fails, try simple string matching
            pattern_clean = pattern.replace(r'\?', '?')
            pattern_count = series.str.lower().str.contains(pattern_clean, na=False, regex=False).sum()
            if pattern_count > 0:
                quality_issues.append(f"{pattern_count} values contain '{pattern_clean}'")
    
    return {
        'exists': True,
        'is_categorical': True,
        'unique_count': len(unique_values),
        'value_counts': value_counts.to_dict(),
        'validation': validation,
        'quality_issues': quality_issues
    }

def analyze_datetime_field(df: pd.DataFrame, field: str) -> Dict[str, Any]:
    """Analyze datetime field for range and consistency"""
    if field not in df.columns:
        return {'exists': False}
    
    series = df[field].dropna()
    if len(series) == 0:
        return {'exists': True, 'no_data': True}
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(series):
        try:
            series = pd.to_datetime(series, errors='coerce')
            series = series.dropna()
        except:
            return {'exists': True, 'not_datetime': True}
    
    if len(series) == 0:
        return {'exists': True, 'no_valid_dates': True}
    
    # Basic datetime statistics
    stats = {
        'earliest': str(series.min()),
        'latest': str(series.max()),
        'span_days': (series.max() - series.min()).days,
        'unique_dates': series.dt.date.nunique()
    }
    
    # Year distribution
    year_counts = series.dt.year.value_counts().sort_index()
    
    # Range validation
    range_validation = {}
    if field in FIELD_SPECS:
        spec = FIELD_SPECS[field]
        if 'min_year' in spec:
            min_year = spec['min_year']
            before_min = series[series.dt.year < min_year]
            range_validation['before_min_year'] = {
                'count': len(before_min),
                'expected_min_year': min_year,
                'actual_min_year': int(series.dt.year.min())
            }
        if 'max_year' in spec:
            max_year = spec['max_year']
            after_max = series[series.dt.year > max_year]
            range_validation['after_max_year'] = {
                'count': len(after_max),
                'expected_max_year': max_year,
                'actual_max_year': int(series.dt.year.max())
            }
    
    return {
        'exists': True,
        'is_datetime': True,
        'stats': stats,
        'year_distribution': year_counts.to_dict(),
        'range_validation': range_validation
    }

def audit_field(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
    """Comprehensive analysis of a single field"""
    
    # Basic field info
    field_info = {
        'field_name': field_name,
        'dtype': str(df[field_name].dtype) if field_name in df.columns else 'missing',
        'specification': FIELD_SPECS.get(field_name, {'description': 'No specification available'})
    }
    
    # Coverage analysis
    coverage = analyze_field_coverage(df, field_name)
    field_info['coverage'] = coverage
    
    if not coverage.get('exists', False):
        return field_info
    
    # Uniqueness analysis
    uniqueness = analyze_field_uniqueness(df, field_name)
    field_info['uniqueness'] = uniqueness
    
    # Type-specific analysis
    if field_name in df.columns:
        dtype = df[field_name].dtype
        
        # Numeric analysis
        if pd.api.types.is_numeric_dtype(dtype) or field_name in ['launch_speed', 'launch_angle', 'temp', 'humidity']:
            numeric_analysis = analyze_numeric_field(df, field_name)
            if numeric_analysis.get('is_numeric'):
                field_info['numeric_analysis'] = numeric_analysis
        
        # Categorical analysis
        if (uniqueness.get('is_likely_categorical') or 
            pd.api.types.is_categorical_dtype(dtype) or 
            dtype == 'object'):
            categorical_analysis = analyze_categorical_field(df, field_name)
            if categorical_analysis.get('is_categorical'):
                field_info['categorical_analysis'] = categorical_analysis
        
        # Datetime analysis
        if (pd.api.types.is_datetime64_any_dtype(dtype) or 
            field_name in ['game_date', 'utc_ts']):
            datetime_analysis = analyze_datetime_field(df, field_name)
            if datetime_analysis.get('is_datetime'):
                field_info['datetime_analysis'] = datetime_analysis
    
    return field_info

def audit_dataset(data_file_path: str = None) -> Dict[str, Any]:
    """Comprehensive audit of an NFL dataset."""
    logger.info("Starting comprehensive dataset audit...")
    
    # Load the dataset
    try:
        if data_file_path:
            # Load from specific file path
            if data_file_path.endswith('.parquet'):
                df = pd.read_parquet(data_file_path)
            elif data_file_path.endswith('.csv'):
                df = pd.read_csv(data_file_path, low_memory=False)
            else:
                raise ValueError(f"Unsupported file format: {data_file_path}")
            logger.info(f"Loaded data from {data_file_path}: {len(df):,} rows")
        else:
            # Use default load_data function
            df = load_data()
            logger.info(f"Loaded data from default location: {len(df):,} rows")
            
        if df.empty:
            return {'error': 'No data found', 'timestamp': datetime.now().isoformat()}
    except Exception as e:
        return {'error': f'Failed to load data: {e}', 'timestamp': datetime.now().isoformat()}

    # Dataset overview
    overview = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'date_range': (
            {
                'earliest': str(df['game_date'].min()),
                'latest': str(df['game_date'].max()),
            } if 'game_date' in df.columns else {'note': 'no game_date; skipping'}
        ),
        'audit_timestamp': datetime.now().isoformat()
    }
    
    # Audit each field
    field_audits = {}
    all_fields = set(df.columns)
    
    for field in sorted(all_fields):
        logger.info(f"Auditing field: {field}")
        field_audits[field] = audit_field(df, field)
    
    # Summary statistics
    summary = {
        'fields_present': len([f for f in field_audits if field_audits[f]['coverage'].get('exists', False)]),
        'fields_missing': len([f for f in field_audits if not field_audits[f]['coverage'].get('exists', False)]),
        'high_coverage_fields': len([f for f in field_audits if field_audits[f]['coverage'].get('coverage_pct', 0) >= 95]),
        'low_coverage_fields': len([f for f in field_audits if 0 < field_audits[f]['coverage'].get('coverage_pct', 0) < 50]),
        'completely_missing_fields': len([f for f in field_audits if field_audits[f]['coverage'].get('coverage_pct', 0) == 0])
    }
    
    return {
        'overview': overview,
        'summary': summary,
        'field_audits': field_audits
    }

def generate_audit_report(data_file_path: str = None) -> str:
    """Generate a human-readable audit report"""
    audit_results = audit_dataset(data_file_path)
    
    if 'error' in audit_results:
        return f"AUDIT FAILED: {audit_results['error']}"
    
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE NFL PLAY-BY-PLAY DATA AUDIT REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {audit_results['overview']['audit_timestamp']}")
    report.append("")
    
    # Overview section
    overview = audit_results['overview']
    report.append("DATASET OVERVIEW")
    report.append("-" * 40)
    report.append(f"Total Rows: {overview['total_rows']:,}")
    report.append(f"Total Columns: {overview['total_columns']}")
    report.append(f"Memory Usage: {overview['memory_usage_mb']} MB")
    if 'date_range' in overview and 'earliest' in overview['date_range']:
        report.append(f"Date Range: {overview['date_range']['earliest']} to {overview['date_range']['latest']}")
    report.append("")
    
    # Summary section
    summary = audit_results['summary']
    report.append("SUMMARY STATISTICS")
    report.append("-" * 40)
    report.append(f"Fields Present: {summary['fields_present']}")
    report.append(f"Fields Missing: {summary['fields_missing']}")
    report.append(f"High Coverage Fields (≥95%): {summary['high_coverage_fields']}")
    report.append(f"Low Coverage Fields (<50%): {summary['low_coverage_fields']}")
    report.append(f"Completely Missing Fields: {summary['completely_missing_fields']}")
    report.append("")
    
    # Field-by-field analysis
    report.append("FIELD-BY-FIELD ANALYSIS")
    report.append("=" * 80)
    
    field_audits = audit_results['field_audits']
    
    # Group fields by coverage level
    high_coverage = []
    medium_coverage = []
    low_coverage = []
    missing_fields = []
    
    for field_name, audit in field_audits.items():
        coverage_pct = audit['coverage'].get('coverage_pct', 0)
        if not audit['coverage'].get('exists', False):
            missing_fields.append(field_name)
        elif coverage_pct >= 95:
            high_coverage.append(field_name)
        elif coverage_pct >= 50:
            medium_coverage.append(field_name)
        else:
            low_coverage.append(field_name)
    
    # Report by coverage level
    for section_name, fields in [
        ("HIGH COVERAGE FIELDS (≥95%)", high_coverage),
        ("MEDIUM COVERAGE FIELDS (50-94%)", medium_coverage), 
        ("LOW COVERAGE FIELDS (<50%)", low_coverage),
        ("MISSING FIELDS", missing_fields)
    ]:
        if fields:
            report.append(f"\n{section_name}")
            report.append("-" * len(section_name))
            
            for field_name in sorted(fields):
                audit = field_audits[field_name]
                coverage = audit['coverage']
                spec = audit['specification']
                
                report.append(f"\n{field_name.upper()}")
                report.append(f"  Description: {spec.get('description', 'No description')}")
                report.append(f"  Data Type: {audit.get('dtype', 'unknown')}")
                
                if coverage.get('exists', False):
                    report.append(f"  Coverage: {coverage['coverage_pct']}% ({coverage['non_null_count']:,} of {coverage['non_null_count'] + coverage['missing_count']:,})")
                    
                    # Uniqueness info
                    if 'uniqueness' in audit:
                        uniq = audit['uniqueness']
                        report.append(f"  Uniqueness: {uniq.get('uniqueness_pct', 0):.1f}% ({uniq.get('unique_count', 0):,} unique values)")
                        
                        if uniq.get('is_likely_categorical'):
                            report.append(f"  Type: Likely Categorical")
                            if 'top_values' in uniq:
                                top_3 = list(uniq['top_values'].items())[:3]
                                report.append(f"  Top Values: {', '.join([f'{k}({v})' for k, v in top_3])}")
                        else:
                            report.append(f"  Type: Likely Continuous")
                    
                    # Numeric analysis
                    if 'numeric_analysis' in audit:
                        num = audit['numeric_analysis']
                        if 'stats' in num:
                            stats = num['stats']
                            report.append(f"  Range: {stats['min']:.2f} to {stats['max']:.2f} (mean: {stats['mean']:.2f})")
                            
                            # Range validation
                            if 'range_validation' in num:
                                range_val = num['range_validation']
                                if 'below_min' in range_val and range_val['below_min']['count'] > 0:
                                    report.append(f"  ⚠️  {range_val['below_min']['count']} values below expected minimum of {range_val['below_min']['expected_min']}")
                                if 'above_max' in range_val and range_val['above_max']['count'] > 0:
                                    report.append(f"  ⚠️  {range_val['above_max']['count']} values above expected maximum of {range_val['above_max']['expected_max']}")
                            
                            # Outliers
                            if 'outliers' in num and num['outliers']['total_outliers'] > 0:
                                outliers = num['outliers']
                                report.append(f"  Outliers: {outliers['total_outliers']} ({outliers['outlier_pct']:.1f}%)")
                    
                    # Categorical analysis
                    if 'categorical_analysis' in audit:
                        cat = audit['categorical_analysis']
                        if 'validation' in cat:
                            val = cat['validation']
                            if 'unexpected_values' in val and val['unexpected_values']:
                                report.append(f"  ⚠️  Unexpected values: {', '.join(val['unexpected_values'][:5])}")
                            if 'missing_expected' in val and val['missing_expected']:
                                report.append(f"  ⚠️  Missing expected values: {', '.join(val['missing_expected'])}")
                        
                        if 'quality_issues' in cat and cat['quality_issues']:
                            for issue in cat['quality_issues']:
                                report.append(f"  ⚠️  Quality issue: {issue}")
                    
                    # Datetime analysis
                    if 'datetime_analysis' in audit:
                        dt = audit['datetime_analysis']
                        if 'stats' in dt:
                            stats = dt['stats']
                            report.append(f"  Date Range: {stats['earliest']} to {stats['latest']} ({stats['span_days']} days)")
                            report.append(f"  Unique Dates: {stats['unique_dates']:,}")
                            
                            if 'range_validation' in dt:
                                range_val = dt['range_validation']
                                if 'before_min_year' in range_val and range_val['before_min_year']['count'] > 0:
                                    report.append(f"  ⚠️  {range_val['before_min_year']['count']} dates before expected minimum year")
                                if 'after_max_year' in range_val and range_val['after_max_year']['count'] > 0:
                                    report.append(f"  ⚠️  {range_val['after_max_year']['count']} dates after expected maximum year")
                else:
                    report.append(f"  Status: MISSING from dataset")
                    if field_name in FIELD_SPECS:
                        report.append(f"  ⚠️  This field is expected but not present")
    
    report.append("\n" + "=" * 80)
    report.append("END OF AUDIT REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)

if __name__ == "__main__":
    # Command line interface
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--json':
            import json
            results = audit_dataset()
            print(json.dumps(results, indent=2, default=str))
        elif sys.argv[1] == '--field' and len(sys.argv) > 2:
            field_name = sys.argv[2]
            df = load_data()
            result = audit_field(df, field_name)
            import json
            print(json.dumps(result, indent=2, default=str))
        else:
            print("Usage: python data_audit.py [--json] [--field FIELD_NAME]")
    else:
        # Generate and print the report
        report = generate_audit_report()
        print(report)
        
        # Also save to file
        output_file = PROJ_ROOT / "data_audit_report.txt"
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}") 