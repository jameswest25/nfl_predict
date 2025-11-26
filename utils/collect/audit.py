"""
Audit utility for the NFL data collection pipeline step.
Runs comprehensive data quality checks on raw NFL play-by-play data.
"""
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.collect.data_audit import NFL_FIELD_SPECS

logger = logging.getLogger(__name__)

def run_collect_audit(df=None, data_file_path=None, chunk_size=100000):
    """
    Run comprehensive audit on collected NFL play-by-play data.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame to audit (if already loaded)
    data_file_path : str, optional
        Path to data file to audit
    chunk_size : int, default 100000
        Chunk size for processing large files
    """
    print("🔍 Running NFL Play-by-Play Data Audit...")
    
    from utils.general.paths import COLLECT_AUDIT_DIR
    audit_dir = COLLECT_AUDIT_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if data_file_path and os.path.exists(data_file_path):
            audit_results = run_chunked_audit(data_file_path, chunk_size)
            audit_report = format_audit_report(audit_results)
        elif df is not None:
            audit_report = generate_comprehensive_audit_report(df)
        else:
            raise ValueError("Either a DataFrame or a valid data_file_path must be provided.")

        latest_file = audit_dir / "latest_raw_audit.txt"
        with open(latest_file, 'w') as f:
            f.write(audit_report)
        
        print(f"✅ Raw data audit completed")
        print(f"📄 Report saved: {latest_file}")
        
        # Print key insights
        # ... (printing logic remains the same)
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during raw data audit: {e}", exc_info=True)
        return False

def run_chunked_audit(file_path: str, chunk_size: int) -> dict:
    """
    Performs a memory-efficient audit on a large data file by processing it in chunks.
    """
    from collections import defaultdict
    
    total_rows = 0
    file_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
    all_cols = sorted(list(set(file_cols) | set(NFL_FIELD_SPECS.keys())))
    
    column_stats = {
        col: {'non_null_count': 0, 'unique_values': set()} for col in all_cols
    }

    reader = pd.read_csv(file_path, chunksize=chunk_size, iterator=True, low_memory=False)
    
    for i, chunk in enumerate(reader):
        print(f"  - Processing chunk {i+1}...")
        total_rows += len(chunk)
        for col in chunk.columns:
            if col in column_stats:
                column_stats[col]['non_null_count'] += chunk[col].count()
                if len(column_stats[col]['unique_values']) < 5000:
                    column_stats[col]['unique_values'].update(chunk[col].dropna().unique())

    field_audits = {}
    for col_name in all_cols:
        stats = column_stats.get(col_name, {})
        spec = NFL_FIELD_SPECS.get(col_name, {})
        exists = col_name in file_cols
        
        non_null = stats.get('non_null_count', 0)
        uniques = stats.get('unique_values', set())
        
        field_audits[col_name] = {
            'coverage': {
                'exists': exists,
                'non_null_count': non_null,
                'missing_count': total_rows - non_null if exists else total_rows,
                'coverage_pct': (non_null / total_rows * 100) if total_rows > 0 and exists else 0
            },
            'uniqueness': {
                'unique_count': len(uniques),
                'uniqueness_pct': (len(uniques) / non_null * 100) if non_null > 0 else 0
            },
            'specification': spec
        }

    return {
        'overview': {'total_rows': total_rows, 'total_columns': len(file_cols), 'audit_timestamp': datetime.now().isoformat()},
        'field_audits': field_audits
    }

def format_audit_report(audit_results: dict) -> str:
    """Formats the results from a chunked audit into a human-readable report."""
    report = []
    overview = audit_results.get('overview', {})
    report.append("=" * 80)
    report.append("NFL PLAY-BY-PLAY DATA AUDIT REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {overview.get('audit_timestamp')}")
    report.append(f"\nTotal Rows: {overview.get('total_rows', 0):,}")
    report.append(f"Total Columns: {overview.get('total_columns', 0)}")
    
    report.append("\nFIELD-BY-FIELD ANALYSIS")
    report.append("=" * 80)
    
    for field_name, audit in audit_results.get('field_audits', {}).items():
        coverage = audit.get('coverage', {})
        uniqueness = audit.get('uniqueness', {})
        spec = audit.get('specification', {})
        
        report.append(f"\n{field_name.upper()}")
        report.append(f"  Description: {spec.get('description', 'N/A')}")
        
        if coverage.get('exists'):
            report.append(f"  Coverage: {coverage.get('coverage_pct', 0):.2f}% ({coverage.get('non_null_count', 0):,} non-null)")
            report.append(f"  Uniqueness: {uniqueness.get('uniqueness_pct', 0):.1f}% ({uniqueness.get('unique_count', 0):,} unique values)")
        else:
            report.append("  Status: MISSING from dataset")
            
    return "\n".join(report)

def generate_comprehensive_audit_report(df: pd.DataFrame) -> str:
    """Generate the full comprehensive audit report directly from DataFrame"""
    from utils.collect.data_audit import audit_field, NFL_FIELD_SPECS
    from datetime import datetime
    
    # Run the full audit
    audit_results = audit_dataset_from_df(df)
    
    # Use the comprehensive report formatting from data_audit.py
    if 'error' in audit_results:
        return f"AUDIT FAILED: {audit_results['error']}"
    
    report = []
    report.append("=" * 80)
    report.append("NFL PLAY-BY-PLAY DATA AUDIT REPORT")
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
    
    # Field-by-field analysis (the valuable part!)
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
    
    # Report by coverage level with full details
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
                    if field_name in NFL_FIELD_SPECS:
                        report.append(f"  ⚠️  This field is expected but not present")
    
    report.append("\n" + "=" * 80)
    report.append("END OF AUDIT REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)

def audit_dataset_from_df(df: pd.DataFrame):
    """Run audit directly on NFL play-by-play DataFrame"""
    from utils.collect.data_audit import audit_field, NFL_FIELD_SPECS
    from datetime import datetime
    
    # Dataset overview
    overview = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'date_range': {
            'earliest': str(df['game_date'].min()) if 'game_date' in df.columns else 'Unknown',
            'latest': str(df['game_date'].max()) if 'game_date' in df.columns else 'Unknown'
        } if 'game_date' in df.columns else {'error': 'No game_date column'},
        'audit_timestamp': datetime.now().isoformat()
    }
    
    # Audit each field
    field_audits = {}
    all_fields = set(df.columns) | set(NFL_FIELD_SPECS.keys())
    
    for field in sorted(all_fields):
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

if __name__ == "__main__":
    run_collect_audit() 