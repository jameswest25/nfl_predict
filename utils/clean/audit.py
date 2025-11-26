"""
Audit utility for the NFL clean pipeline step.
Runs comprehensive data quality checks on cleaned NFL data.
"""
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ------------------------------------------------------------------
# Columns with *expected* sparsity (NFL-specific)
# key   : column name (str) — or tuple of names if they share logic
# value : dict(reason=str, acceptable_missing=float)  0–1 range
# ------------------------------------------------------------------
EXPECTED_SPARSE = {
    # Player-specific fields - only present on certain play types
    ('passer_player_id', 'passer_player_name', 'passing_yards', 'air_yards'): {
        'reason': 'Only present on passing plays',
        'acceptable_missing': 0.50
    },
    
    ('rusher_player_id', 'rusher_player_name', 'rushing_yards'): {
        'reason': 'Only present on rushing plays',
        'acceptable_missing': 0.70
    },
    
    ('receiver_player_id', 'receiver_player_name', 'receiving_yards', 'yards_after_catch'): {
        'reason': 'Only present on pass plays with a target',
        'acceptable_missing': 0.60
    },
    
    # Advanced metrics - not always available
    ('cp', 'cpoe', 'xyac_epa', 'xyac_success', 'xyac_fd'): {
        'reason': 'Advanced passing metrics not available on all plays',
        'acceptable_missing': 0.55
    },
    
    # Penalty information
    ('penalty_player_id', 'penalty_player_name', 'penalty_yards', 'penalty_type'): {
        'reason': 'Only present on plays with penalties',
        'acceptable_missing': 0.95
    },
    
    # Player involvement fields
    ('assist_tackle_1_player_id', 'assist_tackle_2_player_id'): {
        'reason': 'Tackle assists don\'t occur on every play',
        'acceptable_missing': 0.80
    },
}

def is_expected_sparse(col_name: str, miss_frac: float) -> tuple[bool, str]:
    """
    Return (should_ignore, note) for a given column.
    
    Args:
        col_name: Column name to check
        miss_frac: Missing fraction (0.0 to 1.0)
        
    Returns:
        (should_ignore: bool, reason: str)
    """
    for key, meta in EXPECTED_SPARSE.items():
        if isinstance(key, tuple):
            if col_name in key:
                return True, meta['reason']
        elif col_name == key:
            return True, meta['reason']
    return False, ""

def run_clean_audit(data_file_path=None, chunk_size=250000):
    """
    Run comprehensive audit on cleaned NFL data, using chunking for large files.
    """
    print("🔍 Running Cleaned NFL Data Audit...")
    
    from utils.general.paths import CLEAN_AUDIT_DIR
    audit_dir = CLEAN_AUDIT_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if data_file_path is None:
            # Auto-detect daily-partitioned folder
            cleaned_dir = Path("data/cleaned")
            if cleaned_dir.exists():
                data_file_path = cleaned_dir
            else:
                raise FileNotFoundError("data/cleaned directory not found")

        # Use memory-efficient chunked audit
        audit_report = run_chunked_clean_audit(str(data_file_path), chunk_size)
        
        latest_file = audit_dir / "latest_cleaned_audit.txt"
        with open(latest_file, 'w') as f:
            f.write(audit_report)
        
        timestamped_file = audit_dir / f"cleaned_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(timestamped_file, 'w') as f:
            f.write(audit_report)
        
        print(f"✅ Audit complete! Reports saved to:")
        print(f"   - {latest_file}")
        print(f"   - {timestamped_file}")
        
        return audit_report
        
    except FileNotFoundError as e:
        print(f"❌ Audit failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Audit failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

def should_ignore_sparse_column(col_name: str) -> bool:
    """Check if a column is expected to be sparse and should be ignored in audit warnings."""
    is_sparse, _ = is_expected_sparse(col_name, 0.0)
    return is_sparse

def run_chunked_clean_audit(file_path: str, chunk_size: int) -> str:
    """
    Memory-efficient chunked audit for NFL data.
    
    Args:
        file_path: Path to cleaned data directory or file
        chunk_size: Number of rows per chunk
        
    Returns:
        Audit report as string
    """
    from pathlib import Path
    
    p = Path(file_path)
    
    # Find all parquet files
    if p.is_dir():
        parquet_files = sorted(p.glob("season=*/week=*/date=*/part.parquet"))
        if not parquet_files:
            parquet_files = sorted(p.glob("**/*.parquet"))
    else:
        parquet_files = [p]
    
    if not parquet_files:
        return f"❌ No parquet files found in {file_path}"
    
    # Initialize aggregates
    total_rows = 0
    column_stats = {}
    
    # Process chunks
    for pq_file in parquet_files:
        try:
            df = pd.read_parquet(pq_file)
            total_rows += len(df)
            
            for col in df.columns:
                if col not in column_stats:
                    column_stats[col] = {
                        'count': 0,
                        'null_count': 0,
                        'dtype': str(df[col].dtype)
                    }
                
                column_stats[col]['count'] += len(df)
                column_stats[col]['null_count'] += df[col].isna().sum()
                
        except Exception as e:
            print(f"Warning: Could not process {pq_file}: {e}")
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("NFL CLEANED DATA AUDIT REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("")
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"Total Rows: {total_rows:,}")
    report_lines.append(f"Total Columns: {len(column_stats)}")
    report_lines.append(f"Files Processed: {len(parquet_files)}")
    report_lines.append("")
    
    # Column coverage summary
    high_coverage = sum(1 for stats in column_stats.values() if (stats['count'] - stats['null_count']) / stats['count'] >= 0.95)
    low_coverage = sum(1 for stats in column_stats.values() if (stats['count'] - stats['null_count']) / stats['count'] < 0.50)
    
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 40)
    report_lines.append(f"High Coverage Fields (≥95%): {high_coverage}")
    report_lines.append(f"Low Coverage Fields (<50%): {low_coverage}")
    report_lines.append("")
    
    # Detailed column stats
    report_lines.append("COLUMN COVERAGE DETAILS")
    report_lines.append("-" * 40)
    for col, stats in sorted(column_stats.items()):
        coverage_pct = ((stats['count'] - stats['null_count']) / stats['count']) * 100
        report_lines.append(f"{col}: {coverage_pct:.1f}% coverage ({stats['dtype']})")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF AUDIT REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    run_clean_audit()
