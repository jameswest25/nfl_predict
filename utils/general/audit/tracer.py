import pandas as pd
import logging
from pathlib import Path

# Configure a specific logger for the tracer
trace_log_path = Path(__file__).resolve().parents[2] / 'logs' / 'pipeline_trace.log'
trace_log_path.parent.mkdir(exist_ok=True)

def clear_trace_log():
    """Clears the trace log file."""
    if trace_log_path.exists():
        try:
            with open(trace_log_path, 'w'):
                pass
            logging.getLogger('DataTracer').info("Trace log cleared.")
        except Exception as e:
            logging.getLogger('DataTracer').error(f"Failed to clear trace log: {e}")

# Use append mode for the file handler so that sequential runs add to the log
file_handler = logging.FileHandler(trace_log_path, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

tracer_logger = logging.getLogger('DataTracer')
tracer_logger.setLevel(logging.INFO)
tracer_logger.addHandler(file_handler)
tracer_logger.propagate = False # Prevent logging to the root logger

class DataTracer:
    def __init__(self, existing_df: pd.DataFrame = None):
        if existing_df is not None:
            self._previous_cols = set(existing_df.columns)
            self._previous_rows = len(existing_df)
        else:
            self._previous_cols = set()
            self._previous_rows = 0

    def trace(self, df: pd.DataFrame, step_name: str):
        """
        Logs a detailed summary of the DataFrame's state at a given step.
        """
        if not isinstance(df, pd.DataFrame):
            tracer_logger.info(f"--- TRACE: {step_name} ---")
            tracer_logger.info("  Object is not a DataFrame. Skipping trace.")
            return

        header = f"--- TRACE: {step_name} ---"
        tracer_logger.info(header)

        # --- Shape and Memory ---
        current_rows = len(df)
        current_cols = set(df.columns)
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        row_delta = current_rows - self._previous_rows if self._previous_rows else 0
        
        tracer_logger.info(f"  Shape: ({current_rows:,}, {len(current_cols):,}) | Rows changed: {row_delta:+,} | Memory: {memory_usage:.2f} MB")

        # --- Column Changes ---
        if self._previous_cols:
            added_cols = current_cols - self._previous_cols
            removed_cols = self._previous_cols - current_cols
            if added_cols:
                tracer_logger.info(f"  ++ {len(added_cols)} COLS ADDED: {sorted(list(added_cols))}")
            if removed_cols:
                tracer_logger.info(f"  -- {len(removed_cols)} COLS REMOVED: {sorted(list(removed_cols))}")

        # --- NA Values ---
        total_na = df.isna().sum().sum()
        tracer_logger.info(f"  NA Values: {total_na:,} total")
        if total_na > 0:
            na_per_col = df.isna().sum()
            top_5_na = na_per_col[na_per_col > 0].sort_values(ascending=False).head(5)
            tracer_logger.info("  Top 5 Columns with NAs:")
            for col, count in top_5_na.items():
                pct = (count / current_rows) * 100
                tracer_logger.info(f"    - {col}: {count:,} ({pct:.1f}%)")
        
        # --- Key Column Stats ---
        numeric_cols_to_check = ['launch_speed', 'release_speed', 'pfx_x', 'pfx_z']
        tracer_logger.info("  Key Numeric Stats:")
        for col in numeric_cols_to_check:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                stats = df[col].dropna().describe()
                tracer_logger.info(f"    - {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

        categorical_cols_to_check = ['events', 'pitch_name', 'game_type', 'type']
        tracer_logger.info("  Key Categorical Values:")
        for col in categorical_cols_to_check:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                top_3_values = df[col].value_counts().nlargest(3)
                tracer_logger.info(f"    - {col} (Top 3): {top_3_values.to_dict()}")

        tracer_logger.info("-" * (len(header) - 1))
        
        # Update state for next trace
        self._previous_cols = current_cols
        self._previous_rows = current_rows 