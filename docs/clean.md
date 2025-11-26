# CLEAN_MODULE_GUIDE.md

The `clean.py` script is the first processing step in the pipeline. Its primary responsibility is to take the raw, messy CSV data (potentially enriched with weather data) and produce a clean, standardized, and correctly-typed dataset, ready for feature engineering.

## Execution

To run the cleaning script, execute the following command from the project's root directory:

```bash
python pipeline/clean.py
```

-   **Input:** Reads from `data/raw/pitch_level_data.csv`.
-   **Output:** Writes to `data/processed/cleaned_pitch_data.parquet`.

## Key Operations

This script performs several critical cleaning and transformation steps, orchestrated within the `clean_raw` function in `pipeline/clean.py`.

## Derived NFL Columns

The cleaning pass standardises a set of NFL-specific per-play metrics so downstream feature builders can rely on consistent naming:

- Offensive volume: `target`, `reception`, `carry`, `pass_attempt`, `completion`, `touchdown`.
- Situational splits:
  - `red_zone_target` / `red_zone_carry` – defined by `yardline_100 ≤ 20`.
  - `goal_to_go_target` / `goal_to_go_carry` – driven by the play’s `goal_to_go` flag.

Each column is stored as an `int8` flag so they aggregate cleanly inside Polars and the rolling-window machinery.

### 1. Initial Loading and Filtering
- **Loading:** The process begins by loading the entire raw dataset into a pandas DataFrame.
- **Dropping Unused Columns:** A predefined list of columns is dropped immediately. These include identifiers (`player_name`, `pitcher_1`, `fielder_2`, etc.), descriptive fields not used in modeling (`des`), and redundant or high-cardinality fields (`sv_id`, `umpire`). This step significantly reduces the memory footprint.
- **Filtering Pitch Types:** Only standard pitches are kept. Intentional balls (`IN`) and pitch-outs (`PO`) are removed as they do not represent typical batter-pitcher interactions.

### 2. Type Conversion
- **Datetime Conversion:** The `game_date` column is converted from a string to a proper datetime object, which is essential for all time-based calculations, such as recency weighting.
- **Numeric Conversion:** All columns that are expected to be numeric are coerced into numeric types. An important list of columns (e.g., `release_speed`, `launch_angle`, `pfx_x`) are converted to `float32` to save memory while retaining necessary precision.

### 3. Creating Core Dependent Variables (Labels)
This is a critical step where the target variables for a potential model are created.
- **`is_hit`**: A binary (1/0) flag is created. It is set to 1 if the `events` column corresponds to one of the defined `HIT_TYPES` (e.g., 'single', 'double', 'home_run').
- **`is_2_plus_base_hit`**: A binary flag for extra-base hits. It is set to 1 for events in `TWO_PLUS_BASE_HIT_TYPES` ('double', 'triple', 'home_run').
- **`bases`**: An integer column representing the value of a hit. It is mapped directly from the `events` column (e.g., 'single' -> 1, 'double' -> 2). For non-hit events, the value is 0.

### 4. Handling Missing Values (NA)
- A significant number of missing values are handled in this step. The `DataTracer` log shows a large reduction in `na_count` at this stage.
- Many columns related to batted ball events (e.g., `launch_speed`, `launch_angle`) are naturally null for non-contact pitches (strikes, balls). These are filled with 0. This is a modeling decision that treats non-contact events as having zero impact in these feature columns.
- Other miscellaneous `NaN`s are filled where appropriate.

### 5. Saving Output
- The final, cleaned DataFrame is saved to `data/processed/cleaned_pitch_data.parquet`.
- Parquet is chosen over CSV for its superior performance: it offers faster read/write speeds, smaller file sizes, and, crucially, it preserves data types, preventing ambiguity in subsequent pipeline steps. 