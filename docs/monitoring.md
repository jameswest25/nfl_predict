# Monitoring & Sanity Checks

## Leak Guards
- `tests/test_training_config_features.py` ensures `columns_to_discard` never overlaps with `other_features_to_include` and enforces snapshot coverage thresholds.
- `tests/test_data_leak_guard.py` blocks leak-prone features (explicit and via prefixes) from entering problems such as `anytime_td`.
- `tests/test_offense_context.py` verifies the vacated-usage helper never relies on un-timestamped roster status.
- `pipeline/feature.build_feature_matrix` guards all snapshots (`injury`, `roster`, `odds`, `weather`) and logs the percentage of rows filtered when snapshots are missing or post-cutoff.

## Snapshot Coverage Workflow
1. Collect ESPN roster snapshots periodically:
   ```bash
   PYTHONPATH=. ./venv/bin/python utils/collect/espn_roster_snapshots.py --seasons 2024 2025
   ```
2. Rebuild `data/processed/asof_metadata.parquet` via `pipeline/feature.py`.
3. Generate coverage report + historical metrics:
   ```bash
   PYTHONPATH=. ./venv/bin/python scripts/monitoring/asof_coverage_report.py
   ```
   - Markdown output: `docs/monitoring/asof_coverage.md`
   - Historical time series: `output/metrics/coverage/history.parquet`
4. Investigate alerts whenever coverage drops below:
   - Injury/Roster/Weather: 90%
   - Odds: 60% (until feed improvements land)

## Odds & Availability Backtests
- **Odds-only baseline**: `scripts/eval/odds_backtest.py --score-column market_anytime_td_prob --tag cutoff_v2`
- **Full model backtest**: `scripts/backtest/run_backtest.py --problem anytime_td=pred_anytime_td --season 2024 --tag release_candidate`
- **Diff vs prior release**: `scripts/backtest/compare_baselines.py --current <new>.json --previous <old>.json`

## Horizon & Feature Impact Analytics
- `scripts/analysis/horizon_eval.py` aggregates AUC/Brier/log-loss across decision horizons (1.5h/3h/6h).
- `scripts/analysis/weather_travel_impact.py` plots hit rates by `weather_bad_passing_flag`, `roof_is_indoor_flag`, `travel_rest_days`, etc., storing charts under `docs/analysis/weather_travel/`.

## Importance Monitoring
- Training now emits JSON summaries at `output/metrics/.../importance_highlights/`, emphasizing new feature families (`drive_hist_`, `role_`, `weather_`, `travel_`).

Keep these tooling scripts on a cron or CI schedule to detect regressions before pushing new models live. Whenever coverage or baseline metrics regress, halt releases until data feeds are rebuilt and backtests stabilize.***

