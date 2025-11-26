# NFL Anytime TD Prediction

## Key Scripts
- `pipeline/feature.py` – builds the player-game feature matrix, enforcing as-of leakage guards.
- `pipeline/train.py` – trains the multi-stage modeling graph (availability → usage → efficiency → TD).
- `pipeline/predict.py` – runs inference with the latest artifacts.
- `scripts/eval/odds_backtest.py` – compute odds-only baselines across horizons.
- `scripts/backtest/run_backtest.py` – evaluate existing prediction columns (AUC/Brier/log-loss) and write baseline summaries.
- `scripts/backtest/compare_baselines.py` – diff two baseline JSON files and highlight regressions.
- `scripts/analysis/horizon_eval.py` – consolidate metrics across decision horizons (input multiple prediction files labelled by horizon).
- `scripts/analysis/weather_travel_impact.py` – stratify TD hit rates by weather/travel features and emit plots (`docs/analysis/weather_travel/`) + JSON summaries.
- `scripts/monitoring/asof_coverage_report.py` – summarize snapshot coverage, write Markdown to `docs/monitoring/asof_coverage.md`, and append metrics history to `output/metrics/coverage/history.parquet`.

## Monitoring
- `docs/monitoring.md` documents leak guards, snapshot coverage expectations, roster snapshot regeneration steps, and instructions for running odds/backtests & analytics scripts.
- `docs/analysis/weather_travel.md` lists the latest weather/travel diagnostic artifacts.

## Quick Start
1. Build features:
   ```bash
   PYTHONPATH=. ./venv/bin/python pipeline/feature.py --start 2024-09-01 --end 2024-12-31
   ```
2. Train models:
   ```bash
   PYTHONPATH=. ./venv/bin/python pipeline/train.py --config config/training.yaml
   ```
3. Predict upcoming games:
   ```bash
   PYTHONPATH=. ./venv/bin/python pipeline/predict.py --date 2024-12-24 --days 3
   ```

See `docs/monitoring.md` for ongoing maintenance workflows (roster snapshots, coverage dashboards, odds/availability backtests).

