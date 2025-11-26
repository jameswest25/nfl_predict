# Weather & Travel Impact Analysis

The weather/travel diagnostics are generated via
`scripts/analysis/weather_travel_impact.py`. The script reads the merged feature
matrix (`data/processed/final/processed.parquet`), computes hit rates by key
flags (e.g., `weather_bad_passing_flag`, `roof_is_indoor_flag`,
`travel_rest_days`), and saves plots + JSON summaries.

## Usage

```bash
PYTHONPATH=. ./venv/bin/python scripts/analysis/weather_travel_impact.py \
    --feature-path data/processed/final/processed.parquet \
    --target anytime_td
```

- Plots → `docs/analysis/weather_travel/`
- Summaries → `output/metrics/weather_travel/weather_travel_<timestamp>.json`

The JSON payload can be folded into monitoring dashboards or referenced when
tuning the weight of new weather/travel features in meta-models.

