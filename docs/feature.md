# NFL Feature Engineering Pipeline

This pipeline turns cleaned NFL play-by-play into player-ready feature matrices. It is implemented in `utils/feature/` and orchestrated by `pipeline/feature.py`.

## Stage Overview
- **Play Level (`utils/feature/play_level.py`)**
  - Scans daily cleaned Parquet partitions.
  - Filters to regular-season plays (`season_type == "REG"`).
  - Emits weekly partitions with scoring attribution (`touchdown_player_id`), venue metadata (`stadium_key`, `stadium_name`, `stadium_tz`), and derived columns (`target`, `carry`, `pass_attempt`, etc.).

- **Player-Drive Level (`utils/feature/player_drive_level.py`)**
  - Aggregates play rows into per-player / per-drive totals.
  - Attributes touchdowns to the actual scorer before summing.
  - Joins weekly roster metadata (position, depth chart order, injury status) when available from `nfl_data_py.import_weekly_rosters`.

- **Player-Game Level (`utils/feature/player_game_level.py`)**
  - Builds one row per player/game with passing, rushing, and receiving totals.
  - Re-uses the touchdown attribution logic to keep QBs from inheriting receiving TD credit.
  - Enriches with roster context the same way as the drive-level output.

- **Daily Totals & Rolling Windows**
  - `utils/feature/daily_totals.py` caches per-player daily stats for fast lookups.
  - `utils/feature/rolling_window.py` now supports true N-game windows (1g/2g/3g/4g) alongside season and lifetime spans without leaking current-game stats.

## Key Columns
- `touchdown_player_id` – player who actually scored on the play.
- `stadium_key` / `stadium_name` / `stadium_tz` – supports neutral-site weather and venue-aware features.
- `season_type` – retained for downstream auditing; non-regular-season rows are excluded from feature builds.
- Roster fields (`position`, `position_group`, `depth_chart_position`, `depth_chart_order`) – added when roster data is available. Inactive designations released after the 90-minute cutoff are intentionally excluded from the model input.
- `decision_horizon_hours` – annotated horizon (in hours before kickoff) that the feature matrix was built for. The pipeline now materialises multiple horizons (e.g., 90 minutes, 3 hours, 6 hours, 12 hours) in a single run so that training and evaluation can compare performance across decision windows.
- Situational usage columns:
  - `red_zone_target` / `red_zone_carry` – targets and rush attempts snapped inside the opponent’s 20.
  - `goal_to_go_target` / `goal_to_go_carry` – usage on declared goal-to-go downs.
  - `target_share`, `carry_share`, `pass_attempt_share`, `red_zone_target_share`, `red_zone_carry_share`, `goal_to_go_target_share`, `goal_to_go_carry_share` – per-player proportions of team volume computed during aggregation.

## Weather & Venue Support
- Neutral and international venues are defined in `data/raw/config/stadium_coords.json` and normalised through `stadium_key`.
- Weather attachment respects `stadium_key` so London, Germany, and Mexico games use the correct coordinates and timezone.

## Odds Features (Roadmap)
`utils/feature/odds_features.py` now backfills the earliest available “open” snapshot (up to five days before kickoff) before computing deltas, falling back to cached schedule odds only when the API misses a snapshot.

## Maintenance Notes
- Roster enrichment is best-effort; failures are logged and do not abort the build.
- Weekly rosters are cached under `cache/feature/rosters/` (one Parquet per season). The pipeline reuses cached seasons and raises if required roster columns are missing so depth information remains consistent.
- Rolling-window caches depend on `cache/feature/daily_totals`; clear that directory if you change stat definitions.
- Always run `pipeline/feature.py` after collecting/cleaning to keep derived tables in sync.

