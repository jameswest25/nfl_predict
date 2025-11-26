Pre-Snap Feature Semantics
==========================

Scope of the namespaces added in the Tier-2 pass to keep train vs. inference aligned:

- `ps_game_*`: realized same-game pre-snap stats (labels/diagnostics only).
- `ps_hist_*`: historical projections derived from rolling windows and shifted one game; these are the only pre-snap inputs consumed by models.
- `ps_tracking_*` (base columns) and `ps_hist_tracking_*` (shifted): coverage signals describing whether tracking data exists and whether a baseline was used.
- `ps_baseline_source`: string diagnostic used at inference (`actual`, `baseline`, `none`).

Coverage signals
----------------
- `ps_tracking_team_dropbacks`: copy of `ps_team_dropbacks` for coverage; not a feature input but rolled into `ps_hist_tracking_team_dropbacks_*`.
- `ps_tracking_has_game_data`: 1 if either dropbacks or touches were observed for the player; shifted into `ps_hist_tracking_has_game_data_*` for modeling.
- `ps_tracking_has_actual`: row-level flag derived during prediction to indicate that true tracking features were present.
- `ps_tracking_used_baseline`: row-level flag indicating a carry-forward baseline fill was applied.
- `ps_baseline_source`: qualitative source indicator, never used as a model feature.

Scripted play heuristic
-----------------------
- Scripted plays are now: the first 20 offensive snaps **and** within the first ~20 minutes of game time, excluding two-minute/hurry-up situations and late downs.
- Exclusions:
  - Two-minute-like situations (Q2/Q4 with < 2:00 remaining).
  - No-huddle plays deep into a quarter (< 4:00 remaining).
  - 3rd/4th downs (kept to 1st/2nd down to avoid obvious situational scripting breaks).
- Resulting labels: `ps_game_scripted_touches`, `ps_game_scripted_touch_share` (for targets/carries), with historical projections carried under the `ps_hist_*` namespace.

Usage in models
---------------
- Only `ps_hist_*` features (including the `ps_hist_tracking_*` coverage rolls) are allowed as inputs via `feature_prefixes_to_include`.
- Base `ps_game_*` and `ps_tracking_*` columns remain in the data for diagnostics/labeling but are blocked by `columns_to_discard` and `LEAK_PRONE_COLUMNS`.

Decision-time odds snapshots
----------------------------
- Odds features are now filtered to a single horizon via `odds_horizon` in `config/training.yaml` (e.g., `cutoff` keeps the decision-time snapshot and drops `_2h/_6h/_24h/_open` variants).
- Feature builds emit a schema snapshot under `audit/feature/schema/anytime_td/` capturing dtypes and any leak-guard flags for traceability.
