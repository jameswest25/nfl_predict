Overall approach
Goal: enumerate concrete, code-level steps to bring labels, usage/TD composition, availability, roles, horizons, and rolling history closer to “real football reality”, with an eye toward higher AUC/PR‑AUC and better calibration.
Scope: plan only (no edits now), but each item will specify: files to touch, data/feature changes, model/config changes, and tests/metrics to validate.

---

Train / Inference Feature Parity Contract
----------------------------------------

Purpose: Ensure that any feature used by the models is computed in **exactly the same way** at training time and inference time, without ad-hoc reimplementations or one-off fixes.

Core principles
---------------

1. **Single source of truth for feature logic**
   - All non-trivial feature transformations must live in shared helpers under `utils/feature/` and be reused by both:
     - Training pipeline (`pipeline/feature.py`)
     - Prediction pipeline (`pipeline/predict.py`)
   - Examples of shared helpers that now enforce parity:
     - `utils/feature/shared.py`
       - `finalize_drive_history_features`
       - `attach_td_rate_history_features` (team/opp & position TD rates)
     - `utils/feature/offense_context.py`
       - `add_offense_context_features_training`
       - `add_offense_context_features_inference`
     - `utils/feature/rolling_window.py` + `utils/feature/stats.py`
       - `add_rolling_features` with `ROLLING_FEATURE_STATS`, `ROLLING_WINDOWS`, `ROLLING_CONTEXTS`
     - `utils/feature/weather_features.py`
       - `add_weather_forecast_features_training` / `add_weather_forecast_features_inference`
     - `utils/feature/odds.py`
       - `add_nfl_odds_features_to_df` (shared between train/predict)
     - `utils/feature/team_context.py`
       - `compute_team_context_history`, `add_team_context_features`

2. **Pipelines call helpers, not inline logic**
   - `pipeline/feature.py` (training) is responsible for:
     - Loading `player_game_by_week` and related artifacts.
     - Calling shared helpers in a clear order (team context, offense context, weather, odds, rolling, TD rates, hist_* shares, etc.).
   - `pipeline/predict.py` (inference) is responsible for:
     - Building the prediction scaffold (schedule + rosters).
     - Re-using the same shared helpers on that scaffold, with only differences being:
       - As-of cutoffs (no future leakage).
       - Training-specific labels/targets, which are **never** recomputed at inference time.
   - Inline, one-off feature code in either pipeline is discouraged; if you find yourself writing non-trivial transforms in `pipeline/feature.py` or `pipeline/predict.py`, move them into `utils/feature/` instead.

3. **As-of safety and history windows**
   - Any helper that uses history must:
     - Accept or derive a cutoff (`decision_cutoff_ts` or `max(game_date)` in the current frame).
     - Restrict its history reads to `<= cutoff` (no future games, odds, or weather).
   - Examples:
     - `attach_td_rate_history_features` filters `player_game_by_week` by `game_date <= max(game_date in enriched)` before computing TD rates.
     - Weather and odds helpers perform as-of joins on snapshot timestamps vs a cutoff column.

4. **Symmetric null handling / defaults**
   - Where null handling or defaults are required (e.g. rate denominators are zero, no history exists yet), they must be:
     - Implemented inside shared helpers, not ad-hoc in one pipeline.
     - Applied identically in train and predict.
   - Acceptable patterns:
     - `pl.when(denom > 0).then(num / denom).otherwise(0.0)` for rates and shares.
     - `fill_null(0.0)` on component counts used to define per-game flags (e.g. “has TD”).
   - Unacceptable patterns:
     - Inference-only magic constants (e.g. default `team_implied_total = 22.5`).
     - Train-only fill strategies that are not mirrored at inference.

5. **Strict parity test as a guardrail**
   - `tests/verify_parity_strict.py` is the canonical check that:
     - Runs `_build_feature_matrix_internal` for a specific slate (e.g. 2024-11-10).
     - Rebuilds team context and offense context histories from `player_game_by_week`.
     - Runs `_compute_features` (prediction pipeline) on a scaffold built from the same slate/player-games.
     - Normalizes TD rates and offense-context shares on both frames using:
       - `attach_td_rate_history_features`
       - The same share formulas as `add_offense_context_features_inference`
     - Compares only the features actually used by the flat and structured anytime-TD models (as defined in `config/training.yaml`) and asserts exact equality.
   - This script should be updated whenever:
     - New shared helpers are introduced for high-value feature families.
     - New anytime-TD feature prefixes or individual features are added to `training.yaml`.

How to add a new feature without breaking parity
------------------------------------------------

1. **Put the logic in a shared helper**
   - Create or extend a module under `utils/feature/`, e.g.:
     - `utils/feature/my_feature.py` with `add_my_feature(df, *, mode="train"/"predict")`
     - Or mirror the offense-context pattern:
       - `add_my_feature_training(df, history_path=...)`
       - `add_my_feature_inference(df, history_path=..., cutoff_column=...)`

2. **Wire it into both pipelines**
   - In `pipeline/feature.py`:
     - Call the helper from `_build_feature_matrix_internal` after required upstream dependencies are available.
   - In `pipeline/predict.py`:
     - Call the same helper from `_compute_features` at the corresponding stage.

3. **Enforce as-of safety**
   - If the feature depends on historical tables:
     - Filter those tables to `game_date <= max(game_date for the slate)` in training, or `<= cutoff_ts` in inference, inside the shared helper.
   - Avoid using any field that is only known post-game (labels, same-game outcomes).

4. **Update training config and parity test**
   - In `config/training.yaml`:
     - If the new feature is used by the anytime-TD models, ensure its prefix or name is included in:
       - `feature_prefixes_to_include` or `other_features_to_include` for the relevant problem blocks.
   - In `tests/verify_parity_strict.py`:
     - The parity script already scopes to the anytime-TD feature sets from `training.yaml`. If your feature is exposed via those config blocks, it will be covered automatically.

5. **Rely on the strict parity test before trusting new features**
   - After wiring a new shared helper into train and predict, run:
     - `./venv/bin/python tests/verify_parity_strict.py`
   - Do not consider the new feature “ready” for production until:
     - The parity script reports `✅ STRICT PARITY ACHIEVED` for the anytime-TD problems.
1. Fix anytime‑TD label semantics to match the true betting product
1.1. Define a “skill TD only” label version
Intent: anytime TD = player personally scores a TD (rush/rec, maybe returns), excluding passing TDs for QBs.
Files:
utils/feature/labels.py
utils/feature/targets.py
utils/general/constants.py (if it references target label lists)
Steps:
Add new label version in LABEL_VERSIONS:
e.g. v2_any_skill with:
primary="anytime_td_skill"
LabelRules(include_rush=True, include_rec=True, include_pass_thrown=False, include_other=<decide if you want returns>)
Aliases: "anytime_td": "anytime_td_skill" for this version; define td_count_skill analog if needed.
Ensure compute function emits new columns:
Update compute_td_labels to:
Add td_count_skill (if you want counts for rush+rec only).
Create anytime_td_skill and other variants as per rules.
Keep v1_any_offense for backwards compatibility:
Keep its current semantics but document explicitly: “includes passing TDs, for QB‑centric analyses.”
1.2. Wire the new label into the training config
Files:
config/training.yaml
pipeline/train.py (for require_target_column usage and label_version handling)
Steps:
In the problems: section, modify final anytime‑TD problem:
name: 'anytime_td'
target_col: 'anytime_td'
label_version: 'v2_any_skill'
Ensure any problems that rely on TD counts (efficiency_tds) are:
Either pointed at skill-only counts (td_count_skill) or
Explicitly parameterized to use td_count_offense vs skill counts, with clear comments.
In pipeline/train.py, where self.target_col and self.label_version are used:
Make sure calls to require_target_column pass the per‑problem label_version.
Add defensive logging when the resolved alias isn’t what we expect (e.g., “anytime_td → anytime_td_skill under v2_any_skill”).
1.3. Label generation in feature pipeline
Files:
utils/feature/player_game_level.py (where base TD components and labels are computed)
pipeline/feature.py
Steps:
Confirm player_game_level produces base TD components: rushing_td_count, receiving_td_count, passing_td, touchdowns.
Insert or update a call to compute_td_labels(df, version=<label_version>):
For training feature builds, call with the same label_version as in training.yaml (use DEFAULT_LABEL_VERSION or explicit arg from config).
Ensure the produced columns include anytime_td_skill and anytime_td alias.
1.4. Testing & validation
Files:
tests/labels/test_anytime_td_semantics.py (new)
Potentially extend tests/labels or tests/output
Tests:
Synthetic player‑game rows:
Only rushing TD → label=1.
Only receiving TD → label=1.
Only passing TD → label=0.
Mix of all three → still 1.
Assert that for v2_any_skill, anytime_td alias == anytime_td_skill.
Regression test to ensure old v1_any_offense semantics unchanged.
2. Make usage / TD composition share‑aware and team‑consistent
2.1. Reframe usage models to predict shares instead of absolute counts
Files:
config/training.yaml (usage_targets, usage_carries problem definitions)
utils/feature/game_level.py and/or player_game_level.py (where team usage aggregates live)
pipeline/train.py (any post‑processing relying on absolute target/carry predictions)
Steps:
Define labels for usage share:
At player‑game level, compute:
targets_team_total and carries_team_total per (season, week, team, game_id).
Labels:
target_share_label = target / max(targets_team_total, eps)
carry_share_label = carry / max(carries_team_total, eps).
Modify problem configs:
For usage_targets:
target_col: 'target_share_label'
Clarify comments that model predicts share, not raw count.
For usage_carries:
target_col: 'carry_share_label'
Feature alignment:
Keep all existing hist_*_share_prev/l3 and team context features; they become even more natural predictors for share.
Inference reconstruction:
In pipeline/train.py or a dedicated post‑processing utility:
At inference time, compute:
expected_team_pass_plays (from pred_team_pace and team pass rate features).
expected_team_rush_plays.
Reconstruct:
pred_usage_targets = target_share_pred * expected_team_pass_plays
pred_usage_carries = carry_share_pred * expected_team_rush_plays.
2.2. Enforce consistency between team‑level volume and per‑player expected opportunities
Files:
pipeline/train.py (_derive_expected_features)
Steps:
After reconstructing pred_usage_targets/pred_usage_carries:
Group by (season, week, game_id, team) and compute:
Sum of predicted targets/carries vs expected_team_pass_plays / expected_team_rush_plays.
Compute per‑team scaling ratio:
scale_targets = expected_team_pass_plays / max(sum_pred_targets, eps)
scale_carries = expected_team_rush_plays / max(sum_pred_carries, eps).
Rescale individual predictions within each team group:
pred_usage_targets *= scale_targets
pred_usage_carries *= scale_carries.
Recompute:
expected_targets, expected_carries, expected_opportunities, etc. from these rescaled values.
2.3. Role‑aware team total adjustment
Files:
utils/train/team_total.py
pipeline/train.py (_maybe_wrap_team_total_adjuster and config initialization)
Steps:
Extend TeamTotalConfig to include:
weight_features: List[str] (optional list of columns used to compute intra‑team weights).
Modify adjust_probabilities:
For each group (team/week):
Build a per‑row weight based on:
off_ctx_player_red_zone_share, off_ctx_player_goal_to_go_share
hist_target_share_prev/l3, hist_carry_share_prev/l3
expected_opportunities.
Normalize weights so that sum(weight_i) = 1 within group.
Instead of pure uniform scaling, define:
Effective per‑row scale = base_scale * (1 + alpha*(weight_i - 1/N)) (where N is team size), or directly weight the adjustments so star players absorb more up/down‑scaling.
In pipeline/train.py where TeamTotalConfig is constructed:
Populate weight_features with the most stable, leak‑safe role and usage share signals.
Make alpha a config parameter (e.g., team_total_role_intensity).
2.4. Tests and diagnostics
Files:
New tests in tests/train/test_team_total_adjustment.py
Update or add tests/train/test_usage_share_consistency.py
Checks:
For synthetic teams, verify:
Sum of expected_targets ≈ expected_team_pass_plays.
Scaling moves more probability into high‑share players.
For real data:
Add diagnostics/plots (in output/error_analysis scripts) comparing:
Pre/post team‑total consistency.
Calibration by team implied total bucket.
3. Deepen availability / injury realism
3.1. Explicit “expected snaps” latent
Files:
config/training.yaml (availability_snapshare problem)
utils/feature/player_game_level.py
Steps:
In player‑game aggregation, compute:
offense_snaps and offense_pct as you already do.
Add a derived label:
expected_snaps_label = offense_snaps (for availability_snapshare model), or possibly offense_snaps_active if you have that.
Update availability_snapshare configuration:
Consider predicting snaps instead of percentage, or add a parallel problem availability_snaps.
At inference:
Compute:
pred_expected_snaps = pred_availability * pred_snaps_if_active (if still modeling two stages).
Feed expected_snaps into usage and efficiency models as a feature.
3.2. Model uncertainty in availability
Files:
utils/feature/player_game_level.py (or an injury aggregation module)
pipeline/train.py (feature derivation)
Steps:
From injury_inactive_probability:
Create explicit features:
availability_uncertainty = prob * (1 - prob) or min(prob, 1-prob) to peak at 0.5.
A binned version (high/medium/low).
In _derive_expected_features:
Compute:
uncertainty_weighted_snaps = expected_snaps * (1 - availability_uncertainty).
Or pass both expected_snaps and availability_uncertainty so downstream models can treat “fragile volume” differently.
Tests:
Unit tests to ensure:
These features are present and correctly transformed.
No leakage from post‑cutoff injury info (still relying on as‑of metadata).
4. Richer role modeling and scripted vs unscripted usage
4.1. Standalone role classifiers
Files:
New module: utils/feature/roles.py
config/training.yaml (add problems)
Steps:
Define role labels at player‑game level:
role_is_primary_red_zone_target
role_is_secondary_red_zone_target
role_is_goal_line_back
Potential “gadget” role.
Label generation:
Using past few games (or rolling stats) per team:
Rank players by red‑zone targets, goal‑to‑go carries, etc.
Assign roles based on rank thresholds and minimum volume.
Add new classification problems in training.yaml:
role_red_zone_target, role_goal_line_back, etc.
Train these role models separately and persist predictions as pred_role_* features.
4.2. Integrate role predictions into usage & TD models
Files:
config/training.yaml
pipeline/train.py (_derive_expected_features)
Steps:
For usage_targets / usage_carries / efficiency_tds / anytime_td:
Add pred_role_* to input_predictions and feature_prefixes_to_include (e.g., pred_role_).
Optionally:
Build interaction features: pred_role_goal_line_back * expected_carries.
4.3. Scripted vs unscripted usage
Files:
utils/feature/drive_level.py and player_drive_level.py
utils/feature/player_game_level.py
Steps:
In drive‑level aggregation:
Mark early drives by script (e.g., first X offensive plays, or first N drives).
Aggregate scripted_targets, scripted_carries, scripted_td_drives.
In player‑game frame:
Expose:
scripted_target_share, scripted_carry_share.
Optionally, define a dedicated problem:
pre_snap_scripted_td_chance (regression or classification).
Feed scripted signals into anytime_td and efficiency_tds via feature_prefixes_to_include (e.g., scripted_, off_ctx_ interactions).
5. Horizon‑specific modeling (1.5h vs 6h vs 12h, etc.)
5.1. Problem duplication per horizon
Files:
config/training.yaml
pipeline/feature.py (build_feature_matrix already produces multiple horizons)
Steps:
For the final anytime‑TD problem:
Either:
Create separate problems: anytime_td_h1p5, anytime_td_h6, etc., each pointing to the appropriate features file (by cutoff label), or
Keep one problem but train multiple models per cutoff label, using the same problem name but different run tags.
Ensure ModelTrainer can:
Loop over cutoff‑specific feature matrices or accept a cutoff_label override.
5.2. Horizon‑aware features
Files:
pipeline/feature.py
utils/feature/asof_metadata.py
Steps:
Add a feature decision_horizon_hours (already being written) and ensure it is:
Available as an input feature to horizon‑agnostic models.
Optionally, encode:
Snapshot coverage differences per horizon (e.g., odds/forecast coverage flags) so the model can understand information regimes.
6. Enrich rolling context beyond vs_any where data supports it
6.1. Enable with_team contexts
Files:
utils/feature/stats.py
utils/feature/rolling_window.py
Steps:
In ROLLING_CONTEXTS, add "with_team" where coverage exists.
Confirm your daily totals cache stores team and opponent so the ctx keys can be built for with_team.
Start with a subset of stats:
e.g., targets, carries, receiving_yards, rushing_yards, TD‑related stats.
In pipeline/feature.py, keep call to add_rolling_features the same but now with extended contexts.
6.2. Optional: selective vs_team where dense
Files:
rolling_window.py (no changes if general mechanism works)
A small pre‑analysis script under scripts/analysis/ to check sample size per (player, opponent).
Steps:
Only enable vs_team if n_games_vs_team >= k (e.g. 4) for a player.
Could be implemented as a filter in daily totals generation or by zeroing out features for sparse combos.
7. Evaluation and monitoring aligned with business metrics
7.1. Metric config
Files:
config/training.yaml
utils/train/metrics.py
scripts/eval/ (if present)
Steps:
For anytime_td:
Emphasize:
auc, pr_auc, brier_score, log_loss.
Precision/recall at relevant thresholds (e.g., implied odds buckets).
Ensure your evaluation pipeline:
Logs PR‑AUC and calibration curves per:
Team implied total buckets,
Position groups, and
Horizon.
7.2. Diagnostics specific to new changes
Files:
output/error_analysis scripts (or new ones)
Checks:
Compare:
Label base rates before/after label semantics change.
Per‑team sum of predicted anytime‑TD probabilities vs implied totals before/after share‑aware adjustments.
PR‑AUC and Brier score per position and per horizon.