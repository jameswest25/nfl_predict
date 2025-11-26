# train/ module — Architectural README

This document explains the post-refactor training stack: how `train.py` orchestrates a full run, what each utility module in `utils/train/` does, how configuration/CLI overrides work, and how artifacts/metrics are produced. It's written so you (or another engineer/agent) can safely extend/debug the pipeline without spelunking through code.

## High-level flow

```
config/training.yaml ──▶ train.py (ModelTrainer.run)
                          ├─ data.py: Parquet I/O + chronological splits
                          ├─ feature artifact fitting / caching (via data.py)
                          ├─ (optional) tuning.py: hyperparameter tuning with base-config centering
                          ├─ per-model:
                          │   ├─ xgb_utils.py: ES→refit→calibration/thresholding/eval (regular models)
                          │   └─ selective/: trainer.py + gate.py + wrapper.py (selective models)
                          ├─ calibration.py: classification calibration + threshold selection
                          ├─ conformal.py: regression mean calibration + split conformal
                          ├─ metrics.py: confidence binning, regression metrics
                          ├─ (optional) persist.py: slim prediction analysis
                          ├─ persist.py: save model + metrics + artifacts
                          ├─ (post) conformal_composite.py: composite conformal sums
                          └─ run_manifest.py: run manifest (start/end)
```

> **Note on feature hygiene**  
> During artifact fitting we now coerce all datetime features to millisecond floats and persist the list of converted columns per problem. This keeps XGBoost happy (no timezone-aware objects) and ensures inference runs reuse the same transformation.

## Repo layout (training-related)

```
train.py                          # Orchestrator (entry point + CLI)
utils/train/
  ├─ __init__.py                  # Module initialization
  ├─ data.py                      # Parquet I/O + chronological splitting + feature artifacts
  ├─ persist.py                   # Save/load: models/metrics/artifacts/analysis + versioning
  ├─ model_factory.py             # Get estimator instances and task typing helpers
  ├─ xgb_utils.py                 # Early stopping fit, best-iter refit, batched predict_proba, patience
  ├─ metrics.py                   # Confidence binning, regression metrics (tail-aware), etc.
  ├─ calibration.py               # Classification calibration + threshold selection wrappers
  ├─ conformal.py                 # Regression mean calibration + split conformal wrappers
  ├─ conformal_composite.py       # Composite/team conformal sum artifacts (post-run)
  ├─ tuning.py                    # Optuna-based hyperparameter tuning with base-config centering
  ├─ config_types.py              # Dataclasses: Paths, TrainingCfg
  ├─ seed.py                      # Deterministic global seeding helpers (fixed seed per run)
  ├─ run_manifest.py              # Run-ID generation + manifest writing
  ├─ ordinal_ev_integration.py    # Ordinal EV integration utilities
  ├─ ordinal_ev_core.py           # Core ordinal EV functionality
  ├─ thresholds.py                # Threshold selection helpers
  ├─ xgb_utils.py                 # XGBoost-specific utilities
  ├─ model_factory.py             # Model factory for estimator creation
  ├─ purged_group_time_series_split.py # Time-series CV with purging
  ├─ feature_tester.py            # Feature testing framework
  ├─ feature_tester_bridge.py     # Bridge for feature testing
  ├─ rolling_forward_calibration.py # Rolling calibration utilities
  ├─ constants.py                 # Training-related constants
  └─ selective/                   # Selective Risk Minimization components
      ├─ __init__.py              # Package initialization
      ├─ gate.py                  # Tiny MLP gate network + training utilities
      ├─ loss.py                  # Selective loss functions and utilities
      ├─ trainer.py               # EM-like training orchestration + dual-ascent
      ├─ wrapper.py               # SelectiveClassifier wrapper + decision rules
      └─ __pycache__/             # Python bytecode cache
```

The refactored architecture splits functionality into focused, single-responsibility modules. The goal is to make `train.py` read like orchestration logic only, with all heavy lifting delegated to specialized utility modules.

## Configuration

Primary file: `config/training.yaml`. Key sections used by `ModelTrainer`:

```yaml
data:
  feature_matrix_path: data/feature_matrix.parquet
  model_output_dir: artifacts/models
  metrics_output_dir: artifacts/metrics

data_split:
  time_col: game_date
  # any additional split knobs consumed by utils.train.data.split_data_chronologically

training:
  production_mode: false
  models_to_train: [xgboost]          # extendable via model_factory
  hyperparameter_tuning:
    run_tuning: true
    # (tuning.py reads additional tunables per-model)
  calibrate: true                      # classification prob calibration toggle
  calibration_method: isotonic         # e.g. isotonic / platt / histogram
  calibration_cv_folds: 3
  threshold_selection:
    # config for utils.train.calibration.select_threshold family
    # (e.g., maximize F1@support >= k, or Youden's J, etc.)
  regression_mean_calibration:
    enabled: false
  regression_intervals:
    enabled: true
    alpha: 0.10
    method: naive                      # wrapper decides exact approach
  mu_tail_5plus: 5.6                   # used by tail-aware metrics (if configured)
  base_seed: 42
  write_run_manifest: true
  versioning_mode: run_id              # run_id | legacy
  run_tag: ""                          # appended to run_id for grouping
  use_cached_artifacts: true
  save_prediction_analysis: false
  save_clf_report: false               # classification_report + confusion_matrix

problems:
  - name: anytime_td
    target_col: anytime_td
    feature_prefixes_to_include: [1g_, 2g_, 3g_, 4g_, seasong_, lifetimeg_]
    other_features_to_include:
      - season
      - week
    columns_to_discard: []
    # regression_tail (optional): overrides global tail config for metrics
```

**Production mode**: when `training.production_mode: true`, `df_test` is empty by design. The pipeline still trains (with a recent hold-out used for ES and calibration), saves the model + minimal metrics, and skips test evaluation.

## CLI

`train.py` exposes a polished CLI:

### Full training + Ordinal EV (default)
```bash
python train.py --config config/training.yaml
# Runs training + automatically generates ordinal expected value predictions
```

### Full training without Ordinal EV
```bash
python train.py --config config/training.yaml --no-ordinal-ev
# Runs training but skips ordinal EV generation
```

### Select problems/models
```bash
python train.py --problems anytime_td --models xgboost
```

### Production override (no test eval), disable tuning
```bash
python train.py --prod --no-tuning
```

### Run tagging & versioning
```bash
python train.py --run-tag daily_2025_08_20 --versioning run_id
```

### Feature-parameter search (Optuna) for one problem
```bash
python train.py --tune-features --problem anytime_td --n-trials 25
```

### Ordinal EV Only (skip training)
```bash
python train.py --ordinal-only
# Skip training, just generate ordinal EV using latest trained models
```

CLI overrides are plumbed into the `TrainingCfg`/config before the run and logged.

Automated leak checks live in `tests/test_data_leak_guard.py`; CI will fail if a post-cutoff field such as `status` re-enters the training feature list.

## Reproducibility & versioning

### Seeds
A **global deterministic seed** is applied early via `utils.train.seed.set_global_seed(base_seed)`. This covers Python, NumPy, and downstream model seeds (where available). This ensures:

- **Same base_seed ⇒ identical results across all problems and runs**
- **Global consistency** for cross-problem comparisons and debugging
- **Critical fix**: Previously used per-problem seeding which caused inconsistent hyperparameter optimization and performance degradation

**Note**: The original per-problem seeding approach (`deterministic_seed(base_seed, problem_name)`) was found to cause significant performance differences by changing the hyperparameter optimization starting points between problems.

### Run IDs & manifests
Each invocation generates a `run_id` (optionally suffixed by `run_tag`) and writes a manifest at start/end via `utils.train.run_manifest.write_manifest(...)`. The end manifest includes which (problem, model) pairs produced artifacts.

### Versioning mode
`persist.py` uses `versioning_mode`:
- `run_id` → artifacts written under a versioned folder per run.
- `legacy` → flat/non-versioned layout (backwards compatible).

## Data handling

### Loading
- **`utils.train.data.load_feature_matrix(...)`**: Reads Parquet with column projection (pyarrow) when possible; falls back to fastparquet if needed.
- If cached inference artifacts exist, we only read the frozen feature columns + target + meta; otherwise we project by feature prefixes and "other features" first.
- Handles schema reading with fallback mechanisms for robust data loading.

### Chronological splitting
```python
utils.train.data.split_data_chronologically(df, time_col, split_cfg, production_mode)
```

Splits by time order into train / val / test (exact scheme is encapsulated inside the helper; production mode yields empty test).

All downstream logic assumes no shuffling (i.e., time-respecting split to avoid leakage).

### Feature artifact management
The `data.py` module now handles all feature artifact operations:

#### Feature Artifact Fitting
```python
# Build candidate features from prefixes + other_features, drop discarded columns
# Warn about constants but retain them (they might vary later)
# Freeze feature_columns (canonical feature list)
# Record categorical columns and their allowed levels
```

Situational usage features (e.g., `1g_red_zone_target_per_game`, `seasong_target_share`) flow through the same prefix/gating logic as traditional counting stats, so no special configuration is required. Player depth/injury context is sourced from cached weekly roster files in `cache/feature/rosters/`; the pipeline will raise if those caches are missing or malformed to keep roster-aware features trustworthy.

#### Feature Artifact Validation
```json
{
  "feature_columns": [...],
  "categorical_features": [...],
  "category_levels": {col: [levels...]},
  "imputation_values": {...}   # currently empty / placeholder
}
```

#### Cache Validation
- **Structural check**: required keys present
- **Data check**: all `feature_columns` still exist in the input Parquet schema
- **Auto-invalidation**: stale artifacts are deleted and rebuilt

#### Artifact Application
`_apply_feature_artifacts(df, problem_name)` reindexes to the frozen list and re-applies categorical levels for consistent encoding across time slices.



## Hyperparameter tuning (optional)

Controlled by `training.hyperparameter_tuning.run_tuning`.

**Hook**: `utils.train.tuning.tune_hyperparameters(trainer, model_name, problem_config, X_train, y_train, groups_train)`.

### Critical Architecture: Base-Config Centering

**🔴 CRITICAL FIX**: The tuning module implements **base-config centering** to ensure consistent optimization across problems:

```python
# Instead of using tuning range minimums as starting points:
# OLD: max_leaves started at 128 (range min)
# NEW: max_leaves centers around base_config value (512)

if cfg['low'] <= base_value <= cfg['high']:
    # Center the optimization range around the base value
    low = max(cfg['low'], int(base_value * 0.5))
    high = min(cfg['high'], int(base_value * 1.5))
    params[param_name] = trial.suggest_int(param_name, low, high)
```

**Why this matters**: The original implementation caused massive performance differences (0.84 vs 0.91 AUC) by starting optimization at different points for each problem. Base-config centering ensures identical optimization behavior across all problems.

### Parameter Specification
The `tuning.py` module uses a declarative parameter specification:

```python
param_specs = {
    'n_estimators': ('int', 0.5, 1.5),
    'learning_rate': ('float_log', 0.5, 2.0),
    'colsample_bytree': ('float', 0.5, 1.5),
    # ... etc
}
```

### Time-aware CV with Purging
- Uses `PurgedGroupTimeSeriesSplit` for time-aware cross-validation
- Prevents data leakage in time-series optimization
- Groups by calendar day to reduce temporal leakage

Best params are stored on `trainer.best_params` keyed by `"{problem}_{model}"` and picked up by the training loop.

## Training loop per problem

### Modular Architecture Overview
The training loop now delegates to specialized utility modules:

```python
# Data preparation
utils.train.data.load_feature_matrix(...)
utils.train.data.split_data_chronologically(...)
utils.train.data._fit_feature_artifacts(...)
utils.train.data._apply_feature_artifacts(...)

# Hyperparameter tuning
utils.train.tuning.tune_hyperparameters(...)

# Model training & evaluation
utils.train.xgb_utils.fit_model_es(...)
utils.train.xgb_utils.retrain_with_best_iter(...)
utils.train.xgb_utils.predict_proba_batched(...)

# Calibration & conformal
utils.train.calibration.calibrate_and_select_threshold(...)
utils.train.conformal.fit_split_conformal_regression(...)

# Metrics & analysis
utils.train.metrics.compute_regression_metrics(...)
utils.train.metrics.confidence_bins_report(...)
```

### Grouping for time-aware calibration
- Build `groups_train = factorize(df_train[time_col].dt.date)` for tuning.
- Build `_groups_index_map[problem]` on train+val to reindex groups to `X_cal.index` later (used by calibration CV to avoid temporal leakage).

### Merge train+val → train_full
- Join and re-sort by time (if val is empty, copy train).

### For each model in `models_to_train`

1. **Prepare params**.
   - Start from tuned params if available, else default model params from problem config.
   - For XGBoost, bump `n_estimators` high (5k) and reduce histogram memory pressure via `max_bin≤128`.

2. **Hold-out scheme (non-prod)**:
   - Reserve newest 20% of `train_full` → `holdout_full`.
   - Split `holdout_full` into ES (older half) and CAL (newer half).

3. **Downcast numerics to float32**. In classification, cast labels to int8.

4. **Fit with Early Stopping (ES) then refit at best-iter**.
   - **Stage 1**: `utils.train.xgb_utils.fit_model_es(...)` trains with ES against the ES slice; `compute_patience(...)` sets ES patience (reasonable for large `n_estimators`).
   - **Stage 2**: `retrain_with_best_iter(...)` trains a fresh model fixed at `best_iteration`.

5. **Task branches**.

#### Classification

**Calibration/threshold selection (CAL)**
`utils.train.calibration.calibrate_and_select_threshold(...)`:

- Respects `training.calibrate` toggle.
- Supports methods like isotonic/Platt/histogram (config driven).
- Uses `groups_cal` derived from `_groups_index_map` for a time-aware CV during OOF calibration, which yields a robust ECE estimate.
- Computes the decision threshold according to `training.threshold_selection` (Youden, F1@support≥k, etc.).
- Returns a final model wrapper (e.g., `CalibratedClassifierCV` or a small ensemble wrapper) + `(best_thresh, best_method, best_ece)`.

**Evaluation (non-prod test)**
- Predict probs with `predict_proba_batched(...)`, threshold at selected `best_thresh` (falls back to 0.5 if not set).
- **Metrics saved**:
  - `auc`, `pr_auc` (rare event diagnostic), `precision_at_thresh`, `recall_at_thresh`,
  - `brier_score`, `log_loss`,
  - optional `classification_report`, `confusion_matrix`,
  - `confidence_analysis` (bin calibration around the operational threshold),
  - `calibration_method`, `calibration_ece_oof`.
- Optional: saves pre/post-calibration PR curves for auditability.
- Feature importances saved from the base estimator via `_unwrap_base_model(...)`.

**Production mode**
Same ES→refit approach, then calibrate if enabled on the recent CAL slice; no test evaluation. Metrics note clarifies "production run." Feature importances are still saved.

#### Regression

**Optional isotonic mean calibration (CAL)**
`regression_mean_calibrate(model, X_cal, y_cal)` if `regression_mean_calibration.enabled: true` and `len(X_cal)≥200`. This aligns mean predictions to the target scale without touching variance.

**Optional split conformal intervals (CAL)**
`fit_split_conformal_regression(model, X_cal, y_cal, pi_cfg)` if enabled and `len(X_cal)≥100`. The wrapper adds `predict_interval(X)` and holds calibrated residual distributions.

**Evaluation (non-prod test)**
- `rmse`, `mae`, `r2` via `utils.train.metrics.compute_regression_metrics(...)` (tail-aware diagnostics optionally included via `regression_tail` cfg).
- If conformal active: report `pi_coverage`, `pi_width_mean`, `pi_width_median`.
- Feature importances saved (from base estimator).

**Production mode**
Skips classification calibrator; may apply mean calibration + conformal (same gating). No test eval.

## Prediction analysis (optional)

If `training.save_prediction_analysis: true`:

- Loads the full dataset once and merges corresponding test rows (or falls back to the current test slice).
- Writes a "slim" Parquet with the operational columns (classification: `true_label`/`pred_label`/`prob`/...; regression: `true`/`pred`/`errors`/...) plus the exact feature columns used.
- By default this goes under a versioned folder (if using the Stage-4 patch); otherwise under `analysis_dir`.

## Save artifacts

For each (problem, model): save model + metrics + feature importance under a versioned directory (in `run_id` mode).

After finishing all models for a problem, save the inference artifacts again to ensure the final frozen list/levels are available for downstream inference code.

## Composite conformal sums (post-run)

After all problems complete:

`fit_composite_conformal_sums(self)` and `fit_team_conformal_sums(self)` consume the stashed calibration/test predictions for regression problems:

During training we optionally stash predictions + metadata:
- **Calibration store**: indices, `y_cal`, `y_hat_cal`, and meta (`game_pk`, home/away, `inning_topbot`, `time_col` when available).
- **Test store**: indices, `y_test`, `y_hat_test`.

Post-run functions materialize per-game / per-team conformal diagnostics and intervals for sums of component models (e.g., summing EVs across multiple targets).

Artifacts are saved via `persist.py` and obey the same versioning.

## Files written

### Models
Versioned directory per `(run_id / problem / model)`; format depends on the estimator. XGBoost uses its own serializable wrapper (pickle/joblib).

### Metrics
YAML/JSON in the same versioned directory (includes calibration choices, operational threshold, and diagnostics).

### Feature importance
A CSV or JSON depending on the estimator (gain/weight/cover for tree models, coefficients where applicable).

### Inference artifacts
`inference_artifacts_{problem}.joblib` at the problem scope (shared across models).

### Slim prediction analysis (optional)
`preds_slim_*.parquet` under a versioned `analysis/` folder (or flat `analysis_dir` if you kept the original path).

### Manifests
`run_manifest_{run_id}.json` (or similar) with start/end entries, config snapshot bits, and a tiny success rollup per (problem, model).

## Design choices (and why)

- **ES→refit**: Early stopping finds capacity on a chronologically valid ES slice; refitting a fresh model fixed at `best_iteration` avoids leakage from the ES validation state and reduces stochasticity.

- **Chronological hold-out + ES/CAL split**: The newest 20% is never used for parameter tuning; its older half is the ES validation, and the newest half is the calibration slice. This keeps calibration "closest in time" to deployment while still leak-free.

- **🔴 Global deterministic seeds**: Fixed critical performance issue where per-problem seeding caused massive AUC differences (0.84 vs 0.91). Now uses `utils.train.seed.set_global_seed(base_seed)` for identical results across problems.

- **🔴 Base-config centering in tuning**: `utils.train.tuning` centers hyperparameter optimization ranges around base configuration values, ensuring consistent optimization behavior across problems instead of starting at range minimums.

- **Frozen feature list**: `utils.train.data` freezes the feature list once on the training slice; we don't "discover" new columns mid-run (which could cause schema drift). Constants are retained to prevent accidental loss if they later become non-constant in new data.

- **Time-aware grouping in calibration**: Even when OOF calibration is used, groups are built by day to reduce time leakage in probability calibration and threshold selection.

- **Graceful fallbacks**: PyArrow→fastparquet, artifact cache invalidation, and robust logging around non-critical errors improve reliability in nightly/cron runs.

## Extending the system

### Add a new model

1. Add a branch in `utils.train.model_factory.get_model_instance(name, problem_cfg, params)` that returns a configured estimator.
2. Implement `is_classification(problem_cfg)` / `is_regression(problem_cfg)` correctly for the new problem (typically problem-scoped).
3. If the estimator supports early stopping, extend `utils.train.xgb_utils.fit_model_es(...)` to handle it (or add a variant).
4. Provide feature importance extraction in `utils.train.persist.save_feature_importance(...)` (via `_unwrap_base_model(...)` if wrapped).
5. (Optional) Extend `utils.train.tuning` with a search space + time-aware CV for this estimator.

## Selective Risk Minimization Models

**Selective models** implement the [Selective Risk Minimization](https://arxiv.org/abs/1901.09192) framework, which learns to abstain from uncertain predictions to achieve higher precision on accepted examples. These models are particularly valuable for high-stakes predictions where precision is more important than coverage.

### Available Selective Models

| Model Name | Base Estimator | Decision Mode | Use Case |
|------------|----------------|---------------|----------|
| `xgb_selective` | XGBoost Classifier | Argmax (recommended) | High-precision classification with controlled coverage |

### Architecture

```
Input Features → Base Classifier (XGBoost)
                    ↓
              Per-sample CE Loss
                    ↓
              Gate Network (MLP)
                    ↓
         SelectiveClassifier Wrapper
                    ↓
        Argmax Decision: POS/NEG/ABSTAIN
```

**Key Components:**
- **Base Classifier**: XGBoost model that learns to predict the target class
- **Gate Network**: Tiny MLP that learns to predict whether to accept/abstain from each prediction
- **Selective Loss**: Penalizes errors only on accepted examples + coverage constraint penalty
- **Decision Rule**: Argmax competition between positive, negative, and abstain logits

### Configuration

Add to `config/training.yaml`:

```yaml
training:
  selective:
    enabled: true                    # Enable selective training
    coverage_target: 0.15            # Target acceptance rate (0.0-1.0)
    lambda_init: 0.01                # Initial Lagrange multiplier
    lambda_step: 0.1                 # Dual-ascent step size
    outer_rounds: 3                  # EM-like alternation rounds
    gate:
      type: "mlp"                    # Gate architecture type
      hidden_units: [64]             # Hidden layer sizes
      dropout: 0.0                   # Dropout rate
      l2: 0.0                        # Weight decay
      epochs: 3                      # Training epochs per round
      batch_size: 8192               # Batch size
      lr: 0.001                      # Learning rate
      device: "cpu"                  # Device (CPU-only)
    decision:
      use_argmax_abstain: true       # Use argmax (recommended for precision)
      abstain_temp: 1.0              # Temperature for abstain logit scaling

# Per-problem overrides
problems:
  - name: gets_hit
    selective:
      coverage_target: 0.12          # Problem-specific target
```

### Training Process

1. **EM-like Alternation**: Alternate between fitting base classifier and gate network
2. **Sample Weighting**: Gate acceptance probabilities weight base classifier training
3. **Coverage Constraint**: Dual-ascent on Lagrange multiplier enforces coverage target
4. **Temperature Calibration**: Post-hoc calibration ensures consistent coverage across folds

### Decision Modes

**Argmax Mode (Recommended)**:
- Competes: `z_pos`, `z_neg`, `z_abstain`
- Naturally learns precision-optimized abstention
- Better for high-precision applications

**Gate-First Mode**:
- Hard threshold on gate logit: accept if `z_gate > 0`
- Simpler but less flexible than argmax

### Temperature Calibration

Post-hoc calibration ensures consistent coverage across folds/trials:

```python
# Binary search calibrates T so coverage ≈ target_coverage
T_star, achieved_cov = model.calibrate_abstain_temp(X_val, target_coverage)
```

**Benefits:**
- Apples-to-apples precision comparisons across folds
- Consistent coverage regardless of base model confidence distribution
- Robust to varying data distributions

### Evaluation Metrics

Comprehensive evaluation with detailed counters:

```
Accepted 125/500 (abstained 375); pred_pos 15 (rate 0.120), TP=12, FP=3, precision_sel=0.800
```

**Key Metrics:**
- **Coverage**: Fraction of examples accepted
- **Precision_sel**: Precision on accepted examples only
- **Recall_on_accepted**: Recall considering only accepted examples
- **TP/FP/FN**: Confusion matrix on accepted subset

### Usage

**Train selective model:**
```bash
# Via main pipeline
python main.py train --models xgb_selective --problems gets_hit

# Direct train.py call
python pipeline/train.py --models xgb_selective --problems gets_hit
```

**Enable for specific problems:**
```yaml
problems:
  - name: gets_hit
    selective:
      enabled: true
      coverage_target: 0.12
```

### Performance Characteristics

**Advantages:**
- ✅ Higher precision on accepted predictions
- ✅ Controlled coverage via coverage_target
- ✅ Robust to class imbalance
- ✅ No post-hoc thresholding needed

**Trade-offs:**
- 🔸 Lower coverage than non-selective models
- 🔸 More complex training (EM alternation)
- 🔸 Additional hyperparameters to tune

### Troubleshooting

**Coverage too high/low:**
- Adjust `lambda_init` (higher = more conservative gate)
- Increase `outer_rounds` for better convergence
- Check `abstain_temp` scaling

**Poor precision:**
- Ensure `use_argmax_abstain: true`
- Verify temperature calibration is working
- Check gate network capacity (`hidden_units`)

**Training instability:**
- Reduce `lambda_step` for gentler convergence
- Increase `gate.epochs` for better gate fitting
- Verify input features are properly sanitized

### Add a new problem

1. Add an entry in `config/training.yaml` under `problems`.
2. Define `feature_prefixes_to_include` / `other_features_to_include` and `target_col`.
3. (Optional) Add problem-specific params per model (e.g., `xgboost_params`) and tail metrics overrides.

### Add a new calibration/thresholding policy (classification)

1. Extend `utils.train.calibration` with a new method.
2. Wire it into the method switch based on `training.calibration_method`.
3. If it needs new config knobs, add them under `training.threshold_selection` or a sibling key.

### Add a new PI method (regression)

1. Extend `utils.train.conformal` to create a wrapper with `predict` and `predict_interval`.
2. Gate with `training.regression_intervals.method`.

## The tune_features(...) helper

`python train.py --tune-features --problem PROB --n-trials N`:

For each Optuna trial:
1. Calls `pipeline/feature.py` via subprocess with trial params (`--half-life`, `--shrink-k`, `--crude-window`) to regenerate features.
2. Runs a lightweight XGBoost (250 trees, single-threaded) using the same modern artifact fitting procedure (freeze on train; apply to val).
3. Scores validation AUC and returns it to Optuna.
4. Logs best params and value at the end.

**Note**: the trial modifies the shared feature matrix file; thus runs serially (`n_jobs=1`) and sets a per-trial timeout.

## Gotchas / FAQs

**🔴 Why did we switch from per-problem to global seeding?**
The original per-problem seeding caused massive performance differences (0.84 vs 0.91 AUC) because hyperparameter optimization started at different points for each problem. Global seeding ensures identical results across problems and consistent optimization behavior.

**🔴 Why base-config centering in tuning?**
The original tuning approach started optimization at the minimum values of parameter ranges, causing inconsistent optimization behavior. Base-config centering ensures that optimization starts near proven good configurations, leading to consistent and better performance across problems.

**Why do constant columns get kept?**
They're constant in the training window but might vary later; dropping them would silently remove a valid signal post-train. We surface a warning instead.

**Where do "imputation values" come from?**
Placeholder for future encoders/imputers if you introduce models that require explicit imputation. Currently models (e.g., XGBoost) are robust to NaNs; we still version the slot to avoid artifact schema churn later.

**Why factorize groups by date (not game or row)?**
Day-level grouping is a good default for time-aware OOF calibration: it's coarser than row-level (reduces leakage) and less brittle than game-level (avoids group explosion).

**What's in _unwrap_base_model?**
It extracts the core estimator from wrappers: `CalibratedClassifierCV`, scikit Pipelines, the conformal/mean-calibrated regressor, or the small ensemble wrapper used by calibration. This ensures FI and serialization see the true model.

## Quick smoke test

### Dry run (no tuning) on one problem
```bash
python train.py --problems gets_2plus_bases --models xgboost --no-tuning
```

**Confirm**:
- A new `run_id` shows in artifacts.
- Under that, per problem/model: model file + metrics + feature importance.
- `inference_artifacts_gets_2plus_bases.joblib` exists at the problem scope.
- `run_manifest_*` has start and end entries with a simple success roll-up.

## Ordinal Expected Value System

The training pipeline includes an **ordinal expected value (EV) system** that automatically generates sophisticated expected bases predictions using ordinal regression heads.

### What It Does

The ordinal EV system:
1. **Trains 5 ordinal regression heads** for cumulative probabilities:
   - `gets_1_plus_bases`: P(Y ≥ 1) - Any hit
   - `gets_2_plus_bases`: P(Y ≥ 2) - Extra base hits
   - `gets_3_plus_bases`: P(Y ≥ 3) - Triple or HR
   - `gets_4_plus_bases`: P(Y ≥ 4) - HR only
   - `gets_5_plus_bases`: P(Y ≥ 5) - Always 0 (for API consistency)

2. **Combines predictions** into a probability mass function (PMF) over [0,1,2,3,4,5+] bases

3. **Computes expected value**: EV = Σ Pr(Y ≥ k) with proper tail handling

4. **Generates comprehensive output** with PMF, survival probabilities, and expected values

### Default Behavior

**Ordinal EV runs automatically after training completes** (unless disabled):

```bash
# Default: training + ordinal EV
python main.py train

# Disable ordinal EV
python main.py train --no-ordinal-ev

# Force ordinal EV (same as default)
python main.py train --ordinal-ev

# Ordinal EV only (skip training)
python main.py train --ordinal-only
```

### Output Files

Ordinal EV generates files in the versioned metrics directory:
```
output/metrics/{run_id}/ordinal_ev_total_bases_{timestamp}.csv
```

With columns:
- **Metadata**: `game_pk`, `batter`, `batter_name`, `game_date`
- **Probability Mass Function**: `p0`, `p1`, `p2`, `p3`, `p4`, `p5plus`
- **Survival Probabilities**: `S>=1`, `S>=2`, `S>=3`, `S>=4`, `S>=5`
- **Expected Value**: `ev_tb` (pseudo expected total bases)

### Architecture

The ordinal EV system consists of:

- **`utils/train/ordinal_ev_core.py`**: Pure mathematical functions for EV calculation
- **`utils/train/ordinal_ev_integration.py`**: Trainer-aware glue code and orchestration
- **`utils/feature/targets.py`**: Ordinal target column creation in feature pipeline
- **Configuration**: `ordinal_ev` section in `config/training.yaml`

### Key Features

- **Monotonicity Enforcement**: Automatically ensures survival probabilities decrease monotonically
- **Tail Synthesis**: Can synthesize 5+ probabilities if head is missing using configurable ratio
- **Versioned Outputs**: Integrates with existing versioning system
- **Memory Efficient**: Uses existing batch prediction infrastructure
- **Robust Error Handling**: Graceful fallbacks and comprehensive logging

## Dependencies

- Python 3.10+
- pandas / numpy / pyarrow (or fastparquet fallback)
- scikit-learn
- xgboost ≥ 1.6 (for sklearn wrapper with callbacks; we guard capability)
- optuna
- joblib
- yaml (PyYAML)

## Contact / ownership

- **Orchestrator & contracts**: `train.py`
- **Data loading & artifacts**: `utils/train/data.py`
- **Serialization/versioning**: `utils/train/persist.py`
- **Model factory**: `utils/train/model_factory.py`
- **XGBoost utilities**: `utils/train/xgb_utils.py`
- **Metrics & evaluation**: `utils/train/metrics.py`
- **Calibration**: `utils/train/calibration.py`
- **Conformal & intervals**: `utils/train/conformal*.py`
- **Hyperparameter tuning**: `utils/train/tuning.py`
- **Time-series CV**: `utils/train/purged_group_time_series_split.py`
- **Seeding**: `utils/train/seed.py`
- **Run manifests**: `utils/train/run_manifest.py`
- **Ordinal EV core**: `utils/train/ordinal_ev_core.py`
- **Ordinal EV integration**: `utils/train/ordinal_ev_integration.py`
- **Target creation**: `utils/feature/targets.py`

If you alter interfaces in any of these modules, scan call-sites in `train.py` and keep signatures stable to preserve the orchestrator's simplicity.

## Critical Performance Fixes Summary

The refactored training system includes two critical fixes that resolved massive performance inconsistencies:

### 🔴 Fix 1: Global Seeding
- **Problem**: Per-problem seeding caused different hyperparameter optimization starting points
- **Solution**: `utils.train.seed.set_global_seed(base_seed)` ensures identical results across problems

### 🔴 Fix 2: Base-Config Centering in Tuning
- **Problem**: Hyperparameter optimization started at range minimums instead of base config values
- **Solution**: `utils.train.tuning` centers optimization ranges around base configuration values

These fixes ensure the refactored system performs identically to the original while providing much better maintainability and modularity. 