0. Design goals (so Opus doesn’t get “creative”)

For each submodel:

snaps

usage_target

usage_carries

usage_target_yards

efficiency_rec_yards_air

efficiency_rec_yards_yac

efficiency_rush_yards

anytime_td_con_rec

anytime_td_con_car

We want to:

Replace one global model with one model per position: RB, WR, TE, QB (only when label makes sense).

Keep the same input features and target per task.

Keep the same output shape/column names going into the final classifier.

Initially retain capping logic (as a safety rail), but separate it from the model itself so it’s easy to remove later.

Add a “shadow” comparison so we know we didn’t nuke performance.

1. High-level refactor pattern

You want to move from this mental model:

# before
model = load_model("usage_target.model")
pred = model.predict(features)


To this:

# after
models = {
    "RB": load_model("usage_target.RB.model"),
    "WR": load_model("usage_target.WR.model"),
    "TE": load_model("usage_target.TE.model"),
    "QB": load_model("usage_target.QB.model"),
}
pos = row["position_group"]  # e.g. RB, WR, TE, QB
pred = models.get(pos, models["WR"]).predict(features_for_row)


Externally, you still just end up with a usage_target_pred column.

So to Opus, the change is:

“Internally, replace single model with a position→model mapping, and route rows by position; externally, nothing changes.”

2. Step-by-step plan for Opus (you can paste this in)

You can give Opus something like this (adapt paths/names as needed):

Task A: Introduce a canonical position_group field

Goal: Every training and prediction row has a normalized position group: RB, WR, TE, QB.

Implementation details:

In the data prep code for NFL player-game rows (where you already have per-player features for these submodels), add a derived column:

def normalize_position(raw_pos: str) -> str:
    raw = (raw_pos or "").upper()
    if raw in {"RB", "HB", "FB"}:
        return "RB"
    if raw in {"WR"}:
        return "WR"
    if raw in {"TE"}:
        return "TE"
    if raw in {"QB"}:
        return "QB"
    # default: treat unknown like WR
    return "WR"


Store this as position_group in your main training dataframe for all submodels.

Add a unit test (or simple assert) that the vast majority of rows fall into these four groups, and log counts by group.

Task B: Split training data per position for ONE submodel first

Pick one submodel as the template — e.g. usage_target.

Before:

Single training dataset: all positions.

One model artifact: usage_target.model.

After:

In the training script for usage_target, instead of one training call, loop over position groups:

POSITION_GROUPS = ["RB", "WR", "TE", "QB"]

for pos in POSITION_GROUPS:
    df_pos = train_df[train_df["position_group"] == pos]
    if len(df_pos) < MIN_ROWS:  # e.g., 500 or 1000
        continue  # optionally skip thin groups like QB for this label

    X_train, y_train = build_features_and_target(df_pos)
    model = train_usage_target_model(X_train, y_train, hyperparams)

    save_model(model, f"usage_target.{pos}.model")


build_features_and_target and train_usage_target_model should be exactly the same as before.

Use the same CV / time split logic as the original single model, just applied to the position-filtered frame.

If a certain label doesn’t apply to a position (e.g. usage_carries for QB), you can either:

Skip training that position_group (no model saved), or

Train but mark it in config as “fallback to RB=0” or similar.

Keep hyperparameters identical to current best model initially, to keep this refactor orthogonal to tuning.

Task C: Change prediction code to route by position

Wherever you currently do:

model = load_model("usage_target.model")
df["usage_target_pred"] = model.predict(X)
df["usage_target_pred"] = cap_usage_target(df["usage_target_pred"])


Change to:

models = {
    pos: load_model(f"usage_target.{pos}.model")
    for pos in ["RB", "WR", "TE", "QB"]
    if model_file_exists(f"usage_target.{pos}.model")
}

def predict_usage_target_for_row(row):
    pos = row["position_group"]
    model = models.get(pos)
    # Fallback logic if position has no trained model:
    if model is None:
        # e.g., default to WR model or global baseline
        model = models.get("WR")

    x_row = extract_features_for_row(row)  # same as before
    raw_pred = model.predict(x_row)[0]
    return raw_pred

df["usage_target_pred_raw"] = df.apply(predict_usage_target_for_row, axis=1)
df["usage_target_pred"] = cap_usage_target(df["usage_target_pred_raw"])


Key points:

New column usage_target_pred_raw stores the uncapped prediction.

usage_target_pred remains the capped one, so downstream code sees no change.

This gives you visibility so you can later decide to remove or relax caps.

Task D: Add a “shadow” comparison to avoid regression

We want to know: did the per-position models actually help?

For the same eval dataset:

Load old global model (unchanged) and compute:

df["usage_target_pred_old"] = old_model.predict(X)
df["usage_target_pred_old"] = cap_usage_target(df["usage_target_pred_old"])


Compute metrics by position and overall:

for pos in ["RB", "WR", "TE", "QB"]:
    d = df[df["position_group"] == pos]
    print(pos, "R2_old:", r2(d["y"], d["usage_target_pred_old"]),
              "R2_new:", r2(d["y"], d["usage_target_pred"]))


Also compute global metrics and maybe MAE/RMSE.

Only switch the production code path to the per-position models once you’ve verified:

New ≥ old metrics for most positions, or at least no obvious collapse.

If a position regresses, you can:

Temporarily keep using the old global model for that position.

Or investigate that specific segment.

This gives you a clear A/B check without touching the final classifier yet.

Task E: Generalize pattern to all submodels

Once usage_target is working cleanly, Opus can:

Abstract the pattern into a utility:

def train_per_position_models(task_name, train_df, train_fn, feature_fn, target_col, positions):
    for pos in positions:
        df_pos = train_df[train_df["position_group"] == pos]
        if len(df_pos) < MIN_ROWS:
            continue
        X, y = feature_fn(df_pos), df_pos[target_col]
        model = train_fn(X, y)
        save_model(model, f"{task_name}.{pos}.model")


Abstract prediction routing:

class PerPositionModel:
    def __init__(self, task_name, positions, feature_fn, cap_fn=None, fallback_pos="WR"):
        self.models = {
            pos: load_model(f"{task_name}.{pos}.model")
            for pos in positions
            if model_file_exists(f"{task_name}.{pos}.model")
        }
        self.feature_fn = feature_fn
        self.cap_fn = cap_fn
        self.fallback_pos = fallback_pos

    def predict_for_df(self, df, out_col):
        def _predict_row(row):
            pos = row["position_group"]
            model = self.models.get(pos) or self.models[self.fallback_pos]
            X_row = self.feature_fn(row)
            return model.predict(X_row)[0]

        df[out_col + "_raw"] = df.apply(_predict_row, axis=1)
        if self.cap_fn is not None:
            df[out_col] = self.cap_fn(df[out_col + "_raw"])
        else:
            df[out_col] = df[out_col + "_raw"]
        return df


For each task, instantiate:

usage_target_model = PerPositionModel(
    task_name="usage_target",
    positions=["RB", "WR", "TE", "QB"],
    feature_fn=build_usage_target_features_for_row,
    cap_fn=cap_usage_target,  # keep for now
    fallback_pos="WR",
)
usage_target_model.predict_for_df(df, out_col="usage_target_pred")


Repeat for:

snaps_pred

usage_carries_pred

usage_target_yards_pred

efficiency_rec_yards_air_pred

efficiency_rec_yards_yac_pred

efficiency_rush_yards_pred

anytime_td_con_rec_pred

anytime_td_con_car_pred

Each one reuses the same PerPositionModel wrapper, just with a different task_name, feature_fn, and optional cap_fn.

Task F: Keep top-level classifier unchanged for now

Important: Do not change the final anytime-TD classifier yet.

It should still consume the same columns as before:

snaps_pred

usage_target_pred

usage_carries_pred

etc.

All we’ve changed is how those columns are produced (per-position models instead of one global model).

Later, when you’re confident:

You can remove caps or relax them.

You can retrain the final classifier and see if performance improves.

3. Stage toward Mixture-of-Experts

With this structure in place, you’ve already:

Introduced the idea of routing by position_group.

Wrapped the logic in a reusable PerPositionModel abstraction.

Kept the external interface the same.

To move toward a richer Mixture-of-Experts later:

Add more granular labels like role_group (slot, outside, inline, 3rd-down RB).

Either:

Feed role_group as a feature to each per-position model, or

Extend the wrapper to support experts keyed by (position_group, role_group).

But for now, you’re taking a minimal, safe, structural step that:

Removes the need for hacky caps over time.

Improves per-position calibration.

Doesn’t force massive changes in the rest of your pipeline.