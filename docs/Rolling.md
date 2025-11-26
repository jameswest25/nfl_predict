# Feature Engineering Pipeline

This document specifies the **final implementation contract** for the rolling‑window feature system that powers our MLB prediction pipeline.  All decisions below are settled and should be treated as source‑of‑truth for the first coding pass.

---

## 0  Environment assumptions

* **Python** 3.13.3
* **Polars** ≥ 0.20 (add to `requirements.txt` if not present)
* All Parquet reading/writing will honor `pyarrow` back‑end.

---

## 1  Project layout

*(paths reflect the current repo structure)*

```
utils/feature/
    io.py               # Polars scan & sink helpers
    pitch_level.py      # pitch‑level orchestrator
    pa_level.py         # PA‑level orchestrator
    game_level.py       # game‑level orchestrator
    rolling_window.py   # unified RollingWindow stats module
    feature_registry.md # auto‑generated list of final feature names/descriptions
cache/feature/
    rolling_stats.parquet   # on‑disk cache for windowed aggregates
```

---

## 2  Uniqueness keys / primary IDs

| Grain         | Unique columns (composite key)             |
| ------------- | ------------------------------------------ |
| **Pitch row** | `game_pk`, `at_bat_number`, `pitch_number` |
| **PA row**    | `game_pk`, `at_bat_number`                 |
| **Game row**  | `game_pk`                                  |

`rolling_window.py` **must not** assume any other uniqueness.

---

## 3  RollingWindow module (final spec)

### 3.1 Core responsibilities

* Compute rolling metrics for **exclusive** time windows – rows **before** the current row/day only.
* Maintain an **append‑only cache** at **`cache/feature/rolling_stats.parquet  # append‑only window cache`**.
* Support entity‑agnostic operation: `entity_id` resolves to `batter_id` *(side="batter")* or `pitcher_id` *(side="pitcher")*.

### 3.2 Supported windows

* `3d`, `7d`, `30d`          – previous *N* calendar days, **excluding** current date.
* `season`                    – from **Jan 1** of `game_year` **up to but not including** `utc_ts`.
* `lifetime`                  – all rows **strictly before** current `utc_ts`.

### 3.3 Target keys (numerators)

```
TARGET_KEYS = [
    "is_rbi", "is_hit", "is_2_plus_base_hit", "bases", "is_home_run",
    "is_weak", "is_topped", "is_under", "is_flare_burner",
    "is_solid", "is_barrel"
]
```

*All numerators are treated as simple **counts** (summing 1s or integer `bases`).*

### 3.4 Granularities & denominators

| Granularity            | Denominator         | Evaluated on |
| ---------------------- | ------------------- | ------------ |
| `pitch` / `pitch_type` | count of pitch rows | pitch frame  |
| `pa`                   | count of PA rows    | PA frame     |
| `game`                 | count of game rows  | game frame   |

### 3.5 Context definitions

| Context ID   | Group‑by columns           | Selection rule                                                  |
| ------------ | -------------------------- | --------------------------------------------------------------- |
| `all`        | `entity_id`                | none (versus everyone)                                          |
| `vs_hand`    | `entity_id`                | **filter** rows where `batter_hand == pitcher_hand`             |
| `vs_pitcher` | `entity_id`, `pitcher_id`  | versus specific pitcher                                         |
| `vs_bullpen` | `entity_id`, `reliever_id` | **placeholder** – will activate once reliever ID logic is ready |

### 3.6 Feature naming convention

```
{window}_{key}_per_{granularity}_{context}
```

Examples:

* `30d_is_barrel_per_pitch_vs_hand`
* `season_bases_per_FF_vs_pitcher`
* `7d_is_rbi_per_pa` *(context defaults to `all` when omitted)*

---

## 4  Incremental workflow (updated paths)

1. **Ingest** new pitch rows (3‑day chunk) → append to *long* Parquet store under **`data/cleaned/statcast/YYYY‑MM‑DD.parquet`**.
2. **Pitch‑level features**: run `RollingWindow.compute` on **all** pitch scans (Polars prunes to necessary dates).
3. **Aggregate → PA** and compute PA‑level windows.
4. **Aggregate → Game** and compute game‑level windows.
5. **Upsert** resulting `(batter_id, game_pk)` rows into **`data/processed/statcast/features_master.parquet`** using Polars `unique(keep="last")`.

*(Nulls are **retained**; no drop or imputation in aggregation passes.)*

---

*End of implementation‑ready spec.*
