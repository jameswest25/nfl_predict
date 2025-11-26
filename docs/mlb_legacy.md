# Legacy MLB / Statcast Pipeline

The original version of this repository powered an MLB Statcast prediction stack.
Those components are still checked in for posterity (and to reference the
rolling-window implementation), but they are no longer maintained or part of the
NFL production flow.

## Current policy

- The MLB code path is **quarantined by default**. None of the nightly jobs,
  feature builders, or training scripts rely on it.
- Integration tests that hit Statcast-specific modules are marked with
  `@pytest.mark.mlb` and are **skipped unless you explicitly opt in**.
- Documentation that references MLB features is for historical context only.

## Running the legacy pipeline (if you really need to)

1. Make sure you have the Statcast parquet inputs populated under
   `data/cleaned/statcast`.
2. Run pytest with `--run-mlb` to enable the skipped tests:

   ```bash
   venv/bin/python -m pytest --run-mlb tests/test_rolling_window_extended.py
   ```

3. Expect to wire up custom paths—most modules (e.g. `utils.feature.pitch_level`)
   still work but are effectively frozen snapshots.

## Rationale

- The NFL pipeline now owns all production resources. Keeping MLB code active was
  surfacing noisy failures and slowing CI.
- Quarantining lets us keep the proven rolling-window reference without forcing
  strict compatibility on every refactor.

If you plan to revive the MLB stack, fork these modules into a separate package
or add new entry points rather than re-entangling them with the NFL codebase.

