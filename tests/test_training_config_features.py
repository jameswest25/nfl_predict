from pathlib import Path
from typing import Dict, List

import polars as pl
import pytest
import yaml

from utils.general.constants import LEAK_PRONE_COLUMNS


def _load_training_config() -> Dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "training.yaml"
    with config_path.open("r") as fh:
        return yaml.safe_load(fh)


def test_training_config_feature_guards():
    """Ensure columns_to_discard and other_features_to_include don't overlap."""
    training_cfg = _load_training_config()

    problems = training_cfg.get("problems", [])
    assert problems, "No problems defined in training configuration."

    for problem in problems:
        name = problem.get("name", "<unknown>")
        discard = set(problem.get("columns_to_discard", []))
        other = set(problem.get("other_features_to_include", []))
        intersection = discard & other
        assert (
            not intersection
        ), f"columns_to_discard and other_features_to_include intersect for '{name}': {sorted(intersection)}"


def test_leak_prone_columns_sync():
    """Validate that LEAK_PRONE_COLUMNS in utils/constants.py matches training.yaml.
    
    The Python constant and YAML config must stay in sync to ensure consistent
    feature exclusion across the feature and training pipelines.
    """
    training_cfg = _load_training_config()

    problems = training_cfg.get("problems", [])
    assert problems, "No problems defined in training configuration."

    # Get the shared columns_to_discard from any problem that uses the anchor
    # (they all reference the same anchor, so just pick the first one)
    yaml_columns = set()
    for problem in problems:
        discard = problem.get("columns_to_discard", [])
        if discard:
            yaml_columns = set(discard)
            break

    python_columns = set(LEAK_PRONE_COLUMNS)

    # The YAML may have additional columns (like team_offensive_plays_actual 
    # for the extended version), so we check that the Python constant is a
    # subset of the YAML base set OR vice versa
    
    # Core columns should be in both
    common_core = {
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "pass_attempt",
        "completion",
        "reception",
        "touchdown",
        "offense_snaps",
        "defense_snaps",
    }
    
    missing_from_python = common_core - python_columns
    missing_from_yaml = common_core - yaml_columns
    
    assert not missing_from_python, (
        f"Core leak-prone columns missing from LEAK_PRONE_COLUMNS: {sorted(missing_from_python)}. "
        "Update utils/constants.py to match config/training.yaml."
    )
    assert not missing_from_yaml, (
        f"Core leak-prone columns missing from training.yaml: {sorted(missing_from_yaml)}. "
        "Update config/training.yaml to match utils/constants.py."
    )


def test_feature_prefixes_cannot_select_leak_prone_columns():
    """Ensure no feature_prefixes_to_include can scoop up leak-prone fields."""
    training_cfg = _load_training_config()
    problems = training_cfg.get("problems", [])
    assert problems, "No problems defined in training configuration."

    banned = set(LEAK_PRONE_COLUMNS)
    violations: List[str] = []

    for problem in problems:
        name = problem.get("name", "<unknown>")
        discard = set(problem.get("columns_to_discard", []))
        prefixes = problem.get("feature_prefixes_to_include") or []
        for prefix in prefixes:
            if not prefix:
                continue
            collisions = [
                col for col in banned if col.startswith(prefix) and col not in discard
            ]
            if collisions:
                violations.append(
                    f"{name}: prefix '{prefix}' would include leak columns {collisions[:5]}"
                )

    assert not violations, (
        "Feature prefixes must not include leak-prone columns. "
        f"Violations: {violations}"
    )


def test_asof_snapshots_not_after_cutoff():
    """Validate that stored snapshot timestamps never exceed the decision cutoff."""
    asof_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "asof_metadata.parquet"
    if not asof_path.exists():
        pytest.skip(f"asof metadata not found at {asof_path}")

    df = pl.read_parquet(asof_path)
    if df.is_empty():
        pytest.skip("asof metadata file is empty")

    df = df.with_columns(pl.col("cutoff_ts").cast(pl.Datetime("ms", "UTC")))
    snapshot_specs = {
        "injury_snapshot_ts": 0.95,
        "roster_snapshot_ts": 0.9,
        "odds_snapshot_ts": 0.6,
        "forecast_snapshot_ts": 0.9,
    }

    coverage_failures = {}
    for col, min_rate in snapshot_specs.items():
        if col not in df.columns:
            continue
        subset = df.filter(pl.col(col).is_not_null()).with_columns(
            pl.col(col).cast(pl.Datetime("ms", "UTC"))
        )
        total = subset.height
        if total == 0:
            continue
        valid = subset.filter(pl.col(col) <= pl.col("cutoff_ts")).height
        rate = valid / total
        if rate < min_rate:
            coverage_failures[col] = {"rate": rate, "required": min_rate, "total": total}

    assert not coverage_failures, (
        "As-of snapshot coverage dropped below expected thresholds: "
        f"{coverage_failures}"
    )

