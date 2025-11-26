# tests/conftest.py
from __future__ import annotations
import polars as pl
from pathlib import Path
import pytest
from utils.feature import rolling_window as rwmod


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-mlb",
        action="store_true",
        default=False,
        help="Run legacy MLB/statcast tests (disabled by default).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "mlb: marks tests that exercise the legacy MLB/statcast pipeline (skipped unless --run-mlb is provided)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-mlb"):
        return
    skip_mlb = pytest.mark.skip(reason="Legacy MLB/statcast path is quarantined; pass --run-mlb to execute.")
    for item in items:
        if "mlb" in item.keywords:
            item.add_marker(skip_mlb)


def _cast_u32(df: pl.DataFrame, cols: tuple[str, ...] = ("batter", "pitcher", "game_pk")) -> pl.DataFrame:
    """Make sure ID columns are UInt32 – matches RollingWindow’s internal cast."""
    return df.with_columns([pl.col(c).cast(pl.UInt32) for c in cols if c in df.columns])


@pytest.fixture
def rw(tmp_path: Path):
    """
    Provide an isolated RollingWindow instance.
    """
    return rwmod.RollingWindow(side="batter", row_label="pitch")


@pytest.fixture
def cast_u32():
    return _cast_u32
