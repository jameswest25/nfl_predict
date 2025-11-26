from __future__ import annotations

import logging
from typing import Iterable, Sequence

import polars as pl

from utils.general.paths import WEATHER_FORECAST_DIR

logger = logging.getLogger(__name__)

# Heuristic uncertainty (°F) by forecast lead – used when provider lacks spread
FORECAST_TEMP_UNCERTAINTY = {
    96: 7.5,
    72: 6.0,
    48: 4.5,
    36: 4.0,
    24: 3.0,
    12: 2.2,
    6: 1.5,
    3: 1.2,
    2: 1.0,
    1: 0.8,
}

# Heuristic uncertainty (mph) on sustained wind
FORECAST_WIND_UNCERTAINTY = {
    96: 9.0,
    72: 8.0,
    48: 6.0,
    24: 4.5,
    12: 3.0,
    6: 2.0,
    3: 1.5,
    2: 1.2,
    1: 1.0,
}


WEATHER_FEATURE_COLUMNS: Sequence[tuple[str, pl.datatypes.PolarsDataType]] = (
    ("weather_forecast_lead_hours", pl.Float32),
    ("weather_temp_air_f", pl.Float32),
    ("weather_feels_like_f", pl.Float32),
    ("weather_dew_point_f", pl.Float32),
    ("weather_humidity_pct", pl.Float32),
    ("weather_pressure_mb", pl.Float32),
    ("weather_wind_mph", pl.Float32),
    ("weather_wind_gust_mph", pl.Float32),
    ("weather_precip_probability_pct", pl.Float32),
    ("weather_precip_amount_in", pl.Float32),
    ("weather_snow_amount_in", pl.Float32),
    ("weather_cloud_cover_pct", pl.Float32),
    ("weather_visibility_miles", pl.Float32),
    ("weather_conditions", pl.Utf8),
    ("weather_precip_type", pl.Utf8),
    ("weather_forecast_is_backfill", pl.Int8),
    ("weather_forecast_is_historical", pl.Int8),
    ("weather_forecast_source_detail", pl.Utf8),
    ("weather_forecast_generated_ts", pl.Datetime("ms", "UTC")),
    ("weather_forecast_valid_ts", pl.Datetime("ms", "UTC")),
    ("weather_temp_trend", pl.Float32),
    ("weather_wind_trend", pl.Float32),
    ("weather_precip_prob_trend", pl.Float32),
    ("weather_forecast_uncertainty_temp", pl.Float32),
    ("weather_forecast_uncertainty_wind", pl.Float32),
    ("weather_extreme_wind_flag", pl.Int8),
    ("weather_freezing_flag", pl.Int8),
    ("weather_heavy_precip_flag", pl.Int8),
)

WEATHER_FLAG_DEPENDENCIES: Sequence[tuple[str, pl.datatypes.PolarsDataType]] = (
    ("weather_wind_mph", pl.Float32),
    ("weather_wind_gust_mph", pl.Float32),
    ("weather_precip_probability_pct", pl.Float32),
    ("weather_precip_amount_in", pl.Float32),
    ("weather_snow_amount_in", pl.Float32),
    ("weather_temp_air_f", pl.Float32),
    ("weather_extreme_wind_flag", pl.Int8),
    ("weather_heavy_precip_flag", pl.Int8),
    ("weather_freezing_flag", pl.Int8),
)


def _ensure_columns(df: pl.DataFrame, columns: Sequence[tuple[str, pl.datatypes.PolarsDataType]]) -> pl.DataFrame:
    missing_exprs = []
    for name, dtype in columns:
        if name not in df.columns:
            missing_exprs.append(pl.lit(None, dtype=dtype).alias(name))
    if missing_exprs:
        df = df.with_columns(missing_exprs)
    return df


def _load_forecast_snapshots(
    seasons: Sequence[int] | None,
    game_ids: Iterable[str] | None,
) -> pl.DataFrame:
    if not WEATHER_FORECAST_DIR.exists():
        return pl.DataFrame()

    pattern = WEATHER_FORECAST_DIR / "season=*/week=*/part.parquet"

    try:
        scan = pl.scan_parquet(
            str(pattern),
            glob=True,
            hive_partitioning=True,
        )
    except FileNotFoundError:
        return pl.DataFrame()
    except Exception as exc:
        logger.warning("Failed to scan weather snapshots: %s", exc)
        return pl.DataFrame()

    if seasons:
        scan = scan.filter(pl.col("season").is_in(list(seasons)))
    if game_ids:
        scan = scan.filter(pl.col("game_id").is_in(list(game_ids)))

    try:
        forecast = scan.collect()
    except Exception as exc:
        logger.warning("Failed to collect weather snapshots: %s", exc)
        return pl.DataFrame()

    if forecast.is_empty():
        return forecast

    time_cols = [
        "forecast_generated_ts",
        "forecast_valid_ts",
        "forecast_available_ts",
        "ingested_at",
    ]
    for col in time_cols:
        if col in forecast.columns:
            forecast = forecast.with_columns(pl.col(col).cast(pl.Datetime("ms", "UTC")))

    numeric_casts = [
        "forecast_lead_hours",
        "temp_f",
        "feels_like_f",
        "dew_point_f",
        "humidity_pct",
        "pressure_mb",
        "wind_speed_mph",
        "wind_gust_mph",
        "cloud_cover_pct",
        "visibility_miles",
        "precip_amount_in",
        "precip_probability_pct",
        "snow_amount_in",
    ]
    for col in numeric_casts:
        if col in forecast.columns:
            forecast = forecast.with_columns(pl.col(col).cast(pl.Float32))

    bool_cols = ["forecast_is_backfill", "forecast_is_historical"]
    for col in bool_cols:
        if col in forecast.columns:
            forecast = forecast.with_columns(pl.col(col).cast(pl.Boolean))

    forecast = forecast.sort(["game_id", "forecast_generated_ts"])
    shift_cols = [
        ("temp_f", "prev_temp_f"),
        ("wind_speed_mph", "prev_wind_speed_mph"),
        ("precip_probability_pct", "prev_precip_probability_pct"),
        ("forecast_generated_ts", "prev_forecast_generated_ts"),
    ]
    for src, alias in shift_cols:
        if src in forecast.columns:
            forecast = forecast.with_columns(
                pl.col(src).shift(1).over("game_id").alias(alias)
            )

    desired_columns = [
        "game_id",
        "forecast_generated_ts",
        "forecast_valid_ts",
        "forecast_provider",
        "forecast_is_backfill",
        "forecast_is_historical",
        "forecast_source_detail",
        "forecast_lead_hours",
        "temp_f",
        "feels_like_f",
        "dew_point_f",
        "humidity_pct",
        "pressure_mb",
        "wind_speed_mph",
        "wind_gust_mph",
        "precip_amount_in",
        "precip_probability_pct",
        "snow_amount_in",
        "cloud_cover_pct",
        "visibility_miles",
        "conditions",
        "precip_type",
        "prev_temp_f",
        "prev_wind_speed_mph",
        "prev_precip_probability_pct",
        "prev_forecast_generated_ts",
    ]
    existing_cols = [col for col in desired_columns if col in forecast.columns]
    forecast = forecast.select(existing_cols)

    return forecast


def _compute_uncertainty_expr(column: str, mapping: dict[int, float], default_value: float, alias: str) -> pl.Expr:
    return (
        pl.col(column)
        .round(0)
        .cast(pl.Int32)
        .map_elements(
            lambda v: mapping.get(int(v), default_value) if v is not None else None,
            return_dtype=pl.Float32,
        )
        .alias(alias)
    )


def _derive_weather_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("forecast_lead_hours").alias("weather_forecast_lead_hours"),
            pl.col("temp_f").alias("weather_temp_air_f"),
            pl.col("feels_like_f").alias("weather_feels_like_f"),
            pl.col("dew_point_f").alias("weather_dew_point_f"),
            pl.col("humidity_pct").alias("weather_humidity_pct"),
            pl.col("pressure_mb").alias("weather_pressure_mb"),
            pl.col("wind_speed_mph").alias("weather_wind_mph"),
            pl.col("wind_gust_mph").alias("weather_wind_gust_mph"),
            pl.col("precip_probability_pct").alias("weather_precip_probability_pct"),
            pl.col("precip_amount_in").alias("weather_precip_amount_in"),
            pl.col("snow_amount_in").alias("weather_snow_amount_in"),
            pl.col("cloud_cover_pct").alias("weather_cloud_cover_pct"),
            pl.col("visibility_miles").alias("weather_visibility_miles"),
            pl.col("conditions").alias("weather_conditions"),
            pl.col("precip_type").alias("weather_precip_type"),
            pl.col("forecast_generated_ts").alias("weather_forecast_generated_ts"),
            pl.col("forecast_valid_ts").alias("weather_forecast_valid_ts"),
            pl.col("forecast_is_backfill")
            .fill_null(False)
            .cast(pl.Int8)
            .alias("weather_forecast_is_backfill"),
            pl.col("forecast_is_historical")
            .fill_null(False)
            .cast(pl.Int8)
            .alias("weather_forecast_is_historical"),
            pl.col("forecast_source_detail").alias("weather_forecast_source_detail"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("weather_temp_air_f") - pl.col("prev_temp_f"))
            .alias("weather_temp_trend")
            .cast(pl.Float32),
            (pl.col("weather_wind_mph") - pl.col("prev_wind_speed_mph"))
            .alias("weather_wind_trend")
            .cast(pl.Float32),
            (pl.col("weather_precip_probability_pct") - pl.col("prev_precip_probability_pct"))
            .alias("weather_precip_prob_trend")
            .cast(pl.Float32),
        ]
    )

    df = df.with_columns(
        [
            _compute_uncertainty_expr(
                "weather_forecast_lead_hours",
                FORECAST_TEMP_UNCERTAINTY,
                default_value=4.5,
                alias="weather_forecast_uncertainty_temp",
            ),
            _compute_uncertainty_expr(
                "weather_forecast_lead_hours",
                FORECAST_WIND_UNCERTAINTY,
                default_value=3.5,
                alias="weather_forecast_uncertainty_wind",
            ),
        ]
    )

    df = df.with_columns(
        [
            pl.when(
                pl.max_horizontal(
                    [
                        pl.col("weather_wind_mph").fill_null(0),
                        pl.col("weather_wind_gust_mph").fill_null(0),
                    ]
                )
                >= 20.0
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("weather_extreme_wind_flag"),
            pl.when(pl.col("weather_temp_air_f").fill_null(60) <= 32.0)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("weather_freezing_flag"),
            pl.when(
                (pl.col("weather_precip_probability_pct").fill_null(0) >= 70.0)
                | (pl.col("weather_precip_amount_in").fill_null(0) >= 0.25)
                | (pl.col("weather_snow_amount_in").fill_null(0) >= 0.5)
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("weather_heavy_precip_flag"),
        ]
    )

    drop_cols = [
        "forecast_provider",
        "forecast_lead_hours",
        "forecast_generated_ts",
        "forecast_valid_ts",
        "forecast_available_ts",
        "forecast_is_backfill",
        "forecast_is_historical",
        "forecast_source_detail",
        "ingested_at",
        "temp_f",
        "feels_like_f",
        "dew_point_f",
        "humidity_pct",
        "pressure_mb",
        "wind_speed_mph",
        "wind_gust_mph",
        "precip_amount_in",
        "precip_probability_pct",
        "snow_amount_in",
        "cloud_cover_pct",
        "visibility_miles",
        "conditions",
        "precip_type",
        "prev_temp_f",
        "prev_wind_speed_mph",
        "prev_precip_probability_pct",
        "prev_forecast_generated_ts",
    ]

    existing_drop = [col for col in drop_cols if col in df.columns]
    if existing_drop:
        df = df.drop(existing_drop)

    return df


def append_weather_context_flags(
    df: pl.DataFrame,
    *,
    roof_col: str = "roof",
) -> pl.DataFrame:
    """
    Add leak-safe weather context flags that depend on both forecast values
    and venue metadata (e.g., roof type).
    """
    df = _ensure_columns(df, WEATHER_FLAG_DEPENDENCIES)

    if roof_col not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(roof_col))

    roof_is_indoor_expr = (
        pl.col(roof_col)
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.contains("DOME|INDOOR|RETRACTABLE|CLOSED", literal=False)
        .fill_null(False)
    )

    wind_expr = pl.max_horizontal(
        [
            pl.col("weather_wind_mph").fill_null(0.0),
            pl.col("weather_wind_gust_mph").fill_null(0.0) * 0.8,
        ]
    ) >= 17.0
    precip_expr = (
        (pl.col("weather_precip_probability_pct").fill_null(0.0) >= 60.0)
        | (pl.col("weather_precip_amount_in").fill_null(0.0) >= 0.25)
        | (pl.col("weather_snow_amount_in").fill_null(0.0) >= 0.25)
    )
    severe_weather_expr = (
        wind_expr
        | precip_expr
        | (pl.col("weather_extreme_wind_flag").fill_null(0) == 1)
        | (pl.col("weather_heavy_precip_flag").fill_null(0) == 1)
        | (pl.col("weather_freezing_flag").fill_null(0) == 1)
    )

    df = df.with_columns(
        [
            roof_is_indoor_expr.cast(pl.Int8).alias("roof_is_indoor_flag"),
            (
                severe_weather_expr
                & (~roof_is_indoor_expr)
            )
            .cast(pl.Int8)
            .alias("weather_bad_passing_flag"),
        ]
    )

    return df


def add_weather_forecast_features(
    df: pl.DataFrame,
    *,
    cutoff_column: str = "decision_cutoff_ts",
) -> pl.DataFrame:
    """
    Attach weather forecast-based features to the provided frame.
    """
    if df.is_empty():
        return _ensure_columns(df, WEATHER_FEATURE_COLUMNS)

    if "game_id" not in df.columns:
        logger.warning("Dataframe missing game_id column; weather features skipped.")
        return _ensure_columns(df, WEATHER_FEATURE_COLUMNS)

    if cutoff_column not in df.columns:
        logger.warning(
            "Dataframe missing %s column; weather features skipped.", cutoff_column
        )
        return _ensure_columns(df, WEATHER_FEATURE_COLUMNS)

    df = df.with_columns(pl.col(cutoff_column).cast(pl.Datetime("ms", "UTC")))
    seasons = df.get_column("season").unique().to_list() if "season" in df.columns else None
    game_ids = df.get_column("game_id").unique().to_list()

    forecasts = _load_forecast_snapshots(seasons=seasons, game_ids=game_ids)
    if forecasts.is_empty():
        logger.info("No weather forecast snapshots available; populating default columns.")
        return _ensure_columns(df, WEATHER_FEATURE_COLUMNS)

    forecasts = forecasts.sort(["game_id", "forecast_generated_ts"])

    # Ensure join keys exist and types align
    forecasts = forecasts.with_columns(pl.col("game_id").cast(pl.Utf8))
    df = df.with_columns(pl.col("game_id").cast(pl.Utf8))

    with_idx = df.with_row_count("_weather_row_idx")
    sorted_left = with_idx.sort(["game_id", cutoff_column])

    joined = sorted_left.join_asof(
        forecasts,
        left_on=cutoff_column,
        right_on="forecast_generated_ts",
        by="game_id",
        strategy="backward",
        allow_parallel=True,
    )

    joined = _ensure_columns(joined, WEATHER_FEATURE_COLUMNS)
    joined = _derive_weather_features(joined)
    joined = joined.sort("_weather_row_idx").drop("_weather_row_idx")

    joined = _ensure_columns(joined, WEATHER_FEATURE_COLUMNS)
    return joined


def add_weather_forecast_features_training(
    df: pl.DataFrame,
    *,
    cutoff_column: str = "decision_cutoff_ts",
) -> pl.DataFrame:
    return add_weather_forecast_features(df, cutoff_column=cutoff_column)


def add_weather_forecast_features_inference(
    df: pl.DataFrame,
    *,
    cutoff_column: str = "decision_cutoff_ts",
) -> pl.DataFrame:
    return add_weather_forecast_features(df, cutoff_column=cutoff_column)


__all__ = [
    "add_weather_forecast_features",
    "add_weather_forecast_features_training",
    "add_weather_forecast_features_inference",
    "append_weather_context_flags",
]

