import polars as pl

from utils.feature.weather_features import append_weather_context_flags


def _base_weather_row(**overrides):
    default_row = {
        "roof": overrides.get("roof", "OUTDOORS"),
        "weather_wind_mph": overrides.get("weather_wind_mph", 10.0),
        "weather_wind_gust_mph": overrides.get("weather_wind_gust_mph", 12.0),
        "weather_precip_probability_pct": overrides.get("weather_precip_probability_pct", 10.0),
        "weather_precip_amount_in": overrides.get("weather_precip_amount_in", 0.0),
        "weather_snow_amount_in": overrides.get("weather_snow_amount_in", 0.0),
        "weather_temp_air_f": overrides.get("weather_temp_air_f", 65.0),
        "weather_extreme_wind_flag": overrides.get("weather_extreme_wind_flag", 0),
        "weather_heavy_precip_flag": overrides.get("weather_heavy_precip_flag", 0),
        "weather_freezing_flag": overrides.get("weather_freezing_flag", 0),
    }
    default_row.update(overrides)
    return default_row


def test_append_weather_flags_marks_outdoor_windy_games():
    df = pl.DataFrame(
        [
            _base_weather_row(
                roof="OUTDOORS",
                weather_wind_mph=22.0,
                weather_wind_gust_mph=30.0,
            )
        ]
    )

    result = append_weather_context_flags(df)

    assert result["roof_is_indoor_flag"].to_list() == [0]
    assert result["weather_bad_passing_flag"].to_list() == [1]


def test_append_weather_flags_ignored_for_domes():
    df = pl.DataFrame(
        [
            _base_weather_row(
                roof="DOME",
                weather_wind_mph=22.0,
                weather_wind_gust_mph=30.0,
            )
        ]
    )

    result = append_weather_context_flags(df)

    assert result["roof_is_indoor_flag"].to_list() == [1]
    assert result["weather_bad_passing_flag"].to_list() == [0], "Indoor games should suppress bad-weather flag"


def test_append_weather_flags_handles_missing_roof_column():
    df = pl.DataFrame(
        [
            {
                key: value
                for key, value in _base_weather_row(
                    weather_precip_probability_pct=80.0,
                    weather_precip_amount_in=0.4,
                ).items()
                if key != "roof"
            }
        ]
    )

    result = append_weather_context_flags(df)

    assert "roof" in result.columns, "Helper should add roof column when missing"
    assert result["roof_is_indoor_flag"].to_list() == [0]
    assert result["weather_bad_passing_flag"].to_list() == [1]

