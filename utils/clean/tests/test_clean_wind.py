import pandas as pd
import numpy as np
from utils.clean.clean_wind import add_wind_features

def test_add_wind_features():
    """
    Tests the calculation of sin and cos components for wind direction.
    """
    # Create a sample DataFrame with cardinal and intercardinal directions
    data = {
        'winddir': [0, 45, 90, 180, 270, 360, np.nan],
        'windspeed': [10, 10, 10, 10, 10, 10, 10], # Add windspeed column
        'home_team_abbr': ['NYY'] * 7 # Add home team for roof logic
    }
    df = pd.DataFrame(data)

    # Apply the function
    result_df = add_wind_features(df)

    # Define expected results (winddir is in degrees)
    expected_sin = [
        np.sin(np.deg2rad(0)),
        np.sin(np.deg2rad(45)),
        np.sin(np.deg2rad(90)),
        np.sin(np.deg2rad(180)),
        np.sin(np.deg2rad(270)),
        np.sin(np.deg2rad(360)),
        0.0  # Expect NaN to be filled with 0
    ]
    expected_cos = [
        np.cos(np.deg2rad(0)),
        np.cos(np.deg2rad(45)),
        np.cos(np.deg2rad(90)),
        np.cos(np.deg2rad(180)),
        np.cos(np.deg2rad(270)),
        np.cos(np.deg2rad(360)),
        0.0  # Expect NaN to be filled with 0
    ]

    # Check that the columns were created
    assert 'wind_sin' in result_df.columns
    assert 'wind_cos' in result_df.columns

    # Check that the values are correct within a small tolerance
    assert np.allclose(result_df['wind_sin'], expected_sin)
    assert np.allclose(result_df['wind_cos'], expected_cos)

    # Test with an empty dataframe
    empty_df = pd.DataFrame({'winddir': [], 'windspeed': [], 'home_team_abbr': []})
    result_empty = add_wind_features(empty_df)
    assert result_empty.empty

    print("test_add_wind_features passed.")

if __name__ == "__main__":
    test_add_wind_features() 