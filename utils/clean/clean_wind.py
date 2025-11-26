import logging
import pandas as pd
import numpy as np
from .constants import ROOF_CLOSED_TEAMS


def add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add robust wind encoding that handles domed stadiums and circular directions.
    
    Creates:
    - is_roof_closed: Boolean flag for domed/closed roof stadiums
    - wind_sin, wind_cos: Circular encoding of wind direction (0 if no wind)
    - Properly handles missing wind data for indoor stadiums
    """
    logging.info("Adding wind features with dome/roof handling...")
    
    # Add roof status
    df = _add_roof_status(df)
    
    # Add circular wind encoding
    df = _add_circular_wind_encoding(df)
    
    # Calculate wind components
    df = _calculate_wind_components(df)
    
    # Calculate wind resistance
    df = _calculate_wind_resistance(df)
    
    return df


def _add_roof_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add roof closed status based on team and conditions."""
    
    if 'home_team_abbr' not in df.columns:
        # Try original column name
        if 'home_team' in df.columns:
            df["is_roof_closed"] = df["home_team"].map(ROOF_CLOSED_TEAMS).fillna(False).infer_objects(copy=False).astype('uint8')
            logging.info(f"Added roof status using home_team: {df['is_roof_closed'].sum():,} games in domed/closed stadiums")
        else:
            logging.warning("home_team_abbr column not found - cannot determine roof status")
            df['is_roof_closed'] = 0
    else:
        df["is_roof_closed"] = df["home_team_abbr"].map(ROOF_CLOSED_TEAMS).fillna(False).infer_objects(copy=False).astype('uint8')
        logging.info(f"Added roof status: {df['is_roof_closed'].sum():,} games in domed/closed stadiums")
    
    # For retractable roof teams, check weather conditions
    retractable_teams = ['HOU', 'MIA', 'MIL', 'SEA', 'TEX', 'TOR']
    retractable_mask = df.get('home_team_abbr', df.get('home_team', pd.Series())).isin(retractable_teams)
    
    if retractable_mask.any() and 'conditions' in df.columns:
        # Close roof for bad weather conditions
        bad_weather_conditions = ['rain', 'snow', 'thunderstorm', 'drizzle', 'showers']
        bad_weather_mask = df['conditions'].str.contains(
            '|'.join(bad_weather_conditions), case=False, na=False
        )
        
        # Close roof for retractable teams in bad weather (fix dtype issue)
        df.loc[retractable_mask & bad_weather_mask, 'is_roof_closed'] = 1
    
    closed_count = df['is_roof_closed'].sum()
    total_games = len(df)
    
    logging.info(f"Roof status: {closed_count}/{total_games} games with roof closed ({closed_count/total_games*100:.1f}%)")
    
    return df


def _add_circular_wind_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add circular encoding for wind direction."""
    
    # Circular encoding for wind direction
    if 'winddir' in df.columns and 'windspeed' in df.columns:
        # Convert wind direction to radians
        rad = np.deg2rad(df["winddir"])
        
        # Only encode direction if there's actual wind (speed > 0)
        # This prevents spurious direction signals when there's no wind
        wind_exists = (df["windspeed"] > 0) & df["windspeed"].notna() & df["winddir"].notna()
        
        df["wind_sin"] = np.where(wind_exists, np.sin(rad), 0.0)
        df["wind_cos"] = np.where(wind_exists, np.cos(rad), 0.0)
        logging.info(f"Added circular wind encoding: {wind_exists.sum():,} rows with valid wind data")
        
    else:
        logging.warning("Wind columns (winddir, windspeed) not found - cannot add wind features")
        df["wind_sin"] = 0.0
        df["wind_cos"] = 0.0
    
    return df


def _calculate_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate wind components relative to field orientation."""
    
    required_cols = ['winddir', 'windspeed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.warning(f"Missing columns for wind components: {missing_cols}")
        return df
    
    # Load stadium azimuths (this would need to be imported from config)
    # For now, using a simplified approach
    df['wind_x'] = df['windspeed'] * np.cos(np.radians(df['winddir']))
    df['wind_y'] = df['windspeed'] * np.sin(np.radians(df['winddir']))
    
    # Calculate headwind/tailwind component (simplified)
    # This assumes home plate to center field is 0 degrees
    df['headwind_component'] = df['windspeed'] * np.cos(np.radians(df['winddir']))
    df['crosswind_component'] = df['windspeed'] * np.sin(np.radians(df['winddir']))
    
    logging.info("Calculated wind components (x, y, headwind, crosswind)")
    
    return df


def _calculate_wind_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate wind resistance effects on ball flight."""
    
    if 'windspeed' not in df.columns:
        logging.warning("windspeed column not found - cannot calculate wind resistance")
        return df
    
    # Simplified wind resistance calculation
    # Higher wind speed = more resistance
    df['wind_resistance'] = df['windspeed'] ** 2 * 0.001  # Simplified coefficient
    
    # Wind assistance (positive = helping, negative = hindering)
    if 'headwind_component' in df.columns:
        df['wind_assistance'] = -df['headwind_component']  # Headwind negative, tailwind positive
    
    logging.info("Calculated wind resistance and assistance factors")
    
    return df 