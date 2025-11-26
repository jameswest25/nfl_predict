import logging
import pandas as pd
from .constants import ROOF_CLOSED_TEAMS


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean weather data with enhanced cloudcover imputation."""
    
    # Ensure is_roof_closed exists (should be created by wind cleaning)
    if 'is_roof_closed' not in df.columns:
        # This warning is helpful for debugging but noisy in production if order is guaranteed.
        # logger.warning("is_roof_closed column not found - creating basic version")
        df['is_roof_closed'] = 0 # Assume open roof if not specified
    
    # Clean precipitation
    df = _clean_precipitation(df)
    
    # Clean wind direction
    df = _clean_wind_direction(df)
    
    # Clean cloudcover with enhanced logic
    df = _clean_cloudcover(df)
    
    return df


def _clean_precipitation(df: pd.DataFrame) -> pd.DataFrame:
    """Clean precipitation data."""
    # Fill missing precip with 0
    precip_missing = df['precip'].isna().sum()
    if precip_missing > 0:
        df['precip'] = df['precip'].fillna(0)
        logging.info(f"Filled {precip_missing} missing values for precip with 0")
    
    # Fill missing precipprob with 0
    precipprob_missing = df['precipprob'].isna().sum()
    if precipprob_missing > 0:
        df['precipprob'] = df['precipprob'].fillna(0)
        logging.info(f"Filled {precipprob_missing} missing values for precipprob with 0")
    
    return df


def _clean_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Clean wind direction data."""
    if 'winddir' not in df.columns:
        return df
    
    original_missing = df['winddir'].isna().sum()
    
    # Strategy 1: Use wind bearing as fallback
    if 'wind_bearing' in df.columns:
        fallback_mask = df['winddir'].isna() & df['wind_bearing'].notna()
        fallback_count = fallback_mask.sum()
        if fallback_count > 0:
            df.loc[fallback_mask, 'winddir'] = df.loc[fallback_mask, 'wind_bearing']
            logging.info(f"Used wind_bearing fallback for {fallback_count} winddir values")
    
    # Strategy 2: Set no-wind conditions to 0°
    if 'windspeed' in df.columns:
        no_wind_mask = df['winddir'].isna() & (df['windspeed'] == 0)
        no_wind_count = no_wind_mask.sum()
        if no_wind_count > 0:
            df.loc[no_wind_mask, 'winddir'] = 0
            logging.info(f"Set {no_wind_count} no-wind conditions to 0° winddir")
    
    # Strategy 3: Both winddir and windspeed missing → treat as calm (0°/0 mph)
    if 'windspeed' in df.columns:
        both_missing_mask = df['winddir'].isna() & df['windspeed'].isna()
        both_missing_count = both_missing_mask.sum()
        if both_missing_count > 0:
            df.loc[both_missing_mask, ['winddir', 'windspeed']] = 0
            logging.info(f"Set {both_missing_count} rows with both winddir & windspeed missing to calm conditions (0°/0 mph)")
    
    if original_missing > 0:
        remaining_missing = df['winddir'].isna().sum()
        filled = original_missing - remaining_missing
        logging.info(f"winddir: {original_missing} originally missing, {filled} filled, {remaining_missing} remaining")
    
    return df


def _clean_cloudcover(df: pd.DataFrame) -> pd.DataFrame:
    """Clean cloudcover with enhanced three-tier imputation strategy."""
    if 'cloudcover' not in df.columns:
        return df
    
    original_missing = df['cloudcover'].isna().sum()
    if original_missing == 0:
        return df
    
    # Add missing flag before imputation – removed in refactor
    # df['cloudcover_missing'] = df['cloudcover'].isna().astype(int)
    
    # Strategy 1: Indoor/no-wind conditions = 0% cloudcover
    indoor_mask = df['cloudcover'].isna() & (df['is_roof_closed'] == 1)
    no_wind_mask = df['cloudcover'].isna() & (df.get('windspeed', 0) == 0)
    strategy1_mask = indoor_mask | no_wind_mask
    strategy1_count = strategy1_mask.sum()
    
    if strategy1_count > 0:
        df.loc[strategy1_mask, 'cloudcover'] = 0.0
        logging.info(f"Set {strategy1_count} indoor/no-wind conditions to 0% cloudcover")
    
    # Strategy 2: Estimate from conditions for outdoor stadiums
    strategy2_count = 0
    if 'conditions' in df.columns:
        outdoor_missing_mask = df['cloudcover'].isna() & (df['is_roof_closed'] == 0)
        
        # Cloudcover mapping based on common weather conditions
        condition_cloudcover_map = {
            'clear': 0, 'sunny': 0, 'fair': 10,
            'partly cloudy': 25, 'mostly cloudy': 75,
            'overcast': 100, 'cloudy': 50,
            'rain': 80, 'drizzle': 60, 'showers': 70,
            'thunderstorm': 90, 'snow': 85, 'fog': 95
        }
        
        for condition, cloudcover_val in condition_cloudcover_map.items():
            condition_mask = (outdoor_missing_mask & 
                            df['conditions'].str.contains(condition, case=False, na=False))
            condition_count = condition_mask.sum()
            if condition_count > 0:
                df.loc[condition_mask, 'cloudcover'] = cloudcover_val
                strategy2_count += condition_count
        
        if strategy2_count > 0:
            logging.info(f"Estimated cloudcover from conditions for {strategy2_count} outdoor games")
    
    # Strategy 3: Default value for any remaining missing
    remaining_missing_mask = df['cloudcover'].isna()
    strategy3_count = remaining_missing_mask.sum()
    
    if strategy3_count > 0:
        df.loc[remaining_missing_mask, 'cloudcover'] = 50.0  # Default moderate cloudcover
        logging.info(f"Applied default 50% cloudcover to {strategy3_count} remaining missing values")
    
    # Final statistics
    final_missing = df['cloudcover'].isna().sum()
    total_imputed = original_missing - final_missing
    
    logging.info(f"cloudcover: {original_missing} originally missing, {total_imputed} imputed, {final_missing} remaining")
    
    if final_missing == 0:
        logging.info("✓ All cloudcover values successfully imputed")
    
    return df 