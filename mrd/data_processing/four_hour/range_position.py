"""
Module for calculating the range position feature for 4-hour BTC data.

The range position is calculated as (Close - Low)/(High - Low).
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_range_position_value(df):
    """
    Calculate the range position value for each row.
    
    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns
        
    Returns:
        pd.Series: Series containing the range position values
    """
    try:
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Calculate range
        candle_range = df_copy['high'] - df_copy['low']
        
        # Calculate position: (close - low) / range
        position = (df_copy['close'] - df_copy['low']) / candle_range
        
        # Handle division by zero
        position[candle_range == 0] = 0.5  # Arbitrary value, as the range is zero
        
        return position
    
    except Exception as e:
        logger.error(f"Error calculating range position value: {str(e)}")
        raise

def calculate(df):
    """
    Calculate the range position feature.
    
    Range position is (close - Low)/(High - Low).
    
    Args:
        df (pd.DataFrame): DataFrame with at least 'High', 'Low', 'Close' columns
        
    Returns:
        pd.Series: Series containing the range position values
    """
    try:
        # Validate input
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=df.index, name='range_position')
        
        # Calculate range position using helper function
        range_position = calculate_range_position_value(df)
        
        logger.info(f"Successfully calculated range position for {len(df)} rows")
        return range_position
    
    except Exception as e:
        logger.error(f"Error calculating range position: {str(e)}")
        raise