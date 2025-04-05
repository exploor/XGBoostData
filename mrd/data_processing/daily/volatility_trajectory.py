"""
Module for calculating the volatility trajectory feature for daily BTC data.

The volatility trajectory is calculated as the percentage change in volatility
from the first to the last 4-hour period of the day.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate(df):
    """
    Calculate the volatility trajectory feature for daily data.
    
    Volatility trajectory measures how volatility evolves throughout the day,
    calculated as: (last_volatility_signature - first_volatility_signature) / first_volatility_signature
    
    Args:
        df (pd.DataFrame): DataFrame with 4-hour data including 'volatility_signature' column
                          and datetime index
        
    Returns:
        pd.Series: Series containing the volatility trajectory values for each day
    """
    try:
        # Validate input
        required_columns = ['volatility_signature']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=pd.DatetimeIndex([]), name='volatility_trajectory')
        
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Group by date (day)
        daily_groups = df.groupby(df.index.date)
        
        # Initialize result Series
        result = pd.Series(index=pd.DatetimeIndex([]), name='volatility_trajectory')
        
        # Process each day
        for day, group in daily_groups:
            # Skip days with insufficient data points
            if len(group) < 2:
                logger.warning(f"Insufficient data points for {day} (minimum 2 required)")
                continue
                
            # Get first and last volatility values of the day
            first_volatility = group['volatility_signature'].iloc[0]
            last_volatility = group['volatility_signature'].iloc[-1]
            
            # Calculate trajectory
            if first_volatility == 0:
                # Handle division by zero
                trajectory = np.nan
                logger.warning(f"First volatility value is zero for {day}, cannot calculate trajectory")
            else:
                trajectory = (last_volatility - first_volatility) / first_volatility
            
            # Add to result Series (use day's date as index)
            day_datetime = pd.Timestamp(day)
            result.loc[day_datetime] = trajectory
        
        logger.info(f"Successfully calculated volatility trajectory for {len(result)} days")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating volatility trajectory: {str(e)}")
        raise