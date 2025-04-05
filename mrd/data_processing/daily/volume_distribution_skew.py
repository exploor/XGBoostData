"""
Module for calculating the volume distribution skew feature for daily BTC data.

The volume distribution skew measures the asymmetry of 4-hour volume distribution within a day.
"""
import pandas as pd
import numpy as np
import logging
from scipy import stats

logger = logging.getLogger(__name__)

def calculate(df):
    """
    Calculate the volume distribution skew feature for daily data.
    
    Volume distribution skew measures the asymmetry of intraday volume distribution,
    calculated using scipy.stats.skew on the 4-hour volumes within each day.
    
    Args:
        df (pd.DataFrame): DataFrame with 4-hour data including 'volume' column and datetime index
        
    Returns:
        pd.Series: Series containing the volume distribution skew values for each day
    """
    try:
        # Validate input
        required_columns = ['volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=pd.DatetimeIndex([]), name='volume_distribution_skew')
        
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Group by date (day)
        daily_groups = df.groupby(df.index.date)
        
        # Initialize result Series
        result = pd.Series(index=pd.DatetimeIndex([]), name='volume_distribution_skew')
        
        # Process each day
        for day, group in daily_groups:
            # Skip days with insufficient data points (skewness requires at least 3 points)
            if len(group) < 3:
                logger.warning(f"Insufficient data points for {day} (minimum 3 required for skewness)")
                continue
                
            # Calculate skewness of volume within the day
            volume_skew = stats.skew(group['volume'].values)
            
            # Add to result Series (use day's date as index)
            day_datetime = pd.Timestamp(day)
            result.loc[day_datetime] = volume_skew
        
        logger.info(f"Successfully calculated volume distribution skew for {len(result)} days")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating volume distribution skew: {str(e)}")
        raise