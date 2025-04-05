"""
Module for calculating the volume z-score feature for daily BTC data.

The volume z-score measures how the current day's volume deviates from
the 20-day average in terms of standard deviations.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate(df):
    """
    Calculate the volume z-score feature for daily data.
    
    Volume z-score is calculated as (volume - 20d_avg) / 20d_std, representing
    how many standard deviations the current volume is from the 20-day mean.
    
    Args:
        df (pd.DataFrame): DataFrame with daily data including 'volume' column
                          and datetime index
        
    Returns:
        pd.Series: Series containing the volume z-score values for each day
    """
    try:
        # Validate input
        required_columns = ['volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=df.index, name='volume_zscore')
        
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Calculate rolling mean and standard deviation (20-day window)
        rolling_mean = df['volume'].rolling(window=20, min_periods=1).mean()
        rolling_std = df['volume'].rolling(window=20, min_periods=1).std()
        
        # Calculate z-score
        volume_zscore = (df['volume'] - rolling_mean) / rolling_std
        
        # Replace infinities with NaN
        volume_zscore.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        logger.info(f"Successfully calculated volume z-score for {len(volume_zscore)} days")
        return volume_zscore
        
    except Exception as e:
        logger.error(f"Error calculating volume z-score: {str(e)}")
        raise