"""
Module for calculating the price level 30d feature for daily BTC data.

The price level 30d measures where the current day's close lies within
the high-low range of the previous 30 days.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate(df):
    """
    Calculate the price level 30d feature for daily data.
    
    Price level 30d represents the normalized position of the current close
    within the 30-day high-low range: (close - 30d_low) / (30d_high - 30d_low)
    
    Args:
        df (pd.DataFrame): DataFrame with daily data including 'high', 'low', 'close' columns
                          and datetime index
        
    Returns:
        pd.Series: Series containing the price level 30d values
    """
    try:
        # Validate input
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=df.index, name='price_level_30d')
        
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Calculate rolling 30-day high and low
        rolling_high = df['high'].rolling(window=30, min_periods=1).max()
        rolling_low = df['low'].rolling(window=30, min_periods=1).min()
        
        # Calculate price level 30d
        price_range = rolling_high - rolling_low
        
        # Handle potential division by zero
        mask = price_range > 0
        result = pd.Series(index=df.index, name='price_level_30d')
        
        # Only calculate for non-zero ranges
        result[mask] = (df['close'][mask] - rolling_low[mask]) / price_range[mask]
        
        # Set NaN for zero ranges
        result[~mask] = np.nan
        
        logger.info(f"Successfully calculated price level 30d for {len(result)} days")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating price level 30d: {str(e)}")
        raise