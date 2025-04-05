"""
Module for calculating the body consistency feature for daily BTC data.

The body consistency measures the proportion of 4-hour candles that have
the same direction as the daily candle.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate(df):
    """
    Calculate the body consistency feature for daily data.
    
    Body consistency is the proportion of 4-hour candles that have the same
    direction (bullish/bearish) as the overall daily candle.
    
    Args:
        df (pd.DataFrame): DataFrame with 4-hour data including 'open' and 'close' columns
                          and datetime index
        
    Returns:
        pd.Series: Series containing the body consistency values for each day
    """
    try:
        # Validate input
        required_columns = ['open', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=pd.DatetimeIndex([]), name='body_consistency')
        
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Group by date (day)
        daily_groups = df.groupby(df.index.date)
        
        # Initialize result Series
        result = pd.Series(index=pd.DatetimeIndex([]), name='body_consistency')
        
        # Process each day
        for day, group in daily_groups:
            # Skip days with insufficient data points
            if len(group) < 2:
                logger.warning(f"Insufficient data points for {day} (minimum 2 required)")
                continue
            
            # Determine daily direction (1 for bullish, -1 for bearish, 0 for neutral)
            daily_open = group['open'].iloc[0]
            daily_close = group['close'].iloc[-1]
            
            if daily_close > daily_open:
                daily_direction = 1  # Bullish day
            elif daily_close < daily_open:
                daily_direction = -1  # Bearish day
            else:
                daily_direction = 0  # Neutral day
            
            # If the day is neutral, skip
            if daily_direction == 0:
                logger.warning(f"Neutral day detected for {day}, skipping body consistency calculation")
                continue
            
            # Calculate 4-hour candle directions
            candle_directions = np.sign(group['close'] - group['open'])
            
            # Count candles with the same direction as the daily candle
            if daily_direction == 1:  # Bullish day
                consistent_candles = (candle_directions > 0).sum()
            else:  # Bearish day
                consistent_candles = (candle_directions < 0).sum()
                
            # Calculate consistency ratio
            consistency = consistent_candles / len(group)
            
            # Add to result Series (use day's date as index)
            day_datetime = pd.Timestamp(day)
            result.loc[day_datetime] = consistency
        
        logger.info(f"Successfully calculated body consistency for {len(result)} days")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating body consistency: {str(e)}")
        raise