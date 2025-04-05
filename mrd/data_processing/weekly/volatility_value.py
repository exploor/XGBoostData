"""
Module for calculating the volatility value feature for weekly BTC data.

The volatility value is calculated as the weekly range normalized by a 12-week EMA, scaled by percentile rank over 26 weeks.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate(df):
    """
    Calculate the volatility value feature for weekly data.
    
    Volatility value is the weekly range normalized by 12-week EMA, scaled by percentile rank over 26 weeks.
    
    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', and a datetime index.
        
    Returns:
        pd.Series: Series containing the volatility_value values for each week.
    """
    try:
        # Validate input
        required_columns = ['high', 'low']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=df.index, name='volatility_value')
        
        # Resample to weekly data
        df = df.copy()
        weekly = df.resample('W-MON').agg({
            'high': 'max',
            'low': 'min'
        })
        weekly['range'] = weekly['high'] - weekly['low']
        
        # Calculate 12-week EMA of range with α=0.15
        alpha = 0.15
        weekly['ema_range'] = weekly['range'].ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate ratio
        weekly['ratio'] = weekly['range'] / weekly['ema_range']
        
        # Scale by percentile rank over 26 weeks
        weekly['volatility_value'] = weekly['ratio'].rolling(window=26, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
        )
        volatility_value = weekly['volatility_value']
        volatility_value.name = 'volatility_value'
        
        logger.info(f"Successfully calculated volatility value for {len(weekly)} weeks")
        return volatility_value
    
    except Exception as e:
        logger.error(f"Error calculating volatility value: {str(e)}")
        raise