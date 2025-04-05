"""
Module for calculating the direction value feature for weekly BTC data.

The direction value is calculated as the volume-weighted average of daily direction slopes.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate(df):
    """
    Calculate the direction value feature for weekly data.
    
    Direction value is the volume-weighted average of daily direction slopes.
    
    Args:
        df (pd.DataFrame): DataFrame with 'close', 'volume', and a datetime index.
        
    Returns:
        pd.Series: Series containing the direction_value values for each week.
    """
    try:
        # Validate input
        required_columns = ['close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=df.index, name='direction_value')
        
        # Calculate daily direction slope (log return)
        df = df.copy()
        df['slope'] = np.log(df['close'] / df['close'].shift(1))
        
        # Resample to weekly (Monday start) and calculate volume-weighted average
        weekly = df.resample('W-MON').agg({
            'slope': lambda x: np.average(x.dropna(), weights=df.loc[x.dropna().index, 'volume'], axis=0) if x.notna().any() else np.nan,
            'volume': 'sum'
        })
        direction_value = weekly['slope']
        direction_value.name = 'direction_value'
        
        logger.info(f"Successfully calculated direction value for {len(weekly)} weeks")
        return direction_value
    
    except Exception as e:
        logger.error(f"Error calculating direction value: {str(e)}")
        raise