"""
Module for calculating the volume intensity feature for weekly BTC data.

The volume intensity is calculated as the weekly volume relative to the 20-week SMA.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate(df):
    """
    Calculate the volume intensity feature for weekly data.
    
    Volume intensity is the weekly volume relative to the 20-week SMA.
    
    Args:
        df (pd.DataFrame): DataFrame with 'volume' and a datetime index.
        
    Returns:
        pd.Series: Series containing the volume_intensity values for each week.
    """
    try:
        # Validate input
        required_columns = ['volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=df.index, name='volume_intensity')
        
        # Resample to weekly volume sum
        df = df.copy()
        weekly_volume = df.resample('W-MON')['volume'].sum()
        
        # Calculate 20-week SMA
        sma_20 = weekly_volume.rolling(window=20, min_periods=1).mean()
        
        # Calculate volume intensity
        volume_intensity = weekly_volume / sma_20
        volume_intensity.name = 'volume_intensity'
        
        logger.info(f"Successfully calculated volume intensity for {len(volume_intensity)} weeks")
        return volume_intensity
    
    except Exception as e:
        logger.error(f"Error calculating volume intensity: {str(e)}")
        raise