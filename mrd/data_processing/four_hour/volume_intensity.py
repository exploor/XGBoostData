"""
Module for calculating the volume intensity feature for 4-hour BTC data.

The volume intensity is calculated as the z-score of volume over a 20-period window.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_volume_zscore(df, window=20):
    """
    Calculate the z-score of volume over a specified window.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Volume' column
        window (int, optional): The period over which to calculate the z-score. Defaults to 20.
        
    Returns:
        pd.Series: Series containing the volume z-score values
    """
    try:
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Calculate rolling mean and standard deviation
        vol_mean = df_copy['volume'].rolling(window=window, min_periods=1).mean()
        vol_std = df_copy['volume'].rolling(window=window, min_periods=1).std()
        
        # Calculate z-score: (volume - mean) / std
        zscore = (df_copy['volume'] - vol_mean) / vol_std
        
        # Handle division by zero (when std is 0)
        zscore.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return zscore
    
    except Exception as e:
        logger.error(f"Error calculating volume z-score: {str(e)}")
        raise

def calculate(df):
    """
    Calculate the volume intensity feature.
    
    Volume intensity is the z-score of volume over a 20-period window.
    
    Args:
        df (pd.DataFrame): DataFrame with at least 'Volume' column
        
    Returns:
        pd.Series: Series containing the volume intensity values
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
        
        # Calculate volume intensity using helper function
        volume_intensity = calculate_volume_zscore(df, window=20)
        
        logger.info(f"Successfully calculated volume intensity for {len(df)} rows")
        return volume_intensity
    
    except Exception as e:
        logger.error(f"Error calculating volume intensity: {str(e)}")
        raise