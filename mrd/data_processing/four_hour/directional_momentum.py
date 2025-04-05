"""
Module for calculating the directional momentum feature for 4-hour BTC data.

The directional momentum is calculated as the log return of the price change: ln(close / open).
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_log_return(df):
    """
    Calculate the log return based on close and open prices.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Open' and 'Close' columns
        
    Returns:
        pd.Series: Series containing the log return values
    """
    try:
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Calculate log return: ln(close / open)
        log_return = np.log(df_copy['close'] / df_copy['open'])
        
        # Handle potential infinite values from division by zero or negative prices
        log_return.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return log_return
    
    except Exception as e:
        logger.error(f"Error calculating log return: {str(e)}")
        raise

def calculate(df):
    """
    Calculate the directional momentum feature.
    
    Directional momentum is the log return of the price change: ln(close / open).
    
    Args:
        df (pd.DataFrame): DataFrame with at least 'Open' and 'Close' columns
        
    Returns:
        pd.Series: Series containing the directional momentum values
    """
    try:
        # Validate input
        required_columns = ['open', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=df.index, name='directional_momentum')
        
        # Additional validation for zero or negative values
        if (df['open'] <= 0).any() or (df['close'] <= 0).any():
            logger.warning("DataFrame contains zero or negative prices, which may cause issues in log calculation")
            # Proceed, but NaN will be returned for those rows
        
        # Calculate directional momentum using helper function
        directional_momentum = calculate_log_return(df)
        
        logger.info(f"Successfully calculated directional momentum for {len(df)} rows")
        return directional_momentum
    
    except Exception as e:
        logger.error(f"Error calculating directional momentum: {str(e)}")
        raise