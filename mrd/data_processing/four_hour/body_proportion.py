"""
Module for calculating the body proportion feature for 4-hour BTC data.

The body proportion is calculated as (Close - Open)/(High - Low), adjusted for direction.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_body_proportion_value(df):
    """
    Calculate the body proportion value for each row.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns
        
    Returns:
        pd.Series: Series containing the body proportion values
    """
    try:
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Calculate body and range
        body = df_copy['close'] - df_copy['open']
        candle_range = df_copy['high'] - df_copy['low']
        
        # Calculate proportion: body / range
        proportion = body / candle_range
        
        # Handle division by zero
        proportion[candle_range == 0] = 0
        
        return proportion
    
    except Exception as e:
        logger.error(f"Error calculating body proportion value: {str(e)}")
        raise

def calculate(df):
    """
    Calculate the body proportion feature.
    
    Body proportion is (Close - Open)/(High - Low), adjusted for direction.
    
    Args:
        df (pd.DataFrame): DataFrame with at least 'Open', 'High', 'Low', 'Close' columns
        
    Returns:
        pd.Series: Series containing the body proportion values
    """
    try:
        # Validate input
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=df.index, name='body_proportion')
        
        # Calculate body proportion using helper function
        body_proportion = calculate_body_proportion_value(df)
        
        logger.info(f"Successfully calculated body proportion for {len(df)} rows")
        return body_proportion
    
    except Exception as e:
        logger.error(f"Error calculating body proportion: {str(e)}")
        raise