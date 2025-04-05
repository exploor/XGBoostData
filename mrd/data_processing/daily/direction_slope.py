"""
Module for calculating the direction slope feature for daily BTC data.

The direction slope is calculated as the linear regression slope of 4-hour closes within a single day.
"""
import pandas as pd
import numpy as np
import logging
from scipy import stats

logger = logging.getLogger(__name__)

def calculate_regression_slope(x_values, y_values):
    """
    Calculate the slope of a linear regression line for x and y values.
    
    Args:
        x_values (array-like): Independent variable values (typically time indices)
        y_values (array-like): Dependent variable values (typically prices)
        
    Returns:
        float: Slope coefficient of the regression line
    """
    try:
        if len(x_values) < 2 or len(y_values) < 2:
            logger.warning("Insufficient data points for regression (minimum 2 required)")
            return np.nan
            
        # Perform linear regression (y = mx + b)
        slope, _, _, _, _ = stats.linregress(x_values, y_values)
        return slope
        
    except Exception as e:
        logger.error(f"Error calculating regression slope: {str(e)}")
        raise

def calculate(df):
    """
    Calculate the direction slope feature for daily data.
    
    Direction slope is the linear regression slope of 4-hour closes within a day,
    indicating the strength and direction of the intraday trend.
    
    Args:
        df (pd.DataFrame): DataFrame with 4-hour data including 'close' column and datetime index
        
    Returns:
        pd.Series: Series containing the direction slope values for each day
    """
    try:
        # Validate input
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=pd.DatetimeIndex([]), name='direction_slope')
        
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Group by date (day)
        daily_groups = df.groupby(df.index.date)
        
        # Initialize result Series
        result = pd.Series(index=pd.DatetimeIndex([]), name='direction_slope')
        
        # Process each day
        for day, group in daily_groups:
            # Skip days with insufficient data points
            if len(group) < 2:
                logger.warning(f"Insufficient data points for {day} (minimum 2 required)")
                continue
                
            # Create sequence indices for x values (0, 1, 2, ...)
            x_values = np.arange(len(group))
            
            # Calculate slope of close prices
            slope = calculate_regression_slope(x_values, group['close'].values)
            
            # Add to result Series (use day's date as index)
            day_datetime = pd.Timestamp(day)
            result.loc[day_datetime] = slope
        
        logger.info(f"Successfully calculated direction slope for {len(result)} days")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating direction slope: {str(e)}")
        raise