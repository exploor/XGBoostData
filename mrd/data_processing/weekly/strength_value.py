import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate(df):
    try:
        # Validate input
        required_columns = ['open', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.Series(index=df.index, name='strength_value')
        
        # Ensure the DataFrame index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Ensure DataFrame is sorted by index (resampling assumes chronological order)
        df = df.sort_index()
        
        # Compute weekly data
        weekly = df.resample('W-MON').agg({
            'open': 'first',
            'close': 'last'
        })
        weekly['direction'] = np.where(weekly['close'] > weekly['open'], 1, -1)
        # Convert weekly index to period for consistent mapping
        weekly['week'] = weekly.index.to_period('W-MON')
        
        # Create a copy to avoid modifying the input DataFrame
        df = df.copy()
        
        # Assign week period to each day
        df['week'] = df.index.to_period('W-MON')
        
        # Map weekly direction to each day using period index
        df['weekly_direction'] = df['week'].map(weekly.set_index('week')['direction'])
        
        # Compute daily direction
        df['daily_direction'] = np.where(df['close'] > df['open'], 1, -1)
        
        # Compute whether daily direction matches weekly direction
        df['match'] = df['daily_direction'] == df['weekly_direction']
        
        # Calculate strength value as the mean of matches per week
        strength_value = df.groupby('week')['match'].mean()
        strength_value.name = 'strength_value'
        
        logger.info(f"Successfully calculated strength value for {len(strength_value)} weeks")
        return strength_value
    
    except Exception as e:
        logger.error(f"Error calculating strength value: {str(e)}")
        raise