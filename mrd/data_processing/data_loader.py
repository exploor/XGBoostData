"""
Data loader module for Market Regime Dashboard.
"""
import pandas as pd
from mrd.core.config import Config
from mrd.core.database import Database
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)

def load_raw_4h_data(start_date, end_date):
    """
    Load raw 4-hour Bitcoin data from the database for the given date range.
    
    Parameters:
    - start_date (str): Start date in 'YYYY-MM-DD HH:MM:SS' format.
    - end_date (str): End date in 'YYYY-MM-DD HH:MM:SS' format.
    
    Returns:
    - pd.DataFrame: DataFrame with raw OHLCV data and timestamp as index.
    """
    try:
        # Initialize database connection
        config = Config()  # Will use default 'config.yaml' in same directory as config.py
        database = Database(config)
        
        # Use lowercase column names in the query without quotes
        sql = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE timestamp BETWEEN :start_date AND :end_date
        ORDER BY timestamp
        """
        
        params = {'start_date': start_date, 'end_date': end_date}
        
        # execute_query returns a DataFrame directly
        df = database.execute_query(sql, params)
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded {len(df)} rows of 4-hour data from {start_date} to {end_date}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading 4-hour data: {str(e)}")
        return pd.DataFrame()
    
def load_four_hour_features(start_date, end_date):
    """
    Load precomputed features and targets for the 4-hour timeframe.
    
    Parameters:
    - start_date (str): Start date in 'YYYY-MM-DD HH:MM:SS' format.
    - end_date (str): End date in 'YYYY-MM-DD HH:MM:SS' format.
    
    Returns:
    - pd.DataFrame: DataFrame with available features, timestamp as index.
    """
    try:
        # Initialize database connection
        config = Config()  # Will use default 'config.yaml' in same directory as config.py
        database = Database(config)
        
        # SQL query with :name placeholders
        sql = """
        SELECT timestamp,
               directional_momentum, volatility_signature, volume_intensity,
               body_proportion, range_position,
               future_price_change_4h, future_range_4h, future_volume_change_4h,
               high_volatility
        FROM four_hour_vectors
        WHERE timestamp BETWEEN :start_date AND :end_date
        ORDER BY timestamp
        """
        
        # Define parameters as a dictionary
        params = {'start_date': start_date, 'end_date': end_date}
        
        # Wrap the query in text for proper parameter binding
        sql_text = text(sql)
        
        # Execute the query directly with the engine connection
        with database.engine.connect() as connection:
            df = pd.read_sql(sql_text, connection, params=params)
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded {len(df)} rows of 4-hour features from {start_date} to {end_date}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading 4-hour features: {str(e)}")
        return pd.DataFrame()
    
def load_daily_features(start_date, end_date):
    """
    Load precomputed daily features and targets from the database.
    
    Parameters:
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
    - pd.DataFrame: DataFrame with available features, timestamp as index.
    """
    try:
        # Initialize database connection
        config = Config()  # Will use default 'config.yaml' in same directory as config.py
        database = Database(config)
        
        # SQL query with :name placeholders
        sql = """
        SELECT date,
               direction_slope, volatility_trajectory, volume_distribution_skew,
               body_consistency, price_level_30d, volume_zscore,
               future_price_change_1d, future_range_1d, future_volume_1d,
               high_volatility
        FROM daily_vectors
        WHERE date BETWEEN :start_date AND :end_date
        ORDER BY date
        """
        
        # Define parameters as a dictionary
        params = {'start_date': start_date, 'end_date': end_date}
        
        # Wrap the query in text for proper parameter binding
        sql_text = text(sql)
        
        # Execute the query directly with the engine connection
        with database.engine.connect() as connection:
            df = pd.read_sql(sql_text, connection, params=params)
        
        # Convert date to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        logger.info(f"Loaded {len(df)} rows of daily features from {start_date} to {end_date}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading daily features: {str(e)}")
        return pd.DataFrame()

def load_weekly_features(start_date, end_date):
    """
    Load precomputed weekly features and targets from the database.
    
    Parameters:
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
    - pd.DataFrame: DataFrame with available features, week_start_date as index.
    """
    try:
        config = Config()
        database = Database(config)
        sql = """
        SELECT week_start_date,
               direction_value, strength_value, volatility_value, volume_intensity,
               future_price_change_1w, future_range_1w, future_volume_1w,
               high_volatility
        FROM weekly_vectors
        WHERE week_start_date BETWEEN :start_date AND :end_date
        ORDER BY week_start_date
        """
        params = {'start_date': start_date, 'end_date': end_date}
        sql_text = text(sql)
        with database.engine.connect() as connection:
            df = pd.read_sql(sql_text, connection, params=params)
        df['week_start_date'] = pd.to_datetime(df['week_start_date'])
        df.set_index('week_start_date', inplace=True)
        logger.info(f"Loaded {len(df)} rows of weekly features from {start_date} to {end_date}")
        return df
    except Exception as e:
        logger.error(f"Error loading weekly features: {str(e)}")
        return pd.DataFrame()