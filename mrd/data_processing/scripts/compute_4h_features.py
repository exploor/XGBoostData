"""
Script to compute 4-hour features and targets from raw BTC data.
"""
import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime, timedelta

# Set up logging
from mrd.core.logger import setup_logger
logger = setup_logger(__name__)

# Import data loader
from mrd.data_processing.data_loader import load_raw_4h_data

# Import feature calculation modules
from mrd.data_processing.four_hour import (
    directional_momentum,
    volatility_signature,
    volume_intensity,
    body_proportion,
    range_position
)

# Define maximum lookback period (in 4-hour periods)
MAX_LOOKBACK_PERIODS = 1092  # 26 weeks = 26 * 7 * 6 = 1092 periods

def compute_4h_features(start_date, end_date):
    """
    Compute all 4-hour features and targets for the given date range with sufficient lookback.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD HH:MM:SS' format
        end_date (str): End date in 'YYYY-MM-DD HH:MM:SS' format
        
    Returns:
        pd.DataFrame: DataFrame with computed features and targets
    """
    try:
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Calculate lookback start date
        load_start_dt = start_dt - timedelta(hours=4 * MAX_LOOKBACK_PERIODS)
        load_start_date = load_start_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Loading data from {load_start_date} to {end_date} to cover {MAX_LOOKBACK_PERIODS} periods lookback")
        
        # Load raw data
        df = load_raw_4h_data(load_start_date, end_date)
        
        # Check if DataFrame is empty
        if df.empty:
            logger.warning("No data loaded from the database for the specified range")
            return df

        # Log DataFrame structure for debugging
        logger.info(f"Loaded DataFrame columns: {list(df.columns)}")
        logger.info(f"DataFrame index: {df.index}")

        # Ensure 'timestamp' is a column
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                logger.info("Found 'timestamp' in index; resetting to column")
                df = df.reset_index().rename(columns={'index': 'timestamp'})
            else:
                logger.error("No 'timestamp' column or datetime index found in DataFrame")
                raise KeyError("DataFrame missing 'timestamp' column or datetime index")

        # Verify 'timestamp' is now present
        if 'timestamp' not in df.columns:
            logger.error("Failed to ensure 'timestamp' as a column")
            raise KeyError("'timestamp' column still missing after checks")

        # Filter to user-specified range and compute features
        result_df = df.copy()
        
        # Compute features
        result_df['directional_momentum'] = directional_momentum.calculate(df).values
        result_df['volatility_signature'] = volatility_signature.calculate(df).values
        result_df['volume_intensity'] = volume_intensity.calculate(df).values
        result_df['body_proportion'] = body_proportion.calculate(df).values
        result_df['range_position'] = range_position.calculate(df).values
        
        # Compute targets
        result_df['future_price_change_4h'] = np.log(result_df['close'].shift(-1) / result_df['close'])
        result_df['future_range_4h'] = result_df['high'].shift(-1) - result_df['low'].shift(-1)
        result_df['future_volume_change_4h'] = result_df['volume'].shift(-1)
        
        # Compute condition column
        median_volatility = result_df['volatility_signature'].median()
        result_df['high_volatility'] = (result_df['volatility_signature'] > median_volatility).astype(int)

        # Filter to requested date range
        result_df = result_df[result_df['timestamp'] >= start_dt]
        
        # Drop rows with NaNs in key columns
        key_columns = ['volatility_signature', 'future_price_change_4h', 'future_range_4h', 'future_volume_change_4h']
        result_df = result_df.dropna(subset=key_columns)
        
        # Log the number of rows after dropping NaNs
        logger.info(f"After dropping NaNs in key columns, {len(result_df)} rows remain")
        
        # Log NaNs in the final DataFrame for debugging
        logger.info(f"NaNs in final result_df:\n{result_df.isna().sum()}")
        
        # Select columns to return
        columns_to_save = [
            'timestamp', 'directional_momentum', 'volatility_signature',
            'volume_intensity', 'body_proportion', 'range_position',
            'future_price_change_4h', 'future_range_4h', 'future_volume_change_4h',
            'high_volatility'
        ]
        return result_df[columns_to_save]

    except Exception as e:
        logger.error(f"Error computing 4h features: {str(e)}")
        raise

def save_features_to_db(df):
    """
    Save computed features and targets to the database.
    """
    try:
        if df.empty:
            logger.warning("No data to save to database")
            return False
        
        from mrd.core.config import Config
        from sqlalchemy import create_engine
        
        config = Config()  # Will use default 'config.yaml' in same directory as config.py
        db_config = {
            'host': config.get('database', 'host', 'localhost'),
            'port': config.get('database', 'port', 5432),
            'username': config.get('database', 'username', 'postgres'),
            'password': config.get('database', 'password', 'postgres'),
            'database_name': config.get('database', 'database_name', 'evolabz')
        }
        
        db_url = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database_name']}"
        engine = create_engine(db_url)
        
        df.to_sql('four_hour_vectors', engine, if_exists='replace', index=False)
        logger.info(f"Saved {len(df)} rows to database")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
        return False

if __name__ == "__main__":
    start_date = '2019-01-01 00:00:00'
    end_date = '2024-12-31 23:59:59'
    
    if len(sys.argv) > 2:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    
    logger.info(f"Computing 4-hour features and targets from {start_date} to {end_date}")
    
    try:
        features_df = compute_4h_features(start_date, end_date)
        if not features_df.empty:
            logger.info(f"Computed features for {len(features_df)} rows")
            print(features_df.head())
            save_features_to_db(features_df)
        else:
            logger.warning("No features computed")
            
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        sys.exit(1)