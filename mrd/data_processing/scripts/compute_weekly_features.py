"""
Script to compute weekly features and save them to the database.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Assuming these are part of your project structure
from mrd.core.logger import setup_logger
from mrd.data_processing.data_loader import load_raw_4h_data

# Import the feature modules
from mrd.data_processing.weekly.direction_value import calculate as calculate_direction_value
from mrd.data_processing.weekly.strength_value import calculate as calculate_strength_value
from mrd.data_processing.weekly.volatility_value import calculate as calculate_volatility_value
from mrd.data_processing.weekly.volume_intensity import calculate as calculate_volume_intensity

logger = setup_logger(__name__)

MAX_LOOKBACK_WEEKS = 26  # 1 year lookback for sufficient historical data

def aggregate_to_daily(df_4h):
    """
    Aggregate 4-hour OHLCV data to daily OHLCV data.

    Args:
        df_4h (pd.DataFrame): 4-hour data with datetime index and OHLCV columns.

    Returns:
        pd.DataFrame: Daily OHLCV data with DatetimeIndex.
    """
    # Resample to daily frequency
    daily_open = df_4h['open'].resample('D').first()
    daily_high = df_4h['high'].resample('D').max()
    daily_low = df_4h['low'].resample('D').min()
    daily_close = df_4h['close'].resample('D').last()
    daily_volume = df_4h['volume'].resample('D').sum()

    # Combine into a single DataFrame
    daily_df = pd.DataFrame({
        'open': daily_open,
        'high': daily_high,
        'low': daily_low,
        'close': daily_close,
        'volume': daily_volume
    })

    # Drop rows with NaN values (e.g., if a day has missing data)
    daily_df = daily_df.dropna()

    return daily_df

def compute_weekly_features(start_date, end_date):
    """
    Compute weekly features for the specified date range.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with computed features and targets.
    """
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        load_start_dt = start_dt - timedelta(weeks=MAX_LOOKBACK_WEEKS)
        load_start_date = load_start_dt.strftime('%Y-%m-%d 00:00:00')
        end_date_with_time = f"{end_date} 23:59:59"

        logger.info(f"Loading 4-hour data from {load_start_date} to {end_date_with_time}")
        df_4h = load_raw_4h_data(load_start_date, end_date_with_time)
        if df_4h.empty:
            logger.warning("No 4-hour data loaded")
            return pd.DataFrame()

        # Aggregate 4-hour data to daily data
        df = aggregate_to_daily(df_4h)
        logger.info(f"df index type after aggregation: {type(df.index)}")  # Should be DatetimeIndex

        # Compute features
        weekly_df = pd.DataFrame(index=df.resample('W-MON').last().index)
        weekly_df['direction_value'] = calculate_direction_value(df)
        weekly_df['strength_value'] = calculate_strength_value(df)
        weekly_df['volatility_value'] = calculate_volatility_value(df)
        weekly_df['volume_intensity'] = calculate_volume_intensity(df)

        # Compute targets
        weekly_close = df['close'].resample('W-MON').last()
        weekly_df['future_price_change_1w'] = np.log(weekly_close.shift(-1) / weekly_close)
        weekly_df['future_range_1w'] = (df['high'].resample('W-MON').max() - df['low'].resample('W-MON').min()).shift(-1)
        weekly_df['future_volume_1w'] = df['volume'].resample('W-MON').sum().shift(-1)

        # Compute condition
        median_volatility = weekly_df['volatility_value'].median()
        weekly_df['high_volatility'] = (weekly_df['volatility_value'] > median_volatility).astype(int)

        # Filter to requested range and drop NaNs in key columns
        weekly_df = weekly_df[(weekly_df.index >= start_dt) & (weekly_df.index <= end_dt)]
        weekly_df = weekly_df.dropna(subset=['volatility_value', 'future_price_change_1w', 'future_range_1w', 'future_volume_1w'])

        logger.info(f"Computed {len(weekly_df)} weekly feature rows")
        return weekly_df.reset_index().rename(columns={'index': 'week_start_date'})

    except Exception as e:
        logger.error(f"Error computing weekly features: {str(e)}")
        raise

def save_features_to_db(df):
    """
    Save the computed features to the database.

    Args:
        df (pd.DataFrame): DataFrame with weekly features.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        if df.empty:
            logger.warning("No data to save")
            return False
        from mrd.core.config import Config
        from sqlalchemy import create_engine
        config = Config()
        db_config = {
            'host': config.get('database', 'host', 'localhost'),
            'port': config.get('database', 'port', 5432),
            'username': config.get('database', 'username', 'postgres'),
            'password': config.get('database', 'password', 'postgres'),
            'database_name': config.get('database', 'database_name', 'evolabz')
        }
        db_url = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database_name']}"
        engine = create_engine(db_url)
        df.to_sql('weekly_vectors', engine, if_exists='replace', index=False)
        logger.info(f"Saved {len(df)} rows to weekly_vectors table")
        return True
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
        return False

if __name__ == "__main__":
    start_date = '2019-01-01'
    end_date = '2024-12-31'
    features_df = compute_weekly_features(start_date, end_date)
    if not features_df.empty:
        save_features_to_db(features_df)