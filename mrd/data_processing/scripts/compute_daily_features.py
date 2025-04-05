import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime, timedelta

# Set up logging
from mrd.core.logger import setup_logger
logger = setup_logger(__name__)

# Import data loader
from mrd.data_processing.data_loader import load_raw_4h_data  # Assuming this includes your updates

# Import feature calculation modules
from mrd.data_processing.daily import (
    direction_slope,
    volatility_trajectory,
    volume_distribution_skew,
    body_consistency,
    price_level_30d,
    volume_zscore
)

# Define maximum lookback period (in days)
MAX_LOOKBACK_DAYS = 182  # For price_level_30d calculation

def aggregate_4h_to_daily(df):
    """
    Aggregate 4-hour OHLCV data to daily periods.
    
    Args:
        df (pd.DataFrame): 4-hour data with OHLCV columns
        
    Returns:
        tuple: (daily_df, intraday_data) where daily_df is the aggregated daily data,
               and intraday_data is a dict of 4-hour data per day
    """
    try:
        # Validate input DataFrame
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for aggregation: {missing_columns}")
        
        # Ensure 'timestamp' is the index if not already
        if not isinstance(df.index, pd.DatetimeIndex) and 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        # Create a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Group by date (extracting the date part from the DatetimeIndex)
        daily_grouped = df_copy.groupby(df_copy.index.date)
        
        # Aggregate OHLCV data using agg
        daily_df = daily_grouped.agg({
            'open': 'first',   # First opening price of the day
            'high': 'max',     # Maximum high of the day
            'low': 'min',      # Minimum low of the day
            'close': 'last',   # Last closing price of the day
            'volume': 'sum'    # Sum of volume for the day
        })
        
        # Convert the index (datetime.date objects) to DatetimeIndex
        daily_df.index = pd.to_datetime(daily_df.index)
        
        # Store the 4-hour data for each day (needed for some features)
        intraday_data = {}
        for date, group in daily_grouped:
            intraday_data[pd.Timestamp(date)] = group
            
        logger.info(f"Successfully aggregated 4-hour data to {len(daily_df)} daily periods")
        return daily_df, intraday_data
        
    except Exception as e:
        logger.error(f"Error aggregating 4-hour data to daily: {str(e)}")
        raise

def compute_daily_features(start_date, end_date):
    """
    Compute all daily features and targets for the given date range with sufficient lookback.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: DataFrame with computed daily features and targets
    """
    try:
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Calculate lookback start date (4-hour periods per day = 6)
        load_start_dt = start_dt - timedelta(days=MAX_LOOKBACK_DAYS + 1)  # Add 1 for safety
        load_start_date = load_start_dt.strftime('%Y-%m-%d 00:00:00')
        
        logger.info(f"Loading 4-hour data from {load_start_date} to {end_date} to cover {MAX_LOOKBACK_DAYS} days lookback")
        
        # Load raw 4-hour data using the updated loader
        four_hour_df = load_raw_4h_data(load_start_date, end_date)
        
        # Check if DataFrame is empty
        if four_hour_df.empty:
            logger.warning("No 4-hour data loaded from the database for the specified range")
            return pd.DataFrame()
            
        logger.info(f"Loaded 4-hour DataFrame with {len(four_hour_df)} rows")
        
        # Aggregate 4-hour data to daily
        daily_df, intraday_data = aggregate_4h_to_daily(four_hour_df)
        
        # Create a result DataFrame
        result_df = daily_df.copy()
        
        # Ensure DataFrame has datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            result_df.index = pd.to_datetime(result_df.index)
        
        # Calculate features
        logger.info("Calculating daily features...")
        
        # Direction Slope
        try:
            direction_slope_series = direction_slope.calculate(four_hour_df)
            result_df['direction_slope'] = direction_slope_series.reindex(result_df.index)
            logger.info("Calculated direction_slope feature")
        except Exception as e:
            logger.error(f"Failed to calculate direction_slope: {str(e)}")
            result_df['direction_slope'] = np.nan
        
        # Volatility Trajectory
        try:
            if 'volatility_signature' not in four_hour_df.columns:
                from mrd.data_processing.four_hour import volatility_signature as vs
                four_hour_df['volatility_signature'] = vs.calculate(four_hour_df)
            volatility_trajectory_series = volatility_trajectory.calculate(four_hour_df)
            result_df['volatility_trajectory'] = volatility_trajectory_series.reindex(result_df.index)
            logger.info("Calculated volatility_trajectory feature")
        except Exception as e:
            logger.error(f"Failed to calculate volatility_trajectory: {str(e)}")
            result_df['volatility_trajectory'] = np.nan
        
        # Volume Distribution Skew
        try:
            volume_distribution_skew_series = volume_distribution_skew.calculate(four_hour_df)
            result_df['volume_distribution_skew'] = volume_distribution_skew_series.reindex(result_df.index)
            logger.info("Calculated volume_distribution_skew feature")
        except Exception as e:
            logger.error(f"Failed to calculate volume_distribution_skew: {str(e)}")
            result_df['volume_distribution_skew'] = np.nan
        
        # Body Consistency
        try:
            body_consistency_series = body_consistency.calculate(four_hour_df)
            result_df['body_consistency'] = body_consistency_series.reindex(result_df.index)
            logger.info("Calculated body_consistency feature")
        except Exception as e:
            logger.error(f"Failed to calculate body_consistency: {str(e)}")
            result_df['body_consistency'] = np.nan
        
        # Price Level 30D
        try:
            result_df['price_level_30d'] = price_level_30d.calculate(result_df)
            logger.info("Calculated price_level_30d feature")
        except Exception as e:
            logger.error(f"Failed to calculate price_level_30d: {str(e)}")
            result_df['price_level_30d'] = np.nan
        
        # Volume Z-Score
        try:
            result_df['volume_zscore'] = volume_zscore.calculate(result_df)
            logger.info("Calculated volume_zscore feature")
        except Exception as e:
            logger.error(f"Failed to calculate volume_zscore: {str(e)}")
            result_df['volume_zscore'] = np.nan
        
        # Compute target variables
        logger.info("Computing target variables...")
        
        result_df['future_price_change_1d'] = np.log(result_df['close'].shift(-1) / result_df['close'])
        result_df['future_range_1d'] = (result_df['high'].shift(-1) - result_df['low'].shift(-1)) / result_df['close']
        result_df['future_volume_1d'] = result_df['volume'].shift(-1)
        
        # High volatility condition
        median_volatility = result_df['price_level_30d'].median()
        result_df['high_volatility'] = (result_df['price_level_30d'] > median_volatility).astype(int)
        
        # Filter to requested date range
        result_df = result_df[(result_df.index >= start_dt) & (result_df.index <= end_dt)]
        
        # Drop rows with NaNs in key columns
        key_columns = ['price_level_30d', 'future_price_change_1d', 'future_range_1d', 'future_volume_1d']
        result_df = result_df.dropna(subset=key_columns)
        
        logger.info(f"Final daily DataFrame contains {len(result_df)} days")
        logger.info(f"NaNs in final result_df:\n{result_df.isna().sum()}")
        
        # Reset index to make 'date' a column
        result_df = result_df.reset_index().rename(columns={'index': 'date'})
        
        # Select columns to return
        columns_to_save = [
            'date', 'direction_slope', 'volatility_trajectory',
            'volume_distribution_skew', 'body_consistency',
            'price_level_30d', 'volume_zscore',
            'future_price_change_1d', 'future_range_1d', 'future_volume_1d',
            'high_volatility'
        ]
        return result_df[columns_to_save]
        
    except Exception as e:
        logger.error(f"Error computing daily features: {str(e)}")
        raise

def save_features_to_db(df):
    """
    Save computed daily features and targets to the database.
    
    Args:
        df (pd.DataFrame): DataFrame with daily features and targets
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if df.empty:
            logger.warning("No data to save to database")
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
        
        # Save to daily_vectors table, appending to existing data
        df.to_sql('daily_vectors', engine, if_exists='replace', index=False)
        logger.info(f"Saved {len(df)} rows to daily_vectors table")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
        return False

if __name__ == "__main__":
    # Default date range: last 5 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # Override with command line arguments if provided
    if len(sys.argv) > 2:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    
    logger.info(f"Computing daily features and targets from {start_date} to {end_date}")
    
    try:
        # Compute daily features
        features_df = compute_daily_features(start_date, end_date)
        
        if not features_df.empty:
            logger.info(f"Computed daily features for {len(features_df)} days")
            print(features_df.head())
            
            # Save to database
            save_result = save_features_to_db(features_df)
            if save_result:
                logger.info("Daily features successfully saved to database")
            else:
                logger.warning("Failed to save daily features to database")
        else:
            logger.warning("No daily features computed")
            
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        sys.exit(1)
        
    logger.info("Daily feature computation completed successfully")