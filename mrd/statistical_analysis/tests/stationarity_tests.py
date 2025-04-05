import logging
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from mrd.core.database import get_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(timeframe):
    """
    Load data from the database based on the specified timeframe.

    Args:
        timeframe (str): The timeframe ('4h', 'daily', 'weekly').

    Returns:
        pd.DataFrame: The data loaded from the corresponding table, indexed by 'timestamp'.

    Raises:
        ValueError: If the timeframe is invalid.
        Exception: If data loading fails.
    """
    table_map = {
        '4h': 'four_hour_vectors',
        'daily': 'daily_vectors',
        'weekly': 'weekly_vectors'
    }
    table = table_map.get(timeframe)
    if not table:
        logger.error(f"Invalid timeframe: {timeframe}")
        raise ValueError(f"Invalid timeframe: {timeframe}")

    engine = get_connection()
    query = f"SELECT * FROM {table} ORDER BY timestamp"
    try:
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        df.set_index('timestamp', inplace=True)
        logger.info(f"Loaded {len(df)} rows from {table}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {table}: {e}")
        raise

def get_features_and_target(timeframe):
    """
    Define features and target variable for the given timeframe.

    Args:
        timeframe (str): The timeframe ('4h', 'daily', 'weekly').

    Returns:
        tuple: (list of feature names, target variable name).

    Raises:
        ValueError: If the timeframe is invalid.
    """
    feature_map = {
        '4h': ['directional_momentum', 'volatility_signature', 'volume_intensity', 'body_proportion', 'range_position'],
        'daily': ['direction_slope', 'volatility_trajectory', 'volume_distribution_skew', 'body_consistency', 'price_level_30d', 'volume_zscore'],
        'weekly': ['direction_value', 'strength_value', 'volatility_value', 'volume_intensity']
    }
    target_map = {
        '4h': 'future_price_change_4h',
        'daily': 'future_price_change_1d',
        'weekly': 'future_price_change_1w'
    }

    features = feature_map.get(timeframe)
    target = target_map.get(timeframe)
    if not features or not target:
        logger.error(f"Invalid timeframe: {timeframe}")
        raise ValueError(f"Invalid timeframe: {timeframe}")

    return features, target

def test_stationarity(feature_series, timeframe='4h'):
    """
    Perform the Augmented Dickey-Fuller test to check stationarity of the feature series.
    
    Args:
        feature_series (pd.Series): Feature data (e.g., price series).
        timeframe (str): Timeframe context for logging (e.g., '4h', 'daily', 'weekly'). Defaults to '4h'.
    
    Returns:
        dict: Contains 'test_name' and 'results' with ADF test metrics, or 'error' if failed.
    """
    try:
        # Drop NaNs and check for sufficient data
        clean_series = feature_series.dropna()
        if len(clean_series) < 10:
            raise ValueError("Fewer than 10 valid data points for ADF test")
        
        # Perform ADF test
        result = adfuller(clean_series)
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]  # Dictionary of critical values (1%, 5%, 10%)
        
        logger.info(f"Completed stationarity test for {timeframe}")
        return {
            'test_name': 'stationarity',
            'results': {
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_stationary': p_value < 0.05  # Stationary if p-value < 0.05
            }
        }
    except Exception as e:
        logger.error(f"Error in {timeframe} stationarity test: {str(e)}")
        return {'test_name': 'stationarity', 'error': str(e)}

def run_stationarity_analysis(timeframe):
    """
    Run stationarity analysis for all features for the given timeframe.

    Args:
        timeframe (str): The timeframe ('4h', 'daily', 'weekly').

    Returns:
        dict: Feature names mapped to their stationarity test results.
    """
    logger.info(f"Starting stationarity analysis for {timeframe}")
    try:
        df = load_data(timeframe)
        features, _ = get_features_and_target(timeframe)  # Target is not used
        results = {}
        for feature in features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in {timeframe} data")
                results[feature] = {'test_name': 'stationarity', 'error': f"Feature {feature} not found"}
                continue
            feature_series = df[feature]
            result = test_stationarity(feature_series, timeframe)
            results[feature] = result
        logger.info(f"Completed stationarity analysis for {timeframe}")
        return results
    except Exception as e:
        logger.error(f"Error in {timeframe} stationarity analysis: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Test the script with 'daily' timeframe
    timeframe = 'daily'
    results = run_stationarity_analysis(timeframe)
    print(f"Stationarity analysis for {timeframe}:")
    for feature, result in results.items():
        print(f"{feature}: {result}")