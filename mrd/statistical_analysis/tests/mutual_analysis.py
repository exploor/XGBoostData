import logging
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
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
        pd.DataFrame: The data loaded from the corresponding table.

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
    query = f"SELECT * FROM {table}"
    try:
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
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

def compute_mutual_information(feature_series, target_series, timeframe='4h'):
    """
    Compute mutual information between feature and target series.
    
    Args:
        feature_series (pd.Series): Feature data (e.g., volume).
        target_series (pd.Series): Target data (e.g., price direction).
        timeframe (str): Timeframe context for logging (e.g., '4h', 'daily', 'weekly'). Defaults to '4h'.
    
    Returns:
        dict: Contains 'test_name' and 'results' with mutual information, or 'error' if failed.
    """
    try:
        # Combine feature and target, drop NaNs
        combined = pd.concat([feature_series, target_series], axis=1).dropna()
        if len(combined) < 2:
            raise ValueError("Fewer than 2 valid data points after removing NaNs")
        
        # Compute mutual information
        mi = mutual_info_regression(
            combined.iloc[:, 0].values.reshape(-1, 1),
            combined.iloc[:, 1]
        )
        
        logger.info(f"Completed mutual information analysis for {timeframe}")
        return {
            'test_name': 'mutual_information',
            'results': {
                'mutual_information': mi[0]
            }
        }
    except Exception as e:
        logger.error(f"Error in {timeframe} mutual information analysis: {str(e)}")
        return {'test_name': 'mutual_information', 'error': str(e)}

def run_mutual_information_analysis(timeframe):
    """
    Run mutual information analysis for all features against the target for the given timeframe.

    Args:
        timeframe (str): The timeframe ('4h', 'daily', 'weekly').

    Returns:
        dict: Feature names mapped to their mutual information analysis results.
    """
    logger.info(f"Starting mutual information analysis for {timeframe}")
    try:
        df = load_data(timeframe)
        features, target = get_features_and_target(timeframe)
        target_series = df[target]
        results = {}
        for feature in features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in {timeframe} data")
                results[feature] = {'test_name': 'mutual_information', 'error': f"Feature {feature} not found"}
                continue
            feature_series = df[feature]
            result = compute_mutual_information(feature_series, target_series, timeframe)
            results[feature] = result
        logger.info(f"Completed mutual information analysis for {timeframe}")
        return results
    except Exception as e:
        logger.error(f"Error in {timeframe} mutual information analysis: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Test the script with 'daily' timeframe
    timeframe = 'daily'
    results = run_mutual_information_analysis(timeframe)
    print(f"Mutual information analysis for {timeframe}:")
    for feature, result in results.items():
        print(f"{feature}: {result}")