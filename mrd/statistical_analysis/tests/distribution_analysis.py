import logging
import pandas as pd
from scipy import stats
from mrd.core.database import get_connection

# Configure logging with a basic setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(timeframe):
    """
    Load data from the database based on the specified timeframe.

    Args:
        timeframe (str): The timeframe for which to load data (e.g., '4h', 'daily', 'weekly').

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        ValueError: If the timeframe is invalid.
        Exception: If there's an error loading the data.
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
    Define features and target variable based on the timeframe.

    Args:
        timeframe (str): The timeframe for which to define features (e.g., '4h', 'daily', 'weekly').

    Returns:
        tuple: A tuple containing a list of feature names and the target variable name.

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

def analyze_distribution(feature_series, timeframe='4h'):
    """
    Compute descriptive statistics for the feature series.
    
    Args:
        feature_series (pd.Series): Feature data (e.g., price momentum).
        timeframe (str): Timeframe context for logging (e.g., '4h', 'daily', 'weekly'). Defaults to '4h'.
    
    Returns:
        dict: Contains 'test_name' and 'results' with distribution stats, or 'error' if failed.
    """
    try:
        # Drop NaNs and check for sufficient data
        clean_series = feature_series.dropna()
        if len(clean_series) < 1:
            raise ValueError("No valid data points after removing NaNs")
        
        # Compute statistics
        mean = clean_series.mean()
        std = clean_series.std()
        skew = stats.skew(clean_series)
        kurtosis = stats.kurtosis(clean_series)
        
        logger.info(f"Completed distribution analysis for {timeframe}")
        return {
            'test_name': 'distribution',
            'results': {
                'mean': mean,
                'std': std,
                'skew': skew,
                'kurtosis': kurtosis
            }
        }
    except Exception as e:
        logger.error(f"Error in {timeframe} distribution analysis: {str(e)}")
        return {'test_name': 'distribution', 'error': str(e)}

def run_distribution_analysis(timeframe):
    """
    Run distribution analysis for all features for the given timeframe.

    Args:
        timeframe (str): The timeframe for the analysis (e.g., '4h', 'daily', 'weekly').

    Returns:
        dict: A dictionary with feature names as keys and their distribution analysis results as values.
    """
    logger.info(f"Starting distribution analysis for {timeframe}")
    try:
        df = load_data(timeframe)
        features, _ = get_features_and_target(timeframe)  # Target is ignored as it's not needed here
        results = {}
        for feature in features:
            feature_series = df[feature]
            result = analyze_distribution(feature_series, timeframe)
            results[feature] = result
        logger.info(f"Completed distribution analysis for {timeframe}")
        return results
    except Exception as e:
        logger.error(f"Error in {timeframe} distribution analysis: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Example: Run analysis for 'daily' timeframe
    timeframe = 'daily'
    results = run_distribution_analysis(timeframe)
    print(f"Distribution analysis for {timeframe}:")
    for feature, result in results.items():
        print(f"{feature}: {result}")