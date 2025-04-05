import logging
import pandas as pd
from scipy import stats
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

def get_features_and_condition(timeframe):
    """
    Define features and condition column for the given timeframe.

    Args:
        timeframe (str): The timeframe ('4h', 'daily', 'weekly').

    Returns:
        tuple: (list of feature names, condition column name).

    Raises:
        ValueError: If the timeframe is invalid.
    """
    feature_map = {
        '4h': ['directional_momentum', 'volatility_signature', 'volume_intensity', 'body_proportion', 'range_position'],
        'daily': ['direction_slope', 'volatility_trajectory', 'volume_distribution_skew', 'body_consistency', 'price_level_30d', 'volume_zscore'],
        'weekly': ['direction_value', 'strength_value', 'volatility_value', 'volume_intensity']
    }
    condition_map = {
        '4h': 'is_high_volatility_4h',
        'daily': 'is_high_volatility_daily',
        'weekly': 'is_high_volatility_weekly'
    }

    features = feature_map.get(timeframe)
    condition = condition_map.get(timeframe)
    if not features or not condition:
        logger.error(f"Invalid timeframe: {timeframe}")
        raise ValueError(f"Invalid timeframe: {timeframe}")

    return features, condition

def perform_t_test(feature_series, condition_series, timeframe='4h'):
    """
    Perform a t-test on feature values split by a binary condition, with extensive debugging logs.
    
    Args:
        feature_series (pd.Series): Feature data (e.g., 'volume_intensity').
        condition_series (pd.Series): Condition data with exactly two unique values (e.g., 'high_volatility').
        timeframe (str): Timeframe context for logging (e.g., '4h'). Defaults to '4h'.
    
    Returns:
        dict: Contains 'test_name' and 'results' with t-test metrics, or 'error' if failed.
    """
    try:
        # Log original state of condition_series
        original_unique = condition_series.unique()
        logger.info(f"Original condition_series unique values: {original_unique}")
        
        # Log NaN counts in both series
        feature_nans = feature_series.isna().sum()
        condition_nans = condition_series.isna().sum()
        logger.info(f"Number of NaNs in feature_series: {feature_nans}")
        logger.info(f"Number of NaNs in condition_series: {condition_nans}")
        
        # Log number of rows before dropping NaNs
        original_rows = len(feature_series)
        logger.info(f"Number of rows before dropna: {original_rows}")
        
        # Combine feature and condition series, drop rows with NaNs in either
        combined = pd.concat([feature_series, condition_series], axis=1).dropna()
        
        # Log number of rows after dropping NaNs
        remaining_rows = len(combined)
        logger.info(f"Number of rows after dropna: {remaining_rows}")
        
        # Get unique conditions after dropping NaNs
        unique_conditions = combined.iloc[:, 1].unique()
        logger.info(f"After dropna, condition_series unique values: {unique_conditions}")
        
        # Check if there are exactly two unique conditions
        if len(unique_conditions) != 2:
            raise ValueError(
                f"Condition series must have exactly two unique values, "
                f"but has {len(unique_conditions)}: {unique_conditions}"
            )
        
        # Split into two groups based on the two unique conditions
        group1 = combined[combined.iloc[:, 1] == unique_conditions[0]].iloc[:, 0]
        group2 = combined[combined.iloc[:, 1] == unique_conditions[1]].iloc[:, 0]
        
        # Log the size of each group
        logger.info(f"Group 1 size: {len(group1)}, Group 2 size: {len(group2)}")
        
        # Check if both groups have at least two data points (required for t-test)
        if len(group1) < 2 or len(group2) < 2:
            raise ValueError("Fewer than 2 valid data points in one or both groups")
        
        # Perform Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        
        logger.info(f"Completed t-test for {timeframe}")
        return {
            'test_name': 't_test',
            'results': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05  # Significant if p-value < 0.05
            }
        }
    except Exception as e:
        logger.error(f"Error in {timeframe} t-test analysis: {str(e)}")
        return {'test_name': 't_test', 'error': str(e)}

def run_t_test_analysis(timeframe):
    """
    Run t-test analysis for all features against the condition for the given timeframe.

    Args:
        timeframe (str): The timeframe ('4h', 'daily', 'weekly').

    Returns:
        dict: Feature names mapped to their t-test analysis results.
    """
    logger.info(f"Starting t-test analysis for {timeframe}")
    try:
        df = load_data(timeframe)
        features, condition_column = get_features_and_condition(timeframe)
        if condition_column not in df.columns:
            raise ValueError(f"Condition column '{condition_column}' not found in data")
        condition_series = df[condition_column]
        results = {}
        for feature in features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in {timeframe} data")
                results[feature] = {'test_name': 't_test', 'error': f"Feature {feature} not found"}
                continue
            feature_series = df[feature]
            result = perform_t_test(feature_series, condition_series, timeframe)
            results[feature] = result
        logger.info(f"Completed t-test analysis for {timeframe}")
        return results
    except Exception as e:
        logger.error(f"Error in {timeframe} t-test analysis: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Test the script with 'daily' timeframe
    timeframe = 'daily'
    results = run_t_test_analysis(timeframe)
    print(f"T-test analysis for {timeframe}:")
    for feature, result in results.items():
        print(f"{feature}: {result}")