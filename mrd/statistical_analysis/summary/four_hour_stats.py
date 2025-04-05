import logging
import pandas as pd
from mrd.statistical_analysis.tests import (
    correlation_analysis,
    distribution_analysis,
    stationarity_tests,
    mutual_analysis,
    t_test_analysis
)

# Configure logging globally if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# Mapping of features to their statistical tests
feature_tests = {
    'directional_momentum': [
        {'test': 'correlation', 'target': 'future_price_change_4h'},
        {'test': 'distribution'},
        {'test': 'stationarity'}
    ],
    'volatility_signature': [
        {'test': 'distribution'},
        {'test': 'stationarity'},
        {'test': 'mutual_information', 'target': 'future_range_4h'}
    ],
    'volume_intensity': [
        {'test': 'correlation', 'target': 'future_volume_change_4h'},
        {'test': 'distribution'},
        {'test': 't_test', 'condition': 'high_volatility'}
    ],
    'body_proportion': [
        {'test': 'distribution'},
        {'test': 'stationarity'}
    ],
    'range_position': [
        {'test': 'distribution'},
        {'test': 'stationarity'}
    ]
}

def generate_four_hour_stats(start_date=None, end_date=None):
    """
    Generate statistical analysis for 4-hour data based on predefined tests.

    Args:
        start_date (str, optional): Start date for filtering data (e.g., '2023-01-01')
        end_date (str, optional): End date for filtering data (e.g., '2023-12-31')

    Returns:
        dict: A nested dictionary of results with structure:
              {feature: {test_type: result_dict}}
              or {'error': 'error message'} if a global error occurs.
    """
    try:
        # Load the 4-hour data with date filters
        from mrd.data_processing.data_loader import load_four_hour_features
        df = load_four_hour_features(start_date, end_date)
        
        # Check if DataFrame is valid
        if df.empty:
            logger.warning(f"No data loaded for range {start_date} to {end_date}")
            return {'error': 'No data available for the specified date range'}
        logger.info(f"Processing {len(df)} rows of 4-hour data")

        # Initialize results dictionary
        results = {}

        # Process each feature and its associated tests
        for feature in feature_tests:
            if feature not in df.columns:
                logger.error(f"Feature '{feature}' not found in data")
                results[feature] = {'error': f"Feature '{feature}' not found in data"}
                continue

            results[feature] = {}
            for test in feature_tests[feature]:
                test_type = test['test']
                try:
                    if test_type == 'correlation':
                        target = test['target']
                        if target not in df.columns:
                            raise ValueError(f"Target column '{target}' not found in data")
                        result = correlation_analysis.analyze_correlation(df[feature], df[target], timeframe='4h')
                    elif test_type == 'distribution':
                        result = distribution_analysis.analyze_distribution(df[feature], timeframe='4h')
                    elif test_type == 'stationarity':
                        result = stationarity_tests.test_stationarity(df[feature], timeframe='4h')
                    elif test_type == 'mutual_information':
                        target = test['target']
                        if target not in df.columns:
                            raise ValueError(f"Target column '{target}' not found in data")
                        result = mutual_analysis.compute_mutual_information(df[feature], df[target], timeframe='4h')
                    elif test_type == 't_test':
                        condition = test['condition']  # e.g., 'high_volatility'
                        if condition not in df.columns:
                            raise ValueError(f"Condition column '{condition}' not found in data")
                        result = t_test_analysis.perform_t_test(df[feature], df[condition], timeframe='4h')
                    else:
                        result = {'error': f"Unknown test type: {test_type}"}
                        logger.error(f"Unknown test type '{test_type}' for feature '{feature}'")

                    results[feature][test_type] = result
                    logger.debug(f"Completed {test_type} on {feature}")
                except Exception as e:
                    error_msg = f"Error performing {test_type} on {feature}: {str(e)}"
                    logger.error(error_msg)
                    results[feature][test_type] = {'error': error_msg}

        logger.info("Completed 4-hour statistical analysis")
        return results

    except Exception as e:
        error_msg = f"Error generating 4-hour stats: {str(e)}"
        logger.error(error_msg)
        return {'error': error_msg}

if __name__ == "__main__":
    # Example usage for testing the script standalone
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    results = generate_four_hour_stats(start_date, end_date)
    
    # Print results for verification
    import json
    print(json.dumps(results, indent=2, default=str))  # 'default=str' handles non-serializable objects