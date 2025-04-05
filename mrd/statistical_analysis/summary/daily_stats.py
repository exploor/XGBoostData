"""
Module for generating statistical analyses of daily features.

This module applies statistical tests to daily market features, evaluating their
significance, distributions, and relationships with target variables.
"""
import logging
import pandas as pd
from mrd.data_processing.data_loader import load_daily_features
from mrd.statistical_analysis.tests import (
    correlation_analysis,
    distribution_analysis,
    stationarity_tests,
    mutual_analysis,
    t_test_analysis
)

# Configure logging
logger = logging.getLogger(__name__)

# Mapping of daily features to their statistical tests
feature_tests = {
    'direction_slope': [
        {'test': 'correlation', 'target': 'future_price_change_1d'},
        {'test': 'distribution'},
        {'test': 'stationarity'}
    ],
    'volatility_trajectory': [
        {'test': 'distribution'},
        {'test': 'correlation', 'target': 'future_range_1d'},
        {'test': 'stationarity'}
    ],
    'volume_distribution_skew': [
        {'test': 'distribution'},
        {'test': 'mutual_information', 'target': 'future_volume_1d'}
    ],
    'body_consistency': [
        {'test': 'distribution'},
        {'test': 'correlation', 'target': 'future_price_change_1d'},
        {'test': 't_test', 'condition': 'high_volatility'}
    ],
    'price_level_30d': [
        {'test': 'distribution'},
        {'test': 'stationarity'},
        {'test': 'correlation', 'target': 'future_price_change_1d'}
    ],
    'volume_zscore': [
        {'test': 'distribution'},
        {'test': 'correlation', 'target': 'future_volume_1d'},
        {'test': 'stationarity'}
    ]
}

def generate_daily_stats(start_date, end_date):
    """
    Generate statistical analysis for daily data based on predefined tests.

    Args:
        start_date (str): Start date for data filtering (e.g., '2023-01-01').
        end_date (str): End date for data filtering (e.g., '2023-12-31').

    Returns:
        dict: A nested dictionary of results with structure:
              {feature: {test_type: result_dict}}
              or {'error': 'error message'} if a global error occurs.
    """
    try:
        # Load the daily data with date filters
        df = load_daily_features(start_date, end_date)
        if df.empty:
            logger.warning(f"No data loaded for range {start_date} to {end_date}")
            return {'error': 'No data available for the specified date range'}
        logger.info(f"Loaded {len(df)} rows of daily data from {start_date} to {end_date}")

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
                        result = correlation_analysis.analyze_correlation(df[feature], df[target], 'daily')
                        results[feature][test_type] = result.get('results', {})
                        
                    elif test_type == 'distribution':
                        result = distribution_analysis.analyze_distribution(df[feature], 'daily')
                        results[feature][test_type] = result.get('results', {})
                        
                    elif test_type == 'stationarity':
                        result = stationarity_tests.test_stationarity(df[feature], 'daily')
                        results[feature][test_type] = result.get('results', {})
                        
                    elif test_type == 'mutual_information':
                        target = test['target']
                        if target not in df.columns:
                            raise ValueError(f"Target column '{target}' not found in data")
                        result = mutual_analysis.compute_mutual_information(df[feature], df[target], 'daily')
                        results[feature][test_type] = result.get('results', {})
                        
                    elif test_type == 't_test':
                        condition = test['condition']  # e.g., 'high_volatility'
                        if condition not in df.columns:
                            raise ValueError(f"Condition column '{condition}' not found in data")
                        result = t_test_analysis.perform_t_test(df[feature], df[condition], 'daily')
                        results[feature][test_type] = result.get('results', {})
                        
                    else:
                        error_msg = f"Unknown test type: {test_type}"
                        logger.error(error_msg)
                        results[feature][test_type] = {'error': error_msg}

                    logger.debug(f"Completed {test_type} on {feature}")
                    
                except Exception as e:
                    error_msg = f"Error performing {test_type} on {feature}: {str(e)}"
                    logger.error(error_msg)
                    results[feature][test_type] = {'error': error_msg}

        logger.info("Completed daily statistical analysis")
        return results

    except Exception as e:
        error_msg = f"Error generating daily stats: {str(e)}"
        logger.error(error_msg)
        return {'error': error_msg}

if __name__ == "__main__":
    # Example usage for testing the script standalone
    import json
    from datetime import datetime
    
    # Set fixed start and end dates
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    
    # Generate daily stats for the specified date range
    results = generate_daily_stats(start_date, end_date)
    print(json.dumps(results, indent=2, default=str))