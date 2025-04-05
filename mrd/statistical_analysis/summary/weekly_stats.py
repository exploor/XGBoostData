"""
Module for generating statistical analyses of weekly features.
"""
import pandas as pd
import logging
from sqlalchemy import create_engine  # Added import for database engine creation
from mrd.data_processing.data_loader import load_daily_features  # We'll adapt this for weekly
from mrd.statistical_analysis.tests import (
    correlation_analysis,
    distribution_analysis,
    stationarity_tests,
    mutual_analysis,
    t_test_analysis
)

logger = logging.getLogger(__name__)

# Define statistical tests for weekly features
feature_tests = {
    'direction_value': [
        {'test': 'correlation', 'target': 'future_price_change_1w'},
        {'test': 'stationarity'},
        {'test': 'mutual_information', 'target': 'future_price_change_1w'}
    ],
    'volatility_value': [
        {'test': 'correlation', 'target': 'future_range_1w'},
        {'test': 'distribution'},
        {'test': 't_test', 'condition': 'high_volatility'}
    ],
    'volume_intensity': [
        {'test': 'correlation', 'target': 'future_volume_1w'},
        {'test': 'stationarity'}
    ],
    # 'strength_value' omitted due to missing data in sample; add if populated later
}

def load_weekly_features(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load weekly market features from the weekly_vectors table for a given date range.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Weekly features and targets.
    """
    try:
        # Manually create the database engine
        user = "postgres"
        password = "postgres"
        host = "localhost"
        port = "5432"
        dbname = "evolabz"
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(conn_str)
        
        query = f"""
            SELECT * FROM weekly_vectors 
            WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'
        """
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        if df.empty:
            logger.warning(f"No weekly data for range {start_date} to {end_date}")
        else:
            logger.info(f"Loaded {len(df)} rows of weekly data")
        return df
    except Exception as e:
        logger.error(f"Error loading weekly data: {str(e)}")
        return pd.DataFrame()

def generate_weekly_stats(start_date: str, end_date: str) -> dict:
    """
    Generate statistical analysis for weekly data based on predefined tests.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        dict: {feature: {test_type: result_dict}} or {'error': message}
    """
    try:
        df = load_weekly_features(start_date, end_date)
        if df.empty:
            return {'error': 'No data available for the specified date range'}
        
        results = {}
        for feature, tests in feature_tests.items():
            if feature not in df.columns:
                logger.error(f"Feature '{feature}' not found")
                results[feature] = {'error': f"Feature '{feature}' not found"}
                continue
            
            results[feature] = {}
            for test in tests:
                test_type = test['test']
                try:
                    if test_type == 'correlation':
                        target = test['target']
                        result = correlation_analysis.analyze_correlation(
                            df[feature], df[target], 'weekly'
                        )
                        results[feature][test_type] = result.get('results', {})
                    elif test_type == 'distribution':
                        result = distribution_analysis.analyze_distribution(
                            df[feature], 'weekly'
                        )
                        results[feature][test_type] = result.get('results', {})
                    elif test_type == 'stationarity':
                        result = stationarity_tests.test_stationarity(
                            df[feature], 'weekly'
                        )
                        results[feature][test_type] = result.get('results', {})
                    elif test_type == 'mutual_information':
                        target = test['target']
                        result = mutual_analysis.compute_mutual_information(
                            df[feature], df[target], 'weekly'
                        )
                        results[feature][test_type] = result.get('results', {})
                    elif test_type == 't_test':
                        condition = test['condition']
                        result = t_test_analysis.perform_t_test(
                            df[feature], df[condition], 'weekly'
                        )
                        results[feature][test_type] = result.get('results', {})
                    else:
                        results[feature][test_type] = {'error': f"Unknown test: {test_type}"}
                except Exception as e:
                    results[feature][test_type] = {'error': str(e)}
        
        logger.info("Completed weekly statistical analysis")
        return results
    except Exception as e:
        logger.error(f"Error generating weekly stats: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    results = generate_weekly_stats('2019-01-01', '2019-03-31')
    import json
    print(json.dumps(results, indent=2, default=str))