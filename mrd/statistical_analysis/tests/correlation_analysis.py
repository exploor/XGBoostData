import pandas as pd
from scipy import stats
import logging
from mrd.core.database import get_connection

# Configure logging with a basic setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(timeframe):
    """Load data from the database based on the specified timeframe."""
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
    """Define features and target variable based on the timeframe."""
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

def analyze_correlation(feature_series, target_series, timeframe):
    """Compute Pearson and Spearman correlations between feature and target series for the given timeframe."""
    try:
        combined = pd.concat([feature_series, target_series], axis=1).dropna()
        if len(combined) < 2:
            logger.warning(f"Insufficient data for correlation analysis in {timeframe}")
            return {'error': 'Insufficient data'}
        
        pearson_r, pearson_p = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
        spearman_r, spearman_p = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
        
        result = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }
        logger.info(
            f"Computed correlations for {timeframe}: Pearson r={pearson_r:.4f}, p={pearson_p:.4f}; "
            f"Spearman r={spearman_r:.4f}, p={spearman_p:.4f}"
        )
        return result
    except Exception as e:
        logger.error(f"Error in correlation analysis for {timeframe}: {e}")
        return {'error': str(e)}

def run_correlation_analysis(timeframe):
    """Run correlation analysis for all features against the target for the given timeframe."""
    logger.info(f"Starting correlation analysis for {timeframe}")
    try:
        df = load_data(timeframe)
        features, target = get_features_and_target(timeframe)
        
        results = {}
        for feature in features:
            feature_series = df[feature]
            target_series = df[target]
            result = analyze_correlation(feature_series, target_series, timeframe)
            results[feature] = result
        
        logger.info(f"Completed correlation analysis for {timeframe}")
        return results
    except Exception as e:
        logger.error(f"Error in {timeframe} correlation analysis: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Example: Run analysis for '4h' timeframe
    timeframe = '4h'
    results = run_correlation_analysis(timeframe)
    print(f"Correlation analysis for {timeframe}:")
    for feature, result in results.items():
        print(f"{feature}: {result}")