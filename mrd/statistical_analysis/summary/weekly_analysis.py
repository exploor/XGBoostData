"""
Module for analyzing and visualizing weekly statistical results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any
from .weekly_stats import generate_weekly_stats, load_weekly_features  # Corrected import
from .daily_analysis import (generate_distribution_analysis, generate_stationarity_analysis,
                             generate_mutual_information_analysis, generate_t_test_analysis)  # Added import

logger = logging.getLogger(__name__)

def analyze_weekly_data(stats_results: Dict[str, Dict[str, Any]], 
                        df: pd.DataFrame,
                        start_date: str,
                        end_date: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Process statistical results for weekly market data.
    
    Args:
        stats_results: Dict from generate_weekly_stats
        df: DataFrame with weekly market data
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
    
    Returns:
        Dict: {feature: {test_type: {visualization, explanation, ...}}}
    """
    analysis_results = {}
    feature_descriptions = {
        "direction_value": "Volume-weighted average of daily slopes, indicating weekly trend direction.",
        "volatility_value": "Weekly range relative to 12-week EMA, scaled by 26-week percentile rank.",
        "volume_intensity": "Weekly volume relative to 20-week SMA, highlighting participation."
        # Add "strength_value" if data becomes available
    }

    for feature, tests in stats_results.items():
        analysis_results[feature] = {}
        summary_text = feature_descriptions.get(feature, "No description available.")
        analysis_results[feature]["summary"] = {"explanation": summary_text}

        for test_type, test_results in tests.items():
            try:
                if test_type == "correlation":
                    analysis_results[feature][test_type] = generate_correlation_analysis(
                        feature, test_results, df
                    )
                elif test_type == "distribution":
                    analysis_results[feature][test_type] = generate_distribution_analysis(
                        feature, test_results, df
                    )
                elif test_type == "stationarity":
                    analysis_results[feature][test_type] = generate_stationarity_analysis(
                        feature, test_results, df
                    )
                elif test_type == "mutual_information":
                    analysis_results[feature][test_type] = generate_mutual_information_analysis(
                        feature, test_results, df
                    )
                elif test_type == "t_test":
                    analysis_results[feature][test_type] = generate_t_test_analysis(
                        feature, test_results, df
                    )
            except Exception as e:
                analysis_results[feature][test_type] = {
                    "visualization": None,
                    "explanation": f"Error: {str(e)}",
                    "hypothesis": "N/A",
                    "significance": "N/A",
                    "model_implications": "N/A"
                }
    
    return analysis_results

def generate_correlation_analysis(feature_name: str, test_results: Dict[str, Any], 
                                  df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate correlation analysis for a feature against a target.
    
    Args:
        feature_name: Name of the feature being analyzed
        test_results: Statistical results from correlation test
        df: DataFrame containing the feature and target data
    
    Returns:
        Dict with visualization, explanation, and analysis details
    """
    target_name = 'future_price_change_1w' if 'price_change' in test_results.get('target', '') else test_results.get('target', 'future_range_1w')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df[feature_name], y=df[target_name], ax=ax)
    ax.set_title(f"{feature_name} vs {target_name}")
    pearson_r = test_results.get("pearson_r", 0)
    explanation = f"Pearson correlation: {pearson_r:.2f} (p={test_results.get('pearson_p', 1):.4f})"
    return {
        "visualization": fig,
        "explanation": explanation,
        "hypothesis": f"H0: No correlation between {feature_name} and {target_name}",
        "significance": "Significant" if test_results.get('pearson_p', 1) < 0.05 else "Not significant",
        "model_implications": "Influences Direction dimension" if abs(pearson_r) > 0.3 else "Weak predictor"
    }

if __name__ == "__main__":
    start_date, end_date = '2019-01-01', '2019-03-31'
    df = load_weekly_features(start_date, end_date)
    stats = generate_weekly_stats(start_date, end_date)
    results = analyze_weekly_data(stats, df, start_date, end_date)
    for feature, tests in results.items():
        print(f"\n{feature}:")
        for test, details in tests.items():
            print(f"  {test}: {details['explanation']}")