"""
Module for analyzing and visualizing daily statistical results.

This module processes statistical results for daily market features,
generating visualizations and explanations for each feature and test.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import logging
from typing import Dict, Any, Optional, Tuple, List

from mrd.statistical_analysis.summary.daily_stats import generate_daily_stats
from mrd.data_processing.data_loader import load_daily_features

# Configure logger for this module
logger = logging.getLogger(__name__)

def analyze_daily_data(stats_results: Dict[str, Dict[str, Any]], 
                       df: pd.DataFrame,
                       start_date: str,
                       end_date: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Process statistical results for daily market data, generating visualizations
    and explanations for each feature and test.

    Parameters:
    - stats_results: Dict containing statistical test results from daily_stats
    - df: DataFrame with daily market data
    - start_date: Start date string in format 'YYYY-MM-DD'
    - end_date: End date string in format 'YYYY-MM-DD'

    Returns:
    - Dict with structure {feature: {test_type: {visualization, explanation, hypothesis, significance, model_implications}}}
    """
    analysis_results = {}
    feature_descriptions = {
        "direction_slope": "Measures the linear regression slope of 4-hour closes within a day, indicating the strength and direction of the intraday trend.",
        "volatility_trajectory": "Captures the evolution of volatility throughout the day, showing whether volatility increased or decreased intraday.",
        "volume_distribution_skew": "Quantifies the asymmetry of intraday volume distribution, indicating front or back-loaded trading activity.",
        "body_consistency": "Represents the proportion of 4-hour candles that match the direction of the daily candle, indicating trend coherence.",
        "price_level_30d": "Measures where the closing price sits within the 30-day high-low range, indicating potential overbought or oversold conditions.",
        "volume_zscore": "Normalizes daily volume against a 20-day window, highlighting unusual trading activity."
    }

    for feature, tests in stats_results.items():
        analysis_results[feature] = {}
        # Generate feature summary
        summary_text = feature_descriptions.get(feature, "No description available.")
        analysis_results[feature]["summary"] = {"explanation": summary_text}

        # Analyze each test for the feature
        for test_type, test_results in tests.items():
            try:
                if test_type == "correlation":
                    analysis_results[feature][test_type] = generate_correlation_analysis(feature, test_results, df)
                elif test_type == "distribution":
                    analysis_results[feature][test_type] = generate_distribution_analysis(feature, test_results, df)
                elif test_type == "stationarity":
                    analysis_results[feature][test_type] = generate_stationarity_analysis(feature, test_results, df)
                elif test_type == "mutual_information":
                    analysis_results[feature][test_type] = generate_mutual_information_analysis(feature, test_results, df)
                elif test_type == "t_test":
                    analysis_results[feature][test_type] = generate_t_test_analysis(feature, test_results, df)
            except Exception as e:
                analysis_results[feature][test_type] = handle_analysis_error(feature, test_type, e)

    return analysis_results

def generate_correlation_analysis(feature_name: str,
                                  test_results: Dict[str, Any],
                                  df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate visualization and explanation for correlation test results.
    """
    target_name = next((col for col in df.columns if col.startswith('future_') and feature_name in test_results), 'future_price_change_1d')
    
    fig = create_scatter_plot(df[feature_name], df[target_name],
                              x_label=feature_name, y_label=target_name)
    
    pearson_r = test_results.get("pearson_r", 0)
    pearson_p = test_results.get("pearson_p", 1)
    spearman_r = test_results.get("spearman_r", 0)
    spearman_p = test_results.get("spearman_p", 1)
    
    explanation = (f"The Pearson correlation between {feature_name} and {target_name} is {pearson_r:.2f} "
                   f"(p-value: {format_p_value(pearson_p)}). This indicates a {format_correlation_strength(pearson_r)} "
                   f"linear relationship. The Spearman correlation is {spearman_r:.2f} (p-value: {format_p_value(spearman_p)}), "
                   f"which measures monotonic relationship strength.")
    
    hypothesis = f"H0: No linear correlation exists between {feature_name} and {target_name}."
    
    significance = "Statistically significant" if pearson_p < 0.05 else "Not statistically significant"
    
    model_implications = generate_model_implications(feature_name, "correlation", test_results)
    
    return {
        "visualization": fig,
        "explanation": explanation,
        "hypothesis": hypothesis,
        "significance": significance,
        "model_implications": model_implications
    }

def generate_distribution_analysis(feature_name: str,
                                   test_results: Dict[str, Any],
                                   df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate visualization and explanation for distribution test results.
    """
    fig = create_histogram_plot(df[feature_name], feature_name)
    
    mean = test_results.get("mean", 0)
    std = test_results.get("std", 0)
    skew = test_results.get("skew", 0)
    kurtosis = test_results.get("kurtosis", 0)
    
    explanation = (f"The distribution of {feature_name} has a mean of {mean:.4f} and standard deviation of {std:.4f}. "
                   f"The distribution has a skewness of {skew:.4f} ({format_skewness(skew)}) and "
                   f"kurtosis of {kurtosis:.4f} ({format_kurtosis(kurtosis)}).")
    
    hypothesis = "Descriptive statistics analysis with no formal hypothesis test."
    
    significance = "Not applicable for distribution analysis."
    
    model_implications = generate_model_implications(feature_name, "distribution", test_results)
    
    return {
        "visualization": fig,
        "explanation": explanation,
        "hypothesis": hypothesis,
        "significance": significance,
        "model_implications": model_implications
    }

def generate_stationarity_analysis(feature_name: str,
                                   test_results: Dict[str, Any],
                                   df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate visualization and explanation for stationarity test results.
    """
    fig = create_rolling_statistics_plot(df[feature_name])
    
    adf_statistic = test_results.get("adf_statistic", 0)
    p_value = test_results.get("p_value", 1)
    is_stationary = test_results.get("is_stationary", False)
    critical_values = test_results.get("critical_values", {})
    
    explanation = (f"The Augmented Dickey-Fuller test for {feature_name} produces a test statistic of {adf_statistic:.4f} "
                   f"with a p-value of {format_p_value(p_value)}. ")
    
    if is_stationary:
        explanation += "The series is stationary, meaning its statistical properties don't change over time."
    else:
        explanation += "The series is non-stationary, meaning its statistical properties change over time."
    
    if critical_values:
        explanation += f" Critical values: 1%: {critical_values.get('1%', 'N/A')}, 5%: {critical_values.get('5%', 'N/A')}, 10%: {critical_values.get('10%', 'N/A')}."
    
    hypothesis = "H0: The time series is non-stationary (has a unit root)."
    
    significance = "Statistically significant (reject H0)" if is_stationary else "Not statistically significant (fail to reject H0)"
    
    model_implications = generate_model_implications(feature_name, "stationarity", test_results)
    
    return {
        "visualization": fig,
        "explanation": explanation,
        "hypothesis": hypothesis,
        "significance": significance,
        "model_implications": model_implications
    }

def generate_mutual_information_analysis(feature_name: str,
                                         test_results: Dict[str, Any],
                                         df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate explanation for mutual information test results.
    """
    fig = None
    
    mutual_info = test_results.get("mutual_information", 0)
    
    target_name = next((col for col in df.columns if col.startswith('future_')), 'target variable')
    
    explanation = (f"The mutual information between {feature_name} and {target_name} is {mutual_info:.4f}. "
                   f"This measures the reduction in uncertainty about the target when knowing the feature value. "
                   f"Higher values indicate stronger non-linear relationships.")
    
    hypothesis = "No formal hypothesis test for mutual information. This is a non-parametric measure of dependency."
    
    significance = format_mutual_information_significance(mutual_info)
    
    model_implications = generate_model_implications(feature_name, "mutual_information", test_results)
    
    return {
        "visualization": fig,
        "explanation": explanation,
        "hypothesis": hypothesis,
        "significance": significance,
        "model_implications": model_implications
    }

def generate_t_test_analysis(feature_name: str,
                             test_results: Dict[str, Any],
                             df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate visualization and explanation for t-test results.
    """
    condition_name = 'high_volatility'  # Default
    for col in df.columns:
        if col.endswith('_condition') or col == 'high_volatility':
            condition_name = col
            break
    
    fig = create_box_plot(df, feature_name, condition_name)
    
    t_statistic = test_results.get("t_statistic", 0)
    p_value = test_results.get("p_value", 1)
    significant = test_results.get("significant", False)
    
    explanation = (f"The t-test for {feature_name} between {condition_name} groups yields a t-statistic of {t_statistic:.4f} "
                   f"with a p-value of {format_p_value(p_value)}. ")
    
    if significant:
        explanation += f"There is a statistically significant difference in {feature_name} between different {condition_name} conditions."
    else:
        explanation += f"There is no statistically significant difference in {feature_name} between different {condition_name} conditions."
    
    hypothesis = f"H0: There is no difference in the mean of {feature_name} between {condition_name} groups."
    
    significance = "Statistically significant (reject H0)" if significant else "Not statistically significant (fail to reject H0)"
    
    model_implications = generate_model_implications(feature_name, "t_test", test_results)
    
    return {
        "visualization": fig,
        "explanation": explanation,
        "hypothesis": hypothesis,
        "significance": significance,
        "model_implications": model_implications
    }

def create_scatter_plot(x: pd.Series, y: pd.Series, x_label: str, y_label: str, title: str = None) -> plt.Figure:
    """Create a scatter plot with regression line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha': 0.5})
    ax.set_xlabel(x_label.replace("_", " ").title())
    ax.set_ylabel(y_label.replace("_", " ").title())
    ax.set_title(title or f"{x_label.replace('_', ' ').title()} vs {y_label.replace('_', ' ').title()}")
    plt.tight_layout()
    return fig

def create_histogram_plot(series: pd.Series, feature_name: str, bins: int = 30, kde: bool = True) -> plt.Figure:
    """Create a histogram with optional KDE for a feature."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(series, bins=bins, kde=kde, ax=ax)
    ax.set_xlabel(feature_name.replace("_", " ").title())
    ax.set_title(f"Distribution of {feature_name.replace('_', ' ').title()}")
    plt.tight_layout()
    return fig

def create_rolling_statistics_plot(series: pd.Series, window: int = 20, title: str = None) -> plt.Figure:
    """Create a plot with original series and rolling mean/std."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(series, label="Original")
    ax.plot(rolling_mean, label=f"Rolling Mean (window={window})")
    ax.plot(rolling_std, label=f"Rolling Std (window={window})")
    
    ax.set_xlabel("Date")
    ax.set_ylabel(series.name.replace("_", " ").title())
    ax.set_title(title or f"Rolling Statistics of {series.name.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_box_plot(data: pd.DataFrame, feature: str, groupby: str, title: str = None) -> plt.Figure:
    """Create a box plot comparing a feature across groups."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=groupby, y=feature, data=data, ax=ax)
    
    ax.set_xlabel(groupby.replace("_", " ").title())
    ax.set_ylabel(feature.replace("_", " ").title())
    ax.set_title(title or f"{feature.replace('_', ' ').title()} by {groupby.replace('_', ' ').title()}")
    
    plt.tight_layout()
    return fig

def generate_model_implications(feature_name: str, test_name: str, test_results: Dict[str, Any]) -> str:
    """Generate explanations of how this feature/test relates to the XGBoost model for market regime classification."""
    feature_dimensions = {
        'direction_slope': ['Direction', 'Momentum'],
        'volatility_trajectory': ['Volatility'],
        'volume_distribution_skew': ['Participation'],
        'body_consistency': ['Direction', 'Confidence'],
        'price_level_30d': ['Direction', 'Momentum'],
        'volume_zscore': ['Participation']
    }
    
    dimensions = feature_dimensions.get(feature_name, ['Unknown'])
    
    implications = f"This feature influences the {', '.join(dimensions)} dimensions in the daily market regime classification. "
    
    if test_name == "correlation":
        pearson_r = test_results.get("pearson_r", 0)
        abs_corr = abs(pearson_r)
        
        if abs_corr > 0.5:
            implications += "The strong correlation suggests this is a key predictor for these dimensions."
        elif abs_corr > 0.3:
            implications += "The moderate correlation indicates reasonable predictive value."
        else:
            implications += "The weak correlation suggests this feature may be less important in isolation but could still contribute in combination with others."
    
    elif test_name == "stationarity":
        is_stationary = test_results.get("is_stationary", False)
        
        if is_stationary:
            implications += "Its stationarity makes it reliable for consistent model interpretation across different market conditions."
        else:
            implications += "Its non-stationarity suggests the model needs to account for changing statistical properties over time."
    
    elif test_name == "mutual_information":
        mi = test_results.get("mutual_information", 0)
        
        if mi > 0.1:
            implications += f"The high mutual information score of {mi:.4f} indicates strong non-linear predictive power."
        elif mi > 0.05:
            implications += f"The moderate mutual information score of {mi:.4f} suggests some non-linear relationship exists."
        else:
            implications += f"The low mutual information score of {mi:.4f} indicates limited non-linear predictive power."
    
    elif test_name == "t_test":
        significant = test_results.get("significant", False)
        
        if significant:
            implications += "The significant difference across conditions suggests this feature helps differentiate between market states."
        else:
            implications += "The lack of significant difference suggests this feature may not be effective at distinguishing between certain market conditions."
    
    elif test_name == "distribution":
        skew = test_results.get("skew", 0)
        kurtosis = test_results.get("kurtosis", 0)
        
        implications += "Its distribution characteristics inform appropriate preprocessing steps. "
        
        if abs(skew) > 1:
            implications += "The pronounced skewness suggests transformation might improve model performance. "
        
        if kurtosis > 3:
            implications += "Heavy tails indicate potential for outlier values that may affect prediction."
    
    return implications

def format_p_value(p_value: float) -> str:
    """Format p-value with appropriate scientific notation."""
    if p_value < 0.001:
        return f"{p_value:.2e}"
    else:
        return f"{p_value:.4f}"

def format_correlation_strength(correlation: float) -> str:
    """Interpret the strength of a correlation coefficient."""
    abs_corr = abs(correlation)
    if abs_corr > 0.7:
        return "strong"
    elif abs_corr > 0.5:
        return "moderate to strong"
    elif abs_corr > 0.3:
        return "moderate"
    elif abs_corr > 0.1:
        return "weak to moderate"
    else:
        return "weak"

def format_skewness(skew: float) -> str:
    """Interpret skewness value."""
    if skew > 1:
        return "strongly positively skewed"
    elif skew > 0.5:
        return "moderately positively skewed"
    elif skew > 0.1:
        return "slightly positively skewed"
    elif skew > -0.1:
        return "approximately symmetric"
    elif skew > -0.5:
        return "slightly negatively skewed"
    elif skew > -1:
        return "moderately negatively skewed"
    else:
        return "strongly negatively skewed"

def format_kurtosis(kurtosis: float) -> str:
    """Interpret kurtosis value."""
    if kurtosis > 3:
        return "leptokurtic (heavy-tailed)"
    elif kurtosis < -1:
        return "platykurtic (light-tailed)"
    else:
        return "mesokurtic (normal-like tails)"

def format_mutual_information_significance(mi: float) -> str:
    """Interpret mutual information significance."""
    if mi > 0.3:
        return "Very strong non-linear relationship"
    elif mi > 0.1:
        return "Strong non-linear relationship"
    elif mi > 0.05:
        return "Moderate non-linear relationship"
    elif mi > 0.01:
        return "Weak non-linear relationship"
    else:
        return "Very weak or no non-linear relationship"

def handle_analysis_error(feature_name: str, test_name: str, error: Exception) -> Dict[str, Any]:
    """Handle errors in analysis process and return fallback content."""
    logger.error(f"Error analyzing {test_name} for {feature_name}: {str(error)}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, f"Analysis error: {str(error)}", ha='center', va='center', fontsize=12)
    ax.set_axis_off()
    
    return {
        "visualization": fig,
        "explanation": f"Error during analysis: {str(error)}",
        "hypothesis": "Analysis could not be completed due to error",
        "significance": "Unknown due to error",
        "model_implications": "Cannot determine implications due to error"
    }

def analyze_sample_data():
    """Run analysis on sample data to test functionality."""
    from datetime import datetime, timedelta
    
    # Define the date range
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    print(f"Starting analysis from {start_date} to {end_date}")
    
    try:
        # Load the data
        df = load_daily_features(start_date, end_date)
        print(f"Data loaded with shape: {df.shape}")
        
        # Check if data is empty
        if df.empty:
            print("No daily data available for the selected date range.")
            return
        
        # Generate statistics
        stats_results = generate_daily_stats(start_date, end_date)
        print("Daily stats generated")
        
        # Analyze the data
        analysis_results = analyze_daily_data(stats_results, df, start_date, end_date)
        print("Daily data analyzed")
        
        # Print the results
        if not analysis_results:
            print("No analysis results to display.")
        else:
            for feature, tests in analysis_results.items():
                print(f"\nFeature: {feature}")
                for test_type, test_analysis in tests.items():
                    if test_type == "summary":
                        print(f"  Summary: {test_analysis['explanation']}")
                    else:
                        print(f"  Test: {test_type}")
                        print(f"  Explanation: {test_analysis['explanation'][:100]}...")
                        if 'significance' in test_analysis:
                            print(f"  Significance: {test_analysis['significance']}")
                        else:
                            print("  Significance: Not available")
    
    except Exception as e:
        print(f"Error in analysis: {str(e)}")

# Ensure the function runs when the module is executed
if __name__ == "__main__":
    analyze_sample_data()